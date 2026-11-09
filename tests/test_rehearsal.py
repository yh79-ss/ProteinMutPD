"""Rehearsal: execute the sbatch scripts that `mpd submit` writes, with bash, on the CPU, in dependency order.

Nothing else in the suite runs a rendered script end to end: the stage functions are tested in-process, the renderer as text and the
submit logic against a fake Slurm. This file closes the gap with the real planner and renderer, fx_slurm's fake sbatch, `module` and
`nvidia-smi`, made-up SLURM_* variables, a venv whose interpreter reaches the test environment through a .pth file, tiny random
models and tiny data (tests/fx_rehearsal.py has the toolbox and its limits).

Two tiers. Unmarked (about a minute and a half on a loaded login node): one ESM2 pair with `direct-all` and `score-all` on the Grace
profile, run concurrently, plus the seams of the script prologue and environment, the GPU guard, the offline flags, a rerun after
success and a training job killed at its time limit. `@pytest.mark.rehearsal` (several minutes): the other recipes, the Vision profile
with its two-process training, chunked arrays, a SaProt student and the maintainers command.

What a pass means: the scripts of this code run on a CPU from their first line to MPD_STAGE_DONE and leave the files `mpd status`,
`mpd results` and `--force` read. What it does not mean: anything about Grace's modules, drivers, NCCL, memory, queue or Slurm
version. Nothing here ran on Grace, on a GPU or through a real scheduler.
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

if sys.platform != "linux" or shutil.which("bash") is None:
    pytest.skip("the rehearsal runs bash scripts as a Linux cluster would (process groups, fcntl, /usr/bin)", allow_module_level=True)
pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from fx_rehearsal import (WORKERS, Node, Recipe, environment, job_logs, jobs_for, kill_all, kill_and_resume, make_home,  # noqa: E402
                          model_variables, net_attempts, prepare_campaign, read_json, rehearse, rehearsal_venv, run_job,  # noqa: F401
                          run_prepared, run_stages, shards_of, training_difference)
from fx_saprot import fake_calls, install_fake_foldseek, tiny_saprot_models  # noqa: E402,F401
from mpd.cli import main  # noqa: E402
from mpd.slurm.campaign import CAUSE_ADVICE, find_cause, stage_log, stage_progress  # noqa: E402
from mpd.slurm.submit import submit  # noqa: E402

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "maintainers" / "rehearse_campaign.py"
TOLERANCE = 1e-6                    # absolute; the stage tests hold in-process resumes to the same bound. On CPU in float32 it measures 0.0
SUMMARY_KEYS = {"campaign", "objective", "primary_readout", "primary_readout_basis", "eval_data", "units", "assays_in_dataset",
                "results", "dms_labels_opened", "provenance"}          # docs/RESULTS.md
ENTRY_KEYS = {"dataset", "unit", "readout", "assays", "assays_finite", "macro_spearman", "primary"}


@pytest.fixture(autouse=True)
def _no_stray_processes():
    yield
    kill_all()


# ---------------------------------------------------------------------------------------------------- shared checks
def poison_python_path(root: Path) -> Path:
    """A folder that breaks `import numpy`: a job that inherits PYTHONPATH from the shell that submitted it dies at once."""
    folder = root / "poisoned_pythonpath"
    (folder / "numpy").mkdir(parents=True, exist_ok=True)
    (folder / "numpy" / "__init__.py").write_text("raise ImportError('the PYTHONPATH of the submitting shell reached the job')\n")
    return folder


def stray_shell(node: Node, root: Path, tiny_models: dict, *, tiny_saprot: dict | None = None, without=(), **extra) -> dict:
    """The submitting shell of these tests: the model overrides (MPD_MODEL_<KEY>), a small MPD_CKPT_EVERY, and variables that must
    not hurt a job (a poisoned PYTHONPATH, a PYTHONHOME, switches that would turn the offline flags off)."""
    env = {**model_variables(tiny_models, tiny_saprot), "MPD_CKPT_EVERY": "4", "PYTHONPATH": str(poison_python_path(root)),
           "PYTHONHOME": str(root / "no_such_pythonhome"), "HF_HUB_OFFLINE": "0", "TRANSFORMERS_OFFLINE": "0", **extra}
    for name in without:
        env.pop(name)
    return node.shell(**env)


def solo(tmp_path, venv, tiny_models, *, cluster: str = "grace", structures: bool = False, home_name: str = "mpd", **shell_options):
    """A fresh node, home and shell for a test that needs its own."""
    node = Node(tmp_path / "node")
    home = make_home(tmp_path, cluster, venv, structures=structures, name=home_name)
    return SimpleNamespace(node=node, home=home, cluster=cluster, models=tiny_models, root=tmp_path,
                           shell=stray_shell(node, tmp_path, tiny_models, **shell_options))


def prepared(env, recipe: Recipe | None = None, **plan_options):
    return prepare_campaign(env.node, env.home, env.cluster, recipe or Recipe(), env.shell, **plan_options)


def assert_ran(rh) -> None:
    assert rh.run.ok, rh.run.report()
    for stage in rh.plan.stages:
        outcomes = rh.run.outcomes[stage.full_name]
        assert len(outcomes) == (len(stage.tasks) if stage.is_array else 1), stage.full_name
        for o in outcomes:
            lines = o.out.splitlines()
            starts = [ln for ln in lines if ln.startswith("MPD_JOB_START")]
            expected = rf"MPD_JOB_START host=\S* job={o.job.job_id} stage={re.escape(stage.full_name)}"
            assert len(starts) == 1 and re.fullmatch(expected, starts[0]), o.describe()
            assert lines[-1] == f"MPD_STAGE_DONE {stage.full_name}", o.describe()


def assert_outputs(rh) -> None:
    """The files each stage leaves behind (mpd.slurm.campaign.task_marker names them: `mpd status` and `--force` read the same) and no
    temporary file."""
    root, spec = rh.root, rh.plan.spec
    manifest = read_json(root / "contexts" / "MANIFEST.json")
    assert manifest["dms_labels_opened"] is False and manifest["units"] == list(spec.units)
    assert any((root / "contexts").glob("union.*"))
    if spec.backend == "live":
        done = read_json(root / "teacher" / "COMPLETE.json")
        assert done["status"] == "PASS" and done["teacher"] == spec.teacher and (root / "teacher" / "logits.npy").is_file()
        assert done["provenance"]["slurm_job_id"] == rh.ids["teacher"]          # SLURM_JOB_ID reached python
    for unit in spec.units:
        train = root / "train" / unit
        complete = read_json(train / "COMPLETE.json")
        assert complete["status"] == "PASS" and complete["updates"] == rh.plan.resolved["updates"]
        assert complete["provenance"]["slurm_job_id"] in {j.job_id for j in rh.jobs["train"]}
        assert "gpu" not in complete["provenance"] and complete["peak_gpu_memory_gb"] is None        # ran on the CPU: MPD_REQUIRE_CUDA=0
        chosen = read_json(train / "SELECTED.json")
        assert (train / f"checkpoint_{chosen['step']:05d}" / "adapter_model.safetensors").is_file()
        for name in ("CONFIG.json", "dev_curve.jsonl", "train_steps.csv"):
            assert (train / name).is_file(), name
        assert not (root / "train" / f"{unit}.partial").exists()                # the resumable state is gone once the unit is complete
    assert (root / "evaluate" / "per_assay.csv").is_file()
    assert not list(root.glob("**/*.tmp*")), "a temporary file was left behind"


def assert_shards(rh) -> None:
    """Array tasks write disjoint files: each task its own shard, together every context exactly once."""
    found = shards_of(rh)
    for stage in (s for s in rh.plan.stages if s.name in ("score-base", "score")):
        for task, job in zip(stage.tasks, rh.jobs[stage.full_name]):
            record = found[(task["dataset"], task["unit"])][task["shard"]]
            assert record["num_shards"] == task["num_shards"] == record["_npz_num"] and record["_npz_shard"] == task["shard"]
            assert record["provenance"]["slurm_job_id"] == job.job_id, (stage.full_name, task)      # the file is that task's own
            assert record["dms_labels_opened"] is False
    for (dataset, unit), shards in found.items():
        num = {r["num_shards"] for r in shards.values()}
        assert len(num) == 1 and sorted(shards) == list(range(num.pop())), (dataset, unit, sorted(shards))
        rows = [row for r in shards.values() for row in r["_rows"]]
        assert sorted(rows) == list(range(len(rows))) and all(r["contexts"] == len(r["_rows"]) for r in shards.values())


def assert_summary(rh) -> None:
    summary = read_json(rh.root / "evaluate" / "summary.json")
    spec = rh.plan.spec
    assert set(summary) >= SUMMARY_KEYS
    assert summary["campaign"] == rh.plan.campaign_id and summary["objective"] == spec.objective and summary["dms_labels_opened"] is True
    assert summary["primary_readout"] == ("delta" if spec.objective == "direct_logits" else "dual")
    assert "assumed, not confirmed" in summary["primary_readout_basis"]                       # the maintainer's convention, said as such
    assert summary["units"] == ["base", *spec.units] and summary["eval_data"] == list(spec.eval_data)
    assert summary["assays_in_dataset"] == {"proteingym": 3}
    assert summary["provenance"]["slurm_job_id"] == rh.ids["evaluate"]
    entries = summary["results"]
    assert {(e["dataset"], e["unit"], e["readout"]) for e in entries} == {
        (ds, unit, readout) for ds in spec.eval_data for unit in summary["units"] for readout in ("delta", "dual")}     # both, always
    for e in entries:
        assert set(e) >= ENTRY_KEYS and e["assays_finite"] == e["assays"] and np.isfinite(e["macro_spearman"])
        assert e["primary"] == (e["readout"] == summary["primary_readout"])
        if e["unit"] == "base":
            assert e["assays"] == 3 and "delta_vs_base" not in e
        else:
            assert e["delta_vs_base"] == pytest.approx(e["macro_spearman"] - e["base_macro_same_assays"])
    rows = (rh.root / "evaluate" / "per_assay.csv").read_text().splitlines()
    assert rows[0] == "dataset,unit,assay_id,readout,n_variants,spearman"
    assert len(rows) - 1 == sum(e["assays"] for e in entries)


def assert_results_cli(rh, capsys) -> None:
    """`mpd results` prints both readouts for the frozen student and for the distilled one, with the numbers of summary.json."""
    capsys.readouterr()
    assert main(["results", str(rh.root)]) == 0
    text = capsys.readouterr().out
    summary = read_json(rh.root / "evaluate" / "summary.json")
    base, tuned = text.split("distilled student against the same student before training")
    assert "frozen student (before training)" in base and "primary readout by this tool's convention" in tuned
    assert "an assumption of the maintainer" in tuned
    for e in summary["results"]:
        part = base if e["unit"] == "base" else tuned
        line = next((ln for ln in part.splitlines() if ln.split()[:1] == [e["dataset"]] and e["readout"] in ln.split()[1:2]
                     and f"{e['macro_spearman']:.4f}" in ln and (e["unit"] == "base" or e["unit"] in ln.split())), None)
        assert line is not None, (e, text)
    assert str(rh.root / "evaluate" / "summary.json") in text and "adapter_model.safetensors" in text
    capsys.readouterr()
    assert main(["results", rh.plan.campaign_id, "--home", str(rh.home)]) == 0              # by id, as a student types it
    assert capsys.readouterr().out == text


def assert_logs(rh, capsys) -> None:
    """Every job's logs are in <campaign>/logs under the names `mpd status` and `mpd logs` look for, and those commands find them."""
    found = job_logs(rh)
    for stage in rh.plan.stages:
        for job in rh.jobs[stage.full_name]:
            task = job.task                                              # None for a stage that is not an array
            mine = {p.suffix[1:]: p for s, j, t, p in found if (s, j, t) == (stage.full_name, job.array_id or job.job_id, task)}
            assert set(mine) == {"out", "err"}, (stage.full_name, job.job_id, [p.name for *_, p in found])
            assert mine["out"].parent == rh.root / "logs" and mine["out"] == job.logs()[0]
        base = stage.full_name.split(".")[0]
        log = stage_log(rh.root, base, err=False)                          # what `mpd status` reads to explain a failure
        assert log is not None and "MPD_STAGE_DONE" in log.read_text(), base
    capsys.readouterr()
    assert main(["logs", str(rh.root)]) == 0
    listing = capsys.readouterr().out
    for stage in rh.plan.stages:
        assert re.search(rf"^{re.escape(stage.name)}\s", listing, re.M), (stage.name, listing)
    assert main(["logs", str(rh.root), "train"]) == 0
    assert "MPD_STAGE_DONE train" in capsys.readouterr().out


def tree_digest(root: Path, *parts: str) -> dict[str, str]:
    out = {}
    for part in parts:
        for p in sorted((root / part).rglob("*")):
            if p.is_file():
                out[p.relative_to(root).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


# ---------------------------------------------------------------------------------------------------- the pair (default tier)
class Experiment(threading.Thread):
    """kill_and_resume beside the pair: it starts once the teacher table of direct-all exists, so its two training runs overlap with
    the pair's own instead of adding their time to it. The error of the thread, if any, is raised by the test that reads it."""

    def __init__(self, second, semaphore):
        super().__init__(daemon=True)
        self.second, self.semaphore, self.reference, self.result, self.error = second, semaphore, None, None, None

    def begin(self, reference) -> None:
        self.reference = reference
        self.start()

    def run(self) -> None:
        try:
            self.result = kill_and_resume(self.reference, self.second, semaphore=self.semaphore)
        except BaseException as exc:
            self.error = exc


@pytest.fixture(scope="module")
def pair(tmp_path_factory, rehearsal_venv, tiny_models):
    """`esm2-35m -> esm2-8m` with direct-all and score-all on the Grace profile, run once, concurrently, as scripts."""
    root = tmp_path_factory.mktemp("pair")
    node, home = Node(root / "node"), make_home(root, "grace", rehearsal_venv)
    shell = stray_shell(node, root, tiny_models)
    semaphore = threading.Semaphore(WORKERS)
    (root / "resume").mkdir()
    other = solo(root / "resume", rehearsal_venv, tiny_models)          # a home of its own for the TIMEOUT experiment
    experiment = Experiment(prepared(other, Recipe("direct-all")), semaphore)

    def on_done(rh, stage):
        if rh.recipe.method == "direct-all" and stage == "teacher":
            experiment.begin(rh)

    started = time.monotonic()
    runs = rehearse(node, home, "grace", [Recipe("direct-all"), Recipe("score-all")], shell, semaphore=semaphore, on_done=on_done)
    if experiment.reference is not None:
        experiment.join(timeout=600)
    yield SimpleNamespace(node=node, home=home, shell=shell, root=root, runs={rh.recipe.method: rh for rh in runs},
                          experiment=experiment, seconds=time.monotonic() - started)
    kill_all()


@pytest.fixture(params=["direct-all", "score-all"])
def campaign(request, pair):
    return pair.runs[request.param]


def test_every_stage_of_the_pair_runs_to_its_end_in_dependency_order(campaign):
    assert_ran(campaign)
    outcomes = campaign.run.outcomes
    for stage in campaign.plan.stages:
        for dep in stage.depends_on:
            assert min(o.started for o in outcomes[stage.full_name]) >= max(o.ended for o in outcomes[dep]), \
                f"{stage.full_name} started before {dep} had finished"


def test_each_stage_left_the_files_the_next_one_and_status_look_for(campaign):
    assert_ran(campaign)
    assert_outputs(campaign)
    assert dict(stage_progress(campaign.root)) == {s.name: (len(s.tasks), len(s.tasks)) for s in campaign.plan.stages if s.part == 0}


def test_the_array_tasks_write_disjoint_files(campaign):
    assert_shards(campaign)
    assert len(campaign.jobs["score"]) == len(campaign.jobs["score-base"]) == 2              # --score-shards proteingym=2: every task ran


def test_evaluate_wrote_the_summary_with_both_readouts(campaign):
    assert_summary(campaign)


def test_mpd_results_prints_both_readouts_for_the_base_and_the_distilled_student(campaign, capsys):
    assert_results_cli(campaign, capsys)


def test_the_logs_carry_the_names_mpd_status_and_mpd_logs_look_for(campaign, capsys):
    assert_logs(campaign, capsys)


def test_gpu_stages_ran_on_the_cpu_because_mpd_require_cuda_0_is_exported_after_the_guard(campaign):
    text = campaign.scripts["train"].read_text()
    assert text.index("MPD_REQUIRE_CUDA=1") < text.index("MPD_REQUIRE_CUDA=0") < text.index("nvidia-smi -L")
    assert "MPD_REQUIRE_CUDA" not in campaign.scripts["prepare"].read_text().replace("export MPD_REQUIRE_CUDA=0", "")
    assert "GPU 0: NVIDIA" in campaign.run.outcomes["train"][0].out                  # the script's `nvidia-smi -L` passed and printed
    assert campaign.run.outcomes["train"][0].job.env["CUDA_VISIBLE_DEVICES"] == ""


def test_mpd_ckpt_every_of_the_submitting_shell_set_the_cadence_of_the_resumable_saves(campaign):
    log = campaign.run.outcomes["train"][0].out
    saves = [int(m) for m in re.findall(r"resumable checkpoint at update (\d+)", log)]
    assert saves == [4, 8, 12, 16, 20]                           # MPD_CKPT_EVERY=4 of the submitting shell reached python; 24 is the end
    assert read_json(campaign.root / "train" / "all" / "COMPLETE.json")["resumed_from"] == []


def test_nothing_connected_to_the_network_and_nothing_was_written_under_home(pair):
    assert net_attempts(pair.home) == []
    assert list(pair.node.userhome.iterdir()) == [], "a job wrote under $HOME (10 GB / 10,000 files on Grace)"


def test_mpd_status_shows_every_stage_finished(pair, monkeypatch, capsys):
    rh = pair.runs["direct-all"]
    fs = pair.node.fs
    for stage, job_id in rh.ids.items():
        fs.set_state(job_id, "COMPLETED", elapsed="00:00:30")
    monkeypatch.setenv("PATH", fs.path)
    monkeypatch.setenv("USER", fs.user)
    capsys.readouterr()
    assert main(["status", str(rh.root)]) == 0
    out = capsys.readouterr().out
    lines = out.splitlines()
    header = next(i for i, ln in enumerate(lines) if ln.split()[:2] == ["stage", "state"])
    table = []
    for ln in lines[header + 1:]:
        if not ln.strip():
            break
        table.append(ln.split())
    assert [row[0] for row in table] == [s.full_name for s in rh.plan.stages]
    columns = lines[header].split()
    for row in table:
        stage = next(s for s in rh.plan.stages if s.full_name == row[0])
        assert row[columns.index("state")] == "COMPLETED", out
        assert row[columns.index("tasks")] == f"{len(stage.tasks)}/{len(stage.tasks)}", out      # read from the files the stages wrote
    assert "problems:" not in out and f"mpd results {rh.root}" in out


def test_submitting_again_after_a_real_run_finds_every_stage_finished(pair):
    """The files the real scripts leave are what `mpd submit --force` takes for finished work: nothing is queued again."""
    rh, fs = pair.runs["direct-all"], pair.node.fs
    for job_id in rh.ids.values():
        fs.set_state(job_id, "COMPLETED", elapsed="00:00:30")
    queued, said = len(fs.submitted()), []
    with environment({"USER": fs.user, "LOGNAME": fs.user}):
        submit(rh.plan, rh.scripts, force=True, sbatch=str(fs.bin / "sbatch"), squeue=str(fs.bin / "squeue"),
               scancel=str(fs.bin / "scancel"), echo=said.append, sleep=fs.sleep, clock=fs.clock)
    for stage in rh.plan.stages:
        assert any(line.split()[:2] == ["finished", stage.full_name] and "(its output files exist)" in line for line in said), said
    assert len(fs.submitted()) == queued and any(line.startswith("nothing to submit") for line in said), said


def test_running_finished_stages_again_changes_nothing(pair):
    rh = pair.runs["direct-all"]
    before = tree_digest(rh.root, "teacher", "train", "score")
    manifest = read_json(rh.root / "contexts" / "MANIFEST.json")
    summary = {k: v for k, v in read_json(rh.root / "evaluate" / "summary.json").items() if k != "provenance"}
    per_assay = (rh.root / "evaluate" / "per_assay.csv").read_bytes()
    again = {name: [dataclasses.replace(j, job_id=str(7000 + 10 * i + k), env={**j.env, "SLURM_JOB_ID": str(7000 + 10 * i + k)})
                    for k, j in enumerate(jobs)] for i, (name, jobs) in enumerate(rh.jobs.items())}
    semaphore = threading.Semaphore(WORKERS)
    # the stages that skip themselves read the tables, so prepare (which writes them again) and evaluate run on their own afterwards
    gpu_stages = {"teacher", "train", "score-base", "score"}
    skipping = run_stages(rh.plan, again, only=gpu_stages, ignore_dependencies=True, semaphore=semaphore)
    assert skipping.ok, skipping.report()
    assert "already computed" in skipping.outcomes["teacher"][0].out and "skipping" in skipping.outcomes["teacher"][0].out
    assert "all: already complete" in skipping.outcomes["train"][0].out
    for stage in ("score-base", "score"):
        assert all(".npz: already done" in o.out for o in skipping.outcomes[stage]), stage
    for stage in ("prepare", "evaluate"):
        redone = run_stages(rh.plan, again, only={stage}, semaphore=semaphore)
        assert redone.ok, redone.report()
    assert tree_digest(rh.root, "teacher", "train", "score") == before          # byte for byte: nothing was computed or rewritten
    assert read_json(rh.root / "contexts" / "MANIFEST.json") == manifest
    assert {k: v for k, v in read_json(rh.root / "evaluate" / "summary.json").items() if k != "provenance"} == summary
    assert (rh.root / "evaluate" / "per_assay.csv").read_bytes() == per_assay


def test_a_training_job_killed_at_its_time_limit_continues_from_its_last_save(pair):
    """The claim behind the TIMEOUT advice: SIGTERM after the first save, run the same script again, and the result is the one an
    uninterrupted run gives (the pair's direct-all training, on the same contexts and teacher table)."""
    experiment = pair.experiment
    assert experiment.reference is not None, "the experiment never started: the teacher stage of direct-all did not finish"
    if experiment.error is not None:
        raise experiment.error
    result, ref = experiment.result, pair.runs["direct-all"]
    killed, resumed, updates = result["killed"], result["resumed"], ref.plan.resolved["updates"]
    assert killed["returncode"] == -signal.SIGTERM and not killed["finished_before_the_kill"] and not killed["complete"]
    assert killed["saved_update"] % 4 == 0 and 0 < killed["saved_update"] < updates          # MPD_CKPT_EVERY=4, and not the end
    assert "MPD_STAGE_DONE" not in killed["log"] and "resumable checkpoint" in killed["log"]            # unbuffered: the log was live
    assert resumed.rc == 0, resumed.describe()
    assert f"{result['unit']}: resuming from update {killed['saved_update']} of {updates}" in resumed.out, resumed.out
    complete = read_json(result["second"] / "COMPLETE.json")
    assert complete["resumed_from"] == [killed["saved_update"]] and complete["status"] == "PASS"
    assert complete["provenance"]["slurm_job_id"] == resumed.job.job_id                  # the second job, not the one that was killed
    assert read_json(result["reference"] / "COMPLETE.json")["resumed_from"] == []
    assert read_json(result["reference"] / "CONFIG.json")["config_digest"] == read_json(result["second"] / "CONFIG.json")["config_digest"]
    diff = training_difference(result["reference"], result["second"])
    assert diff and max(diff.values()) <= TOLERANCE, f"resumed run differs from the uninterrupted run by {diff} (tolerance {TOLERANCE})"
    assert not (result["second"].parent / f"{result['unit']}.partial").exists()


# ---------------------------------------------------------------------------------------------------- the GPU guard
def run_one(rh, stage: str, *, script: str | None = None, task: int = 0, job_id: str = "9001", env: dict | None = None, drop=()):
    """Run a stage's script once more under another job id, with `env` added to and `drop` removed from the job's environment."""
    job = rh.job(stage, task)
    job = dataclasses.replace(job, job_id=job_id, env={**{k: v for k, v in job.env.items() if k not in drop},
                                                       "SLURM_JOB_ID": job_id, **(env or {})})
    if script is not None:
        path = job.script.with_name(f"{job.script.stem}.mutated.sbatch")
        path.write_text(script)
        job = dataclasses.replace(job, script=path)
    return run_job(job)


def test_a_gpu_stage_without_a_usable_gpu_exits_31_with_mpd_no_gpu(tmp_path, rehearsal_venv, tiny_models):
    """nvidia-smi is fine and the interpreter is started, but torch sees no CUDA device: the guard in the stage is what stops it."""
    env = solo(tmp_path, rehearsal_venv, tiny_models)
    rh = prepared(env, Recipe().without("MPD_REQUIRE_CUDA"))
    text = rh.scripts["teacher"].read_text()
    assert "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MPD_REQUIRE_CUDA=1" in text and "MPD_REQUIRE_CUDA=0" not in text
    done = run_one(rh, "teacher")
    assert done.rc == 31, done.describe()
    refusal = r"^MPD_NO_GPU: torch\.cuda\.is_available\(\) is False CUDA_VISIBLE_DEVICES= torch=\S+ cuda="
    assert re.search(refusal, done.err, re.M), done.err
    assert "MPD_JOB_START" in done.out and "MPD_STAGE_DONE" not in done.out          # the script got as far as python, which refused
    assert not (rh.root / "teacher").exists()


def test_a_gpu_stage_on_a_node_whose_nvidia_smi_fails_exits_31_before_python_starts(tmp_path, rehearsal_venv, tiny_models):
    env = solo(tmp_path, rehearsal_venv, tiny_models)
    rh = prepared(env, Recipe().without("MPD_REQUIRE_CUDA"))
    env.node.fs.configure(nvidia_smi={"fail": True})
    done = run_one(rh, "teacher")
    assert done.rc == 31, done.describe()
    assert re.search(r"^MPD_NO_GPU: no usable GPU on \S+; resubmit with --sbatch-arg=--exclude=\S+$", done.err, re.M), done.err
    assert "MPD_JOB_START" not in done.out and not (rh.root / "teacher").exists()
    assert "NVIDIA-SMI has failed" in done.err
    cpu = run_one(rh, "prepare", job_id="9002")                                    # a CPU stage never asks for a GPU
    assert cpu.rc == 0, cpu.describe()


def test_a_cpu_stage_has_no_gpu_guard_to_pass(tmp_path, rehearsal_venv, tiny_models):
    env = solo(tmp_path, rehearsal_venv, tiny_models)
    rh = prepared(env)
    for stage in ("prepare", "evaluate"):
        text = rh.scripts[stage].read_text()
        assert "nvidia-smi" not in text and "MPD_NO_GPU" not in text and "PYTORCH_CUDA_ALLOC_CONF" not in text


# ---------------------------------------------------------------------------------------------------- the environment of a job
# Slurm hands a job the environment of the shell that ran sbatch (--export=ALL is the default, and mpd's profiles add neither
# --export nor --get-user-env). So what a job sees is what the script's prologue leaves of that environment: it drops PYTHONPATH,
# PYTHONHOME and every MPD_* variable and then exports the planned ones; everything else (PATH, LD_LIBRARY_PATH, CONDA_*) is inherited.
STRAY = {"MPD_ALLOW_HUB": "1", "MPD_STRUCTURE_LEGACY_MASK": "1", "MPD_NOT_ON_THE_LIST": "sentinel", "UNRELATED_SENTINEL": "sentinel",
         "LD_LIBRARY_PATH": "/stray/lib"}


def python_wrapper(env, name: str = "job_python") -> Path:
    """An interpreter stand-in that writes the environment the script hands to python (NUL separated: a value may hold a newline)
    and then runs the real interpreter of the home."""
    path = env.root / name
    path.write_text(f'#!/bin/bash\nenv -0 > "$MPD_HOME/job_env.bin"\nexec {env.home}/env/bin/python "$@"\n')
    path.chmod(0o755)
    return path


def job_environment(home: Path) -> dict[str, str]:
    items = (home / "job_env.bin").read_bytes().split(b"\0")
    return dict(item.decode().split("=", 1) for item in items if b"=" in item)


@pytest.fixture(scope="module")
def stray(tmp_path_factory, rehearsal_venv, tiny_models):
    """One prepare job started from a shell full of variables, with the environment python got written down."""
    root = tmp_path_factory.mktemp("stray")
    env = solo(root, rehearsal_venv, tiny_models, MPD_KEEP_ALL_CHECKPOINTS="1", MPD_DTYPE="float32", MPD_CKPT_EVERY="7", **STRAY)
    rh = prepared(env, python=str(python_wrapper(env)))
    forwarded = ("MPD_KEEP_ALL_CHECKPOINTS", "MPD_DTYPE", "MPD_CKPT_EVERY", "MPD_MODEL_ESM2_8M", "MPD_MODEL_ESM2_35M")
    rh.jobs = jobs_for(rh.plan, rh.scripts, rh.ids, env.node, {**env.shell, **dict.fromkeys(forwarded, "stale")})   # not what the plan saw
    run = run_stages(rh.plan, rh.jobs, only={"prepare"})
    return SimpleNamespace(env=env, rh=rh, run=run, seen=job_environment(env.home) if run.ok else {})


def test_the_scripts_export_exactly_the_allowlist_and_the_env_pairs(stray):
    text = stray.rh.scripts["prepare"].read_text()
    exported = {name for line in text.splitlines() if line.startswith("export ") for name in re.findall(r"\b(MPD_[A-Z0-9_]+)=", line)}
    assert exported == {"MPD_HOME", "MPD_CLUSTER", "MPD_KEEP_ALL_CHECKPOINTS", "MPD_CKPT_EVERY", "MPD_DTYPE", "MPD_MODEL_ESM2_8M",
                        "MPD_MODEL_ESM2_35M", "MPD_REQUIRE_CUDA"}        # the last is an --env pair of the recipe
    for name in (*STRAY, "/stray/lib"):
        assert name not in text, name
    assert any("MPD_STRUCTURE_LEGACY_MASK is set in this shell but jobs do not inherit it" in w for w in stray.rh.plan.warnings)


def test_python_gets_the_planned_values_of_the_variables_the_scripts_set(stray):
    """The shell the job inherits from held "stale" in every allowlisted variable: the export lines of the script must win."""
    assert stray.run.ok, stray.run.report()
    seen, home = stray.seen, stray.env.home
    assert seen["MPD_HOME"] == str(home) and seen["MPD_CLUSTER"] == "grace"                   # the plan's, for the messages of a job
    assert seen["MPD_KEEP_ALL_CHECKPOINTS"] == "1" and seen["MPD_CKPT_EVERY"] == "7"
    assert seen["MPD_DTYPE"] == "float32" and seen["MPD_MODEL_ESM2_8M"] == str(stray.env.models["student"])
    assert seen["MPD_MODEL_ESM2_35M"] == str(stray.env.models["teacher"])
    assert seen["HF_HUB_OFFLINE"] == "1" and seen["TRANSFORMERS_OFFLINE"] == "1"             # the shell exported 0 for both
    assert seen["PYTHONNOUSERSITE"] == "1" and seen["PYTHONDONTWRITEBYTECODE"] == "1" and seen["PYTHONUNBUFFERED"] == "1"
    assert seen["OMP_NUM_THREADS"] == "1" and seen["MPD_REQUIRE_CUDA"] == "0"                   # --env pairs are exported last and win
    for name in ("HF_HOME", "XDG_CACHE_HOME", "TORCH_HOME", "TRITON_CACHE_DIR", "MPLCONFIGDIR"):
        assert seen[name].startswith(f"{home}/"), f"{name}={seen[name]}: every cache lives under the mpd home, never under $HOME"
    assert seen["HOME"] == str(stray.env.node.userhome) and list(stray.env.node.userhome.iterdir()) == []


def test_the_prologue_purged_and_loaded_exactly_the_recorded_modules_in_that_order(stray):
    calls = [c.argv for c in stray.env.node.fs.calls("module")]
    assert calls == [["purge"], ["load", "Anaconda3/2024.02-1"]], calls
    assert stray.seen["LOADEDMODULES"] == "Anaconda3/2024.02-1"
    assert stray.rh.plan.modules == ("Anaconda3/2024.02-1",) and stray.rh.plan.env_record["cluster"] == "grace"       # from mpd_env.json


@pytest.mark.parametrize("name", ["PYTHONPATH", "PYTHONHOME", "MPD_ALLOW_HUB", "MPD_STRUCTURE_LEGACY_MASK", "MPD_NOT_ON_THE_LIST"])
def test_a_variable_the_scripts_do_not_forward_does_not_reach_python(stray, name):
    assert stray.run.ok, stray.run.report()
    assert name not in stray.seen, f"{name}={stray.seen.get(name)!r} is in the environment of the job's python"


def test_variables_that_are_not_mpd_variables_are_inherited_from_the_submitting_shell(stray):
    """Not a promise of the scripts but a fact the docs state: only PYTHONPATH, PYTHONHOME and the MPD_* variables are dropped, so the
    rest of the submitting shell (PATH, LD_LIBRARY_PATH, VIRTUAL_ENV, CONDA_*, whatever the user exported) reaches the job, as
    Slurm's default --export=ALL has it. The Grace job scripts that ran inherit LD_LIBRARY_PATH too, and modules add to it."""
    assert stray.seen["UNRELATED_SENTINEL"] == "sentinel" and stray.seen["LD_LIBRARY_PATH"] == "/stray/lib"


def test_the_hub_switch_of_a_login_shell_does_not_switch_off_the_weights_check_of_a_job(tmp_path, rehearsal_venv, tiny_models):
    env = solo(tmp_path, rehearsal_venv, tiny_models, without=("MPD_MODEL_ESM2_8M",), MPD_ALLOW_HUB="1")
    done = run_one(prepared(env), "prepare")
    assert done.rc != 0 and "weights for esm2-8m are missing" in done.err, done.describe()


# ---------------------------------------------------------------------------------------------------- offline, weights, the prologue
def test_a_model_without_a_local_folder_fails_with_the_canonical_message_and_no_download(tmp_path, rehearsal_venv, tiny_models):
    env = solo(tmp_path, rehearsal_venv, tiny_models, without=("MPD_MODEL_ESM2_8M",))
    done = run_one(prepared(env), "prepare")
    assert done.rc != 0 and "MPD_STAGE_DONE" not in done.out, done.describe()
    assert (f"weights for esm2-8m are missing ({env.home}/models/esm2-8m). Run on a login node: mpd fetch model esm2-8m --cluster grace"
            in done.err), done.err
    assert net_attempts(env.home) == []


def test_weights_that_vanish_after_prepare_stop_the_gpu_stage_without_a_download(tmp_path, rehearsal_venv, tiny_models):
    """The scratch purge case: prepare passed, then the folder is gone. The job is offline (HF_HUB_OFFLINE=1) and mpd never falls back
    to the Hub id, so the answer is the message that says what to run, not a connection attempt."""
    env = solo(tmp_path, rehearsal_venv, tiny_models)
    rh = prepared(env)
    assert run_one(rh, "prepare").rc == 0
    lines = rh.scripts["teacher"].read_text().splitlines()
    purged = "\n".join(line for line in lines if not line.startswith("export MPD_MODEL_ESM2_35M="))
    done = run_one(rh, "teacher", script=purged, job_id="9004", drop=("MPD_MODEL_ESM2_35M",))
    assert done.rc != 0 and "MPD_STAGE_DONE" not in done.out and "MPD_NO_GPU" not in done.err, done.describe()
    assert f"weights for esm2-35m are missing ({env.home}/models/esm2-35m)" in done.err
    assert f"Run on a login node: mpd fetch model esm2-35m --cluster grace --home {shlex.quote(str(env.home))}" in done.err
    assert net_attempts(env.home) == [] and not (rh.root / "teacher" / "COMPLETE.json").exists()


def test_set_u_comes_after_the_module_block_so_an_lmod_that_reads_unset_variables_survives(tmp_path, rehearsal_venv, tiny_models):
    """The fake `module` reads an unset variable, as Grace's /etc/bashrc does ("BASHRCSOURCED: unbound variable"). The same script
    with `set -u` moved to the top must die of exactly that, which shows the fake reproduces the failure it stands in for."""
    env = solo(tmp_path, rehearsal_venv, tiny_models)
    rh = prepared(env)
    text = rh.scripts["prepare"].read_text()
    assert run_one(rh, "prepare").rc == 0
    top = run_one(rh, "prepare", script=text.replace("set -eo pipefail\n", "set -euo pipefail\n", 1), job_id="9005")
    assert top.rc != 0 and "BASHRCSOURCED: unbound variable" in top.err and "MPD_JOB_START" not in top.out, top.describe()


def test_a_module_the_cluster_does_not_have_stops_the_job_before_python_starts(tmp_path, rehearsal_venv, tiny_models):
    env = solo(tmp_path, rehearsal_venv, tiny_models)
    rh = prepared(env)
    env.node.fs.set_modules(["WebProxy"])
    done = run_one(rh, "prepare")
    assert done.rc == 1 and "MPD_MODULE_LOAD_FAILED Anaconda3/2024.02-1" in done.err, done.describe()
    assert "MPD_JOB_START" not in done.out and not (rh.root / "contexts").exists()


@pytest.mark.parametrize("cause", ["no_gpu", "weights", "module"])
def test_mpd_status_explains_the_log_that_a_failed_job_really_wrote(cause, tmp_path, rehearsal_venv, tiny_models, monkeypatch, capsys):
    """The advice of `mpd status` is keyed on the markers mpd's own scripts and stages print. Here the log is the one a script wrote
    in a failing run, not a hand-made one, so a change to a marker in either place shows."""
    env = solo(tmp_path, rehearsal_venv, tiny_models, **({"without": ("MPD_MODEL_ESM2_8M",)} if cause == "weights" else {}))
    rh = prepared(env, Recipe().without("MPD_REQUIRE_CUDA") if cause == "no_gpu" else Recipe())
    if cause == "module":
        env.node.fs.set_modules(["WebProxy"])
    failing, code = ("teacher", 31) if cause == "no_gpu" else ("prepare", 1)
    if cause == "no_gpu":
        assert run_job(rh.job("prepare")).rc == 0                                   # prepare passed, the GPU stage after it did not
    done = run_job(rh.job(failing))
    assert done.rc == code and find_cause(done.err) == cause, done.describe()
    fs, names = env.node.fs, [s.full_name for s in rh.plan.stages]
    for name in names:                                                              # what Slurm would report for it
        if names.index(name) < names.index(failing):
            fs.set_state(rh.ids[name], "COMPLETED", elapsed="00:00:03")
        elif name == failing:
            fs.set_state(rh.ids[name], "FAILED", exitcode=f"{code}:0", elapsed="00:00:05", node="fakenode001")
        else:
            fs.set_state(rh.ids[name], "CANCELLED", reason="DependencyNeverSatisfied")
    monkeypatch.setenv("PATH", fs.path)
    monkeypatch.setenv("USER", fs.user)
    capsys.readouterr()
    assert main(["status", str(rh.root)]) == 0
    said = " ".join(capsys.readouterr().out.split())                                 # the status text wraps its lines
    assert f"{failing} (job {rh.ids[failing]}) failed with exit code {code}" in said and "last lines of the error log" in said
    assert " ".join(CAUSE_ADVICE[cause].split("{")[0].split()) in said, said
    assert ("--sbatch-arg=--exclude=fakenode001" in said) == (cause == "no_gpu")


def test_the_network_tripwire_refuses_what_leaves_the_machine_and_lets_localhost_through(tmp_path, rehearsal_venv, tiny_models):
    """Without this the 'no download' assertions above could pass because nothing tried, or because DNS happens to fail."""
    env = solo(tmp_path, rehearsal_venv, tiny_models)
    code = ("import socket\ntry:\n    socket.create_connection(('huggingface.co', 443), timeout=2)\nexcept OSError as exc:\n"
            "    print('refused:', exc)\nserver = socket.socket()\nserver.bind(('127.0.0.1', 0))\nserver.listen(1)\n"
            "socket.create_connection(server.getsockname(), timeout=2).close()\nprint('local connection ok')\n")
    done = subprocess.run([str(env.home / "env" / "bin" / "python"), "-c", code], env={"PATH": env.node.path, "MPD_HOME": str(env.home)},
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert "refused: [Errno 101]" in done.stdout and "local connection ok" in done.stdout
    assert net_attempts(env.home) == ["getaddrinfo huggingface.co"]


def test_the_maintainers_command_has_help_and_a_dry_run_that_runs_nothing(tmp_path):
    helped = subprocess.run([sys.executable, str(SCRIPTS), "--help"], capture_output=True, text=True, timeout=120)
    assert helped.returncode == 0 and "says nothing about Grace" in helped.stdout and "--kill-resume" in helped.stdout
    done = subprocess.run([sys.executable, str(SCRIPTS), "--dry-run", "--work", str(tmp_path / "work"), "--method", "direct-all"],
                          capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    assert "dry run: nothing was run" in done.stdout and "would run  bash" in done.stdout and "PASSED" not in done.stdout
    assert "submitted to the fake sbatch: prepare=" in done.stdout
    assert not list((tmp_path / "work" / "mpd" / "runs").glob("*/teacher"))


# ---------------------------------------------------------------------------------------------------- the rest of the matrix (rehearsal)
MATRIX = [("grace", "direct-one"), ("grace", "score-one"), ("vision", "direct-all"), ("vision", "score-all"), ("vision", "direct-one"),
          ("vision", "score-one")]


@pytest.fixture(scope="module")
def matrix(tmp_path_factory, rehearsal_venv, tiny_models):
    """Every recipe and profile the pair does not cover, run at once: Grace with one-assay training (array of two trainings), Vision with
    its default two-process training (`torch.distributed.run` with gloo) and with one-assay training."""
    semaphore = threading.Semaphore(WORKERS)
    names = {"grace": "mpd", "vision": "h+o@m,e=1:2"}          # the second is a home path made of every unusual character a plan accepts
    envs = {cluster: solo(tmp_path_factory.mktemp(f"matrix_{cluster}"), rehearsal_venv, tiny_models, cluster=cluster,
                          home_name=names[cluster]) for cluster in ("grace", "vision")}
    prepared_ = [prepared(envs[cluster], Recipe(method)) for cluster, method in MATRIX]
    run_prepared(prepared_, semaphore=semaphore)
    yield SimpleNamespace(envs=envs, runs=dict(zip(MATRIX, prepared_)))
    kill_all()


@pytest.mark.rehearsal
@pytest.mark.parametrize("cluster, method", MATRIX)
def test_every_other_recipe_runs_end_to_end(matrix, cluster, method, capsys):
    rh = matrix.runs[(cluster, method)]
    assert_ran(rh)
    assert_outputs(rh)
    assert_shards(rh)
    assert_summary(rh)
    assert_results_cli(rh, capsys)
    assert_logs(rh, capsys)
    assert net_attempts(rh.home) == []
    if method.endswith("one"):                                                    # an array of two independent trainings
        train = next(stage for stage in rh.plan.stages if stage.name == "train")
        assert len(rh.jobs["train"]) == 2 and {task["unit"] for task in train.tasks} == {"ASSAY_A", "ASSAY_B"}


@pytest.mark.rehearsal
@pytest.mark.parametrize("cluster", ["grace", "vision"])
def test_the_matrix_wrote_nothing_under_home(matrix, cluster):
    assert list(matrix.envs[cluster].node.userhome.iterdir()) == []


@pytest.mark.rehearsal
@pytest.mark.parametrize("method", ["direct-all", "score-all"])
def test_vision_trains_on_two_processes_started_by_torch_distributed_run(matrix, method):
    rh = matrix.runs[("vision", method)]
    text = rh.scripts["train"].read_text()
    assert "-m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 -m mpd.stages.train " in text
    assert "#SBATCH --gres=gpu:H200:2" in text and "#SBATCH --qos=standard" in text and "#SBATCH --partition=def" in text
    assert rh.plan.resolved["world_size"] == 2 and read_json(rh.root / "train" / "all" / "CONFIG.json")["world_size"] == 2
    assert "module load CUDA/12.8.0 Python/3.11.5" in text                                  # the profile's module, then the recorded one
    assert ["load", "CUDA/12.8.0", "Python/3.11.5"] in [c.argv for c in matrix.envs["vision"].node.fs.calls("module")]


@pytest.mark.rehearsal
def test_chunked_arrays_hand_each_job_its_offset(monkeypatch, tmp_path, rehearsal_venv, tiny_models, capsys):
    """Past 1,000 tasks a stage is split into chunks; here 2, so that three score shards make a chunk of two and a chunk of one. The
    single task of a longer list must still be told which one it is (a literal offset, no SLURM_ARRAY_TASK_ID)."""
    monkeypatch.setattr("mpd.slurm.plan.MAX_ARRAY_TASKS", 2)
    env = solo(tmp_path, rehearsal_venv, tiny_models)
    [rh] = rehearse(env.node, env.home, "grace", [Recipe("score-all", score_shards=3)], env.shell)
    names = [s.full_name for s in rh.plan.stages]
    assert {"score-base", "score-base.1", "score", "score.1"} <= set(names), names
    assert "--task-index $((SLURM_ARRAY_TASK_ID + 0))" in rh.scripts["score-base"].read_text()
    last = rh.scripts["score-base.1"].read_text()
    assert " --task-index 2\n" in last and "SLURM_ARRAY_TASK_ID" not in last and "#SBATCH --array" not in last
    assert_ran(rh)
    assert_shards(rh)
    assert_summary(rh)
    assert stage_progress(rh.root)["score-base"] == (3, 3) and stage_progress(rh.root)["score"] == (3, 3)
    assert sorted(shards_of(rh)[("proteingym", "base")]) == [0, 1, 2]
    assert_logs(rh, capsys)


# ---------------------------------------------------------------------------------------------------- SaProt (rehearsal)
@pytest.fixture
def saprot_home(tmp_path, rehearsal_venv, tiny_models, tiny_saprot_models):
    fake = install_fake_foldseek(tmp_path / "fakebin")                        # not on any PATH; MPD_FOLDSEEK is its only address
    calls = tmp_path / "fakebin" / "calls.log"
    return SimpleNamespace(fake=fake, calls=calls, tmp=tmp_path, venv=rehearsal_venv, models=tiny_models, saprot=tiny_saprot_models)


@pytest.mark.rehearsal
def test_a_saprot_student_gets_its_structure_strings_from_the_foldseek_named_by_mpd_foldseek(saprot_home, capsys):
    s = saprot_home
    env = solo(s.tmp, s.venv, s.models, structures=True, tiny_saprot=s.saprot, MPD_FOLDSEEK=str(s.fake), FAKE_FOLDSEEK_LOG=str(s.calls))
    assert str(s.fake.parent) not in env.node.path.split(":")                 # no PATH lookup can find it
    [rh] = rehearse(env.node, env.home, "grace", [Recipe("score-all", teacher="esm2-8m", student="saprot-35m")], env.shell)
    assert f"export MPD_FOLDSEEK={s.fake}\n" in rh.scripts["prepare"].read_text()          # absolute, from the submitting shell
    assert_ran(rh)
    assert_outputs(rh)
    assert_shards(rh)
    assert_summary(rh)
    assert_results_cli(rh, capsys)
    assert fake_calls(s.fake) and all("structureto3didescriptor" in line for line in fake_calls(s.fake))      # prepare ran the binary
    manifest = read_json(rh.root / "contexts" / "MANIFEST.json")
    digest = manifest["structures"]["digest"]
    assert (rh.root / "contexts" / "structures.json").is_file() and manifest["structures"]["engine"]["name"] == "foldseek"
    assert read_json(rh.root / "train" / "all" / "CONFIG.json")["structures_digest"] == digest
    assert all(r["structures_digest"] == digest for shards in shards_of(rh).values() for r in shards.values())
    assert "structures_digest" not in read_json(rh.root / "teacher" / "COMPLETE.json")           # an ESM2 teacher reads no structure
    assert net_attempts(env.home) == [] and list(env.node.userhome.iterdir()) == []


@pytest.mark.rehearsal
def test_a_foldseek_on_path_alone_is_not_found(saprot_home):
    s = saprot_home
    env = solo(s.tmp, s.venv, s.models, structures=True, tiny_saprot=s.saprot, FAKE_FOLDSEEK_LOG=str(s.calls))
    env.shell["PATH"] = f"{s.fake.parent}:{env.shell['PATH']}"               # `foldseek` is on PATH here, and MPD_FOLDSEEK is not set
    rh = prepared(env, Recipe("score-all", teacher="esm2-8m", student="saprot-35m"))
    assert "MPD_FOLDSEEK" not in rh.scripts["prepare"].read_text()
    done = run_one(rh, "prepare")
    assert done.rc != 0 and "Foldseek was not found" in done.err and "mpd fetch foldseek --cluster grace" in done.err, done.describe()
    assert fake_calls(s.fake) == [] and not (rh.root / "contexts" / "MANIFEST.json").exists()


# ------------------------------------------------------------------------------------------------ the maintainers command (rehearsal)
@pytest.mark.rehearsal
def test_the_maintainers_command_rehearses_campaigns_and_prints_a_transcript(tmp_path):
    done = subprocess.run([sys.executable, str(SCRIPTS), "--work", str(tmp_path / "work"), "--method", "direct-all", "--kill-resume"],
                          capture_output=True, text=True, timeout=1500)
    out = done.stdout
    assert done.returncode == 0, out[-4000:] + done.stderr[-2000:]
    assert re.search(r"^direct-all\s+PASSED$", out, re.M) and re.search(r"^kill-resume\s+PASSED$", out, re.M)
    assert re.search(r"^PASSED in \d+ s\. The jobs ran on the CPU with fake tools; this says nothing about Grace", out, re.M)
    assert "frozen student (before training)" in out and "SIGTERM after the save at update" in out and "largest difference" in out
