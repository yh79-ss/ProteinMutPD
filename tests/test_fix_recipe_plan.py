"""Four problems an independent review found in the recipe, the plan and the job scripts, each with the test that keeps it fixed.

F3: the same command typed with the datasets or assays in another order (or the default teacher backend spelled out) is the same
campaign, and `--force` continues it instead of calling it a different recipe. ml-03: `--kl-weight` is refused for the score
methods, which have no KL term. ml-04: `--seed` outside 0..2**32-1 is refused when the plan is made, not hours into the queue.
G7: a profile loaded from a YAML file is not exported as MPD_CLUSTER, because the fetch command a job prints would name it.
Nothing here talks to a real cluster: the Slurm clients are the fakes of fx_slurm.py.
"""

import dataclasses
import json
import os
import shlex
from pathlib import Path

import numpy as np
import pytest
import yaml

from fx_cli import FAKE_ACCOUNT, cli_env, ready_home, recipe, run_cli  # noqa: F401
from fx_render import ScriptRun, bash_n, clean_mpd_env, make, render_all, spec  # noqa: F401  (clean_mpd_env is autouse)
from fx_slurm import clean_path, fake_slurm  # noqa: F401
from mpd import cli, registry
from mpd.data import models
from mpd.slurm.campaign import SubmitError, write_marker
from mpd.slurm.cluster import Cluster, available_clusters
from mpd.slurm.submit import submit as real_submit
from mpd.slurm.submit import write_campaign
from mpd.spec import MAX_SEED, RunSpec, SpecError

pytestmark = pytest.mark.usefixtures("cli_env")
CLUSTERS_DIR = Path(__file__).resolve().parents[1] / "src" / "mpd" / "clusters"


@pytest.fixture(autouse=True)
def no_real_slurm(monkeypatch):
    """Even a test that forgets `fake_slurm` cannot find the real sbatch/squeue/scancel/sacct on PATH."""
    monkeypatch.setenv("PATH", clean_path(os.environ.get("PATH", "")))


@pytest.fixture
def world(cli_env, monkeypatch, fake_slurm):
    """A ready Grace home, the fake Slurm, and a `submit` whose waits are virtual (the CLI would sleep for real)."""
    ready_home(cli_env, monkeypatch)
    monkeypatch.setattr(cli, "submit", lambda plan, scripts, **kw: real_submit(plan, scripts, sleep=fake_slurm.sleep, clock=fake_slurm.clock, **kw))
    return cli_env, fake_slurm


def command(env, *extra, eval_data=("proteingym", "venusmuthub"), method="score-all", force=False) -> list[str]:
    """The student's `mpd submit ...` command; `extra` are the options they add."""
    return ["submit", *recipe(env.home, "--account", FAKE_ACCOUNT, *extra, eval_data=eval_data, method=method), "--yes", *(["--force"] if force else [])]


def campaigns(env) -> list[Path]:
    runs = env.home / "runs"
    return sorted(p for p in runs.iterdir() if p.is_dir()) if runs.is_dir() else []


def recipe_files(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): p.read_text(encoding="utf-8") for p in sorted(root.rglob("*.json"))
            if p.name != "plan.json" and p.name != "SUBMITTED.json"}


# ---------------------------------------------------------------- F3: the order a command was typed in is not part of the recipe
def test_a_spec_does_not_remember_the_order_it_was_typed_in():
    typed = spec(method="score-one", assays=("C", "A", "B"), eval_data=("venusmuthub", "proteingym"))
    sorted_ = spec(method="score-one", assays=("A", "B", "C"), eval_data=("proteingym", "venusmuthub"))
    assert typed == sorted_ and typed.assays == ("A", "B", "C") and typed.eval_data == ("proteingym", "venusmuthub")
    assert typed.to_json() == sorted_.to_json() and typed.campaign_id() == sorted_.campaign_id()
    assert json.loads(typed.to_json())["assays"] == ["A", "B", "C"] and typed.units == ("A", "B", "C")
    old = json.loads(sorted_.to_json())
    old.update(assays=["C", "A", "B"], eval_data=["venusmuthub", "proteingym"])             # a spec.json written in typing order
    assert RunSpec.from_json(json.dumps(old)) == sorted_


def test_a_dataset_or_an_assay_listed_twice_is_still_refused_after_sorting():
    with pytest.raises(SpecError, match="--eval-data lists a dataset twice"):
        spec(eval_data=("venusmuthub", "proteingym", "venusmuthub")).validate()
    with pytest.raises(SpecError, match="--assay lists an assay twice"):
        spec(method="score-one", assays=("B", "A", "B")).validate()


def test_the_recipe_files_come_out_equal_whatever_the_typing_order(tmp_path):
    kw = dict(method="score-one", assays=("ASSAY_C", "ASSAY_A", "ASSAY_B"))
    typed = make("grace", home=tmp_path / "home", spec_kw=dict(kw, eval_data=("venusmuthub", "proteingym")))
    sorted_ = make("grace", home=tmp_path / "home", spec_kw=dict(method="score-one", assays=("ASSAY_A", "ASSAY_B", "ASSAY_C"),
                                                                  eval_data=("proteingym", "venusmuthub")))
    assert typed.campaign_id == sorted_.campaign_id
    assert typed.spec.to_json() == sorted_.spec.to_json() and typed.resolved == sorted_.resolved
    assert [(s.full_name, s.tasks, s.task_offset) for s in typed.stages] == [(s.full_name, s.tasks, s.task_offset) for s in sorted_.stages]
    assert typed.resolved["assays"] == ["ASSAY_A", "ASSAY_B", "ASSAY_C"] and typed.resolved["eval_data"] == ["proteingym", "venusmuthub"]
    train = next(s for s in typed.stages if s.name == "train")
    assert [t["unit"] for t in train.tasks] == ["ASSAY_A", "ASSAY_B", "ASSAY_C"]


def test_force_accepts_the_same_recipe_typed_in_another_order(tmp_path):
    first = make("grace", home=tmp_path / "home", spec_kw=dict(method="score-one", assays=("ASSAY_A", "ASSAY_C")))
    write_campaign(first)
    write_marker(first.campaign_dir, {"campaign_id": first.campaign_id, "state": "submitted", "jobs": {}})
    before = recipe_files(first.campaign_dir)
    again = make("grace", home=tmp_path / "home", nonce="anothernonce0",
                 spec_kw=dict(method="score-one", assays=("ASSAY_C", "ASSAY_A"), eval_data=("venusmuthub", "proteingym")))
    assert again.campaign_dir == first.campaign_dir
    write_campaign(again, force=True)                                   # raised "a different recipe" before
    assert recipe_files(first.campaign_dir) == before and "anothernonce0" in (first.campaign_dir / "scripts" / "train.sbatch").read_text()


def test_the_default_teacher_backend_spelled_out_is_the_same_recipe(monkeypatch):
    assert spec(teacher_backend="live").to_json() == spec().to_json()                  # every teacher runs live today
    assert json.loads(spec(teacher_backend="live").to_json())["teacher_backend"] is None
    assert spec(teacher_backend="live").backend == "live" and spec(teacher_backend="live").campaign_id() == spec().campaign_id()
    offers_cache = dataclasses.replace(registry.get("esm2-3b"), teacher_backends=("cache", "live"))
    monkeypatch.setitem(registry.MODELS, "esm2-3b", offers_cache)                       # now the default is cache
    assert spec(teacher_backend="cache").to_json() == spec().to_json() and spec(teacher_backend="cache").campaign_id() == spec().campaign_id()
    live = spec(teacher_backend="live")
    assert json.loads(live.to_json())["teacher_backend"] == "live" and live.campaign_id() != spec().campaign_id()      # a real choice stays
    assert RunSpec.from_json(live.to_json()).backend == "live" and RunSpec.from_json(spec().to_json()).backend == "cache"


def test_an_unknown_teacher_is_reported_by_validate_not_by_to_json():
    bad = spec(teacher="nope", teacher_backend="live")
    assert json.loads(bad.to_json())["teacher_backend"] == "live"
    with pytest.raises(SpecError, match="unknown model"):
        bad.validate()


def test_the_datasets_in_another_order_continue_the_same_campaign(capsys, world):
    env, fs = world
    assert run_cli(capsys, *command(env))[0] == 0
    (root,) = campaigns(env)
    queued = len(fs.submitted())
    rc, out, err = run_cli(capsys, *command(env, eval_data=("venusmuthub", "proteingym"), force=True))
    assert rc == 0, err                                                                   # "different recipe", exit 2, before the fix
    assert campaigns(env) == [root] and len(fs.submitted()) == queued and f"nothing to submit: every stage of {root.name}" in out
    assert json.loads((root / "spec.json").read_text())["eval_data"] == ["proteingym", "venusmuthub"]


def test_the_assays_in_another_order_continue_the_same_campaign(capsys, world):
    env, fs = world
    assert run_cli(capsys, *command(env, "--assay", "ASSAY_C", "ASSAY_A", method="score-one", eval_data=("proteingym",)))[0] == 0
    (root,) = campaigns(env)
    queued = len(fs.submitted())
    rc, out, err = run_cli(capsys, *command(env, "--assay", "ASSAY_A", "ASSAY_C", method="score-one", eval_data=("proteingym",), force=True))
    assert rc == 0, err
    assert campaigns(env) == [root] and len(fs.submitted()) == queued and "nothing to submit" in out
    assert json.loads((root / "spec.json").read_text())["assays"] == ["ASSAY_A", "ASSAY_C"]
    assert [t["unit"] for t in json.loads((root / "tasks" / "train.json").read_text())] == ["ASSAY_A", "ASSAY_C"]


def test_the_default_backend_typed_out_continues_the_same_campaign(capsys, world):
    env, fs = world
    assert run_cli(capsys, *command(env))[0] == 0
    (root,) = campaigns(env)
    queued = len(fs.submitted())
    rc, out, err = run_cli(capsys, *command(env, "--teacher-backend", "live", force=True))
    assert rc == 0, err
    assert campaigns(env) == [root] and len(fs.submitted()) == queued and "nothing to submit" in out


def test_a_timeout_is_recovered_by_a_command_typed_in_another_order(capsys, world):
    """The scenario of the review: a new shell, the variables of the README gone, the datasets typed the other way round."""
    env, fs = world
    assert run_cli(capsys, *command(env, "--train-time", "01:00:00"))[0] == 0
    (root,) = campaigns(env)
    for rel in ("contexts/MANIFEST.json", "teacher/COMPLETE.json"):                       # prepare and teacher finished
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text("{}", encoding="utf-8")
    fs.set_state("prepare", "COMPLETED")
    fs.set_state("teacher", "COMPLETED")
    fs.set_state("train", "TIMEOUT", elapsed="01:00:07", exitcode="0:15", node="g003")
    for stage in ("score-base", "score", "evaluate"):
        fs.set_state(stage, "CANCELLED", reason="DependencyNeverSatisfied")
    queued = len(fs.submitted())
    rc, out, err = run_cli(capsys, *command(env, "--train-time", "10:00:00", eval_data=("venusmuthub", "proteingym"), force=True))
    assert rc == 0, err
    assert campaigns(env) == [root] and [j["stage"] for j in fs.submitted()[queued:]] == ["train", "score-base", "score", "evaluate"]
    assert "#SBATCH --time=10:00:00" in (root / "scripts" / "train.sbatch").read_text(encoding="utf-8")


# ---------------------------------------------------------------- F3: a real difference is shown in full
def submitted(tmp_path, **kw):
    plan = make("grace", home=tmp_path / "home", **kw)
    write_campaign(plan)
    write_marker(plan.campaign_dir, {"campaign_id": plan.campaign_id, "state": "submitted", "jobs": {}})
    return plan


def edit(path: Path, change) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def refusal(plan) -> str:
    """What the same command says when it is repeated with --force (with a new nonce, as a new shell would give)."""
    again = make("grace", home=plan.campaign_dir.parents[1], nonce="anothernonce0",
                 spec_kw={"method": plan.spec.method, "assays": plan.spec.assays})
    with pytest.raises(SubmitError) as exc:
        write_campaign(again, force=True)
    return " ".join(str(exc.value).split())


def test_a_value_that_really_differs_is_printed_in_full_not_cut_at_forty_characters(tmp_path):
    plan = submitted(tmp_path)
    long = "esm2-3b-" + "a-name-that-is-longer-than-forty-characters-" * 2
    edit(plan.campaign_dir / "resolved.json", lambda d: d.update(teacher=long))
    text = refusal(plan)
    assert f'resolved.json: teacher: "{long}" on disk, "esm2-3b" now' in text
    assert "so --force cannot continue it" in text and "change --seed to start a new campaign" in text


def test_two_lists_that_differ_only_in_order_say_so(tmp_path):
    plan = submitted(tmp_path)
    edit(plan.campaign_dir / "spec.json", lambda d: d.update(eval_data=["venusmuthub", "proteingym"]))      # as an older mpd wrote it
    text = refusal(plan)
    assert 'spec.json: eval_data: ["venusmuthub", "proteingym"] on disk, ["proteingym", "venusmuthub"] now (the same entries in another order)' in text


def test_two_long_lists_that_start_alike_show_where_they_differ(tmp_path):
    ids = ("A0A140D2T1_ZIKV_Sourisseau_2019", "GFP_AEQVI_Sarkisyan_2016")                   # the first 40 characters of both lists are equal
    plan = submitted(tmp_path, spec_kw=dict(method="score-one", assays=ids))
    edit(plan.campaign_dir / "spec.json", lambda d: d.update(assays=[ids[0], "GFP_AEQVI_Sarkisyan_2017"]))
    text = refusal(plan)
    assert f'spec.json: assays: ["{ids[0]}", "GFP_AEQVI_Sarkisyan_2017"] on disk, ["{ids[0]}", "{ids[1]}"] now' in text
    assert "another order" not in text


def test_task_lists_of_the_same_length_say_what_differs(tmp_path):
    plan = submitted(tmp_path, spec_kw=dict(method="score-one", assays=("ASSAY_A", "ASSAY_B")))
    path = plan.campaign_dir / "tasks" / "score.json"
    tasks = json.loads(path.read_text())
    assert len(tasks) == 4
    edit(path, lambda data: data[3].update(num_shards=9))
    other = {**tasks[3], "num_shards": 9}
    text = refusal(plan)
    assert (f"tasks/score.json: 4 tasks on disk, 4 now; the first difference is task 3: {json.dumps(other)} on disk, "
            f"{json.dumps(tasks[3])} now") in text
    path.write_text(json.dumps(list(reversed(tasks)), indent=1) + "\n")
    assert "tasks/score.json: 4 tasks on disk, 4 now (the same tasks in another order)" in refusal(plan)
    path.write_text(json.dumps(tasks[:3], indent=1) + "\n")
    assert "tasks/score.json: 3 tasks on disk, 4 now" in refusal(plan)


# ---------------------------------------------------------------- ml-03: --kl-weight belongs to the direct methods
@pytest.mark.parametrize("method", ["score-all", "score-one"])
@pytest.mark.parametrize("weight", [1.0, 0.0])
def test_kl_weight_is_refused_for_the_score_methods(method, weight, tmp_path):
    with pytest.raises(SpecError) as exc:
        make("grace", home=tmp_path / "home", spec_kw=dict(method=method, kl_weight=weight))
    text = str(exc.value)
    assert text.startswith("--kl-weight is only valid with --method direct-all or direct-one.") and "no KL term" in text
    assert text.endswith("Leave --kl-weight out.")
    with pytest.raises(SpecError, match="--kl-weight is only valid with"):
        spec(method=method, kl_weight=weight).validate()                                  # zero is a value too: it would be hashed into the id


@pytest.mark.parametrize("method", ["direct-all", "direct-one"])
def test_kl_weight_still_reaches_the_direct_methods(method):
    plan = make("grace", spec_kw=dict(method=method, kl_weight=0.25))
    assert plan.resolved["kl_weight"] == 0.25 and plan.campaign_id != make("grace", spec_kw=dict(method=method)).campaign_id
    assert make("grace", spec_kw=dict(method=method)).resolved["kl_weight"] == 1.0


def test_a_score_campaign_records_no_kl_weight_it_did_not_use():
    for method in ("score-all", "score-one"):
        assert make("grace", spec_kw=dict(method=method)).resolved["kl_weight"] == 0.0


def test_the_command_line_says_what_to_do_about_kl_weight(capsys, cli_env):
    rc, out, err = run_cli(capsys, "plan", *recipe(cli_env.home, "--kl-weight", "1.0"))
    assert rc == 2 and out == "" and "Traceback" not in err
    assert err.startswith("error: --kl-weight is only valid with --method direct-all or direct-one. ") and err.rstrip().endswith("Leave --kl-weight out.")
    rc, out, err = run_cli(capsys, "plan", *recipe(cli_env.home, "--kl-weight", "0.5", method="direct-all"))
    assert rc == 0, err
    assert "kl 0.5," in out


# ---------------------------------------------------------------- ml-04: --seed has to fit the generator that training seeds
@pytest.mark.parametrize("seed", [-1, MAX_SEED + 1, 20261109123000, 1.5])
def test_a_seed_the_training_stage_cannot_use_is_refused_by_the_plan(seed, tmp_path):
    with pytest.raises(SpecError) as exc:
        make("grace", home=tmp_path / "home", spec_kw=dict(seed=seed))
    assert str(exc.value).startswith(f"--seed must be a whole number from 0 to 4294967295, not {seed}. ")
    assert "for example a date such as 20261109" in str(exc.value)
    if isinstance(seed, int):
        with pytest.raises(ValueError, match="Seed must be between 0 and 2"):             # the reason for the limit: training calls this
            np.random.seed(seed)


@pytest.mark.parametrize("seed", [0, 7, 20260922, MAX_SEED])
def test_every_seed_the_generator_takes_is_planned(seed):
    np.random.seed(seed)                                    # what the training stage does with it
    assert f"seed{seed}__" in make("grace", spec_kw=dict(seed=seed)).campaign_id


def test_a_date_time_seed_is_refused_before_anything_is_queued(capsys, world):
    env, fs = world
    rc, out, err = run_cli(capsys, *command(env, "--seed", "20261109123000"))
    assert rc == 2 and err.startswith("error: --seed must be a whole number from 0 to 4294967295, not 20261109123000.")
    assert "Traceback" not in err and fs.submitted() == [] and campaigns(env) == []


# ---------------------------------------------------------------- G7: only a packaged profile name is exported to the jobs
def yaml_profile(tmp_path, name="mycluster") -> str:
    """The packaged Grace profile under another name, loaded from a YAML path as docs/CLUSTERS.md describes."""
    data = yaml.safe_load((CLUSTERS_DIR / "grace.yaml").read_text(encoding="utf-8"))
    data["name"] = name
    path = tmp_path / f"{name}.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    assert Cluster.load(str(path)).name == name and name not in available_clusters()
    return str(path)


def test_a_profile_loaded_from_a_yaml_file_is_not_exported_to_the_jobs(tmp_path):
    plan = make(yaml_profile(tmp_path), home=tmp_path / "home")
    scripts = render_all(plan)
    assert len(scripts) == 6
    for name, text in scripts.items():
        assert "MPD_CLUSTER" not in text and "mycluster" not in text, name
        home = shlex.quote(str(plan.options.home))
        assert f"export MPD_HOME={home} HF_HOME={shlex.quote(str(plan.options.home / 'hf_cache'))} XDG_CACHE_HOME=" in text, name
        assert bash_n(text, tmp_path).returncode == 0, name


@pytest.mark.parametrize("cluster", available_clusters())
def test_the_packaged_profiles_are_exported_as_before(cluster, tmp_path):
    plan = make(cluster, home=tmp_path / "home")
    for name, text in render_all(plan).items():
        assert f"export MPD_HOME={shlex.quote(str(plan.options.home))} MPD_CLUSTER={cluster} HF_HOME=" in text, name


def test_the_fetch_command_a_job_prints_can_be_run(tmp_path, monkeypatch):
    """The job of a YAML profile has MPD_HOME but no MPD_CLUSTER, even when the submitting shell had one (the activate file of
    scripts/setup_env.sh exports the path it was given), so the fix command it prints is one `mpd fetch` accepts."""
    profile = yaml_profile(tmp_path)
    plan = make(profile, home=tmp_path / "home")
    run = ScriptRun(tmp_path, plan, "prepare", module=None, env={"MPD_CLUSTER": profile})
    assert run.run().returncode == 0
    job = {k: v for k, v in run.job_env().items() if k.startswith("MPD_")}
    assert job["MPD_HOME"] == str(plan.options.home) and "MPD_CLUSTER" not in job
    for name in [n for n in os.environ if n.startswith("MPD_")]:
        monkeypatch.delenv(name)
    for name, value in job.items():
        monkeypatch.setenv(name, value)
    with pytest.raises(models.ModelNotReady) as exc:
        models.resolve("esm2-8m")
    text = str(exc.value)
    assert "mpd fetch model esm2-8m" in text and "--cluster" not in text and f"--home {plan.options.home}" in text
