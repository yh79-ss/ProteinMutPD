"""The rendered sbatch scripts: their structure for every method, cluster and stage kind, and what the prologue does when run.

Layers: (1) text and `bash -n` for the whole matrix; (2) full-text goldens in tests/golden_render (rewrite them with
UPDATE_GOLDEN=1 after a deliberate change); (3) the scripts executed under bash with stand-ins for module, nvidia-smi and python.
"""

import dataclasses
import getpass
import os
import re
import subprocess
from pathlib import Path

import pytest

from fx_render import (HOME, METHODS, NONCE, ScriptRun, bash_n, clean_mpd_env, directives, make,  # noqa: F401
                       render_all, replace_options, spec, stage)
from fx_slurm import fake_slurm  # noqa: F401
from mpd.slurm.cluster import Cluster, format_walltime
from mpd.slurm.plan import PlanError, PlanOptions, make_plan
from mpd.slurm.render import render_script

GOLDEN = Path(__file__).parent / "golden_render"
CLUSTERS = ("grace", "vision")
STAGE_KINDS = {"prepare", "teacher", "train", "score-base", "score", "evaluate"}


# ---------------------------------------------------------------- the expected shape, written out independently of render.py
# what the prologue drops from the environment of the shell that ran sbatch, before the planned exports put theirs back
SCRUB = '''for _v in $(compgen -e); do case $_v in MPD_*) unset "$_v";; esac; done; unset _v PYTHONPATH PYTHONHOME'''


def expected_header(plan, st) -> list[str]:
    cluster = plan.options.cluster
    gpu = cluster.gpus[st.gpu] if st.gpu else None
    out = f"{plan.campaign_dir}/logs/" + ("%x_%A_%a" if st.is_array else "%x_%j")
    lines = [f"#SBATCH --job-name=mpd-{st.full_name}-{plan.spec.student.replace('-', '')[:10]}",
             f"#SBATCH --comment=mpd:{plan.campaign_id}:{plan.nonce}:{st.full_name}",
             f"#SBATCH --time={format_walltime(st.walltime_s)}", "#SBATCH --nodes=1", "#SBATCH --ntasks=1",
             f"#SBATCH --cpus-per-task={st.cpus}", f"#SBATCH --mem={st.mem_gb}G"]
    if gpu:
        lines.append(f"#SBATCH --gres=gpu:{gpu.gres}:{st.gpus}")
        if gpu.partition:
            lines.append(f"#SBATCH --partition={gpu.partition}")
    elif cluster.cpu_partition:
        lines.append(f"#SBATCH --partition={cluster.cpu_partition}")
    if plan.options.account:
        lines.append(f"#SBATCH --account={plan.options.account}")
    if st.is_array:
        lines.append(f"#SBATCH --array=0-{len(st.tasks) - 1}%{plan.options.array_throttle or cluster.array_throttle}")
    lines += [f"#SBATCH --output={out}.out", f"#SBATCH --error={out}.err", "#SBATCH --open-mode=append"]
    lines += [f"#SBATCH {flag}" for flag in (*cluster.sbatch_extra, *plan.options.extra_sbatch)]
    return lines


def in_order(lines: list[str], needles: list[str]) -> list[int]:
    """Index of the first line starting with each needle, which must be strictly increasing."""
    found, start = [], 0
    for needle in needles:
        idx = next((i for i in range(start, len(lines)) if lines[i].startswith(needle)), None)
        assert idx is not None, f"{needle!r} missing or out of order"
        found.append(idx)
        start = idx + 1
    return found


def assert_structure(plan, st, text: str, tmp_path: Path) -> None:
    cluster, gpu = plan.options.cluster, plan.options.cluster.gpus.get(st.gpu or "")
    where = f"{cluster.name}/{plan.spec.method}/{st.full_name}"
    assert text.endswith("\n") and not text.endswith("\n\n") and "\r" not in text, where
    check = bash_n(text, tmp_path)
    assert check.returncode == 0, f"{where}: {check.stderr}"
    lines = text.split("\n")
    assert lines[0] == "#!/bin/bash" and directives(text) == expected_header(plan, st), where
    assert lines[1:1 + len(directives(text))] == directives(text), f"{where}: directives are not one block on top"

    modules = plan.modules
    prologue = ["set -eo pipefail"]
    prologue += (["type module >/dev/null 2>&1 || source /etc/profile", "type module >/dev/null 2>&1 || {", "module purge",
                  "module load " + " ".join(modules)] if modules else ["if type module >/dev/null 2>&1; then module purge; fi"])
    prologue += [SCRUB, "set -u", "export MPD_HOME=", "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1"]
    if gpu:
        prologue += ["export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MPD_REQUIRE_CUDA=1", "nvidia-smi -L || {"]
    prologue += ['echo "MPD_JOB_START host=$(hostname) job=${SLURM_JOB_ID:-} stage=' + st.full_name + '"', "cd ",
                 plan.options.python if st.launcher != "ddp" else f"{plan.options.python} -m torch.distributed.run",
                 'echo "MPD_STAGE_DONE ' + st.full_name + '"']
    in_order(lines, prologue)

    # forbidden: the flags that broke or hung Grace jobs, anything relative to $HOME, a user name nobody passed
    for bad in ("--export", "--get-user-env", "--wrap", "$HOME", "${HOME", "~", "--requeue"):
        assert bad not in text, f"{where}: {bad}"
    user = getpass.getuser()
    assert len(user) < 5 or user not in text
    if not plan.options.account:
        assert "--account" not in text, where

    # the interpreter line: always a task index, an array uses the array id, a lone chunk the chunk's offset
    cmd = next(ln for ln in lines if f"-m {st.module} " in ln)
    index = f"$((SLURM_ARRAY_TASK_ID + {st.task_offset}))" if st.is_array else str(st.task_offset)
    assert cmd.endswith(f"--stage {st.name} --task-index {index}"), f"{where}: {cmd}"
    assert cmd.count("--task-index") == 1 and f"--campaign-dir {plan.campaign_dir} " in cmd
    if st.launcher == "ddp":
        assert f"--standalone --nnodes=1 --nproc_per_node={st.gpus} -m {st.module}" in cmd and "bin/torchrun" not in cmd

    # GPU rails only on GPU stages, typed GRES exactly as the profile spells it
    if gpu:
        assert f"#SBATCH --gres=gpu:{gpu.gres}:{st.gpus}" in text, where
    else:
        assert "--gres" not in text, where
    assert ("MPD_REQUIRE_CUDA=1" in text) == bool(gpu) and ("nvidia-smi" in text) == bool(gpu) and ("exit 31" in text) == bool(gpu)
    assert text.count("MPD_JOB_START") == 1 and text.count("MPD_STAGE_DONE") == 1


def matrix_plans():
    for cluster in CLUSTERS:
        for method in METHODS:
            yield pytest.param(make(cluster, spec_kw=dict(method=method)), id=f"{cluster}-{method}")


@pytest.mark.parametrize("plan", list(matrix_plans()))
def test_every_stage_of_every_method_on_both_clusters(plan, tmp_path):
    scripts = render_all(plan)
    assert {s.name for s in plan.stages} == STAGE_KINDS
    for st in plan.stages:
        assert_structure(plan, st, scripts[st.full_name], tmp_path)
    one = plan.spec.scope == "one"
    train = stage(plan, "train")
    assert train.is_array == one and (train.launcher == "ddp") == (plan.resolved["world_size"] > 1)
    assert stage(plan, "score-base").is_array and stage(plan, "score").is_array and not stage(plan, "prepare").is_array


@pytest.mark.parametrize("cluster,kw,ddp", [("grace", {}, False), ("grace", {"train_gpus": 2}, True), ("vision", {"train_gpus": 1}, False),
                                           ("vision", {}, True)])
def test_one_gpu_and_ddp_training_scripts(cluster, kw, ddp, tmp_path):
    plan = make(cluster, spec_kw=dict(method="direct-all", **kw))
    train = stage(plan, "train")
    assert (train.launcher == "ddp") == ddp
    assert_structure(plan, train, render_script(plan, train), tmp_path)
    text = render_script(plan, train)
    assert ("torch.distributed.run" in text) == ddp
    assert f"OMP_NUM_THREADS={train.cpus // train.gpus}" in text


def test_a_stage_with_one_task_is_not_an_array(tmp_path):
    plan = make("grace", spec_kw=dict(eval_data=("venusmuthub",)))
    base = stage(plan, "score-base")
    assert len(base.tasks) == 1 and not base.is_array
    text = render_script(plan, base)
    assert "--array" not in text and "%x_%j.out" in text and text.rstrip().splitlines()[-2].endswith("--task-index 0")
    assert_structure(plan, base, text, tmp_path)


def test_job_name_and_comment_tag(tmp_path):
    plan = make("grace")
    text = render_script(plan, stage(plan, "score-base"))
    assert "#SBATCH --job-name=mpd-score-base-esm2650m" in text
    assert re.search(rf"^#SBATCH --comment=mpd:{re.escape(plan.campaign_id)}:{NONCE}:score-base$", text, re.M)
    assert plan.campaign_id.startswith(plan.spec.campaign_id() + "__")
    assert "#SBATCH --open-mode=append" in text


def test_a_plan_that_was_not_built_by_make_plan_does_not_render():
    plan = make("grace")
    with pytest.raises(PlanError, match="nonce"):
        render_script(dataclasses.replace(plan, nonce=""), stage(plan, "train"))
    with pytest.raises(PlanError, match="Python interpreter"):
        render_script(replace_options(plan, python=None), stage(plan, "train"))


# ---------------------------------------------------------------- Grace header: no --export, no --get-user-env
def test_grace_script_follows_the_documented_conventions():
    plan = make("grace", spec_kw=dict(train_gpus=2))
    text = render_script(plan, stage(plan, "train"))
    for needle in ("#SBATCH --gres=gpu:a100:2", "#SBATCH --partition=gpu", "#SBATCH --account=123456789012",
                   "#SBATCH --cpus-per-task=48", "#SBATCH --mem=360G", "HF_HUB_OFFLINE=1", "--nproc_per_node=2"):
        assert needle in text, needle
    assert "--export" not in text and "--get-user-env" not in text            # Slurm 25.05 rejects =L; no verified Grace job used either
    assert "module load" not in text                                           # the profile needs no module by default
    assert text.count("#SBATCH") == len(directives(text))


def test_a40_jobs_name_their_partition():
    plan = make("grace", gpu="a40")
    assert "#SBATCH --partition=gpu-a40" in render_script(plan, stage(plan, "train"))
    assert "#SBATCH --gres=gpu:a40:1" in render_script(plan, stage(plan, "train"))


def test_array_script_has_throttle_and_offset():
    plan = make("grace", spec_kw=dict(method="direct-one"))
    text = render_script(plan, stage(plan, "train"))
    assert "#SBATCH --array=0-2%8" in text and "SLURM_ARRAY_TASK_ID + 0" in text
    plan = make("grace", spec_kw=dict(method="direct-one"), array_throttle=3)
    assert "#SBATCH --array=0-2%3" in render_script(plan, stage(plan, "train"))


def test_cpu_stage_has_no_gres_or_partition_on_grace():
    plan = make("grace")
    text = render_script(plan, stage(plan, "prepare"))
    assert "--gres" not in text and "--partition" not in text                   # Grace routes CPU jobs by walltime
    assert "nvidia-smi" not in text and "MPD_REQUIRE_CUDA" not in text


def test_vision_script_uses_its_qos_modules_and_case_sensitive_gres():
    plan = make("vision")
    text = render_script(plan, stage(plan, "train"))
    assert "#SBATCH --qos=standard" in text and "module load CUDA/12.8.0 ||" in text
    assert "#SBATCH --gres=gpu:H200:4" in text and "h200" not in text.replace("MPD", "")
    assert text.index("module purge") < text.index("module load") < text.index("compgen -e") < text.index("set -u")


def test_cpu_gres_option_gives_cpu_stages_a_gpu_request_on_sites_that_demand_one():
    cluster = dataclasses.replace(Cluster.load("vision"), cpu_gres="gpu:1")
    plan = make_plan(spec(), PlanOptions(cluster=cluster, home=HOME, python="/env/bin/python", nonce=NONCE))
    assert "#SBATCH --gres=gpu:1\n" in render_script(plan, stage(plan, "prepare"))
    assert "#SBATCH --gres=gpu:1\n" in render_script(plan, stage(plan, "evaluate"))
    assert "--gres=gpu:H200" in render_script(plan, stage(plan, "train")) and "--gres=gpu:1\n" not in render_script(plan, stage(plan, "train"))
    assert "MPD_REQUIRE_CUDA" not in render_script(plan, stage(plan, "prepare"))


# ---------------------------------------------------------------- chunked arrays: the single-task last chunk (B2, D8)
@pytest.mark.parametrize("n_assays,chunks", [(1000, [(1000, 0)]), (1001, [(1000, 0), (1, 1000)]),
                                             (2001, [(1000, 0), (1000, 1000), (1, 2000)])])
def test_task_index_of_every_chunk(n_assays, chunks, tmp_path):
    assays = tuple(f"assay{i}" for i in range(n_assays))
    plan = make("grace", spec_kw=dict(method="direct-one", assays=assays, eval_data=("venusmuthub",)))
    parts = [s for s in plan.stages if s.name == "train"]
    assert [(len(p.tasks), p.task_offset) for p in parts] == chunks
    for part in parts:
        text = render_script(plan, part)
        if len(part.tasks) > 1:
            assert f"--task-index $((SLURM_ARRAY_TASK_ID + {part.task_offset}))" in text
            assert f"#SBATCH --array=0-{len(part.tasks) - 1}%8" in text
        else:
            assert text.rstrip().splitlines()[-2].endswith(f"--stage train --task-index {part.task_offset}")
            assert "--array" not in text
        assert_structure(plan, part, text, tmp_path)
    assert [s.full_name for s in parts] == ["train"] + [f"train.{k}" for k in range(1, len(chunks))]
    indices = sorted(p.task_offset + i for p in parts for i in range(len(p.tasks)))
    assert indices == list(range(n_assays))                                      # every task is run exactly once


def test_the_single_task_chunk_runs_the_right_task(tmp_path):
    assays = tuple(f"assay{i}" for i in range(1001))
    plan = make("grace", home=tmp_path / "home", spec_kw=dict(method="direct-one", assays=assays, eval_data=("venusmuthub",)))
    run = ScriptRun(tmp_path, plan, "train.1", module=None)
    done = run.run()
    assert done.returncode == 0, done.stderr
    assert run.args()[-2:] == ["--task-index", "1000"]
    first = ScriptRun(tmp_path, plan, "train", module=None, env={"SLURM_ARRAY_TASK_ID": "999"})
    assert first.run().returncode == 0 and first.args()[-2:] == ["--task-index", "999"]


# ---------------------------------------------------------------- the prologue, executed
def test_gpu_stage_prologue_sets_a_clean_environment(tmp_path):
    plan = make("grace", home=tmp_path / "home", spec_kw=dict(train_gpus=2), extra_modules=("Anaconda3/2024.02-1",))
    stray = dict(MPD_ALLOW_HUB="1", MPD_STRUCTURE_LEGACY_MASK="1", MPD_CLUSTER="not-this-one", MPD_HOME="/stray/home",
                 MPD_ACCOUNT="999", MPD_SOMETHING_ELSE="x", LD_LIBRARY_PATH="/stray/lib", MY_OWN_SETTING="kept")
    run = ScriptRun(tmp_path, plan, "train", env=stray)
    done = run.run()
    assert done.returncode == 0, done.stderr
    env = run.job_env()
    assert "PYTHONPATH" not in env and "PYTHONHOME" not in env                    # stray paths of the submitting shell are gone
    for name in ("MPD_ALLOW_HUB", "MPD_STRUCTURE_LEGACY_MASK", "MPD_ACCOUNT", "MPD_SOMETHING_ELSE"):
        assert name not in env, f"{name} of the submitting shell reached the job"
    assert env["MPD_CLUSTER"] == "grace" and env["MPD_HOME"] == str(plan.options.home)      # the plan's, not the shell's
    assert env["LD_LIBRARY_PATH"] == "/stray/lib" and env["MY_OWN_SETTING"] == "kept"       # everything else is inherited, as Slurm does
    home = str(plan.options.home)
    assert env["MPD_HOME"] == home and env["HF_HOME"] == f"{home}/hf_cache" and env["XDG_CACHE_HOME"] == f"{home}/cache"
    assert env["TORCH_HOME"] == f"{home}/cache/torch" and env["TRITON_CACHE_DIR"] == f"{home}/cache/triton"
    assert env["MPLCONFIGDIR"] == f"{home}/cache/mpl"
    for name, value in dict(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1",
                            PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
                            MPD_REQUIRE_CUDA="1", OMP_NUM_THREADS="24").items():
        assert env[name] == value, name
    assert run.module_calls() == ["purge", "load Anaconda3/2024.02-1"]
    args = run.args()
    assert args[:5] == ["-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc_per_node=2"]
    assert args[-6:] == ["--campaign-dir", str(plan.campaign_dir), "--stage", "train", "--task-index", "0"]
    assert re.search(r"^MPD_JOB_START host=\S+ job=4242 stage=train$", done.stdout, re.M)
    assert done.stdout.rstrip().endswith("MPD_STAGE_DONE train") and "GPU 0: NVIDIA Fake" in done.stdout


def test_array_task_id_becomes_the_task_index(tmp_path):
    plan = make("grace", home=tmp_path / "home", spec_kw=dict(method="score-one"))
    run = ScriptRun(tmp_path, plan, "score", module=None)
    assert run.run().returncode == 0
    assert run.args()[-2:] == ["--task-index", "3"]                               # SLURM_ARRAY_TASK_ID=3 plus offset 0


def test_no_gpu_means_exit_31_and_nothing_runs(tmp_path):
    plan = make("grace", home=tmp_path / "home")
    run = ScriptRun(tmp_path, plan, "teacher", gpu_ok=False)
    done = run.run()
    assert done.returncode == 31 and "MPD_NO_GPU: no usable GPU on" in done.stderr and "--exclude=" in done.stderr
    assert not (run.out / "args").exists() and "MPD_JOB_START" not in done.stdout


def test_cpu_stages_do_not_need_a_gpu(tmp_path):
    plan = make("grace", home=tmp_path / "home")
    run = ScriptRun(tmp_path, plan, "prepare", gpu_ok=False)
    assert run.run().returncode == 0 and "MPD_REQUIRE_CUDA" not in run.job_env()


def test_module_purge_runs_even_without_modules_and_is_skipped_without_a_module_command(tmp_path):
    plan = make("grace", home=tmp_path / "home")
    with_module = ScriptRun(tmp_path, plan, "prepare", module="log")
    assert with_module.run().returncode == 0 and with_module.module_calls() == ["purge"]
    without = ScriptRun(tmp_path, plan, "evaluate", module=None)
    assert without.run().returncode == 0 and without.module_calls() == []
    assert "PYTHONPATH" not in without.job_env()


def test_a_module_that_fails_to_load_stops_the_job_loudly(tmp_path):
    plan = make("grace", home=tmp_path / "home", extra_modules=("Missing/1.0", "Other/2"))
    run = ScriptRun(tmp_path, plan, "train", module_fails_on="Missing/1.0")
    done = run.run()
    assert done.returncode == 1 and "MPD_MODULE_LOAD_FAILED Missing/1.0 Other/2" in done.stderr
    assert not (run.out / "args").exists()


def test_modules_without_a_module_command_fail_with_a_message(tmp_path):
    plan = make("grace", home=tmp_path / "home", extra_modules=("Anaconda3/2024.02-1",))
    run = ScriptRun(tmp_path, plan, "prepare", module=None)
    # a developer's /etc/profile may define `module`; point the script at a profile that does not exist
    run.script.write_text(run.script.read_text().replace("source /etc/profile", "source /nonexistent/profile"))
    done = run.run()
    assert done.returncode == 1 and "MPD_MODULE_LOAD_FAILED the module command is not available (Anaconda3/2024.02-1)" in done.stderr
    assert not (run.out / "args").exists()


def test_set_u_comes_after_the_module_block(tmp_path):
    """The stand-in `module` reads an unset variable, as Lmod's init does on Grace: it must run before `set -u`."""
    plan = make("grace", home=tmp_path / "home", extra_modules=("A/1",))
    run = ScriptRun(tmp_path, plan, "prepare")
    assert run.run().returncode == 0
    text = run.script.read_text()
    assert text.index("module load") < text.index("\nset -u\n") < text.index("export MPD_HOME")


def test_set_u_catches_a_typo_after_the_prologue(tmp_path):
    plan = make("grace", home=tmp_path / "home")
    run = ScriptRun(tmp_path, plan, "prepare", module=None)
    run.script.write_text(run.script.read_text().replace("echo \"MPD_JOB_START", 'echo "$NOT_DEFINED_ANYWHERE" ; echo "MPD_JOB_START'))
    done = run.run()
    assert done.returncode != 0 and "unbound variable" in done.stderr


# ---------------------------------------------------------------- forwarded and user environment
def test_forwarding_allowlist_and_absolute_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name, value in dict(MPD_MODEL_ESM2_650M="weights/esm2-650m", MPD_MODELS_DIR="models", MPD_FOLDSEEK="tools/foldseek/bin/foldseek",
                            MPD_DTYPE="float32", MPD_TF32="1", MPD_KEEP_ALL_CHECKPOINTS="1", MPD_3DI_ENGINE="foldseek",
                            MPD_CKPT_EVERY="100", MPD_STRUCTURE_LEGACY_MASK="1", MPD_HOME="/should/not/leak", MPD_CLUSTER="not-this-one",
                            MPD_ACCOUNT="999", MPD_PYTHON="/nope/python", MPD_ALLOW_HUB="1", MPD_REQUIRE_CUDA="0",
                            HF_HUB_OFFLINE="0", TRANSFORMERS_OFFLINE="0").items():
        monkeypatch.setenv(name, value)
    plan = make("grace")
    text = render_script(plan, stage(plan, "score-base"))
    exported = {m.group(1): m.group(2) for m in re.finditer(r"^export (MPD_[A-Z0-9_]+)=(\S+)$", text, re.M)}
    assert exported == {"MPD_3DI_ENGINE": "foldseek", "MPD_CKPT_EVERY": "100", "MPD_DTYPE": "float32",
                        "MPD_FOLDSEEK": f"{tmp_path}/tools/foldseek/bin/foldseek", "MPD_KEEP_ALL_CHECKPOINTS": "1",
                        "MPD_MODELS_DIR": f"{tmp_path}/models", "MPD_MODEL_ESM2_650M": f"{tmp_path}/weights/esm2-650m", "MPD_TF32": "1"}
    assert "MPD_STRUCTURE_LEGACY_MASK" not in text                                           # only through --env, see the next test
    for private in ("/should/not/leak", "not-this-one", "MPD_ALLOW_HUB", "/nope/python", "MPD_ACCOUNT"):
        assert private not in text, private
    assert " MPD_CLUSTER=grace " in text                                                      # the plan's cluster, for the messages of a job
    assert text.count("MPD_REQUIRE_CUDA=1") == 1 and "MPD_REQUIRE_CUDA=0" not in text      # the planner decides, not the shell
    assert "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 " in text and "HF_HUB_OFFLINE=0" not in text      # jobs stay offline
    assert "#SBATCH --account=123456789012" in text                                           # MPD_ACCOUNT is the cli's default only if no --account


def test_the_legacy_mask_and_other_numerics_flags_reach_a_job_through_env(tmp_path):
    plan = make("grace", home=tmp_path / "home", extra_env={"MPD_STRUCTURE_LEGACY_MASK": "1"})
    for text in render_all(plan).values():
        assert "\nexport MPD_STRUCTURE_LEGACY_MASK=1\n" in text
    run = ScriptRun(tmp_path, plan, "prepare", module=None)
    assert run.run().returncode == 0 and run.job_env()["MPD_STRUCTURE_LEGACY_MASK"] == "1"


def test_a_job_can_not_be_allowed_to_reach_the_hub():
    with pytest.raises(PlanError, match="MPD_ALLOW_HUB cannot be set for jobs: it is for login nodes and laptops"):
        make("grace", extra_env={"MPD_ALLOW_HUB": "1"})


def test_a_bad_account_in_the_environment_is_named_as_such(monkeypatch):
    monkeypatch.setenv("MPD_ACCOUNT", "no good")
    with pytest.raises(PlanError, match="MPD_ACCOUNT 'no good'"):
        make("grace", account=None)
    with pytest.raises(PlanError, match="--account 'bad one'"):
        make("grace", account="bad one")


def test_account_from_the_environment_when_no_flag(monkeypatch):
    monkeypatch.setenv("MPD_ACCOUNT", "env_acct_1")
    plan = make("grace", account=None)
    assert "#SBATCH --account=env_acct_1" in render_script(plan, stage(plan, "train"))
    assert not any("no --account" in w for w in plan.warnings)
    plan = make("grace", account="flag_acct_2")
    assert "#SBATCH --account=flag_acct_2" in render_script(plan, stage(plan, "train"))               # the flag beats the variable


HOSTILE = ["'; touch PWNED; '", "$(touch PWNED)", "`touch PWNED`", "a b\tc", "\"double\" 'single' \\back", "*", "~/x", "${HOME}",
           "café ü", "=leading=equals", "#not a comment", "x" * 4000, "$((1+1))", "!history", ""]


@pytest.mark.parametrize("value", HOSTILE)
def test_user_environment_values_arrive_unchanged_and_nothing_runs(value, tmp_path):
    plan = make("grace", home=tmp_path / "home", extra_env={"MY_SETTING": value})
    run = ScriptRun(tmp_path, plan, "evaluate", module=None)
    done = run.run()
    assert done.returncode == 0, done.stderr
    assert run.job_env()["MY_SETTING"] == value
    assert not (run.plan.campaign_dir / "PWNED").exists() and not (tmp_path / "PWNED").exists() and not Path("PWNED").exists()


def test_forwarded_values_are_quoted_too(tmp_path, monkeypatch):
    monkeypatch.setenv("MPD_MODEL_ESM2_8M", str(tmp_path / "w e ird; touch PWNED"))
    plan = make("grace", home=tmp_path / "home")
    run = ScriptRun(tmp_path, plan, "evaluate", module=None)
    assert run.run().returncode == 0
    assert run.job_env()["MPD_MODEL_ESM2_8M"] == str(tmp_path / "w e ird; touch PWNED") and not Path("PWNED").exists()


def test_user_environment_comes_last_and_can_override_defaults(tmp_path):
    plan = make("grace", home=tmp_path / "home", extra_env={"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128", "NCCL_DEBUG": "INFO"})
    run = ScriptRun(tmp_path, plan, "train")
    assert run.run().returncode == 0
    assert run.job_env()["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:128" and run.job_env()["NCCL_DEBUG"] == "INFO"
    lines = run.script.read_text().splitlines()
    assert lines.index("export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128") > lines.index(
        "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MPD_REQUIRE_CUDA=1")


@pytest.mark.parametrize("name,value,message", [
    ("1BAD", "x", "not an environment variable name"), ("BAD NAME", "x", "not an environment variable name"),
    ("A=B", "x", "not an environment variable name"), ("", "x", "not an environment variable name"),
    ("FOO", "line1\nline2", "one line"), ("FOO", "carriage\rreturn", "one line"), ("FOO", "nul\x00byte", "one line"),
    ("FOO", "x" * 5000, "at most"), ("MPD_ALLOW_HUB", "1", "login nodes"), ("FOO;rm", "x", "not an environment variable name")])
def test_bad_environment_pairs_are_plan_errors(name, value, message):
    with pytest.raises(PlanError, match=message):
        make("grace", extra_env={name: value})


def test_forwarded_variable_with_a_newline_is_refused(monkeypatch):
    monkeypatch.setenv("MPD_DTYPE", "float32\nexport EVIL=1")
    with pytest.raises(PlanError, match="MPD_DTYPE"):
        make("grace")


# ---------------------------------------------------------------- values that land on #SBATCH lines
@pytest.mark.parametrize("account", ["1\n#SBATCH --mem=1", "a b", "x;y", "", "$(id)", "acct\r", "a/b", "a=b"])
def test_a_hostile_account_never_reaches_a_directive(account):
    plan = make("grace")
    bad = replace_options(plan, account=account)
    if account:
        with pytest.raises(PlanError, match="account"):
            render_script(bad, stage(plan, "train"))
        with pytest.raises(PlanError, match="account"):
            make("grace", account=account)
    else:                                          # empty = not given: the profile (none) or the Slurm default applies
        assert "--account" not in render_script(bad, stage(plan, "train"))


@pytest.mark.parametrize("account", ["123456789012", "7" * 13, "123456", "proj_1.a-b", "abc"])
def test_account_has_no_length_rule(account):
    plan = make("grace", account=account)
    assert f"#SBATCH --account={account}\n" in render_script(plan, stage(plan, "train"))


@pytest.mark.parametrize("flag", ["--wrap=rm -rf ~", "--exclude=g001\n#SBATCH --mem=1", "-w g001", "--exclude g001", "exclude=g001",
                                  "--mail-user=a b", "--x;y", "--", "--output=/tmp/x", "--comment=hi", "--job-name=x", "--array=1-3",
                                  "--dependency=afterok:1", "--chdir=/tmp", "--open-mode=truncate", ""])
def test_unsafe_or_mpd_owned_sbatch_flags_are_refused(flag):
    with pytest.raises(PlanError, match="sbatch-arg"):
        make("grace", extra_sbatch=(flag,))
    plan = make("grace")
    with pytest.raises(PlanError):
        render_script(replace_options(plan, extra_sbatch=(flag,)), stage(plan, "train"))


def test_ordinary_sbatch_flags_are_written_after_the_managed_directives():
    plan = make("grace", extra_sbatch=("--exclude=g001", "--mail-type=END,FAIL", "--constraint=a100|a40", "--signal=B:USR1@600"))
    text = render_script(plan, stage(plan, "train"))
    tail = directives(text)[-4:]
    assert tail == ["#SBATCH --exclude=g001", "#SBATCH --mail-type=END,FAIL", "#SBATCH --constraint=a100|a40",
                    "#SBATCH --signal=B:USR1@600"]
    assert directives(text)[-5] == "#SBATCH --open-mode=append"


def test_overriding_a_planned_resource_is_allowed_but_warned():
    plan = make("grace", extra_sbatch=("--mem=100G",))
    assert any("overrides a value mpd chose" in w for w in plan.warnings)
    assert "#SBATCH --mem=100G" in render_script(plan, stage(plan, "train"))


@pytest.mark.parametrize("module", ["bad name", "a;b", "$(x)", "a\nb", "-x", "", "a b"])
def test_hostile_module_names_are_refused(module):
    with pytest.raises(PlanError, match="module"):
        make("grace", extra_modules=(module,))
    plan = make("grace")
    with pytest.raises(PlanError):
        render_script(replace_options(plan, extra_modules=(module,)), stage(plan, "train"))


def test_a_home_that_slurm_cannot_carry_is_refused():
    for bad in ("/nonexistent/my home", "/nonexistent/a%b", "/nonexistent/$x", "/nonexistent/a;b", "/nonexistent/'q'", "/nonexistent/#x"):
        with pytest.raises(PlanError, match="Slurm directives"):
            make("grace", home=bad)
    make("grace", home="/nonexistent/mpd-home_1.2+x@y:z,w=v")


# ---------------------------------------------------------------- full-text goldens
def golden_cases():
    grace = make("grace", spec_kw=dict(method="score-all", train_gpus=2), extra_modules=("Anaconda3/2024.02-1",),
                 extra_env={"NCCL_DEBUG": "WARN"}, extra_sbatch=("--exclude=g001",))
    vision = make("vision", spec_kw=dict(method="direct-one", assays=("ASSAY_A", "ASSAY_B", "ASSAY_C")))
    return {"grace_train_ddp.sbatch": render_script(grace, stage(grace, "train")),
            "vision_score_array.sbatch": render_script(vision, stage(vision, "score")),
            "vision_prepare.sbatch": render_script(vision, stage(vision, "prepare"))}


@pytest.mark.parametrize("name", ["grace_train_ddp.sbatch", "vision_score_array.sbatch", "vision_prepare.sbatch"])
def test_full_text_goldens(name):
    text = golden_cases()[name]
    path = GOLDEN / name
    if os.environ.get("UPDATE_GOLDEN") == "1":
        GOLDEN.mkdir(exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="")
    assert path.read_text(encoding="utf-8") == text, f"{name} changed; if that is intended re-run with UPDATE_GOLDEN=1 and review the diff"


# ---------------------------------------------------------------- accepted by the fake sbatch (case-sensitive GRES, no refused flags)
@pytest.mark.parametrize("cluster,types", [("grace", ["a100", "a40", "rtx", "t4"]), ("vision", ["H200"])])
def test_fake_sbatch_accepts_every_script(cluster, types, tmp_path, fake_slurm):
    fake_slurm.configure(sbatch={"gres_types": types, "reject_flags": {"--get-user-env": "sbatch: error: option '--get-user-env' doesn't allow an argument"}})
    for method in METHODS:
        plan = make(cluster, spec_kw=dict(method=method))
        for st in plan.stages:
            path = tmp_path / f"{cluster}_{method}_{st.full_name}.sbatch"
            path.write_text(render_script(plan, st), encoding="utf-8", newline="")
            done = subprocess.run(["sbatch", "--test-only", str(path)], env=fake_slurm.env(), capture_output=True, text=True)
            assert done.returncode == 0, f"{path.name}: {done.stderr}"
    assert fake_slurm.submitted() == []                                          # test-only calls queue nothing


def test_the_fake_sbatch_would_refuse_a_lower_case_vision_type(tmp_path, fake_slurm):
    fake_slurm.configure(sbatch={"gres_types": ["H200"]})
    plan = make("vision")
    path = tmp_path / "train.sbatch"
    path.write_text(render_script(plan, stage(plan, "train")).replace("gpu:H200:", "gpu:h200:"))
    done = subprocess.run(["sbatch", "--test-only", str(path)], env=fake_slurm.env(), capture_output=True, text=True)
    assert done.returncode == 1 and "not available" in done.stderr
