"""The gate on the job scripts: every combination of method x cluster x teacher backend x student x GPU is rendered and every script
must satisfy the same invariants. `fx_render.render_matrix()` generates the combinations; the integrator also runs it to call
`sbatch --test-only` on the Vision ones. Nothing here runs Slurm, a GPU or torch.
"""

import dataclasses
import getpass
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from fx_render import (MATRIX_BACKENDS, MATRIX_STUDENTS, METHODS, clean_mpd_env, make, matrix_gpus, render_all,  # noqa: F401
                       render_matrix, stage)
from mpd.slurm.cluster import format_walltime, parse_walltime
from mpd.slurm.plan import MAX_ARRAY_TASKS

CLUSTERS = ("grace", "vision")
SLURM_DEFAULT_MAX_ARRAY_SIZE = 1001            # Slurm's default and the value of the Vision controller; Grace's is not verified
# every option a generated script may carry; the profile's own `sbatch_extra` flags are added per cluster
ALLOWED_OPTIONS = {"job-name", "comment", "time", "nodes", "ntasks", "cpus-per-task", "mem", "gres", "partition", "account", "array",
                   "output", "error", "open-mode"}
PLACEHOLDER_NUMBERS = {"123456789012", "1234567890123"}      # CONTRACTS 9.8: these and runs of all 0 or all 9
# written in pieces so that the identity scan of the repository does not find this file
FORBIDDEN_TEXT = ("yi" + "hong", "prj-0" + "2", "mol-" + "cell", "Protein" + "OPD", "/scratch/" + "project", "vs" + "ts", "gh" + "p_",
                  "github_" + "pat_")


def forbidden_in(text: str) -> list[str]:
    """CONTRACTS 1.5: strings that must never reach a file of the repository (the integrator greps for the same)."""
    found = [bad for bad in FORBIDDEN_TEXT if bad in text]
    found += [host for host in re.findall(r"[\w.-]*tamu\.edu", text) if not re.fullmatch(r"(?:[\w-]+\.)*hprc\.tamu\.edu", host)]
    found += [n for n in re.findall(r"(?<!\d)\d{12,13}(?!\d)", text) if n not in PLACEHOLDER_NUMBERS and not re.fullmatch(r"0+|9+", n)]
    user = getpass.getuser()
    if len(user) >= 5 and user in text:
        found.append("the user name of the person running the tests")
    return found


def directive_lines(text: str) -> list[str]:
    return [ln for ln in text.split("\n") if ln.startswith("#SBATCH")]


def options_of(text: str) -> dict[str, str]:
    out = {}
    for ln in directive_lines(text):
        name, _, value = ln.rstrip("\r")[len("#SBATCH "):].partition("=")
        out[name.removeprefix("--")] = value
    return out


def bash_syntax_error(text: str, where: Path) -> str:
    where.write_text(text, encoding="utf-8", newline="")
    done = subprocess.run(["bash", "-n", str(where)], capture_output=True, text=True)
    return done.stderr.strip() if done.returncode else ""


def check_script(plan, st, text: str, scratch: Path) -> list[str]:
    """Every invariant of one script; returns what is wrong, empty when nothing is."""
    cluster = plan.options.cluster
    gpu = cluster.gpus[st.gpu] if st.gpu else None
    problems: list[str] = []

    def bad(message: str) -> None:
        problems.append(message)

    # form
    if (error := bash_syntax_error(text, scratch)):
        bad(f"bash -n: {error}")
    if "\r" in text:
        bad("carriage return")
    if not text.endswith("\n") or text.endswith("\n\n"):
        bad("must end with exactly one newline")
    lines = text.split("\n")
    if lines[0] != "#!/bin/bash":
        bad(f"first line is {lines[0]!r}")
    # what is never allowed
    for flag in ("--export", "--get-user-env", "--dependency", "--kill-on-invalid-dep", "afterok", "--wrap", "--requeue"):
        if flag in text:
            bad(f"contains {flag}")
    if forbidden_in(text):
        bad(f"forbidden text: {forbidden_in(text)}")
    # the directives
    opts = options_of(text)
    unexpected = set(opts) - ALLOWED_OPTIONS - {flag.split("=", 1)[0].removeprefix("--") for flag in cluster.sbatch_extra}
    if unexpected:
        bad(f"unexpected #SBATCH options {sorted(unexpected)}")
    if directive_lines(text) != lines[1:1 + len(directive_lines(text))]:
        bad("directives are not one block directly under the first line")
    if opts.get("comment") != f"mpd:{plan.campaign_id}:{plan.nonce}:{st.full_name}":
        bad(f"comment tag is {opts.get('comment')!r}")
    limit = gpu.max_walltime_s if gpu else cluster.cpu_max_walltime_s
    if "time" not in opts or opts["time"] != format_walltime(st.walltime_s) or parse_walltime(opts["time"]) > limit:
        bad(f"walltime {opts.get('time')!r} (the stage plans {format_walltime(st.walltime_s)}, the partition allows {format_walltime(limit)})")
    if int(opts["cpus-per-task"]) > cluster.node_cores or int(opts["mem"].removesuffix("G")) > cluster.node_mem_gb:
        bad(f"cpus {opts['cpus-per-task']} / mem {opts['mem']} exceed a {cluster.node_cores}-core, {cluster.node_mem_gb} GB node")
    logs = f"{plan.campaign_dir}/logs/"
    for name, suffix in (("output", ".out"), ("error", ".err")):
        value = opts.get(name, "")
        if not (os.path.isabs(value) and value.startswith(logs) and value.endswith(suffix)):
            bad(f"--{name}={value!r} is not an absolute path under {logs}")
    if plan.options.account:
        if opts.get("account") != plan.options.account:
            bad(f"account is {opts.get('account')!r}")
    elif "account" in opts:
        bad("an account that nobody passed")
    # GPU or not
    guard = "nvidia-smi -L || {" in text and "exit 31" in text and "MPD_NO_GPU" in text
    if gpu is not None:
        if opts.get("gres") != f"gpu:{gpu.gres}:{st.gpus}":
            bad(f"gres is {opts.get('gres')!r}, the profile spells {gpu.gres!r} and the stage asks for {st.gpus}")
        if st.gpus > gpu.per_node:
            bad(f"{st.gpus} GPUs on a node of {gpu.per_node}")
        if opts.get("partition") != (gpu.partition or opts.get("partition")):
            bad(f"partition is {opts.get('partition')!r}, the profile says {gpu.partition!r}")
        if not guard:
            bad("the exit-31 GPU guard is missing")
        if "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MPD_REQUIRE_CUDA=1" not in text:
            bad("MPD_REQUIRE_CUDA is not exported")
    else:
        if "gres" in opts or "nvidia-smi" in text or "exit 31" in text or "MPD_REQUIRE_CUDA" in text or "MPD_NO_GPU" in text:
            bad("a CPU stage carries GPU rails")
        if opts.get("partition") != cluster.cpu_partition and (cluster.cpu_partition or "partition" in opts):
            bad(f"CPU partition is {opts.get('partition')!r}, the profile says {cluster.cpu_partition!r}")
    # the order of the prologue: modules, then `set -u`
    if lines.count("set -u") != 1:
        bad(f"set -u appears {lines.count('set -u')} times")
    else:
        at = lines.index("set -u")
        module_lines = [i for i, ln in enumerate(lines) if re.search(r"\bmodule (purge|load)\b|\btype module\b|/etc/profile", ln)]
        if any(i > at for i in module_lines):
            bad("a module command comes after set -u")
        if not ("set -eo pipefail" in lines[:at] and "unset PYTHONPATH PYTHONHOME" in lines[:at]):
            bad("set -eo pipefail and unset PYTHONPATH PYTHONHOME must come before set -u")
        first_export = next((i for i, ln in enumerate(lines) if ln.startswith("export ")), len(lines))
        if first_export < at:
            bad("an export comes before set -u")
    if not any(ln.startswith("export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1") for ln in lines):
        bad("jobs must stay offline (HF_HUB_OFFLINE=1)")
    # arrays and the command line
    index = f"$((SLURM_ARRAY_TASK_ID + {st.task_offset}))" if st.is_array else str(st.task_offset)
    command = next((ln for ln in lines if f"-m {st.module} " in ln), "")
    if not command.endswith(f"--stage {st.name} --task-index {index}") or f"--campaign-dir {plan.campaign_dir} " not in command:
        bad(f"command line is {command!r}")
    if st.launcher == "ddp" and f"--standalone --nnodes=1 --nproc_per_node={st.gpus} -m {st.module}" not in command:
        bad("a DDP stage must start torch.distributed.run with one process per GPU")
    if st.is_array:
        m = re.fullmatch(r"0-(\d+)%(\d+)", opts.get("array", ""))
        if not m or int(m.group(1)) + 1 != len(st.tasks):
            bad(f"array {opts.get('array')!r} for {len(st.tasks)} tasks")
        elif int(m.group(1)) + 1 > MAX_ARRAY_TASKS or int(m.group(1)) + 1 > SLURM_DEFAULT_MAX_ARRAY_SIZE or int(m.group(2)) != cluster.array_throttle:
            bad(f"array {opts['array']} breaks the limit of {MAX_ARRAY_TASKS} tasks or the throttle of {cluster.array_throttle}")
    elif "array" in opts:
        bad("a single task must not be an array")
    return problems


def check_plan(plan, scripts: dict[str, str]) -> list[str]:
    """Invariants across the stages of one plan: every stage has a script, chunks tile the task list, dependencies exist."""
    problems = []
    names = {s.full_name for s in plan.stages}
    if set(scripts) != names:
        problems.append(f"scripts {sorted(scripts)} != stages {sorted(names)}")
    for s in plan.stages:
        missing = [d for d in s.depends_on if d not in names]
        if missing:
            problems.append(f"{s.full_name} depends on {missing}, which do not exist")
    for base in {s.name for s in plan.stages}:
        parts = [s for s in plan.stages if s.name == base]
        offsets = [s.task_offset for s in parts]
        if offsets != [sum(len(p.tasks) for p in parts[:i]) for i in range(len(parts))] or any(len(p.tasks) > MAX_ARRAY_TASKS for p in parts):
            problems.append(f"chunks of {base} do not tile its task list: offsets {offsets}, sizes {[len(p.tasks) for p in parts]}")
    return problems


# ---------------------------------------------------------------- the matrix
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("cluster", CLUSTERS)
def test_every_script_of_every_combination_satisfies_the_invariants(cluster, method, tmp_path):
    problems, plans, scripts_seen = [], 0, 0
    for label, plan, scripts in render_matrix(clusters=(cluster,), methods=(method,)):          # the default home: a path that does not exist
        plans += 1
        problems += [f"{label}: {p}" for p in check_plan(plan, scripts)]
        for st in plan.stages:
            scripts_seen += 1
            problems += [f"{label}/{st.full_name}: {p}" for p in check_script(plan, st, scripts[st.full_name], tmp_path / "script.sbatch")]
    widths = 2 if cluster == "grace" and method.endswith("all") else 1           # plus --train-gpus 2 on Grace
    assert plans == len(matrix_gpus(cluster)) * len(MATRIX_BACKENDS) * len(MATRIX_STUDENTS) * widths
    assert scripts_seen >= 5 * plans
    assert not problems, "\n".join(problems[:25]) + f"\n... {len(problems)} problems in all"


def test_the_matrix_has_the_documented_shape_and_nothing_is_refused():
    refused: list = []
    plans = {label: plan for label, plan, _ in render_matrix(refused=refused)}
    assert refused == [] and len(plans) == len(set(plans)) == 96
    assert {p.options.cluster.name for p in plans.values()} == set(CLUSTERS)
    assert {p.spec.method for p in plans.values()} == set(METHODS)
    assert {p.spec.student for p in plans.values()} == set(MATRIX_STUDENTS)
    assert {p.spec.backend for p in plans.values()} == set(MATRIX_BACKENDS)
    assert {(p.options.cluster.name, p.resolved["gpu"]) for p in plans.values()} == {("grace", "a100"), ("grace", "a40"), ("vision", "h200")}
    assert not any(g in ("rtx", "t4") for _, g in {(p.options.cluster.name, p.resolved["gpu"]) for p in plans.values()})     # Turing: refused
    wide = [p for label, p in plans.items() if label.endswith("train-gpus-2")]
    assert len(wide) == 24 and all(p.options.cluster.name == "grace" and p.resolved["world_size"] == 2 and p.spec.method.endswith("all") for p in wide)
    assert all(not any(s.name == "teacher" for s in p.stages) == (p.spec.backend == "cache") for p in plans.values())
    cache_plans = [p for p in plans.values() if p.spec.backend == "cache"]
    assert cache_plans and all(next(s for s in p.stages if s.name == "train").depends_on == ["prepare"] for p in cache_plans)


def test_the_matrix_changes_nothing_around_it(tmp_path, monkeypatch):
    from mpd import registry
    for name, value in dict(MPD_ACCOUNT="999999999999", MPD_FOLDSEEK="/maintainer/foldseek", MPD_HOME=str(tmp_path / "other"),
                            SLURM_JOB_ID="31337").items():
        monkeypatch.setenv(name, value)
    before = dict(os.environ), dict(registry.MODELS), sorted(p.name for p in tmp_path.iterdir())
    home = tmp_path / "home"
    seen = [(label, scripts) for label, _, scripts in render_matrix(home, clusters=("vision",))]
    assert (dict(os.environ), dict(registry.MODELS), sorted(p.name for p in tmp_path.iterdir())) == before and not home.exists()
    assert seen and not any("999999999999" in t or "/maintainer" in t for _, scripts in seen for t in scripts.values())    # the shell did not leak in
    assert registry.get("esm2-3b").teacher_backends == ("live",)


def test_the_matrix_imports_and_runs_without_torch():
    code = ("import sys; sys.path.insert(0, 'tests'); from fx_render import render_matrix; "
            "n = sum(1 for _ in render_matrix(clusters=('vision',))); assert n == 24, n; assert 'torch' not in sys.modules")
    done = subprocess.run([sys.executable, "-s", "-c", code], capture_output=True, text=True, cwd=Path(__file__).parents[1],
                          env={"PYTHONPATH": str(Path(__file__).parents[1] / "src"), "PATH": os.environ["PATH"]})
    assert done.returncode == 0, done.stderr


def test_the_typed_gres_of_each_profile_is_what_the_scripts_ask_for():
    spelled = {}
    for _, plan, scripts in render_matrix(clusters=("grace", "vision"), methods=("score-all",)):
        for st in plan.stages:
            if st.gpu:
                spelled.setdefault((plan.options.cluster.name, st.gpu), set()).add(options_of(scripts[st.full_name])["gres"].rsplit(":", 1)[0])
    assert spelled == {("grace", "a100"): {"gpu:a100"}, ("grace", "a40"): {"gpu:a40"}, ("vision", "h200"): {"gpu:H200"}}      # Vision's type is case sensitive


# ---------------------------------------------------------------- more than the default
@pytest.mark.parametrize("cluster", CLUSTERS)
@pytest.mark.parametrize("n_assays", [1000, 1001, 2001])
@pytest.mark.parametrize("method", ["direct-one", "score-one"])
def test_large_campaigns_are_split_into_arrays_that_fit(cluster, method, n_assays, tmp_path):
    assays = tuple(f"assay{i}" for i in range(n_assays))
    plan = make(cluster, spec_kw=dict(method=method, assays=assays, eval_data=("venusmuthub",)))
    scripts = render_all(plan)
    problems = check_plan(plan, scripts)
    for st in plan.stages:
        problems += check_script(plan, st, scripts[st.full_name], tmp_path / "s.sbatch")
    assert not problems, problems[:10]
    arrays = [options_of(scripts[s.full_name]).get("array") for s in plan.stages if s.name == "train"]
    assert max(len(s.tasks) for s in plan.stages) <= MAX_ARRAY_TASKS and len(arrays) == -(-n_assays // 1000)


@pytest.mark.parametrize("gpu", ["rtx", "t4"])
def test_turing_gpus_render_only_with_the_override_and_then_in_float32(gpu, tmp_path):
    plan = make("grace", gpu=gpu, allow_unsupported_gpu=True, spec_kw=dict(teacher="esm2-8m", student="esm2-8m", method="score-all"))
    scripts = render_all(plan)
    for st in plan.stages:
        assert check_script(plan, st, scripts[st.full_name], tmp_path / "s.sbatch") == []
        assert "export MPD_DTYPE=float32\n" in scripts[st.full_name]
        if st.gpu:
            assert options_of(scripts[st.full_name])["gres"] == f"gpu:{gpu}:{st.gpus}"


# ---------------------------------------------------------------- the checker itself: each invariant fails on a script that breaks it
GOOD = make("grace", spec_kw=dict(method="score-all", train_gpus=2))
GOOD_TRAIN = stage(GOOD, "train")
GOOD_PREPARE = stage(GOOD, "prepare")
NEGATIVE_CONTROLS = [
    ("export", "train", lambda t: t.replace("#SBATCH --open-mode=append", "#SBATCH --open-mode=append\n#SBATCH --export=NONE"), "--export"),
    ("get-user-env", "train", lambda t: t.replace("#SBATCH --open-mode=append", "#SBATCH --open-mode=append\n#SBATCH --get-user-env=L"), "--get-user-env"),
    ("dependency", "train", lambda t: t.replace("#SBATCH --open-mode=append", "#SBATCH --open-mode=append\n#SBATCH --dependency=afterok:1"), "--dependency"),
    ("unknown option", "train", lambda t: t.replace("#SBATCH --open-mode=append", "#SBATCH --open-mode=append\n#SBATCH --mail-user=a@b.c"), "unexpected #SBATCH options"),
    ("lower case gres", "train", lambda t: t.replace("gpu:a100:2", "gpu:A100:2"), "gres is"),
    ("wrong gpu count", "train", lambda t: t.replace("gpu:a100:2", "gpu:a100:1"), "gres is"),
    ("gres on a CPU stage", "prepare", lambda t: t.replace("#SBATCH --open-mode=append", "#SBATCH --open-mode=append\n#SBATCH --gres=gpu:1"), "GPU rails"),
    ("set -u before the modules", "train", lambda t: t.replace("\nset -u\n", "\n", 1).replace("set -eo pipefail\n", "set -eo pipefail\nset -u\n", 1),
     "module command comes after set -u"),
    ("no set -u", "train", lambda t: t.replace("\nset -u\n", "\n", 1), "set -u appears 0 times"),
    ("an export before set -u", "train", lambda t: t.replace("\nset -u\n", "\n", 1).replace("unset PYTHONPATH PYTHONHOME\n", "unset PYTHONPATH PYTHONHOME\nexport A=1\nset -u\n", 1),
     "an export comes before set -u"),
    ("no guard", "train", lambda t: t.replace("exit 31", "exit 1"), "exit-31 GPU guard"),
    ("guard on a CPU stage", "prepare", lambda t: t.replace("export HF_HUB_OFFLINE", "nvidia-smi -L || { exit 31; }\nexport HF_HUB_OFFLINE"), "GPU rails"),
    ("a day too long", "train", lambda t: re.sub(r"--time=\S+", "--time=9-00:00:00", t), "walltime"),
    ("relative log path", "train", lambda t: t.replace("--output=/", "--output=./"), "absolute path"),
    ("log outside the campaign", "train", lambda t: t.replace(f"{GOOD.campaign_dir}/logs/", "/tmp/logs/"), "absolute path"),
    ("online", "train", lambda t: t.replace("export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1", "export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0"), "offline"),
    ("a real-looking account number", "prepare", lambda t: t.replace("123456789012", "98765" + "4321098"), "forbidden text"),
    ("a maintainer path", "prepare", lambda t: t + "\n# /scratch/" + "project/x\n", "forbidden text"),
    ("crlf", "prepare", lambda t: t.replace("\n", "\r\n"), "carriage return"),
    ("no final newline", "prepare", lambda t: t.rstrip("\n"), "newline"),
    ("syntax error", "prepare", lambda t: t + "if then\n", "bash -n"),
    ("task index missing", "train", lambda t: t.replace(" --task-index 0", ""), "command line"),
    ("a wrong comment tag", "train", lambda t: t.replace(":train\n", ":trainx\n", 1), "comment tag"),
]


@pytest.mark.parametrize("name,which,break_it,expect", NEGATIVE_CONTROLS, ids=[c[0] for c in NEGATIVE_CONTROLS])
def test_the_checker_catches_a_script_that_breaks_an_invariant(name, which, break_it, expect, tmp_path):
    st = GOOD_TRAIN if which == "train" else GOOD_PREPARE
    good = render_all(GOOD)[st.full_name]
    assert check_script(GOOD, st, good, tmp_path / "s.sbatch") == []
    broken = break_it(good)
    assert broken != good, name
    found = check_script(GOOD, st, broken, tmp_path / "s.sbatch")
    assert any(expect in problem for problem in found), (name, found)


def test_the_checker_catches_an_array_beyond_the_limit_and_chunks_that_do_not_tile(tmp_path):
    big = make("grace", spec_kw=dict(method="score-one", assays=tuple(f"a{i}" for i in range(1500)), eval_data=("venusmuthub",)))
    scripts = render_all(big)
    assert check_plan(big, scripts) == []
    train = next(s for s in big.stages if s.name == "train")
    fat = scripts[train.full_name].replace(f"--array=0-{len(train.tasks) - 1}", "--array=0-1500")
    assert any("array" in p for p in check_script(big, train, fat, tmp_path / "s.sbatch"))
    overlapping = dataclasses.replace(big, stages=[dataclasses.replace(s, task_offset=10) if s.full_name == "train.1" else s for s in big.stages])
    assert any("do not tile" in p for p in check_plan(overlapping, scripts))
    assert any("do not exist" in p for p in check_plan(dataclasses.replace(
        big, stages=[dataclasses.replace(s, depends_on=["nowhere"]) if s.name == "evaluate" else s for s in big.stages]), scripts))
