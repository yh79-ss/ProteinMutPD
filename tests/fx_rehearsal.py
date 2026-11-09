"""Run the sbatch scripts that `mpd submit` writes, end to end, on the CPU: the toolbox of tests/test_rehearsal.py and of
scripts/maintainers/rehearse_campaign.py.

    from fx_rehearsal import rehearsal_venv, ...      # rehearsal_venv is a session fixture

Why it exists. The stage functions are tested in-process, the renderer as text, and the submit logic against a fake Slurm that
never runs a script. The first failures on a real cluster live in the seams between those: the prologue of a script (modules,
environment record, PYTHONPATH), the variables it exports, the task-index plumbing, paths, the GPU guard, the offline flags,
resuming. So this module executes the real scripts with `bash`, in dependency order, started the way Slurm starts them, with

  * fx_slurm's fake `module` (entered through a read of an unset variable, as Grace's /etc/bashrc does) and fake `nvidia-smi`;
  * made-up SLURM_* variables, an empty $HOME and a PATH without any Slurm client (real or fake);
  * a venv at <home>/env whose interpreter reaches the test environment and the repository through a .pth file (no pip, no
    network), with a tripwire (imported by a .pth line) that records and refuses every connection leaving the machine;
  * tiny random-weight models and the tiny datasets of tests/tinydata.py.

The campaign itself is real: make_plan, write_campaign and submit run as `mpd submit` runs them, against fx_slurm's fake sbatch,
so the script that is executed is the script that was "queued" and the log names carry the job ids that SUBMITTED.json records.

What a passing rehearsal shows: the scripts of this code run on a CPU, from their first line to MPD_STAGE_DONE, in the right order,
and leave the files that `mpd status`, `mpd results` and `mpd submit --force` look for. What it does not show: anything about Grace's
module names, GPU drivers, NCCL, memory or queue, the Slurm version, or whether a real GPU run is fast enough. Nothing here ran on Grace.
"""

from __future__ import annotations

import concurrent.futures as futures
import contextlib
import csv
import importlib.util
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import sysconfig
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pytest

from fx_slurm import FakeSlurm, clean_path
from tinydata import write_tiny_proteingym

from mpd.slurm.campaign import task_marker
from mpd.slurm.cluster import Cluster
from mpd.slurm.plan import Plan, PlanOptions, make_plan
from mpd.slurm.submit import submit, write_campaign
from mpd.spec import RunSpec

ACCOUNT = "123456789012"                                      # the documented fake account
MODULES = {"grace": ("Anaconda3/2024.02-1",), "vision": ("Python/3.11.5",)}      # names the fake node's `module` knows (fx_slurm)
WORKERS = int(os.environ.get("MPD_REHEARSAL_JOBS") or min(4, os.cpu_count() or 2))   # jobs running at once, over every campaign
JOB_TIMEOUT_S = float(os.environ.get("MPD_REHEARSAL_TIMEOUT") or 900)
BASH = shutil.which("bash") or "/bin/bash"
SRC = Path(__file__).resolve().parents[1] / "src"
CPU_ENV = (("OMP_NUM_THREADS", "1"), ("MKL_NUM_THREADS", "1"), ("MPD_REQUIRE_CUDA", "0"))     # --env pairs of a CPU run


# ---------------------------------------------------------------------------------------------- the venv and its tripwire
TRIPWIRE_SOURCE = '''\
"""Rehearsal tripwire (tests/fx_rehearsal.py): record and refuse every connection that would leave this machine."""
import errno
import ipaddress
import os
import socket

_LOCAL = {"", "localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback"}


def _install():
    real = {n: getattr(socket, n) for n in ("getaddrinfo", "gethostbyname", "gethostbyname_ex")}
    connect, connect_ex = socket.socket.connect, socket.socket.connect_ex
    own = []

    def addresses():
        if not own:
            for name in {socket.gethostname(), os.uname().nodename}:
                try:
                    own.extend(info[4][0] for info in real["getaddrinfo"](name, None))
                except OSError:
                    pass
            own.append("")
        return own

    def local(host):
        if isinstance(host, bytes):
            host = host.decode("ascii", "replace")
        if host is None or str(host).lower() in _LOCAL:
            return True
        host = str(host).lower()
        mine = {socket.gethostname().lower(), os.uname().nodename.lower()}
        if host in mine or host.split(".")[0] in {m.split(".")[0] for m in mine}:
            return True
        try:
            ip = ipaddress.ip_address(host.split("%")[0])
        except ValueError:
            return False                                  # a name that is not this machine's: resolving it would need DNS
        return ip.is_loopback or ip.is_unspecified or host in addresses()

    def refuse(kind, target):
        home = os.environ.get("MPD_HOME")
        if home:
            try:
                with open(os.path.join(home, "net_attempts.log"), "a") as fh:
                    fh.write(f"{kind} {target}\\n")
            except OSError:
                pass
        return OSError(errno.ENETUNREACH, f"rehearsal tripwire: {kind} {target} would leave this machine")

    def guarded(name):
        def call(host, *args, **kwargs):
            if not local(host):
                raise refuse(name, host)
            return real[name](host, *args, **kwargs)
        return call

    def connect_guard(self, address):
        if isinstance(address, tuple) and not local(address[0]):
            raise refuse("connect", address[0])
        return connect(self, address)

    def connect_ex_guard(self, address):
        if isinstance(address, tuple) and not local(address[0]):
            refuse("connect_ex", address[0])
            return errno.ENETUNREACH
        return connect_ex(self, address)

    for name in real:
        setattr(socket, name, guarded(name))
    socket.socket.connect, socket.socket.connect_ex = connect_guard, connect_ex_guard


try:
    _install()
except Exception:                                         # never stop an interpreter from starting
    pass
'''


def build_venv(dest: Path) -> Path:
    """A venv at `dest` that imports mpd from the repository and everything else from the environment running the tests.

    `python -m venv --without-pip` plus a .pth file: no install, no network. The .pth lists src/ first (the repository wins over any
    installed copy of mpd) and then the folders torch, transformers, peft, numpy, scipy, safetensors and yaml are found in."""
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(dest)], check=True, capture_output=True)
    site = next((dest / "lib").glob("python*/site-packages"))
    roots = [SRC]
    for name in ("torch", "transformers", "peft", "numpy", "scipy", "yaml", "safetensors"):
        spec = importlib.util.find_spec(name)             # where it is, without importing it (peft alone takes seconds)
        assert spec and spec.origin, f"{name} is not installed in the environment that runs the tests"
        roots.append(Path(spec.origin).resolve().parents[1])
    roots += [Path(p) for p in (sysconfig.get_path("purelib"), sysconfig.get_path("platlib"))]
    (site / "mpd_rehearsal.pth").write_text("import mpd_tripwire\n" + "".join(f"{r}\n" for r in dict.fromkeys(roots)))
    (site / "mpd_tripwire.py").write_text(TRIPWIRE_SOURCE)          # a .pth import line runs on every Python; a sitecustomize is shadowed by Debian's
    return dest


@pytest.fixture(scope="session")
def rehearsal_venv(tmp_path_factory):
    """The template venv, built once per session and removed at the end; every home gets a copy at <home>/env."""
    root = tmp_path_factory.mktemp("rehearsal_venv")
    yield build_venv(root / "env")
    shutil.rmtree(root, ignore_errors=True)


# ---------------------------------------------------------------------------------------------- the fake node
# fx_slurm's `module` function, called after a read of a variable nothing defines. Grace's /etc/bashrc (sourced from Lmod's init)
# reads $BASHRCSOURCED like that, so a `set -u` that is already on when a script loads its modules dies with "unbound variable".
BASH_ENV = '''\
. "@FAKE@"
eval "$(declare -f module | sed '1s/^module/fx_lmod/')"
module() { : "$BASHRCSOURCED"; fx_lmod "$@"; }
export -f module
'''


class Node:
    """What a compute node adds to a job: `module`, `nvidia-smi`, SLURM_* variables, an empty $HOME and no Slurm client on PATH.

    `fs` is the fx_slurm toolbox behind it; the same object also serves as the fake sbatch/squeue/sacct of the campaign."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.fs = FakeSlurm(self.root / "slurm")
        self.bin = self.root / "nodebin"
        self.bin.mkdir()
        (self.bin / "nvidia-smi").symlink_to(self.fs.bin / "nvidia-smi")
        self.userhome = self.root / "userhome"
        self.userhome.mkdir()
        self.submit_dir = self.root / "submit"
        self.submit_dir.mkdir()
        self.bash_env = self.root / "bash_env.sh"
        self.bash_env.write_text(BASH_ENV.replace("@FAKE@", str(self.fs.bash_env)))
        self.path = f"{self.bin}{os.pathsep}{clean_path('/usr/bin:/bin')}"

    def shell(self, **extra: str) -> dict[str, str]:
        """The environment of the shell that ran `mpd submit`, which Slurm's default (--export=ALL) hands to every job."""
        env = {"PATH": self.path, "HOME": str(self.userhome), "USER": self.fs.user, "LOGNAME": self.fs.user, "LANG": "C.UTF-8",
               "BASH_ENV": str(self.bash_env), "LOADEDMODULES": "", "CUDA_VISIBLE_DEVICES": ""}
        env.update(extra)
        return env


@contextlib.contextmanager
def environment(updates: dict | None = None):
    """Edit os.environ inside a with-block: the MPD_* and SLURM_* variables of the developer's shell are hidden, `updates` is set."""
    saved = dict(os.environ)
    for name in list(os.environ):
        if name.startswith(("MPD_", "SLURM_")):
            del os.environ[name]
    os.environ.update(updates or {})
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


# ---------------------------------------------------------------------------------------------- home, recipe, campaign
def make_home(root: Path, cluster: str, venv: Path, *, structures: bool = False, modules: tuple[str, ...] | None = None,
              name: str = "mpd") -> Path:
    """<root>/<name> with the tiny ProteinGym, a copy of the venv at env/ and the environment record `setup_env.sh` would write."""
    from mpd.data.sources import PROTEINGYM
    home = Path(root) / name
    if structures:
        from fx_saprot import write_tiny_structure_home
        write_tiny_structure_home(home)
    else:
        (home / "data" / PROTEINGYM.dirname).mkdir(parents=True)
        write_tiny_proteingym(home / "data" / PROTEINGYM.dirname)
    shutil.copytree(venv, home / "env", symlinks=True)
    command = [str(home / "env" / "bin" / "python"), "-m", "mpd.envrecord", "write", "--home", str(home), "--cluster", cluster,
               "--modules", " ".join(modules or MODULES.get(cluster, ()))]
    done = subprocess.run(command, capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"}, timeout=120)
    assert done.returncode == 0, done.stderr
    return home


@dataclass(frozen=True)
class Recipe:
    """A tiny campaign: 24 updates at a global batch of 4 on the three assays of tests/tinydata.py."""
    method: str = "direct-all"
    teacher: str = "esm2-35m"           # a different tiny model from the student's: with equal weights the loss starts at zero
    student: str = "esm2-8m"
    train_data: str = "proteingym"
    eval_data: tuple[str, ...] = ("proteingym",)
    assays: tuple[str, ...] = ()
    updates: int = 24
    global_batch: int = 4
    score_shards: int = 2
    train_gpus: int | None = None
    dev_fraction: float = 0.3
    env: tuple[tuple[str, str], ...] = CPU_ENV

    @property
    def label(self) -> str:
        return f"{self.teacher}->{self.student} {self.method}"

    def spec(self) -> RunSpec:
        assays = self.assays or (("ASSAY_A", "ASSAY_B") if self.method.endswith("one") else ())
        return RunSpec(teacher=self.teacher, student=self.student, train_data=self.train_data, method=self.method,
                       eval_data=self.eval_data, assays=assays, updates=self.updates, global_batch=self.global_batch,
                       dev_fraction=self.dev_fraction, train_gpus=self.train_gpus)

    def without(self, *names: str) -> Recipe:
        return replace(self, env=tuple(pair for pair in self.env if pair[0] not in names))


def model_variables(tiny_models: dict, tiny_saprot: dict | None = None) -> dict[str, str]:
    """MPD_MODEL_<KEY>: the user-override route to local weights, which skips the pin check exactly as a user's override does."""
    out = {"MPD_MODEL_ESM2_8M": str(tiny_models["student"]), "MPD_MODEL_ESM2_35M": str(tiny_models["teacher"])}
    if tiny_saprot:
        out.update(MPD_MODEL_SAPROT_35M=str(tiny_saprot["student"]), MPD_MODEL_SAPROT_650M=str(tiny_saprot["teacher"]))
    return out


def plan_recipe(home: Path, cluster: str, recipe: Recipe, shell: dict, *, python: str | None = None, nonce: str | None = None):
    """make_plan as `mpd plan` runs it, in a process whose MPD_* variables are those of `shell`."""
    options = PlanOptions(cluster=Cluster.load(cluster), home=home, python=python, nonce=nonce, extra_env=dict(recipe.env),
                          account=ACCOUNT if cluster == "grace" else None,
                          score_shards={ds: recipe.score_shards for ds in recipe.eval_data})
    with environment({k: v for k, v in shell.items() if k.startswith("MPD_")}):
        return make_plan(recipe.spec(), options)


def submit_to_fake(plan, node: Node, shell: dict) -> tuple[dict[str, Path], dict[str, str]]:
    """write_campaign + submit against the fake sbatch: the scripts on disk, and stage -> job id as SUBMITTED.json records it.

    The scripts read the forwarded MPD_* variables when they are rendered, which is here, so `shell` is the environment of this step."""
    fs = node.fs
    with environment({**{k: v for k, v in shell.items() if k.startswith("MPD_")}, "USER": fs.user, "LOGNAME": fs.user}):
        scripts = write_campaign(plan)
        ids = submit(plan, scripts, sbatch=str(fs.bin / "sbatch"), squeue=str(fs.bin / "squeue"), scancel=str(fs.bin / "scancel"),
                     echo=lambda line: None, sleep=fs.sleep, clock=fs.clock)
    return scripts, ids


# ---------------------------------------------------------------------------------------------- jobs
def sbatch_option(script: Path, name: str) -> str | None:
    """The value of `#SBATCH --name=value` in a script (directives end at the first command)."""
    for line in script.read_text(encoding="utf-8").splitlines()[1:]:
        if line.startswith(f"#SBATCH --{name}="):
            return line.split("=", 1)[1]
        if line.strip() and not line.startswith("#"):
            break
    return None


@dataclass
class Job:
    """One run of one script: a stage, or one task of an array."""
    stage: str
    script: Path
    job_id: str                     # SLURM_JOB_ID
    env: dict
    array_id: str | None = None     # SLURM_ARRAY_JOB_ID
    task: int | None = None         # SLURM_ARRAY_TASK_ID
    cwd: Path | None = None

    def logs(self) -> tuple[Path, Path]:
        """Where Slurm would write stdout and stderr: the --output/--error patterns with %x %j %A %a filled in."""
        name = sbatch_option(self.script, "job-name") or "job"
        fill = lambda pattern: Path(pattern.replace("%x", name).replace("%A", self.array_id or "").replace(
            "%a", "" if self.task is None else str(self.task)).replace("%j", self.job_id))
        return fill(sbatch_option(self.script, "output")), fill(sbatch_option(self.script, "error"))

    def start(self) -> subprocess.Popen:
        out, err = self.logs()
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "ab") as so, open(err, "ab") as se:                  # --open-mode=append
            proc = subprocess.Popen([BASH, str(self.script)], env=self.env, cwd=self.cwd, stdin=subprocess.DEVNULL, stdout=so, stderr=se,
                                    start_new_session=True)               # its own process group, so a kill reaches python too
        _LIVE.add(proc)
        return proc

    def text(self, which: str = "out") -> str:
        path = self.logs()[0 if which == "out" else 1]
        return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""


@dataclass
class Outcome:
    job: Job
    rc: int
    seconds: float
    started: float = 0.0            # wall clock, time.time()
    ended: float = 0.0

    @property
    def out(self) -> str:
        return self.job.text("out")

    @property
    def err(self) -> str:
        return self.job.text("err")

    def describe(self, lines: int = 25) -> str:
        tail = lambda text: "\n".join(text.rstrip().splitlines()[-lines:])
        return (f"[{self.job.stage} job {self.job.job_id}] rc={self.rc} after {self.seconds:.1f} s\n--- stdout (tail)\n{tail(self.out)}"
                f"\n--- stderr (tail)\n{tail(self.err)}")


_LIVE: set[subprocess.Popen] = set()


def kill_group(proc: subprocess.Popen, sig: int = signal.SIGKILL) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, sig)


def kill_all() -> None:
    for proc in list(_LIVE):
        kill_group(proc)


def run_job(job: Job, *, semaphore: threading.Semaphore | None = None, timeout: float | None = None) -> Outcome:
    with semaphore or contextlib.nullcontext():
        started, wall = time.monotonic(), time.time()
        proc = job.start()
        try:
            rc = proc.wait(timeout=timeout or JOB_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            kill_group(proc)
            proc.wait()
            rc = -int(signal.SIGKILL)
        finally:
            _LIVE.discard(proc)
        return Outcome(job, rc, time.monotonic() - started, wall, time.time())


def slurm_variables(node: Node, stage, job_id: str, *, array_id: str | None = None, task: int | None = None) -> dict[str, str]:
    env = {"SLURM_JOB_ID": job_id, "SLURM_JOB_NAME": f"mpd-{stage.full_name}", "SLURM_JOB_NODELIST": "fakenode001",
           "SLURM_SUBMIT_DIR": str(node.submit_dir), "SLURM_CPUS_PER_TASK": str(stage.cpus), "SLURM_NNODES": "1", "SLURM_NTASKS": "1"}
    if array_id is not None:
        env.update(SLURM_ARRAY_JOB_ID=array_id, SLURM_ARRAY_TASK_ID=str(task))
    return env


def jobs_for(plan, scripts: dict[str, Path], ids: dict[str, str], node: Node, shell: dict) -> dict[str, list[Job]]:
    """stage -> its jobs: one, or one per task of an array (each task has a job id of its own, as in Slurm)."""
    out: dict[str, list[Job]] = {}
    for stage in plan.stages:
        name, sid = stage.full_name, ids[stage.full_name]
        if not stage.is_array:
            out[name] = [Job(name, scripts[name], sid, {**shell, **slurm_variables(node, stage, sid)}, cwd=node.submit_dir)]
            continue
        out[name] = []
        for task in range(len(stage.tasks)):
            job_id = str(int(sid) * 1000 + task + 1)
            env = {**shell, **slurm_variables(node, stage, job_id, array_id=sid, task=task)}
            out[name].append(Job(name, scripts[name], job_id, env, array_id=sid, task=task, cwd=node.submit_dir))
    return out


def missing_markers(plan, stage) -> list[Path]:
    """The files that tell `mpd status` and `--force` a task finished (campaign.task_marker) and that are not there."""
    out = []
    for task in stage.tasks:
        marker = task_marker(plan.campaign_dir, stage.name, task)
        if marker is None or not marker.is_file():
            out.append(marker or plan.campaign_dir / f"<no marker known for {stage.name}>")
    return out


@dataclass
class Run:
    outcomes: dict[str, list[Outcome]] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)          # stages that never started: a stage they wait for failed
    missing: dict[str, list[Path]] = field(default_factory=dict)      # marker files absent when the last job of a stage ended
    seconds: float = 0.0

    @property
    def failures(self) -> list[Outcome]:
        return [o for outs in self.outcomes.values() for o in outs if o.rc != 0]

    @property
    def ok(self) -> bool:
        return not self.failures and not self.skipped and not any(self.missing.values())

    def report(self) -> str:
        return "\n".join([*(o.describe() for o in self.failures[:2]), *(f"skipped: {s}" for s in self.skipped),
                          *(f"{s}: missing {[p.name for p in paths]}" for s, paths in self.missing.items() if paths)])


def run_stages(plan, jobs: dict[str, list[Job]], *, only=None, ignore_dependencies: bool = False,
               semaphore: threading.Semaphore | None = None, timeout: float | None = None, on_done=None, on_job=None) -> Run:
    """Run the stages of `plan` like Slurm with afterok dependencies: a stage starts when the stages it waits for have finished,
    every task of an array at once, and nothing starts behind a failure. At most the semaphore's permits jobs run at a time.
    `on_done(stage)` is called, from the thread that schedules, each time a stage has finished with its files in place, and
    `on_job(outcome)` each time a job has ended."""
    names = [s.full_name for s in plan.stages if only is None or s.full_name in only]
    stages = {s.full_name: s for s in plan.stages}
    wait_for = {n: [] if ignore_dependencies else [d for d in stages[n].depends_on if d in names] for n in names}
    run, done, failed = Run(), set(), set()
    left: dict[str, int] = {}
    started = time.monotonic()
    pool = futures.ThreadPoolExecutor(max_workers=max(1, sum(len(jobs[n]) for n in names)))
    running: dict[futures.Future, str] = {}
    try:
        def launch() -> None:
            for name in list(names):
                if any(d in failed for d in wait_for[name]):
                    names.remove(name)
                    failed.add(name)
                    run.skipped.append(name)
                elif all(d in done for d in wait_for[name]):
                    names.remove(name)
                    left[name] = len(jobs[name])
                    for job in jobs[name]:
                        running[pool.submit(run_job, job, semaphore=semaphore, timeout=timeout)] = name

        launch()
        while running:
            finished, _ = futures.wait(running, return_when=futures.FIRST_COMPLETED)
            for future in finished:
                name = running.pop(future)
                outcome = future.result()
                run.outcomes.setdefault(name, []).append(outcome)
                if on_job is not None:
                    on_job(outcome)
                left[name] -= 1
                if outcome.rc != 0:
                    failed.add(name)
                if left[name] == 0:
                    run.missing[name] = [] if name in failed else missing_markers(plan, stages[name])
                    if run.missing[name]:
                        failed.add(name)
                    elif name not in failed:
                        done.add(name)
                        if on_done is not None:
                            on_done(name)
            launch()
    except BaseException:
        kill_all()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    for outs in run.outcomes.values():
        outs.sort(key=lambda o: o.job.task or 0)
    run.seconds = time.monotonic() - started
    return run


@dataclass
class Rehearsal:
    """One campaign that was planned, submitted to the fake sbatch and run."""
    recipe: Recipe
    cluster: str
    home: Path
    node: Node
    shell: dict
    plan: Plan
    scripts: dict[str, Path]
    ids: dict[str, str]
    jobs: dict[str, list[Job]]
    run: Run

    @property
    def root(self) -> Path:
        return self.plan.campaign_dir

    def job(self, stage: str, task: int = 0) -> Job:
        return self.jobs[stage][task]


def prepare_campaign(node: Node, home: Path, cluster: str, recipe: Recipe, shell: dict, **plan_options) -> Rehearsal:
    """Plan, write and submit (to the fake sbatch); nothing runs yet."""
    plan = plan_recipe(home, cluster, recipe, shell, **plan_options)
    scripts, ids = submit_to_fake(plan, node, shell)
    return Rehearsal(recipe, cluster, home, node, shell, plan, scripts, ids, jobs_for(plan, scripts, ids, node, shell), Run())


def run_prepared(prepared: list[Rehearsal], *, semaphore: threading.Semaphore | None = None, timeout: float | None = None,
                 on_done=None, on_job=None) -> None:
    """Run campaigns that prepare_campaign made, all at once (at most WORKERS jobs at a time over all of them), and store each Run in
    its Rehearsal. `on_done(rehearsal, stage)` is called whenever a stage has finished, `on_job(rehearsal, outcome)` whenever a job
    has."""
    semaphore = semaphore or threading.Semaphore(WORKERS)
    with futures.ThreadPoolExecutor(max_workers=max(1, len(prepared))) as outer:
        runs = [outer.submit(run_stages, rh.plan, rh.jobs, semaphore=semaphore, timeout=timeout,
                             on_done=None if on_done is None else (lambda stage, rh=rh: on_done(rh, stage)),
                             on_job=None if on_job is None else (lambda outcome, rh=rh: on_job(rh, outcome))) for rh in prepared]
        for rh, future in zip(prepared, runs):
            rh.run = future.result()


def rehearse(node: Node, home: Path, cluster: str, recipes, shell: dict, *, semaphore: threading.Semaphore | None = None,
             on_done=None, on_job=None, **plan_options) -> list[Rehearsal]:
    """Prepare every recipe (plan, write, submit to the fake sbatch), then run the campaigns: see run_prepared."""
    prepared = [prepare_campaign(node, home, cluster, recipe, shell, **plan_options) for recipe in recipes]
    run_prepared(prepared, semaphore=semaphore, on_done=on_done, on_job=on_job)
    return prepared


# ---------------------------------------------------------------------------------------------- what to look at afterwards
def job_logs(rh: Rehearsal) -> list[tuple[str, str, int | None, Path]]:
    """(stage, job id, task, file) for every log file under <campaign>/logs."""
    pattern = re.compile(r"^mpd-(?P<stage>.+)-[^-_]+_(?P<job>\d+)(?:_(?P<task>\d+))?\.(?P<ext>out|err)$")
    found = []
    for path in sorted((rh.root / "logs").glob("*")):
        m = pattern.match(path.name)
        if m:
            found.append((m["stage"], m["job"], None if m["task"] is None else int(m["task"]), path))
    return found


def problems(rh: Rehearsal) -> list[str]:
    """What is wrong with a finished rehearsal, in words ([] when nothing is): failed or skipped jobs, markers that are missing, a job
    that did not reach MPD_STAGE_DONE, no summary, a connection the tripwire refused, something written under the fake $HOME."""
    out = [f"{o.job.stage} (job {o.job.job_id}) exited with {o.rc}" for o in rh.run.failures]
    out += [f"{name} never started: a stage it waits for failed" for name in rh.run.skipped]
    out += [f"{name}: {path.name} is missing" for name, paths in rh.run.missing.items() for path in paths]
    out += [f"{o.job.stage} (job {o.job.job_id}) did not print MPD_STAGE_DONE" for outs in rh.run.outcomes.values() for o in outs
            if o.rc == 0 and f"MPD_STAGE_DONE {o.job.stage}" not in o.out]
    summary = rh.root / "evaluate" / "summary.json"
    if not summary.is_file():
        out.append("evaluate/summary.json does not exist")
    elif not {"results", "primary_readout", "units"} <= set(read_json(summary)):
        out.append("evaluate/summary.json lacks results, primary_readout or units")
    out += [f"the tripwire refused a connection: {line}" for line in net_attempts(rh.home)]
    written = sorted(p.name for p in rh.node.userhome.iterdir())
    out += [f"something was written under the fake $HOME: {', '.join(written)}"] if written else []
    return out


def net_attempts(home: Path) -> list[str]:
    """The connections the tripwire refused in jobs run under this home."""
    log = Path(home) / "net_attempts.log"
    return log.read_text().splitlines() if log.is_file() else []


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def shards_of(rh: Rehearsal) -> dict[tuple[str, str], dict[int, dict]]:
    """(dataset, unit) -> {shard: its JSON record} of every finished score shard, with the contexts (`rows`) of its .npz."""
    out: dict[tuple[str, str], dict[int, dict]] = {}
    for path in sorted((rh.root / "score").glob("*/*/shard_*_of_*.json")):
        record = read_json(path)
        with np.load(path.with_suffix(".npz")) as z:
            record["_rows"] = z["rows"].tolist()
            record["_npz_shard"], record["_npz_num"] = int(z["shard"]), int(z["num_shards"])
        out.setdefault((path.parent.parent.name, path.parent.name), {})[record["shard"]] = record
    return out


def training_difference(a: Path, b: Path) -> dict[str, float]:
    """Largest absolute differences between two train/<unit> folders: logged losses, the development curve and the selected adapter.
    A difference of kind (other steps, another selected update, other tensor names) is infinite."""
    from safetensors.numpy import load_file

    def lines(path: Path) -> list[str]:
        return path.read_text(encoding="utf-8").splitlines()

    log_a, log_b = list(csv.DictReader(lines(a / "train_steps.csv"))), list(csv.DictReader(lines(b / "train_steps.csv")))
    if [r["step"] for r in log_a] != [r["step"] for r in log_b]:
        return {"steps": float("inf")}
    loss = max((abs(float(x[c]) - float(y[c])) for x, y in zip(log_a, log_b) for c in ("loss", "huber", "lr", "grad_norm")), default=0.0)
    curve_a, curve_b = [json.loads(x) for x in lines(a / "dev_curve.jsonl")], [json.loads(x) for x in lines(b / "dev_curve.jsonl")]
    if [r["step"] for r in curve_a] != [r["step"] for r in curve_b]:
        return {"dev_steps": float("inf")}
    dev = max((abs(x[k] - y[k]) for x, y in zip(curve_a, curve_b) for k in ("dev_loss", "dev_huber", "dev_rho_ST")), default=0.0)
    step = read_json(a / "SELECTED.json")["step"]
    if step != read_json(b / "SELECTED.json")["step"]:
        return {"selected_step": float("inf")}
    ta = load_file(str(a / f"checkpoint_{step:05d}" / "adapter_model.safetensors"))
    tb = load_file(str(b / f"checkpoint_{step:05d}" / "adapter_model.safetensors"))
    adapter = float("inf") if sorted(ta) != sorted(tb) else max(float(np.max(np.abs(ta[k] - tb[k]))) for k in ta)
    return {"loss": loss, "dev": dev, "adapter": adapter}


def start_and_kill_after_first_save(job: Job, partial: Path, *, timeout: float = 300.0,
                                    semaphore: threading.Semaphore | None = None) -> dict:
    """Start a training script and stop it the way Slurm stops a job at its time limit (SIGTERM to the process group) once its first
    resumable save, `partial`/STATE.json, exists. The group is frozen first, so nothing moves between the look and the kill.
    Returns the update that was saved, the return code of the script and whether the unit had already completed."""
    with semaphore or contextlib.nullcontext():
        return _kill_after_first_save(job, partial, timeout)


def _kill_after_first_save(job: Job, partial: Path, timeout: float) -> dict:
    proc = job.start()
    deadline = time.monotonic() + timeout
    try:
        while not (partial / "STATE.json").is_file():
            if proc.poll() is not None:
                ended = Outcome(job, proc.returncode, 0).describe()
                raise AssertionError(f"the training script ended before its first resumable save:\n{ended}")
            if time.monotonic() > deadline:
                raise AssertionError(f"no resumable save after {timeout:.0f} s")
            time.sleep(0.01)
        os.killpg(proc.pid, signal.SIGSTOP)
        saved = partial if (partial / "STATE.json").is_file() else partial.with_name(partial.name + ".old")      # between its two renames
        state = read_json(saved / "STATE.json")
        finished = (partial.parent / partial.name.removesuffix(".partial") / "COMPLETE.json").is_file()
        os.killpg(proc.pid, signal.SIGTERM)
        os.killpg(proc.pid, signal.SIGCONT)
        rc = proc.wait(timeout=60)
    finally:
        kill_group(proc)
        _LIVE.discard(proc)
    return {"saved_update": state["step"], "returncode": rc, "finished_before_the_kill": finished}


def kill_and_resume(reference: Rehearsal, second: Rehearsal, *, semaphore: threading.Semaphore | None = None,
                    job_id: str = "7777") -> dict:
    """The claim behind the TIMEOUT advice ("training continues from the latest save"), as an experiment.

    `second` is a campaign with the recipe of `reference` that has been planned and submitted to the fake sbatch but not run; the
    contexts and the teacher table of `reference` are copied into it, so the training is the only thing that differs. Its train
    script is stopped after the first resumable save (SIGTERM), then run again under another job id. Compare the result with the
    uninterrupted `reference` using training_difference() once the reference has finished its own training."""
    assert second.plan.campaign_id == reference.plan.campaign_id, "the second campaign must have the recipe of the first"
    for part in ("contexts", "teacher"):
        shutil.copytree(reference.root / part, second.root / part)
    unit = second.plan.stages[[s.name for s in second.plan.stages].index("train")].tasks[0]["unit"]
    job = second.job("train")
    killed = start_and_kill_after_first_save(job, second.root / "train" / f"{unit}.partial", semaphore=semaphore)
    killed.update(log=job.text("out"), complete=(second.root / "train" / unit / "COMPLETE.json").exists())
    again = replace(job, job_id=job_id, env={**job.env, "SLURM_JOB_ID": job_id})
    return {"unit": unit, "killed": killed, "resumed": run_job(again, semaphore=semaphore),
            "reference": reference.root / "train" / unit, "second": second.root / "train" / unit}
