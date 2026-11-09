"""Helpers for the planner, profile and script-renderer tests.

    from fx_render import clean_mpd_env, make, opts, spec, stage  # noqa: F401

`clean_mpd_env` is an autouse fixture once imported into a test module: it removes every MPD_* variable and SLURM_* so a
developer's shell cannot change a plan. `ScriptRun` executes a rendered sbatch script under bash with stand-ins for
`module`, `nvidia-smi` and the Python interpreter, which is how the prologue is tested for what it does, not only for its text.
"""

from __future__ import annotations

import dataclasses
import os
import stat
import subprocess
from pathlib import Path

import pytest

from mpd.slurm.cluster import Cluster
from mpd.slurm.plan import PlanOptions, make_plan
from mpd.slurm.render import render_script
from mpd.spec import RunSpec

HOME = Path("/nonexistent/mpd")    # a fixed made-up path: nothing is written there, plans only compute with it
NONCE = "0123456789ab"
ACCOUNT = "123456789012"          # the documented fake account
METHODS = ("direct-all", "direct-one", "score-all", "score-one")


@pytest.fixture(autouse=True)
def clean_mpd_env(monkeypatch):
    for name in list(os.environ):
        if name.startswith(("MPD_", "SLURM_")):
            monkeypatch.delenv(name)


def spec(**kw) -> RunSpec:
    base = dict(teacher="esm2-3b", student="esm2-650m", train_data="proteingym", method="score-all",
                eval_data=("proteingym", "venusmuthub"))
    base.update(kw)
    if base["method"].endswith("one") and not base.get("assays"):
        base["assays"] = ("A", "B", "C")
    return RunSpec(**base)


def opts(home=HOME, cluster="grace", **kw) -> PlanOptions:
    kw.setdefault("account", ACCOUNT if cluster == "grace" else None)
    kw.setdefault("nonce", NONCE)
    kw.setdefault("python", "/env/bin/python")
    return PlanOptions(cluster=Cluster.load(cluster), home=Path(home), **kw)


def make(cluster="grace", home=HOME, spec_kw=None, **kw):
    return make_plan(spec(**(spec_kw or {})), opts(home, cluster, **kw))


def stage(plan, name):
    return next(s for s in plan.stages if s.full_name == name)


def render_all(plan) -> dict[str, str]:
    return {s.full_name: render_script(plan, s) for s in plan.stages}


def bash_n(text: str, tmp_path: Path) -> subprocess.CompletedProcess:
    path = tmp_path / "script.sbatch"
    path.write_text(text, encoding="utf-8", newline="")
    return subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True)


def directives(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if ln.startswith("#SBATCH")]


def replace_options(plan, **kw):
    return dataclasses.replace(plan, options=dataclasses.replace(plan.options, **kw))


def _executable(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


class ScriptRun:
    """Run one rendered stage script in a sandbox of stand-ins and look at what it did.

    `python` is a stand-in that writes its arguments and its environment to files instead of importing mpd; `module` is a shell
    function (installed through BASH_ENV, as Lmod is on a cluster) that logs its calls and can be told to fail; `nvidia-smi`
    succeeds or fails on request. Nothing here touches Slurm, a GPU or the network.
    """

    def __init__(self, tmp_path: Path, plan, name: str, *, module: str | None = "log", gpu_ok: bool = True,
                 module_fails_on: str | None = None, env: dict | None = None):
        self.root = tmp_path / f"run_{name.replace('.', '_')}"
        self.bin = self.root / "bin"
        self.bin.mkdir(parents=True)
        self.out = self.root / "out"
        self.out.mkdir()
        self.plan, self.stage = plan, stage(plan, name)
        self.python = _executable(self.bin / "python", f'#!/bin/bash\nprintf "%s\\n" "$@" > {self.out}/args\nenv > {self.out}/env\n')
        _executable(self.bin / "nvidia-smi", "#!/bin/bash\n" + ('echo "GPU 0: NVIDIA Fake (UUID: x)"\n' if gpu_ok else "exit 9\n"))
        self.module_log = self.root / "module.log"
        self.bash_env = None
        if module:
            fail = f'if [ "$1 $2" = "load {module_fails_on}" ]; then return 1; fi; ' if module_fails_on else ""
            # reads an unset variable on purpose: Lmod's init does, and `set -u` must not be active yet
            self.bash_env = _executable(
                self.root / "bash_env.sh", f'module() {{ echo "$*" >> {self.module_log}; : "$UNSET_LIKE_LMOD"; {fail}return 0; }}\n')
        self.plan = dataclasses.replace(plan, options=dataclasses.replace(plan.options, python=str(self.python)))
        assert tmp_path in self.plan.campaign_dir.parents, "build the plan with home=tmp_path / 'home': the script cd's into it"
        self.plan.campaign_dir.mkdir(parents=True, exist_ok=True)
        self.script = self.root / "job.sbatch"
        self.script.write_text(render_script(self.plan, self.stage), encoding="utf-8", newline="")
        self.env = {"PATH": f"{self.bin}:/usr/bin:/bin", "HOME": str(self.root), "SLURM_JOB_ID": "4242",
                    "SLURM_ARRAY_TASK_ID": "3", "PYTHONPATH": "/stray/pythonpath", "PYTHONHOME": "/stray/pythonhome",
                    **({"BASH_ENV": str(self.bash_env)} if self.bash_env else {}), **(env or {})}

    def run(self) -> subprocess.CompletedProcess:
        return subprocess.run(["bash", str(self.script)], env=self.env, capture_output=True, text=True, timeout=60)

    def args(self) -> list[str]:
        return (self.out / "args").read_text(encoding="utf-8").splitlines()

    def job_env(self) -> dict[str, str]:
        out: dict[str, str] = {}
        key = None
        for line in (self.out / "env").read_text(encoding="utf-8").splitlines():     # values may span lines only if they contain newlines
            if "=" in line and line.split("=", 1)[0].isidentifier():
                key, value = line.split("=", 1)
                out[key] = value
            elif key:
                out[key] += "\n" + line
        return out

    def module_calls(self) -> list[str]:
        return self.module_log.read_text(encoding="utf-8").splitlines() if self.module_log.exists() else []
