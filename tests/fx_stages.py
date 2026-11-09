"""Helpers for the stage tests: a planned campaign on the tiny data, stage runners, readers for what training writes.

Import what you need into the test module (`from fx_stages import env, plan_for  # noqa: F401`); `env` is a fixture.
"""

from __future__ import annotations

import csv
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from mpd.slurm.cluster import Cluster
from mpd.slurm.plan import PlanOptions, make_plan
from mpd.slurm.submit import write_campaign
from mpd.spec import RunSpec


@pytest.fixture
def env(monkeypatch, tiny_home, tiny_models):
    monkeypatch.setenv("MPD_HOME", str(tiny_home))
    monkeypatch.setenv("MPD_DTYPE", "float32")
    monkeypatch.setenv("MPD_MODEL_ESM2_8M", str(tiny_models["student"]))
    monkeypatch.setenv("MPD_MODEL_ESM2_35M", str(tiny_models["teacher"]))
    return tiny_home


def plan_for(home, score_shards=2, **kw):
    """A written campaign on the tiny data. Defaults: ESM2-35M teacher, ESM2-8M student, ProteinGym for training and
    evaluation; any RunSpec field (teacher, student, train_data, eval_data, method, ...) can be given."""
    kw = {"teacher": "esm2-35m", "student": "esm2-8m", "train_data": "proteingym", "eval_data": ("proteingym",),
          "method": "direct-all", "updates": 6, "global_batch": 4, "dev_fraction": 0.3, **kw}
    if kw["method"].endswith("all"):
        kw.setdefault("train_gpus", 1)
    spec = RunSpec(**kw)
    plan = make_plan(spec, PlanOptions(cluster=Cluster.load("vision"), home=home, python=sys.executable,
                                       score_shards={dataset: score_shards for dataset in kw["eval_data"]}))
    write_campaign(plan)
    return plan


def run_stages(plan, only=None):
    for stage in plan.stages:
        if only and stage.name not in only:
            continue
        module = importlib.import_module(stage.module)
        for i in range(len(stage.tasks)):
            module.main(["--campaign-dir", str(plan.campaign_dir), "--stage", stage.name, "--task-index", str(i + stage.task_offset)])


def curve(path):
    return [json.loads(line) for line in open(path / "dev_curve.jsonl")]


def step_log(path):
    return list(csv.DictReader(open(path / "train_steps.csv")))


def adapters(train_dir: Path) -> dict[int, dict[str, np.ndarray]]:
    """{update: {tensor name: array}} for every adapter kept in a train/<unit> folder."""
    from safetensors.numpy import load_file
    return {int(p.name.split("_")[1]): load_file(str(p / "adapter_model.safetensors"))
            for p in sorted(train_dir.glob("checkpoint_*"))}


def assert_same_training(a: Path, b: Path) -> None:
    """Two train/<unit> folders hold the same run: logged losses, development curve and every kept adapter."""
    la, lb = step_log(a), step_log(b)
    assert [r["step"] for r in la] == [r["step"] for r in lb]
    for col in ("loss", "huber", "lr", "grad_norm"):
        np.testing.assert_allclose([float(r[col]) for r in la], [float(r[col]) for r in lb], rtol=0, atol=1e-6, err_msg=col)
    ca, cb = curve(a), curve(b)
    assert [r["step"] for r in ca] == [r["step"] for r in cb]
    for ra, rb in zip(ca, cb):
        for key in ("dev_loss", "dev_huber", "dev_rho_ST"):
            assert ra[key] == pytest.approx(rb[key], abs=1e-6), (key, ra, rb)
    ta, tb = adapters(a), adapters(b)
    assert sorted(ta) == sorted(tb) and ta
    for step in ta:
        assert sorted(ta[step]) == sorted(tb[step])
        for name in ta[step]:
            np.testing.assert_allclose(ta[step][name], tb[step][name], rtol=0, atol=1e-6, err_msg=f"{name} at update {step}")


class Killed(Exception):
    """Raised from the on_update hook to stand in for a job killed at its walltime."""


def kill_after(update: int):
    def hook(step):
        if step == update:
            raise Killed(f"simulated kill after update {update}")
    return hook


RUNNER = '''\
import os
import sys

from mpd.stages import train

stop_after, only_rank = int(sys.argv[1]), int(sys.argv[2])


def hook(step):
    if step == stop_after and only_rank in (-1, int(os.environ["RANK"])):
        raise RuntimeError(f"simulated kill after update {step}")


train.main(sys.argv[3:], on_update=hook if stop_after else None)
'''


def start_ddp_train(campaign, tmp_path: Path, name: str, stop_after: int = 0, only_rank: int = -1):
    """Launch the train stage of a campaign (a plan or its directory) on 2 gloo processes, as the sbatch script does.

    With `stop_after`, a small runner script is launched instead and the processes (only `only_rank` if given) raise
    after that update.
    """
    import mpd
    args = ["--campaign-dir", str(getattr(campaign, "campaign_dir", campaign)), "--stage", "train"]
    if stop_after:
        script = tmp_path / "ddp_runner.py"
        script.write_text(RUNNER)
        target = [str(script), str(stop_after), str(only_rank), *args]
    else:
        target = ["-m", "mpd.stages.train", *args]
    src = str(Path(mpd.__file__).resolve().parents[1])
    cmd = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc_per_node=2", *target]
    log_path = tmp_path / f"{name}.log"
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, text=True,
                                env={**os.environ, "OMP_NUM_THREADS": "1",
                                     "PYTHONPATH": os.pathsep.join(filter(None, [src, os.environ.get("PYTHONPATH")]))})
    proc.log_path = log_path
    return proc


def finish_ddp_train(proc, timeout: int = 900) -> tuple[int, str]:
    """(exit code, everything the launch printed); quote `log[-3000:]` in assertion messages."""
    code = proc.wait(timeout=timeout)
    return code, proc.log_path.read_text()
