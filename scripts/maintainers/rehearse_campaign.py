#!/usr/bin/env python3
"""Rehearse campaigns: run the sbatch scripts that `mpd submit` writes, with bash, on this machine's CPU, and print a transcript.

    python scripts/maintainers/rehearse_campaign.py                         # two tiny campaigns in a throwaway home
    python scripts/maintainers/rehearse_campaign.py --method score-one --cluster vision
    python scripts/maintainers/rehearse_campaign.py --dry-run               # plan, write and show the scripts, run nothing
    python scripts/maintainers/rehearse_campaign.py --home DIR --python DIR/env/bin/python --teacher esm2-35m --student esm2-8m ...

The campaign is planned, written and "submitted" exactly as `mpd submit` does it (make_plan, write_campaign, submit), against
fx_slurm's fake sbatch. Then every script is run with `bash`, in dependency order and every task of an array, with a fake `module`
(entered through a read of an unset variable, as Grace's /etc/bashrc does), a fake `nvidia-smi`, made-up SLURM_* variables, an empty
$HOME and a PATH without a Slurm client. The jobs run on the CPU (`--env MPD_REQUIRE_CUDA=0`, which the scripts export after the GPU
guard), and a tripwire in the job interpreter refuses every connection that would leave the machine. The same toolbox runs under
tests/test_rehearsal.py (tests/fx_rehearsal.py).

By default the home is synthetic: tiny random ESM2 models for the two keys and the three tiny assays of tests/tinydata.py, in a
folder that is removed afterwards (--keep, or --work DIR, keeps it). With --home an existing home is used as it is, with the data and
weights `mpd fetch` left in it (the models named in MPD_MODEL_<KEY> variables of this shell count too); the stages then run real
weights on the CPU, which is slow and meant for a small student and a handful of updates.

A pass means the scripts of this checkout run on a CPU from their first line to MPD_STAGE_DONE and leave the files that `mpd status`,
`mpd results` and `mpd submit --force` read. It says nothing about Grace's module names, GPU drivers, NCCL, memory or queue, or about
the Slurm version: nothing here ran on Grace, on a GPU or through a real scheduler. No Slurm client is ever called.

Needs the packages of the test environment (`pip install -e '.[train,test]'`). Exit status: 0 when every campaign passed, 1 otherwise.
"""

from __future__ import annotations

import argparse
import datetime
import os
import re
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "src"), str(REPO / "tests")]

try:
    import fx_rehearsal as rt
except ImportError as exc:                                  # pytest, torch, transformers or peft is missing
    sys.exit(f"rehearse_campaign.py needs the test environment: pip install -e '.[train,test]' ({exc})")

from mpd.results import render as render_results          # noqa: E402
from mpd.slurm.plan import describe, parse_env_pair       # noqa: E402

METHODS = ("direct-all", "direct-one", "score-all", "score-one")


def parse(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="A pass shows the scripts run on a CPU. It says nothing about Grace, a GPU or a real scheduler.")
    recipe = ap.add_argument_group("recipe (as for `mpd plan`; the defaults are the tiny synthetic one)")
    recipe.add_argument("--method", nargs="+", choices=METHODS, default=["direct-all", "score-all"], help="one campaign per method")
    recipe.add_argument("--teacher", default="esm2-35m")
    recipe.add_argument("--student", default="esm2-8m")
    recipe.add_argument("--train-data", default="proteingym", choices=["proteingym", "venusmuthub"])
    recipe.add_argument("--eval-data", nargs="+", default=["proteingym"], choices=["proteingym", "venusmuthub"])
    recipe.add_argument("--assay", nargs="+", default=[], help="training assays of the *-one methods (synthetic home: ASSAY_A ASSAY_B)")
    recipe.add_argument("--updates", type=int, default=24)
    recipe.add_argument("--global-batch", type=int, default=4)
    recipe.add_argument("--score-shards", type=int, default=2, help="score tasks per evaluation dataset")
    recipe.add_argument("--train-gpus", type=int, help="training width (default: the profile's; two processes on Vision)")
    where = ap.add_argument_group("where")
    where.add_argument("--cluster", default="grace", help="the cluster profile whose scripts are rehearsed (grace or vision)")
    where.add_argument("--home", type=Path, help="use this home instead of a synthetic one")
    where.add_argument("--python", help="interpreter of the jobs (default: the home's environment record, else <home>/env/bin/python)")
    where.add_argument("--env", action="append", default=[], metavar="K=V", help="more job environment, as `mpd submit --env`")
    where.add_argument("--work", type=Path, help="folder for the synthetic home and the logs (kept); default: a temporary one")
    where.add_argument("--keep", action="store_true", help="do not remove the temporary folder at the end")
    run = ap.add_argument_group("running")
    run.add_argument("--jobs", type=int, default=rt.WORKERS, help="jobs running at a time over all campaigns")
    run.add_argument("--timeout", type=float, default=rt.JOB_TIMEOUT_S, help="seconds one job may take")
    run.add_argument("--dry-run", action="store_true", help="plan, write and list the scripts in the order they would run; run nothing")
    run.add_argument("--kill-resume", action="store_true",
                     help="also stop a training job after its first save and run it again, as a TIMEOUT does (a *-all recipe)")
    args = ap.parse_args(argv)
    if args.jobs < 1:
        ap.error("--jobs must be at least 1")
    try:
        args.env = [parse_env_pair(pair) for pair in args.env]
    except ValueError as exc:
        ap.error(str(exc))
    if args.home is not None and args.kill_resume:
        ap.error("--kill-resume needs the synthetic home (it copies a finished campaign into a second home); drop --home")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.cluster):
        ap.error("--cluster takes the name of a profile (grace, vision, ...), not a path")
    return args


class Transcript:
    def __init__(self) -> None:
        self.start, self.lock = time.monotonic(), threading.Lock()

    def __call__(self, text: str = "") -> None:
        with self.lock:
            print(text, flush=True)

    def stamp(self) -> str:
        return f"[+{time.monotonic() - self.start:6.1f}s]"


def recipes_of(args) -> list:
    synthetic = args.home is None
    assays = tuple(args.assay) or (("ASSAY_A", "ASSAY_B") if synthetic else ())
    env = (*rt.CPU_ENV, *args.env)
    return [rt.Recipe(method=m, teacher=args.teacher, student=args.student, train_data=args.train_data, eval_data=tuple(args.eval_data),
                      assays=assays if m.endswith("one") else (), updates=args.updates, global_batch=args.global_batch,
                      score_shards=args.score_shards, train_gpus=args.train_gpus if m.endswith("all") else None, env=env)
            for m in args.method]


def setup_synthetic(args, work: Path, say: Transcript):
    """A node, a home with tiny data and a venv, and the MPD_MODEL_<KEY> variables of two tiny random models."""
    say("building the synthetic home: a venv that reaches this environment through a .pth file, tiny random ESM2 models, tiny data")
    if args.dry_run:                                            # nothing runs, so empty folders do for the paths the scripts name
        models = {"MPD_MODEL_ESM2_8M": work / "models" / "student", "MPD_MODEL_ESM2_35M": work / "models" / "teacher"}
        for folder in models.values():
            folder.mkdir(parents=True, exist_ok=True)
    else:
        from tinydata import make_tiny_esm
        models = {"MPD_MODEL_ESM2_8M": make_tiny_esm(work / "models" / "student", seed=1),
                  "MPD_MODEL_ESM2_35M": make_tiny_esm(work / "models" / "teacher", seed=2)}
    wanted = {f"MPD_MODEL_{key.upper().replace('-', '_')}" for key in (args.teacher, args.student)}
    if not wanted <= set(models):
        sys.exit(f"the synthetic home has tiny weights for esm2-8m (student) and esm2-35m (teacher) only, not for {args.teacher} -> "
                 f"{args.student}; use --home for other models")
    venv = rt.build_venv(work / "venv" / "env")
    node = rt.Node(work / "node")
    home = rt.make_home(work, args.cluster, venv)
    every = {"MPD_CKPT_EVERY": str(max(1, args.updates // 6))} if args.kill_resume else {}     # a save to stop after, not the last
    return node, home, node.shell(**{k: str(v) for k, v in models.items()}, **every), venv


def setup_existing(args, work: Path):
    node = rt.Node(work / "node")
    return node, args.home.resolve(), node.shell(**{k: v for k, v in os.environ.items() if k.startswith("MPD_")})


def show_campaign(rh, say: Transcript) -> None:
    say()
    say(f"=== {rh.recipe.label}")
    say(describe(rh.plan))
    say()
    say("submitted to the fake sbatch: " + ", ".join(f"{stage}={job}" for stage, job in rh.ids.items()))
    for stage in rh.plan.stages:
        n = len(rh.jobs[stage.full_name])
        tasks = f"   x {n} (SLURM_ARRAY_TASK_ID 0..{n - 1})" if stage.is_array else ""
        say(f"  would run  bash {rh.scripts[stage.full_name]}{tasks}")


def report(rh, say: Transcript, synthetic: bool) -> list[str]:
    say()
    say(f"=== {rh.recipe.label}: {rh.plan.campaign_id}")
    say(f"{'stage':<14}{'jobs':>5}{'exit':>6}{'seconds':>9}  files that mark it finished")
    for stage in rh.plan.stages:
        outcomes = rh.run.outcomes.get(stage.full_name, [])
        codes = ",".join(sorted({str(o.rc) for o in outcomes})) or "-"
        seconds = f"{max((o.seconds for o in outcomes), default=0.0):.1f}"
        gone = rh.run.missing.get(stage.full_name) or []
        marks = "not started" if stage.full_name in rh.run.skipped else f"missing {len(gone)}" if gone or not outcomes else "all present"
        say(f"{stage.full_name:<14}{len(outcomes):>5}{codes:>6}{seconds:>9}  {marks}")
    text, _ = render_results(rh.root)
    say()
    say("mpd results (random tiny models: the numbers mean nothing)" if synthetic else "mpd results")
    say(text)
    found = rt.problems(rh)
    for line in found:
        say(f"PROBLEM  {line}")
    for outcome in rh.run.failures[:2]:
        say(outcome.describe())
    return found


def kill_resume(args, work: Path, venv: Path, reference, say: Transcript, semaphore) -> list[str]:
    say()
    say(f"=== TIMEOUT experiment: stop the training of {reference.recipe.label} after its first save, run the same script again")
    node = rt.Node(work / "node_resume")
    home = rt.make_home(work / "resume", args.cluster, venv)
    shell = dict(node.shell(**{k: v for k, v in reference.shell.items() if k.startswith("MPD_")}))
    second = rt.prepare_campaign(node, home, args.cluster, reference.recipe, shell)
    try:
        result = rt.kill_and_resume(reference, second, semaphore=semaphore)
    except AssertionError as exc:
        return [f"the experiment could not run: {exc}"]
    killed, resumed = result["killed"], result["resumed"]
    say(f"SIGTERM after the save at update {killed['saved_update']} (exit {killed['returncode']}); the unit was not complete")
    say(f"the same script again: exit {resumed.rc} in {resumed.seconds:.1f} s")
    diff = rt.training_difference(result["reference"], result["second"])
    say(f"largest difference to the uninterrupted run: {diff}")
    bad = [f"the resumed job exited with {resumed.rc}"] if resumed.rc else []
    return bad + ([f"the resumed run differs from the uninterrupted one by {diff}"] if not diff or max(diff.values()) > 1e-6 else [])


def main(argv: list[str] | None = None) -> int:
    args = parse(argv)
    say = Transcript()
    synthetic = args.home is None
    temporary = args.work is None
    work = Path(args.work or tempfile.mkdtemp(prefix="mpd-rehearsal-")).resolve()
    work.mkdir(parents=True, exist_ok=True)
    say(f"rehearsal {datetime.datetime.now().isoformat(timespec='seconds')}  cluster {args.cluster}  python {sys.version.split()[0]}")
    say("This runs the job scripts of this checkout on the CPU with fake tools: it proves nothing about Grace, a GPU or a scheduler.")
    venv = None
    try:
        node, home, shell, *rest = setup_synthetic(args, work, say) if synthetic else (*setup_existing(args, work), None)
        venv = rest[0]
        options = {"python": args.python} if args.python else {}
        prepared = [rt.prepare_campaign(node, home, args.cluster, recipe, shell, **options) for recipe in recipes_of(args)]
        known = node.fs.scenario()["module"]["available"]
        recorded = {m for rh in prepared for m in rh.plan.modules}
        node.fs.set_modules(sorted({*known, *recorded}))                                     # the fake cluster has the recorded modules
        for rh in prepared:
            show_campaign(rh, say)
        if args.dry_run:
            say()
            say("dry run: nothing was run")
            return 0
        say()
        say(f"running the scripts, at most {args.jobs} at a time (a script waits for the stages it depends on, as with afterok)")
        semaphore = threading.Semaphore(args.jobs)
        rt.run_prepared(prepared, semaphore=semaphore, timeout=args.timeout, on_job=lambda rh, o: say(
            f"{say.stamp()} {rh.recipe.method:<11} {o.job.stage:<12} job {o.job.job_id:<8} exit {o.rc:<3} {o.seconds:5.1f} s"))
        wrong = {rh.recipe.method: report(rh, say, synthetic) for rh in prepared}
        if args.kill_resume:
            reference = next((rh for rh in prepared if rh.recipe.method.endswith("all") and rh.run.ok), None)
            wrong["kill-resume"] = (kill_resume(args, work, venv, reference, say, semaphore) if reference
                                    else ["no *-all campaign passed, so there is no uninterrupted training to compare with"])
        say()
        for name, found in wrong.items():
            say(f"{name:<12} {'PASSED' if not found else 'FAILED'}" + "".join(f"\n    {line}" for line in found))
        verdict = "PASSED" if not any(wrong.values()) else "FAILED"
        say(f"{verdict} in {time.monotonic() - say.start:.0f} s. The jobs ran on the CPU with fake tools; this says nothing about "
            "Grace's modules, drivers, NCCL, memory or queue.")
        return 0 if verdict == "PASSED" else 1
    finally:
        rt.kill_all()
        if temporary and not args.keep:
            shutil.rmtree(work, ignore_errors=True)
        else:
            say(f"kept: {work}")


if __name__ == "__main__":
    sys.exit(main())
