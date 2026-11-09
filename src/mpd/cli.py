"""`mpd` command line. Planning and submission only: no torch import, safe on a login node."""

from __future__ import annotations

import argparse
import importlib
import os
import re
import shutil
import sys
import textwrap
import time
from collections import deque
from pathlib import Path

from . import __version__, preflight, registry, results
from .checks import Check
from .constants import DATASETS, METHODS
from .data import sources as data_sources
from .data.fetch import TimeBudgetReached, fetch_source
from .results import table
from .slurm.cluster import Cluster, ClusterError, available_clusters
from .slurm.plan import PlanOptions, default_python, describe, make_plan, parse_env_pair, parse_score_shards
from .slurm.submit import FORCE_APPLIES, submit, write_campaign
from .spec import RunSpec

DEFAULT_MAX_SU = 10_000
EXIT_BUDGET = 3                       # `mpd fetch model` stopped before the login-node time limit
FAILURE_STATES = ("FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL", "DEADLINE")
NO_HOME = "give --home, or --cluster (or set MPD_HOME / MPD_CLUSTER)"

EPILOG = """\
typical order (details in README.md and docs/GRACE.md):
  bash scripts/setup_env.sh --cluster grace         build the job environment (login node)
  mpd fetch proteingym                              download data and weights (login node; compute nodes are offline)
  mpd fetch model esm2-3b esm2-650m
  mpd doctor                                        check what a run needs: environment, data, weights, quota, Slurm
  mpd plan ...   then   mpd submit ...              see the plan and cost, then submit
  mpd status | logs | cancel | results CAMPAIGN     follow a campaign

Run `mpd doctor` first when something does not work, and paste `mpd doctor --report` into an issue."""


class CliError(Exception):
    """A problem with how the command was used; the message says how to fix it."""


# ---------------------------------------------------------------- argument types
def _number(kind, text: str):
    try:
        return kind(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a {'whole number' if kind is int else 'number'}") from None


def _positive_int(text: str) -> int:
    value = _number(int, text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1; got {value}")
    return value


def _positive_float(text: str) -> float:
    value = _number(float, text)
    if not value > 0:
        raise argparse.ArgumentTypeError(f"must be greater than 0; got {text}")
    return value


def _nonnegative_float(text: str) -> float:
    value = _number(float, text)
    if not value >= 0:
        raise argparse.ArgumentTypeError(f"must be 0 or more; got {text}")
    return value


# ---------------------------------------------------------------- parsers
def _debug_parent() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--debug", action="store_true", default=argparse.SUPPRESS,
                   help="show the traceback of an error instead of the one-line message")
    return p


def _recipe_args(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("what to distill")
    g.add_argument("--teacher", required=True, help="teacher model key (see `mpd models`)")
    g.add_argument("--student", required=True, help="student model key (see `mpd models`)")
    g.add_argument("--train-data", required=True, choices=DATASETS, help="benchmark whose sequences/positions the student is trained on")
    g.add_argument("--method", required=True, choices=list(METHODS),
                   help="direct-all | direct-one | score-all | score-one  (objective x training scope)")
    g.add_argument("--assay", nargs="+", default=[], metavar="ASSAY",
                   help="training assay id(s); required for *-one methods (list them with `mpd assays`)")
    g.add_argument("--assays-file", type=Path, help="file with one training assay id per line")
    g.add_argument("--eval-data", nargs="+", required=True, choices=DATASETS, help="benchmark(s) to evaluate on")
    g.add_argument("--teacher-backend", choices=["cache", "live"], help="default: cache if the teacher supports it, else live")
    g.add_argument("--seed", type=int, default=RunSpec.seed)

    r = p.add_argument_group("recipe overrides (defaults are the per-family presets in mpd/presets.py, taken from the "
                             "earlier study's runs; `mpd plan` prints the recipe it will use)")
    r.add_argument("--updates", type=_positive_int)
    r.add_argument("--global-batch", type=_positive_int)
    r.add_argument("--lr", type=_positive_float, dest="peak_lr")
    r.add_argument("--kl-weight", type=_nonnegative_float)
    r.add_argument("--dev-fraction", type=float)

    c = p.add_argument_group("cluster", "Another --cluster, --gpu, --train-gpus or --score-shards (and another --eval-data) is another campaign "
                                        "that computes everything again. The walltimes, --array-throttle, --account, --home and --python only decide "
                                        "where and when the jobs run: they keep the campaign, and `mpd submit --force` applies them to the stages "
                                        "it queues again (after a TIMEOUT, a longer --train-time).")
    c.add_argument("--cluster", default=os.environ.get("MPD_CLUSTER"),
                   help=f"profile ({', '.join(available_clusters())}) or YAML path (default: $MPD_CLUSTER)")
    c.add_argument("--account", help="billing account (Grace: the number `myproject` prints). Default: $MPD_ACCOUNT, then "
                                     "the cluster profile, then your Slurm default account")
    c.add_argument("--gpu", help="GPU type for GPU stages (default from the cluster profile; see `mpd clusters`)")
    c.add_argument("--train-gpus", type=int, help="data-parallel GPUs for *-all training, on one node (default from the profile)")
    c.add_argument("--home", help="working root for data, models, environment and runs (default: $MPD_HOME, else the cluster "
                                  "profile; on Grace never under /home)")
    c.add_argument("--python", help="interpreter the jobs run (default: $MPD_PYTHON, else the one recorded by "
                                    "scripts/setup_env.sh, else <home>/env/bin/python)")
    c.add_argument("--train-time", help="override the estimated training walltime, e.g. 08:00:00")
    c.add_argument("--score-time", help="walltime per scoring task, e.g. 02:00:00 (default: estimated)")
    c.add_argument("--teacher-time", help="walltime for live teacher inference, e.g. 06:00:00 (default: estimated)")
    c.add_argument("--array-throttle", type=_positive_int, help="max concurrent array tasks")
    c.add_argument("--score-shards", help="positions split per dataset, e.g. proteingym=4,venusmuthub=1")
    c.add_argument("--allow-unsupported-gpu", action="store_true",
                   help="train on a GPU without native bf16 (Grace rtx, t4) in float32 instead of refusing")

    j = p.add_argument_group("job script (repeatable)", "These keep the campaign, except MPD_DTYPE, MPD_TF32 and MPD_STRUCTURE_LEGACY_MASK, which "
                                                         "change results and so start their own. After a bad node: --sbatch-arg=--exclude=NODE and --force.")
    j.add_argument("--env", action="append", default=[], metavar="K=V", help="export K=V in every job script")
    j.add_argument("--module", action="append", default=[], metavar="M", help="also `module load` M in every job script")
    j.add_argument("--sbatch-arg", action="append", default=[], metavar="ARG",
                   help="extra #SBATCH flag; write it with an equals sign, e.g. --sbatch-arg=--exclude=g006")
    p.add_argument("--max-su", type=_positive_float, default=DEFAULT_MAX_SU, metavar="N",
                   help=f"refuse to submit when the estimated upper bound exceeds N service units (default {DEFAULT_MAX_SU:,})")


def _parser() -> argparse.ArgumentParser:
    debug = _debug_parent()
    parser = argparse.ArgumentParser(
        prog="mpd", parents=[debug], formatter_class=argparse.RawDescriptionHelpFormatter, epilog=EPILOG,
        description="Distill protein mutation-effect predictors into smaller students, plan the work and submit it "
                    "to Slurm. It runs on a login node and never imports torch.")
    parser.add_argument("--version", action="version", version=f"mpd {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True, metavar="COMMAND")

    def add(name, fn, help_):
        p = sub.add_parser(name, help=help_, description=help_, parents=[debug])
        p.set_defaults(fn=fn)
        return p

    def where(p):
        p.add_argument("--cluster", default=os.environ.get("MPD_CLUSTER"), help="cluster profile (default: $MPD_CLUSTER)")
        p.add_argument("--home", help="working root (default: $MPD_HOME, else the cluster profile)")

    p = add("plan", cmd_plan, "show the stages, resources, cost estimate and what is still missing; writes nothing")
    _recipe_args(p)
    p = add("submit", cmd_submit, "write the scripts, check them and submit them to Slurm with dependencies")
    _recipe_args(p)
    p.add_argument("--dry-run", action="store_true",
                   help="write the scripts and run the checks (including `sbatch --test-only` when sbatch exists, which validates "
                        "the scripts and queues nothing), but submit no job")
    p.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    p.add_argument("--force", action="store_true",
                   help="continue or repair a campaign that was already submitted: queued, running and finished stages are kept and missing ones "
                        "are queued, with the walltimes and sbatch options of this command (a longer --train-time after a TIMEOUT, "
                        "--sbatch-arg=--exclude=NODE after a bad node). To run it again from scratch change --seed")

    p = add("status", cmd_status, "list campaigns, or show the Slurm state, progress and failures of one")
    p.add_argument("campaign", nargs="?", help="campaign directory, id or unique id prefix (default: list them all)")
    where(p)
    p = add("logs", cmd_logs, "show the newest log of a stage of a campaign")
    p.add_argument("campaign", help="campaign directory, id or unique id prefix")
    p.add_argument("stage", nargs="?", help="stage name, e.g. train or score-base (default: list the logs)")
    p.add_argument("--err", action="store_true", help="show the error log (.err) instead of the output log (.out)")
    p.add_argument("--tail", type=_positive_int, default=50, metavar="N", help="last N lines (default 50)")
    where(p)
    p = add("cancel", cmd_cancel, "cancel the queued and running jobs of a campaign")
    p.add_argument("campaign", help="campaign directory, id or unique id prefix")
    where(p)
    p = add("results", cmd_results, "print the evaluation tables of a finished campaign and where its adapters are")
    p.add_argument("campaign", help="campaign directory, id or unique id prefix")
    where(p)

    p = add("fetch", cmd_fetch, "download pinned data, model weights or Foldseek, checksum-verified (run on a login node)")
    p.add_argument("source", nargs="?", choices=[*dict.fromkeys([*data_sources.SOURCES, "model", "foldseek"])],
                   help="a data source, `model` (then name the models) or `foldseek`; none lists them")
    p.add_argument("keys", nargs="*", metavar="KEY", help="model keys for `fetch model` (see `mpd models`)")
    p.add_argument("--list", action="store_true", help="show the pinned sources (with `model`: the pinned models)")
    p.add_argument("--all", action="store_true", help="with `model`: every model whose status is available")
    where(p)
    p.add_argument("--from-dir", type=Path, help="copy the files from a local folder instead of downloading (offline clusters)")
    p.add_argument("--zenodo-record", help="Zenodo record id, for a source without a pinned record")
    p.add_argument("--no-unpack", action="store_true", help="download and verify only")
    p.add_argument("--models-dir", type=Path, help="with `model`: keep the weights here instead of <home>/models (or $MPD_MODELS_DIR)")
    p.add_argument("--verify", action="store_true", help="with `model`: re-hash every file already present")
    p.add_argument("--force", action="store_true", help="with `model`: download again even if the files look complete")
    p.add_argument("--dry-run", action="store_true", help="with `model`: say what would be downloaded")
    p.add_argument("--time-budget", type=float, metavar="MIN",
                   help="with `model`: stop cleanly after MIN minutes so a login-node limit does not kill the download "
                        "(default: the profile's limit minus 5; 0 = no limit). Run the same command again to resume")
    p.add_argument("--from-file", type=Path, help="with `foldseek`: use this downloaded tarball instead of the network")

    p = add("models", cmd_models, "list models: roles, status, licence, download size and whether the weights are fetched")
    where(p)
    p.add_argument("--all", action="store_true", help="also list the models this release does not offer (licence-locked)")
    add("clusters", cmd_clusters, "list cluster profiles")
    p = add("assays", cmd_assays, "list the assay ids of a fetched dataset")
    p.add_argument("--dataset", choices=DATASETS, help="default: both")
    p.add_argument("--grep", metavar="TEXT", help="only ids containing TEXT (case-insensitive); exit 1 if none")
    where(p)
    p = add("doctor", cmd_doctor, "check that this machine is ready: environment, data, weights, quota, Slurm tools")
    where(p)
    p.add_argument("--deep", action="store_true", help="also import torch in the job Python (slow on a cold login node)")
    p.add_argument("--verify", action="store_true", help="re-hash the fetched model weights")
    p.add_argument("--report", action="store_true",
                   help="print a summary with account numbers, user names and home paths removed, to paste into an issue")
    return parser


# ---------------------------------------------------------------- resolving where things are
def _absolute(path) -> Path:
    return Path(os.path.abspath(os.path.expanduser(str(path))))


def _load_cluster(name: str | None, *, required: bool = True) -> Cluster | None:
    if name:
        return Cluster.load(name)
    if required:
        raise CliError("--cluster is required (or set MPD_CLUSTER). Available: " + ", ".join(available_clusters()))
    return None


def _home(cluster: Cluster, override: str | None) -> Path:
    """The working root as an absolute path; the profile refuses a root under a forbidden prefix (Grace: /home)."""
    raw = override or os.environ.get("MPD_HOME")
    if raw and "{env:" not in raw:
        raw = str(_absolute(raw))
    return _absolute(cluster.storage_dir(raw or None))


def _optional_home(cluster: Cluster | None, override: str | None) -> Path | None:
    """The working root if it is known; a profile that cannot name one ($SCRATCH unset) gives None unless --home was given."""
    raw = override or os.environ.get("MPD_HOME")
    if cluster is None:
        return _absolute(raw) if raw else None
    try:
        return _home(cluster, override)
    except ClusterError:
        if override:
            raise
        return None


def _python(arg: str | None, home: Path) -> str:
    """--python, else $MPD_PYTHON, else the interpreter the environment record names, else <home>/env/bin/python."""
    return str(_absolute(arg)) if arg else default_python(home)


def _plan_options(args, cluster: Cluster, home: Path) -> PlanOptions:
    env: dict[str, str] = {}
    for text in args.env:
        key, value = parse_env_pair(text)
        if env.get(key, value) != value:
            raise CliError(f"--env sets {key} twice with different values")
        env[key] = value
    return PlanOptions(
        cluster=cluster, home=home, python=_python(args.python, home), gpu=args.gpu,
        account=args.account or os.environ.get("MPD_ACCOUNT") or None, train_time=args.train_time,
        score_time=args.score_time, teacher_time=args.teacher_time, array_throttle=args.array_throttle,
        score_shards=parse_score_shards(args.score_shards) if args.score_shards else None,
        extra_sbatch=tuple(args.sbatch_arg), extra_env=env, extra_modules=tuple(args.module),
        allow_unsupported_gpu=args.allow_unsupported_gpu)


def _assays_from_file(path: Path | None) -> list[str]:
    if not path:
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise CliError(f"cannot read --assays-file {path}: {exc.strerror or exc}") from None
    return [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]


def _build(args) -> tuple[RunSpec, PlanOptions]:
    cluster = _load_cluster(args.cluster)
    assays = [*args.assay, *_assays_from_file(args.assays_file)]
    spec = RunSpec(teacher=args.teacher, student=args.student, train_data=args.train_data, method=args.method,
                   eval_data=tuple(args.eval_data), assays=tuple(assays), teacher_backend=args.teacher_backend, seed=args.seed,
                   updates=args.updates, global_batch=args.global_batch, peak_lr=args.peak_lr,
                   kl_weight=args.kl_weight, dev_fraction=args.dev_fraction, train_gpus=args.train_gpus)
    return spec, _plan_options(args, cluster, _home(cluster, args.home))


# ---------------------------------------------------------------- the library calls other areas provide
def _need(module: str, what: str):
    """Import `module` lazily; a build without it says which command cannot work."""
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        if exc.name == module:
            raise CliError(f"{what} is not available in this build ({module} is missing)") from None
        raise


def _campaigns():
    return _need("mpd.slurm.campaign", "following campaigns")


def _models():
    return _need("mpd.data.models", "model weights")


# ---------------------------------------------------------------- plan and submit
def _ref(campaign_dir: Path, home: Path | None) -> str:
    """How to name the campaign in a command the user will paste: the id if $MPD_HOME finds it, else the path."""
    env_home = os.environ.get("MPD_HOME")
    return campaign_dir.name if home is not None and env_home and _absolute(env_home) == home else str(campaign_dir)


def cmd_plan(args) -> int:
    spec, opts = _build(args)
    plan = make_plan(spec, opts)
    print(describe(plan))
    checks = preflight.run_checks(spec, opts, plan=plan, level="plan", max_su=args.max_su)
    print("\n" + preflight.render(checks, mode="plan"))
    return 0


def _already_submitted(name: str, state) -> str:
    if state.state == "unreadable":
        return (f"{state.message or 'the SUBMITTED.json of this campaign cannot be read'}. Something may be queued: look at "
                f"`squeue -u $USER` before doing anything else. Nothing was written.")
    queued = ", ".join(f"{stage} {job}" for stage, job in state.jobs.items()) or "none recorded"
    applies = textwrap.fill(FORCE_APPLIES, width=100, initial_indent=" " * 6, subsequent_indent=" " * 6,
                            break_long_words=False, break_on_hyphens=False)
    if state.state == "partial":
        return (f"campaign {name} was only partly submitted (queued: {queued}). Nothing was written.\n"
                f"  mpd status {name}    see what is queued\n  mpd cancel {name}    cancel it\n"
                f"  add --force to this command to continue the chain: the stages not yet queued are submitted\n{applies}")
    return (f"campaign {name} was already submitted (jobs: {queued}). Nothing was written.\n"
            f"  mpd status {name}    follow it\n"
            f"  mpd cancel {name}    stop its queued and running jobs; after that the same command submits it again (finished stages are kept)\n"
            "  add --force to this command to continue or repair it: stages that are queued, running or finished are kept and the "
            f"others are queued\n{applies}\n"
            "  or change --seed for an independent run")


def _refusal(checks: list[Check]) -> str:
    errors = [c for c in checks if c.level == "error"]
    lines = [f"not submitting: {len(errors)} problem(s) to fix first (the table above has the rest)"]
    for c in errors:
        lines.append(f"  {c.message}" + (f"\n    fix: {c.fix}" if c.fix and c.fix not in c.message else ""))
    return "\n".join(lines)


def cmd_submit(args) -> int:
    spec, opts = _build(args)
    plan = make_plan(spec, opts)
    name = plan.campaign_dir.name
    print(describe(plan))
    api = _campaigns()
    state = api.campaign_state(plan.campaign_dir)
    blocked = state is not None and state.state in api.BLOCKING
    if blocked and not args.force:
        raise CliError(_already_submitted(name, state))          # before the checks, and before anything is written
    checks = preflight.run_checks(spec, opts, plan=plan, level="submit", max_su=args.max_su, dry_run=args.dry_run)
    print("\n" + preflight.render(checks, mode="dry-run" if args.dry_run else "submit"))
    if preflight.has_errors(checks) and not args.dry_run:
        raise CliError(_refusal(checks))
    if blocked and args.dry_run:
        print(f"\ndry run: the campaign is {state.state}, so nothing was written")
        return 0
    if state is not None and not blocked and args.dry_run:         # a real submit says it itself, with the reason, once the job queue is read
        print(f"\nthe earlier submission of this campaign ended as {state.state}; a real submit queues it again and keeps its finished stages")
    scripts = write_campaign(plan, force=args.force)               # with --force the recipe files must be unchanged
    print(f"\nscripts written to {plan.campaign_dir / 'scripts'}")
    if args.dry_run:
        print("dry run: nothing submitted")
        return 0
    if not args.yes:
        if not sys.stdin.isatty():
            print("refusing to submit without --yes in a non-interactive shell", file=sys.stderr)
            return 2
        if input(f"\nSubmit {len(scripts)} job(s), up to {plan.total_su:,.0f} SU? [y/N] ").strip().lower() != "y":
            print(f"not submitted (the scripts stay in {plan.campaign_dir / 'scripts'})")
            return 1
    submit(plan, scripts, force=args.force)                        # prints what was queued and the follow-up commands
    if _ref(plan.campaign_dir, opts.home) != plan.campaign_dir.name:
        print(f"those commands find the campaign through MPD_HOME; from another shell give its path instead: {plan.campaign_dir}")
    return 0


# ---------------------------------------------------------------- following a campaign
def _campaign_from(args) -> tuple[Path, Path | None]:
    home = _optional_home(_load_cluster(args.cluster, required=False), args.home)
    return Path(_campaigns().resolve_campaign(args.campaign, home)), home


def _field(states: dict, job: str, name: str) -> str:
    value = getattr(states.get(job), name, None)
    return str(value) if value not in (None, "") else "-"


def _list_campaigns(args, api) -> int:
    home = _optional_home(_load_cluster(args.cluster, required=False), args.home)
    if home is None:
        raise CliError(NO_HOME)
    found = api.list_campaigns(home)
    if not found:
        print(f"no campaigns under {home / 'runs'}")
        return 0
    width = max(len(c.campaign_id) for c in found)
    print(f"{'campaign':<{width}}  {'state':<12}  jobs  submitted")
    for c in found:
        print(f"{c.campaign_id:<{width}}  {c.state:<12}  {len(c.jobs):>4}  {c.submitted_at or '-'}")
    print("\nmpd status <campaign id or unique prefix>   shows one")
    return 0


BAD_STATES = (*FAILURE_STATES, "CANCELLED")


def _word(value) -> str:
    """A field of the status table as one word: runs of spaces and commas become '_', nothing becomes '-'."""
    return re.sub(r"[\s,]+", "_", str(value or "").strip().strip("()")).strip("_") or "-"


def _wrapped(text: str, indent: int = 2) -> str:
    width = max(70, min(shutil.get_terminal_size((110, 24)).columns, 120))
    return textwrap.fill(text, width=width, initial_indent=" " * indent, subsequent_indent=" " * (indent + 2),
                         break_long_words=False, break_on_hyphens=False)         # never split --sbatch-arg: people paste these


def _advice_block(said: list[str]) -> list[str]:
    """What happened at the margin, what to do under it."""
    return [_wrapped(said[0])] + [_wrapped(x, 4) for x in said[1:]] if said else []


def _next_steps(ref: str, state: str, failed: str | None, states: dict, jobs: dict, have_results: bool) -> list[str]:
    if state == "partial":
        return [f"mpd cancel {ref}    cancel the jobs that were queued", "repeat your mpd submit command with --force to continue the chain"]
    if state == "cancelled":
        return ["this campaign was cancelled: repeat your mpd submit command, with any option you want to change, to queue it again; stages that already finished are kept"]
    if state == "failed":
        return [f"this campaign ended as {state}: fix the cause (mpd logs {ref} <stage> --err), then repeat your mpd submit command"]
    if failed:
        return [f"mpd logs {ref} {failed.split('.')[0]} --err    the whole error log", "docs/TROUBLESHOOTING.md explains the usual failures",
                "repeat your mpd submit command with --force to continue: stages that are queued, running or finished are kept, the "
                "others are queued again, with any --train-time, --sbatch-arg or similar option you changed"]
    if have_results:
        return [f"mpd results {ref}"]
    live = [s for s, j in jobs.items() if _field(states, j, "state").split()[0] in ("PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "-")]
    if live:
        return [f"mpd status {ref}    again later", f"mpd logs {ref} {live[0].split('.')[0]}    output of a stage while it runs"]
    return [f"mpd results {ref}"]


def _status_rows(st, states: dict, reasons: dict, progress: dict) -> list[list[str]]:
    """One row per stage: stage, state, elapsed, reason, then job, exit, finished tasks and a note. The first four are one word each."""
    rows = []
    for stage, job in st.jobs.items():
        js = states.get(job)
        base = stage.split(".")[0]
        done = progress.get(base) if stage == base else None
        state = js.state.split()[0] if js and js.state else ""
        reason = "" if state in ("RUNNING", "COMPLETING", "CONFIGURING") else reasons.get(job)    # an array's waiting tasks have reasons of their own
        rows.append([stage, _word(state), _word(js.elapsed if js else ""), _word(reason), job, _word(js.exitcode if js else ""),
                     f"{done[0]}/{done[1]}" if done else "-", js.detail if js and js.detail else ""])
    return rows


def _problems(path: Path, st, rows: list[list[str]], states: dict, reasons: dict, plans: dict, ref: str, api) -> tuple[list[str], str | None]:
    """Plain words and exact commands for every stage that failed or was cancelled; the stage a student should look at first."""
    lines: list[str] = []
    first = None
    if st.state == "cancelled":                              # `mpd cancel` stopped it: one sentence for all the stages it cancelled
        stopped = [r[0] for r in rows if r[1] == "CANCELLED"]
        lines += _advice_block(api.advise_cancelled(stopped) if stopped else [])
    for stage, state, _elapsed, _reason, job, *_ in rows:
        if state not in BAD_STATES or (state == "CANCELLED" and st.state == "cancelled"):
            continue
        js, reason = states.get(job), reasons.get(job, "")
        err, out = api.log_tail(path, stage, 200, err=True, job=job), api.log_tail(path, stage, 200, err=False, job=job)
        said = api.advise(stage, job, js, ref=ref, reason=reason, err=err, out=out, plan=plans.get(stage), cluster=st.cluster,
                          cancelled=st.state == "cancelled")
        if state == "CANCELLED" and first:
            said = said[:1]                                  # a stage cancelled after another one failed: that one is the story
        lines += _advice_block(said)
        if first is None and state in FAILURE_STATES:
            first = stage
            tail = api.failure_tail(path, stage, 20, job=job)
            lines += [f"  last lines of the error log of {stage} (job {job}):", textwrap.indent(tail or "(empty)", "    ")]
    return lines, first


def _waiting(rows: list[list[str]], reasons: dict, plans: dict, ref: str, api) -> list[str]:
    """One line for every stage that has not started and Slurm's reason, in words."""
    lines = []
    ended_badly = {r[0] for r in rows if r[1] in BAD_STATES}
    for stage, state, _elapsed, reason, job, *_ in rows:
        if state != "PENDING" or reason == "-":
            continue
        text = api.pending_help(reasons.get(job) or reason, ref)
        deps = (plans.get(stage) or {}).get("depends_on") or []
        if reason == "Dependency" and deps:
            stuck = [d for d in deps if d in ended_badly]
            text = (f"it waits for {', '.join(stuck)}, which did not end well (see problems above); it cannot start before that is fixed."
                    if stuck else f"it waits for {', '.join(deps)} to finish. Nothing to do.")
        lines.append(_wrapped(f"{stage:<11} {reason}: {text}"))
    return lines


def _show_campaign(path: Path, home: Path | None, api) -> int:
    st = api.campaign_state(path)
    if st is None:
        print(f"{path} has no SUBMITTED.json: it was planned or dry-run but never submitted.\n"
              "Submit it with the same mpd submit command.", file=sys.stderr)
        return 1
    ids = list(dict.fromkeys(st.jobs.values()))
    states = api.job_states(ids) if ids else {}
    reasons = api.pending_reasons(ids, any_state=True) if ids else {}
    progress, plans, ref = api.stage_progress(path), api.stage_plan(path), _ref(path, home)
    print(f"campaign   {st.campaign_id}\ndirectory  {path}\nstate      {st.state}"
          + (f"   (submitted {st.submitted_at})" if st.submitted_at else ""))
    if st.message:
        print(f"           {st.message}")
    replaced = (api.read_marker(path) or {}).get("replaced") or {}
    if replaced:
        print("earlier    " + "; ".join(f"{s} {', '.join(map(str, j))}" for s, j in replaced.items())
              + "   (jobs that --force replaced; their logs stay in logs/)")
    if ids and not states:
        print("slurm      no job states: sacct and squeue are not installed here, the controller did not answer, or the jobs are too new")
    rows = _status_rows(st, states, reasons, progress)
    header = ["stage", "state", "elapsed", "reason", "job", "exit", "tasks", "note"]
    widths = [max([len(h), *(len(r[i]) for r in rows)]) for i, h in enumerate(header)]
    print()
    for row in [header, *rows]:
        print("  ".join(cell.ljust(w) for cell, w in zip(row, widths)).rstrip())
    problems, failed = _problems(path, st, rows, states, reasons, plans, ref, api)
    if problems:
        print("\nproblems:")
        print("\n".join(problems))
    waiting = _waiting(rows, reasons, plans, ref, api)
    if waiting:
        print("\nwaiting:")
        print("\n".join(waiting))
    print("\nnext:")
    for line in _next_steps(ref, st.state, failed, states, st.jobs, (path / "evaluate" / "summary.json").is_file()):
        print("  " + line)
    return 0


def cmd_status(args) -> int:
    api = _campaigns()
    if not args.campaign:
        return _list_campaigns(args, api)
    path, home = _campaign_from(args)
    return _show_campaign(path, home, api)


_LOG_NAME = re.compile(r"^mpd-(?P<stage>.+)-[^-_]*_(?P<job>\d+)(?:_(?P<task>\d+))?\.(?P<ext>out|err)$")


def _log_files(path: Path) -> dict[str, list[tuple[int, int, str, Path]]]:
    """{stage: [(job id, task, ext, file)]} for the log files Slurm wrote into <campaign>/logs."""
    out: dict[str, list[tuple[int, int, str, Path]]] = {}
    for f in sorted((path / "logs").glob("mpd-*")):
        m = _LOG_NAME.match(f.name)
        if m:
            out.setdefault(m["stage"], []).append((int(m["job"]), int(m["task"] or 0), m["ext"], f))
    return out


def _size(n: int) -> str:
    return f"{n} B" if n < 1024 else f"{n / 1024:.1f} KB" if n < 1 << 20 else f"{n / (1 << 20):.1f} MB"


def cmd_logs(args) -> int:
    path, _ = _campaign_from(args)
    logs = _log_files(path)
    if not logs:
        print(f"no log files under {path / 'logs'} yet (the jobs have not started, or the campaign was not submitted)", file=sys.stderr)
        return 1
    if not args.stage:
        rows = []
        for stage, files in logs.items():
            job = max(f[0] for f in files)
            mine = [f for f in files if f[0] == job]
            sizes = {ext: sum(f[3].stat().st_size for f in mine if f[2] == ext) for ext in ("out", "err")}
            rows.append([stage, str(job), str(len({f[1] for f in mine})), _size(sizes["out"]), _size(sizes["err"])])
        print(f"newest job of each stage, in {path / 'logs'}")
        print("\n".join(table(["stage", "job", "tasks", ".out", ".err"], rows, left=(0,))))
        print("\nmpd logs <campaign> <stage> [--err] [--tail N]   shows one")
        return 0
    if args.stage not in logs:
        raise CliError(f"no logs for stage {args.stage!r}; stages with logs: {', '.join(logs)}")
    ext = "err" if args.err else "out"
    mine = sorted((f for f in logs[args.stage] if f[2] == ext), key=lambda f: (f[0], f[1]))
    if not mine:
        print(f"stage {args.stage} has no .{ext} file yet", file=sys.stderr)
        return 1
    job, task, _, target = mine[-1]
    print(f"==> {target} (job {job}" + (f", task {task}; {len(mine) - 1} other task log(s) in the same folder" if len(mine) > 1 else "") + ") <==")
    with open(target, errors="replace") as fh:
        sys.stdout.write("".join(deque(fh, maxlen=args.tail)))
    return 0


def cmd_cancel(args) -> int:
    path, _ = _campaign_from(args)
    _campaigns().cancel(path, echo=print)
    return 0


def cmd_results(args) -> int:
    path, _ = _campaign_from(args)
    text, found = results.render(path)
    print(text)
    return 0 if found else 1


# ---------------------------------------------------------------- data, models, clusters, assays
def _list_sources() -> int:
    for s in data_sources.SOURCES.values():
        if s.zenodo_record:
            where = f"zenodo.org/records/{s.zenodo_record} (doi:{s.doi})"
        elif s.hf_repo:
            where = f"huggingface.co/datasets/{s.hf_repo} at revision {s.hf_revision[:8]}, every file pinned by sha256"
        else:
            where = "not yet published (use --from-dir, or --zenodo-record once there is one)"
        print(f"{s.key:<17}{s.title}\n{'':<17}{where}  license {s.license}")
        if s.note:
            print(f"{'':<17}{s.note}")
    print(f"{'model KEY ...':<17}pinned model weights, so compute nodes can run offline: `mpd models`, `mpd fetch model --list`")
    for tool in getattr(data_sources, "TOOLS", {}).values():
        print(f"{tool.key:<17}{tool.title}\n{'':<17}{tool.url}, archive and binary pinned by sha256  license {tool.license}\n"
              f"{'':<17}{tool.note}")
    return 0


def _model_state(models, key: str, home: Path | None) -> str:
    if home is None:
        return "-"
    try:
        state = models.status(key, home).state
    except Exception as exc:                                   # one broken folder must not break the listing
        return f"error ({type(exc).__name__})"
    return "-" if state == "unpinned" else state


def cmd_models(args) -> int:
    home = _optional_home(_load_cluster(args.cluster, required=False), args.home)
    models = _models()
    pins = models.pins()
    width = max(60, shutil.get_terminal_size((100, 24)).columns)
    header = ["key", "family", "params", "roles", "status", "license", "download"] + (["weights"] if home else [])
    rows = []
    offered = [m for m in registry.MODELS.values() if args.all or not m.output_family_lock]       # a student sees what it can use
    for m in offered:
        pin = pins.get(m.key)
        rows.append([m.key, m.family, f"{m.params_m / 1000:g}B" if m.params_m >= 1000 else f"{m.params_m:g}M",
                     "+".join(m.roles), m.status, m.license or "-", models.human_bytes(pin.total_bytes) if pin else "-"]
                    + ([_model_state(models, m.key, home)] if home else []))
    lines = table(header, rows, left=(0, 1, 3, 4, 5, 7))
    print(lines[0])
    for line, m in zip(lines[1:], offered):
        print(line)
        for label, text in (("note", m.note), ("license terms", m.license_note)):
            if text:
                print(textwrap.fill(f"{label}: {text}", width=width, initial_indent=" " * 4, subsequent_indent=" " * 6))
    if not home:
        print("\nadd --cluster or --home (or set MPD_HOME) to see which weights are fetched")
    print("\nfetch weights on a login node: mpd fetch model KEY [KEY ...]    (compute nodes have no internet)")
    return 0


def cmd_clusters(_args) -> int:
    for name in available_clusters():
        c = Cluster.load(name)
        print(f"{name:<8}{c.description}")
        root = re.sub(r"\{env:(\w+)\}", r"$\1", c.storage_root)               # the profile's {env:SCRATCH} is $SCRATCH to a user
        print(f"         home {root}, login node: {c.login_max_cores} cores / {c.login_max_minutes} min")
        for g in c.gpus.values():
            extra = "" if getattr(g, "bf16", True) else "  no native bf16: training is refused (--allow-unsupported-gpu)"
            print(f"   {g.name:<5} {g.per_node}/node  {g.mem_gb:g} GB  partition={g.partition or 'auto'}{extra}")
    return 0


def cmd_assays(args) -> int:
    cluster = _load_cluster(args.cluster, required=False)
    home = _optional_home(cluster, args.home)
    if home is None:
        raise CliError(NO_HOME)
    shown_total = 0
    for dataset in [args.dataset] if args.dataset else list(DATASETS):
        check = preflight.check_dataset(dataset, preflight.Context(cluster, home))
        if check.level == "error":
            raise CliError(check.message)
        ids = preflight.assay_ids(dataset, home)
        shown = [a for a in ids if not args.grep or args.grep.lower() in a.lower()]
        shown_total += len(shown)
        print(f"{dataset}: {len(shown)} of {len(ids)} assay ids" + (f" containing {args.grep!r}" if args.grep else ""))
        for a in shown:
            print(f"  {a}")
    return 0 if shown_total else 1


def _list_models(args, models, home: Path | None) -> int:
    pins = models.pins()
    rows = []
    for key in args.keys or [k for k, m in registry.MODELS.items() if not m.output_family_lock]:
        info, pin = registry.get(key), pins.get(key)
        source = f"{info.hf_id} @ {pin.revision[:10]}" if pin else f"{info.hf_id} (never fetched)"
        rows.append([key, source, str(len(pin.files)) if pin else "-", models.human_bytes(pin.total_bytes) if pin else "-",
                     info.license or "-", _model_state(models, key, home)])
    print("\n".join(table(["key", "source @ pinned commit", "files", "size", "license", "state"], rows, left=(0, 1, 4, 5))))
    if home is None:
        print("\nadd --cluster or --home to see which are fetched")
    return 0


def _model_keys(args, models) -> list[str]:
    if args.all:
        if args.keys:
            raise CliError("give model keys or --all, not both")
        return models.fetchable_keys()
    if not args.keys:
        raise CliError("name the models to fetch, or use --all or --list. `mpd models` lists the keys, e.g. "
                       "`mpd fetch model esm2-3b esm2-650m --cluster grace`")
    unknown = [k for k in args.keys if k not in registry.MODELS]
    if unknown:
        raise CliError(f"unknown model {unknown[0]!r}; choose from: {', '.join(models.pinned_keys())}")
    return list(dict.fromkeys(args.keys))


def _fetch_models(args) -> int:
    models = _models()
    cluster = _load_cluster(args.cluster, required=False)
    models_dir = args.models_dir or os.environ.get("MPD_MODELS_DIR")
    try:
        home = _optional_home(cluster, args.home)
    except ClusterError:
        if not models_dir:
            raise
        home = None
    if args.list:
        return _list_models(args, models, home)
    keys = _model_keys(args, models)
    if home is None and not models_dir:
        raise SystemExit("error: give --home, or --cluster (or set MPD_HOME / MPD_CLUSTER / MPD_MODELS_DIR)")
    minutes = args.time_budget if args.time_budget is not None else (cluster.login_max_minutes - 5 if cluster else None)
    deadline = time.monotonic() + max(minutes, 1.0) * 60 if minutes and minutes > 0 else None
    saved = os.environ.get("MPD_MODELS_DIR")
    if args.models_dir:
        os.environ["MPD_MODELS_DIR"] = str(_absolute(args.models_dir))
    try:
        for key in keys:
            models.fetch_model(key, home, from_dir=args.from_dir, verify=args.verify, force=args.force,
                               dry_run=args.dry_run, deadline=deadline, log=print)
    except TimeBudgetReached as exc:
        print(exc, file=sys.stderr)
        return EXIT_BUDGET
    finally:
        if args.models_dir:
            if saved is None:
                os.environ.pop("MPD_MODELS_DIR", None)
            else:
                os.environ["MPD_MODELS_DIR"] = saved
    return 0


def _fetch_foldseek(args, home: Path) -> int:
    tools = _need("mpd.structure.tools", "fetching Foldseek")
    if args.from_dir:
        raise CliError("for foldseek use --from-file TARBALL (the downloaded foldseek-linux-avx2.tar.gz), not --from-dir")
    print(f"foldseek: {tools.fetch_foldseek(home, from_file=args.from_file, log=print)}")
    return 0


def cmd_fetch(args) -> int:
    if args.source == "model":
        return _fetch_models(args)
    if args.list or not args.source:
        return _list_sources()
    if args.keys:
        raise CliError(f"model keys ({', '.join(args.keys)}) only go with `mpd fetch model`; `mpd fetch {args.source}` takes none")
    if args.all:
        raise CliError("--all only goes with `mpd fetch model`")
    cluster = _load_cluster(args.cluster, required=False)
    if cluster is not None:
        home = _home(cluster, args.home)                       # raises the profile's own message when $SCRATCH is unset
    elif args.home or os.environ.get("MPD_HOME"):
        home = _absolute(args.home or os.environ["MPD_HOME"])
    else:
        raise SystemExit(f"error: {NO_HOME}")
    if args.source == "foldseek":
        return _fetch_foldseek(args, home)
    source = data_sources.get(args.source)
    fetch_source(source, home / "data" / source.dirname, from_dir=args.from_dir,
                 zenodo_record=args.zenodo_record, unpack=not args.no_unpack)
    return 0


# ---------------------------------------------------------------- doctor
def cmd_doctor(args) -> int:
    cluster = _load_cluster(args.cluster, required=False)
    if cluster is None and not (args.home or os.environ.get("MPD_HOME")):
        raise CliError("give --cluster (or set MPD_CLUSTER) or --home (or set MPD_HOME)")
    try:
        home = _home(cluster, args.home) if cluster is not None else _optional_home(None, args.home)
    except ClusterError as exc:                                # a bad or unknown home is a finding, not a usage error
        text = preflight.render([Check("home", "error", str(exc))], mode="doctor")
        print(preflight.scrub(text) if args.report else text)
        return 1
    if home is None:
        raise CliError(NO_HOME)
    ctx = preflight.Context(cluster, home, _python(None, home), os.environ.get("MPD_ACCOUNT") or None)
    checks = preflight.run_checks(None, ctx, level="doctor", deep=args.deep, verify=args.verify)
    print(preflight.report(checks, ctx) if args.report else preflight.render(checks, mode="doctor"))
    return 1 if preflight.has_errors(checks) else 0


# ---------------------------------------------------------------- entry point
def _expected(exc: Exception) -> bool:
    """Errors with a message written for the user: our own exception classes, bad values, and file system trouble."""
    return isinstance(exc, (ValueError, OSError)) or type(exc).__module__.startswith("mpd")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        status = args.fn(args)
        sys.stdout.flush()                                         # a broken pipe must show up here, not in the interpreter's exit
        return status
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except BrokenPipeError:                                        # `mpd assays | head`: the reader left, which is not an error
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0
    except Exception as exc:
        if getattr(args, "debug", False):
            raise
        print(f"error: {exc}", file=sys.stderr)
        if not _expected(exc):
            print(f"({type(exc).__name__} was not expected; run again with --debug for the traceback and please report it)",
                  file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
