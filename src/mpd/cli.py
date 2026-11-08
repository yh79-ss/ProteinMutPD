"""`mpd` command line. Planning and submission only: no torch import, safe on a login node."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from . import __version__, registry
from .constants import DATASETS, METHODS
from .data import sources as data_sources
from .data.fetch import FetchError, fetch_source
from .slurm.cluster import Cluster, ClusterError, available_clusters
from .slurm.plan import PlanError, PlanOptions, default_python, describe, make_plan
from .slurm.submit import SubmitError, submit, write_campaign
from .spec import RunSpec, SpecError


def _common(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("what to distill")
    g.add_argument("--teacher", required=True, help="teacher model key (see `mpd models`)")
    g.add_argument("--student", required=True, help="student model key (see `mpd models`)")
    g.add_argument("--train-data", required=True, choices=DATASETS, help="benchmark whose sequences/positions the student is trained on")
    g.add_argument("--method", required=True, choices=list(METHODS),
                   help="direct-all | direct-one | score-all | score-one  (objective x training scope)")
    g.add_argument("--assay", nargs="+", default=[], metavar="ASSAY", help="training assay id(s); required for *-one methods")
    g.add_argument("--assays-file", type=Path, help="file with one training assay id per line")
    g.add_argument("--eval-data", nargs="+", required=True, choices=DATASETS, help="benchmark(s) to evaluate on")
    g.add_argument("--teacher-backend", choices=["cache", "live"], help="default: cache if the teacher supports it, else live")
    g.add_argument("--seed", type=int, default=RunSpec.seed)

    r = p.add_argument_group("recipe overrides (defaults reproduce the paper runs)")
    r.add_argument("--updates", type=int)
    r.add_argument("--global-batch", type=int)
    r.add_argument("--lr", type=float, dest="peak_lr")
    r.add_argument("--kl-weight", type=float)
    r.add_argument("--dev-fraction", type=float)

    c = p.add_argument_group("cluster")
    c.add_argument("--cluster", default=os.environ.get("MPD_CLUSTER"), help=f"profile ({', '.join(available_clusters())}) or YAML path")
    c.add_argument("--account", help="Slurm/HPRC billing account (Grace: a 12-digit project number)")
    c.add_argument("--gpu", help="GPU type for GPU stages (default from the cluster profile)")
    c.add_argument("--train-gpus", type=int, help="data-parallel GPUs for *-all training (one node)")
    c.add_argument("--home", help="working root for data, models and runs (default: cluster profile / $MPD_HOME)")
    c.add_argument("--python", help="interpreter on compute nodes (default: $MPD_PYTHON or <home>/env/bin/python)")
    c.add_argument("--train-time", help="override the estimated training walltime, e.g. 08:00:00")
    c.add_argument("--score-time", help="walltime per scoring task (default 02:00:00)")
    c.add_argument("--teacher-time", help="walltime for live teacher inference (default 06:00:00)")
    c.add_argument("--array-throttle", type=int, help="max concurrent array tasks")
    c.add_argument("--score-shards", help="positions split per dataset, e.g. proteingym=4,venusmuthub=1")


def _build(args) -> tuple[RunSpec, PlanOptions]:
    if not args.cluster:
        raise SystemExit("error: --cluster is required (or set MPD_CLUSTER). Available: " + ", ".join(available_clusters()))
    assays = list(args.assay)
    if args.assays_file:
        assays += [ln.strip() for ln in args.assays_file.read_text().splitlines() if ln.strip() and not ln.startswith("#")]
    spec = RunSpec(teacher=args.teacher, student=args.student, train_data=args.train_data, method=args.method,
                   eval_data=tuple(args.eval_data), assays=tuple(assays), teacher_backend=args.teacher_backend, seed=args.seed,
                   updates=args.updates, global_batch=args.global_batch, peak_lr=args.peak_lr,
                   kl_weight=args.kl_weight, dev_fraction=args.dev_fraction, train_gpus=args.train_gpus)
    cluster = Cluster.load(args.cluster)
    home = cluster.storage_dir(args.home or os.environ.get("MPD_HOME"))
    shards = None
    if args.score_shards:
        shards = {k: int(v) for k, v in (kv.split("=") for kv in args.score_shards.split(","))}
    opts = PlanOptions(cluster=cluster, home=home, python=args.python or default_python(home), gpu=args.gpu,
                       account=args.account, train_time=args.train_time, score_time=args.score_time,
                       teacher_time=args.teacher_time, array_throttle=args.array_throttle, score_shards=shards)
    return spec, opts


def cmd_plan(args) -> int:
    spec, opts = _build(args)
    print(describe(make_plan(spec, opts)))
    return 0


def cmd_submit(args) -> int:
    spec, opts = _build(args)
    plan = make_plan(spec, opts)
    print(describe(plan))
    scripts = write_campaign(plan)
    print(f"\nscripts written to {plan.campaign_dir / 'scripts'}")
    if args.dry_run:
        print("dry run: nothing submitted")
        return 0
    if not args.yes:
        if not sys.stdin.isatty():
            print("refusing to submit without --yes in a non-interactive shell", file=sys.stderr)
            return 2
        if input(f"\nSubmit {len(scripts)} job(s), up to {plan.total_su:,.0f} SU? [y/N] ").strip().lower() != "y":
            print("not submitted")
            return 1
    submit(plan, scripts, force=args.force)
    print(f"\nfollow with: mpd status {plan.campaign_dir}")
    return 0


def cmd_status(args) -> int:
    root = Path(args.campaign)
    marker = root / "SUBMITTED.json"
    if not marker.is_file():
        print(f"{root} has no SUBMITTED.json (not submitted yet?)", file=sys.stderr)
        return 1
    info = json.loads(marker.read_text())
    ids = ",".join(info["jobs"].values())
    print(f"campaign {info['campaign_id']}  submitted {info['submitted_at']}")
    fmt = "JobID%-18,JobName%-26,State%-12,Elapsed,ExitCode"
    out = subprocess.run(["sacct", "-X", "-n", "-j", ids, f"--format={fmt}"], capture_output=True, text=True)
    print(out.stdout.rstrip() or out.stderr.rstrip() or "(no accounting records yet)")
    for name, pattern in (("teacher", "COMPLETE.json"), ("train", "COMPLETE.json"), ("score", "shard_*.json"),
                          ("evaluate", "summary.json")):
        d = root / name
        if d.is_dir():
            print(f"{name:<9} finished: {len(list(d.rglob(pattern)))}")
    return 0


def cmd_fetch(args) -> int:
    if args.list or not args.source:
        for s in data_sources.SOURCES.values():
            where = f"zenodo.org/records/{s.zenodo_record} (doi:{s.doi})" if s.zenodo_record else "not yet published (use --from-dir)"
            print(f"{s.key:<17}{s.title}\n{'':<17}{where}  license {s.license}")
            if s.note:
                print(f"{'':<17}{s.note}")
        return 0
    home = args.home or os.environ.get("MPD_HOME")
    if args.cluster:
        home = Cluster.load(args.cluster).storage_dir(home)
    elif home:
        home = Path(home)
    else:
        raise SystemExit("error: give --home, or --cluster (or set MPD_HOME / MPD_CLUSTER)")
    source = data_sources.get(args.source)
    fetch_source(source, Path(home) / "data" / source.dirname, from_dir=args.from_dir,
                 zenodo_record=args.zenodo_record, unpack=not args.no_unpack)
    return 0


def cmd_models(_args) -> int:
    print(f"{'key':<14}{'family':<8}{'params':>8}  {'roles':<17}{'status':<10}notes")
    for m in registry.MODELS.values():
        p = f"{m.params_m / 1000:g}B" if m.params_m >= 1000 else f"{m.params_m:g}M"
        print(f"{m.key:<14}{m.family:<8}{p:>8}  {'+'.join(m.roles):<17}{m.status:<10}{m.note[:70]}")
        if m.license_note:
            print(f"{'':<14}license: {m.license_note}")
    return 0


def cmd_clusters(_args) -> int:
    for name in available_clusters():
        c = Cluster.load(name)
        print(f"{name:<8}{c.description}")
        for g in c.gpus.values():
            print(f"   {g.name:<5} {g.per_node}/node  {g.mem_gb:g} GB  partition={g.partition or 'auto'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mpd", description=__doc__)
    parser.add_argument("--version", action="version", version=f"mpd {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="show stages, resources and the SU estimate; writes nothing")
    _common(p)
    p.set_defaults(fn=cmd_plan)
    p = sub.add_parser("submit", help="write scripts and submit them with dependencies")
    _common(p)
    p.add_argument("--dry-run", action="store_true", help="write scripts but do not call sbatch")
    p.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    p.add_argument("--force", action="store_true", help="submit again even if SUBMITTED.json exists")
    p.set_defaults(fn=cmd_submit)
    p = sub.add_parser("status", help="show Slurm state and stage markers for a campaign directory")
    p.add_argument("campaign")
    p.set_defaults(fn=cmd_status)
    p = sub.add_parser("fetch", help="download pinned benchmark data (checksum-verified); run on a login node")
    p.add_argument("source", nargs="?", choices=list(data_sources.SOURCES))
    p.add_argument("--list", action="store_true", help="show the pinned sources")
    p.add_argument("--cluster", default=os.environ.get("MPD_CLUSTER"), help="cluster profile; chooses the storage root")
    p.add_argument("--home", help="working root (default: cluster profile / $MPD_HOME)")
    p.add_argument("--from-dir", type=Path, help="copy the files from a local folder instead of downloading (offline clusters)")
    p.add_argument("--zenodo-record", help="Zenodo record id, for a source without a pinned record")
    p.add_argument("--no-unpack", action="store_true", help="download and verify only")
    p.set_defaults(fn=cmd_fetch)
    sub.add_parser("models", help="list models and their roles").set_defaults(fn=cmd_models)
    sub.add_parser("clusters", help="list cluster profiles").set_defaults(fn=cmd_clusters)
    args = parser.parse_args(argv)
    try:
        return args.fn(args)
    except (SpecError, PlanError, ClusterError, SubmitError, FetchError, data_sources.SourceError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
