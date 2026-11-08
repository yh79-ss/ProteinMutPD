"""Write a campaign directory and submit its stages with Slurm dependencies."""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from .plan import Plan, Stage
from .render import render_script


class SubmitError(RuntimeError):
    pass


def write_campaign(plan: Plan) -> dict[str, Path]:
    """Create the campaign directory with spec, resolved recipe, task lists and sbatch scripts."""
    root = plan.campaign_dir
    for sub in ("scripts", "tasks", "logs"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    (root / "spec.json").write_text(plan.spec.to_json() + "\n", encoding="utf-8")
    (root / "resolved.json").write_text(json.dumps(plan.resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    all_tasks: dict[str, list[dict]] = {}
    for stage in plan.stages:
        all_tasks.setdefault(stage.name, []).extend(stage.tasks)
    for name, tasks in all_tasks.items():
        (root / "tasks" / f"{name}.json").write_text(json.dumps(tasks, indent=1) + "\n", encoding="utf-8")

    scripts = {}
    for stage in plan.stages:
        path = root / "scripts" / f"{stage.full_name}.sbatch"
        path.write_text(render_script(plan, stage), encoding="utf-8")
        scripts[stage.full_name] = path
    summary = {"stages": [{**{k: v for k, v in asdict(s).items() if k != "tasks"}, "n_tasks": len(s.tasks)}
                          for s in plan.stages], "estimated_su_upper_bound": plan.total_su,
               "warnings": plan.warnings}
    (root / "plan.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return scripts


def submit(plan: Plan, scripts: dict[str, Path], *, sbatch: str = "sbatch", force: bool = False,
           echo=print) -> dict[str, str]:
    """Submit stages in order. On any failure the jobs already queued for this campaign are cancelled."""
    marker = plan.campaign_dir / "SUBMITTED.json"
    if marker.exists() and not force:
        raise SubmitError(f"{marker} exists: this campaign was already submitted. Use `mpd status` to follow it, "
                          "or --force to submit a new copy.")
    job_ids: dict[str, str] = {}
    by_name = {s.full_name: s for s in plan.stages}
    try:
        for stage in plan.stages:
            cmd = [sbatch, "--parsable"]
            deps = [job_ids[d] for d in stage.depends_on]
            if deps:
                cmd += ["--dependency=afterok:" + ":".join(deps), "--kill-on-invalid-dep=yes"]
            cmd.append(str(scripts[stage.full_name]))
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.stderr.strip():
                # Grace's job_submit plugin prints the SU charge here; show it to the user.
                echo(proc.stderr.strip())
            if proc.returncode != 0:
                raise SubmitError(f"sbatch rejected stage {stage.full_name}: {proc.stderr.strip() or proc.stdout.strip()}")
            job_ids[stage.full_name] = proc.stdout.strip().split(";")[0]
            echo(f"submitted {stage.full_name:<12} job {job_ids[stage.full_name]}")
    except Exception:
        if job_ids:
            subprocess.run(["scancel", *job_ids.values()], capture_output=True)
            echo(f"cancelled already-queued jobs: {', '.join(job_ids.values())}")
        raise
    marker.write_text(json.dumps({"submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "jobs": job_ids,
                                  "campaign_id": plan.spec.campaign_id()}, indent=2) + "\n", encoding="utf-8")
    return job_ids
