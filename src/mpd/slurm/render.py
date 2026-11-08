"""Render sbatch scripts for a Plan."""

from __future__ import annotations

import shlex

from .cluster import format_walltime
from .plan import Plan, Stage


def _quote(path) -> str:
    return shlex.quote(str(path))


def render_script(plan: Plan, stage: Stage) -> str:
    opts, cluster = plan.options, plan.options.cluster
    gpu = cluster.gpus[stage.gpu] if stage.gpu else None
    logs = plan.campaign_dir / "logs"
    short = plan.spec.student.replace("-", "")[:10]
    job = f"mpd-{stage.full_name}-{short}"
    array = stage.is_array
    out = f"{logs}/%x_%A_%a.out" if array else f"{logs}/%x_%j.out"
    err = out[:-4] + ".err"

    head = ["#!/bin/bash",
            f"#SBATCH --job-name={job}",
            f"#SBATCH --time={format_walltime(stage.walltime_s)}",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            f"#SBATCH --cpus-per-task={stage.cpus}",
            f"#SBATCH --mem={stage.mem_gb}G"]
    if gpu is not None:
        head.append(f"#SBATCH --gres=gpu:{gpu.name}:{stage.gpus}")
        partition = gpu.partition
    else:
        partition = cluster.cpu_partition
    if partition:
        head.append(f"#SBATCH --partition={partition}")
    account = opts.account or cluster.account
    if account:
        head.append(f"#SBATCH --account={account}")
    if array:
        throttle = opts.array_throttle or cluster.array_throttle
        head.append(f"#SBATCH --array=0-{len(stage.tasks) - 1}%{throttle}")
    head += [f"#SBATCH --output={out}", f"#SBATCH --error={err}"]
    head += [f"#SBATCH {flag}" for flag in (*cluster.sbatch_extra, *opts.extra_sbatch)]

    body = ["", "set -euo pipefail"]
    if cluster.modules:
        body += ["module purge", "module load " + " ".join(cluster.modules)]
    body += [
        f"export MPD_HOME={_quote(opts.home)}",
        f"export HF_HOME={_quote(opts.home / 'hf_cache')}",
        "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1   # weights are staged by `mpd fetch`",
        "export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1",
        f"export OMP_NUM_THREADS={max(1, stage.cpus // max(1, stage.gpus or 1))}",
        f"cd {_quote(plan.campaign_dir)}",
    ]
    py = _quote(opts.python)
    args = f"--campaign-dir {_quote(plan.campaign_dir)} --stage {stage.name}"
    if array:
        args += f" --task-index $((SLURM_ARRAY_TASK_ID + {stage.task_offset}))"
    if stage.launcher == "ddp":
        # `python -m torch.distributed.run`, not bin/torchrun: relocated environments can keep a
        # stale interpreter in torchrun's shebang (this broke a real job).
        cmd = f"{py} -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node={stage.gpus} -m {stage.module} {args}"
    else:
        cmd = f"{py} -m {stage.module} {args}"
    body += ["", cmd, f'echo "MPD_STAGE_DONE {stage.full_name}"']
    return "\n".join(head + body) + "\n"
