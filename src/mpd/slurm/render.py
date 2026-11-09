"""Render sbatch scripts for a Plan."""

from __future__ import annotations

import shlex

from .cluster import format_walltime
from .plan import (Plan, PlanError, Stage, check_slurm_path, job_env, validate_account, validate_module,
                   validate_sbatch_flag)


def _quote(value) -> str:
    return shlex.quote(str(value))


def _exports(env: dict[str, str]) -> list[str]:
    return [f"export {name}={_quote(value)}" for name, value in env.items()]


def render_script(plan: Plan, stage: Stage) -> str:
    """The sbatch script of one stage (CONTRACTS.md section 6). Everything user-supplied is validated again here, because a
    `#SBATCH` line is read verbatim by Slurm and the rest is run by bash."""
    opts, cluster = plan.options, plan.options.cluster
    if not plan.nonce or not plan.campaign_id or not opts.python:
        raise PlanError("this plan has no campaign id, nonce or Python interpreter; build it with make_plan")
    gpu = cluster.gpus[stage.gpu] if stage.gpu else None
    logs = plan.campaign_dir / "logs"             # created by write_campaign before sbatch: Slurm does not create it
    check_slurm_path(logs)
    short = plan.spec.student.replace("-", "")[:10]
    array = stage.is_array
    out = f"{logs}/%x_%A_%a.out" if array else f"{logs}/%x_%j.out"
    err = out[:-4] + ".err"

    head = ["#!/bin/bash",
            f"#SBATCH --job-name=mpd-{stage.full_name}-{short}",
            # `squeue -u $USER -o %k` finds this submission's jobs even when sbatch reported an error but queued the job
            f"#SBATCH --comment=mpd:{plan.campaign_id}:{plan.nonce}:{stage.full_name}",
            f"#SBATCH --time={format_walltime(stage.walltime_s)}",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            f"#SBATCH --cpus-per-task={stage.cpus}",
            f"#SBATCH --mem={stage.mem_gb}G"]
    if gpu is not None:
        head.append(f"#SBATCH --gres=gpu:{gpu.gres}:{stage.gpus}")     # always typed: an untyped request ran on a T4 node
        partition = gpu.partition
    else:
        partition = cluster.cpu_partition
        if cluster.cpu_gres:
            head.append(f"#SBATCH --gres={cluster.cpu_gres}")
    if partition:
        head.append(f"#SBATCH --partition={partition}")
    account = opts.account or cluster.account
    if account:
        head.append(f"#SBATCH --account={validate_account(account)}")
    if array:
        head.append(f"#SBATCH --array=0-{len(stage.tasks) - 1}%{opts.array_throttle or cluster.array_throttle}")
    head += [f"#SBATCH --output={out}", f"#SBATCH --error={err}", "#SBATCH --open-mode=append"]
    head += [f"#SBATCH {validate_sbatch_flag(flag, f'{cluster.name} profile sbatch_extra')}" for flag in cluster.sbatch_extra]
    head += [f"#SBATCH {validate_sbatch_flag(flag)}" for flag in opts.extra_sbatch]

    # The job inherits the environment of the shell that ran sbatch (Slurm's default), so drop what must not arrive: PYTHONPATH,
    # PYTHONHOME and every MPD_* variable (MPD_ALLOW_HUB, MPD_STRUCTURE_LEGACY_MASK, ...); the exports below put back the planned ones.
    # LD_LIBRARY_PATH and the like are left alone: the Grace job scripts that ran inherit them, and a module may add to them.
    # `set -u` comes after the module block: Lmod's init and /etc/bashrc read unset variables (Grace: "BASHRCSOURCED: unbound variable").
    body = ["", "set -eo pipefail"]
    modules = [validate_module(m) for m in plan.modules]
    if modules:
        names = " ".join(modules)
        no_module = f'echo "MPD_MODULE_LOAD_FAILED the module command is not available ({names})" >&2; exit 1'
        body += ["type module >/dev/null 2>&1 || source /etc/profile 2>/dev/null || true",
                 f"type module >/dev/null 2>&1 || {{ {no_module}; }}",
                 "module purge",
                 f'module load {names} || {{ echo "MPD_MODULE_LOAD_FAILED {names}" >&2; exit 1; }}']
    else:
        body.append("if type module >/dev/null 2>&1; then module purge; fi")
    body += ['for _v in $(compgen -e); do case $_v in MPD_*) unset "$_v";; esac; done; unset _v PYTHONPATH PYTHONHOME', "set -u"]

    cache = opts.home / "cache"
    planner_env, user_env = job_env(opts)
    body += [
        f"export MPD_HOME={_quote(opts.home)} MPD_CLUSTER={_quote(cluster.name)} HF_HOME={_quote(opts.home / 'hf_cache')} "
        f"XDG_CACHE_HOME={_quote(cache)} "
        f"TORCH_HOME={_quote(cache / 'torch')} TRITON_CACHE_DIR={_quote(cache / 'triton')} MPLCONFIGDIR={_quote(cache / 'mpl')}",
        # weights are staged by `mpd fetch` and compute nodes have no internet
        "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 "
        f"PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS={max(1, stage.cpus // max(1, stage.gpus or 1))}",
        *_exports(planner_env),
    ]
    if gpu is not None:
        body.append("export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MPD_REQUIRE_CUDA=1")
    body += _exports(user_env)                    # last, so --env can override anything above
    if gpu is not None:
        # a node whose GPU is broken lists it in nvidia-smi or not at all, and code that falls back to the CPU burns the allocation
        body.append('nvidia-smi -L || { echo "MPD_NO_GPU: no usable GPU on $(hostname -s); resubmit with '
                    '--sbatch-arg=--exclude=$(hostname -s)" >&2; exit 31; }')
    body += [f'echo "MPD_JOB_START host=$(hostname) job=${{SLURM_JOB_ID:-}} stage={stage.full_name}"',
             f"cd {_quote(plan.campaign_dir)}"]

    # always pass the index: a chunk of one task is not an array, and the task list holds every chunk's tasks
    index = f"$((SLURM_ARRAY_TASK_ID + {stage.task_offset}))" if array else str(stage.task_offset)
    args = f"--campaign-dir {_quote(plan.campaign_dir)} --stage {stage.name} --task-index {index}"
    py = _quote(opts.python)
    if stage.launcher == "ddp":
        # `python -m torch.distributed.run`, not bin/torchrun: relocated environments can keep a
        # stale interpreter in torchrun's shebang (this broke a real job).
        cmd = f"{py} -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node={stage.gpus} -m {stage.module} {args}"
    else:
        cmd = f"{py} -m {stage.module} {args}"
    body += ["", cmd, f'echo "MPD_STAGE_DONE {stage.full_name}"']
    return "\n".join(head + body) + "\n"
