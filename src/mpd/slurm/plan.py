"""Turn a RunSpec into a staged, policy-checked job plan. No torch, no network."""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .. import registry
from ..presets import H200_MINUTES_PER_1000_UPDATES, preset_for
from ..spec import RunSpec
from .cluster import Cluster, GpuType, format_walltime, parse_walltime

# Default scoring shards per evaluation dataset (positions are split across array tasks).
DEFAULT_SCORE_SHARDS = {"proteingym": 4, "venusmuthub": 1}
MAX_ARRAY_TASKS = 1000        # Slurm's default MaxArraySize is 1001 (indices 0-1000)
TIME_SLACK = 1.5
GPU_MEM_USABLE = 0.9          # leave headroom for CUDA context and fragmentation


class PlanError(ValueError):
    """The request cannot be planned. The message says how to fix it."""


@dataclass
class PlanOptions:
    cluster: Cluster
    home: Path
    python: str
    gpu: str | None = None
    account: str | None = None
    train_time: str | None = None
    score_time: str | None = None
    teacher_time: str | None = None
    array_throttle: int | None = None
    score_shards: dict | None = None
    extra_sbatch: tuple[str, ...] = ()


@dataclass
class Stage:
    name: str                      # base name, e.g. "train"
    part: int                      # array chunk index (0 for the first / only chunk)
    kind: str                      # cpu | gpu
    module: str                    # python module run as `-m`
    gpu: str | None
    gpus: int
    cpus: int
    mem_gb: int
    walltime_s: int
    tasks: list[dict]              # tasks in this chunk
    task_offset: int               # index of tasks[0] within the stage's full task list
    launcher: str                  # python | ddp
    depends_on: list[str]
    su_estimate: float
    notes: list[str] = field(default_factory=list)

    @property
    def full_name(self) -> str:
        return f"{self.name}.{self.part}" if self.part else self.name

    @property
    def is_array(self) -> bool:
        return len(self.tasks) > 1


@dataclass
class Plan:
    spec: RunSpec
    options: PlanOptions
    campaign_dir: Path
    stages: list[Stage]
    resolved: dict
    warnings: list[str]

    @property
    def total_su(self) -> float:
        return sum(s.su_estimate for s in self.stages)


def _choose_gpus(model: registry.ModelInfo, gpu: GpuType) -> int:
    need = model.weights_gb + model.act_gb
    usable = gpu.mem_gb * GPU_MEM_USABLE
    if need <= usable:
        return 1
    if model.shard_across_gpus:
        for n in range(2, gpu.per_node + 1):
            if need <= n * usable:
                return n
    raise PlanError(
        f"{model.key} needs about {need:.0f} GB (weights {model.weights_gb:.1f} + activations "
        f"{model.act_gb:.0f}, estimated) but {gpu.name} offers {usable:.0f} GB usable per GPU"
        + (f" and {gpu.per_node * usable:.0f} GB across a node" if model.shard_across_gpus else "")
        + ". Choose a larger GPU with --gpu or a smaller model.")


def _round_up_minutes(seconds: float, step: int = 15) -> int:
    minutes = max(step, math.ceil(seconds / 60 / step) * step)
    return minutes * 60


def make_plan(spec: RunSpec, options: PlanOptions) -> Plan:
    spec.validate()
    cluster = options.cluster
    gpu = cluster.gpu(options.gpu)
    teacher, student = registry.get(spec.teacher), registry.get(spec.student)
    warnings: list[str] = []
    errors: list[str] = []

    if options.account is None and cluster.account is None:
        warnings.append(f"no --account given: SUs go to your default {cluster.name} account "
                        "(on Grace, `myproject` lists accounts)")

    # ---- resolve the training recipe -----------------------------------
    preset = preset_for(student.family, spec.objective)
    updates = spec.updates or preset.updates
    global_batch = spec.global_batch or preset.global_batch
    micro = preset.microbatch

    train_gpus_model = _choose_gpus(student, gpu)
    if spec.scope == "all" and not student.shard_across_gpus:
        if spec.train_gpus:
            dp = max(1, min(spec.train_gpus, gpu.per_node))
        else:
            # Largest width up to 4 GPUs (the paper's setup) that splits the global batch evenly,
            # e.g. 2 of the 3 GPUs on an A40 node.
            dp = max(n for n in range(1, min(gpu.per_node, 4) + 1) if global_batch % (n * micro) == 0) \
                if any(global_batch % (n * micro) == 0 for n in range(1, min(gpu.per_node, 4) + 1)) else 1
    else:
        dp = 1
    per_step = dp * micro
    if global_batch % per_step:
        raise PlanError(f"global batch {global_batch} is not divisible by GPUs x microbatch = {dp} x {micro}; "
                        f"pick --global-batch as a multiple of {per_step} or change --train-gpus")
    accumulation = global_batch // per_step
    train_gpus = train_gpus_model if student.shard_across_gpus else dp

    resolved = {
        "campaign_id": spec.campaign_id(),
        "teacher": spec.teacher, "student": spec.student, "train_data": spec.train_data,
        "objective": spec.objective, "scope": spec.scope, "assays": list(spec.assays),
        "eval_data": list(spec.eval_data), "teacher_backend": spec.backend, "seed": spec.seed,
        "updates": updates, "global_batch": global_batch, "microbatch": micro,
        "accumulation": accumulation, "world_size": dp, "peak_lr": spec.peak_lr or preset.peak_lr,
        "kl_weight": preset.kl_weight if spec.kl_weight is None else spec.kl_weight,
        "support": preset.support, "lora_rank": preset.lora_rank, "lora_alpha": preset.lora_alpha,
        "dev_fraction": spec.dev_fraction or preset.dev_fraction, "warmup_fraction": preset.warmup_fraction,
        "min_lr_ratio": preset.min_lr_ratio, "grad_clip": preset.grad_clip, "weight_decay": preset.weight_decay,
        "gpu": gpu.name, "cluster": cluster.name,
    }
    if not student.act_gb_measured:
        warnings.append(f"{student.key}: GPU memory need ({student.weights_gb + student.act_gb:.0f} GB) is an "
                        "estimate, not a measurement; the first job may need more GPUs or a larger GPU")

    # ---- stage construction --------------------------------------------
    stages: list[Stage] = []
    campaign_dir = options.home / "runs" / spec.campaign_id()
    max_array = MAX_ARRAY_TASKS

    def gpu_stage(name, module, tasks, n_gpus, walltime_s, launcher, depends, note=None):
        cpus, mem = cluster.cpus_and_mem(gpu, n_gpus)
        for level, msg in cluster.check_gpu_job(gpu, n_gpus, walltime_s):
            (errors.append(f"[{name}] {msg}") if level == "error" else warnings.append(msg))
        parts = [tasks[i:i + max_array] for i in range(0, len(tasks), max_array)] or [[]]
        names = []
        for k, chunk in enumerate(parts):
            su = cluster.estimate_su(gpu=gpu, n_gpus=n_gpus, cpus=cpus, mem_gb=mem,
                                     walltime_s=walltime_s, tasks=len(chunk))
            stage = Stage(name=name, part=k, kind="gpu", module=module, gpu=gpu.name, gpus=n_gpus, cpus=cpus,
                          mem_gb=mem, walltime_s=walltime_s, tasks=chunk, task_offset=k * max_array,
                          launcher=launcher, depends_on=list(depends), su_estimate=su,
                          notes=[note] if note else [])
            stages.append(stage)
            names.append(stage.full_name)
        return names

    def cpu_stage(name, module, cpus, mem_gb, walltime_s, depends):
        for level, msg in cluster.check_cpu_job(walltime_s, cpus):
            (errors.append(f"[{name}] {msg}") if level == "error" else warnings.append(msg))
        su = cluster.estimate_su(gpu=None, n_gpus=0, cpus=cpus, mem_gb=mem_gb, walltime_s=walltime_s)
        stage = Stage(name=name, part=0, kind="cpu", module=module, gpu=None, gpus=0, cpus=cpus, mem_gb=mem_gb,
                      walltime_s=walltime_s, tasks=[{}], task_offset=0, launcher="python",
                      depends_on=list(depends), su_estimate=su)
        stages.append(stage)
        return [stage.full_name]

    prepare = cpu_stage("prepare", "mpd.stages.prepare", cpus=8, mem_gb=32, walltime_s=4 * 3600, depends=[])
    upstream = list(prepare)

    if spec.backend == "live":
        t_gpus = _choose_gpus(teacher, gpu)
        t_time = parse_walltime(options.teacher_time or "06:00:00")
        upstream = gpu_stage("teacher", "mpd.stages.teacher", [{}], t_gpus, t_time, "python", prepare,
                             note="computes the AA20 logit targets the student will imitate")

    # training
    unit_tasks = [{"unit": u} for u in spec.units]
    if options.train_time:
        train_s = parse_walltime(options.train_time)
    else:
        minutes = H200_MINUTES_PER_1000_UPDATES * updates / 1000.0 / max(1, dp) * gpu.time_factor * TIME_SLACK
        train_s = _round_up_minutes(minutes * 60)
        warnings.append(f"train walltime {format_walltime(train_s)} is estimated from H200 timings x "
                        f"{gpu.time_factor:g} (assumed {gpu.name} slowdown) x {TIME_SLACK}; override with --train-time")
    train_names = gpu_stage("train", "mpd.stages.train", unit_tasks, train_gpus, train_s,
                            "ddp" if dp > 1 else "python", upstream)

    # scoring: the frozen base student runs as soon as inputs exist; adapters wait for training
    shards = {**DEFAULT_SCORE_SHARDS, **(options.score_shards or {})}
    score_s = parse_walltime(options.score_time or "02:00:00")
    score_gpus = _choose_gpus(student, gpu)

    def score_tasks(units):
        return [{"unit": u, "dataset": ds, "shard": s, "num_shards": shards[ds]}
                for u in units for ds in spec.eval_data for s in range(shards[ds])]

    base_names = gpu_stage("score-base", "mpd.stages.score", score_tasks(["base"]), score_gpus, score_s,
                           "python", prepare, note="frozen student, no adapter (reference for deltas)")
    score_names = gpu_stage("score", "mpd.stages.score", score_tasks(list(spec.units)), score_gpus, score_s,
                            "python", train_names)
    cpu_stage("evaluate", "mpd.stages.evaluate", cpus=4, mem_gb=16, walltime_s=3600,
              depends=base_names + score_names)

    if errors:
        raise PlanError("\n".join(errors))
    if spec.scope == "one" and len(spec.assays) > 20:
        warnings.append(f"{len(spec.assays)} one-assay trainings will be submitted as arrays "
                        f"(throttle {options.array_throttle or cluster.array_throttle} at a time)")
    return Plan(spec=spec, options=options, campaign_dir=campaign_dir, stages=stages, resolved=resolved,
                warnings=list(dict.fromkeys(warnings)))


def default_python(home: Path) -> str:
    return os.environ.get("MPD_PYTHON") or str(home / "env" / "bin" / "python")


def describe(plan: Plan) -> str:
    c = plan.options.cluster
    lines = [f"campaign   {plan.spec.campaign_id()}",
             f"cluster    {c.name}  gpu={plan.resolved['gpu']}  home={plan.options.home}",
             f"teacher    {plan.spec.teacher} ({plan.spec.backend})   student {plan.spec.student}",
             f"train      {plan.spec.train_data}  method={plan.spec.method}  units={len(plan.spec.units)}",
             f"eval       {', '.join(plan.spec.eval_data)}",
             f"recipe     {plan.resolved['updates']} updates, global batch {plan.resolved['global_batch']} "
             f"(= {plan.resolved['world_size']} GPU x {plan.resolved['microbatch']} micro x "
             f"{plan.resolved['accumulation']} accum), lr {plan.resolved['peak_lr']}, "
             f"kl {plan.resolved['kl_weight']}, support {plan.resolved['support']}", "",
             f"{'stage':<14}{'tasks':>6}  {'gpu':<9}{'cpus':>5}{'mem':>6}  {'walltime':<11}{'SU (max)':>10}  after"]
    for s in plan.stages:
        gpu = f"{s.gpus}x{s.gpu}" if s.gpus else "-"
        lines.append(f"{s.full_name:<14}{len(s.tasks):>6}  {gpu:<9}{s.cpus:>5}{s.mem_gb:>5}G  "
                     f"{format_walltime(s.walltime_s):<11}{s.su_estimate:>10,.0f}  {', '.join(s.depends_on) or '-'}")
    lines += ["", f"estimated upper bound: {plan.total_su:,.0f} SU (requested walltime; actual charge follows runtime)"]
    for w in plan.warnings:
        lines.append(f"warning: {w}")
    return "\n".join(lines)
