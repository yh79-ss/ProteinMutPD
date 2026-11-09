"""Turn a RunSpec into a staged, policy-checked job plan. No torch, no network."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
from dataclasses import dataclass, field, replace
from pathlib import Path

from .. import envrecord, registry
from ..constants import DATASETS
from ..presets import (BENCHMARK_SIZES, H200_FORWARD_TOKENS_PER_S, H200_TRAIN_MIN_PER_1000, MEASURED_STUDENTS,
                       REFERENCE_PARAMS_M, SIZE_EXPONENT, preset_for)
from ..spec import RunSpec
from .cluster import (ACCOUNT_RE, MODULE_RE, SBATCH_FLAG_RE, Cluster, ClusterError, GpuType, format_walltime,
                      parse_walltime)

# Default scoring shards per evaluation dataset (positions are split across array tasks).
DEFAULT_SCORE_SHARDS = {"proteingym": 4, "venusmuthub": 1}
MAX_SCORE_SHARDS = 1000
MAX_ARRAY_TASKS = 1000        # Slurm's default MaxArraySize is 1001 (indices 0-1000)
TIME_SLACK = 1.5
STARTUP_S = 600               # imports, CUDA init and model load before the first batch (a cold `import torch` took 3 min on Grace)
PREPARE_S, EVALUATE_S = 4 * 3600, 3600
GPU_MEM_USABLE = 0.85         # keep 15% free: padded batch widths vary, which fragments the allocator
FIXED_GB = 2.5                # CUDA context, NCCL buffers, LoRA optimizer state, backward workspace (an estimate)
ACT_REFERENCE_MICROBATCH = 2  # ModelInfo.act_gb is for micro-batch 2 and scales linearly with it
INFERENCE_ACT_FRACTION = 0.2  # no_grad batch-8 activations as a share of act_gb: about 5 GB for 3B, 2 GB for 650M (gpu_path_audit M1)

# What the job scripts export from the submitting shell (values are read when the scripts are rendered), besides MPD_MODEL_<KEY>.
FORWARDED_ENV = ("MPD_MODELS_DIR", "MPD_FOLDSEEK", "MPD_3DI_ENGINE", "MPD_DTYPE", "MPD_KEEP_ALL_CHECKPOINTS", "MPD_TF32",
                 "MPD_CKPT_EVERY", "MPD_STRUCTURE_LEGACY_MASK")
FORWARDED_PREFIX = "MPD_MODEL_"
_DIR_ENV = ("MPD_MODELS_DIR",)                 # always a path; MPD_MODEL_<KEY> too. MPD_FOLDSEEK may be a bare command name
RESERVED_ENV = {"MPD_ALLOW_HUB": "it is for login nodes and laptops; jobs run offline (HF_HUB_OFFLINE=1)"}
_NUMERICS_ENV = ("MPD_DTYPE", "MPD_TF32", "MPD_STRUCTURE_LEGACY_MASK")      # change results, so they are part of a campaign's identity

_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SAFE_ID = re.compile(r"[A-Za-z0-9_.+-]+")
_SAFE_PATH = re.compile(r"[A-Za-z0-9_.+@:,=/-]+")      # lands on `--output=` lines, which Slurm reads verbatim
_NONCE = re.compile(r"[A-Za-z0-9]{6,32}")
MAX_ENV_VALUE = 4096
# sbatch options mpd writes itself: a copy would be silently overridden, or break `mpd status/cancel/logs`
_OWNED_FLAGS = {"--job-name", "--comment", "--output", "--error", "--array", "--dependency", "--wrap", "--chdir", "--open-mode"}
_PLANNED_FLAGS = {"--time", "--nodes", "--ntasks", "--cpus-per-task", "--mem", "--gres", "--partition", "--account"}


class PlanError(ValueError):
    """The request cannot be planned. The message says how to fix it."""


# ---- option validators (the cli calls these so a typo never shows a traceback) -------------------------------------
def check_env(name: str, value: str) -> None:
    if not isinstance(name, str) or not _ENV_NAME.fullmatch(name):
        raise PlanError(f"{name!r} is not an environment variable name (letters, digits and '_', not starting with a digit)")
    if name in RESERVED_ENV:
        raise PlanError(f"{name} cannot be set for jobs: {RESERVED_ENV[name]}")
    if not isinstance(value, str) or len(value) > MAX_ENV_VALUE or any((ord(c) < 32 and c != "\t") or ord(c) == 127 for c in value):
        raise PlanError(f"the value of {name} must be one line of at most {MAX_ENV_VALUE} characters without control characters")


def parse_env_pair(text: str) -> tuple[str, str]:
    """``NAME=VALUE`` from --env."""
    name, sep, value = text.partition("=")
    if not sep:
        raise PlanError(f"--env expects NAME=VALUE, got {text!r}")
    check_env(name, value)
    return name, value


def check_score_shards(shards: dict) -> dict[str, int]:
    for key, n in shards.items():
        if key not in DATASETS:
            raise PlanError(f"--score-shards: {key!r} is not a dataset; use {' or '.join(DATASETS)}")
        if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= MAX_SCORE_SHARDS:
            raise PlanError(f"--score-shards: {key}={n!r} must be a whole number from 1 to {MAX_SCORE_SHARDS}")
    return dict(shards)


def parse_score_shards(text: str) -> dict[str, int]:
    """``proteingym=4,venusmuthub=1`` from --score-shards."""
    shards: dict[str, int] = {}
    for part in text.split(","):
        key, sep, number = (piece.strip() for piece in part.partition("="))
        if not sep or not key:
            raise PlanError(f"--score-shards expects DATASET=N pairs such as proteingym=4,venusmuthub=1, got {text!r}")
        if key in shards:
            raise PlanError(f"--score-shards lists {key} twice")
        try:
            shards[key] = int(number)
        except ValueError:
            raise PlanError(f"--score-shards: {number!r} (for {key}) is not a whole number") from None
    return check_score_shards(shards)


def validate_account(account: str, source: str = "--account") -> str:
    if not isinstance(account, str) or not ACCOUNT_RE.fullmatch(account):
        raise PlanError(f"{source} {account!r} may only contain letters, digits, '_', '.' and '-'")
    return account


def validate_module(name: str) -> str:
    if not isinstance(name, str) or not MODULE_RE.fullmatch(name):
        raise PlanError(f"--module {name!r} is not a module name (letters, digits, '_', '.', '+', '/', '-'; one module per --module)")
    return name


def validate_sbatch_flag(flag: str, source: str = "--sbatch-arg") -> str:
    """One long sbatch option, ``--name`` or ``--name=value``, without spaces: it becomes a `#SBATCH` line."""
    if not isinstance(flag, str) or not SBATCH_FLAG_RE.fullmatch(flag):
        raise PlanError(f"{source} {flag!r} must be one long option written --name or --name=value, without spaces "
                        "(for example --sbatch-arg=--exclude=g001)")
    name = flag.split("=", 1)[0]
    if name in _OWNED_FLAGS:
        raise PlanError(f"{source} {flag}: mpd sets {name} itself and relies on it; remove it")
    return flag


def _user_walltime(text: str, flag: str) -> int:
    try:
        return parse_walltime(text, strict=True)
    except ClusterError as exc:
        raise PlanError(f"{flag}: {exc}") from None


# ---- options and plan ------------------------------------------------------------------------------------------------
@dataclass
class PlanOptions:
    cluster: Cluster
    home: Path
    python: str | None = None      # None: decide (see resolve_python)
    gpu: str | None = None
    account: str | None = None
    train_time: str | None = None
    score_time: str | None = None
    teacher_time: str | None = None
    array_throttle: int | None = None
    score_shards: dict | None = None
    extra_sbatch: tuple[str, ...] = ()
    extra_env: dict[str, str] = field(default_factory=dict)
    extra_modules: tuple[str, ...] = ()
    nonce: str | None = None       # per-submission tag in `--comment`; None: a fresh random one
    allow_unsupported_gpu: bool = False


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
    walltime_basis: str = "fixed"  # fixed | user | measured (scaled from timed runs) | estimate (not measured)

    @property
    def full_name(self) -> str:
        return f"{self.name}.{self.part}" if self.part else self.name

    @property
    def is_array(self) -> bool:
        return len(self.tasks) > 1


@dataclass
class Plan:
    spec: RunSpec
    options: PlanOptions           # normalised: absolute home, resolved python and account
    campaign_dir: Path
    stages: list[Stage]
    resolved: dict
    warnings: list[str]
    nonce: str = ""                # per-submission; written into every job's --comment
    campaign_id: str = ""          # spec.campaign_id() + "__" + hash of the options that change tasks or scripts
    env_record: dict | None = None # <home>/env/mpd_env.json when it exists and fits this cluster and home

    @property
    def total_su(self) -> float:
        return sum(s.su_estimate for s in self.stages)

    @property
    def modules(self) -> tuple[str, ...]:
        """Modules the job scripts load: the profile's, then the environment record's, then --module."""
        return effective_modules(self.options.cluster, self.env_record, self.options.extra_modules)


def effective_modules(cluster: Cluster, record: dict | None, extra: tuple[str, ...] = ()) -> tuple[str, ...]:
    seen: list[str] = []
    for name in (*cluster.modules, *((record or {}).get("modules") or ()), *extra):
        if name not in seen:
            seen.append(name)
    return tuple(seen)


def effective_account(options: PlanOptions) -> str | None:
    """--account, else $MPD_ACCOUNT, else the profile's, else None (Slurm bills the default account)."""
    return options.account or os.environ.get("MPD_ACCOUNT") or options.cluster.account or None


def forwarded_env(environ=None) -> dict[str, str]:
    """The MPD_* variables of the submitting shell that job scripts export (paths made absolute: a job changes directory)."""
    environ = os.environ if environ is None else environ
    out: dict[str, str] = {}
    for name in sorted(environ):
        if name in FORWARDED_ENV or (name.startswith(FORWARDED_PREFIX) and _ENV_NAME.fullmatch(name)):
            value = environ[name]
            check_env(name, value)
            if name in _DIR_ENV or name.startswith(FORWARDED_PREFIX) or (name == "MPD_FOLDSEEK" and "/" in value):
                value = os.path.abspath(os.path.expanduser(value))
            out[name] = value
    return out


def job_env(options: PlanOptions, environ=None) -> tuple[dict[str, str], dict[str, str]]:
    """(MPD_* variables forwarded from the submitting shell, the --env pairs). The scripts export --env last, so it wins."""
    return forwarded_env(environ), dict(options.extra_env)


_READ = object()


def default_python(home: Path, record=_READ) -> str:
    """$MPD_PYTHON, else the interpreter in the environment record, else <home>/env/bin/python."""
    if os.environ.get("MPD_PYTHON"):
        return os.environ["MPD_PYTHON"]
    if record is _READ:
        record = envrecord.read(home)
        if record and record.get("home") and os.path.realpath(record["home"]) != os.path.realpath(home):
            record = None                      # written for another home: its interpreter is not this home's
    if record and record.get("python"):
        return record["python"]
    return str(Path(home) / "env" / "bin" / "python")


def resolve_python(options: PlanOptions, record=_READ) -> str:
    """The interpreter the jobs call: --python, else $MPD_PYTHON, else the environment record's, else <home>/env/bin/python."""
    return options.python or default_python(Path(options.home), record)


# ---- memory and time models --------------------------------------------------------------------------------------------
def _usable_gb(gpu: GpuType) -> float:
    return gpu.mem_gb * GPU_MEM_USABLE


def _need_parts(model: registry.ModelInfo, gpu: GpuType, micro: int | None) -> tuple[float, float]:
    """(weights, activations) in GB for one process. ``micro`` is the training micro-batch; None means no_grad inference at batch 8."""
    scale = 1.0 if gpu.bf16 else 2.0                     # float32 weights and activations where bfloat16 is missing
    if micro is None:
        activations = INFERENCE_ACT_FRACTION * model.act_gb
    else:
        activations = model.act_gb * micro / ACT_REFERENCE_MICROBATCH
    return model.weights_gb * scale, activations * scale


def _need_gb(model: registry.ModelInfo, gpu: GpuType, micro: int | None) -> float:
    return sum(_need_parts(model, gpu, micro)) + FIXED_GB


def _no_fit(model: registry.ModelInfo, gpu: GpuType, micro: int | None) -> PlanError:
    weights, activations = _need_parts(model, gpu, micro)
    usable = _usable_gb(gpu)
    return PlanError(
        f"{model.key} needs about {weights + activations + FIXED_GB:.0f} GB (weights {weights:.1f} + activations {activations:.1f} "
        f"+ {FIXED_GB:g} fixed; estimated{'' if gpu.bf16 else ', float32'}) but {gpu.name} offers {usable:.0f} GB usable per GPU"
        + (f" and {gpu.per_node * usable:.0f} GB across a node" if model.shard_across_gpus else "")
        + ". Choose a larger GPU with --gpu or a smaller model.")


def _gpus_for(model: registry.ModelInfo, gpu: GpuType, micro: int | None) -> int:
    """GPUs one process group needs for the model: one, or a node's worth for a model that is sharded across GPUs."""
    need, usable = _need_gb(model, gpu, micro), _usable_gb(gpu)
    if need <= usable:
        return 1
    if model.shard_across_gpus:
        for n in range(2, gpu.per_node + 1):
            if need <= n * usable:
                return n
    raise _no_fit(model, gpu, micro)


def _round_up_minutes(seconds: float, step: int = 15) -> int:
    minutes = max(step, math.ceil(seconds / 60 / step) * step)
    return minutes * 60


def size_factor(model: registry.ModelInfo) -> float:
    return (model.params_m / REFERENCE_PARAMS_M) ** SIZE_EXPONENT


def train_walltime_s(student: registry.ModelInfo, scope: str, updates: int, dp: int, gpu: GpuType) -> int:
    """Walltime of one training run: H200 minutes per 1,000 updates (measured for esm2-650m) x model size x GPU slowdown,
    divided by the data-parallel width, plus slack and the start-up allowance."""
    minutes = H200_TRAIN_MIN_PER_1000[scope] * size_factor(student) * updates / 1000.0 / max(1, dp) * gpu.time_factor
    return _round_up_minutes(minutes * TIME_SLACK * 60 + STARTUP_S)


def forward_walltime_s(model: registry.ModelInfo, tokens: float, gpu: GpuType) -> int:
    """Walltime of a forward-only pass over ``tokens`` (teacher and score tasks)."""
    seconds = tokens / H200_FORWARD_TOKENS_PER_S * size_factor(model) * gpu.time_factor
    return _round_up_minutes(seconds * TIME_SLACK + STARTUP_S)


def union_tokens(spec: RunSpec, support: str) -> float:
    """Tokens the teacher walks through: the training union (an upper bound: assays on one protein share contexts)."""
    size = BENCHMARK_SIZES[spec.train_data]["dense" if support == "aa20_dense" else "observed"]
    if spec.scope == "all":
        return float(size.tokens)
    return float(min(size.tokens, len(spec.assays) * size.assay_mean_tokens))


def scored_on_own_assay(spec: RunSpec, unit: str, dataset: str) -> bool:
    """A one-assay adapter is scored on its own assay when the training benchmark is evaluated (mirrors
    stages.common.Campaign.evaluated_on_own_assay; a test keeps the two equal)."""
    return spec.scope == "one" and unit != "base" and dataset == spec.train_data


def score_task_tokens(dataset: str, shards: int, own_assay: bool) -> float:
    size = BENCHMARK_SIZES[dataset]["observed"]
    return float(size.assay_max_tokens if own_assay else size.tokens / shards)


# ---- planning ---------------------------------------------------------------------------------------------------------
def _absolute(path, what: str) -> Path:
    path = Path(path)
    if not path.is_absolute():
        raise PlanError(f"{what} {path} is a relative path; jobs change directory, so it would not resolve. Give an absolute path")
    return path


def check_slurm_path(path) -> None:
    """Refuse a path that cannot be written on a `#SBATCH --output=` line (Slurm reads those verbatim)."""
    if not _SAFE_PATH.fullmatch(str(path)):
        raise PlanError(f"{path} has characters that Slurm directives cannot carry (spaces, quotes, '%', '$', '#' ...). "
                        "Use a path made of letters, digits and _ . + @ : , = / -")


def _usable_record(home: Path, cluster: Cluster) -> tuple[dict | None, str | None]:
    """The environment record, unless it is damaged or belongs to another cluster or home."""
    reason = envrecord.problem(home)
    if reason:
        return None, f"ignoring the environment record: {reason}; re-run `bash scripts/setup_env.sh --cluster {cluster.name}`"
    record = envrecord.read(home)
    if record is None:
        return None, None
    if record.get("cluster") != cluster.name:
        return None, (f"ignoring {envrecord.record_path(home)}: the environment was built for cluster "
                      f"{record.get('cluster')!r}, not {cluster.name!r}")
    if record.get("home") and os.path.realpath(record["home"]) != os.path.realpath(home):
        return None, (f"ignoring {envrecord.record_path(home)}: it was written for {record['home']}, not {home}; "
                      f"re-run `bash scripts/setup_env.sh --cluster {cluster.name} --home {home}`")
    return record, None


def _identity(spec: RunSpec, options: PlanOptions, gpu: GpuType, dp: int, shards: dict, throttle: int,
              user_seconds: dict, planner_env: dict) -> str:
    """Eight hex digits of everything that changes the tasks or scripts of a campaign but is not already in
    ``spec.campaign_id()``: evaluation data, cluster, GPU type, training width, score shards, array throttle, walltime
    overrides, extra sbatch options, extra modules and --env pairs, and the numerics variables (MPD_DTYPE, MPD_TF32,
    MPD_STRUCTURE_LEGACY_MASK).
    Deliberately absent: home, python, account and nonce, so moving or re-pointing a campaign keeps its identity."""
    env = {**planner_env, **options.extra_env}
    payload = {
        "eval_data": sorted(spec.eval_data), "cluster": options.cluster.name, "gpu": gpu.name, "train_gpus": dp,
        "score_shards": {ds: shards[ds] for ds in sorted(spec.eval_data)}, "array_throttle": throttle,
        "walltimes": user_seconds, "extra_sbatch": list(options.extra_sbatch), "extra_modules": list(options.extra_modules),
        "extra_env": {k: v for k, v in sorted(options.extra_env.items()) if k not in _NUMERICS_ENV},
        "numerics": {k: env[k] for k in _NUMERICS_ENV if k in env},
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:8]


def make_plan(spec: RunSpec, options: PlanOptions) -> Plan:
    spec.validate()
    cluster = options.cluster
    gpu = cluster.gpu(options.gpu)
    teacher, student = registry.get(spec.teacher), registry.get(spec.student)
    warnings: list[str] = []
    errors: list[str] = []

    # ---- options: absolute paths, safe strings, the environment record -----------------------------------------
    home = _absolute(options.home, "home")
    check_slurm_path(home)
    record, note = _usable_record(home, cluster)
    if note:
        warnings.append(note)
    python = resolve_python(replace(options, home=home), record)
    if not os.path.isabs(python):
        if "/" in python:
            raise PlanError(f"--python {python} is a relative path; jobs change directory, so it would not resolve. "
                            "Give an absolute path")
        warnings.append(f"python {python!r} is looked up on PATH inside the job (after the modules load); "
                        "an absolute path in the environment is more reproducible")
    account = effective_account(options)
    if account is not None:
        validate_account(account, "--account" if options.account else "MPD_ACCOUNT")
    extra_sbatch = tuple(validate_sbatch_flag(f) for f in options.extra_sbatch)
    extra_modules = tuple(validate_module(m) for m in options.extra_modules)
    for name, value in options.extra_env.items():
        check_env(name, value)
    if options.nonce and not _NONCE.fullmatch(options.nonce):
        raise PlanError(f"nonce {options.nonce!r} must be 6-32 letters or digits")
    env = dict(options.extra_env)
    if options.allow_unsupported_gpu and not gpu.bf16:
        env.setdefault("MPD_DTYPE", "float32")
    options = replace(options, home=home, python=python, account=account, extra_sbatch=extra_sbatch,
                      extra_modules=extra_modules, extra_env=env)
    planner_env, _ = job_env(options)
    for flag in extra_sbatch:
        if flag.split("=", 1)[0] in _PLANNED_FLAGS:
            warnings.append(f"--sbatch-arg {flag} overrides a value mpd chose; the plan's resources, walltime and SU figures "
                            "do not reflect it")
    if account is None and cluster.bills_su:
        hint = " (`myproject` lists your accounts; pass --account or export MPD_ACCOUNT)" if cluster.name == "grace" else ""
        warnings.append(f"no --account given: SUs go to your default {cluster.name} account{hint}")
    for number, flag in ((spec.peak_lr, "--lr"), (spec.kl_weight, "--kl-weight")):
        if number is not None and not math.isfinite(number):
            raise PlanError(f"{flag} must be a finite number, got {number!r}")
    if spec.peak_lr is not None and spec.peak_lr <= 0:
        raise PlanError(f"--lr must be positive, got {spec.peak_lr:g}")
    if spec.kl_weight is not None and spec.kl_weight < 0:
        raise PlanError(f"--kl-weight must be zero or positive, got {spec.kl_weight:g}")
    if options.array_throttle is not None and options.array_throttle < 1:
        raise PlanError(f"--array-throttle must be at least 1, got {options.array_throttle}")
    throttle = options.array_throttle or cluster.array_throttle
    shards = {**DEFAULT_SCORE_SHARDS, **check_score_shards(options.score_shards or {})}
    unused = sorted(set(options.score_shards or {}) - set(spec.eval_data))
    if unused:
        warnings.append(f"--score-shards for {', '.join(unused)} is ignored: it is not in --eval-data")
    user_seconds = {flag: (_user_walltime(text, f"--{flag}-time") if text else None) for flag, text in
                    (("train", options.train_time), ("score", options.score_time), ("teacher", options.teacher_time))}
    if not gpu.bf16:
        if not options.allow_unsupported_gpu:
            better = " or ".join(f"--gpu {k}" for k, g in cluster.gpus.items() if g.bf16) or "a GPU with bfloat16"
            raise PlanError(f"{gpu.name} cannot train here: it has no native bfloat16 (Turing) and mpd trains in bfloat16. "
                            f"Use {better}. To run anyway in float32 (slow, about twice the memory, never tested) "
                            "pass --allow-unsupported-gpu")
        warnings.append(f"{gpu.name} has no native bfloat16: --allow-unsupported-gpu runs every stage in float32 "
                        "(MPD_DTYPE=float32 is exported unless --env sets it); slow, about twice the memory, never tested")

    # ---- the training recipe: micro-batch from memory, width from the profile ----------------------------------
    preset = preset_for(student.family, spec.objective)
    updates = spec.updates or preset.updates
    global_batch = spec.global_batch or preset.global_batch
    if student.shard_across_gpus:
        micro, train_gpus_model = preset.microbatch, _gpus_for(student, gpu, preset.microbatch)
    else:
        micro, train_gpus_model = next(
            ((m, 1) for m in range(preset.microbatch, 0, -1) if _need_gb(student, gpu, m) <= _usable_gb(gpu)), (0, 1))
        if not micro:
            raise _no_fit(student, gpu, 1)

    if spec.scope == "all" and not student.shard_across_gpus:
        if spec.train_gpus is not None:
            if spec.train_gpus < 1:
                raise PlanError(f"--train-gpus must be at least 1, got {spec.train_gpus}; leave it out to use the profile's default")
            problem = cluster.gpu_count_error(gpu, spec.train_gpus)
            if problem:
                raise PlanError(f"[train] {problem}")
            dp = spec.train_gpus
        else:
            # profile default (Grace: one GPU), else the widest of up to 4 GPUs (the paper's setup) that splits the global batch
            cap = min(cluster.default_train_gpus or min(gpu.per_node, 4), gpu.per_node)
            fits = [n for n in range(1, cap + 1) if global_batch % (n * micro) == 0]
            dp = max(fits) if fits else 1
    else:
        dp = 1
    per_step = dp * micro
    if global_batch % per_step:
        raise PlanError(f"global batch {global_batch} is not divisible by GPUs x microbatch = {dp} x {micro}; "
                        f"pick --global-batch as a multiple of {per_step} or change --train-gpus")
    accumulation = global_batch // per_step
    train_gpus = train_gpus_model if student.shard_across_gpus else dp

    campaign_id = f"{spec.campaign_id()}__{_identity(spec, options, gpu, dp, shards, throttle, user_seconds, planner_env)}"
    if not _SAFE_ID.fullmatch(campaign_id):
        raise PlanError(f"campaign id {campaign_id!r} has characters outside letters, digits and _ . + -; check the assay ids")
    campaign_dir = home / "runs" / campaign_id
    nonce = options.nonce or secrets.token_hex(6)

    resolved = {
        "campaign_id": campaign_id,
        "teacher": spec.teacher, "student": spec.student, "train_data": spec.train_data,
        "objective": spec.objective, "scope": spec.scope, "assays": list(spec.assays),
        "eval_data": list(spec.eval_data), "teacher_backend": spec.backend, "seed": spec.seed,
        "updates": updates, "global_batch": global_batch, "microbatch": micro,
        "accumulation": accumulation, "world_size": dp, "train_gpus": train_gpus,
        "peak_lr": preset.peak_lr if spec.peak_lr is None else spec.peak_lr,
        "kl_weight": preset.kl_weight if spec.kl_weight is None else spec.kl_weight,
        "support": preset.support, "lora_rank": preset.lora_rank, "lora_alpha": preset.lora_alpha,
        "dev_fraction": spec.dev_fraction or preset.dev_fraction, "warmup_fraction": preset.warmup_fraction,
        "min_lr_ratio": preset.min_lr_ratio, "grad_clip": preset.grad_clip, "weight_decay": preset.weight_decay,
        "gpu": gpu.name, "cluster": cluster.name,
        "score_shards": {ds: shards[ds] for ds in spec.eval_data},      # stages count only shard files with this `num`
    }
    if not student.act_gb_measured:
        warnings.append(
            f"{student.key}: GPU memory ({_need_gb(student, gpu, micro):.0f} GB of {_usable_gb(gpu):.0f} GB usable on {gpu.name}, "
            f"micro-batch {micro}) is an estimate, not a measurement. If the first job runs out of memory use a larger GPU "
            "(--gpu) or a smaller student: more GPUs do not help, because training is data-parallel and every GPU holds a full copy")
    if micro < preset.microbatch:
        warnings.append(f"{student.key} trains at micro-batch {micro} on {gpu.name} (memory); the global batch of "
                        f"{global_batch} is kept through {accumulation} accumulation steps")

    # ---- stage construction ------------------------------------------------------------------------------------
    stages: list[Stage] = []
    max_array = MAX_ARRAY_TASKS

    def pick_walltime(label, flag, key, estimate_s, model):
        """The user's override, else the estimate; every number that is not a measurement says so."""
        if user_seconds[key] is not None:
            if user_seconds[key] < estimate_s:
                warnings.append(f"{flag} {format_walltime(user_seconds[key])} is shorter than the estimate "
                                f"{format_walltime(estimate_s)} for {label}; the job may stop at its limit (TIMEOUT)")
            return user_seconds[key], "user", True
        measured = model.key in MEASURED_STUDENTS and gpu.time_factor == 1.0
        warnings.append(f"{label} walltime {format_walltime(estimate_s)} is "
                        + ("scaled from timed H200 runs of esm2-650m" if measured else "an estimate, not measured")
                        + f" (x {TIME_SLACK:g} slack + {STARTUP_S // 60} min start-up"
                        + ("" if model.key in MEASURED_STUDENTS else f"; extrapolated to {model.key}")
                        + ("" if gpu.time_factor == 1.0 else f"; {gpu.name} slowdown {gpu.time_factor:g} is assumed")
                        + f"); override with {flag}")
        return estimate_s, "measured" if measured else "estimate", False

    def gpu_stage(name, module, tasks, n_gpus, walltime_s, launcher, depends, *, note=None, basis="fixed", user_set=False):
        cpus, mem = cluster.cpus_and_mem(gpu, n_gpus)
        for level, msg in cluster.check_gpu_job(gpu, n_gpus, walltime_s, user_set=user_set):
            (errors.append(f"[{name}] {msg}") if level == "error" else warnings.append(msg))
        parts = [tasks[i:i + max_array] for i in range(0, len(tasks), max_array)] or [[]]
        names = []
        for k, chunk in enumerate(parts):
            su = cluster.estimate_su(gpu=gpu, n_gpus=n_gpus, cpus=cpus, mem_gb=mem,
                                     walltime_s=walltime_s, tasks=len(chunk))
            stage = Stage(name=name, part=k, kind="gpu", module=module, gpu=gpu.name, gpus=n_gpus, cpus=cpus,
                          mem_gb=mem, walltime_s=walltime_s, tasks=chunk, task_offset=k * max_array,
                          launcher=launcher, depends_on=list(depends), su_estimate=su,
                          notes=[note] if note else [], walltime_basis=basis)
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

    prepare = cpu_stage("prepare", "mpd.stages.prepare", cpus=8, mem_gb=32, walltime_s=PREPARE_S, depends=[])
    upstream = list(prepare)

    if spec.backend == "live":
        t_gpus = _gpus_for(teacher, gpu, None)
        tokens = union_tokens(spec, preset.support)
        t_time, basis, user_set = pick_walltime(
            f"teacher ({tokens / 1e6:.1f}M tokens)", "--teacher-time", "teacher", forward_walltime_s(teacher, tokens, gpu), teacher)
        upstream = gpu_stage("teacher", "mpd.stages.teacher", [{}], t_gpus, t_time, "python", prepare,
                             note="computes the AA20 logit targets the student will imitate", basis=basis, user_set=user_set)

    # training
    unit_tasks = [{"unit": u} for u in spec.units]
    train_s, basis, user_set = pick_walltime(
        f"train ({updates} updates, {dp} GPU{'s' if dp > 1 else ''})", "--train-time", "train",
        train_walltime_s(student, spec.scope, updates, dp, gpu), student)
    train_names = gpu_stage("train", "mpd.stages.train", unit_tasks, train_gpus, train_s,
                            "ddp" if dp > 1 else "python", upstream, basis=basis, user_set=user_set)

    # scoring: the frozen base student runs as soon as inputs exist; adapters wait for training
    score_gpus = _gpus_for(student, gpu, None)

    def score_tasks(units):
        tasks, heaviest = [], 0.0
        for u in units:
            for ds in spec.eval_data:
                own = scored_on_own_assay(spec, u, ds)
                n = 1 if own else shards[ds]                    # a one-assay table is far too small to split
                heaviest = max(heaviest, score_task_tokens(ds, n, own))
                tasks += [{"unit": u, "dataset": ds, "shard": s, "num_shards": n} for s in range(n)]
        return tasks, heaviest

    base, heavy = score_tasks(["base"])
    s_time, basis, user_set = pick_walltime(f"score-base (up to {heavy / 1e6:.1f}M tokens per task)", "--score-time", "score",
                                            forward_walltime_s(student, heavy, gpu), student)
    base_names = gpu_stage("score-base", "mpd.stages.score", base, score_gpus, s_time, "python", prepare,
                           note="frozen student, no adapter (reference for deltas)", basis=basis, user_set=user_set)
    units, heavy = score_tasks(list(spec.units))
    s_time, basis, user_set = pick_walltime(f"score (up to {heavy / 1e6:.1f}M tokens per task)", "--score-time", "score",
                                            forward_walltime_s(student, heavy, gpu), student)
    score_names = gpu_stage("score", "mpd.stages.score", units, score_gpus, s_time, "python", train_names,
                            basis=basis, user_set=user_set)
    cpu_stage("evaluate", "mpd.stages.evaluate", cpus=4, mem_gb=16, walltime_s=EVALUATE_S, depends=base_names + score_names)

    if errors:
        raise PlanError("\n".join(errors))
    if spec.scope == "one" and len(spec.assays) > 20:
        warnings.append(f"{len(spec.assays)} one-assay trainings will be submitted as arrays (throttle {throttle} at a time)")
    return Plan(spec=spec, options=options, campaign_dir=campaign_dir, stages=stages, resolved=resolved,
                warnings=list(dict.fromkeys(warnings)), nonce=nonce, campaign_id=campaign_id, env_record=record)


def describe(plan: Plan) -> str:
    c = plan.options.cluster
    modules = f"   modules {' '.join(plan.modules)}" if plan.modules else ""
    lines = [f"campaign   {plan.campaign_id}",
             f"directory  {plan.campaign_dir}",
             f"cluster    {c.name}  gpu={plan.resolved['gpu']}  home={plan.options.home}",
             f"python     {plan.options.python}{modules}",
             f"teacher    {plan.spec.teacher} ({plan.spec.backend})   student {plan.spec.student}",
             f"train      {plan.spec.train_data}  method={plan.spec.method}  units={len(plan.spec.units)}",
             f"eval       {', '.join(plan.spec.eval_data)}",
             f"recipe     {plan.resolved['updates']} updates, global batch {plan.resolved['global_batch']} "
             f"(= {plan.resolved['world_size']} GPU x {plan.resolved['microbatch']} micro x "
             f"{plan.resolved['accumulation']} accum), lr {plan.resolved['peak_lr']}, "
             f"kl {plan.resolved['kl_weight']}, support {plan.resolved['support']}", "",
             f"{'stage':<14}{'tasks':>6}  {'gpu':<9}{'cpus':>5}{'mem':>6}  {'walltime':<16}{'SU (max)':>10}  after"]
    tag = {"estimate": " est", "measured": " meas", "user": " set"}
    for s in plan.stages:
        gpu = f"{s.gpus}x{s.gpu}" if s.gpus else "-"
        su = f"{s.su_estimate:>10,.0f}" if c.bills_su else f"{'n/a':>10}"
        lines.append(f"{s.full_name:<14}{len(s.tasks):>6}  {gpu:<9}{s.cpus:>5}{s.mem_gb:>5}G  "
                     f"{format_walltime(s.walltime_s) + tag.get(s.walltime_basis, ''):<16}{su}  {', '.join(s.depends_on) or '-'}")
    lines += ["", "walltime: est = estimate, not measured; meas = scaled from timed runs; set = given by you"]
    if c.bills_su:
        lines += [f"estimated upper bound: {plan.total_su:,.0f} SU (every task runs to its requested walltime and the node share "
                  "it asks for; the charge follows actual runtime). Estimate, not measured.",
                  f"SU rule: (effective cores + GPU rate x GPUs) x hours; memory counts as cores. {c.display_name}'s sbatch prints a "
                  "\"Requested SUs\" line that counts CPUs only, so it is lower than the total above; `myproject` shows your balance."]
    else:
        lines.append(f"estimated upper bound: not billed (the {c.name} profile has no SU rates)")
    for w in plan.warnings:
        lines.append(f"warning: {w}")
    return "\n".join(lines)
