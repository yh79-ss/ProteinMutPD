"""Cluster profiles: resource shapes, policy checks and a service-unit estimate.

Profiles live in ``mpd/clusters/*.yaml`` (or any YAML path). Nothing here imports torch,
so planning is safe on a login node.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import yaml

# SUs follow the AMS "Service Unit" rule (hprc.tamu.edu/kb/User-Guides/AMS, non-exclusive jobs):
#   memory_equivalent_core = min(m, ceil(mem / total_memory * m))      m = cores per node
#   effective_core         = max(requested cores, memory_equivalent_core)
#   SU                     = (effective_core + SU rate of the GPU * GPUs) * hours
# Grace: m = 48, total_memory = 351.5625 GB. `maxconfig` agrees (24 cores + 180 GB -> 25 effective cores; one A100 for
# 96 h = 9,312 SU) and so does the job_submit message of a real Grace job (4 cores + 100 GB, 4 h, 5 array tasks ->
# "Requested SUs: 280"). That message counts CPUs only: the GPU part is charged when the job ends.

# Values that land on `#SBATCH` lines or after `module load` are read verbatim by Slurm and bash: accept only
# what cannot end the line, split an option in two or add a command.
ACCOUNT_RE = re.compile(r"[A-Za-z0-9_.-]+")                     # HPRC samples show 6, 12 and 13 digits: no length rule
MODULE_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.+/-]*")
SBATCH_FLAG_RE = re.compile(r"--[A-Za-z][A-Za-z0-9-]*(=[A-Za-z0-9_.:,@%/+*=|&()\[\]-]*)?")
GRES_TYPE_RE = re.compile(r"[A-Za-z0-9_.-]+")                   # the H200 in --gres=gpu:H200:1
CPU_GRES_RE = re.compile(r"gpu(:[A-Za-z0-9_.-]+)?:[0-9]+")      # e.g. gpu:1
NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


class ClusterError(ValueError):
    pass


_WALLTIME = re.compile(r"(?:(?P<d>[0-9]+)-)?(?P<a>[0-9]+)(?::(?P<b>[0-9]+))?(?::(?P<c>[0-9]+))?")
MAX_WALLTIME_S = 366 * 86400


def parse_walltime(value, *, strict: bool = False) -> int:
    """Slurm walltime to seconds: M, M:S, H:M:S, D-H, D-H:M, D-H:M:S.

    ``strict`` is for what a user types: it refuses bare minutes and M:S (Slurm reads "08:00" as 8 minutes),
    which people mean as hours.
    """
    text = str(value).strip()
    m = _WALLTIME.fullmatch(text)
    if not m:
        raise ClusterError(f"{value!r} is not a walltime. Write HH:MM:SS (08:00:00 is eight hours) "
                           "or D-HH:MM:SS (1-12:00:00 is a day and a half)")
    day, a, b, c = m.group("d"), m.group("a"), m.group("b"), m.group("c")
    if day is not None:                    # D-H, D-H:M, D-H:M:S
        days, hours, minutes, seconds = int(day), int(a), int(b or 0), int(c or 0)
        checks = (("hour", hours, 23), ("minute", minutes, 59), ("second", seconds, 59))
    elif c is not None:                    # H:M:S, the leading field may exceed 23
        days, hours, minutes, seconds = 0, int(a), int(b), int(c)
        checks = (("minute", minutes, 59), ("second", seconds, 59))
    else:                                  # M or M:S, which is what Slurm makes of a bare number or "08:00"
        days, hours, minutes, seconds = 0, 0, int(a), int(b or 0)
        checks = (("second", seconds, 59),)
    for unit, number, top in checks:
        if number > top:
            raise ClusterError(f"{value!r}: the {unit} field must be 0-{top}")
    total = ((days * 24 + hours) * 60 + minutes) * 60 + seconds
    if total <= 0:
        raise ClusterError(f"{value!r} is zero; Slurm treats a zero time limit as unlimited. Give a positive walltime")
    if strict and day is None and c is None:
        if b is None:
            raise ClusterError(f"{value!r} is ambiguous: Slurm reads it as {minutes} minutes. Write {minutes:02d}:00:00 "
                               f"if you mean hours or {minutes // 60:02d}:{minutes % 60:02d}:00 if you mean minutes")
        reads = f"{minutes} minutes" + (f" {seconds} seconds" if seconds else "")
        raise ClusterError(f"{value!r} is ambiguous: Slurm reads it as {reads}. Write {minutes:02d}:{seconds:02d}:00 "
                           f"for hours and minutes or 00:{minutes:02d}:{seconds:02d} for minutes and seconds")
    if total > MAX_WALLTIME_S:
        raise ClusterError(f"{value!r} is longer than a year")
    return total


def format_walltime(seconds: int) -> str:
    seconds = int(math.ceil(seconds))
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    base = f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{days}-{base}" if days else base


@dataclass(frozen=True)
class GpuType:
    name: str
    per_node: int
    mem_gb: float
    su_per_gpu_hour: float
    partition: str | None
    max_walltime_s: int
    max_nodes: int | None
    time_factor: float            # planning slowdown vs one H200; an ASSUMPTION, see profile
    unverified: str | None = None
    gres: str = ""                # type in `--gres=gpu:<gres>:N`, spelled as the site does (Vision: H200); default = the key
    bf16: bool = True             # False on Turing (T4, RTX 6000): mpd trains in bfloat16

    def __post_init__(self) -> None:
        if not self.gres:
            object.__setattr__(self, "gres", self.name)


@dataclass(frozen=True)
class Cluster:
    name: str
    description: str
    account: str | None
    node_cores: int
    node_mem_gb: int
    sbatch_extra: tuple[str, ...]
    modules: tuple[str, ...]
    gpus: dict[str, GpuType]
    cpu_partition: str | None
    cpu_max_walltime_s: int
    su_per_core_hour: float
    default_gpu: str
    array_throttle: int
    resource_rule: str
    fixed_per_gpu: tuple[int, int] | None
    storage_root: str
    login_max_cores: int
    login_max_minutes: int
    su_mem_total_gb: float = 0.0              # memory that bills a whole node (AMS rule); 0 -> node_mem_gb
    forbidden_storage_prefixes: tuple[str, ...] = ()
    default_train_gpus: int | None = None     # width of *-all training when --train-gpus is absent; None -> up to 4
    home_max_files: int | None = None
    home_max_gb: float | None = None
    file_quota_warning: int | None = None
    cpu_gres: str | None = None               # e.g. "gpu:1": a GPU request on CPU stages for sites that reject jobs without one

    def __post_init__(self) -> None:
        if not self.su_mem_total_gb:
            object.__setattr__(self, "su_mem_total_gb", float(self.node_mem_gb))

    # ---- loading ---------------------------------------------------------
    @classmethod
    def load(cls, name_or_path: str) -> "Cluster":
        arg = str(name_or_path)
        path = Path(arg)
        if path.suffix in (".yaml", ".yml") or os.sep in arg:
            if not path.is_file():
                raise ClusterError(f"cluster profile {arg} not found")
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                raise ClusterError(f"cannot read cluster profile {arg}: {exc}") from None
            source = arg
        else:
            known = ", ".join(available_clusters())
            if not re.fullmatch(r"[A-Za-z0-9_-]+", arg):
                raise ClusterError(f"unknown cluster {arg!r}; use one of {known} or a path to a profile YAML")
            name = arg.lower()                  # Grace and GRACE are grace; the spelling of a path to a profile file is kept
            try:
                text = resources.files("mpd.clusters").joinpath(f"{name}.yaml").read_text(encoding="utf-8")
            except FileNotFoundError:
                raise ClusterError(f"unknown cluster {arg!r}; use one of {known} or a path to a profile YAML") from None
            source = f"{name}.yaml"
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ClusterError(f"{source}: not valid YAML ({str(exc).splitlines()[0] if str(exc) else 'parse error'})") from None
        try:
            return cls.from_dict(data)
        except ClusterError as exc:
            raise ClusterError(f"{source}: {exc}") from None

    @classmethod
    def from_dict(cls, d: dict) -> "Cluster":
        """Build a profile from parsed YAML. Every problem is a ClusterError that names the field."""
        top = _section(d, "profile", _TOP_FIELDS)
        node = _section(_required(top, "node", "profile", dict), "node", _NODE_FIELDS)
        gpus_in = _required(top, "gpus", "profile", dict)
        if not gpus_in:
            raise ClusterError("gpus: list at least one GPU type")
        gpus = {}
        for key, g in gpus_in.items():
            where = f"gpus.{key}"
            if not isinstance(key, str) or not NAME_RE.fullmatch(key):
                raise ClusterError(f"{where}: GPU keys are letters, digits, '_', '.' or '-' (got {key!r})")
            g = _section(g, where, _GPU_FIELDS)
            gres = _opt_str(g, "gres", where, GRES_TYPE_RE) or key
            gpus[key] = GpuType(
                name=key, per_node=_int(_required(g, "per_node", where), f"{where}.per_node", minimum=1),
                mem_gb=_num(_required(g, "mem_gb", where), f"{where}.mem_gb", minimum=0, strict=True),
                su_per_gpu_hour=_num(g.get("su_per_gpu_hour", 0), f"{where}.su_per_gpu_hour", minimum=0),
                partition=_opt_str(g, "partition", where, NAME_RE),
                max_walltime_s=_walltime(_required(g, "max_walltime", where), f"{where}.max_walltime"),
                max_nodes=None if g.get("max_nodes") is None else _int(g["max_nodes"], f"{where}.max_nodes", minimum=1),
                time_factor=_num(g.get("time_factor", 1.0), f"{where}.time_factor", minimum=0, strict=True),
                unverified=_opt_str(g, "unverified", where), gres=gres, bf16=_bool(g.get("bf16", True), f"{where}.bf16"))
        default_gpu = _required(top, "default_gpu", "profile")
        if default_gpu not in gpus:
            raise ClusterError(f"default_gpu: {default_gpu!r} is not one of the profile's GPUs ({', '.join(gpus)})")

        cpu = _section(top.get("cpu") or {}, "cpu", _CPU_FIELDS)
        login = _section(top.get("login_node") or {}, "login_node", _LOGIN_FIELDS)
        storage = _section(top.get("storage") or {}, "storage", _STORAGE_FIELDS)
        rule = top.get("resource_rule", "proportional_to_node")
        if rule not in ("proportional_to_node", "fixed_per_gpu"):
            raise ClusterError(f"resource_rule: expected proportional_to_node or fixed_per_gpu, got {rule!r}")
        fixed = None
        if rule == "fixed_per_gpu":
            f = _section(_required(top, "fixed_per_gpu", "profile (resource_rule is fixed_per_gpu)", dict),
                         "fixed_per_gpu", {"cpus", "mem_gb"})
            fixed = (_int(_required(f, "cpus", "fixed_per_gpu"), "fixed_per_gpu.cpus", minimum=1),
                     _int(_required(f, "mem_gb", "fixed_per_gpu"), "fixed_per_gpu.mem_gb", minimum=1))
        extra = top.get("sbatch_extra") or []
        modules = top.get("modules") or []
        if not isinstance(extra, list) or not isinstance(modules, list):
            raise ClusterError("sbatch_extra and modules must be lists")
        for flag in extra:
            if not isinstance(flag, str) or not SBATCH_FLAG_RE.fullmatch(flag):
                raise ClusterError(f"sbatch_extra: {flag!r} is not a --name or --name=value option without spaces")
        for module in modules:
            if not isinstance(module, str) or not MODULE_RE.fullmatch(module):
                raise ClusterError(f"modules: {module!r} is not a module name (letters, digits, '_', '.', '+', '/', '-')")
        account = top.get("account")
        if isinstance(account, int) and not isinstance(account, bool):
            account = str(account)             # YAML reads an unquoted 123456789012 as a number
        if account is not None and (not isinstance(account, str) or not ACCOUNT_RE.fullmatch(account)):
            raise ClusterError(f"account: {account!r} must be letters, digits, '_', '.' or '-'")
        node_mem = _num(_required(node, "usable_mem_gb", "node"), "node.usable_mem_gb", minimum=0, strict=True)
        prefixes = storage.get("forbidden_prefixes") or []
        if not isinstance(prefixes, list) or not all(isinstance(p, str) and p.startswith("/") for p in prefixes):
            raise ClusterError("storage.forbidden_prefixes: expected a list of absolute paths such as [\"/home\"]")
        return cls(
            name=_name(_required(top, "name", "profile"), "name"), description=str(top.get("description", "")),
            account=account, node_cores=_int(_required(node, "cores", "node"), "node.cores", minimum=1),
            node_mem_gb=int(node_mem), sbatch_extra=tuple(extra), modules=tuple(modules), gpus=gpus,
            cpu_partition=_opt_str(cpu, "partition", "cpu", NAME_RE),
            cpu_max_walltime_s=_walltime(cpu.get("max_walltime", "1-00:00:00"), "cpu.max_walltime"),
            su_per_core_hour=_num(cpu.get("su_per_core_hour", 0), "cpu.su_per_core_hour", minimum=0),
            default_gpu=default_gpu, array_throttle=_int(top.get("array_throttle", 4), "array_throttle", minimum=1),
            resource_rule=rule, fixed_per_gpu=fixed, storage_root=str(storage.get("root", "{env:MPD_HOME}")),
            login_max_cores=_int(login.get("max_cores", 8), "login_node.max_cores", minimum=1),
            login_max_minutes=_int(login.get("max_minutes", 60), "login_node.max_minutes", minimum=1),
            su_mem_total_gb=_num(node.get("su_mem_total_gb", node_mem), "node.su_mem_total_gb", minimum=0, strict=True),
            forbidden_storage_prefixes=tuple(p.rstrip("/") or "/" for p in prefixes),
            default_train_gpus=None if top.get("default_train_gpus") is None
            else _int(top["default_train_gpus"], "default_train_gpus", minimum=1),
            home_max_files=None if storage.get("home_max_files") is None
            else _int(storage["home_max_files"], "storage.home_max_files", minimum=1),
            home_max_gb=None if storage.get("home_max_gb") is None
            else _num(storage["home_max_gb"], "storage.home_max_gb", minimum=0, strict=True),
            file_quota_warning=None if storage.get("file_quota_warning") is None
            else _int(storage["file_quota_warning"], "storage.file_quota_warning", minimum=1),
            cpu_gres=_opt_str(cpu, "gres", "cpu", CPU_GRES_RE))

    # ---- resources -------------------------------------------------------
    @property
    def display_name(self) -> str:
        return self.name.capitalize()

    @property
    def bills_su(self) -> bool:
        return bool(self.su_per_core_hour or any(g.su_per_gpu_hour for g in self.gpus.values()))

    def gpu(self, name: str | None) -> GpuType:
        key = name or self.default_gpu
        if key not in self.gpus:
            raise ClusterError(f"{self.name} has no GPU type {key!r}; choose from {', '.join(self.gpus)}")
        return self.gpus[key]

    def cpus_and_mem(self, gpu: GpuType, n_gpus: int) -> tuple[int, int]:
        """CPU cores and memory (GB) to request alongside ``n_gpus`` of ``gpu``."""
        if self.resource_rule == "fixed_per_gpu":
            cpus, mem = self.fixed_per_gpu
            return cpus * n_gpus, mem * n_gpus
        share = n_gpus / gpu.per_node
        return max(1, int(self.node_cores * share)), max(4, int(self.node_mem_gb * share))

    def storage_dir(self, override: str | None = None) -> Path:
        """Absolute working root. A relative path would not resolve inside a job (it changes directory)."""
        template = override or self.storage_root
        match = re.search(r"\{env:([A-Za-z_][A-Za-z0-9_]*)\}", template)
        if match:
            var = match.group(1)
            if not os.environ.get(var):
                raise ClusterError(
                    f"{self.name}: storage root {template!r} needs ${var}, which is not set. "
                    "Pass --home /path/to/mpd-home or export MPD_HOME.")
            template = template.replace(match.group(0), os.environ[var])
        path = Path(os.path.abspath(os.path.expanduser(template)))
        real = Path(os.path.realpath(path))
        for prefix in self.forbidden_storage_prefixes:
            root = Path(prefix)
            if path.is_relative_to(root) or real.is_relative_to(root):
                raise ClusterError(self._home_text(path, prefix))
        return path

    def _home_text(self, path: Path, prefix: str) -> str:
        limits = [f"{self.home_max_gb:g} GB"] if self.home_max_gb else []
        if self.home_max_files:
            limits.append(f"{self.home_max_files:,} files")
        where = f" ({' / '.join(limits)} on {self.display_name})" if limits else ""
        m = re.fullmatch(r"\{env:([A-Za-z_][A-Za-z0-9_]*)\}(/.*)?", self.storage_root)
        hint = f"export MPD_HOME=${m.group(1)}{m.group(2) or ''}" if m and m.group(1) != "MPD_HOME" \
            else f"pass --home with a path outside {prefix}"
        return f"{path} is under {prefix}{where}. Use scratch: {hint}"

    # ---- policy ----------------------------------------------------------
    def gpu_count_error(self, gpu: GpuType, n_gpus: int) -> str | None:
        if n_gpus < 1:
            return "a GPU job needs at least one GPU"
        if n_gpus > gpu.per_node:
            return (f"{n_gpus} x {gpu.name} does not fit one node "
                    f"({gpu.per_node} per node on {self.name}); multi-node jobs are not generated. "
                    f"Use fewer GPUs (gradient accumulation keeps the global batch) or another GPU type.")
        return None

    def check_gpu_job(self, gpu: GpuType, n_gpus: int, walltime_s: int, *,
                      user_set: bool = False) -> list[tuple[str, str]]:
        """Return (level, message) pairs; level is 'error' or 'warning'."""
        issues: list[tuple[str, str]] = []
        problem = self.gpu_count_error(gpu, n_gpus)
        if problem:
            issues.append(("error", problem))
        if walltime_s > gpu.max_walltime_s:
            fix = (f"Give a --train-time/--score-time/--teacher-time of at most {format_walltime(gpu.max_walltime_s)}."
                   if user_set else "Reduce --updates, use more GPUs (--train-gpus) or a faster GPU (--gpu).")
            issues.append(("error", f"requested {format_walltime(walltime_s)} exceeds the {gpu.name} partition limit "
                                    f"{format_walltime(gpu.max_walltime_s)}; Slurm would hold the job with "
                                    f"PartitionTimeLimit. {fix}"))
        if gpu.unverified:
            issues.append(("warning", f"{self.name}/{gpu.name}: {gpu.unverified}"))
        return issues

    def check_cpu_job(self, walltime_s: int, cpus: int) -> list[tuple[str, str]]:
        issues = []
        if walltime_s > self.cpu_max_walltime_s:
            issues.append(("error", f"CPU job walltime {format_walltime(walltime_s)} exceeds "
                                    f"{format_walltime(self.cpu_max_walltime_s)}"))
        if cpus > self.node_cores:
            issues.append(("error", f"{cpus} cores exceed a {self.node_cores}-core node"))
        return issues

    # ---- accounting ------------------------------------------------------
    def cpu_billing(self, cpus: int, mem_gb: float) -> float:
        """Effective cores charged for ``cpus`` cores and ``mem_gb`` of memory (AMS rule, see the top of this file)."""
        memory_cores = min(self.node_cores, math.ceil(mem_gb / self.su_mem_total_gb * self.node_cores - 1e-9))
        return float(max(cpus, memory_cores))

    def estimate_su(self, *, gpu: GpuType | None, n_gpus: int, cpus: int, mem_gb: float,
                    walltime_s: int, tasks: int = 1) -> float:
        """Upper bound on SUs: (CPU billing + GPU rate x GPUs) x hours x tasks."""
        hours = walltime_s / 3600.0
        billing = self.cpu_billing(cpus, mem_gb)
        if gpu is None:
            rate = billing * self.su_per_core_hour
        else:
            rate = billing + gpu.su_per_gpu_hour * n_gpus
        return rate * hours * tasks


def available_clusters() -> list[str]:
    return sorted(p.name[:-5] for p in resources.files("mpd.clusters").iterdir() if p.name.endswith(".yaml"))


# ---- profile validation helpers ------------------------------------------------------------------------------------
_TOP_FIELDS = {"name", "description", "account", "node", "sbatch_extra", "modules", "login_node", "gpus", "cpu",
               "default_gpu", "default_train_gpus", "array_throttle", "resource_rule", "fixed_per_gpu", "storage"}
_NODE_FIELDS = {"cores", "usable_mem_gb", "su_mem_total_gb"}
_GPU_FIELDS = {"per_node", "mem_gb", "su_per_gpu_hour", "time_factor", "partition", "max_walltime", "max_nodes",
               "unverified", "gres", "bf16"}
_CPU_FIELDS = {"partition", "max_walltime", "su_per_core_hour", "gres"}
_LOGIN_FIELDS = {"max_cores", "max_minutes"}
_STORAGE_FIELDS = {"root", "file_quota_warning", "forbidden_prefixes", "home_max_files", "home_max_gb"}


def _section(value, where: str, allowed: set[str]) -> dict:
    if not isinstance(value, dict):
        raise ClusterError(f"{where}: expected a mapping, got {type(value).__name__}")
    unknown = sorted(str(k) for k in value if k not in allowed)
    if unknown:
        raise ClusterError(f"{where}: unknown field {', '.join(unknown)}; valid fields are {', '.join(sorted(allowed))}")
    return value


def _required(d: dict, key: str, where: str, kind=None):
    if key not in d or d[key] is None:
        raise ClusterError(f"{where}: missing required field {key!r}")
    if kind is not None and not isinstance(d[key], kind):
        raise ClusterError(f"{where}: {key} must be a {kind.__name__}")
    return d[key]


def _num(value, field: str, *, minimum: float | None = None, strict: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ClusterError(f"{field}: expected a number, got {value!r}")
    if minimum is not None and (value <= minimum if strict else value < minimum):
        raise ClusterError(f"{field}: must be {'greater than' if strict else 'at least'} {minimum:g}, got {value!r}")
    return float(value)


def _int(value, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ClusterError(f"{field}: expected a whole number, got {value!r}")
    if value < minimum:
        raise ClusterError(f"{field}: must be at least {minimum}, got {value!r}")
    return value


def _bool(value, field: str) -> bool:
    if not isinstance(value, bool):
        raise ClusterError(f"{field}: expected true or false, got {value!r}")
    return value


def _opt_str(d: dict, key: str, where: str, pattern: re.Pattern | None = None) -> str | None:
    value = d.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or (pattern and not pattern.fullmatch(value)):
        raise ClusterError(f"{where}.{key}: unusable value {value!r}")
    return value


def _name(value, field: str) -> str:
    if not isinstance(value, str) or not NAME_RE.fullmatch(value):
        raise ClusterError(f"{field}: expected letters, digits, '_', '.' or '-', got {value!r}")
    return value


def _walltime(value, field: str) -> int:
    if not isinstance(value, str):
        # YAML reads an unquoted 48:00:00 as a base-60 integer
        raise ClusterError(f"{field}: write the walltime as a quoted string such as \"48:00:00\" (got {value!r})")
    try:
        return parse_walltime(value)
    except ClusterError as exc:
        raise ClusterError(f"{field}: {exc}") from None
