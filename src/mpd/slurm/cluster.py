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

# Calibrated on the three SU examples in the HPRC Grace primer (1 core/4 GB/1.5 h = 1.5 SU;
# 24 cores/48 GB/6.5 h = 156 SU; 24 cores + 1 A100/180 GB/96 h = 9,312 SU). sbatch on Grace
# prints the authoritative "Requested SUs", which `mpd submit` echoes.
GB_PER_BILLED_CORE = 7.2


class ClusterError(ValueError):
    pass


def parse_walltime(value) -> int:
    """Slurm walltime to seconds: M, M:S, H:M:S, D-H, D-H:M, D-H:M:S."""
    text = str(value).strip()
    days = 0
    if "-" in text:
        d, text = text.split("-", 1)
        days = int(d)
        parts = [int(p) for p in text.split(":")]
        hours, minutes, seconds = (parts + [0, 0])[:3]
    else:
        parts = [int(p) for p in text.split(":")]
        if len(parts) == 1:
            hours, minutes, seconds = 0, parts[0], 0
        elif len(parts) == 2:
            hours, minutes, seconds = 0, parts[0], parts[1]
        else:
            hours, minutes, seconds = parts[:3]
    return ((days * 24 + hours) * 60 + minutes) * 60 + seconds


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

    # ---- loading ---------------------------------------------------------
    @classmethod
    def load(cls, name_or_path: str) -> "Cluster":
        path = Path(name_or_path)
        if path.suffix in (".yaml", ".yml") and path.is_file():
            text = path.read_text(encoding="utf-8")
        else:
            try:
                text = resources.files("mpd.clusters").joinpath(f"{name_or_path}.yaml").read_text(encoding="utf-8")
            except FileNotFoundError:
                raise ClusterError(
                    f"unknown cluster {name_or_path!r}; use one of {', '.join(available_clusters())} "
                    "or a path to a profile YAML") from None
        return cls.from_dict(yaml.safe_load(text))

    @classmethod
    def from_dict(cls, d: dict) -> "Cluster":
        gpus = {}
        for key, g in d["gpus"].items():
            gpus[key] = GpuType(
                name=key, per_node=int(g["per_node"]), mem_gb=float(g["mem_gb"]),
                su_per_gpu_hour=float(g.get("su_per_gpu_hour", 0)), partition=g.get("partition"),
                max_walltime_s=parse_walltime(g["max_walltime"]), max_nodes=g.get("max_nodes"),
                time_factor=float(g.get("time_factor", 1.0)), unverified=g.get("unverified"))
        fixed = d.get("fixed_per_gpu")
        return cls(
            name=d["name"], description=d.get("description", ""), account=d.get("account"),
            node_cores=int(d["node"]["cores"]), node_mem_gb=int(d["node"]["usable_mem_gb"]),
            sbatch_extra=tuple(d.get("sbatch_extra", ())), modules=tuple(d.get("modules", ())),
            gpus=gpus, cpu_partition=d.get("cpu", {}).get("partition"),
            cpu_max_walltime_s=parse_walltime(d.get("cpu", {}).get("max_walltime", "1-00:00:00")),
            su_per_core_hour=float(d.get("cpu", {}).get("su_per_core_hour", 0)),
            default_gpu=d["default_gpu"], array_throttle=int(d.get("array_throttle", 4)),
            resource_rule=d.get("resource_rule", "proportional_to_node"),
            fixed_per_gpu=(int(fixed["cpus"]), int(fixed["mem_gb"])) if fixed else None,
            storage_root=d.get("storage", {}).get("root", "{env:MPD_HOME}"),
            login_max_cores=int(d.get("login_node", {}).get("max_cores", 8)),
            login_max_minutes=int(d.get("login_node", {}).get("max_minutes", 60)))

    # ---- resources -------------------------------------------------------
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
        template = override or self.storage_root
        match = re.search(r"\{env:([A-Za-z_][A-Za-z0-9_]*)\}", template)
        if match:
            var = match.group(1)
            if var not in os.environ:
                raise ClusterError(
                    f"{self.name}: storage root {template!r} needs ${var}, which is not set. "
                    "Pass --home /path/to/mpd-home or export MPD_HOME.")
            template = template.replace(match.group(0), os.environ[var])
        return Path(template).expanduser()

    # ---- policy ----------------------------------------------------------
    def check_gpu_job(self, gpu: GpuType, n_gpus: int, walltime_s: int) -> list[tuple[str, str]]:
        """Return (level, message) pairs; level is 'error' or 'warning'."""
        issues: list[tuple[str, str]] = []
        if n_gpus < 1:
            issues.append(("error", "a GPU job needs at least one GPU"))
        if n_gpus > gpu.per_node:
            issues.append(("error", f"{n_gpus} x {gpu.name} does not fit one node "
                                    f"({gpu.per_node} per node on {self.name}); multi-node jobs are not generated. "
                                    f"Use fewer GPUs (gradient accumulation keeps the global batch) or another GPU type."))
        if walltime_s > gpu.max_walltime_s:
            issues.append(("error", f"requested {format_walltime(walltime_s)} exceeds the {gpu.name} partition limit "
                                    f"{format_walltime(gpu.max_walltime_s)}; Slurm would hold the job with "
                                    f"PartitionTimeLimit. Reduce --updates or add GPUs."))
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
        return max(float(cpus), math.ceil(mem_gb / GB_PER_BILLED_CORE))

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
