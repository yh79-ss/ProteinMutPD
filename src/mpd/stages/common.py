"""What every stage shares: finding the campaign, its task, and where outputs go.

Campaign layout (all under <home>/runs/<campaign_id>/):

    spec.json resolved.json plan.json tasks/<stage>.json scripts/ logs/ SUBMITTED.json   written by `mpd submit`
    contexts/union.*  contexts/unit__<unit>.*  contexts/eval__<dataset>[__<unit>].*   prepare
    contexts/structures.json                                                           prepare, only if a model reads structure
    teacher/{logits.npy,COMPLETE.json}                                                 teacher
    train/<unit>/{CONFIG.json,dev_curve.jsonl,train_steps.csv,checkpoint_*/,SELECTED.json,COMPLETE.json}
    train/<unit>.partial/{trainable.pt,optimizer.pt,STATE.json}                        train, until the unit is complete
    score/<dataset>/<unit>/shard_<i>_of_<n>.{npz,json}                                 score
    evaluate/{per_assay.csv,summary.json}                                              evaluate

This module must not import torch at import time: the login-node tools reach it too.
"""

from __future__ import annotations

import argparse
import datetime
import functools
import hashlib
import json
import os
import platform
import re
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .. import registry
from ..data.benchmarks import Benchmark, open_benchmark
from ..spec import RunSpec

NO_GPU_EXIT = 31                       # exit code of a GPU stage that cannot use a GPU (message starts MPD_NO_GPU)
SHARD_JSON = re.compile(r"shard_(\d+)_of_(\d+)\.json$")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


# -- GPU guard, numerics switch, provenance -------------------------------------

def _probe_cuda(index: int) -> None:
    """Put one element on cuda:`index` and read it back: what a broken driver or a bad node fails at."""
    import torch
    probe = torch.ones(1, device=torch.device("cuda", index))
    torch.cuda.synchronize(index)
    if float(probe.cpu()[0]) != 1.0:
        raise RuntimeError("a one-element tensor came back wrong from the GPU")


def require_cuda(local_rank: int | None = None) -> None:
    """GPU stages call this first: with MPD_REQUIRE_CUDA=1 (the job script sets it) exit 31 unless CUDA works."""
    if os.environ.get("MPD_REQUIRE_CUDA") != "1":
        return                                        # tests and laptops: unchanged
    import torch
    index = int(os.environ.get("LOCAL_RANK", 0)) if local_rank is None else local_rank       # each DDP rank checks its own GPU
    try:
        if not torch.cuda.is_available():
            reason = "torch.cuda.is_available() is False"
        elif index >= torch.cuda.device_count():
            reason = f"cuda:{index} is needed but {torch.cuda.device_count()} device(s) are visible"
        else:
            _probe_cuda(index)
            return
    except Exception as exc:                       # CUDA runtime errors surface as several exception types
        reason = f"{type(exc).__name__}: {exc}"
    print(f"MPD_NO_GPU: {' '.join(reason.split())} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')} "
          f"torch={torch.__version__} cuda={torch.version.cuda}", file=sys.stderr, flush=True)
    sys.exit(NO_GPU_EXIT)


def configure_torch() -> None:
    """MPD_TF32=1 lets float32 matmuls (the attention scores run in float32) use TF32. Off by default: it changes numbers."""
    if os.environ.get("MPD_TF32") == "1":
        import torch
        torch.set_float32_matmul_precision("high")
        print("MPD_TF32=1: float32 matmuls may use TF32", flush=True)


@functools.lru_cache(maxsize=1)
def _driver_version() -> str | None:
    try:
        done = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None
    words = done.stdout.split() if done.returncode == 0 else []
    return words[0] if words else None


def _package_version(name: str) -> str | None:
    from importlib import metadata
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def provenance() -> dict:
    """Where a result came from: library versions, host, Slurm job and the GPU (if CUDA is in use). Holds no paths."""
    try:
        from .. import __version__
    except ImportError:
        __version__ = None
    info = {"mpd": __version__, "python": platform.python_version(), "platform": platform.platform(),
            "transformers": _package_version("transformers"), "peft": _package_version("peft"),
            "host": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    try:
        import torch
    except ImportError:
        return {**info, "torch": None}
    info.update(torch=torch.__version__, torch_cuda=torch.version.cuda,
                float32_matmul_precision=torch.get_float32_matmul_precision())
    if torch.cuda.is_available():
        info.update(gpu=torch.cuda.get_device_name(torch.cuda.current_device()), gpu_count=torch.cuda.device_count(),
                    driver=_driver_version())
    return info


def peak_gpu_gb() -> float | None:
    """Most GPU memory this process has had allocated (None without CUDA): the measurement the planner only estimates."""
    try:
        import torch
    except ImportError:
        return None
    return round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if torch.cuda.is_available() else None


# -- campaign ---------------------------------------------------------------------

@dataclass
class Campaign:
    root: Path
    spec: RunSpec
    resolved: dict

    @classmethod
    def load(cls, root: Path) -> "Campaign":
        root = Path(root)
        return cls(root, RunSpec.from_json((root / "spec.json").read_text(encoding="utf-8")),
                   json.loads((root / "resolved.json").read_text(encoding="utf-8")))

    @property
    def home(self) -> Path:
        return Path(os.environ.get("MPD_HOME") or self.root.parents[1])

    def tasks(self, stage: str) -> list[dict]:
        return json.loads((self.root / "tasks" / f"{stage}.json").read_text(encoding="utf-8"))

    def benchmark(self, name: str) -> Benchmark:
        return open_benchmark(name, self.home)

    @property
    def units(self) -> list[str]:
        return list(self.spec.units)

    # -- paths ----------------------------------------------------------------
    def stem_union(self) -> Path:
        return self.root / "contexts" / "union"

    def stem_unit(self, unit: str) -> Path:
        return self.root / "contexts" / f"unit__{unit}"

    def stem_eval(self, dataset: str, unit: str | None = None) -> Path:
        """Contexts to score. `unit` is given only when that unit is evaluated on its own assay."""
        return self.root / "contexts" / (f"eval__{dataset}" + (f"__{unit}" if unit else ""))

    def evaluated_on_own_assay(self, unit: str, dataset: str) -> bool:
        """One-assay adapters are scored on their own assay when it belongs to the evaluation dataset."""
        return self.spec.scope == "one" and unit != "base" and dataset == self.spec.train_data

    def eval_assay_ids(self, unit: str, dataset: str, all_ids: list[str]) -> list[str]:
        return [unit] if self.evaluated_on_own_assay(unit, dataset) else all_ids

    def stem_eval_for(self, unit: str, dataset: str) -> Path:
        return self.stem_eval(dataset, unit if self.evaluated_on_own_assay(unit, dataset) else None)

    def stem_structures(self) -> Path:
        return self.root / "contexts" / "structures.json"

    @property
    def teacher_logits(self) -> Path:
        return self.root / "teacher" / "logits.npy"

    @property
    def teacher_complete(self) -> Path:
        return self.root / "teacher" / "COMPLETE.json"

    def train_dir(self, unit: str) -> Path:
        return self.root / "train" / unit

    def score_dir(self, dataset: str, unit: str) -> Path:
        return self.root / "score" / dataset / unit

    @property
    def evaluate_dir(self) -> Path:
        return self.root / "evaluate"

    # -- structures (SaProt) ----------------------------------------------------
    def needs_structure(self, role: str) -> bool:
        """Does the campaign's "teacher" or "student" read protein structure?"""
        return registry.get(getattr(self.spec, role)).needs_structure

    def load_structures(self, dataset: str | None = None):
        """contexts/structures.json as `dataset` sees it, or None if `prepare` wrote none.

        Name the dataset: a sequence two benchmarks share has a string in each, and `None` gives the base set (the
        training dataset's), which is wrong for scoring any other dataset."""
        if not self.stem_structures().is_file():
            return None
        from ..structure.set import StructureSet
        return StructureSet.load(self.stem_structures()).scoped(dataset)

    def structures_digest(self) -> str | None:
        """Digest of structures.json as `prepare` froze it (the one MANIFEST.json records), None if there is none."""
        if not self.stem_structures().is_file():
            return None
        from ..structure.set import StructureSet
        return StructureSet.load(self.stem_structures()).digest()

    def structures_for(self, role: str, dataset: str):
        """What `model.encode` takes for `role`: the structure strings of `dataset` if that model reads structure, else None."""
        if not self.needs_structure(role):
            return None
        structures = self.load_structures(dataset)
        if structures is None:
            raise FileNotFoundError(f"{getattr(self.spec, role)} reads protein structure, but {self.stem_structures()} does not "
                                    f"exist. The prepare stage writes it: delete {self.root / 'contexts' / 'MANIFEST.json'} and "
                                    "repeat your `mpd submit` command with --force to run that stage again.")
        return structures

    # -- score shards ---------------------------------------------------------
    @functools.cached_property
    def _shard_plan(self) -> dict[tuple[str, str], set[int]]:
        """(dataset, unit) -> the shard counts named by that pair's score tasks (one value in a sound plan)."""
        counts: dict[tuple[str, str], set[int]] = {}
        for stage in ("score-base", "score"):
            try:
                tasks = self.tasks(stage)
            except (OSError, ValueError):
                continue
            for t in tasks:
                if {"dataset", "unit", "num_shards"} <= t.keys():
                    counts.setdefault((t["dataset"], t["unit"]), set()).add(int(t["num_shards"]))
        return counts

    def expected_shards(self, dataset: str, unit: str) -> int | None:
        """Score shards the plan wants for (dataset, unit); files for any other count are leftovers. None if it recorded none."""
        # the pair's own tasks first: a unit scored on its own small table can have fewer shards than the dataset default
        planned = self._shard_plan.get((dataset, unit), set())
        if len(planned) == 1:
            return next(iter(planned))
        per_dataset = self.resolved.get("score_shards")
        if isinstance(per_dataset, dict) and per_dataset.get(dataset) is not None:
            return int(per_dataset[dataset])
        return None


def shard_name(shard: int, num: int) -> str:
    return f"shard_{shard:03d}_of_{num:03d}"


def stage_main(run, description: str, argv: list[str] | None = None):
    """Parse the arguments the sbatch script passes and call `run(campaign, task)`.

    The script always passes --task-index (a one-task chunk of a longer list passes its offset); a one-task stage may omit it."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--campaign-dir", type=Path, required=True)
    ap.add_argument("--stage", required=True, help="stage name as in tasks/<stage>.json")
    ap.add_argument("--task-index", type=int, help="position in tasks/<stage>.json; required when the stage has several tasks")
    args = ap.parse_args(argv)
    campaign = Campaign.load(args.campaign_dir)
    tasks = campaign.tasks(args.stage)
    index = args.task_index
    if index is None:
        if len(tasks) != 1:
            ap.error(f"stage {args.stage} has {len(tasks)} tasks and no --task-index was given; the sbatch script must pass it "
                     "(array jobs: $((SLURM_ARRAY_TASK_ID + offset)); a one-task chunk of a longer list: its offset)")
        index = 0
    if not 0 <= index < len(tasks):
        ap.error(f"--task-index {index} outside 0..{len(tasks) - 1}")
    return run(campaign, tasks[index])
