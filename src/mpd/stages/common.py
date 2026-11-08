"""What every stage shares: finding the campaign, its task, and where outputs go.

Campaign layout (all under <home>/runs/<campaign_id>/):

    spec.json resolved.json tasks/<stage>.json scripts/ logs/   written by `mpd submit`
    contexts/union.*  contexts/unit__<unit>.*  contexts/eval__<dataset>[__<unit>].*   prepare
    teacher/logits.npy                                                                 teacher
    train/<unit>/{CONFIG.json,dev_curve.jsonl,train_steps.csv,checkpoint_*/,SELECTED.json,COMPLETE.json}
    score/<dataset>/<unit>/shard_<i>_of_<n>.{npz,json}                                 score
    evaluate/{per_assay.csv,summary.json}                                              evaluate
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from ..data.benchmarks import Benchmark, open_benchmark
from ..spec import RunSpec


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

    @property
    def teacher_logits(self) -> Path:
        return self.root / "teacher" / "logits.npy"

    def train_dir(self, unit: str) -> Path:
        return self.root / "train" / unit

    def score_dir(self, dataset: str, unit: str) -> Path:
        return self.root / "score" / dataset / unit

    @property
    def evaluate_dir(self) -> Path:
        return self.root / "evaluate"


def stage_main(run, description: str, argv: list[str] | None = None):
    """Parse the arguments the generated sbatch script passes and call `run(campaign, task)`."""
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--campaign-dir", type=Path, required=True)
    ap.add_argument("--stage", required=True, help="stage name as in tasks/<stage>.json")
    ap.add_argument("--task-index", type=int)
    args = ap.parse_args(argv)
    campaign = Campaign.load(args.campaign_dir)
    tasks = campaign.tasks(args.stage)
    if args.task_index is None:
        if len(tasks) != 1:
            ap.error(f"stage {args.stage} has {len(tasks)} tasks; give --task-index")
        index = 0
    else:
        index = args.task_index
    if not 0 <= index < len(tasks):
        ap.error(f"--task-index {index} outside 0..{len(tasks) - 1}")
    return run(campaign, tasks[index])
