"""Stage `score` (GPU): write the student's AA20 logits at every evaluation context of one shard.

Task: {unit, dataset, shard, num_shards}. unit "base" is the frozen student; any other unit loads the checkpoint
its training run selected. No labels are read; `evaluate` turns these logits into variant scores.
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch

from .. import students
from ..contexts import ContextTable
from .common import Campaign, file_sha256, stage_main, write_json
from .teacher import aa20_logits, device


def selected_adapter(c: Campaign, unit: str):
    if unit == "base":
        return None, None
    train = c.train_dir(unit)
    if not (train / "COMPLETE.json").is_file():
        raise FileNotFoundError(f"training of {unit} is not complete ({train})")
    chosen = json.loads((train / "SELECTED.json").read_text(encoding="utf-8"))
    return train / f"checkpoint_{chosen['step']:05d}", chosen


def shard_path(c: Campaign, dataset: str, unit: str, shard: int, num: int):
    return c.score_dir(dataset, unit) / f"shard_{shard:03d}_of_{num:03d}.npz"


def run(c: Campaign, task: dict) -> None:
    unit, dataset, shard, num = task["unit"], task["dataset"], task["shard"], task["num_shards"]
    target = shard_path(c, dataset, unit, shard, num)
    if target.with_suffix(".json").is_file():
        print(f"{target.name}: already done", flush=True)
        return
    table = ContextTable.load(c.stem_eval_for(unit, dataset))
    rows = np.arange(len(table))[shard::num]
    adapter, chosen = selected_adapter(c, unit)
    started = time.monotonic()
    model = students.load(c.spec.student, device(), adapter=adapter)
    # same longest-first batching as the teacher stage, on this shard's rows only
    sub = table.subset(rows)
    scores = aa20_logits(model, sub, batch=8)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp.npz")
    np.savez(tmp, rows=rows.astype(np.int64), scores=scores)
    tmp.replace(target)
    write_json(target.with_suffix(".json"), {
        "unit": unit, "dataset": dataset, "shard": shard, "num_shards": num, "contexts": int(len(rows)),
        "of": int(len(table)), "adapter_step": None if chosen is None else chosen["step"],
        "scores_sha256": file_sha256(target), "seconds": round(time.monotonic() - started, 1),
        "dms_labels_opened": False})
    print(f"scored {len(rows)} contexts ({unit}, {dataset}, shard {shard + 1}/{num})", flush=True)


def main(argv=None):
    return stage_main(run, __doc__, argv)


if __name__ == "__main__":
    main()
