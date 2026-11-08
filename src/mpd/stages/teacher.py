"""Stage `teacher` (GPU): run the frozen teacher once over every training context and store its AA20 logits.

The student never sees the teacher at training time, only this table (cached-target distillation).
"""

from __future__ import annotations

import time

import numpy as np
import torch

from .. import students
from ..contexts import ContextTable
from .common import Campaign, file_sha256, stage_main, write_json

BATCH = 8


def device() -> torch.device:
    return torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")


def aa20_logits(model, table: ContextTable, batch: int = BATCH) -> np.ndarray:
    """[N, 20] float32. Contexts are processed longest-first so padding stays small."""
    model.encode(table)
    lengths = np.asarray([min(len(table.sequences[s]), 1022) for s in table.seq_id])
    order = np.argsort(-lengths, kind="stable")
    out = np.zeros((len(table), 20), dtype=np.float32)
    model.module.eval()
    with torch.no_grad():
        for start in range(0, len(order), batch):
            rows = order[start:start + batch]
            out[rows] = model.logits(table, rows).cpu().numpy()
    if not np.isfinite(out).all():
        raise RuntimeError("the teacher produced non-finite logits")
    return out


def run(c: Campaign, task: dict) -> None:
    table = ContextTable.load(c.stem_union())
    started = time.monotonic()
    model = students.load(c.spec.teacher, device())
    out = aa20_logits(model, table)
    path = c.teacher_logits
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("logits.tmp.npy")
    np.save(tmp, out)
    tmp.replace(path)
    write_json(path.parent / "COMPLETE.json", {
        "status": "PASS", "teacher": c.spec.teacher, "contexts": len(table), "context_digest": table.digest(),
        "logits_sha256": file_sha256(path), "seconds": round(time.monotonic() - started, 1),
        "order": "AA20 = ACDEFGHIKLMNPQRSTVWY, raw masked-position logits"})
    print(f"teacher {c.spec.teacher}: {len(table)} contexts in {time.monotonic() - started:.0f}s", flush=True)


def main(argv=None):
    return stage_main(run, __doc__, argv)


if __name__ == "__main__":
    main()
