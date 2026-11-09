"""Stage `teacher` (GPU): run the frozen teacher once over every training context and store its AA20 logits.

The student never sees the teacher at training time, only this table (cached-target distillation). A finished table
is reused: the stage does nothing when teacher/COMPLETE.json names this teacher and these contexts (and, for a teacher
that reads protein structure, these structure strings), so resubmitting a campaign does not pay for the teacher twice.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .. import students
from ..contexts import ContextTable
from .common import (Campaign, configure_torch, file_sha256, peak_gpu_gb, provenance, require_cuda, stage_main,
                     write_json)

BATCH = 8


def device() -> torch.device:
    return torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")


def aa20_logits(model, table: ContextTable, structures=None, batch: int = BATCH) -> np.ndarray:
    """[N, 20] float32. Contexts are processed longest-first so padding stays small."""
    model.encode(table, structures)
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


def teacher_state(c: Campaign, union: ContextTable) -> tuple[dict | None, str]:
    """(COMPLETE.json, "") when teacher/ holds this campaign's teacher output for `union`, else (None, why not)."""
    if not c.teacher_complete.is_file():
        return None, "teacher/COMPLETE.json does not exist"
    try:
        info = json.loads(c.teacher_complete.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, "teacher/COMPLETE.json cannot be read"
    if info.get("status") != "PASS":
        return None, "teacher/COMPLETE.json does not say PASS"
    if info.get("teacher") != c.spec.teacher:
        return None, f"it was computed by {info.get('teacher')}, this campaign distills {c.spec.teacher}"
    if info.get("context_digest") != union.digest():
        return None, (f"it was computed for contexts {str(info.get('context_digest'))[:12]}, "
                      f"this campaign's union table is {union.digest()[:12]}")
    if c.needs_structure("teacher"):               # a structure-aware teacher's table depends on the strings it was given
        now = c.structures_digest()
        if info.get("structures_digest") != now:
            return None, (f"it was computed with structures {str(info.get('structures_digest'))[:12]}, this campaign's "
                          f"structures.json is {'missing' if now is None else now[:12]}")
    if not c.teacher_logits.is_file():
        return None, "teacher/logits.npy does not exist"
    if info.get("logits_sha256") != file_sha256(c.teacher_logits):
        return None, "teacher/logits.npy does not match the sha256 in COMPLETE.json"
    return info, ""


def save_logits(path: Path, logits: np.ndarray) -> None:
    """Write through a temporary name and rename, so an interrupted job never leaves a half-written table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, logits)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)


def run(c: Campaign, task: dict) -> None:
    require_cuda()
    configure_torch()
    table = ContextTable.load(c.stem_union())
    # the training dataset's own strings; read before anything is deleted or loaded, so a campaign without them fails at once
    structures = c.structures_for("teacher", c.spec.train_data)
    done, why = teacher_state(c, table)
    if done is not None:
        print(f"teacher {c.spec.teacher}: {len(table)} contexts already computed (digest {table.digest()[:12]}); "
              "skipping. Delete teacher/COMPLETE.json to force a recomputation.", flush=True)
        return
    if c.teacher_complete.is_file():
        print(f"teacher {c.spec.teacher}: recomputing, {why}", flush=True)
        c.teacher_complete.unlink()                # a crash below must not leave a PASS for the old table
    started = time.monotonic()
    model = students.load(c.spec.teacher, device())
    out = aa20_logits(model, table, structures)
    save_logits(c.teacher_logits, out)
    record = {
        "status": "PASS", "teacher": c.spec.teacher, "contexts": len(table), "context_digest": table.digest(),
        "logits_sha256": file_sha256(c.teacher_logits), "seconds": round(time.monotonic() - started, 1),
        "dtype": str(next(model.module.parameters()).dtype), "peak_gpu_memory_gb": peak_gpu_gb(),
        "order": "AA20 = ACDEFGHIKLMNPQRSTVWY, raw masked-position logits", "provenance": provenance()}
    if structures is not None:
        record["structures_digest"] = c.structures_digest()
    write_json(c.teacher_complete, record)
    print(f"teacher {c.spec.teacher}: {len(table)} contexts in {time.monotonic() - started:.0f}s", flush=True)


def main(argv=None):
    return stage_main(run, __doc__, argv)


if __name__ == "__main__":
    main()
