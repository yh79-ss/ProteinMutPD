"""Stage `train` (GPU): LoRA-distill the student on cached teacher logits. One task per training unit.

The loss reads only the teacher table and, for the score objective, which variants were measured. No DMS
score is read anywhere in this stage; checkpoints are chosen on a position-held-out development split of the
teacher targets.
"""

from __future__ import annotations

import contextlib
import csv
import json
import math
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from scipy.stats import spearmanr

from .. import students
from ..contexts import ContextTable
from ..objectives import direct_loss, flatten_variants, score_from_logits, score_loss
from .common import Campaign, stage_main, write_json

DEV_BATCH = 8


# -- recipe pieces (pure) -----------------------------------------------------

def learning_rate(step: int, total: int, peak: float, warmup_fraction: float = 0.05, min_ratio: float = 0.1) -> float:
    """Linear warm-up for 5% of the updates, then cosine decay to `min_ratio` x peak."""
    warmup = max(1, int(round(total * warmup_fraction)))
    if step <= warmup:
        return peak * step / warmup
    phase = (step - warmup) / max(1, total - warmup)
    return peak * (min_ratio + (1 - min_ratio) * (1 + math.cos(math.pi * phase)) / 2)


def monitor_steps(updates: int) -> list[int]:
    """Three evenly spaced checkpoints ending at the last update (701, 1402, 2103 for 2,103 updates)."""
    return sorted({max(1, round(updates * k / 3)) for k in (1, 2, 3)})


def update_stream(train_rows: np.ndarray, updates: int, global_batch: int, seed: int) -> list[np.ndarray]:
    """`updates` batches of `global_batch` rows, drawn from repeated shuffles of the training rows."""
    rng = np.random.default_rng(seed)
    need = updates * global_batch
    pieces, count = [], 0
    while count < need:
        piece = rng.permutation(train_rows)
        pieces.append(piece)
        count += len(piece)
    return list(np.split(np.concatenate(pieces)[:need], updates))


def select_checkpoint(curve: list[dict]) -> dict:
    """Lowest development loss among the checkpoints (steps > 0), preferring ones that beat the untrained start."""
    start = next(r for r in curve if r["step"] == 0)
    cands = [r for r in curve if r["step"] > 0 and math.isfinite(r["dev_loss"])]
    if not cands:
        raise RuntimeError("no finite development checkpoint")
    better = [r for r in cands if r["dev_loss"] < start["dev_loss"]]
    best = dict(min(better or cands, key=lambda r: (r["dev_loss"], r["step"])))
    best["start_dev_loss"] = start["dev_loss"]
    best["improved_over_start"] = bool(better)
    return best


# -- objective-specific batch loss and development metrics --------------------

class Objective:
    def __init__(self, c: Campaign, table: ContextTable, teacher: torch.Tensor, device):
        self.kind, self.table, self.teacher, self.device = c.spec.objective, table, teacher, device
        self.kl_weight = float(c.resolved["kl_weight"])
        self.wt = torch.as_tensor(table.wt.astype(np.int64), device=device)

    def loss(self, student: torch.Tensor, rows: np.ndarray):
        target = self.teacher[torch.as_tensor(rows, device=self.device)]
        if self.kind == "direct_logits":
            loss, parts = direct_loss(student, target, self.kl_weight)
            return loss, parts
        row, mt = flatten_variants(self.table.var_ptr, self.table.var_mt, rows)
        wt = self.wt[torch.as_tensor(rows, device=self.device)]
        loss, parts, _, _ = score_loss(student, target, wt, torch.as_tensor(row, device=self.device),
                                       torch.as_tensor(mt.astype(np.int64), device=self.device))
        return loss, parts

    @torch.no_grad()
    def dev_metrics(self, model, dev_rows: np.ndarray) -> dict:
        model.module.eval()
        huber = kl = 0.0
        n_items = 0
        pred_parts, target_parts = [], []
        for start in range(0, len(dev_rows), DEV_BATCH):
            rows = dev_rows[start:start + DEV_BATCH]
            student = model.logits(self.table, rows)
            target = self.teacher[torch.as_tensor(rows, device=self.device)]
            if self.kind == "direct_logits":
                huber += float(F.smooth_l1_loss(student, target, beta=1.0, reduction="sum"))
                kl += float(F.kl_div(F.log_softmax(student, dim=-1), F.softmax(target, dim=-1), reduction="sum"))
                n_items += len(rows)
                pred_parts.append(student.cpu().numpy().ravel())
                target_parts.append(target.cpu().numpy().ravel())
            else:
                row, mt = flatten_variants(self.table.var_ptr, self.table.var_mt, rows)
                wt = self.wt[torch.as_tensor(rows, device=self.device)]
                r = torch.as_tensor(row, device=self.device)
                m = torch.as_tensor(mt.astype(np.int64), device=self.device)
                gs, gt = score_from_logits(student, wt, r, m), score_from_logits(target, wt, r, m)
                huber += float(F.smooth_l1_loss(gs, gt, beta=1.0, reduction="sum"))
                n_items += len(mt)
                pred_parts.append(gs.cpu().numpy())
                target_parts.append(gt.cpu().numpy())
        pred, target = np.concatenate(pred_parts), np.concatenate(target_parts)
        rho = float(spearmanr(pred, target).statistic) if len(pred) > 2 else float("nan")
        if self.kind == "direct_logits":
            h, k = huber / (n_items * 20), kl / n_items
            out = {"dev_huber": h, "dev_kl": k, "dev_loss": h + self.kl_weight * k}
        else:
            out = {"dev_huber": huber / n_items, "dev_loss": huber / n_items}
        model.module.train()
        return {**out, "dev_rho_ST": rho, "dev_contexts": int(len(dev_rows)), "dev_items": int(n_items)}


# -- the stage -----------------------------------------------------------------

def run(c: Campaign, task: dict) -> None:
    unit, spec, r = task["unit"], c.spec, c.resolved
    rank, local_rank, world = (int(os.environ.get(k, d)) for k, d in (("RANK", 0), ("LOCAL_RANK", 0), ("WORLD_SIZE", 1)))
    if world != r["world_size"]:
        raise RuntimeError(f"launched with {world} process(es) but the plan has {r['world_size']}")
    out = c.train_dir(unit)
    if (out / "COMPLETE.json").is_file():
        print(f"{unit}: already complete", flush=True)
        return
    cuda = torch.cuda.is_available()
    device = torch.device("cuda", local_rank) if cuda else torch.device("cpu")
    if cuda:
        torch.cuda.set_device(local_rank)
    if world > 1:
        dist.init_process_group("nccl" if cuda else "gloo")
    random.seed(spec.seed)
    np.random.seed(spec.seed)
    torch.manual_seed(spec.seed)
    if cuda:
        torch.cuda.manual_seed_all(spec.seed)

    if rank == 0:
        if out.exists():
            aside = out.with_name(f"{out.name}.incomplete-{int(time.time())}")
            out.rename(aside)
            print(f"moved an incomplete earlier attempt to {aside.name}", flush=True)
        out.mkdir(parents=True)
    if world > 1:
        dist.barrier()

    table = ContextTable.load(c.stem_unit(unit))
    teacher = torch.as_tensor(np.load(c.teacher_logits, mmap_mode="r")[table.union_index], dtype=torch.float32, device=device)
    dev = table.dev_mask(r["dev_fraction"])
    train_rows, dev_rows = np.flatnonzero(~dev), np.flatnonzero(dev)
    if len(train_rows) == 0 or len(dev_rows) == 0:
        raise RuntimeError(f"unit {unit}: {len(train_rows)} training and {len(dev_rows)} development contexts; need both")

    model = students.load(spec.student, device, lora={"rank": r["lora_rank"], "alpha": r["lora_alpha"]})
    model.encode(table)
    trainable = model.trainable()
    if not trainable:
        raise RuntimeError("no trainable LoRA parameters")
    forward = (torch.nn.parallel.DistributedDataParallel(model.module, device_ids=[local_rank] if cuda else None,
                                                         find_unused_parameters=False) if world > 1 else model.module)
    optimizer = torch.optim.AdamW(trainable, lr=r["peak_lr"], betas=(0.9, 0.999), eps=1e-8, weight_decay=r["weight_decay"])
    objective = Objective(c, table, teacher, device)
    steps = update_stream(train_rows, r["updates"], r["global_batch"], spec.seed)
    monitors = set(monitor_steps(r["updates"]))
    accumulation = r["accumulation"]
    started = time.monotonic()

    if rank == 0:
        write_json(out / "CONFIG.json", {
            "unit": unit, "student": spec.student, "teacher": spec.teacher, "objective": spec.objective,
            "train_contexts": int(len(train_rows)), "dev_contexts": int(len(dev_rows)),
            "variants": None if table.var_mt is None else int(len(table.var_mt)), "world_size": world, "monitors": sorted(monitors),
            "context_digest": table.digest(), "dms_labels_opened": False, **r})
        first = {**objective.dev_metrics(model, dev_rows), "step": 0, "elapsed_seconds": 0.0}
        with open(out / "dev_curve.jsonl", "w", encoding="utf-8") as fh:
            fh.write(json.dumps(first) + "\n")
        print(json.dumps(first), flush=True)
        log = open(out / "train_steps.csv", "w", newline="", encoding="utf-8")
        writer = csv.DictWriter(log, fieldnames=["step", "loss", "huber", "lr", "grad_norm", "elapsed_seconds"])
        writer.writeheader()
    if world > 1:
        dist.barrier()

    model.module.train()
    for step, batch in enumerate(steps, start=1):
        lr = learning_rate(step, len(steps), r["peak_lr"], r["warmup_fraction"], r["min_lr_ratio"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        mine = np.split(batch[rank::world], accumulation)
        total = huber = 0.0
        for k, rows in enumerate(mine):
            sync = forward.no_sync() if world > 1 and k < len(mine) - 1 else contextlib.nullcontext()
            with sync:
                loss, parts = objective.loss(model.logits(table, rows, forward=forward), rows)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"non-finite loss at update {step}")
                (loss / accumulation).backward()
            total += float(loss.detach()) / accumulation
            huber += float(parts["huber"]) / accumulation
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, r["grad_clip"]))
        if not math.isfinite(grad_norm):
            raise RuntimeError(f"non-finite gradient norm at update {step}")
        optimizer.step()
        if rank == 0:
            writer.writerow({"step": step, "loss": total, "huber": huber, "lr": lr, "grad_norm": grad_norm,
                             "elapsed_seconds": round(time.monotonic() - started, 1)})
            if step % 20 == 0:
                log.flush()
        if step in monitors:
            if world > 1:
                dist.barrier()
            if rank == 0:
                metrics = {**objective.dev_metrics(model, dev_rows), "step": step,
                           "elapsed_seconds": round(time.monotonic() - started, 1)}
                with open(out / "dev_curve.jsonl", "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(metrics) + "\n")
                model.save_adapter(out / f"checkpoint_{step:05d}")
                log.flush()
                print(json.dumps(metrics), flush=True)
            if world > 1:
                dist.barrier()

    if rank == 0:
        log.close()
        curve = [json.loads(line) for line in open(out / "dev_curve.jsonl", encoding="utf-8")]
        chosen = select_checkpoint(curve)
        write_json(out / "SELECTED.json", chosen)
        for ckpt in out.glob("checkpoint_*"):
            if ckpt.name != f"checkpoint_{chosen['step']:05d}" and os.environ.get("MPD_KEEP_ALL_CHECKPOINTS") != "1":
                shutil.rmtree(ckpt)
        write_json(out / "COMPLETE.json", {"status": "PASS", "updates": len(steps), "selected_step": chosen["step"],
                                           "seconds": round(time.monotonic() - started, 1)})
        print(f"{unit}: selected update {chosen['step']} (dev loss {chosen['dev_loss']:.4f}, start {chosen['start_dev_loss']:.4f})",
              flush=True)
    if world > 1:
        dist.destroy_process_group()


def main(argv=None):
    return stage_main(run, __doc__, argv)


if __name__ == "__main__":
    main()
