"""Stage `train` (GPU): LoRA-distill the student on cached teacher logits. One task per training unit.

The loss reads only the teacher table and, for the score objective, which variants were measured. No DMS
label is read anywhere in this stage; checkpoints are chosen on a position-held-out development split of the
teacher targets.

A job killed by its walltime is resumed by running the stage again: see "resumable state" below. A student that reads
protein structure (SaProt) is given the strings `prepare` froze in contexts/structures.json; their digest is part of the
run's configuration, so a checkpoint made with other strings is not resumed.
"""

from __future__ import annotations

import contextlib
import csv
import datetime
import hashlib
import inspect
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
from .common import (Campaign, configure_torch, file_sha256, peak_gpu_gb, provenance, require_cuda, stage_main,
                     write_json)
from .teacher import teacher_state

DEV_BATCH = 8
CKPT_EVERY = 200                                   # updates between resumable checkpoints; MPD_CKPT_EVERY overrides, 0 turns them off
GROUP_TIMEOUT = datetime.timedelta(minutes=60)     # rank 0 evaluates alone while the other ranks wait for it
STATE_SCHEMA = 1
STATE_FILES = ("trainable.pt", "optimizer.pt")


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


# -- resumable state ------------------------------------------------------------
#
# Every `ckpt_every` updates rank 0 writes train/<unit>.partial/: trainable.pt (the LoRA tensors), optimizer.pt (AdamW
# moments and step counts) and STATE.json (update, configuration digest, size and sha256 of both files; written last).
# It is built under a temporary name and renamed into place, and only the latest is kept. That is all a rerun needs to
# continue exactly where the run stopped: the learning rate is a function of the update number, the update stream a
# function of (rows, updates, batch, seed), and the loop draws no random number (LoRA and model dropout are 0), so no RNG
# state is saved; a student with dropout would have to add it. The monitored adapters (checkpoint_*) and the two logs
# stay in train/<unit>/ and are cut back to the resumed update.
# Disk: 3 x (LoRA parameters) x 4 bytes, about 32 MB for a 650M student and 71 MB for a 3B one, once (the previous copy is
# deleted when the new one is in place).

def checkpoint_interval(arg: int | None) -> int:
    """Updates between resumable checkpoints: `arg`, else $MPD_CKPT_EVERY, else CKPT_EVERY. 0 turns them off."""
    raw = arg if arg is not None else os.environ.get("MPD_CKPT_EVERY", "").strip() or CKPT_EVERY
    try:
        value = int(raw)
    except ValueError:
        value = -1
    if value < 0:
        raise SystemExit(f"MPD_CKPT_EVERY must be a whole number of updates, 0 or more (0 turns resumable checkpoints off); "
                         f"got {raw!r}")
    return value


def partial_dir(out: Path) -> Path:
    return out.with_name(out.name + ".partial")


def partial_leftovers(out: Path) -> list[Path]:
    final = partial_dir(out)
    return [final.with_name(final.name + s) for s in (".old", ".tmp")]


def discard_partial(out: Path) -> None:
    for folder in (partial_dir(out), *partial_leftovers(out)):
        shutil.rmtree(folder, ignore_errors=True)


def config_digest(config: dict) -> str:
    """Identity of a run: everything that decides its numbers (the CONFIG record without provenance and the digest itself)."""
    body = {k: v for k, v in config.items() if k not in ("provenance", "config_digest", "nonce")}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()


def _dump(path: Path, obj) -> None:
    with open(path, "wb") as fh:
        torch.save(obj, fh)
        fh.flush()
        os.fsync(fh.fileno())


def save_partial(out: Path, *, step: int, total: int, digest: str, names: list[str], trainable, optimizer,
                 elapsed: float, resumed_from: list[int]) -> None:
    final = partial_dir(out)
    old, tmp = partial_leftovers(out)
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    try:
        _dump(tmp / "trainable.pt", {"names": names, "tensors": [p.detach().cpu() for p in trainable]})
        _dump(tmp / "optimizer.pt", optimizer.state_dict())
        write_json(tmp / "STATE.json", {
            "schema": STATE_SCHEMA, "step": step, "updates": total, "config_digest": digest, "elapsed_seconds": elapsed,
            "resumed_from": resumed_from,
            "files": {n: {"bytes": (tmp / n).stat().st_size, "sha256": file_sha256(tmp / n)} for n in STATE_FILES}})
        shutil.rmtree(old, ignore_errors=True)
        if final.exists():
            final.rename(old)
        tmp.rename(final)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    shutil.rmtree(old, ignore_errors=True)


def check_partial(folder: Path, digest: str, total: int) -> dict:
    """STATE.json of a resumable checkpoint, or ValueError saying why it cannot be used for this run."""
    try:
        state = json.loads((folder / "STATE.json").read_text(encoding="utf-8"))
        files, step = state["files"], int(state["step"])
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ValueError(f"STATE.json is unreadable ({type(exc).__name__})") from None
    if not isinstance(files, dict):
        raise ValueError("STATE.json lists no files")
    if state.get("schema") != STATE_SCHEMA:
        raise ValueError(f"format {state.get('schema')!r} is not the one this version writes ({STATE_SCHEMA})")
    if state.get("config_digest") != digest:
        raise ValueError("it was written by a run with a different configuration")
    if not 0 < step < total:
        raise ValueError(f"update {step} is outside 1..{total - 1}")
    for name in STATE_FILES:
        meta, path = files.get(name), folder / name
        if (not isinstance(meta, dict) or not path.is_file() or path.stat().st_size != meta.get("bytes")
                or file_sha256(path) != meta.get("sha256")):
            raise ValueError(f"{name} is missing or damaged")
        try:
            torch.load(path, map_location="cpu", weights_only=True)          # a format this torch cannot read is as good as damaged
        except Exception as exc:
            raise ValueError(f"{name} cannot be loaded ({type(exc).__name__})") from None
    return state


def _rows(path: Path) -> list[dict]:
    """The readable lines of a JSON-lines file (a line cut off by a kill is skipped)."""
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        with contextlib.suppress(ValueError):
            row = json.loads(line)
            if isinstance(row, dict) and isinstance(row.get("step"), int):
                rows.append(row)
    return rows


def _logged_steps(path: Path) -> list[int]:
    with open(path, newline="", encoding="utf-8") as fh:
        steps = []
        for row in list(csv.reader(fh))[1:]:
            with contextlib.suppress(ValueError, IndexError):
                steps.append(int(row[0]))
    return steps


def check_run_dir(out: Path, step: int, monitors) -> None:
    """The logs and monitored adapters in `out` must cover everything up to `step`, or the run cannot continue from it."""
    if not (out / "CONFIG.json").is_file():
        raise ValueError("CONFIG.json is missing")
    try:
        seen = {row.get("step") for row in _rows(out / "dev_curve.jsonl")}
        logged = _logged_steps(out / "train_steps.csv")
    except OSError as exc:
        raise ValueError(f"the logs cannot be read ({exc.strerror})") from None
    due = [0, *(m for m in sorted(monitors) if m <= step)]
    if not set(due) <= seen:
        raise ValueError(f"dev_curve.jsonl lacks update(s) {sorted(set(due) - seen)}")
    if logged[:step] != list(range(1, step + 1)):
        raise ValueError(f"train_steps.csv lacks rows up to update {step}")
    for m in due[1:]:
        if not (out / f"checkpoint_{m:05d}").is_dir():
            raise ValueError(f"checkpoint_{m:05d} is missing")


def prune_run_dir(out: Path, step: int) -> None:
    """Cut the logs and monitored adapters back to `step`; the resumed run writes the later ones again."""
    curve = out / "dev_curve.jsonl"
    keep = [row for row in _rows(curve) if row["step"] <= step]
    tmp = curve.with_name(curve.name + ".tmp")
    tmp.write_text("".join(json.dumps(row) + "\n" for row in keep), encoding="utf-8")
    tmp.replace(curve)
    log = out / "train_steps.csv"
    with open(log, newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    tmp = log.with_name(log.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        csv.writer(fh).writerows([rows[0], *(r for r in rows[1:] if r and r[0].isdigit() and int(r[0]) <= step)])
    tmp.replace(log)
    for ckpt in out.glob("checkpoint_*"):
        if ckpt.name.split("_", 1)[1].isdigit() and int(ckpt.name.split("_", 1)[1]) > step:
            shutil.rmtree(ckpt)
    for stray in out.glob(".tmp-checkpoint_*"):
        shutil.rmtree(stray, ignore_errors=True)
    (out / "SELECTED.json").unlink(missing_ok=True)


def claim_output(out: Path, digest: str, total: int, monitors) -> dict | None:
    """Rank 0, before training: the state of a usable `.partial` checkpoint with `out` cut back to it, else None.

    In the second case whatever an earlier attempt left is moved aside (`*.incomplete-<epoch>`) and `out` is created empty."""
    final = partial_dir(out)
    old, tmp = partial_leftovers(out)
    why = []
    for folder in (final, old):
        if not folder.is_dir():
            continue
        try:
            state = check_partial(folder, digest, total)
            check_run_dir(out, state["step"], monitors)
        except ValueError as exc:
            why.append(f"{folder.name}: {exc}")
            continue
        if folder != final:                        # save_partial stopped between its two renames
            shutil.rmtree(final, ignore_errors=True)
            folder.rename(final)
        shutil.rmtree(old, ignore_errors=True)
        shutil.rmtree(tmp, ignore_errors=True)
        prune_run_dir(out, state["step"])
        return state
    stamp = int(time.time())
    for path in (out, final, old):
        if path.exists():
            aside = path.with_name(f"{path.name}.incomplete-{stamp}")
            path.rename(aside)
            print(f"moved an incomplete earlier attempt to {aside.name}", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
    if why:
        print("not resuming: " + "; ".join(why), flush=True)
    out.mkdir(parents=True)
    return None


def load_partial(folder: Path, trainable, names: list[str], optimizer) -> None:
    """Put the saved LoRA tensors and optimizer state back (every rank does this; the files were verified by rank 0)."""
    saved = torch.load(folder / "trainable.pt", map_location="cpu", weights_only=True)
    if saved["names"] != names:
        raise RuntimeError(f"{folder}: the saved parameters are not this model's trainable parameters")
    with torch.no_grad():
        for p, t in zip(trainable, saved["tensors"]):
            if p.shape != t.shape:
                raise RuntimeError(f"{folder}: shape {tuple(t.shape)} saved for a parameter of shape {tuple(p.shape)}")
            p.copy_(t)
    optimizer.load_state_dict(torch.load(folder / "optimizer.pt", map_location="cpu", weights_only=True))


def save_checkpoint(model, out: Path, step: int) -> None:
    """Adapter at a monitored update, written under a hidden name and renamed: a checkpoint_* folder is always whole."""
    tmp, final = out / f".tmp-checkpoint_{step:05d}", out / f"checkpoint_{step:05d}"
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_adapter(tmp)
    shutil.rmtree(final, ignore_errors=True)
    tmp.rename(final)


# -- process group -------------------------------------------------------------------

def init_group(cuda: bool, device: torch.device) -> None:
    """Start the process group with a 60-minute timeout: rank 0 evaluates alone while the others wait for it."""
    options = {"timeout": GROUP_TIMEOUT}
    if cuda and "device_id" in inspect.signature(dist.init_process_group).parameters:
        options["device_id"] = device              # connects NCCL at once; older torch in the supported range lacks the argument
    dist.init_process_group("nccl" if cuda else "gloo", **options)


def broadcast_int(value: int, device: torch.device) -> int:
    """Rank 0's `value` on every rank (also a synchronisation point)."""
    t = torch.tensor([value], dtype=torch.long, device=device)
    dist.broadcast(t, src=0)
    return int(t.item())


# -- the stage -----------------------------------------------------------------

def load_teacher(c: Campaign, table: ContextTable, device) -> tuple[torch.Tensor, str]:
    """Teacher logits of the unit's contexts, once COMPLETE.json confirms they belong to this campaign's union table."""
    info, why = teacher_state(c, ContextTable.load(c.stem_union()))
    if info is None:
        first = f"delete {c.teacher_complete} and " if c.teacher_complete.is_file() else ""
        raise RuntimeError(f"the teacher output cannot be used for this campaign: {why}. Run the teacher stage again (it "
                           f"recomputes when its contexts, its teacher or its structure strings differ): {first}repeat your "
                           f"`mpd submit` command with --force, which queues the stages whose output is missing and keeps "
                           f"the finished ones.")
    logits = np.load(c.teacher_logits, mmap_mode="r")[table.union_index]
    return torch.as_tensor(logits, dtype=torch.float32, device=device), info["logits_sha256"]


def config_record(c: Campaign, unit: str, table: ContextTable, train_rows, dev_rows, monitors, world: int,
                  teacher_sha: str, dtype: str) -> dict:
    record = {"unit": unit, "student": c.spec.student, "teacher": c.spec.teacher, "objective": c.spec.objective,
              "train_contexts": int(len(train_rows)), "dev_contexts": int(len(dev_rows)),
              "variants": None if table.var_mt is None else int(len(table.var_mt)), "world_size": world,
              "monitors": sorted(monitors), "context_digest": table.digest(), "teacher_logits_sha256": teacher_sha,
              "dtype": dtype, "dms_labels_opened": False, **c.resolved}
    if c.needs_structure("student"):               # other strings are another run: this goes into the resume digest
        record["structures_digest"] = c.structures_digest()
    return record


def train_unit(c: Campaign, unit: str, rank: int, world: int, local_rank: int, device, every: int, on_update) -> None:
    spec, r = c.spec, c.resolved
    out = c.train_dir(unit)
    random.seed(spec.seed)
    np.random.seed(spec.seed)
    torch.manual_seed(spec.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(spec.seed)

    table = ContextTable.load(c.stem_unit(unit))
    structures = c.structures_for("student", spec.train_data)        # None for a student that reads sequence only
    teacher, teacher_sha = load_teacher(c, table, device)
    dev = table.dev_mask(r["dev_fraction"])
    train_rows, dev_rows = np.flatnonzero(~dev), np.flatnonzero(dev)
    if len(train_rows) == 0 or len(dev_rows) == 0:
        raise RuntimeError(f"unit {unit}: {len(train_rows)} training and {len(dev_rows)} development contexts; need both")

    model = students.load(spec.student, device, lora={"rank": r["lora_rank"], "alpha": r["lora_alpha"]})
    model.encode(table, structures)
    trainable = model.trainable()
    if not trainable:
        raise RuntimeError("no trainable LoRA parameters")
    names = [n for n, p in model.module.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=r["peak_lr"], betas=(0.9, 0.999), eps=1e-8, weight_decay=r["weight_decay"])
    objective = Objective(c, table, teacher, device)
    steps = update_stream(train_rows, r["updates"], r["global_batch"], spec.seed)
    total = len(steps)
    monitors = set(monitor_steps(r["updates"]))
    accumulation = r["accumulation"]
    config = config_record(c, unit, table, train_rows, dev_rows, monitors, world, teacher_sha,
                           str(next((p.dtype for p in model.module.parameters() if not p.requires_grad), None)))
    digest = config_digest(config)

    # resume or start fresh: rank 0 decides, everyone follows
    state = claim_output(out, digest, total, monitors) if rank == 0 else None
    start = broadcast_int(state["step"] if state else 0, device) if world > 1 else (state["step"] if state else 0)
    if start:
        load_partial(partial_dir(out), trainable, names, optimizer)
    forward = (torch.nn.parallel.DistributedDataParallel(model.module, device_ids=[local_rank] if device.type == "cuda" else None,
                                                         find_unused_parameters=False) if world > 1 else model.module)
    resumed_from = [*state["resumed_from"], start] if state else []
    started = time.monotonic() - (state["elapsed_seconds"] if state else 0.0)

    log = writer = None
    try:
        if rank == 0:
            fields = ["step", "loss", "huber", "lr", "grad_norm", "elapsed_seconds"]
            if state is None:
                write_json(out / "CONFIG.json", {**config, "config_digest": digest, "provenance": provenance()})
                first = {**objective.dev_metrics(model, dev_rows), "step": 0, "elapsed_seconds": 0.0}
                with open(out / "dev_curve.jsonl", "w", encoding="utf-8") as fh:
                    fh.write(json.dumps(first) + "\n")
                print(json.dumps(first), flush=True)
                log = open(out / "train_steps.csv", "w", newline="", encoding="utf-8")
                writer = csv.DictWriter(log, fieldnames=fields)
                writer.writeheader()
            else:
                print(f"{unit}: resuming from update {start} of {total} ({partial_dir(out).name})", flush=True)
                log = open(out / "train_steps.csv", "a", newline="", encoding="utf-8")
                writer = csv.DictWriter(log, fieldnames=fields)
        if world > 1:
            dist.barrier()

        model.module.train()
        for step in range(start + 1, total + 1):
            batch = steps[step - 1]
            lr = learning_rate(step, total, r["peak_lr"], r["warmup_fraction"], r["min_lr_ratio"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            mine = np.split(batch[rank::world], accumulation)
            loss_sum = huber = 0.0
            for k, rows in enumerate(mine):
                sync = forward.no_sync() if world > 1 and k < len(mine) - 1 else contextlib.nullcontext()
                with sync:
                    loss, parts = objective.loss(model.logits(table, rows, forward=forward), rows)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"non-finite loss at update {step}")
                    (loss / accumulation).backward()
                loss_sum += float(loss.detach()) / accumulation
                huber += float(parts["huber"]) / accumulation
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, r["grad_clip"]))
            if not math.isfinite(grad_norm):
                raise RuntimeError(f"non-finite gradient norm at update {step}")
            optimizer.step()
            if rank == 0:
                writer.writerow({"step": step, "loss": loss_sum, "huber": huber, "lr": lr, "grad_norm": grad_norm,
                                 "elapsed_seconds": round(time.monotonic() - started, 1)})
                if step % 20 == 0:
                    log.flush()
            if step in monitors:
                if rank == 0:
                    metrics = {**objective.dev_metrics(model, dev_rows), "step": step,
                               "elapsed_seconds": round(time.monotonic() - started, 1)}
                    save_checkpoint(model, out, step)          # before the curve line, so a line always has its adapter
                    with open(out / "dev_curve.jsonl", "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(metrics) + "\n")
                    log.flush()
                    print(json.dumps(metrics), flush=True)
                if world > 1:
                    dist.barrier()
            if every and step % every == 0 and step < total and rank == 0:
                log.flush()
                try:
                    save_partial(out, step=step, total=total, digest=digest, names=names, trainable=trainable,
                                 optimizer=optimizer, elapsed=time.monotonic() - started, resumed_from=resumed_from)
                    print(f"{unit}: resumable checkpoint at update {step}", flush=True)
                except OSError as exc:                         # losing the ability to resume is not worth losing the run
                    print(f"{unit}: could not write the resumable checkpoint at update {step} ({exc}); continuing without it",
                          flush=True)
            if on_update is not None:
                on_update(step)

        if rank == 0:
            log.close()
            log = None
            curve = _rows(out / "dev_curve.jsonl")
            chosen = select_checkpoint(curve)
            write_json(out / "SELECTED.json", chosen)
            for ckpt in out.glob("checkpoint_*"):
                if ckpt.name != f"checkpoint_{chosen['step']:05d}" and os.environ.get("MPD_KEEP_ALL_CHECKPOINTS") != "1":
                    shutil.rmtree(ckpt)
            write_json(out / "COMPLETE.json", {
                "status": "PASS", "updates": total, "selected_step": chosen["step"],
                "seconds": round(time.monotonic() - started, 1), "resumed_from": resumed_from,
                "config_digest": digest, "peak_gpu_memory_gb": peak_gpu_gb(), "provenance": provenance()})
            discard_partial(out)
            print(f"{unit}: selected update {chosen['step']} (dev loss {chosen['dev_loss']:.4f}, start {chosen['start_dev_loss']:.4f})",
                  flush=True)
        if world > 1:
            dist.barrier()                                     # nobody leaves while rank 0 is still finishing
    finally:
        if log is not None:
            log.close()


def run(c: Campaign, task: dict, *, ckpt_every: int | None = None, on_update=None) -> None:
    """Train one unit. `ckpt_every` overrides MPD_CKPT_EVERY; `on_update(step)` runs after every update (tests raise from it)."""
    require_cuda()
    configure_torch()
    unit, r = task["unit"], c.resolved
    rank, local_rank, world = (int(os.environ.get(k, d)) for k, d in (("RANK", 0), ("LOCAL_RANK", 0), ("WORLD_SIZE", 1)))
    if world != r["world_size"]:
        raise RuntimeError(f"launched with {world} process(es) but the plan has {r['world_size']}")
    out = c.train_dir(unit)
    if (out / "COMPLETE.json").is_file():
        print(f"{unit}: already complete", flush=True)
        if rank == 0:
            discard_partial(out)
        return
    every = checkpoint_interval(ckpt_every)
    cuda = torch.cuda.is_available()
    device = torch.device("cuda", local_rank) if cuda else torch.device("cpu")
    if cuda:
        torch.cuda.set_device(local_rank)
    if world > 1:
        init_group(cuda, device)
    try:
        train_unit(c, unit, rank, world, local_rank, device, every, on_update)
    finally:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


def main(argv=None, **options):
    return stage_main(lambda c, task: run(c, task, **options), __doc__, argv)


if __name__ == "__main__":
    main()
