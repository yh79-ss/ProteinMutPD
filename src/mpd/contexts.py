"""Masked-position contexts: the unit of work for teachers, students and scorers. numpy only, no labels.

A context is one (sequence, position) pair, scored by masking that position and reading the 20 amino-acid
logits. Contexts are unique per table and sorted by (sequence hash, position), so a table built twice from the
same benchmark is identical.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .constants import AA_INDEX
from .data.benchmarks import Benchmark

DEV_SALT = "MPD_POSITION_DEV_V1"


def sequence_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.encode("ascii")).hexdigest()


@dataclass
class ContextTable:
    sequences: list[str]
    seq_id: np.ndarray                       # [N] int32, index into `sequences`
    pos: np.ndarray                          # [N] int32, 0-based
    wt: np.ndarray                           # [N] uint8, AA20 index of the residue at `pos`
    var_ptr: np.ndarray | None = None        # [N+1] int64: variants of context i are var_mt[var_ptr[i]:var_ptr[i+1]]
    var_mt: np.ndarray | None = None         # [M] uint8, observed mutant AA20 indices
    union_index: np.ndarray | None = None    # [N] int64, row in the campaign-wide union table

    def __len__(self) -> int:
        return len(self.pos)

    @property
    def hashes(self) -> list[str]:
        return [sequence_hash(s) for s in self.sequences]

    def keys(self) -> list[tuple[str, int]]:
        h = self.hashes
        return [(h[s], int(p)) for s, p in zip(self.seq_id, self.pos)]

    def variants_of(self, i: int) -> np.ndarray:
        if self.var_ptr is None:
            raise ValueError("this table has no variant lists")
        return self.var_mt[self.var_ptr[i]:self.var_ptr[i + 1]]

    def dev_mask(self, fraction: float) -> np.ndarray:
        h = self.hashes
        return dev_mask([h[s] for s in self.seq_id], [int(p) for p in self.pos], fraction)

    def save(self, stem: Path) -> None:
        stem.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"seq_id": self.seq_id, "pos": self.pos, "wt": self.wt}
        for name in ("var_ptr", "var_mt", "union_index"):
            if getattr(self, name) is not None:
                arrays[name] = getattr(self, name)
        np.savez(stem.with_suffix(".npz"), **arrays)
        stem.with_suffix(".json").write_text(json.dumps({"sequences": self.sequences}) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, stem: Path) -> "ContextTable":
        seqs = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))["sequences"]
        with np.load(stem.with_suffix(".npz"), allow_pickle=False) as z:
            return cls(seqs, z["seq_id"], z["pos"], z["wt"], *(z[k] if k in z else None
                                                                 for k in ("var_ptr", "var_mt", "union_index")))

    def subset(self, index: np.ndarray) -> "ContextTable":
        """Rows `index` of this table (variants are carried along); `union_index` points back at this table."""
        index = np.asarray(index, dtype=np.int64)
        if self.var_ptr is None:
            ptr = mt = None
        else:
            lens = self.var_ptr[index + 1] - self.var_ptr[index]
            ptr = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
            mt = np.concatenate([self.var_mt[self.var_ptr[i]:self.var_ptr[i + 1]] for i in index]) if len(index) \
                else np.empty(0, dtype=np.uint8)
        return ContextTable(self.sequences, self.seq_id[index], self.pos[index], self.wt[index], ptr, mt, index)

    def digest(self) -> str:
        h = hashlib.sha256()
        for s in self.sequences:
            h.update(s.encode("ascii") + b"\n")
        for a in (self.seq_id, self.pos, self.wt, self.var_ptr, self.var_mt):
            if a is not None:
                h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()


def build_table(benchmark: Benchmark, assay_ids: list[str], with_variants: bool = True, dense: bool = False) -> ContextTable:
    """Unique (sequence, position) contexts of the given assays.

    Observed (default): the positions that were assayed, each with the mutants observed there.
    Dense: every position of each assay's region (no variant lists), as in dense AA20 distillation.
    """
    if dense and with_variants:
        raise ValueError("a dense table has no variant lists")
    seen: dict[tuple[str, int], set[int]] = defaultdict(set)
    sequences: dict[str, str] = {}
    for aid in assay_ids:
        assay = benchmark.assay(aid)
        h = sequence_hash(assay.sequence)
        sequences[h] = assay.sequence
        start, end = assay.span()
        v = benchmark.variants(aid)
        if len(v) and (v.pos.min() < start or v.pos.max() >= end):
            raise ValueError(f"{aid}: assayed positions fall outside the assay region {start + 1}-{end}")
        if dense:
            for p in range(start, end):
                seen.setdefault((h, p), set())
        else:
            for p, m in zip(v.pos, v.mt):
                seen[(h, int(p))].add(int(m))
    order = sorted(sequences)
    index = {h: i for i, h in enumerate(order)}
    keys = sorted(seen)
    seq_id = np.asarray([index[h] for h, _ in keys], dtype=np.int32)
    pos = np.asarray([p for _, p in keys], dtype=np.int32)
    wt = np.asarray([AA_INDEX[sequences[h][p]] for h, p in keys], dtype=np.uint8)
    ptr = mt = None
    if with_variants:
        lists = [sorted(seen[k]) for k in keys]
        ptr = np.concatenate([[0], np.cumsum([len(x) for x in lists])]).astype(np.int64)
        mt = np.asarray([m for x in lists for m in x], dtype=np.uint8)
    return ContextTable([sequences[h] for h in order], seq_id, pos, wt, ptr, mt)


def union(tables: list[ContextTable]) -> ContextTable:
    """Unique contexts across tables, variants dropped. Use `locate` to find a table's rows in the result."""
    sequences: dict[str, str] = {}
    keys: set[tuple[str, int]] = set()
    for t in tables:
        for s, h in zip(t.sequences, t.hashes):
            sequences[h] = s
        keys.update(t.keys())
    order = sorted(sequences)
    index = {h: i for i, h in enumerate(order)}
    ordered = sorted(keys)
    out = ContextTable(
        [sequences[h] for h in order],
        np.asarray([index[h] for h, _ in ordered], dtype=np.int32),
        np.asarray([p for _, p in ordered], dtype=np.int32),
        np.asarray([AA_INDEX[sequences[h][p]] for h, p in ordered], dtype=np.uint8))
    return out


def locate(table: ContextTable, parent: ContextTable) -> np.ndarray:
    """Row of each of `table`'s contexts inside `parent` (which must contain them)."""
    where = {k: i for i, k in enumerate(parent.keys())}
    return np.asarray([where[k] for k in table.keys()], dtype=np.int64)


def dev_mask(seq_hashes: list[str], positions: list[int], fraction: float = 0.05) -> np.ndarray:
    """Position-grouped development split: whole positions go to dev, never single variants.

    Each context joins dev when a hash of (salt, sequence, position) falls below `fraction`; every sequence with
    at least two contexts keeps at least one in training and at least min(5, n-1) in dev.
    """
    grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for index, (h, p) in enumerate(zip(seq_hashes, positions)):
        key = f"{DEV_SALT}:{h}:{p}".encode("ascii")
        grouped[h].append((index, int.from_bytes(hashlib.sha256(key).digest()[:8], "big")))
    mask = np.zeros(len(positions), dtype=np.bool_)
    cutoff = int(round(fraction * (1 << 64)))
    for entries in grouped.values():
        if len(entries) < 2:
            continue                     # a lone context cannot be split; it stays in training
        for index, value in entries:
            mask[index] = value < cutoff
        minimum = min(5, len(entries) - 1)
        if sum(bool(mask[i]) for i, _ in entries) < minimum:
            for i, _ in sorted(entries, key=lambda e: e[1])[:minimum]:
                mask[i] = True
        if all(mask[i] for i, _ in entries):
            mask[max(entries, key=lambda e: e[1])[0]] = False
    return mask
