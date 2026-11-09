"""Benchmark access with a label firewall.

`variants()` returns which substitutions were measured and never touches the score column. `labels()` returns
the scores and is called only by the evaluation stage (tests/test_stages.py enforces this).
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..constants import AA20, AA_INDEX

_SINGLE = re.compile(r"^([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])$")


@dataclass(frozen=True)
class Assay:
    assay_id: str
    protein_id: str              # assays on the same protein share it
    sequence: str
    structure: Path | None = None
    structure_scale: str | None = None   # meaning of the B-factor column: see data.venus_structures.SCALE_TO_PERCENT
    region: tuple[int, int] | None = None  # 0-based [start, end) of the assayed region; None = the whole sequence
    structure_span: tuple[int, int] | None = None   # 0-based [start, end) of the sequence the structure covers; None = all of it
    structure_chain: str | None = None   # chain to read when the structure file has several; None = its only chain

    def span(self) -> tuple[int, int]:
        return self.region or (0, len(self.sequence))


@dataclass(frozen=True)
class Variants:
    """Single substitutions of one assay, row-aligned with `Benchmark.labels`."""
    mutants: tuple[str, ...]     # e.g. "A123C" (numbering as published)
    pos: np.ndarray              # 0-based index into the assay sequence, int32
    wt: np.ndarray               # AA20 index, uint8
    mt: np.ndarray               # AA20 index, uint8

    def __len__(self) -> int:
        return len(self.mutants)


class Benchmark:
    """A set of assays. Subclasses implement `assays`, `_rows` and `labels`."""
    name: str

    def assays(self) -> list[Assay]:
        raise NotImplementedError

    def assay(self, assay_id: str) -> Assay:
        for a in self.assays():
            if a.assay_id == assay_id:
                return a
        raise KeyError(f"{self.name} has no assay {assay_id!r}")

    def variants(self, assay_id: str) -> Variants:
        raise NotImplementedError

    def labels(self, assay_id: str) -> np.ndarray:
        """Scores aligned with `variants(assay_id)`. Evaluation stage only."""
        raise NotImplementedError


def build_variants(mutants: list[str], sequence: str, offset: int = 1) -> tuple[Variants, list[int]]:
    """Parse `X123Y` strings against `sequence`. Returns the variants and the input indices that were kept.

    Strings that are not a single canonical substitution (multi-mutants, indels) are skipped. A substitution
    whose wild-type residue disagrees with the sequence is an error, not a skip.
    """
    keep, pos, wt, mt = [], [], [], []
    for i, m in enumerate(mutants):
        match = _SINGLE.match(m)
        if not match:
            continue
        w, p, t = match.group(1), int(match.group(2)) - offset, match.group(3)
        if not 0 <= p < len(sequence) or sequence[p] != w:
            raise ValueError(f"{m}: wild type {w} does not match the sequence at position {p + offset}")
        keep.append(i)
        pos.append(p)
        wt.append(AA_INDEX[w])
        mt.append(AA_INDEX[t])
    return Variants(tuple(mutants[i] for i in keep), np.asarray(pos, dtype=np.int32),
                    np.asarray(wt, dtype=np.uint8), np.asarray(mt, dtype=np.uint8)), keep


_PDB_RANGE = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")


def pdb_span(assay_id: str, pdb_range: str, seq_len: int) -> tuple[int, int] | None:
    """ProteinGym's `pdb_range` ("a-b", 1-based inclusive over `target_seq`) as a 0-based half-open span clipped to the
    sequence (one structure is longer than its assay's sequence). Empty means the whole sequence."""
    text = (pdb_range or "").strip()
    if not text:
        return None
    match = _PDB_RANGE.match(text)
    if not match:
        raise ValueError(f"{assay_id}: pdb_range {text!r} is not of the form 'a-b'")
    a, b = int(match.group(1)), int(match.group(2))
    if not 1 <= a <= b or a > seq_len:
        raise ValueError(f"{assay_id}: pdb_range {text} does not fit the {seq_len}-residue sequence")
    return a - 1, min(b, seq_len)


class ProteinGym(Benchmark):
    """ProteinGym v1.3 substitutions as unpacked by `mpd fetch proteingym`.

    Mutations index the full `target_seq` 1-based in every assay (docs/DATA.md). Structures are the official AF2 models
    (pLDDT 0-100 in the B-factor column); `pdb_range` gives the part of `target_seq` a model covers. An entry that
    lists several structures (`|` in `pdb_file` or `pdb_range`, absent from v1.3) gets no structure.
    """
    name = "proteingym"

    def __init__(self, root: Path):
        self.root = Path(root)
        ref = self.root / "DMS_substitutions.csv"
        if not ref.is_file():
            raise FileNotFoundError(f"{ref} not found; run `mpd fetch proteingym`")
        with open(ref, newline="", encoding="utf-8") as fh:
            self._ref = {r["DMS_id"]: r for r in csv.DictReader(fh)}
        self._assays: list[Assay] | None = None

    def assays(self) -> list[Assay]:
        if self._assays is None:
            structures = self.root / "ProteinGym_AF2_structures"
            self._assays = []
            for k, r in sorted(self._ref.items()):
                pdb_file = (r.get("pdb_file") or "").strip()
                single = bool(pdb_file) and "|" not in pdb_file and "|" not in (r.get("pdb_range") or "")
                self._assays.append(Assay(
                    assay_id=k, protein_id=r["UniProt_ID"], sequence=r["target_seq"],
                    structure=structures / pdb_file if single else None,
                    structure_scale="percent_0_to_100" if single else None,
                    region=(int(r["MSA_start"]) - 1, int(r["MSA_end"])) if r.get("MSA_start") else None,
                    structure_span=pdb_span(k, r.get("pdb_range", ""), len(r["target_seq"])) if single else None))
        return self._assays

    def _read(self, assay_id: str, score: bool) -> tuple[list[str], list[float]]:
        row = self._ref[assay_id]
        path = self.root / "DMS_ProteinGym_substitutions" / row["DMS_filename"]
        mutants, scores = [], []
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for r in reader:
                mutants.append(r["mutant"])
                if score:
                    scores.append(float(r["DMS_score"]))
        return mutants, scores

    def variants(self, assay_id: str) -> Variants:
        mutants, _ = self._read(assay_id, score=False)
        return build_variants(mutants, self._ref[assay_id]["target_seq"])[0]

    def labels(self, assay_id: str) -> np.ndarray:
        mutants, scores = self._read(assay_id, score=True)
        _, keep = build_variants(mutants, self._ref[assay_id]["target_seq"])
        return np.asarray([scores[i] for i in keep], dtype=np.float64)


def open_benchmark(name: str, home: Path) -> Benchmark:
    from .sources import PROTEINGYM
    if name == "proteingym":
        return ProteinGym(Path(home) / "data" / PROTEINGYM.dirname)
    if name == "venusmuthub":
        from .venusmuthub import VenusMutHub
        return VenusMutHub.open(Path(home))
    raise ValueError(f"unknown benchmark {name!r}")


__all__ = ["AA20", "Assay", "Benchmark", "ProteinGym", "Variants", "build_variants", "open_benchmark", "pdb_span"]
