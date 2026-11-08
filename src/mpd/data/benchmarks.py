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


class ProteinGym(Benchmark):
    """ProteinGym v1.3 substitutions as unpacked by `mpd fetch proteingym`.

    Mutations index the full `target_seq` 1-based in every assay (docs/DATA.md).
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
            self._assays = [
                Assay(assay_id=k, protein_id=r["UniProt_ID"], sequence=r["target_seq"],
                      structure=structures / r["pdb_file"] if r.get("pdb_file") else None,
                      region=(int(r["MSA_start"]) - 1, int(r["MSA_end"])) if r.get("MSA_start") else None)
                for k, r in sorted(self._ref.items())]
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


__all__ = ["AA20", "Assay", "Benchmark", "ProteinGym", "Variants", "build_variants", "open_benchmark"]
