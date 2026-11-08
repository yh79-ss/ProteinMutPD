"""VenusMutHub (AI4Protein/VenusMutHub, MIT) single-mutant assays, as unpacked by `mpd fetch venusmuthub`.

The matched cohort is 793 assays on 489 proteins with 25,044 rows. Relative to the 796 public files:

* three assays on two proteins are left out (`EXCLUDED_ASSAYS`): the input curation of the original study
  marked them unsupported, so they have no structure in the structure set and every model is scored on the
  same 793 assays;
* two rows whose label is not a number (`NDA`, `1.25 ± 0.04`) are dropped;
* 198 identity rows (`M1M`: wild type to itself, a measurement of the wild type) are kept.

`variants()` never uses a label value; it only needs to know which rows have a numeric label.
"""

from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path

import numpy as np

from .benchmarks import Assay, Benchmark, Variants, build_variants
from .sources import VENUS_STRUCTURES, VENUSMUTHUB
from . import venus_structures as vs

# assay id -> why it is not part of the matched cohort
EXCLUDED_ASSAYS = {
    "P14779_ee": "protein of 1,048 residues without a structure in the matched set",
    "PPB_Affinity_2NYY_A_kd": "protein of 1,267 residues without a structure in the matched set",
    "PPB_Affinity_2NZ9_A_kd": "protein of 1,267 residues without a structure in the matched set",
}
TASKS = {"activity": "Activity", "dti_binding": "Binding", "ppi_binding": "Binding",
         "selectivity": "Selectivity", "stability": "Stability"}


def protein_id(sequence: str) -> str:
    return vs.protein_id_for(sequence)


class _Parsed:
    def __init__(self, sequence: str, mutants: list[str], scores: np.ndarray, task: str, dropped: int):
        self.sequence, self.mutants, self.scores, self.task, self.dropped = sequence, mutants, scores, task, dropped


class VenusMutHub(Benchmark):
    name = "venusmuthub"

    def __init__(self, root: Path, structures: Path | None = None):
        self.root = Path(root)
        self.structures = Path(structures) if structures else None
        folder = self.root / "single_mutant"
        if not folder.is_dir():
            raise FileNotFoundError(f"{folder} not found; run `mpd fetch venusmuthub`")
        self._files = {p.stem: p for p in sorted(folder.glob("*/*.csv")) if p.stem not in EXCLUDED_ASSAYS}
        self._parsed: dict[str, _Parsed] = {}
        self._structure_rows: dict[str, dict] | None = None
        self._assays: list[Assay] | None = None

    @classmethod
    def open(cls, home: Path) -> "VenusMutHub":
        structures = Path(home) / "data" / VENUS_STRUCTURES.dirname
        return cls(Path(home) / "data" / VENUSMUTHUB.dirname, structures if structures.is_dir() else None)

    # -- parsing ---------------------------------------------------------------
    def _parse(self, assay_id: str) -> _Parsed:
        if assay_id in self._parsed:
            return self._parsed[assay_id]
        path = self._files[assay_id]
        rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
        mutants, sequences, scores, dropped = [], [], [], 0
        for r in rows:
            try:
                y = float(r["fitness_score"])
            except ValueError:
                y = math.nan
            if not math.isfinite(y):
                dropped += 1
                continue
            mutants.append(r["mutant"])
            sequences.append(r["mutated_sequence"])
            scores.append(y)
        wt = _wild_type(assay_id, mutants, sequences)
        parsed = _Parsed(wt, mutants, np.asarray(scores, dtype=np.float64), TASKS[path.parent.name], dropped)
        self._parsed[assay_id] = parsed
        return parsed

    def _structure_table(self) -> dict[str, dict]:
        if self._structure_rows is None:
            self._structure_rows = {}
            if self.structures is not None and (self.structures / vs.MANIFEST_NAME).is_file():
                self._structure_rows = {r["protein_id"]: r for r in vs.read_manifest(self.structures / vs.MANIFEST_NAME)}
        return self._structure_rows

    # -- Benchmark -------------------------------------------------------------
    def assays(self) -> list[Assay]:
        if self._assays is None:
            table = self._structure_table()
            out = []
            for aid in self._files:
                seq = self._parse(aid).sequence
                pid = protein_id(seq)
                row = table.get(pid)
                out.append(Assay(aid, pid, seq, self.structures / row["file"] if row and self.structures else None,
                                 row["confidence_scale"] if row else None))
            self._assays = out
        return self._assays

    def task(self, assay_id: str) -> str:
        return self._parse(assay_id).task

    def variants(self, assay_id: str) -> Variants:
        p = self._parse(assay_id)
        return build_variants(p.mutants, p.sequence)[0]

    def labels(self, assay_id: str) -> np.ndarray:
        p = self._parse(assay_id)
        _, keep = build_variants(p.mutants, p.sequence)
        return p.scores[keep]


def _wild_type(assay_id: str, mutants: list[str], sequences: list[str]) -> str:
    """The wild-type sequence implied by the rows. Every row of a file must imply the same one."""
    import re
    single = re.compile(r"^([A-Z])(\d+)([A-Z])$")
    wts = set()
    for m, s in zip(mutants, sequences):
        match = single.match(m)
        if not match:
            continue
        w, pos, t = match.group(1), int(match.group(2)) - 1, match.group(3)
        if not 0 <= pos < len(s) or s[pos] != t:
            raise ValueError(f"{assay_id}: {m} does not match its mutated_sequence")
        wts.add(s[:pos] + w + s[pos + 1:])
    if len(wts) != 1:
        raise ValueError(f"{assay_id}: rows imply {len(wts)} different wild-type sequences")
    return wts.pop()
