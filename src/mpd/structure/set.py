"""The structure strings of a campaign: one 3Di-or-`#` string per sequence, frozen as `contexts/structures.json`.

A sidecar to `ContextTable` rather than a field of it, so campaigns that never use structure keep their file
formats and digests. Keyed by the sequence hash (`contexts.sequence_hash`), because a string belongs to a sequence and
its structure source, not to one context row. numpy-free, stdlib only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from .vocab import STATES

SCHEMA = 1


def _lines(states: dict[str, str]) -> str:
    return "".join(f"{h}\t{s}\n" for h, s in sorted(states.items()))


@dataclass(frozen=True)
class StructureSet:
    states: dict[str, str]                       # sha256(sequence) -> lower-case 3Di letters or '#', one per residue
    meta: dict = field(default_factory=dict)     # engine, masking policy, per-sequence provenance: see build.build_set
    # Sequences that two datasets of one campaign carry with different structures: the first dataset's string is in
    # `states`, the other datasets' strings are here (dataset -> hash -> string). Empty in nearly every campaign.
    scopes: dict[str, dict[str, str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        bad = [h for h, s in self.states.items() if set(s) - set(STATES)]
        if bad:
            raise ValueError(f"structure string of sequence {bad[0][:12]} has characters outside the 3Di alphabet")

    def __len__(self) -> int:
        return len(self.states)

    def scoped(self, dataset: str | None) -> "StructureSet":
        """The set as seen by `dataset`: its own strings for the sequences it shares with another dataset."""
        return StructureSet({**self.states, **self.scopes.get(dataset or "", {})}, self.meta)

    def for_table(self, table) -> list[str]:
        """One string per entry of `table.sequences` (same order). Raises KeyError naming the first missing sequence."""
        shared = {h for own in self.scopes.values() for h in own}
        if shared and not shared.isdisjoint(table.hashes):
            raise ValueError("this table holds a sequence that two datasets give different structures; "
                             "look it up in StructureSet.scoped(<dataset>)")
        out = []
        for seq, h in zip(table.sequences, table.hashes):
            s = self.states.get(h)
            if s is None:
                raise KeyError(f"no structure string for the sequence {seq[:12]}... ({len(seq)} residues, sha256 {h[:12]}); "
                               f"it was not in the set built by `prepare`")
            if len(s) != len(seq):
                raise ValueError(f"the structure string for sequence sha256 {h[:12]} has {len(s)} residues, the sequence {len(seq)}")
            out.append(s)
        return out

    def digest(self) -> str:
        """sha256 of the sorted `hash<TAB>string` lines (and, if any, each dataset's own lines after an `@dataset` line)."""
        text = _lines(self.states) + "".join(f"@{d}\n" + _lines(s) for d, s in sorted(self.scopes.items()) if s)
        return hashlib.sha256(text.encode("ascii")).hexdigest()

    def masked_fraction(self) -> float:
        total = sum(len(s) for s in self.states.values())
        return sum(s.count("#") for s in self.states.values()) / total if total else 0.0

    def manifest_block(self) -> dict:
        """What `contexts/MANIFEST.json` records about this set."""
        return {"digest": self.digest(), "sequences": len(self), "masked_fraction": round(self.masked_fraction(), 6),
                "engine": self.meta.get("engine"), "policy": self.meta.get("policy"), "datasets": self.meta.get("datasets"),
                "conflicts": len(self.meta.get("conflicts", [])), "dms_labels_opened": False}

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        body = {"schema": SCHEMA, "digest": self.digest(), "meta": self.meta, "states": self.states, "scopes": self.scopes}
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(body, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "StructureSet":
        body = json.loads(Path(path).read_text(encoding="utf-8"))
        if body.get("schema") != SCHEMA:
            raise ValueError(f"{path}: structures.json schema {body.get('schema')!r}, expected {SCHEMA}")
        out = cls(body["states"], body.get("meta", {}), body.get("scopes", {}))
        if out.digest() != body["digest"]:
            raise ValueError(f"{path}: the strings do not match the recorded digest; the file was edited or truncated")
        return out
