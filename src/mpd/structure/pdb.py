"""C-alpha records of a PDB file: the amino-acid string and the B-factor column, per chain. Stdlib only.

The B-factor of the CA atom is the residue's confidence value (AlphaFold DB and ESMFold write pLDDT there). The
rule matches the earlier study's: ATOM records of the first model, alternate location ' ' or 'A', one residue per
(chain, residue number, insertion code).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

AA3 = {"ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F", "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K",
       "LEU": "L", "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R", "SER": "S", "THR": "T", "VAL": "V",
       "TRP": "W", "TYR": "Y"}


class PdbError(ValueError):
    pass


@dataclass(frozen=True)
class CaChain:
    chain: str
    aa: str                          # one letter per CA record; X for a residue outside the 20 standard ones
    bfactor: tuple[float, ...]       # same length as `aa`


def ca_chains(path: Path) -> dict[str, CaChain]:
    """Every chain of the first model, in file order."""
    aa: dict[str, list[str]] = {}
    bfac: dict[str, list[float]] = {}
    seen: set[tuple[str, str, str]] = set()
    with open(path, encoding="utf-8", errors="replace") as fh:
        for number, line in enumerate(fh, start=1):
            if line.startswith("ENDMDL"):
                break
            if not line.startswith("ATOM") or line[12:16].strip() != "CA" or line[16:17] not in (" ", "A"):
                continue
            chain = line[21:22]
            key = (chain, line[22:26], line[26:27])
            if key in seen:
                continue
            seen.add(key)
            try:
                value = float(line[60:66])
            except ValueError:
                value = math.nan
            if not math.isfinite(value):
                raise PdbError(f"{Path(path).name}, line {number}: no usable B-factor in columns 61-66")
            aa.setdefault(chain, []).append(AA3.get(line[17:20], "X"))
            bfac.setdefault(chain, []).append(value)
    return {c: CaChain(c, "".join(aa[c]), tuple(bfac[c])) for c in aa}


def ca_records(path: Path, chain: str | None = None) -> CaChain:
    """One chain's CA records. Without `chain` the file must hold exactly one chain."""
    chains = ca_chains(path)
    name = Path(path).name
    if not chains:
        raise PdbError(f"{name}: no CA atoms in ATOM records")
    if chain is None:
        if len(chains) > 1:
            raise PdbError(f"{name} has {len(chains)} chains ({', '.join(chains)}); choose one with the assay's structure_chain")
        return next(iter(chains.values()))
    if chain not in chains:
        raise PdbError(f"{name} has no chain {chain!r} (chains: {', '.join(chains)})")
    return chains[chain]
