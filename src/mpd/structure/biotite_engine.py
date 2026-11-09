"""`MPD_3DI_ENGINE=biotite`: biotite's NumPy port of Foldseek's 3Di encoder, for a machine without a Foldseek binary.

biotite is BSD-3-Clause and optional (`pip install biotite`). On the 675 structures of ProteinGym v1.3 and the Venus set it
matched Foldseek on 674; in the other one 1 residue of 2,832 differed. Foldseek defines 3Di and the recorded results, so it
stays the default and the engine is recorded in `structures.json`.
"""

from __future__ import annotations

from pathlib import Path

from . import pdb
from .foldseek import Chain, EngineError


class BiotiteEngine:
    name = "biotite"

    def __init__(self):
        try:
            import biotite
            import biotite.structure.alphabet  # noqa: F401
            import biotite.structure.io.pdb  # noqa: F401
        except ImportError as exc:
            raise EngineError("MPD_3DI_ENGINE=biotite needs biotite: pip install biotite") from exc
        self.version = biotite.__version__

    @property
    def id(self) -> str:
        return f"biotite-{self.version}"

    def describe(self) -> dict:
        return {"name": self.name, "version": self.version, "binary_sha256": None,
                "command": "biotite.structure.alphabet.to_3di", "pinned_release": False, "reference_build": False}

    def chains(self, path: Path | str) -> list[Chain]:
        import biotite.structure as struc
        import biotite.structure.io.pdb as pdbio
        from biotite.structure.alphabet import to_3di
        path = Path(path)
        atoms = pdbio.PDBFile.read(str(path)).get_structure(model=1)
        atoms = atoms[struc.filter_amino_acids(atoms)]
        try:
            sequences, starts = to_3di(atoms)
        except Exception as exc:                      # biotite raises plain ValueErrors for missing backbone atoms
            raise EngineError(f"{path.name}: biotite could not compute 3Di: {exc}") from exc
        records = pdb.ca_chains(path)
        out = []
        for chain, states in zip(atoms.chain_id[starts], sequences):
            states = str(states).upper()
            aa = records[str(chain)].aa if str(chain) in records else ""
            if len(aa) != len(states):
                raise EngineError(f"{path.name}, chain {chain}: biotite found {len(states)} residues, the CA records {len(aa)}; use Foldseek")
            out.append(Chain(f"{path.name}_{chain}", str(chain), aa, states))
        if not out:
            raise EngineError(f"{path.name}: no amino-acid chains")
        return out
