"""Map a structure's chain onto an assay's full sequence. No fuzzy alignment: residues match one to one or it is an error.

Handled: the whole sequence; a part of it (ProteinGym `pdb_range`); a structure with extra residues around the
sequence (an expression tag); up to `max_subs` single-residue differences between the construct and the model. The
amino-acid channel of the model input always stays the assay's residue; only the 3Di state is taken from the structure.
"""

from __future__ import annotations

from dataclasses import dataclass


class AlignmentError(ValueError):
    pass


@dataclass(frozen=True)
class Alignment:
    seq_start: int                      # 0-based index in the assay sequence of the first residue the structure covers
    chain_start: int                    # 0-based index in the structure chain of that residue
    n: int                              # residues covered
    substitutions: tuple[str, ...]      # assay residue, 1-based assay position, structure residue: "R72P"


def _mismatches(seq: str, seq_start: int, chain: str, chain_start: int, n: int) -> list[int]:
    return [i for i in range(n) if seq[seq_start + i] != chain[chain_start + i]]


def align_to_sequence(seq: str, chain: str, span: tuple[int, int] | None = None, max_subs: int = 3) -> Alignment:
    """Place `chain` (the structure's amino acids) on `seq`. With `span` (0-based, half-open) the chain starts at the
    span's start; without one it must have the sequence's length or contain the sequence exactly once."""
    if span is not None:
        start, end = span
        if not 0 <= start < len(seq):
            raise AlignmentError(f"structure range starts at residue {start + 1}, outside the {len(seq)}-residue sequence")
        seq_start, chain_start = start, 0
        n = min(len(chain), end - start, len(seq) - start)
    elif len(chain) == len(seq):
        seq_start, chain_start, n = 0, 0, len(seq)
    elif len(chain) > len(seq):
        found, i = [], chain.find(seq)
        while i != -1:
            found.append(i)
            i = chain.find(seq, i + 1)
        if not found:
            raise AlignmentError(
                f"the {len(chain)}-residue structure does not contain the {len(seq)}-residue sequence; insertions and "
                f"deletions are not aligned. If the structure covers only part of the sequence, give the assay a structure range.")
        if len(found) > 1:
            raise AlignmentError(f"the sequence occurs {len(found)} times in the {len(chain)}-residue structure (ambiguous)")
        seq_start, chain_start, n = 0, found[0], len(seq)
    else:
        raise AlignmentError(
            f"the structure has {len(chain)} residues but the sequence has {len(seq)}; without a structure range the "
            f"structure must cover the whole sequence")
    if n <= 0:
        raise AlignmentError("the structure and the assay sequence do not overlap")
    bad = _mismatches(seq, seq_start, chain, chain_start, n)
    if len(bad) > max_subs:
        i = bad[0]
        raise AlignmentError(
            f"{len(bad)} of {n} structure residues differ from the sequence (more than {max_subs}); first at sequence "
            f"position {seq_start + i + 1}: sequence {seq[seq_start + i]}, structure {chain[chain_start + i]}. "
            f"Insertions and deletions are not aligned.")
    subs = tuple(f"{seq[seq_start + i]}{seq_start + i + 1}{chain[chain_start + i]}" for i in bad)
    return Alignment(seq_start, chain_start, n, subs)
