"""The SaProt vocabulary: 5 special tokens plus every (amino acid, 3Di state) pair, 446 tokens in all.

A residue is one token `AA + state`, e.g. `Ev` (glutamate, 3Di state `v`); `#` stands for a masked amino acid or an
unknown / low-confidence state. Ids come from this table, not from a tokenizer, so they cannot drift with the
installed `transformers`; `load_vocab` checks a model folder's own `vocab.txt` against it. Stdlib and numpy only.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..constants import AA20

SPECIAL = ("<cls>", "<pad>", "<eos>", "<unk>", "<mask>")
CLS, PAD, EOS, UNK, MASK = range(5)
AA_SYM = "ACDEFGHIKLMNPQRSTVWY#"          # '#' = masked amino acid
STATES = "pynwrqhgdlvtmfsaeikc#"          # 3Di letters in SaProt's order; '#' = unknown / masked state
VOCAB_SIZE = len(SPECIAL) + len(AA_SYM) * len(STATES)
VOCAB_SHA256 = "d4289b3ff55d50b1aa33a3141ec344fc2a69cb3dca43aeefe9fccfee371ffd28"   # vocab.txt of both SaProt checkpoints


class VocabError(ValueError):
    pass


def build_vocab() -> list[str]:
    return list(SPECIAL) + [a + s for a in AA_SYM for s in STATES]


def vocab_text(tokens: list[str] | None = None) -> str:
    return "\n".join(tokens or build_vocab()) + "\n"


def load_vocab(path: Path) -> list[str]:
    """Tokens of a model folder's `vocab.txt`, refused unless it is byte-identical to the generated vocabulary."""
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != VOCAB_SHA256:
        raise VocabError(f"{path}: vocab.txt has sha256 {digest[:16]}..., expected {VOCAB_SHA256[:16]}... "
                         f"(not a SaProt AA+3Di vocabulary)")
    return data.decode("ascii").splitlines()


@lru_cache(maxsize=None)
def token_ids() -> dict[str, int]:
    return {t: i for i, t in enumerate(build_vocab())}


def token_id(aa: str, state: str) -> int:
    return token_ids()[aa + state]


@lru_cache(maxsize=None)
def _aa_state_ids() -> np.ndarray:
    ids = np.asarray([[token_ids()[a + s] for s in STATES] for a in AA20], dtype=np.int64)
    # the 20 x 21 block is contiguous because the vocabulary lists the amino acids in AA20 order
    assert (ids == np.arange(len(SPECIAL), len(SPECIAL) + 20 * 21).reshape(20, 21)).all()
    return ids


def aa_state_ids() -> np.ndarray:
    """[20, 21] token ids of every amino acid (rows, `AA20` order) in every state (columns, `STATES` order)."""
    return _aa_state_ids().copy()


def mask_ids() -> dict[str, int]:
    """State -> id of the masked-amino-acid token that keeps that state (`#d` for state `d`; `##` for `#`)."""
    return {s: token_ids()["#" + s] for s in STATES}


def encode(sequence: str, states: str) -> list[int]:
    """`<cls>`, one token per residue, `<eos>`. Letters outside the 20 amino acids become `#`; the state string is
    lower-case 3Di letters or `#`."""
    if len(sequence) != len(states):
        raise ValueError(f"sequence has {len(sequence)} residues but the structure string has {len(states)}")
    ids = token_ids()
    out = [CLS]
    for i, (a, s) in enumerate(zip(sequence, states)):
        a = a.upper()
        if a not in AA20:
            a = "#"
        s = s.lower()
        if s not in STATES:
            raise ValueError(f"residue {i + 1}: {states[i]!r} is not a 3Di state (expected one of {STATES})")
        out.append(ids[a + s])
    out.append(EOS)
    return out
