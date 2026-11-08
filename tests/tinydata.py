"""Builders for the tiny fixtures (no pytest fixtures here, so tests can import them directly)."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import pytest

ESM_VOCAB = ["<cls>", "<pad>", "<eos>", "<unk>", "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K", "Q", "N",
             "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z", "O", ".", "-", "<null_1>", "<mask>"]
AA = "ACDEFGHIKLMNPQRSTVWY"


def make_tiny_esm(folder: Path, seed: int) -> Path:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "vocab.txt").write_text("\n".join(ESM_VOCAB) + "\n")
    tok = transformers.EsmTokenizer(str(folder / "vocab.txt"))
    cfg = transformers.EsmConfig(vocab_size=33, mask_token_id=32, pad_token_id=1, hidden_size=32, num_hidden_layers=2,
                                 num_attention_heads=2, intermediate_size=64, max_position_embeddings=1026,
                                 position_embedding_type="rotary", token_dropout=True, emb_layer_norm_before=False,
                                 hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)   # as in the real ESM2 checkpoints
    torch.manual_seed(seed)
    transformers.EsmForMaskedLM(cfg).save_pretrained(folder)
    tok.save_pretrained(folder)
    return folder


def write_tiny_proteingym(root: Path, seed: int = 0) -> dict[str, str]:
    """Three assays: A and C on one protein (shared contexts), B on another. Returns {assay_id: sequence}."""
    rng = random.Random(seed)
    seqs = {"P1": "".join(rng.choice(AA) for _ in range(40)), "P2": "".join(rng.choice(AA) for _ in range(34))}
    assays = {"ASSAY_A": ("P1", range(2, 14)), "ASSAY_B": ("P2", range(1, 13)), "ASSAY_C": ("P1", range(8, 22))}
    (root / "DMS_ProteinGym_substitutions").mkdir(parents=True, exist_ok=True)
    ref = []
    for aid, (prot, positions) in assays.items():
        seq = seqs[prot]
        rows = []
        for p in positions:
            for a in AA:
                if a != seq[p]:
                    rows.append((f"{seq[p]}{p + 1}{a}", seq[:p] + a + seq[p + 1:], rng.gauss(0, 1), rng.randint(0, 1)))
        rows.append((f"{seq[3]}4{AA[0] if seq[3] != AA[0] else AA[1]}:{seq[5]}6{AA[0] if seq[5] != AA[0] else AA[1]}", seq, 0.0, 0))
        with open(root / "DMS_ProteinGym_substitutions" / f"{aid}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["mutant", "mutated_sequence", "DMS_score", "DMS_score_bin"])
            w.writerows(rows)
        start, end = (2, 30) if aid == "ASSAY_B" else (1, len(seq))      # MSA_start/MSA_end are 1-based and inclusive
        ref.append({"DMS_id": aid, "DMS_filename": f"{aid}.csv", "UniProt_ID": prot, "target_seq": seq, "pdb_file": f"{prot}.pdb",
                    "MSA_start": start, "MSA_end": end})
    with open(root / "DMS_substitutions.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ref[0]))
        w.writeheader()
        w.writerows(ref)
    return {aid: seqs[prot] for aid, (prot, _) in assays.items()}
