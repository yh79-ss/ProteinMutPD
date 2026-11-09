"""SaProt (amino acid + Foldseek 3Di structure token) as a masked-position AA20 logit model. Students and teachers.

One token per residue. The focal residue is replaced by `#` plus its own 3Di state, so the model sees the local fold
but not the amino acid; the 21 logits of one amino acid (one per state) are combined with a log-sum-exp, which gives the
AA20 log-marginal the official SaProt mutation score is built from.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from .. import registry
from ..contexts import ContextTable
from ..structure import vocab
from .esm2 import LORA_TARGETS, default_dtype, model_path, optimal_window


class Saprot:
    family = "saprot"

    def __init__(self, key: str, device: torch.device, dtype: torch.dtype | None = None, *,
                 lora: dict | None = None, adapter: Path | None = None):
        from transformers import AutoModelForMaskedLM
        self.key, self.device = key, device
        info = registry.get(key)
        path = model_path(key)
        if (Path(path) / "vocab.txt").is_file():
            vocab.load_vocab(Path(path) / "vocab.txt")           # ids come from mpd.structure.vocab; refuse a different table
        if lora is not None and adapter is not None:
            raise ValueError("pass either lora (to train) or adapter (to load a trained one)")
        hub = {"revision": info.revision} if path == info.hf_id and info.revision else {}     # only the opt-in Hub route
        # The 650M checkpoint ships with dropout 0.1; distillation and scoring run with none (the 35M config has 0.0).
        model = AutoModelForMaskedLM.from_pretrained(path, torch_dtype=dtype or default_dtype(), attn_implementation="eager",
                                                     hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0, **hub)
        if model.config.vocab_size != vocab.VOCAB_SIZE:
            raise RuntimeError(f"{key}: the model has {model.config.vocab_size} tokens, a SaProt AA+3Di model has {vocab.VOCAB_SIZE}")
        active = sorted({m.p for m in model.modules() if isinstance(m, torch.nn.Dropout)} - {0.0})
        if active:
            raise RuntimeError(f"{key}: dropout {active} is still active after loading with dropout 0")
        # Same as Esm2: a saved adapter names the public checkpoint, not the local folder it was trained from.
        model.name_or_path = model.config.name_or_path = info.hf_id
        if lora is not None:
            from peft import LoraConfig, get_peft_model
            model = get_peft_model(model, LoraConfig(r=lora["rank"], lora_alpha=lora["alpha"], lora_dropout=0.0,
                                                     bias="none", target_modules=LORA_TARGETS,
                                                     task_type="FEATURE_EXTRACTION"))
            if not os.environ.get("MPD_MODEL_" + key.upper().replace("-", "_")):    # an override folder is not the pinned weights
                model.peft_config["default"].revision = info.revision
        elif adapter is not None:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
        self.module = model.to(device)
        self.aa_index = torch.as_tensor(vocab.aa_state_ids().reshape(-1), dtype=torch.long, device=device)   # [20 * 21]
        self._mask_id = vocab.mask_ids()
        self._tokens: list[list[int]] = []
        self._states: list[str] = []

    def trainable(self) -> list[torch.nn.Parameter]:
        return [p for p in self.module.parameters() if p.requires_grad]

    def save_adapter(self, path: Path) -> None:
        self.module.save_pretrained(path, safe_serialization=True)

    def encode(self, table: ContextTable, structures=None) -> None:
        """Tokenize the table's sequences with their structure strings (`structures.for_table`)."""
        if structures is None:
            raise ValueError(f"{self.key} reads protein structure: pass the campaign's structure set (contexts/structures.json)")
        self._states = structures.for_table(table)
        self._tokens = [vocab.encode(s, st) for s, st in zip(table.sequences, self._states)]

    def _batch(self, table: ContextTable, rows: np.ndarray):
        seqs, focal = [], []
        for r in rows:
            sid, pos = int(table.seq_id[r]), int(table.pos[r])
            tokens, f = optimal_window(self._tokens[sid], pos)
            tokens = list(tokens)
            tokens[f] = self._mask_id[self._states[sid][pos]]      # '#' + this residue's own 3Di state
            seqs.append(tokens)
            focal.append(f)
        width = max(map(len, seqs))
        ids = torch.full((len(rows), width), vocab.PAD, dtype=torch.long)
        att = torch.zeros_like(ids)
        for i, t in enumerate(seqs):
            ids[i, :len(t)] = torch.tensor(t, dtype=torch.long)
            att[i, :len(t)] = 1
        return ids.to(self.device), att.to(self.device), torch.as_tensor(focal, dtype=torch.long, device=self.device)

    def logits(self, table: ContextTable, rows: np.ndarray, forward=None) -> torch.Tensor:
        """AA20 log-marginals at the masked position of each context in `rows`: [len(rows), 20], float32."""
        ids, att, focal = self._batch(table, rows)
        out = (forward or self.module)(input_ids=ids, attention_mask=att).logits
        sel = out[torch.arange(len(rows), device=self.device), focal].float()
        return torch.logsumexp(sel.index_select(-1, self.aa_index).reshape(len(rows), 20, 21), dim=-1)
