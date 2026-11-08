"""ESM2 as a masked-position AA20 logit model (sequence only). Used for both students and teachers."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from .. import registry
from ..constants import AA20
from ..contexts import ContextTable

LORA_TARGETS = ["query", "key", "value", "attention.output.dense"]
MAX_TOKENS = 1024


def model_path(key: str) -> str:
    """`MPD_MODEL_<KEY>` (e.g. MPD_MODEL_ESM2_650M) overrides the Hugging Face id with a local folder."""
    override = os.environ.get("MPD_MODEL_" + key.upper().replace("-", "_"))
    return override or registry.get(key).hf_id


def default_dtype() -> torch.dtype:
    name = os.environ.get("MPD_DTYPE")
    if name:
        return getattr(torch, name)
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


def optimal_window(token_ids: list[int], residue_pos: int, max_tokens: int = MAX_TOKENS) -> tuple[list[int], int]:
    """ProteinGym's position-centered window for sequences longer than the model's context.

    `token_ids` includes <cls> and <eos>. Returns the window and the masked token's index inside it.
    """
    token_pos = residue_pos + 1
    if not 1 <= token_pos < len(token_ids) - 1:
        raise ValueError("residue position outside the tokenized sequence")
    if len(token_ids) <= max_tokens:
        return token_ids, token_pos
    half = max_tokens // 2
    if token_pos < half:
        start = 0
    elif token_pos >= len(token_ids) - half:
        start = len(token_ids) - max_tokens
    else:
        start = token_pos - half
    return token_ids[start:start + max_tokens], token_pos - start


class Esm2:
    family = "esm2"

    def __init__(self, key: str, device: torch.device, dtype: torch.dtype | None = None, *,
                 lora: dict | None = None, adapter: Path | None = None):
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        self.key, self.device = key, device
        path = model_path(key)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        model = AutoModelForMaskedLM.from_pretrained(path, torch_dtype=dtype or default_dtype(),
                                                     attn_implementation="eager")
        if lora is not None and adapter is not None:
            raise ValueError("pass either lora (to train) or adapter (to load a trained one)")
        if lora is not None:
            from peft import LoraConfig, get_peft_model
            model = get_peft_model(model, LoraConfig(r=lora["rank"], lora_alpha=lora["alpha"], lora_dropout=0.0,
                                                     bias="none", target_modules=LORA_TARGETS,
                                                     task_type="FEATURE_EXTRACTION"))
        elif adapter is not None:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter, is_trainable=False)
        self.module = model.to(device)
        self.aa_ids = [self.tokenizer.convert_tokens_to_ids(a) for a in AA20]
        if len(set(self.aa_ids)) != 20 or self.tokenizer.unk_token_id in self.aa_ids:
            raise RuntimeError(f"{key}: the tokenizer does not map the 20 amino acids to distinct tokens")
        self.aa_index = torch.as_tensor(self.aa_ids, dtype=torch.long, device=device)
        self._tokens: list[list[int]] = []

    def trainable(self) -> list[torch.nn.Parameter]:
        return [p for p in self.module.parameters() if p.requires_grad]

    def save_adapter(self, path: Path) -> None:
        self.module.save_pretrained(path, safe_serialization=True)

    def encode(self, table: ContextTable, structures=None) -> None:
        self._tokens = [self.tokenizer.encode(s, add_special_tokens=True) for s in table.sequences]
        for s, t in zip(table.sequences, self._tokens):
            if len(t) != len(s) + 2:
                raise RuntimeError(f"{self.key}: tokenization is not one token per residue")

    def _batch(self, table: ContextTable, rows: np.ndarray):
        mask_id, pad_id = self.tokenizer.mask_token_id, self.tokenizer.pad_token_id
        seqs, focal = [], []
        for r in rows:
            tokens, f = optimal_window(self._tokens[int(table.seq_id[r])], int(table.pos[r]))
            tokens = list(tokens)
            tokens[f] = mask_id
            seqs.append(tokens)
            focal.append(f)
        width = max(map(len, seqs))
        ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
        att = torch.zeros_like(ids)
        for i, t in enumerate(seqs):
            ids[i, :len(t)] = torch.tensor(t, dtype=torch.long)
            att[i, :len(t)] = 1
        return ids.to(self.device), att.to(self.device), torch.as_tensor(focal, dtype=torch.long, device=self.device)

    def logits(self, table: ContextTable, rows: np.ndarray, forward=None) -> torch.Tensor:
        """Raw AA20 logits at the masked position of each context in `rows`: [len(rows), 20], float32."""
        ids, att, focal = self._batch(table, rows)
        out = (forward or self.module)(input_ids=ids, attention_mask=att).logits
        return out[torch.arange(len(rows), device=self.device), focal].float().index_select(-1, self.aa_index)
