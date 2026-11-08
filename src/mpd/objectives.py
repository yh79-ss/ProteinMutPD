"""The two distillation objectives and the two score readouts.

All functions take AA20 logits in the canonical order (`constants.AA20`), shaped [positions, 20].

direct logits   Huber(delta=1) between raw student and teacher logits, plus kl_weight x KL(teacher || student)
                at T=1 over the 20 amino acids.
score           Huber(delta=1) between scalar scores g of student and teacher at the observed variants,
                g(i, a) = log softmax(z_i / T_MUT)[a] - log softmax(z_i / T_WT)[wt_i], no KL.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .constants import SCORE_T_MUT, SCORE_T_WT


def score_from_logits(logits: torch.Tensor, wt: torch.Tensor, row: torch.Tensor, mt: torch.Tensor) -> torch.Tensor:
    """g for each variant: `row` picks its position in `logits`, `mt` the mutant, `wt[row]` the wild type."""
    z = logits[row].float()
    top = F.log_softmax(z / SCORE_T_MUT, dim=-1).gather(-1, mt[:, None])[:, 0]
    bottom = F.log_softmax(z / SCORE_T_WT, dim=-1).gather(-1, wt[row][:, None])[:, 0]
    return top - bottom


def direct_loss(student: torch.Tensor, teacher: torch.Tensor, kl_weight: float):
    huber = F.smooth_l1_loss(student, teacher, beta=1.0)
    kl = F.kl_div(F.log_softmax(student, dim=-1), F.softmax(teacher, dim=-1), reduction="batchmean")
    return huber + kl_weight * kl, {"huber": huber.detach(), "kl": kl.detach()}


def score_loss(student: torch.Tensor, teacher: torch.Tensor, wt: torch.Tensor, row: torch.Tensor, mt: torch.Tensor):
    gs = score_from_logits(student, wt, row, mt)
    with torch.no_grad():
        gt = score_from_logits(teacher, wt, row, mt)
    huber = F.smooth_l1_loss(gs, gt, beta=1.0)
    return huber, {"huber": huber.detach()}, gs.detach(), gt


def flatten_variants(var_ptr: np.ndarray, var_mt: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Observed variants of the contexts `rows`: (batch row of each variant, its mutant AA20 index)."""
    counts = var_ptr[rows + 1] - var_ptr[rows]
    batch_row = np.repeat(np.arange(len(rows)), counts)
    mt = np.concatenate([var_mt[var_ptr[r]:var_ptr[r + 1]] for r in rows]) if len(rows) else np.empty(0, np.uint8)
    return batch_row, mt


def readout_delta(scores: np.ndarray, wt: np.ndarray, mt: np.ndarray) -> np.ndarray:
    """Plain masked-marginal difference s[mut] - s[wt] (the convention of the zero-shot baselines)."""
    return scores[np.arange(len(mt)), mt] - scores[np.arange(len(mt)), wt]


def readout_dual(scores: np.ndarray, wt: np.ndarray, mt: np.ndarray) -> np.ndarray:
    """The Score-KD scalar g on the scored AA20 vector (see module docstring)."""
    z = torch.as_tensor(np.ascontiguousarray(scores), dtype=torch.float32)
    idx = torch.arange(len(mt))
    return score_from_logits(z, torch.as_tensor(wt.astype(np.int64)), idx, torch.as_tensor(mt.astype(np.int64))).numpy()
