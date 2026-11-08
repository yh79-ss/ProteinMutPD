"""Per-family training presets, taken from the runs that produced the paper tables.

Everything here is a default; each field can be overridden on the command line. The
presets are pure data so the planner can read them without torch.

Sources (this project's own audited runs):
  * ESM2 / SaProt direct-logit and Score-KD: rank-8 / alpha-16 Q,K,V,O LoRA, AdamW
    (0.9, 0.999, eps 1e-8, wd 0.01), peak LR 3e-5, 5% warm-up, cosine to 0.1 x peak,
    grad clip 1.0, 2,103 updates, global batch 64 positions.
  * Direct-logit KD = Huber(delta=1) on the 20 AA logits + KL(teacher || student) at
    T=1 over the 20 AAs (weight 1). Score KD = Huber(delta=1) on the scalar
    log_softmax(z/1.0)[mut] - log_softmax(z/1.5)[wt], no KL.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Preset:
    updates: int
    global_batch: int
    microbatch: int
    peak_lr: float
    kl_weight: float
    support: str           # aa20_dense: every position x 20 AAs | observed: only assayed mutants
    lora_rank: int = 8
    lora_alpha: int = 16
    dev_fraction: float = 0.05
    warmup_fraction: float = 0.05
    min_lr_ratio: float = 0.1
    grad_clip: float = 1.0
    weight_decay: float = 0.01


_ESM_LIKE_DIRECT = Preset(updates=2103, global_batch=64, microbatch=2, peak_lr=3e-5, kl_weight=1.0,
                          support="aa20_dense")
_ESM_LIKE_SCORE = Preset(updates=2103, global_batch=64, microbatch=2, peak_lr=3e-5, kl_weight=0.0,
                         support="observed")

PRESETS: dict[tuple[str, str], Preset] = {
    ("esm2", "direct_logits"): _ESM_LIKE_DIRECT,
    ("esm2", "score"): _ESM_LIKE_SCORE,
    ("saprot", "direct_logits"): _ESM_LIKE_DIRECT,
    ("saprot", "score"): _ESM_LIKE_SCORE,
}

# Measured on one H200 for the 650M students: 70.0-111.8 min for 2,103 updates at global
# batch 64 with one GPU (esm_saprot_direct_logits training logs). The planner scales this by
# the GPU count and by each GPU type's time_factor, then adds slack.
H200_MINUTES_PER_1000_UPDATES = 111.8 / 2.103


def preset_for(family: str, objective: str) -> Preset:
    try:
        return PRESETS[(family, objective)]
    except KeyError:
        raise ValueError(f"no training preset for family={family!r} objective={objective!r}") from None
