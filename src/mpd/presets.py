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

# Training time, measured on H200 GPUs for the esm2-650m student (bf16, micro-batch 2, global batch 64, 2,103 updates; sacct and the
# training logs of the earlier study). All-assay: four GPUs took 33.5 min in-process, i.e. 134 GPU-minutes = 63.8 min per 1,000
# updates. One-assay: the slowest run took 111.8 min = 53.2 min per 1,000 updates. Per-GPU time scales with the parameter
# count to the power 0.9 (an extrapolation: no other student was timed).
H200_TRAIN_MIN_PER_1000 = {"all": 63.8, "one": 53.2}
REFERENCE_PARAMS_M = 655.0
SIZE_EXPONENT = 0.9
MEASURED_STUDENTS = ("esm2-650m",)

# Forward-only rate of esm2-650m on one H200 (bf16 weights, fp32 attention, batch 8), fitted to two scoring jobs of the earlier
# study: a ProteinGym shard of 4.9M tokens took 2.6 min and the VenusMutHub table of 2.3M tokens 1.5 min, each including about
# 30 s of start-up (fit: 39.9k tokens/s). Lowered by 10% so a walltime derived from it is generous.
H200_FORWARD_TOKENS_PER_S = 36_000.0


@dataclass(frozen=True)
class TableSize:
    contexts: int            # (sequence, masked position) pairs
    tokens: int              # sum of window widths, each capped at 1,024
    assay_mean_tokens: int   # the same count for one assay's own table: the mean over assays ...
    assay_max_tokens: int    # ... and the largest assay


# What the teacher and score stages walk through, counted with `prepare`'s table builder on the pinned releases (ProteinGym v1.3,
# VenusMutHub single_mutant): `dense` = every residue of the assayed regions (direct-logit training), `observed` = only assayed
# positions (score training and every evaluation table).
BENCHMARK_SIZES = {
    "proteingym": {"assays": 217,
                   "dense": TableSize(62_671, 38_982_915, 216_752, 3_500_032),
                   "observed": TableSize(37_753, 19_686_449, 105_223, 1_400_832)},
    "venusmuthub": {"assays": 793,
                    "dense": TableSize(135_060, 56_749_164, 131_665, 1_484_800),
                    "observed": TableSize(8_945, 2_290_488, 4_151, 115_344)},
}


def preset_for(family: str, objective: str) -> Preset:
    try:
        return PRESETS[(family, objective)]
    except KeyError:
        raise ValueError(f"no training preset for family={family!r} objective={objective!r}") from None
