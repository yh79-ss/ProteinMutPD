"""Shared constants. Keep this module import-light: the planner runs on login nodes."""

# Canonical amino-acid order used by every teacher-target file, student score and
# evaluation join. Teacher caches that use another order must be permuted on load.
AA20 = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: i for i, aa in enumerate(AA20)}

# Dual-temperature Score-KD readout (see objectives.score_from_logits).
SCORE_T_MUT = 1.0
SCORE_T_WT = 1.5

DATASETS = ("proteingym", "venusmuthub")
OBJECTIVES = ("direct_logits", "score")
SCOPES = ("all", "one")

# The four distillation methods exposed on the command line.
METHODS = {
    "direct-all": ("direct_logits", "all"),
    "direct-one": ("direct_logits", "one"),
    "score-all": ("score", "all"),
    "score-one": ("score", "one"),
}
