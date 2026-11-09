"""Model registry: which models can play which role, and what they need.

Adding a model means adding one ``ModelInfo`` here plus a student and/or teacher
adapter module (see docs/ADDING_A_MODEL.md). The planner reads only this table, so
it stays importable without torch.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    key: str
    family: str                       # esm2 | saprot | aido
    hf_id: str
    params_m: float
    roles: tuple[str, ...]            # subset of ("student", "teacher")
    status: str = "available"         # available | planned (planned models are rejected by the planner)
    needs_structure: bool = False     # SaProt: 3Di tokens; AIDO student: aligned structure
    needs_msa: bool = False           # AIDO teacher: full selected MSA
    shard_across_gpus: bool = False   # layers must be split over the GPUs of one node
    teacher_backends: tuple[str, ...] = ("live",)
    revision: str | None = None       # Hugging Face commit of the pinned weights; file hashes are in data/models.tsv
    license: str = ""                 # licence of the weights, shown by `mpd models` and recorded by `mpd fetch model`
    # If set, outputs of this model (as teacher) may only be distilled into models of
    # this family. Enforced in RunSpec.validate; see docs/LICENSES.md.
    output_family_lock: str | None = None
    license_note: str = ""
    # Memory model used only to choose a GPU shape. weights_gb is bf16 weights;
    # act_gb is the estimated activation memory (+ LoRA optimizer state) in GB at
    # micro-batch 2 and scales with the micro-batch. The ESM2 values come from
    # counting the tensors autograd saves for a 2 x 1024-token batch on the meta
    # device; they have NOT been measured on a GPU (act_gb_measured stays False
    # until someone does). Override the result with --gpu / --train-gpus.
    act_gb: float = 8.0
    act_gb_measured: bool = False
    note: str = ""

    @property
    def weights_gb(self) -> float:
        return self.params_m * 2 / 1000.0


GENBIO_LICENSE_NOTE = (
    "GenBio AI Community License, clause 1.2.2: Outputs (any output of the GenBio Materials, including "
    "intermediate activations and embeddings) may not be used to train, optimize or improve machine-learning "
    "models similar to the GenBio Materials, nor for model distillation toward that end, except to train the "
    "GenBio Materials themselves and their downstream Derivative Works. Non-commercial use only."
)

# Revisions are the Hugging Face commits the sha256 manifest in data/models.tsv was built from; regenerate both
# with scripts/maintainers/pin_models.py. ESM2 and SaProt weights are MIT-licensed (model cards, 2026-11-09).
MODELS: dict[str, ModelInfo] = {m.key: m for m in [
    ModelInfo("esm2-8m", "esm2", "facebook/esm2_t6_8M_UR50D", 8, ("student", "teacher"), act_gb=2,
              revision="c731040fcd8d73dceaa04b0a8e6329b345b0f5df", license="MIT"),
    ModelInfo("esm2-35m", "esm2", "facebook/esm2_t12_35M_UR50D", 35, ("student", "teacher"), act_gb=3,
              revision="6fbf070e65b0b7291e7bbcd451118c216cff79d8", license="MIT"),
    ModelInfo("esm2-150m", "esm2", "facebook/esm2_t30_150M_UR50D", 150, ("student", "teacher"), act_gb=9,
              revision="a695f6045e2e32885fa60af20c13cb35398ce30c", license="MIT"),
    ModelInfo("esm2-650m", "esm2", "facebook/esm2_t33_650M_UR50D", 650, ("student", "teacher"), act_gb=12,
              revision="08e4846e537177426273712802403f7ba8261b6c", license="MIT", note="Sequence only."),
    ModelInfo("esm2-3b", "esm2", "facebook/esm2_t36_3B_UR50D", 3000, ("student", "teacher"), act_gb=27,
              revision="476b639933c8baad5ad09a60ac1a87f987b656fc", license="MIT"),
    ModelInfo("saprot-35m", "saprot", "westlake-repl/SaProt_35M_AF2", 35, ("student", "teacher"),
              needs_structure=True, act_gb=3,
              revision="316cd4017d29f4657b959365f24b57f1ee278912", license="MIT",
              note="Reads protein structure (Foldseek 3Di). VenusMutHub needs a structure set that is not "
                   "published yet."),
    ModelInfo("saprot-650m", "saprot", "westlake-repl/SaProt_650M_AF2", 650, ("student", "teacher"),
              needs_structure=True, act_gb=12,
              revision="d9b9ad00ef61c0990e611b2b43f2231c7de24b38", license="MIT",
              note="Reads protein structure (Foldseek 3Di; pLDDT<70 masked). VenusMutHub needs a structure set "
                   "that is not published yet."),
    # AIDO is registered only so that `mpd models` and the planner can explain why it is not offered. v0.1 has no
    # AIDO student/teacher module, `mpd fetch` has no AIDO source, and nothing from GenBio is shipped. Facts for a
    # later port (this project's own runs, 2026-09): bf16 weights 32.13 GB; a 12,800-token full-MSA teacher forward
    # peaked at 30.7 GiB allocated / 31.2 GiB reserved on an A100-40GB (so a live teacher fits one 40 GB card);
    # student-training peak memory was never logged; the checkpoint's own code imports deepspeed, which needs a CUDA
    # toolkit module on GPU nodes; the student uses the checkpoint's built-in LoRA (not PEFT) and a score over all
    # 640 padded-vocabulary logits (T=1.0 for the mutant, 1.5 for the wild type).
    ModelInfo("aido-rag-16b", "aido", "genbio-ai/GB.Protein-RAG-16B-proteingym-dms-zeroshot", 16000,
              ("student", "teacher"), status="planned", needs_structure=True, needs_msa=True,
              shard_across_gpus=True, teacher_backends=("live",), license="GenBio AI Community License",
              output_family_lock="aido", license_note=GENBIO_LICENSE_NOTE, act_gb=20,
              note="Not in v0.1. Outputs may only teach AIDO models: docs/LICENSES.md"),
]}


def get(key: str) -> ModelInfo:
    try:
        return MODELS[key]
    except KeyError:
        raise ValueError(f"unknown model {key!r}; choose from: {', '.join(sorted(MODELS))}") from None


def students() -> list[str]:
    return sorted(k for k, m in MODELS.items() if "student" in m.roles)


def teachers() -> list[str]:
    return sorted(k for k, m in MODELS.items() if "teacher" in m.roles)
