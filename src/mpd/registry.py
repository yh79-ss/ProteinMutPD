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
    revision: str | None = None       # pin before a public release
    # If set, outputs of this model (as teacher) may only be distilled into models of
    # this family. Enforced in RunSpec.validate; see docs/LICENSES.md.
    output_family_lock: str | None = None
    license_note: str = ""
    # Memory model used only to choose a GPU shape. weights_gb is bf16 weights;
    # act_gb is a conservative placeholder for activations + LoRA optimizer state at
    # the default microbatch. It has NOT been measured for most models. Measure on
    # your cluster and override with --gpu / --train-gpus.
    act_gb: float = 8.0
    act_gb_measured: bool = False
    note: str = ""

    @property
    def weights_gb(self) -> float:
        return self.params_m * 2 / 1000.0


GENBIO_LICENSE_NOTE = (
    "GenBio AI Community License, clause 1.2.2: Outputs (including logits and intermediate "
    "activations) may not be used to train similar ML models or for model distillation, "
    "except to train the GenBio Materials themselves and their downstream Derivative Works. "
    "Non-commercial use only."
)

MODELS: dict[str, ModelInfo] = {m.key: m for m in [
    ModelInfo("esm2-8m", "esm2", "facebook/esm2_t6_8M_UR50D", 8, ("student", "teacher"), act_gb=2),
    ModelInfo("esm2-35m", "esm2", "facebook/esm2_t12_35M_UR50D", 35, ("student", "teacher"), act_gb=3),
    ModelInfo("esm2-150m", "esm2", "facebook/esm2_t30_150M_UR50D", 150, ("student", "teacher"), act_gb=6),
    ModelInfo("esm2-650m", "esm2", "facebook/esm2_t33_650M_UR50D", 650, ("student", "teacher"), act_gb=12,
              note="Sequence only."),
    ModelInfo("esm2-3b", "esm2", "facebook/esm2_t36_3B_UR50D", 3000, ("student", "teacher"), act_gb=16),
    ModelInfo("saprot-35m", "saprot", "westlake-repl/SaProt_35M_AF2", 35, ("student", "teacher"),
              needs_structure=True, act_gb=3),
    ModelInfo("saprot-650m", "saprot", "westlake-repl/SaProt_650M_AF2", 650, ("student", "teacher"),
              needs_structure=True, act_gb=12,
              note="Sequence + Foldseek 3Di structure tokens; AF2 pLDDT<70 masked."),
    ModelInfo("aido-rag-16b", "aido", "genbio-ai/GB.Protein-RAG-16B-proteingym-dms-zeroshot", 16000,
              ("student", "teacher"), status="planned", needs_structure=True, needs_msa=True,
              shard_across_gpus=True, teacher_backends=("cache", "live"),
              output_family_lock="aido", license_note=GENBIO_LICENSE_NOTE, act_gb=20,
              note="Planned. 32.1 GB of bf16 weights. A single 12,800-token teacher forward peaked at "
                   "33.2 GB allocated / 36.3 GB reserved on an H200; student training peak memory was "
                   "never logged. Needs deepspeed + GenBio modelgenerator/openfold; the student uses a "
                   "native LoRA (not PEFT) and a 640-wide softmax score."),
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
