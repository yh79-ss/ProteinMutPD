"""The run specification: everything a user chooses on the command line."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field

from . import registry
from .constants import DATASETS, METHODS

MAX_SEED = 2**32 - 1       # the training stage seeds numpy with it, which takes nothing larger


class SpecError(ValueError):
    pass


@dataclass(frozen=True)
class RunSpec:
    teacher: str
    student: str
    train_data: str
    method: str                              # direct-all | direct-one | score-all | score-one
    eval_data: tuple[str, ...]
    assays: tuple[str, ...] = ()             # training assays, required for *-one
    teacher_backend: str | None = None       # cache | live; None = the teacher's default
    seed: int = 20260922
    # Optional overrides of the per-student presets (None = use the preset).
    updates: int | None = None
    global_batch: int | None = None
    peak_lr: float | None = None
    kl_weight: float | None = None
    dev_fraction: float | None = None
    train_gpus: int | None = None            # data-parallel width for *-all training
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # A campaign is the same whatever order the datasets and assays were typed in. Keeping them sorted here means the recipe
        # files (spec.json, resolved.json, tasks/) come out equal, so repeating a command in another order finds its campaign.
        object.__setattr__(self, "eval_data", tuple(sorted(self.eval_data, key=str)))
        object.__setattr__(self, "assays", tuple(sorted(self.assays, key=str)))

    # -- derived ----------------------------------------------------------
    @property
    def objective(self) -> str:
        return METHODS[self.method][0]

    @property
    def scope(self) -> str:
        return METHODS[self.method][1]

    @property
    def backend(self) -> str:
        """Teacher backend: the requested one, else `cache` if the teacher offers it, else `live`."""
        return self.teacher_backend or self._default_backend()

    def _default_backend(self) -> str:
        return "cache" if "cache" in registry.get(self.teacher).teacher_backends else "live"

    @property
    def units(self) -> tuple[str, ...]:
        """Independent training runs: one ('all') or one per assay."""
        return ("all",) if self.scope == "all" else self.assays

    def overrides(self) -> dict:
        keys = ("updates", "global_batch", "peak_lr", "kl_weight", "dev_fraction", "train_gpus")
        return {k: getattr(self, k) for k in keys if getattr(self, k) is not None}

    def campaign_id(self) -> str:
        parts = [self.teacher, self.student, f"{self.train_data}-{self.method}"]
        if self.scope == "one":
            if len(self.assays) == 1:
                parts.append(self.assays[0])
            else:
                parts.append(f"{len(self.assays)}assays-" + _short(",".join(sorted(self.assays))))
        parts.append(f"seed{self.seed}")
        if len(registry.get(self.teacher).teacher_backends) > 1:
            parts.append(self.backend)
        if self.overrides():
            parts.append(_short(json.dumps(self.overrides(), sort_keys=True)))
        return "__".join(parts)

    def to_json(self) -> str:
        data = asdict(self)
        data["eval_data"] = list(self.eval_data)
        data["assays"] = list(self.assays)
        try:
            if self.teacher_backend == self._default_backend():
                data["teacher_backend"] = None      # the default spelled out is the same recipe as leaving the option out
        except ValueError:                          # a teacher the registry does not know: validate() says so
            pass
        return json.dumps(data, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "RunSpec":
        data = json.loads(text)
        data["eval_data"] = tuple(data["eval_data"])
        data["assays"] = tuple(data.get("assays", ()))
        return cls(**data)

    # -- validation -------------------------------------------------------
    def validate(self) -> None:
        if self.method not in METHODS:
            raise SpecError(f"--method must be one of {', '.join(METHODS)}; got {self.method!r}")
        if self.train_data not in DATASETS:
            raise SpecError(f"--train-data must be one of {', '.join(DATASETS)}")
        bad = [d for d in self.eval_data if d not in DATASETS]
        if bad or not self.eval_data:
            raise SpecError(f"--eval-data must be one or more of {', '.join(DATASETS)}")
        if len(set(self.eval_data)) != len(self.eval_data):
            raise SpecError("--eval-data lists a dataset twice")
        try:
            teacher, student = registry.get(self.teacher), registry.get(self.student)
        except ValueError as exc:
            raise SpecError(str(exc)) from None
        if "teacher" not in teacher.roles:
            raise SpecError(f"{self.teacher} cannot be a teacher")
        if "student" not in student.roles:
            raise SpecError(f"{self.student} cannot be a student")
        # A licence lock stays true after the model is implemented, so a "planned" message must not hide it.
        if teacher.output_family_lock and student.family != teacher.output_family_lock:
            raise SpecError(
                f"{teacher.key} -> {student.key} is blocked: outputs of {teacher.key} may only be "
                f"distilled into {teacher.output_family_lock} models. {teacher.license_note} "
                f"mpd has no override for this; choose a teacher without an output restriction "
                f"(see `mpd models`). See docs/LICENSES.md.")
        for model in (teacher, student):
            if model.status != "available":
                raise SpecError(f"{model.key} is {model.status} and cannot be used yet. {model.note}")
        if self.backend not in teacher.teacher_backends:
            raise SpecError(
                f"teacher {self.teacher} supports backends {list(teacher.teacher_backends)}; "
                f"got {self.backend!r}")
        if self.scope == "one" and not self.assays:
            raise SpecError(f"--method {self.method} needs --assay <assay id> [...]")
        if self.scope == "all" and self.assays:
            raise SpecError(f"--assay is only valid with a *-one method, not {self.method}")
        if len(set(self.assays)) != len(self.assays):
            raise SpecError("--assay lists an assay twice")
        if self.global_batch is not None and self.global_batch < 1:
            raise SpecError("--global-batch must be positive")
        if self.updates is not None and self.updates < 1:
            raise SpecError("--updates must be positive")
        if self.dev_fraction is not None and not 0 < self.dev_fraction < 0.5:
            raise SpecError("--dev-fraction must be in (0, 0.5)")
        if self.train_gpus is not None and self.scope == "one":
            raise SpecError("--train-gpus applies to *-all methods; one-assay runs use one device")
        if self.kl_weight is not None and self.objective == "score":
            raise SpecError("--kl-weight is only valid with --method direct-all or direct-one. The score methods have no KL term, so "
                            "the value would change nothing. Leave --kl-weight out.")
        if not isinstance(self.seed, int) or not 0 <= self.seed <= MAX_SEED:
            raise SpecError(f"--seed must be a whole number from 0 to {MAX_SEED}, not {self.seed}. Pick a number in that range, "
                            "for example a date such as 20261109.")


def _short(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:8]
