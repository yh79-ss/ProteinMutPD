"""Structure inputs for SaProt: a Foldseek 3Di string per assay sequence. Torch-free; the model class is in students/saprot.py.

    vocab    the 446-token vocabulary and token ids         pdb     CA records of a PDB file
    align    structure chain -> assay sequence              foldseek  run Foldseek, parse its output
    build    state strings, masking policy, the 3Di cache   set     StructureSet and structures.json
    tools    fetch the pinned Foldseek release
"""

from __future__ import annotations

import os
from pathlib import Path


def _structure_check(dataset: str, home: Path, cluster: str | None, explicit_home: bool = False):
    """Is the structure data of `dataset` on disk? One Check."""
    from ..checks import Check
    from ..data import sources
    from ..data import venus_structures as vs
    from .foldseek import where_flags
    where = where_flags(cluster, home, explicit_home)
    if dataset == "proteingym":
        folder = home / "data" / sources.PROTEINGYM.dirname / "ProteinGym_AF2_structures"
        count = len(list(folder.glob("*.pdb"))) if folder.is_dir() else 0
        if count:
            return Check("structures:proteingym", "ok", f"{count} ProteinGym AF2 structure files")
        archive = next(f for f in sources.PROTEINGYM.files if f.name == "ProteinGym_AF2_structures.zip")
        return Check("structures:proteingym", "error",
                     f"ProteinGym structures not found under {folder}. They are the official AF2 models "
                     f"({archive.name}, {archive.size / 1e6:.1f} MB), part of the ProteinGym download. "
                     f"Run on a login node: mpd fetch proteingym {where}", fix=f"mpd fetch proteingym {where}")
    manifest = home / "data" / sources.VENUS_STRUCTURES.dirname / vs.MANIFEST_NAME
    if manifest.is_file():
        return Check("structures:venusmuthub", "ok", f"Venus structure manifest {manifest}")
    fix = f"mpd fetch venus-structures {where}"
    if sources.VENUS_STRUCTURES.zenodo_record is None:
        return Check("structures:venusmuthub", "error",
                     f"VenusMutHub structures not found under {manifest.parent}. The structure set is not published yet, so "
                     f"SaProt can use VenusMutHub only with a copy of the deposit: {fix} --from-dir <folder with the deposit files>",
                     fix=f"{fix} --from-dir <folder with the deposit files>")
    return Check("structures:venusmuthub", "error", f"VenusMutHub structures not found under {manifest.parent}. "
                 f"Run on a login node: {fix}", fix=fix)


def _engine_check(home: Path, cluster: str | None, explicit_home: bool = False):
    from ..checks import Check
    from ..data.sources import FOLDSEEK, FOLDSEEK_REFERENCE_BUILD
    from . import build, foldseek
    if (os.environ.get(build.ENGINE_ENV) or "").strip().lower() == "biotite":
        try:
            import biotite
        except ImportError:
            return Check("3di-engine", "error", f"{build.ENGINE_ENV}=biotite but biotite is not installed",
                         fix="pip install biotite")
        return Check("3di-engine", "warn", f"3Di from biotite {biotite.__version__} ({build.ENGINE_ENV}=biotite), a NumPy port "
                     f"of Foldseek's encoder: identical to Foldseek on 674 of 675 structures in one test, one residue "
                     f"different in the other. Foldseek defines the recorded results.")
    fix = foldseek.fix_command(cluster, home, explicit_home=explicit_home)
    binary = foldseek.locate(home)
    if binary is None:
        return Check("foldseek", "error", build.missing_foldseek(home, cluster, explicit_home=explicit_home), fix=fix)
    try:
        shown = foldseek.version(binary)
    except foldseek.FoldseekError as exc:
        return Check("foldseek", "error", f"Foldseek at {binary} does not run: {exc}", fix=fix)
    sha = foldseek.sha256_file(binary)
    if sha == FOLDSEEK.binary_sha256:
        return Check("foldseek", "ok", f"Foldseek {shown} (pinned release {FOLDSEEK.release}) at {binary}")
    if sha == FOLDSEEK_REFERENCE_BUILD["binary_sha256"]:
        return Check("foldseek", "ok", f"Foldseek {shown} (the 2026-09-09 build behind the recorded results) at {binary}")
    return Check("foldseek", "warn", f"Foldseek {shown} at {binary} (sha256 {sha[:12]}) is not the pinned release "
                 f"{FOLDSEEK.release}; its 3Di strings have not been compared with the recorded results. Keep it, or "
                 f"install the pinned release.", fix=fix)


def structure_needs(spec) -> tuple[bool, tuple[str, ...]]:
    """(do the training sequences need structure strings, the evaluation datasets that do).

    A structure-aware student reads structure everywhere; a live structure-aware teacher only on the training sequences."""
    from .. import registry
    student, teacher = registry.get(spec.student), registry.get(spec.teacher)
    train = student.needs_structure or (teacher.needs_structure and spec.backend == "live")
    return train, tuple(spec.eval_data) if student.needs_structure else ()


def preflight_checks(spec, home, cluster=None, *, explicit_home: bool = False) -> list:
    """Findings for a run that uses a structure-aware model: Foldseek and the structure data of every dataset it reads.
    Weights are reported by `mpd.data.models.preflight_checks` (one check per model the run loads, SaProt included)."""
    cluster = getattr(cluster, "name", cluster)
    train, evaluation = structure_needs(spec)
    datasets = list(dict.fromkeys(([spec.train_data] if train else []) + list(evaluation)))
    if not datasets:
        return []
    return ([_engine_check(Path(home), cluster, explicit_home)]
            + [_structure_check(d, Path(home), cluster, explicit_home) for d in datasets])
