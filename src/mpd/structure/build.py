"""Structure files -> one SaProt state string per assay sequence, frozen as a `StructureSet`.

The string has the length of the full assay sequence: the 3Di letter of each residue the structure covers where its
confidence is at least 70 pLDDT, `#` everywhere else (low confidence, outside the structure, unknown state). The
amino-acid channel of the model input is always the assay's own sequence. Torch-free; never touches labels.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from ..contexts import sequence_hash
from ..data import venus_structures as vs
from ..data.benchmarks import Assay, Benchmark
from . import foldseek, pdb
from .align import AlignmentError, align_to_sequence
from .foldseek import Chain, EngineError, FoldseekEngine, FoldseekError, normalise_states, sha256_file
from .set import StructureSet
from .vocab import STATES

THRESHOLD = 70.0                       # pLDDT below this is masked; 70.0 itself is kept
LEGACY_ENV = "MPD_STRUCTURE_LEGACY_MASK"
ENGINE_ENV = "MPD_3DI_ENGINE"
CACHE_SCHEMA = 1


class StructureError(RuntimeError):
    pass


@dataclass(frozen=True)
class Policy:
    """How confidence turns into masking. `legacy_mask` reproduces the earlier study, which compared the raw B-factor
    column with 70 whatever its scale and so masked every residue of an ESMFold model (values 0 to 1)."""
    threshold: float = THRESHOLD
    legacy_mask: bool = False

    @classmethod
    def from_env(cls) -> "Policy":
        return cls(legacy_mask=os.environ.get(LEGACY_ENV) == "1")

    def describe(self) -> dict:
        return {"plddt_threshold": self.threshold, "mask_rule": "strictly below the threshold",
                "confidence_rule": "B-factor of the CA atom", "scales": dict(vs.SCALE_TO_PERCENT),
                "experimental_never_masked": True, "legacy_mask": self.legacy_mask, "states": STATES}


def confidence(bfactor: Iterable[float], scale: str | None, policy: Policy) -> list[float] | None:
    """pLDDT on the 0-100 scale per residue; None for experimental B-factors, which are not confidences."""
    if scale not in vs.SCALE_TO_PERCENT:
        raise StructureError(f"unknown structure confidence scale {scale!r}; expected one of {', '.join(vs.SCALE_TO_PERCENT)}")
    factor = vs.SCALE_TO_PERCENT[scale]
    if factor is None:
        return None
    if policy.legacy_mask:
        factor = 1.0
    return [v * factor for v in bfactor]


def _pick_chain(chains: list[Chain], wanted: str | None, name: str) -> Chain:
    if wanted is not None:
        for c in chains:
            if c.chain == wanted:
                return c
        raise StructureError(f"{name} has no chain {wanted!r} (Foldseek found: {', '.join(c.chain for c in chains)})")
    if len(chains) > 1:
        raise StructureError(f"{name} has {len(chains)} chains ({', '.join(c.chain for c in chains)}); "
                             f"set the assay's structure_chain to choose one")
    return chains[0]


def structure_string(assay: Assay, engine, policy: Policy | None = None, *,
                     chains: list[Chain] | None = None) -> tuple[str, dict]:
    """The state string of `assay.sequence` and a provenance record. `chains` skips the engine call (cache)."""
    policy = policy or Policy.from_env()
    path = Path(assay.structure) if assay.structure is not None else None
    if path is None or not path.is_file():
        raise StructureError(f"{assay.assay_id}: structure file {path or '(none recorded)'} not found")
    try:
        chain = _pick_chain(chains if chains is not None else engine.chains(path), assay.structure_chain, path.name)
        ca = pdb.ca_records(path, chain.chain)
        if ca.aa != chain.aa:
            i = next((k for k, (a, b) in enumerate(zip(ca.aa, chain.aa)) if a != b), min(len(ca.aa), len(chain.aa)))
            raise StructureError(f"the CA records ({len(ca.aa)} residues) and Foldseek ({len(chain.aa)}) of {path.name} "
                                 f"disagree, first at residue {i + 1}")
        states, unknown = normalise_states(chain.states)
        aln = align_to_sequence(assay.sequence, chain.aa, assay.structure_span)
        conf = confidence(ca.bfactor, assay.structure_scale, policy)
    except (AlignmentError, pdb.PdbError, EngineError, StructureError) as exc:
        raise StructureError(f"{assay.assay_id}: {exc}") from exc
    out = ["#"] * len(assay.sequence)
    low = 0
    for j in range(aln.n):
        k = aln.chain_start + j
        if conf is not None and conf[k] < policy.threshold:
            low += 1
            continue
        out[aln.seq_start + j] = states[k]
    text = "".join(out)
    info = {"assay": assay.assay_id, "structure": path.name, "chain": chain.chain, "scale": assay.structure_scale,
            "length": len(text), "residues": aln.n, "offset": aln.seq_start, "chain_offset": aln.chain_start,
            "substitutions": list(aln.substitutions), "masked_low_confidence": low, "unknown_states": unknown,
            "uncovered": len(text) - aln.n, "masked": text.count("#")}
    return text, info


# -- engines and the per-file cache ---------------------------------------------------------------------

def missing_foldseek(home: Path | str | None, cluster: str | None = None, *, explicit_home: bool = False) -> str:
    given = foldseek.override()
    if given is not None:
        return f"MPD_FOLDSEEK={given} is not an executable file (and no fallback is used while it is set)"
    where = foldseek.pinned_path(home) if home else "<home>/tools/" + foldseek.FOLDSEEK.dirname
    return f"Foldseek was not found ({where}). Run on a login node: {foldseek.fix_command(cluster, home, explicit_home=explicit_home)}"


def open_engine(home: Path | str | None = None, name: str | None = None, *, cluster: str | None = None):
    """The 3Di source: Foldseek (default) or, with MPD_3DI_ENGINE=biotite, its NumPy port."""
    name = (name or os.environ.get(ENGINE_ENV) or "foldseek").strip().lower()
    if name == "foldseek":
        binary = foldseek.locate(home)
        if binary is None:
            raise FoldseekError(missing_foldseek(home, cluster))
        return FoldseekEngine(binary)
    if name == "biotite":
        from .biotite_engine import BiotiteEngine
        return BiotiteEngine()
    raise ValueError(f"{ENGINE_ENV} must be 'foldseek' or 'biotite'; got {name!r}")


class ChainCache:
    """`<home>/cache/3di/<engine id>/<sha256 of the structure file>.json`: shared by every campaign of a home."""

    def __init__(self, home: Path | str | None, engine_id: str):
        self.dir = Path(home) / "cache" / "3di" / engine_id if home else None

    def get(self, file_sha: str) -> list[Chain] | None:
        if self.dir is None:
            return None
        try:
            body = json.loads((self.dir / f"{file_sha}.json").read_text(encoding="utf-8"))
            if body.get("schema") != CACHE_SCHEMA:
                return None
            return [Chain(c["name"], c["chain"], c["aa"], c["states_upper"]) for c in body["chains"]]
        except (OSError, ValueError, KeyError, TypeError):
            return None                   # absent or damaged: recompute

    def put(self, file_sha: str, chains: list[Chain]) -> None:
        if self.dir is None:
            return
        self.dir.mkdir(parents=True, exist_ok=True)
        body = {"schema": CACHE_SCHEMA, "chains": [{"name": c.name, "chain": c.chain, "aa": c.aa, "states_upper": c.states} for c in chains]}
        target = self.dir / f"{file_sha}.json"
        tmp = target.with_name(f"{target.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        tmp.write_text(json.dumps(body) + "\n", encoding="utf-8")
        tmp.replace(target)


def default_threads() -> int:
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 1
    return max(1, min(n, 16))


# -- sets ---------------------------------------------------------------------------------------------------

def assays_for(benchmark: Benchmark, hashes: Iterable[str]) -> dict[str, Assay]:
    """{sequence hash: the assay that carries it}. Assays sharing a sequence must agree on its structure."""
    want = set(hashes)
    found: dict[str, Assay] = {}
    for a in sorted(benchmark.assays(), key=lambda a: a.assay_id):
        h = sequence_hash(a.sequence)
        if h not in want:
            continue
        first = found.setdefault(h, a)
        same = (a.structure, a.structure_span, a.structure_scale, a.structure_chain) == \
               (first.structure, first.structure_span, first.structure_scale, first.structure_chain)
        if not same:
            raise StructureError(f"{benchmark.name}: assays {first.assay_id} and {a.assay_id} share a sequence but name "
                                 f"different structures; a sequence has one structure per dataset")
    missing = want - set(found)
    if missing:
        raise StructureError(f"{len(missing)} sequence(s) of the context tables are not in the {benchmark.name} benchmark")
    return found


def _flatten(needed: Mapping) -> dict[str, dict[str, Assay]]:
    if needed and isinstance(next(iter(needed.values())), Assay):
        return {"": dict(needed)}
    return {d: dict(m) for d, m in needed.items()}


def build_set(needed: Mapping, home: Path | str | None = None, engine=None, *, policy: Policy | None = None,
              threads: int | None = None, log: Callable[[str], None] = print) -> StructureSet:
    """Structure strings for every sequence in `needed` ({hash: Assay} or {dataset: {hash: Assay}}), as a StructureSet.

    `home` holds the 3Di cache and the Foldseek install (None: no cache). Foldseek runs once per structure file."""
    # A sequence two datasets share keeps the first dataset's string; a different string from another dataset goes
    # under scopes[dataset] (StructureSet.scoped), so each dataset is scored with its own structure.
    policy = policy or Policy.from_env()
    datasets = _flatten(needed)
    engine = engine or open_engine(home)
    tasks = [(d, h, a) for d, m in datasets.items() for h, a in sorted(m.items())]
    for d, h, a in tasks:
        if sequence_hash(a.sequence) != h:
            raise ValueError(f"{a.assay_id}: the hash {h[:12]} is not the sha256 of the assay's sequence")
        if a.structure is None:
            raise StructureError(f"{a.assay_id}: this assay has no structure file recorded")
    files = sorted({Path(a.structure) for _, _, a in tasks})
    cache = ChainCache(home, engine.id)
    errors: list[str] = []
    chains: dict[Path, list[Chain]] = {}
    shas: dict[Path, str] = {}

    def one(path: Path):
        sha = sha256_file(path)
        got = cache.get(sha)
        if got is None:
            got = engine.chains(path)
            cache.put(sha, got)
        return path, sha, got

    log(f"3Di for {len(files)} structure file(s) with {engine.id}")
    present = [p for p in files if p.is_file()]
    errors.extend(f"structure file {p} not found" for p in files if not p.is_file())
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(threads or default_threads(), len(present) or 1))) as pool:
        futures = {pool.submit(one, p): p for p in present}
        for fut in concurrent.futures.as_completed(futures):
            try:
                path, sha, got = fut.result()
                chains[path], shas[path] = got, sha
            except (EngineError, OSError) as exc:
                errors.append(f"{futures[fut].name}: {exc}")

    states: dict[str, str] = {}
    scopes: dict[str, dict[str, str]] = {}
    per_sequence: dict[str, dict] = {}
    conflicts: list[dict] = []
    owner: dict[str, str] = {}
    for dataset, h, a in tasks:
        path = Path(a.structure)
        if path not in chains:
            continue
        try:
            text, info = structure_string(a, engine, policy, chains=chains[path])
        except StructureError as exc:
            errors.append(str(exc))
            continue
        info = {"dataset": dataset, "file_sha256": shas[path], **info}
        if h not in states:
            states[h], owner[h] = text, dataset
            per_sequence[h] = info
        elif states[h] != text:
            scopes.setdefault(dataset, {})[h] = text
            conflicts.append({"sequence_sha256": h, "kept_dataset": owner[h], "other_dataset": dataset,
                              "kept_structure": per_sequence[h]["structure"], "other_structure": info["structure"]})
    if errors:
        shown = "\n  ".join(sorted(errors)[:8])
        more = f"\n  (+{len(errors) - 8} more)" if len(errors) > 8 else ""
        raise StructureError(f"{len(errors)} of {len(tasks)} structure(s) could not be turned into state strings:\n  {shown}{more}")
    residues = sum(len(s) for s in states.values())
    meta = {"schema": 1, "engine": engine.describe(), "policy": policy.describe(), "sequences": len(states),
            "residues": residues, "masked_residues": sum(s.count("#") for s in states.values()),
            "unknown_states": sum(i["unknown_states"] for i in per_sequence.values()),
            "datasets": {d: len(m) for d, m in datasets.items()}, "conflicts": conflicts,
            "per_sequence": per_sequence, "dms_labels_opened": False}
    if conflicts:
        log(f"{len(conflicts)} sequence(s) have a different structure in a second dataset; each dataset keeps its own")
    return StructureSet(states, meta, scopes)
