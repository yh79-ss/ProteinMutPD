"""Stage `prepare` (CPU): freeze the label-free context tables every later stage reads.

Reads assay sequences and which mutants were measured, never the scores. When a model of the run reads protein structure
(SaProt) it also turns the structure files into one Foldseek 3Di string per sequence and freezes those.
"""

from __future__ import annotations

import os

from ..contexts import ContextTable, build_table, locate, union
from ..data import models
from ..structure import build, structure_needs
from .common import Campaign, stage_main, write_json


def build_structures(c: Campaign, train_bm, parent: ContextTable, evals: dict):
    """contexts/structures.json: the 3Di strings of every sequence a structure-aware model reads, or None if none does.

    The training dataset is requested first, so a sequence it shares with an evaluation dataset keeps its own structure in
    the base set and the other dataset's string is stored under that dataset (`load_structures(dataset)`)."""
    train_needs, eval_datasets = structure_needs(c.spec)
    if not (train_needs or eval_datasets):
        return None
    requests: dict[str, dict] = {}
    if train_needs:
        requests[c.spec.train_data] = build.assays_for(train_bm, parent.hashes)
    for dataset in eval_datasets:
        bm, full = evals[dataset]
        requests.setdefault(dataset, {}).update(build.assays_for(bm, full.hashes))
    engine = build.open_engine(c.home, cluster=c.resolved.get("cluster"), explicit_home=True)
    structures = build.build_set(requests, c.home, engine)
    structures.save(c.stem_structures())
    return structures


def run(c: Campaign, task: dict) -> None:
    spec = c.spec
    # This is a CPU job: weights that were never fetched, or were purged, fail here minutes after submission instead of in
    # the GPU queue hours later. MPD_ALLOW_HUB=1 (login node or laptop only) lets `models.resolve` use the Hub id instead.
    if os.environ.get("MPD_ALLOW_HUB") != "1":
        models.require_ready(models.needed(spec), c.home, cluster=c.resolved.get("cluster"), explicit_home=True)
    train_bm = c.benchmark(spec.train_data)
    known = {a.assay_id for a in train_bm.assays()}
    missing = [a for a in spec.assays if a not in known]
    if missing:
        raise SystemExit(f"assays not in {spec.train_data}: {', '.join(missing[:5])}"
                         + (f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""))

    dense = c.resolved["support"] == "aa20_dense"      # direct logits: every residue; score: assayed variants only
    tables = {}
    for unit in c.units:
        ids = sorted(known) if unit == "all" else [unit]
        tables[unit] = build_table(train_bm, ids, with_variants=not dense, dense=dense)
    parent = union(list(tables.values()))
    parent.save(c.stem_union())
    for unit, table in tables.items():
        table.union_index = locate(table, parent)
        table.save(c.stem_unit(unit))

    eval_info, evals = {}, {}
    for dataset in spec.eval_data:
        bm = c.benchmark(dataset)
        ids = sorted(a.assay_id for a in bm.assays())
        full = build_table(bm, ids, with_variants=False)
        full.save(c.stem_eval(dataset))
        evals[dataset] = bm, full
        eval_info[dataset] = {"assays": len(ids), "contexts": len(full), "digest": full.digest()}
        for unit in c.units:
            if c.evaluated_on_own_assay(unit, dataset):
                build_table(bm, c.eval_assay_ids(unit, dataset, ids), with_variants=False).save(c.stem_eval(dataset, unit))

    structures = build_structures(c, train_bm, parent, evals)      # before MANIFEST.json, which marks this stage as finished

    manifest = {
        "train_data": spec.train_data, "units": c.units, "support": c.resolved["support"],
        "union_contexts": len(parent), "union_digest": parent.digest(),
        "unit_contexts": {u: len(t) for u, t in tables.items()},
        "unit_variants": {u: (None if t.var_mt is None else int(len(t.var_mt))) for u, t in tables.items()},
        "eval": eval_info, "dms_labels_opened": False,
    }
    if structures is not None:
        manifest["structures"] = structures.manifest_block()
    write_json(c.root / "contexts" / "MANIFEST.json", manifest)
    print(f"prepared {len(parent)} training contexts for {len(tables)} unit(s); "
          f"eval contexts: { {d: i['contexts'] for d, i in eval_info.items()} }", flush=True)
    if structures is not None:
        print(f"structures: {len(structures)} sequences, {structures.masked_fraction():.1%} of residues masked, "
              f"digest {structures.digest()[:12]}", flush=True)


def main(argv=None):
    return stage_main(run, __doc__, argv)


if __name__ == "__main__":
    main()
