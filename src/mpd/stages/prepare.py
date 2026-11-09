"""Stage `prepare` (CPU): freeze the label-free context tables every later stage reads.

Reads assay sequences and which mutants were measured, never the scores.
"""

from __future__ import annotations

from ..contexts import build_table, locate, union
from .common import Campaign, stage_main, write_json


def run(c: Campaign, task: dict) -> None:
    spec = c.spec
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

    eval_info = {}
    for dataset in spec.eval_data:
        bm = c.benchmark(dataset)
        ids = sorted(a.assay_id for a in bm.assays())
        full = build_table(bm, ids, with_variants=False)
        full.save(c.stem_eval(dataset))
        eval_info[dataset] = {"assays": len(ids), "contexts": len(full), "digest": full.digest()}
        for unit in c.units:
            if c.evaluated_on_own_assay(unit, dataset):
                build_table(bm, c.eval_assay_ids(unit, dataset, ids), with_variants=False).save(c.stem_eval(dataset, unit))

    write_json(c.root / "contexts" / "MANIFEST.json", {
        "train_data": spec.train_data, "units": c.units, "support": c.resolved["support"],
        "union_contexts": len(parent), "union_digest": parent.digest(),
        "unit_contexts": {u: len(t) for u, t in tables.items()},
        "unit_variants": {u: (None if t.var_mt is None else int(len(t.var_mt))) for u, t in tables.items()},
        "eval": eval_info, "dms_labels_opened": False,
    })
    print(f"prepared {len(parent)} training contexts for {len(tables)} unit(s); "
          f"eval contexts: { {d: i['contexts'] for d, i in eval_info.items()} }", flush=True)


def main(argv=None):
    return stage_main(run, __doc__, argv)


if __name__ == "__main__":
    main()
