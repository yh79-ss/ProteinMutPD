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


def _fetch_cluster(c: Campaign) -> str | None:
    """The profile name for the --cluster of a fetch command; None for a profile loaded from a YAML file, which `mpd fetch` cannot find by name."""
    from ..slurm.cluster import available_clusters
    name = c.resolved.get("cluster")
    return name if name in available_clusters() else None


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
    engine = build.open_engine(c.home, cluster=_fetch_cluster(c), explicit_home=True)
    structures = build.build_set(requests, c.home, engine)
    structures.save(c.stem_structures())
    return structures


MIN_TRAIN_CONTEXTS = 10      # a unit with fewer still trains, but a handful of contexts decides its student and its checkpoint


def unit_split(c: Campaign, tables: dict) -> dict[str, list[int]]:
    """[training, development] contexts of each unit, split exactly as `train` will split them (the split never reads labels)."""
    fraction = c.resolved["dev_fraction"]
    return {unit: [int((~dev).sum()), int(dev.sum())] for unit, table in tables.items() for dev in [table.dev_mask(fraction)]}


def check_split(split: dict[str, list[int]]) -> None:
    """Fail now, on a CPU node, for a unit `train` would reject after the teacher stage; say so for a unit that trains on very little."""
    unsplittable = [u for u, (train, dev) in split.items() if not train or not dev]
    if unsplittable:
        shown = ", ".join(unsplittable[:5]) + (f" (+{len(unsplittable) - 5} more)" if len(unsplittable) > 5 else "")
        raise SystemExit(f"{len(unsplittable)} training unit(s) have too few measured positions to be split into training and development "
                         f"contexts: {shown}. A unit needs at least two. Leave them out of --assay / --assays-file and submit again "
                         "(that is a new campaign).")
    thin = [u for u, (train, _) in split.items() if train < MIN_TRAIN_CONTEXTS]
    if thin:
        shown = ", ".join(f"{u} ({split[u][0]} training, {split[u][1]} development)" for u in thin[:5]) + (f" (+{len(thin) - 5} more)" if len(thin) > 5 else "")
        print(f"note: {len(thin)} unit(s) train on fewer than {MIN_TRAIN_CONTEXTS} contexts, so a few positions decide the student and its "
              f"checkpoint: {shown}", flush=True)


def run(c: Campaign, task: dict) -> None:
    spec = c.spec
    # This is a CPU job: weights that were never fetched, or were purged, fail here minutes after submission instead of in
    # the GPU queue hours later. MPD_ALLOW_HUB=1 (login node or laptop only) lets `models.resolve` use the Hub id instead.
    if os.environ.get("MPD_ALLOW_HUB") != "1":
        models.require_ready(models.needed(spec), c.home, cluster=_fetch_cluster(c), explicit_home=True)
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
    split = unit_split(c, tables)
    check_split(split)
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
        "unit_split": split, "dev_fraction": c.resolved["dev_fraction"],
        "eval": eval_info, "dms_labels_opened": False,
    }
    if structures is not None:
        manifest["structures"] = structures.manifest_block()
    write_json(c.root / "contexts" / "MANIFEST.json", manifest)
    print(f"prepared {len(parent)} training contexts for {len(tables)} unit(s); "
          f"eval contexts: { {d: i['contexts'] for d, i in eval_info.items()} }", flush=True)
    dev_total, all_total = sum(d for _, d in split.values()), sum(t + d for t, d in split.values())
    print(f"development split: {dev_total} of {all_total} contexts ({dev_total / max(1, all_total):.1%}); --dev-fraction {c.resolved['dev_fraction']:g} "
          "is the target, and every sequence keeps at least one training context and holds out at least min(5, n-1) development contexts", flush=True)
    if structures is not None:
        print(f"structures: {len(structures)} sequences, {structures.masked_fraction():.1%} of residues masked, "
              f"digest {structures.digest()[:12]}", flush=True)


def main(argv=None):
    return stage_main(run, __doc__, argv)


if __name__ == "__main__":
    main()
