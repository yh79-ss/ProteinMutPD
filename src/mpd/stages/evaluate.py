"""Stage `evaluate` (CPU): the only stage that opens DMS labels.

Joins the scored AA20 logits to each assay's measured variants, turns them into variant scores with two
readouts, and reports assay-macro Spearman for the frozen base student and for every trained unit.

  delta   s[mut] - s[wt]                            the zero-shot masked-marginal convention
  dual    log softmax(s/1.0)[mut] - log softmax(s/1.5)[wt]   the readout Score-KD trains

Both readouts are always reported. Which one is called primary (delta for direct-logit runs, dual for score runs)
is the maintainer's assumption, not a confirmed convention; summary.json says so in `primary_readout_basis`.
"""

from __future__ import annotations

import csv

import numpy as np
from scipy.stats import spearmanr

from ..contexts import ContextTable, sequence_hash
from ..data.benchmarks import Benchmark
from ..objectives import readout_delta, readout_dual
from .common import SHARD_JSON, Campaign, provenance, shard_name, stage_main, write_json

READOUTS = {"delta": readout_delta, "dual": readout_dual}
PRIMARY = {"direct_logits": "delta", "score": "dual"}
PRIMARY_BASIS = "assumed, not confirmed: delta for direct_logits runs, dual for score runs; both readouts are always reported"


def shard_marks(folder) -> list[tuple[int, int]]:
    """(shard, num) of every finished shard file in `folder`."""
    return sorted((int(m.group(1)), int(m.group(2))) for p in folder.glob("shard_*.json") if (m := SHARD_JSON.search(p.name)))


def assemble(c: Campaign, dataset: str, unit: str) -> tuple[ContextTable, np.ndarray]:
    """All shards of (dataset, unit) as one [N, 20] array aligned with the evaluation context table.

    Only shard files whose count is the planned one are read; files of another count are ignored."""
    table = ContextTable.load(c.stem_eval_for(unit, dataset))
    folder = c.score_dir(dataset, unit)
    marks = shard_marks(folder)
    num = c.expected_shards(dataset, unit)
    again = "Run the score stage again for this unit and dataset (`mpd status` shows which tasks failed)."
    if num is None:                           # planned before the count was recorded: the files must agree among themselves
        counts = sorted({n for _, n in marks})
        if len(counts) != 1:
            raise RuntimeError(f"{folder}: shards incomplete (shard files exist for counts {counts or 'none'}, "
                               f"expected exactly one). {again}")
        num = counts[0]
    have = sorted(i for i, n in marks if n == num)
    if have != list(range(num)):
        ignored = sorted({n for _, n in marks if n != num})
        raise RuntimeError(f"{folder}: shards incomplete (have {have} of {num}; missing {sorted(set(range(num)) - set(have))})"
                           + (f"; shard files for other counts {ignored} are ignored" if ignored else "") + f". {again}")
    scores = np.full((len(table), 20), np.nan, dtype=np.float32)
    for shard in range(num):
        with np.load(folder / f"{shard_name(shard, num)}.npz") as z:
            if "num_shards" in z and int(z["num_shards"]) != num:
                raise RuntimeError(f"{folder}: {shard_name(shard, num)}.npz was written for {int(z['num_shards'])} shards")
            scores[z["rows"]] = z["scores"]
    if np.isnan(scores).any():
        raise RuntimeError(f"{folder}: some evaluation contexts were never scored")
    return table, scores


def spearman(pred: np.ndarray, truth: np.ndarray) -> float:
    if len(pred) < 3 or np.ptp(pred) == 0 or np.ptp(truth) == 0:
        return float("nan")
    return float(spearmanr(pred, truth).statistic)


def assay_spearman(bm: Benchmark, assay_id: str, table: ContextTable, scores: np.ndarray, where: dict) -> dict[str, float]:
    assay = bm.assay(assay_id)
    v = bm.variants(assay_id)
    h = sequence_hash(assay.sequence)
    rows = np.asarray([where[(h, int(p))] for p in v.pos])
    z = scores[rows]
    truth = bm.labels(assay_id)
    if len(truth) != len(v):
        raise RuntimeError(f"{assay_id}: {len(truth)} labels for {len(v)} variants")
    return {name: spearman(fn(z, v.wt, v.mt), truth) for name, fn in READOUTS.items()} | {"n": len(v)}


def macro(values) -> float:
    finite = [x for x in values if np.isfinite(x)]
    return float(np.mean(finite)) if finite else float("nan")


def run(c: Campaign, task: dict) -> None:
    primary = PRIMARY[c.spec.objective]
    per_assay: list[dict] = []
    totals: dict[str, int] = {}
    for dataset in c.spec.eval_data:
        bm = c.benchmark(dataset)
        all_ids = sorted(a.assay_id for a in bm.assays())
        totals[dataset] = len(all_ids)
        for unit in ["base", *c.units]:
            table, scores = assemble(c, dataset, unit)
            where = {k: i for i, k in enumerate(table.keys())}
            for aid in c.eval_assay_ids(unit, dataset, all_ids):
                res = assay_spearman(bm, aid, table, scores, where)
                for readout in READOUTS:
                    per_assay.append({"dataset": dataset, "unit": unit, "assay_id": aid, "readout": readout,
                                      "n_variants": res["n"], "spearman": res[readout]})
    out = c.evaluate_dir
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "per_assay.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "unit", "assay_id", "readout", "n_variants", "spearman"])
        w.writeheader()
        w.writerows(per_assay)

    by = {}
    for row in per_assay:
        by.setdefault((row["dataset"], row["unit"], row["readout"]), {})[row["assay_id"]] = row["spearman"]
    summary = []
    for (dataset, unit, readout), values in sorted(by.items()):
        entry = {"dataset": dataset, "unit": unit, "readout": readout, "assays": len(values),
                 "assays_finite": sum(bool(np.isfinite(x)) for x in values.values()),
                 "macro_spearman": macro(values.values()), "primary": readout == primary}
        if unit != "base":
            base = by[(dataset, "base", readout)]
            entry["base_macro_same_assays"] = macro(base[a] for a in values)
            entry["delta_vs_base"] = entry["macro_spearman"] - entry["base_macro_same_assays"]
        summary.append(entry)
    write_json(out / "summary.json", {"campaign": c.resolved["campaign_id"], "objective": c.spec.objective,
                                      "primary_readout": primary, "primary_readout_basis": PRIMARY_BASIS,
                                      "eval_data": list(c.spec.eval_data), "units": ["base", *c.units],
                                      "assays_in_dataset": totals, "results": summary, "dms_labels_opened": True,
                                      "provenance": provenance()})
    for e in summary:
        if e["primary"]:
            extra = f"  base {e['base_macro_same_assays']:.4f}  delta {e['delta_vs_base']:+.4f}" if "delta_vs_base" in e else ""
            print(f"{e['dataset']:<12}{e['unit']:<28}{e['assays']:>4} assays  macro Spearman {e['macro_spearman']:.4f}{extra}", flush=True)


def main(argv=None):
    return stage_main(run, __doc__, argv)


if __name__ == "__main__":
    main()
