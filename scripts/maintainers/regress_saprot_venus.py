#!/usr/bin/env python3
"""Regression of the SaProt code against the earlier study's recorded VenusMutHub outputs (maintainers only; CPU).

Part `structures`: build the 3Di structure strings of the 489 Venus proteins with a Foldseek binary and compare them
string for string with the earlier study's frozen strings, under two masking rules: the earlier one (`legacy`, which
compared the raw B-factor with 70 whatever its scale and so masked every residue of an ESMFold model) and the corrected
one (mpd's default). Part `scores`: run SaProt-650M on the CPU for a subset of positions and compare with the recorded
base-model outputs of the same positions (macro Spearman over the subset's assays, per-position rank agreement, largest
difference of any value).

    PYTHONPATH=src python scripts/maintainers/regress_saprot_venus.py \
        --home <home with data/venusmuthub-v1 and data/venusmuthub-structures-v1> --foldseek <binary> \
        --plan <earlier PLAN.json> --predictions <earlier predictions/saprot_base> --model <SaProt-650M folder> \
        [--part structures|scores|all] [--out report.json]

Gates (from the Phase-1 measurements): the 489 legacy strings equal the frozen ones when the Foldseek build is the one
the earlier study used; mean per-position Spearman of the 19 mutants >= 0.995; largest |difference| <= 0.25 (the recorded
values are bf16 logits from a GPU); macro Spearman of the assays with >= 30 variants within 3e-3. A different Foldseek
build is not a failure of the code: the report then says how many residues differ. Nothing is submitted or uploaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from mpd.constants import AA20
from mpd.contexts import ContextTable, sequence_hash
from mpd.data.benchmarks import open_benchmark
from mpd.data.sources import FOLDSEEK, FOLDSEEK_REFERENCE_BUILD
from mpd.structure import build, foldseek
from mpd.structure.set import StructureSet

EXPECTED_DIGESTS = {"frozen_legacy": "2a99932edcc2fda67bce95fb036342efc28580fa6f54869e8c824507beb6a96d",
                    "corrected": "a885bf3c093c724109802405cb224a0fa2a60fa8a18bcdf54008e6850b6af7ff"}
TYPES = ("afdb", "experimental", "de_novo")                 # the earlier study's names for AlphaFold DB / experimental / ESMFold
GATES = {"per_position_rho": 0.995, "max_abs_delta": 0.25, "macro_diff": 3e-3}


def say(*parts) -> None:
    print(*parts, flush=True)


def digest_by_id(rows: dict[str, str]) -> str:
    return hashlib.sha256("".join(f"{k}\t{v}\n" for k, v in sorted(rows.items())).encode()).hexdigest()


def rho(a, b) -> float:
    return float(spearmanr(a, b).statistic)


# -- part 1: structure strings -------------------------------------------------------------------------------------------

def structure_part(bm, records, engine, work) -> tuple[dict, dict[str, StructureSet]]:
    assays = bm.assays()
    needed = {"venusmuthub": build.assays_for(bm, [sequence_hash(a.sequence) for a in assays])}
    by_protein = {a.protein_id: sequence_hash(a.sequence) for a in assays}
    sets = {"legacy": build.build_set(needed, work, engine, policy=build.Policy(legacy_mask=True), log=say),
            "corrected": build.build_set(needed, work, engine, policy=build.Policy(), log=say)}
    frozen = {pid: r["structure"] for pid, r in records.items()}
    strings = {name: {pid: s.states[by_protein[pid]] for pid in records} for name, s in sets.items()}
    report: dict = {"engine": engine.describe(), "proteins": len(records)}
    for name in ("legacy", "corrected"):
        per_type = defaultdict(lambda: {"proteins": 0, "equal": 0, "residues": 0, "differing_residues": 0, "masked": 0, "frozen_masked": 0})
        for pid, r in records.items():
            row, text = per_type[r["structure_type"]], strings[name][pid]
            row["proteins"] += 1
            row["equal"] += text == frozen[pid]
            row["residues"] += len(text)
            row["differing_residues"] += sum(a != b for a, b in zip(text, frozen[pid]))
            row["masked"] += text.count("#")
            row["frozen_masked"] += frozen[pid].count("#")
        report[name] = {"by_type": dict(per_type), "equal_to_frozen": sum(v["equal"] for v in per_type.values()),
                        "masked_residues": sum(v["masked"] for v in per_type.values()),
                        "residues": sum(v["residues"] for v in per_type.values()), "digest": digest_by_id(strings[name])}
    report["frozen"] = {"digest": digest_by_id(frozen), "masked_residues": sum(s.count("#") for s in frozen.values())}
    same_build = engine.binary_sha256 == FOLDSEEK_REFERENCE_BUILD["binary_sha256"]
    report["gate"] = {
        "foldseek_is_the_build_of_the_earlier_study": same_build,
        "legacy_equals_frozen_489": report["legacy"]["equal_to_frozen"] == len(records),
        "frozen_digest_as_recorded": report["frozen"]["digest"] == EXPECTED_DIGESTS["frozen_legacy"],
        "corrected_digest_as_recorded": report["corrected"]["digest"] == EXPECTED_DIGESTS["corrected"],
        "corrected_equals_frozen_except_esmfold": all(
            v["equal"] == v["proteins"] for t, v in report["corrected"]["by_type"].items() if t != "de_novo")}
    for name in ("legacy", "corrected"):
        say(f"[structures] {name}: {report[name]['equal_to_frozen']} of {len(records)} strings equal the frozen ones; "
            f"masked {report[name]['masked_residues']} of {report[name]['residues']} residues; "
            + "; ".join(f"{t} {v['equal']}/{v['proteins']}" for t, v in sorted(report[name]["by_type"].items())))
    say(f"[structures] digests: frozen {report['frozen']['digest'][:16]}  legacy {report['legacy']['digest'][:16]}  "
        f"corrected {report['corrected']['digest'][:16]}")
    return report, sets


# -- part 2: scores ------------------------------------------------------------------------------------------------------

def choose_subset(records: dict, budget: int, max_length: int) -> list[str]:
    """Per structure type, the shortest proteins (<= max_length residues) while their recorded positions fit `budget`."""
    chosen = []
    for kind in TYPES:
        pool = sorted((r for r in records.values() if r["structure_type"] == kind and len(r["sequence"]) <= max_length),
                      key=lambda r: (len(r["sequence"]), r["protein_id"]))
        used = 0
        for r in pool:
            if used + len(r["positions"]) > budget:
                continue
            chosen.append(r["protein_id"])
            used += len(r["positions"])
            if used >= budget:
                break
    return chosen


class Recorded:
    def __init__(self, predictions: Path, plan: dict):
        self.dir, self.where = predictions, {p: i for i, s in enumerate(plan["shards"]) for p in s}
        self.shards: dict[int, dict] = {}

    def deltas(self, pid: str) -> tuple[list[int], np.ndarray]:
        """Recorded positions and their WT-relative AA20 log-marginals [n, 20] (the WT column is 0)."""
        i = self.where[pid]
        if i not in self.shards:
            z = np.load(self.dir / f"shard_{i:02d}.npz", allow_pickle=False)
            self.shards[i] = {k: z[k] for k in z.files}
        z = self.shards[i]
        k = z["protein_ids"].tolist().index(pid)
        lo, hi = int(z["offsets"][k]), int(z["offsets"][k + 1])
        return z["positions"][lo:hi].tolist(), z["scores"][lo:hi]


def score_positions(model, structures: StructureSet, requests: list[tuple[str, str, list[int]]], batch: int = 8) -> dict:
    """{protein id: [n_positions, 20] WT-relative log-marginals} for (protein id, sequence, positions)."""
    import torch
    sequences = [seq for _, seq, _ in requests]
    seq_id, pos = [], []
    for k, (_, _, positions) in enumerate(requests):
        seq_id += [k] * len(positions)
        pos += list(positions)
    wt = np.asarray([AA20.index(sequences[s][p]) for s, p in zip(seq_id, pos)], dtype=np.uint8)
    table = ContextTable(sequences, np.asarray(seq_id, dtype=np.int32), np.asarray(pos, dtype=np.int32), wt)
    model.encode(table, structures)
    lengths = np.asarray([min(len(sequences[s]), 1022) for s in table.seq_id])
    order = np.argsort(-lengths, kind="stable")
    out = np.zeros((len(table), 20), dtype=np.float32)
    model.module.eval()
    with torch.no_grad():
        for start in range(0, len(order), batch):
            rows = order[start:start + batch]
            out[rows] = model.logits(table, rows).cpu().numpy()
    out -= out[np.arange(len(out)), wt][:, None]
    result, at = {}, 0
    for pid, _, positions in requests:
        result[pid] = out[at:at + len(positions)]
        at += len(positions)
    return result


def compare_values(mine: dict, recorded: Recorded, records: dict) -> dict:
    biggest, total, count, per_position = 0.0, 0.0, 0, []
    for pid, got in mine.items():
        positions, want = recorded.deltas(pid)
        assert positions == records[pid]["positions"], f"{pid}: recorded positions differ from the plan"
        diff = np.abs(got - want)
        biggest, total, count = max(biggest, float(diff.max())), total + float(diff.sum()), count + diff.size
        for i, p in enumerate(positions):
            keep = [j for j in range(20) if j != AA20.index(records[pid]["sequence"][p])]
            per_position.append(rho(got[i][keep], want[i][keep]))
    return {"values": count, "max_abs_delta": biggest, "mean_abs_delta": total / count,
            "positions": len(per_position), "mean_per_position_rho": float(np.mean(per_position)),
            "min_per_position_rho": float(np.min(per_position))}


def assay_part(bm, records, mine: dict, recorded: Recorded, ids: list[str]) -> dict:
    """Macro Spearman against the labels, recorded versus recomputed, over the assays of the chosen proteins."""
    index = {pid: {p: k for k, p in enumerate(recorded.deltas(pid)[0])} for pid in ids}
    rows = []
    for a in bm.assays():
        if a.protein_id not in mine:
            continue
        v, y = bm.variants(a.assay_id), bm.labels(a.assay_id)
        want = recorded.deltas(a.protein_id)[1]
        pick = [(index[a.protein_id][int(p)], int(m)) for p, m in zip(v.pos, v.mt)]
        s_rec = np.asarray([want[i, m] for i, m in pick])
        s_new = np.asarray([mine[a.protein_id][i, m] for i, m in pick])
        if len(y) < 3 or np.ptp(s_rec) == 0 or np.ptp(s_new) == 0:
            continue
        rows.append((a.assay_id, records[a.protein_id]["structure_type"], len(y), rho(s_rec, y), rho(s_new, y)))
    return {"rows": rows}


def summarize_assays(rows: list[tuple]) -> dict:
    def block(sel):
        if not sel:
            return None
        rec, new = np.asarray([r[3] for r in sel]), np.asarray([r[4] for r in sel])
        return {"assays": len(sel), "recorded": float(rec.mean()), "recomputed": float(new.mean()),
                "difference": float((new - rec).mean()), "max_abs_per_assay": float(np.abs(new - rec).max())}
    out = {"all": block(rows), "n_ge_30": block([r for r in rows if r[2] >= 30])}
    for kind in TYPES:
        out[kind] = block([r for r in rows if r[1] == kind])
    return out


def score_part(args, bm, records, sets) -> dict:
    import torch
    os.environ["MPD_MODEL_SAPROT_650M"] = str(args.model)
    os.environ["MPD_DTYPE"] = args.dtype
    torch.set_num_threads(args.threads)
    from mpd.students.saprot import Saprot
    recorded = Recorded(args.predictions, json.loads(args.plan.read_text()))
    ids = choose_subset(records, args.budget, args.max_length)
    n_positions = sum(len(records[p]["positions"]) for p in ids)
    say(f"[scores] subset: {len(ids)} proteins, {n_positions} positions, id digest {hashlib.sha256(chr(10).join(sorted(ids)).encode()).hexdigest()[:16]}")
    t0 = time.time()
    model = Saprot("saprot-650m", torch.device("cpu"))
    say(f"[scores] loaded SaProt-650M ({args.dtype}) in {time.time() - t0:.0f}s")
    requests = [(p, records[p]["sequence"], records[p]["positions"]) for p in ids]
    t0 = time.time()
    mine = score_positions(model, sets["legacy"], requests)             # the inputs of the recorded run
    say(f"[scores] scored {n_positions} positions in {time.time() - t0:.0f}s")
    values = compare_values(mine, recorded, records)
    assays = summarize_assays(assay_part(bm, records, mine, recorded, ids)["rows"])
    report: dict = {"subset": {"proteins": len(ids), "positions": n_positions, "dtype": args.dtype, "threads": args.threads},
                    "values": values, "assays": assays}
    say(f"[scores] values: max|d| {values['max_abs_delta']:.4f}  mean|d| {values['mean_abs_delta']:.5f}  per-position rho "
        f"mean {values['mean_per_position_rho']:.5f} min {values['min_per_position_rho']:.4f}")
    for name, b in assays.items():
        if b:
            say(f"[scores] assays {name:<12} n={b['assays']:>3}  recorded {b['recorded']:.6f}  recomputed {b['recomputed']:.6f}  "
                f"diff {b['difference']:+.6f}  max per-assay |diff| {b['max_abs_per_assay']:.4f}")
    # the same ESMFold proteins with the corrected masking: how much the scale artefact of the recorded run matters
    esm = [p for p in ids if records[p]["structure_type"] == "de_novo"]
    if esm and not args.no_corrected_esmfold:
        t0 = time.time()
        fixed = score_positions(model, sets["corrected"], [(p, records[p]["sequence"], records[p]["positions"]) for p in esm])
        legacy_rows = assay_part(bm, records, {p: mine[p] for p in esm}, recorded, esm)["rows"]
        fixed_rows = assay_part(bm, records, fixed, recorded, esm)["rows"]
        new = {r[0]: r[4] for r in fixed_rows}
        both = [(r[0], r[2], r[3], r[4], new[r[0]]) for r in legacy_rows if r[0] in new]
        report["esmfold_correction"] = {
            "proteins": len(esm), "assays": len(both), "recorded_all_masked": float(np.mean([b[2] for b in both])),
            "recomputed_all_masked": float(np.mean([b[3] for b in both])), "recomputed_corrected": float(np.mean([b[4] for b in both])),
            "mean_change": float(np.mean([b[4] - b[3] for b in both])), "max_abs_change": float(np.max([abs(b[4] - b[3]) for b in both]))}
        say(f"[scores] ESMFold subset ({len(both)} assays, {time.time() - t0:.0f}s): all-masked inputs {report['esmfold_correction']['recomputed_all_masked']:.4f}"
            f" -> corrected masking {report['esmfold_correction']['recomputed_corrected']:.4f}")
    if not args.no_long and args.long_protein in records:
        r = records[args.long_protein]
        ps = r["positions"]
        pick = sorted({ps[0], ps[len(ps) // 2], ps[-1]})
        t0 = time.time()
        got = score_positions(model, sets["legacy"], [(args.long_protein, r["sequence"], pick)])[args.long_protein]
        positions, want = recorded.deltas(args.long_protein)
        row = {int(p): want[positions.index(p)] for p in pick}
        report["long_protein"] = {"id": args.long_protein, "length": len(r["sequence"]), "positions": pick,
                                  "max_abs_delta": [float(np.abs(got[i] - row[p]).max()) for i, p in enumerate(pick)]}
        say(f"[scores] {args.long_protein} (L={len(r['sequence'])}, window 1024) positions {pick}: max|d| "
            f"{[round(x, 4) for x in report['long_protein']['max_abs_delta']]} ({time.time() - t0:.0f}s)")
    big = assays["n_ge_30"]
    report["gate"] = {"per_position_rho": values["mean_per_position_rho"] >= GATES["per_position_rho"],
                      "max_abs_delta": values["max_abs_delta"] <= GATES["max_abs_delta"],
                      "macro_n_ge_30": bool(big) and abs(big["difference"]) <= GATES["macro_diff"]}
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--home", type=Path, required=True)
    ap.add_argument("--plan", type=Path, required=True, help="the earlier study's PLAN.json (frozen strings, positions, shards)")
    ap.add_argument("--foldseek", type=Path, help="Foldseek binary (needed for the structures part and for scores)")
    ap.add_argument("--predictions", type=Path, help="folder with the recorded shard_XX.npz of the SaProt-650M base model")
    ap.add_argument("--model", type=Path, help="SaProt-650M folder (config.json, weights, vocab.txt)")
    ap.add_argument("--part", choices=["structures", "scores", "all"], default="all")
    ap.add_argument("--work", type=Path, help="scratch folder for the 3Di cache (default: a temporary one)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--budget", type=int, default=300, help="recorded positions per structure type in the score subset")
    ap.add_argument("--max-length", type=int, default=400)
    ap.add_argument("--long-protein", default="venus_seq_b7b753f9f5ab", help="a protein longer than 1,024 residues for the window check")
    ap.add_argument("--no-long", action="store_true")
    ap.add_argument("--no-corrected-esmfold", action="store_true")
    ap.add_argument("--out", type=Path, help="write the report as JSON")
    args = ap.parse_args(argv)
    if not args.foldseek:
        ap.error("--foldseek is required")
    if args.part in ("scores", "all") and not (args.predictions and args.model):
        ap.error("--predictions and --model are required for the scores part")
    bm = open_benchmark("venusmuthub", args.home)
    records = {r["protein_id"]: r for r in json.loads(args.plan.read_text())["records"]}
    engine = foldseek.FoldseekEngine(args.foldseek)
    say(f"Foldseek {engine.version}, sha256 {engine.binary_sha256[:16]}"
        f"{' (the build of the earlier study)' if engine.binary_sha256 == FOLDSEEK_REFERENCE_BUILD['binary_sha256'] else ' (a different build)'}"
        f"{' (the pinned release)' if engine.binary_sha256 == FOLDSEEK.binary_sha256 else ''}")
    report: dict = {"python": sys.version.split()[0], "created": time.strftime("%Y-%m-%d")}
    with tempfile.TemporaryDirectory(prefix="mpd-regress-") as tmp:
        work = args.work or Path(tmp)
        t0 = time.time()
        report["structures"], sets = structure_part(bm, records, engine, work)
        say(f"[structures] done in {time.time() - t0:.0f}s")
        if args.part in ("scores", "all"):
            import torch
            import transformers
            report["versions"] = {"torch": torch.__version__, "transformers": transformers.__version__}
            report["scores"] = score_part(args, bm, records, sets)
    gates = {**{f"structures.{k}": v for k, v in report["structures"]["gate"].items()},
             **{f"scores.{k}": v for k, v in report.get("scores", {}).get("gate", {}).items()}}
    say("gates: " + ", ".join(f"{k}={'PASS' if v else 'FAIL'}" for k, v in gates.items()))
    if args.out:
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=float) + "\n")
    required = [v for k, v in gates.items() if k != "structures.foldseek_is_the_build_of_the_earlier_study"]
    if not report["structures"]["gate"]["foldseek_is_the_build_of_the_earlier_study"]:
        required = [v for k, v in gates.items() if k.startswith("scores.")]     # another build: only the report is meaningful
    return 0 if all(required) else 1


if __name__ == "__main__":
    sys.exit(main())
