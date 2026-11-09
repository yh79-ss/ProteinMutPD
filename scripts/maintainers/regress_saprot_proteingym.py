#!/usr/bin/env python3
"""Score ProteinGym assays with SaProt on the CPU and compare each assay's Spearman with the value ProteinGym itself
publishes for that model (maintainers only).

The inputs are the official v1.3 AF2 structures through the same code a campaign uses: `pdb_range`, 3Di from Foldseek,
pLDDT < 70 masked, the assay's own residues as the amino-acid channel, the masked token `#` + its own state, and the
log-marginal readout (`delta`). The published values come from ProteinGym's public benchmark table
(`benchmarks/DMS_zero_shot/substitutions/Spearman/DMS_substitutions_Spearman_DMS_level.csv`, a file of that repository).

    PYTHONPATH=src python scripts/maintainers/regress_saprot_proteingym.py --home <home with data/proteingym-v1.3> \
        --foldseek <binary> --model <SaProt folder> --column "SaProt (35M)" --table <the CSV above> [--assays ID ...]

By default it scores the 14 assays below. Six of them reproduce the published value to within 2e-3 for both SaProt sizes
(they come first); on the other eight (short Tsuboyama domains and one fluorescent protein) the published value is not
reproduced from the v1.3 package files, and the cause is not established. This tool reports both, as found.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from mpd.contexts import build_table
from mpd.data.benchmarks import open_benchmark
from mpd.structure import build, foldseek

REPRODUCIBLE = ["RS15_GEOSE_Tsuboyama_2023_1A32", "CCDB_ECOLI_Adkar_2012", "KCNE1_HUMAN_Muhammad_2023_function",
                "NUD15_HUMAN_Suiter_2020", "P84126_THETH_Chan_2017", "MET_HUMAN_Estevam_2023"]
NOT_REPRODUCED = ["TCRG1_MOUSE_Tsuboyama_2023_1E0L", "SAV1_MOUSE_Tsuboyama_2023_2YSB", "PITX2_HUMAN_Tsuboyama_2023_2L7M",
                  "NUSG_MYCTU_Tsuboyama_2023_2MI6", "RL20_AQUAE_Tsuboyama_2023_1GYZ", "SRBS1_HUMAN_Tsuboyama_2023_2O2W",
                  "CATR_CHLRE_Tsuboyama_2023_2AMI", "Q6WV12_9MAXI_Somermeyer_2022"]


def assay_spearman(model, bm, assay_id: str, engine, cache_home: Path | None) -> dict:
    """Spearman of the log-marginal `delta` score with the assay's labels, and facts about the structure used."""
    import torch
    table = build_table(bm, [assay_id], with_variants=False)
    structures = build.build_set({"proteingym": build.assays_for(bm, table.hashes)}, cache_home, engine, log=lambda m: None)
    model.encode(table, structures)
    logits = np.zeros((len(table), 20), dtype=np.float32)
    model.module.eval()
    with torch.no_grad():
        for start in range(0, len(table), 8):
            rows = np.arange(len(table))[start:start + 8]
            logits[rows] = model.logits(table, rows).numpy()
    row_of = {int(p): i for i, p in enumerate(table.pos)}
    v, y = bm.variants(assay_id), bm.labels(assay_id)
    score = np.asarray([logits[row_of[int(p)], m] - logits[row_of[int(p)], w] for p, w, m in zip(v.pos, v.wt, v.mt)])
    info = next(iter(structures.meta["per_sequence"].values()))
    return {"rho": float(spearmanr(score, y).statistic), "variants": len(y), "length": info["length"],
            "masked": info["masked"], "substitutions": info["substitutions"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--home", type=Path, required=True)
    ap.add_argument("--foldseek", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True, help="folder with the SaProt files")
    ap.add_argument("--key", default="saprot-35m", choices=["saprot-35m", "saprot-650m"])
    ap.add_argument("--column", default="SaProt (35M)", help="the published column to compare with")
    ap.add_argument("--table", type=Path, required=True, help="ProteinGym's DMS-level Spearman table (CSV)")
    ap.add_argument("--assays", nargs="+", default=REPRODUCIBLE + NOT_REPRODUCED)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tolerance", type=float, default=2e-3)
    args = ap.parse_args(argv)
    import torch
    os.environ["MPD_MODEL_" + args.key.upper().replace("-", "_")] = str(args.model)
    os.environ["MPD_DTYPE"] = args.dtype
    torch.set_num_threads(args.threads)
    from mpd.students.saprot import Saprot
    published = {r["DMS ID"]: float(r[args.column]) for r in csv.DictReader(open(args.table, newline="", encoding="utf-8"))
                 if r.get(args.column) not in (None, "", "NaN")}
    bm = open_benchmark("proteingym", args.home)
    engine = foldseek.FoldseekEngine(args.foldseek)
    print(f"Foldseek {engine.version}  model {args.key} ({args.dtype}, {args.threads} threads)", flush=True)
    model = Saprot(args.key, torch.device("cpu"))
    rows = []
    for aid in args.assays:
        t0 = time.time()
        r = assay_spearman(model, bm, aid, engine, None)
        diff = r["rho"] - published[aid]
        rows.append((aid, r["rho"], published[aid], diff))
        flag = "within tolerance" if abs(diff) <= args.tolerance else "differs"
        print(f"{aid:<40} L={r['length']:>4} n={r['variants']:>5} masked={r['masked']:>3} subs={','.join(r['substitutions']) or '-':<5} "
              f"here {r['rho']:.4f}  published {published[aid]:.4f}  diff {diff:+.4f}  {flag}  ({time.time() - t0:.0f}s)", flush=True)
    within = [abs(d) <= args.tolerance for _, _, _, d in rows]
    print(f"{sum(within)} of {len(rows)} assays within {args.tolerance} of the published value; "
          f"max |diff| over the reproducible six: "
          f"{max([abs(d) for a, _, _, d in rows if a in REPRODUCIBLE] or [float('nan')]):.4f}", flush=True)
    wanted = [w for (a, _, _, _), w in zip(rows, within) if a in REPRODUCIBLE]
    return 0 if all(wanted) else 1


if __name__ == "__main__":
    sys.exit(main())
