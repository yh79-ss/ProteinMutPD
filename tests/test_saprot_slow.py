"""Opt-in checks with a real Foldseek, real structures and real SaProt weights. Skipped unless the environment names them.

    MPD_FOLDSEEK=<foldseek binary>
    MPD_TEST_VENUS_HOME=<home holding data/venusmuthub-v1 and data/venusmuthub-structures-v1>
    MPD_TEST_PG_HOME=<home holding data/proteingym-v1.3>
    MPD_MODEL_SAPROT_35M=<folder with the SaProt-35M files>        (only for the ProteinGym table check)
    python -m pytest -q tests/test_saprot_slow.py

The expected strings and digests were produced with the Foldseek build of 2026-09-09 (see docs/SAPROT.md). Whether
Foldseek release 10-941cd33, the pinned download, produces the same 3Di has not been checked: run this file with
`MPD_FOLDSEEK` pointing at it to find out.
"""

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from mpd.contexts import build_table, sequence_hash
from mpd.data.benchmarks import open_benchmark
from mpd.structure import build, foldseek

pytestmark = pytest.mark.slow

FOLDSEEK = os.environ.get("MPD_FOLDSEEK")
VENUS_HOME = os.environ.get("MPD_TEST_VENUS_HOME")
PG_HOME = os.environ.get("MPD_TEST_PG_HOME")
needs_foldseek = pytest.mark.skipif(not FOLDSEEK, reason="set MPD_FOLDSEEK to a Foldseek binary")
needs_venus = pytest.mark.skipif(not (FOLDSEEK and VENUS_HOME), reason="set MPD_FOLDSEEK and MPD_TEST_VENUS_HOME")
needs_pg = pytest.mark.skipif(not (FOLDSEEK and PG_HOME), reason="set MPD_FOLDSEEK and MPD_TEST_PG_HOME")

# Foldseek's 3Di (upper case) and the amino acids it reads, for three Venus structures of three kinds
GOLDEN = {
    "venus_seq_016be5b116b4": ("experimental",                                    # 1PGA chain A
                               "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE",
                               "DKAKEAEAEPPDGDMDIDDDPDLVVVVVVVVVVCVVVPHDADWDADPVRRYIYGHD"),
    "venus_seq_00da05aa2d2c": ("alphafold_db",
                               None,
                               "DPLPPWDKDDWDDDPWKIKIKIAADPVLLVVDDPPDFDDPVVVVVSVDDDDPFWGQDDQPVVHSRITIIIGTDDPDPDD"),
    "venus_seq_b775285baac5": ("esmfold", "FNKEQQNAFYEILHLPNLNEEQRNGFIQSLKDDPSQSANLLAEA",
                               "DPPVLVVLLVVLVPPPPDDPVRSVVLVVVCVVPVVCSVVVSVVD"),
}
DIGESTS = {"venus_frozen_legacy": "2a99932edcc2fda67bce95fb036342efc28580fa6f54869e8c824507beb6a96d",
           "venus_corrected": "a885bf3c093c724109802405cb224a0fa2a60fa8a18bcdf54008e6850b6af7ff",
           "proteingym_official": "baea4585ceef32eb1b9b965b8bdc55b0698b3f34ed0359d3c3a8b4258aba6cbf"}
# ProteinGym's published Spearman for SaProt (35M), from its public benchmark table (DMS-level), on assays whose official
# structures our pipeline reproduces to within 2e-3
PG_TABLE_35M = {"RS15_GEOSE_Tsuboyama_2023_1A32": 0.6460, "CCDB_ECOLI_Adkar_2012": 0.1940, "KCNE1_HUMAN_Muhammad_2023_function": 0.2380,
                "NUD15_HUMAN_Suiter_2020": 0.5640, "P84126_THETH_Chan_2017": 0.6100, "MET_HUMAN_Estevam_2023": 0.5290}


def digest_by_id(rows: dict[str, str]) -> str:
    return hashlib.sha256("".join(f"{k}\t{v}\n" for k, v in sorted(rows.items())).encode()).hexdigest()


@needs_venus
def test_golden_3di_strings_of_three_venus_structures():
    folder = Path(VENUS_HOME) / "data" / "venusmuthub-structures-v1" / "structures"
    for pid, (source, aa, states) in GOLDEN.items():
        chains = foldseek.run(FOLDSEEK, folder / source / f"{pid}.pdb")
        assert len(chains) == 1 and chains[0].states == states, pid
        if aa:
            assert chains[0].aa == aa, pid


@needs_venus
def test_biotite_gives_the_same_3di_for_the_golden_structures():
    pytest.importorskip("biotite")
    from mpd.structure.biotite_engine import BiotiteEngine
    folder = Path(VENUS_HOME) / "data" / "venusmuthub-structures-v1" / "structures"
    for pid, (source, aa, states) in GOLDEN.items():
        assert BiotiteEngine().chains(folder / source / f"{pid}.pdb")[0].states == states, pid


@needs_venus
@pytest.mark.parametrize("legacy, name", [(True, "venus_frozen_legacy"), (False, "venus_corrected")])
def test_venus_structure_strings_match_the_earlier_studys_digest(tmp_path, legacy, name):
    bm = open_benchmark("venusmuthub", Path(VENUS_HOME))
    assays = bm.assays()
    needed = build.assays_for(bm, [sequence_hash(a.sequence) for a in assays])
    s = build.build_set({"venusmuthub": needed}, tmp_path, foldseek.FoldseekEngine(FOLDSEEK),
                        policy=build.Policy(legacy_mask=legacy), log=lambda m: None)
    by_protein = {a.protein_id: s.states[sequence_hash(a.sequence)] for a in assays}
    assert len(by_protein) == 489
    assert digest_by_id(by_protein) == DIGESTS[name]


@needs_pg
def test_proteingym_structure_strings_match_the_recorded_digest(tmp_path):
    bm = open_benchmark("proteingym", Path(PG_HOME))
    needed = build.assays_for(bm, [sequence_hash(a.sequence) for a in bm.assays()])
    s = build.build_set({"proteingym": needed}, tmp_path, foldseek.FoldseekEngine(FOLDSEEK), log=lambda m: None)
    assert len(s) == 187
    partial = {k: v["substitutions"] for k, v in s.meta["per_sequence"].items() if v["substitutions"]}
    assert sorted(sum(partial.values(), [])) == ["R72P"]                 # P53: the one construct that differs from its model
    assert s.digest() == DIGESTS["proteingym_official"]


@needs_pg
@pytest.mark.skipif(not os.environ.get("MPD_MODEL_SAPROT_35M"), reason="set MPD_MODEL_SAPROT_35M to the SaProt-35M folder")
def test_saprot_35m_reproduces_proteingyms_published_spearman_on_six_assays(tmp_path):
    import torch
    from scipy.stats import spearmanr
    from mpd.students.saprot import Saprot
    bm = open_benchmark("proteingym", Path(PG_HOME))
    torch.set_num_threads(4)
    os.environ.setdefault("MPD_DTYPE", "bfloat16")
    model = Saprot("saprot-35m", torch.device("cpu"))
    engine = foldseek.FoldseekEngine(FOLDSEEK)
    for aid, published in PG_TABLE_35M.items():
        table = build_table(bm, [aid], with_variants=False)
        structures = build.build_set({"proteingym": build.assays_for(bm, table.hashes)}, tmp_path, engine, log=lambda m: None)
        model.encode(table, structures)
        logits = np.zeros((len(table), 20), dtype=np.float32)
        model.module.eval()
        with torch.no_grad():
            for start in range(0, len(table), 8):
                rows = np.arange(len(table))[start:start + 8]
                logits[rows] = model.logits(table, rows).numpy()
        row_of = {int(p): i for i, p in enumerate(table.pos)}
        v, y = bm.variants(aid), bm.labels(aid)
        score = np.asarray([logits[row_of[int(p)], m] - logits[row_of[int(p)], w] for p, w, m in zip(v.pos, v.wt, v.mt)])
        assert abs(spearmanr(score, y).statistic - published) <= 2e-3, aid
