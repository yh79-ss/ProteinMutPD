import json
import math

import numpy as np
import pytest

from mpd.constants import AA_INDEX
from mpd.contexts import ContextTable, build_table, dev_mask, locate, union
from mpd.data.benchmarks import ProteinGym, build_variants

from tinydata import AA, write_tiny_proteingym


@pytest.fixture
def bm(tmp_path):
    write_tiny_proteingym(tmp_path)
    return ProteinGym(tmp_path)


# -- benchmark access and the label firewall -------------------------------------

def test_variants_are_single_substitutions_aligned_with_labels(bm):
    v = bm.variants("ASSAY_A")
    assert len(v) == 12 * 19                                      # the multi-mutant row is skipped
    assert len(bm.labels("ASSAY_A")) == len(v)
    seq = bm.assay("ASSAY_A").sequence
    for m, p, w, t in zip(v.mutants[:50], v.pos, v.wt, v.mt):
        assert m == f"{seq[p]}{p + 1}{AA[t]}" and AA[w] == seq[p]


def test_variants_never_read_the_score_column(tmp_path):
    import csv
    write_tiny_proteingym(tmp_path)
    path = tmp_path / "DMS_ProteinGym_substitutions" / "ASSAY_A.csv"
    rows = list(csv.DictReader(open(path)))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({**r, "DMS_score": "NOT_A_NUMBER"} for r in rows)
    bad = ProteinGym(tmp_path)
    assert len(bad.variants("ASSAY_A")) > 0                       # label-free access is unaffected
    with pytest.raises(ValueError):
        bad.labels("ASSAY_A")


def test_wild_type_mismatch_is_an_error_not_a_skip():
    with pytest.raises(ValueError, match="does not match"):
        build_variants(["A3C"], "MKLV")
    variants, keep = build_variants(["M1C", "K2A:L3A", "L3del"], "MKLV")
    assert keep == [0] and variants.mutants == ("M1C",)


# -- context tables -----------------------------------------------------------------

def test_tables_are_unique_sorted_and_deterministic(bm):
    ids = ["ASSAY_A", "ASSAY_B", "ASSAY_C"]
    a, b = build_table(bm, ids), build_table(bm, ids)
    assert a.digest() == b.digest()
    keys = a.keys()
    assert keys == sorted(set(keys))
    # A (positions 2..13) and C (8..21) overlap on the same protein: 20 contexts there, plus 12 on the other protein
    assert len(a) == 20 + 12
    for i in range(len(a)):
        mts = a.variants_of(i)
        assert list(mts) == sorted(set(mts)) and AA_INDEX[a.sequences[a.seq_id[i]][a.pos[i]]] not in mts


def test_union_locate_subset_and_roundtrip(bm, tmp_path):
    ta, tb = build_table(bm, ["ASSAY_A"]), build_table(bm, ["ASSAY_B"])
    both = union([ta, tb])
    assert len(both) == len(ta) + len(tb) and both.var_ptr is None
    idx = locate(ta, both)
    assert both.keys()[idx[0]] == ta.keys()[0]
    ta.union_index = idx
    ta.save(tmp_path / "t")
    back = ContextTable.load(tmp_path / "t")
    assert back.digest() == ta.digest() and (back.union_index == idx).all()
    sub = ta.subset(np.array([3, 0]))
    assert sub.keys() == [ta.keys()[3], ta.keys()[0]] and list(sub.variants_of(0)) == list(ta.variants_of(3))


def test_dev_split_is_by_position_and_deterministic(bm):
    t = build_table(bm, ["ASSAY_A", "ASSAY_B", "ASSAY_C"])
    m1, m2 = t.dev_mask(0.3), t.dev_mask(0.3)
    assert (m1 == m2).all()
    for sid in set(t.seq_id):
        rows = t.seq_id == sid
        assert m1[rows].any() and not m1[rows].all()              # every protein keeps train and dev contexts
    big = dev_mask(["h"] * 4000, list(range(4000)), 0.05)
    assert 0.03 < big.mean() < 0.07


# -- recipe pieces -------------------------------------------------------------------

def test_learning_rate_schedule_matches_the_recipe():
    from mpd.stages.train import learning_rate
    total, peak = 2103, 3e-5
    assert learning_rate(1, total, peak) < peak / 50
    assert learning_rate(105, total, peak) == pytest.approx(peak, rel=0.02)     # end of the 5% warm-up
    assert learning_rate(total, total, peak) == pytest.approx(peak * 0.1, rel=1e-3)
    assert all(learning_rate(s, total, peak) >= peak * 0.1 - 1e-12 for s in range(105, total + 1))   # after warm-up
    assert all(learning_rate(s, total, peak) <= peak * (1 + 1e-9) for s in range(1, total + 1))


def test_monitor_steps_reproduce_the_paper_checkpoints():
    from mpd.stages.train import monitor_steps
    assert monitor_steps(2103) == [701, 1402, 2103]
    assert monitor_steps(2) == [1, 2]


def test_update_stream_is_deterministic_and_covers_training_rows():
    from mpd.stages.train import update_stream
    rows = np.arange(100, 130)
    a, b = update_stream(rows, 7, 8, seed=3), update_stream(rows, 7, 8, seed=3)
    assert all((x == y).all() for x, y in zip(a, b)) and len(a) == 7 and all(len(x) == 8 for x in a)
    flat = np.concatenate(a)
    assert set(flat) == set(rows) and flat.min() >= 100
    assert not (update_stream(rows, 7, 8, seed=4)[0] == a[0]).all()


def test_checkpoint_selection_prefers_improvement_and_early_steps():
    from mpd.stages.train import select_checkpoint
    curve = [{"step": 0, "dev_loss": 1.0}, {"step": 10, "dev_loss": 0.8}, {"step": 20, "dev_loss": 0.8}, {"step": 30, "dev_loss": 0.9}]
    best = select_checkpoint(curve)
    assert best["step"] == 10 and best["improved_over_start"]
    worse = select_checkpoint([{"step": 0, "dev_loss": 0.5}, {"step": 10, "dev_loss": 0.7}, {"step": 20, "dev_loss": 0.6}])
    assert worse["step"] == 20 and not worse["improved_over_start"]
    with pytest.raises(RuntimeError):
        select_checkpoint([{"step": 0, "dev_loss": 1.0}, {"step": 5, "dev_loss": math.nan}])


def test_long_sequences_use_a_position_centered_window():
    pytest.importorskip("torch")
    from mpd.students.esm2 import optimal_window
    short = list(range(12))
    assert optimal_window(short, 4) == (short, 5)
    long = list(range(3000))                                        # 2998 residues + <cls>/<eos>
    window, focal = optimal_window(long, 1500)
    assert len(window) == 1024 and window[focal] == 1501
    assert optimal_window(long, 3)[1] == 4                          # near the start: window begins at 0
    w_end, f_end = optimal_window(long, 2995)
    assert w_end[f_end] == 2996 and w_end[-1] == 2999
    with pytest.raises(ValueError):
        optimal_window(long, 2998)


# -- objectives ---------------------------------------------------------------------

def test_score_readout_matches_its_definition():
    torch = pytest.importorskip("torch")
    from mpd.objectives import readout_dual, readout_delta, score_from_logits
    rng = np.random.default_rng(0)
    z = rng.normal(size=(5, 20)).astype(np.float32)
    wt, mt = np.array([0, 3, 5, 7, 9]), np.array([1, 2, 6, 19, 0])
    expect = [math.log(np.exp(z[i, m]) / np.exp(z[i]).sum()) - math.log(np.exp(z[i, w] / 1.5) / np.exp(z[i] / 1.5).sum())
              for i, (w, m) in enumerate(zip(wt, mt))]
    assert np.allclose(readout_dual(z, wt, mt), expect, atol=1e-5)
    assert np.allclose(readout_delta(z, wt, mt), z[np.arange(5), mt] - z[np.arange(5), wt])
    t = torch.as_tensor(z)
    assert np.allclose(score_from_logits(t, torch.as_tensor(wt), torch.arange(5), torch.as_tensor(mt)).numpy(), expect, atol=1e-5)


def test_direct_loss_is_zero_for_identical_logits_and_positive_otherwise():
    torch = pytest.importorskip("torch")
    from mpd.objectives import direct_loss, score_loss
    t = torch.randn(4, 20)
    loss, parts = direct_loss(t.clone(), t, 1.0)
    assert float(loss) == pytest.approx(0, abs=1e-6)
    shifted, _ = direct_loss(t + 5.0, t, 0.0)
    assert float(shifted) == pytest.approx(4.5)                    # Huber(delta=1) of a constant offset of 5
    only_kl, _ = direct_loss(t + 5.0, t, 1.0)
    assert float(only_kl) == pytest.approx(4.5, abs=1e-4)          # a constant shift changes no softmax: KL stays 0
    wt, row, mt = torch.zeros(4, dtype=torch.long), torch.tensor([0, 0, 2]), torch.tensor([1, 2, 3])
    loss, _, gs, gt = score_loss(t.clone().requires_grad_(), t, wt, row, mt)
    assert float(loss) == pytest.approx(0, abs=1e-6) and gs.shape == gt.shape == (3,)


def test_flatten_variants_lists_the_observed_mutants_of_a_batch(bm):
    from mpd.objectives import flatten_variants
    t = build_table(bm, ["ASSAY_B"])
    row, mt = flatten_variants(t.var_ptr, t.var_mt, np.array([2, 5]))
    assert list(row) == [0] * len(t.variants_of(2)) + [1] * len(t.variants_of(5))
    assert list(mt) == list(t.variants_of(2)) + list(t.variants_of(5))


def test_dense_tables_cover_every_residue_of_the_region_and_carry_no_variants(bm):
    dense = build_table(bm, ["ASSAY_B"], with_variants=False, dense=True)
    assert len(dense) == 29 and dense.var_ptr is None            # region 2-30 (1-based), not the 34-residue sequence
    assert sorted(set(dense.pos)) == list(range(1, 30))
    whole = build_table(bm, ["ASSAY_A", "ASSAY_C"], with_variants=False, dense=True)
    assert len(whole) == 40                                       # two assays on one protein: its residues once
    with pytest.raises(ValueError, match="dense table has no variant"):
        build_table(bm, ["ASSAY_A"], dense=True)


def test_assayed_positions_outside_the_region_are_an_error(tmp_path):
    import csv
    write_tiny_proteingym(tmp_path)
    rows = list(csv.DictReader(open(tmp_path / "DMS_substitutions.csv")))
    for r in rows:
        if r["DMS_id"] == "ASSAY_A":
            r["MSA_start"], r["MSA_end"] = "6", "20"             # assay A measured positions 3..14
    with open(tmp_path / "DMS_substitutions.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(ValueError, match="outside the assay region"):
        build_table(ProteinGym(tmp_path), ["ASSAY_A"])
