"""The five stages end to end on CPU: tiny random ESM2 models, a miniature ProteinGym-format dataset."""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from fx_stages import curve, env, finish_ddp_train, plan_for, run_stages, start_ddp_train  # noqa: F401

STAGES_DIR = Path(__file__).resolve().parents[1] / "src" / "mpd" / "stages"


def test_direct_logit_pipeline_all_assays(env):
    plan = plan_for(env)
    assert [s.name for s in plan.stages] == ["prepare", "teacher", "train", "score-base", "score", "evaluate"]
    run_stages(plan)
    root = plan.campaign_dir

    manifest = json.loads((root / "contexts" / "MANIFEST.json").read_text())
    # dense: every residue of each assay region = 40 (P1, whole) + 29 (P2, region 2-30)
    assert manifest["dms_labels_opened"] is False and manifest["support"] == "aa20_dense" and manifest["union_contexts"] == 69
    assert np.load(root / "teacher" / "logits.npy").shape == (69, 20)

    train = root / "train" / "all"
    steps = [r["step"] for r in curve(train)]
    assert steps == [0, 2, 4, 6]
    chosen = json.loads((train / "SELECTED.json").read_text())
    kept = sorted(p.name for p in train.glob("checkpoint_*"))
    assert kept == [f"checkpoint_{chosen['step']:05d}"] and (train / kept[0] / "adapter_model.safetensors").is_file()

    summary = json.loads((root / "evaluate" / "summary.json").read_text())
    assert summary["primary_readout"] == "delta" and summary["dms_labels_opened"] is True
    primary = {(e["unit"], e["dataset"]): e for e in summary["results"] if e["primary"]}
    assert set(primary) == {("base", "proteingym"), ("all", "proteingym")}
    assert all(np.isfinite(e["macro_spearman"]) and e["assays"] == 3 for e in primary.values())
    assert "delta_vs_base" in primary[("all", "proteingym")]
    rows = list(csv.DictReader(open(root / "evaluate" / "per_assay.csv")))
    assert len(rows) == 3 * 2 * 2                                   # assays x (base, all) x readouts

    # the adapter was actually applied: trained and base scores differ
    base = np.load(root / "score" / "proteingym" / "base" / "shard_000_of_002.npz")["scores"]
    tuned = np.load(root / "score" / "proteingym" / "all" / "shard_000_of_002.npz")["scores"]
    assert base.shape == tuned.shape and not np.array_equal(base, tuned)


def test_score_pipeline_one_assay_scores_its_own_assay(env):
    plan = plan_for(env, method="score-one", assays=("ASSAY_A",))
    run_stages(plan)
    root = plan.campaign_dir
    assert "dev_kl" not in curve(root / "train" / "ASSAY_A")[0]      # score objective has no KL term
    manifest = json.loads((root / "contexts" / "MANIFEST.json").read_text())
    assert manifest["support"] == "observed" and manifest["unit_variants"]["ASSAY_A"] == 12 * 19   # assayed variants only
    summary = json.loads((root / "evaluate" / "summary.json").read_text())
    assert summary["primary_readout"] == "dual"
    rows = list(csv.DictReader(open(root / "evaluate" / "per_assay.csv")))
    unit_rows = [r for r in rows if r["unit"] == "ASSAY_A"]
    assert {r["assay_id"] for r in unit_rows} == {"ASSAY_A"}        # own assay only
    assert {r["assay_id"] for r in rows if r["unit"] == "base"} == {"ASSAY_A", "ASSAY_B", "ASSAY_C"}
    entry = next(e for e in summary["results"] if e["unit"] == "ASSAY_A" and e["primary"])
    assert entry["assays"] == 1 and np.isfinite(entry["delta_vs_base"])


@pytest.mark.parametrize("method", ["direct-all", "score-all"])
def test_a_teacher_identical_to_the_student_starts_at_zero_loss(env, monkeypatch, tiny_models, method):
    """Fails if any context index, amino-acid order, window or union mapping is wrong between teacher and student."""
    monkeypatch.setenv("MPD_MODEL_ESM2_35M", str(tiny_models["student"]))
    plan = plan_for(env, method=method)
    run_stages(plan, only={"prepare", "teacher", "train"})
    start = curve(plan.campaign_dir / "train" / "all")[0]
    # Same weights, so only float32 noise separates teacher and student (the two stages batch and pad differently). The
    # noise depends on the torch release: exactly zero on the pinned 2.10, a few 1e-8 on newer ones. A wrong index, window
    # or amino-acid order gives a loss around 1e-1 and a rank correlation near 0, far outside these bounds.
    assert start["step"] == 0 and start["dev_loss"] < 1e-6
    assert start["dev_rho_ST"] > 0.9999


def test_stages_are_idempotent(env, capsys):
    plan = plan_for(env)
    run_stages(plan, only={"prepare", "teacher", "train", "score-base"})
    before = (plan.campaign_dir / "train" / "all" / "SELECTED.json").read_text()
    run_stages(plan, only={"train", "score-base"})
    out = capsys.readouterr().out
    assert "already complete" in out and "already done" in out
    assert (plan.campaign_dir / "train" / "all" / "SELECTED.json").read_text() == before
    assert not list((plan.campaign_dir / "train").glob("*incomplete*"))


def test_evaluate_refuses_incomplete_scoring(env):
    plan = plan_for(env)
    run_stages(plan, only={"prepare", "teacher", "train", "score-base"})
    (plan.campaign_dir / "score" / "proteingym" / "base" / "shard_001_of_002.json").unlink()
    with pytest.raises(RuntimeError, match="shards incomplete"):
        run_stages(plan, only={"score", "evaluate"} - {"score"})


def test_only_the_evaluate_stage_can_read_labels():
    for path in sorted(STAGES_DIR.glob("*.py")):
        text = path.read_text()
        if path.name == "evaluate.py":
            assert ".labels(" in text
        else:
            assert ".labels(" not in text and "DMS_score" not in text, f"{path.name} touches labels"


def test_two_process_ddp_training_matches_single_process(env, tmp_path):
    """Same global batch of 8: 1 process x 4 micro-batches must equal 2 processes x 2 micro-batches (gloo, CPU).

    Two micro-batches per process take the no_sync branch of the accumulation loop, which real multi-GPU runs rely on.
    """
    # a large learning rate so the dev loss moves by far more than the tolerance below
    single = plan_for(env, train_gpus=1, peak_lr=3e-3, global_batch=8)
    ddp = plan_for(env, train_gpus=2, peak_lr=3e-3, global_batch=8)
    assert single.resolved["world_size"] == 1 and single.resolved["accumulation"] == 4
    assert ddp.resolved["world_size"] == 2 and ddp.resolved["accumulation"] == 2
    run_stages(single, only={"prepare", "teacher", "train"})
    run_stages(ddp, only={"prepare", "teacher"})
    code, log = finish_ddp_train(start_ddp_train(ddp, tmp_path, "ddp"))
    assert code == 0, log[-3000:]
    a, b = curve(single.campaign_dir / "train" / "all"), curve(ddp.campaign_dir / "train" / "all")
    assert [r["step"] for r in a] == [r["step"] for r in b]
    assert abs(a[-1]["dev_loss"] - a[0]["dev_loss"]) > 100 * 2e-5, "training barely moved the loss: the comparison would be vacuous"
    for ra, rb in zip(a, b):
        assert ra["dev_loss"] == pytest.approx(rb["dev_loss"], abs=2e-5), (ra, rb)
    assert (ddp.campaign_dir / "train" / "all" / "COMPLETE.json").is_file()
