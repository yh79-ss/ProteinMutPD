"""The five stages end to end on CPU: tiny random ESM2 models, a miniature ProteinGym-format dataset."""

import csv
import importlib
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from mpd.slurm.cluster import Cluster
from mpd.slurm.plan import PlanOptions, make_plan
from mpd.slurm.submit import write_campaign
from mpd.spec import RunSpec

STAGES_DIR = Path(__file__).resolve().parents[1] / "src" / "mpd" / "stages"


@pytest.fixture
def env(monkeypatch, tiny_home, tiny_models):
    monkeypatch.setenv("MPD_HOME", str(tiny_home))
    monkeypatch.setenv("MPD_DTYPE", "float32")
    monkeypatch.setenv("MPD_MODEL_ESM2_8M", str(tiny_models["student"]))
    monkeypatch.setenv("MPD_MODEL_ESM2_35M", str(tiny_models["teacher"]))
    return tiny_home


def plan_for(home, **kw):
    kw = {"method": "direct-all", "updates": 6, "global_batch": 4, "dev_fraction": 0.3, **kw}
    if kw["method"].endswith("all"):
        kw.setdefault("train_gpus", 1)
    spec = RunSpec(teacher="esm2-35m", student="esm2-8m", train_data="proteingym", eval_data=("proteingym",), **kw)
    plan = make_plan(spec, PlanOptions(cluster=Cluster.load("vision"), home=home, python="python",
                                       score_shards={"proteingym": 2}))
    write_campaign(plan)
    return plan


def run_stages(plan, only=None):
    for stage in plan.stages:
        if only and stage.name not in only:
            continue
        module = importlib.import_module(stage.module)
        for i in range(len(stage.tasks)):
            module.main(["--campaign-dir", str(plan.campaign_dir), "--stage", stage.name, "--task-index", str(i + stage.task_offset)])


def curve(path):
    return [json.loads(line) for line in open(path / "dev_curve.jsonl")]


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
    assert start["step"] == 0 and start["dev_loss"] < 1e-8
    assert start["dev_rho_ST"] > 0.999999


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
