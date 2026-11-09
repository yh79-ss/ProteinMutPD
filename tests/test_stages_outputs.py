"""What the stages write: the documented files and fields (`mpd results` and docs/RESULTS.md read them), and provenance."""

import csv
import json

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from fx_stages import env, plan_for, run_stages  # noqa: F401

PROVENANCE = {"mpd", "python", "platform", "transformers", "peft", "host", "slurm_job_id", "utc", "torch", "torch_cuda",
              "float32_matmul_precision"}


def load(path):
    return json.loads(path.read_text())


@pytest.fixture
def pipeline(env, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "777")
    plan = plan_for(env)
    run_stages(plan)
    return plan.campaign_dir


def test_every_record_carries_provenance_and_no_home_path(pipeline, env):
    train = pipeline / "train" / "all"
    records = [pipeline / "teacher" / "COMPLETE.json", train / "CONFIG.json", train / "COMPLETE.json",
               pipeline / "score" / "proteingym" / "base" / "shard_000_of_002.json",
               pipeline / "score" / "proteingym" / "all" / "shard_001_of_002.json", pipeline / "evaluate" / "summary.json"]
    for path in records:
        info = load(path)
        assert PROVENANCE <= set(info["provenance"]), path
        assert info["provenance"]["slurm_job_id"] == "777"
        assert str(env) not in path.read_text(), f"{path.name} holds the home path"
        assert "gpu" not in info["provenance"]                          # no CUDA here, so no GPU fields


def test_train_records(pipeline):
    train = pipeline / "train" / "all"
    config, done = load(train / "CONFIG.json"), load(train / "COMPLETE.json")
    assert {"unit", "student", "teacher", "objective", "train_contexts", "dev_contexts", "variants", "world_size", "monitors",
            "context_digest", "teacher_logits_sha256", "dtype", "dms_labels_opened", "config_digest", "provenance",
            "updates", "global_batch", "accumulation", "peak_lr", "kl_weight", "lora_rank", "dev_fraction"} <= set(config)
    assert config["dms_labels_opened"] is False and config["dtype"] == "torch.float32" and config["monitors"] == [2, 4, 6]
    assert config["teacher_logits_sha256"] == load(pipeline / "teacher" / "COMPLETE.json")["logits_sha256"]
    assert set(done) == {"status", "updates", "selected_step", "seconds", "resumed_from", "config_digest", "peak_gpu_memory_gb",
                         "provenance"}
    assert done["status"] == "PASS" and done["updates"] == 6 and done["resumed_from"] == [] and done["peak_gpu_memory_gb"] is None
    assert done["config_digest"] == config["config_digest"]
    assert set(load(train / "SELECTED.json")) >= {"step", "dev_loss", "start_dev_loss", "improved_over_start"}
    assert not (pipeline / "train" / "all.partial").exists()


def test_score_shard_records(pipeline):
    shard = pipeline / "score" / "proteingym" / "all" / "shard_001_of_002"
    record = load(shard.with_suffix(".json"))
    assert set(record) == {"unit", "dataset", "shard", "num_shards", "contexts", "of", "adapter_step", "scores_sha256", "seconds",
                           "peak_gpu_memory_gb", "dms_labels_opened", "provenance"}
    assert (record["unit"], record["dataset"], record["shard"], record["num_shards"]) == ("all", "proteingym", 1, 2)
    assert record["adapter_step"] in (2, 4, 6) and record["dms_labels_opened"] is False
    with np.load(shard.with_suffix(".npz")) as z:
        assert set(z.files) == {"rows", "scores", "shard", "num_shards"} and (int(z["shard"]), int(z["num_shards"])) == (1, 2)
        assert z["scores"].shape == (len(z["rows"]), 20) and len(z["rows"]) == record["contexts"]


def test_the_evaluate_summary_has_the_documented_shape(pipeline):
    summary = load(pipeline / "evaluate" / "summary.json")
    assert set(summary) == {"campaign", "objective", "primary_readout", "primary_readout_basis", "eval_data", "units",
                            "assays_in_dataset", "results", "dms_labels_opened", "provenance"}
    assert summary["campaign"] == load(pipeline / "resolved.json")["campaign_id"]
    assert (summary["objective"], summary["primary_readout"], summary["eval_data"]) == ("direct_logits", "delta", ["proteingym"])
    assert "assumed, not confirmed" in summary["primary_readout_basis"]
    assert summary["units"] == ["base", "all"] and summary["assays_in_dataset"] == {"proteingym": 3}
    assert summary["dms_labels_opened"] is True
    # one entry per (dataset, unit, readout); both readouts always; the primary flag marks exactly one readout per unit
    assert {(e["unit"], e["readout"]) for e in summary["results"]} == {(u, r) for u in ("base", "all") for r in ("delta", "dual")}
    for e in summary["results"]:
        base = {"dataset", "unit", "readout", "assays", "assays_finite", "macro_spearman", "primary"}
        assert set(e) == base | ({"base_macro_same_assays", "delta_vs_base"} if e["unit"] != "base" else set())
        assert e["primary"] == (e["readout"] == "delta") and e["assays"] == 3 and e["assays_finite"] <= e["assays"]
        if e["unit"] != "base":
            assert e["delta_vs_base"] == pytest.approx(e["macro_spearman"] - e["base_macro_same_assays"])


def test_per_assay_csv_is_unchanged(pipeline):
    with open(pipeline / "evaluate" / "per_assay.csv", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    assert reader.fieldnames == ["dataset", "unit", "assay_id", "readout", "n_variants", "spearman"]
    assert {(r["unit"], r["readout"]) for r in rows} == {(u, r) for u in ("base", "all") for r in ("delta", "dual")}
    assert {r["assay_id"] for r in rows} == {"ASSAY_A", "ASSAY_B", "ASSAY_C"} and all(int(r["n_variants"]) > 0 for r in rows)


def test_a_unit_scored_on_its_own_assay_reports_one_assay_against_the_same_base_assay(env):
    plan = plan_for(env, method="score-one", assays=("ASSAY_A",))
    run_stages(plan)
    summary = load(plan.campaign_dir / "evaluate" / "summary.json")
    assert (summary["objective"], summary["primary_readout"], summary["units"]) == ("score", "dual", ["base", "ASSAY_A"])
    own = next(e for e in summary["results"] if e["unit"] == "ASSAY_A" and e["readout"] == "dual")
    base_on_a = next(r for r in csv.DictReader(open(plan.campaign_dir / "evaluate" / "per_assay.csv"))
                     if r["unit"] == "base" and r["assay_id"] == "ASSAY_A" and r["readout"] == "dual")
    assert own["assays"] == 1 and own["base_macro_same_assays"] == pytest.approx(float(base_on_a["spearman"]))
