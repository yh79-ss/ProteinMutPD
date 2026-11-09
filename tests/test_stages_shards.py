"""Score shards: only files for the planned count are used, and a leftover count cannot wedge evaluate (M5, C9)."""

import json

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from fx_stages import env, plan_for, run_stages  # noqa: F401
from mpd.stages import evaluate, score

BEFORE_SCORING = {"prepare", "teacher", "train"}


def results(plan):
    return json.loads((plan.campaign_dir / "evaluate" / "summary.json").read_text())["results"]


def plant_leftovers(plan, num, units=("base", "all")):
    """Shard files of an earlier plan with `num` shards, holding scores that would change every result if they were read."""
    eval_contexts = np.load(plan.campaign_dir / "contexts" / "eval__proteingym.npz")["pos"].shape[0]
    for unit in units:
        folder = plan.campaign_dir / "score" / "proteingym" / unit
        folder.mkdir(parents=True, exist_ok=True)
        for shard in range(num):
            rows = np.arange(eval_contexts)[shard::num]
            np.savez(folder / f"shard_{shard:03d}_of_{num:03d}.npz", rows=rows, scores=np.full((len(rows), 20), 99.0, np.float32))
            (folder / f"shard_{shard:03d}_of_{num:03d}.json").write_text("{}")


def test_leftover_shards_of_an_earlier_count_do_not_wedge_a_three_shard_campaign(env):
    plan = plan_for(env, score_shards=3)
    run_stages(plan, only=BEFORE_SCORING)
    plant_leftovers(plan, 2)                                           # an earlier 2-shard plan wrote these
    run_stages(plan, only={"score-base", "score", "evaluate"})         # score must not treat them as done, evaluate must not read them
    folder = plan.campaign_dir / "score" / "proteingym" / "base"
    assert sorted(p.name for p in folder.glob("*.json")) == sorted(
        [f"shard_{i:03d}_of_002.json" for i in range(2)] + [f"shard_{i:03d}_of_003.json" for i in range(3)])
    with_leftovers = results(plan)
    assert all(np.isfinite(e["macro_spearman"]) and e["assays_finite"] == e["assays"] for e in with_leftovers)
    for stale in folder.glob("*_of_002.*"):
        stale.unlink()
    run_stages(plan, only={"evaluate"})
    assert results(plan) == with_leftovers                              # the planted 99.0 scores had no influence


def test_evaluate_ignores_leftovers_when_all_planned_shards_are_present(env):
    plan = plan_for(env, score_shards=3)
    run_stages(plan)
    expected = results(plan)
    plant_leftovers(plan, 2)
    run_stages(plan, only={"evaluate"})
    assert results(plan) == expected


def test_a_missing_planned_shard_is_reported_with_what_to_do_even_when_another_count_is_complete(env):
    plan = plan_for(env, score_shards=3)
    run_stages(plan, only=BEFORE_SCORING | {"score-base"})
    folder = plan.campaign_dir / "score" / "proteingym" / "base"
    plant_leftovers(plan, 2, units=("base",))
    (folder / "shard_002_of_003.json").unlink()
    with pytest.raises(RuntimeError) as exc:
        run_stages(plan, only={"evaluate"})
    message = str(exc.value)
    assert "shards incomplete (have [0, 1] of 3; missing [2])" in message
    assert "shard files for other counts [2] are ignored" in message
    assert "Run the score stage again for this unit and dataset" in message


def test_the_count_in_resolved_json_is_used_when_the_task_files_are_gone(env):
    plan = plan_for(env, score_shards=2)
    run_stages(plan)
    root = plan.campaign_dir
    for name in ("score", "score-base"):
        (root / "tasks" / f"{name}.json").unlink()
    resolved = json.loads((root / "resolved.json").read_text())
    resolved["score_shards"] = {"proteingym": 3}
    (root / "resolved.json").write_text(json.dumps(resolved))
    with pytest.raises(RuntimeError, match=r"have \[\] of 3; missing \[0, 1, 2\]\); shard files for other counts \[2\] are ignored"):
        run_stages(plan, only={"evaluate"})
    resolved["score_shards"] = {"proteingym": 2}
    (root / "resolved.json").write_text(json.dumps(resolved))
    run_stages(plan, only={"evaluate"})


def test_a_campaign_that_records_no_count_keeps_the_old_rule(env):
    plan = plan_for(env, score_shards=2)
    run_stages(plan)
    root = plan.campaign_dir
    for name in ("score", "score-base"):
        (root / "tasks" / f"{name}.json").unlink()
    resolved = json.loads((root / "resolved.json").read_text())
    resolved.pop("score_shards", None)                                 # as a campaign planned before the count was recorded
    (root / "resolved.json").write_text(json.dumps(resolved))
    run_stages(plan, only={"evaluate"})                                # one complete set of files: accepted
    plant_leftovers(plan, 3, units=("base",))
    with pytest.raises(RuntimeError, match=r"shard files exist for counts \[2, 3\], expected exactly one"):
        run_stages(plan, only={"evaluate"})


def test_a_shard_file_written_for_another_count_is_refused(env):
    plan = plan_for(env, score_shards=2)
    run_stages(plan)
    path = plan.campaign_dir / "score" / "proteingym" / "base" / "shard_000_of_002.npz"
    with np.load(path) as z:
        content = {k: z[k] for k in z.files}
    np.savez(path, **{**content, "num_shards": np.int64(3)})
    with pytest.raises(RuntimeError, match="was written for 3 shards"):
        evaluate.assemble(evaluate.Campaign.load(plan.campaign_dir), "proteingym", "base")


def test_a_one_assay_unit_with_a_single_shard_is_assembled_next_to_a_three_shard_base(env):
    """A unit scored on its own small table may have fewer shards than the dataset default; its tasks say how many."""
    plan = plan_for(env, method="score-one", assays=("ASSAY_A",), score_shards=3)
    run_stages(plan, only={"prepare", "teacher", "train", "score-base"})
    root = plan.campaign_dir
    (root / "tasks" / "score.json").write_text(json.dumps([{"unit": "ASSAY_A", "dataset": "proteingym", "shard": 0, "num_shards": 1}]))
    resolved = json.loads((root / "resolved.json").read_text())
    resolved["score_shards"] = {"proteingym": 3}                        # the dataset default; the unit's own tasks win
    (root / "resolved.json").write_text(json.dumps(resolved))
    score.main(["--campaign-dir", str(root), "--stage", "score", "--task-index", "0"])
    run_stages(plan, only={"evaluate"})
    assert [p.name for p in (root / "score" / "proteingym" / "ASSAY_A").glob("*.json")] == ["shard_000_of_001.json"]
    assert len(list((root / "score" / "proteingym" / "base").glob("*.json"))) == 3
    assert {e["unit"] for e in results(plan)} == {"base", "ASSAY_A"}
