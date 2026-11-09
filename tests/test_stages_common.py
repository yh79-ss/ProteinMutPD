"""What every stage shares (task selection, shard counts, provenance). None of it needs torch."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mpd.spec import RunSpec
from mpd.stages import common

SPEC = RunSpec(teacher="esm2-35m", student="esm2-8m", train_data="proteingym", method="score-one",
               eval_data=("proteingym",), assays=("ASSAY_A",))


def campaign_dir(tmp_path: Path, tasks: dict[str, list], resolved: dict | None = None) -> Path:
    root = tmp_path / "runs" / "c1"
    (root / "tasks").mkdir(parents=True)
    (root / "spec.json").write_text(SPEC.to_json())
    (root / "resolved.json").write_text(json.dumps({"campaign_id": "c1", **(resolved or {})}))
    for stage, entries in tasks.items():
        (root / "tasks" / f"{stage}.json").write_text(json.dumps(entries))
    return root


def pick(root: Path, *args: str):
    seen = []
    common.stage_main(lambda c, task: seen.append(task), "test", ["--campaign-dir", str(root), "--stage", "score", *args])
    return seen[0]


# -- task selection (B2) -----------------------------------------------------------

def test_a_long_task_list_is_indexed_through_its_last_single_task_chunk(tmp_path):
    """1001 tasks make a 1000-task array plus a one-task chunk; the script passes that chunk's offset."""
    tasks = [{"unit": f"u{i}", "dataset": "proteingym", "shard": 0, "num_shards": 1} for i in range(1001)]
    root = campaign_dir(tmp_path, {"score": tasks})
    assert pick(root, "--task-index", "1000")["unit"] == "u1000"
    assert pick(root, "--task-index", "0")["unit"] == "u0" and pick(root, "--task-index", "999")["unit"] == "u999"


def test_several_tasks_without_an_index_is_an_error_that_says_so(tmp_path, capsys):
    root = campaign_dir(tmp_path, {"score": [{"unit": "a"}, {"unit": "b"}]})
    with pytest.raises(SystemExit) as exc:
        pick(root)
    err = capsys.readouterr().err
    assert exc.value.code == 2 and "stage score has 2 tasks and no --task-index was given" in err and "sbatch script must pass it" in err


@pytest.mark.parametrize("index", ["2", "-1"])
def test_an_index_outside_the_task_list_is_rejected(tmp_path, capsys, index):
    root = campaign_dir(tmp_path, {"score": [{"unit": "a"}, {"unit": "b"}]})
    with pytest.raises(SystemExit):
        pick(root, "--task-index", index)
    assert "outside 0..1" in capsys.readouterr().err


def test_a_single_task_stage_needs_no_index(tmp_path):
    root = campaign_dir(tmp_path, {"score": [{"unit": "only"}]})
    assert pick(root)["unit"] == "only" and pick(root, "--task-index", "0")["unit"] == "only"


# -- shard counts (M5) ------------------------------------------------------------------

def shard_tasks(unit: str, num: int, dataset: str = "proteingym") -> list[dict]:
    return [{"unit": unit, "dataset": dataset, "shard": s, "num_shards": num} for s in range(num)]


def test_the_shard_count_comes_from_the_units_own_tasks_then_from_resolved_json(tmp_path):
    tasks = {"score-base": shard_tasks("base", 4), "score": shard_tasks("ASSAY_A", 1)}
    c = common.Campaign.load(campaign_dir(tmp_path, tasks, {"score_shards": {"proteingym": 4, "venusmuthub": 1}}))
    assert c.expected_shards("proteingym", "base") == 4
    assert c.expected_shards("proteingym", "ASSAY_A") == 1          # a unit scored on its own small table has one shard
    assert c.expected_shards("venusmuthub", "base") == 1            # no task for this pair: the per-dataset record
    assert c.expected_shards("venusmuthub", "unknown") == 1


def test_a_campaign_without_any_record_has_no_expected_shard_count(tmp_path):
    c = common.Campaign.load(campaign_dir(tmp_path, {}))
    assert c.expected_shards("proteingym", "base") is None


def test_contradicting_tasks_fall_back_to_the_per_dataset_record(tmp_path):
    tasks = {"score": shard_tasks("ASSAY_A", 2) + shard_tasks("ASSAY_A", 3)}
    c = common.Campaign.load(campaign_dir(tmp_path, tasks, {"score_shards": {"proteingym": 3}}))
    assert c.expected_shards("proteingym", "ASSAY_A") == 3


# -- provenance and the torch-free import -------------------------------------------------

def test_provenance_is_plain_json_without_paths():
    info = common.provenance()
    text = json.dumps(info)
    assert {"mpd", "python", "platform", "host", "utc", "torch", "transformers", "peft", "slurm_job_id"} <= set(info)
    assert info["utc"].endswith("Z") and info["mpd"]
    assert "/scratch" not in text and (len(str(Path.home())) < 2 or str(Path.home()) not in text)


def test_provenance_reports_the_slurm_job(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "4242")
    assert common.provenance()["slurm_job_id"] == "4242"


def test_importing_the_shared_stage_module_does_not_import_torch():
    src = str(Path(common.__file__).resolve().parents[2])
    code = (f"import sys; sys.path.insert(0, {src!r}); import mpd.stages.common; "
            "assert 'torch' not in sys.modules, 'torch imported by stages.common'")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    done = subprocess.run([sys.executable, "-s", "-c", code], env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]


def test_the_gpu_guard_does_nothing_without_the_variable(monkeypatch):
    monkeypatch.delenv("MPD_REQUIRE_CUDA", raising=False)
    common.require_cuda()                                            # would need torch if it did anything
    monkeypatch.setenv("MPD_REQUIRE_CUDA", "0")
    common.require_cuda()
