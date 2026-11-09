"""mpd.slurm.submit.preflight_checks: `sbatch --test-only` once per job shape, never queueing anything."""

import os

import pytest

from fx_slurm import BAD_GRES, BAD_TIME, CONTROLLER_DOWN, clean_path, fake_slurm, stub_plan, stub_render  # noqa: F401
from mpd.checks import Check
from mpd.slurm.submit import preflight_checks, write_campaign


@pytest.fixture(autouse=True)
def no_real_slurm(monkeypatch):
    monkeypatch.setenv("PATH", clean_path(os.environ.get("PATH", "")))


@pytest.fixture
def plan(tmp_path):
    return stub_plan(tmp_path)


def by_label(checks):
    return {c.name: c for c in checks}


def test_each_distinct_shape_is_test_submitted_once_and_nothing_is_queued(fake_slurm, plan):
    checks = preflight_checks(plan, render=stub_render)
    assert all(isinstance(c, Check) and c.level == "ok" for c in checks)
    # prepare and evaluate share a CPU shape, score-base and score share an array shape: 6 stages, 4 shapes
    assert sorted(c.name for c in checks) == sorted(["sbatch --test-only (prepare, evaluate)", "sbatch --test-only (teacher)",
                                                     "sbatch --test-only (train)", "sbatch --test-only (score-base, score)"])
    calls = fake_slurm.calls("sbatch")
    assert len(calls) == 4 and all(c.test_only and "--test-only" in c.argv for c in calls)
    assert fake_slurm.jobs() == {} and fake_slurm.submitted() == []                       # the fake saw only --test-only calls
    assert fake_slurm.calls("squeue") == [] and fake_slurm.calls("scancel") == []
    gpu = by_label(checks)["sbatch --test-only (teacher)"]
    assert gpu.message == "Slurm would accept gpu:a100:1 on partition gpu, 01:00:00, 24 CPUs, 180G"
    assert by_label(checks)["sbatch --test-only (prepare, evaluate)"].message == "Slurm would accept no GPU requested, 01:00:00, 4 CPUs, 16G"
    assert by_label(checks)["sbatch --test-only (score-base, score)"].message.endswith(", array")
    assert "123456789012" not in " ".join(c.message for c in checks)               # the account number is never echoed


def test_a_refused_shape_is_an_error_with_the_offending_line_and_slurms_words(fake_slurm, plan):
    fake_slurm.reject_stage("train", BAD_GRES)
    checks = by_label(preflight_checks(plan, render=stub_render))
    bad = checks["sbatch --test-only (train)"]
    assert bad.level == "error" and "#SBATCH --gres=gpu:a100:1" in bad.message and "Requested node configuration is not available" in bad.message
    assert "--gpu" in bad.fix                                       # the same hint submit prints
    assert "#SBATCH --time" not in bad.message                      # only the lines that explain the refusal
    assert [c.level for n, c in checks.items() if n != "sbatch --test-only (train)"] == ["ok"] * 3


def test_a_time_limit_refusal_points_at_the_time_line(fake_slurm, plan):
    fake_slurm.reject_stage("train", BAD_TIME)
    bad = by_label(preflight_checks(plan, render=stub_render))["sbatch --test-only (train)"]
    assert "#SBATCH --time=02:00:00" in bad.message and "--train-time" in bad.fix


def test_existing_scripts_are_tested_as_written_without_dependency_lines(fake_slurm, plan):
    scripts = write_campaign(plan, render=stub_render)
    text = scripts["train"].read_text().replace("#SBATCH --nodes=1\n", "#SBATCH --nodes=1\n#SBATCH --dependency=afterok:555\n#SBATCH --kill-on-invalid-dep=yes\n")
    scripts["train"].write_text(text)
    checks = preflight_checks(plan, scripts)
    assert all(c.level == "ok" for c in checks), [c.message for c in checks]               # an unknown dependency id would be refused
    for call in fake_slurm.calls("sbatch"):
        assert "dependency" not in call.info["directives"] and "kill-on-invalid-dep" not in call.info["directives"]
    assert scripts["train"].read_text() == text                                              # the original is untouched
    assert not [c for c in fake_slurm.calls("sbatch") if os.path.exists(c.info["script"])]    # and the temporary copies are gone


def test_the_largest_array_of_a_shape_stands_for_the_shape(fake_slurm, tmp_path):
    plan = stub_plan(tmp_path, minimal=True)
    base = stub_render(plan, plan.stages[1])
    small = base.replace("#SBATCH --job-name", "#SBATCH --array=0-2%8\n#SBATCH --job-name", 1)
    large = base.replace("#SBATCH --job-name", "#SBATCH --array=0-999%8\n#SBATCH --job-name", 1)
    scripts = {"prepare": tmp_path / "a.sbatch", "train": tmp_path / "b.sbatch", "evaluate": tmp_path / "c.sbatch"}
    scripts["prepare"].write_text(stub_render(plan, plan.stages[0]))
    scripts["train"].write_text(small)
    scripts["evaluate"].write_text(large)
    plan.stages[2].tasks = [{}, {}]                          # evaluate stands in for a second array here
    checks = preflight_checks(plan, scripts)
    arrays = [c.info["directives"].get("array") for c in fake_slurm.calls("sbatch") if "array" in c.info["directives"]]
    assert arrays == ["0-999%8"] and len(checks) == 2


def test_it_is_skipped_when_sbatch_is_not_on_path(fake_slurm, plan):
    fake_slurm.remove("sbatch")
    (check,) = preflight_checks(plan, render=stub_render)
    assert check.level == "skip" and "not a Slurm login node" in check.message and fake_slurm.calls("sbatch") == []


def test_a_silent_controller_gives_one_skip_per_shape_and_one_call_only(fake_slurm, plan):
    fake_slurm.rule(do="fail", message=CONTROLLER_DOWN)           # applies to --test-only as well
    checks = preflight_checks(plan, render=stub_render)
    assert [c.level for c in checks] == ["skip"] * 4 and "did not answer" in checks[0].message
    assert len(fake_slurm.calls("sbatch")) == 1                    # not four hung calls on a login node
    assert "not tested: the Slurm controller did not answer" in checks[1].message


def test_the_whole_test_submission_has_a_time_budget(fake_slurm, plan):
    fake_slurm.configure(sbatch={"duration_s": 100})
    checks = preflight_checks(plan, render=stub_render, budget_s=150, clock=fake_slurm.clock)
    assert [c.level for c in checks] == ["ok", "ok", "skip", "skip"] and len(fake_slurm.calls("sbatch")) == 2
    assert "150 s time budget" in checks[2].message


def test_a_refusal_slurm_words_in_an_unexpected_way_is_still_an_error(fake_slurm, plan):
    fake_slurm.reject_stage("teacher", "sbatch: error: something odd happened")
    bad = by_label(preflight_checks(plan, render=stub_render))["sbatch --test-only (teacher)"]
    assert bad.level == "error" and "something odd" in bad.message and bad.fix is None
    assert "#SBATCH --gres=gpu:a100:1" in bad.message                    # no keyword matched: every defining line is shown
