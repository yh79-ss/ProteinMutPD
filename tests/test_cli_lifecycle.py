"""`mpd status`, `logs`, `cancel` and `results` against hand-built campaign directories and the fake Slurm tools."""

import json

import pytest
from fx_cli import cli_env, result_rows, run_cli, write_adapter, write_logs, write_summary  # noqa: F401
from fx_slurm import fake_slurm, stub_plan, write_stub_campaign  # noqa: F401

from mpd import results
from mpd.slurm import campaign

pytestmark = pytest.mark.usefixtures("cli_env")


def queue(plan, fake_slurm, states):
    """Put one fake job per stage in the queue (state per stage, default COMPLETED); returns {stage: job id}."""
    ids = {}
    for stage in plan.stages:
        state, extra = states.get(stage.full_name, ("COMPLETED", {}))
        jid = fake_slurm.add_job(name=f"mpd-{stage.full_name}-esm2650m", stage=stage.full_name, state=state,
                                 comment=f"mpd:{plan.campaign_id}:{plan.nonce}:{stage.full_name}", reason=extra.get("reason", "None"))
        if "elapsed" in extra or "exitcode" in extra:
            fake_slurm.set_state(jid, state, elapsed=extra.get("elapsed"), exitcode=extra.get("exitcode"))
        ids[stage.full_name] = jid
    return ids


@pytest.fixture
def running(cli_env, fake_slurm):
    """A submitted campaign: prepare and teacher finished, train running, the rest waiting on it."""
    plan = stub_plan(cli_env.tmp)
    ids = queue(plan, fake_slurm, {"prepare": ("COMPLETED", {"elapsed": "00:03:11"}), "teacher": ("COMPLETED", {"elapsed": "00:40:00"}),
                                   "train": ("RUNNING", {"elapsed": "01:10:00"}),
                                   "score": ("PENDING", {"reason": "Dependency"}), "evaluate": ("PENDING", {"reason": "Dependency"})})
    write_stub_campaign(plan, state="submitted", jobs=ids)
    for marker in ("contexts/MANIFEST.json", "teacher/COMPLETE.json"):               # what the finished stages leave behind
        (plan.campaign_dir / marker).parent.mkdir(parents=True, exist_ok=True)
        (plan.campaign_dir / marker).write_text("{}", encoding="utf-8")
    return plan, ids


def status(capsys, plan, *extra, ref=None):
    return run_cli(capsys, "status", ref or plan.campaign_id, "--home", plan.options.home, *extra)


# ---------------------------------------------------------------- status
def test_status_lists_campaigns_and_resolves_a_prefix(capsys, running):
    plan, _ = running
    rc, out, _ = run_cli(capsys, "status", "--home", plan.options.home)
    assert rc == 0 and plan.campaign_id in out and "submitted" in out and "mpd status <campaign id" in out
    rc, out, _ = status(capsys, plan, ref=plan.campaign_id[:20])
    assert rc == 0 and f"campaign   {plan.campaign_id}" in out
    rc, out, _ = status(capsys, plan, ref=str(plan.campaign_dir))
    assert rc == 0 and f"directory  {plan.campaign_dir}" in out


def test_status_with_no_campaigns_and_with_no_home(capsys, cli_env):
    rc, out, _ = run_cli(capsys, "status", "--home", cli_env.home)
    assert rc == 0 and "no campaigns under" in out
    rc, _, err = run_cli(capsys, "status")
    assert rc == 2 and "give --home" in err


def table_rows(out: str) -> dict[str, list[str]]:
    """The status table: stage -> its whitespace-separated fields (the block under the `stage state elapsed reason` header)."""
    lines = out.splitlines()
    first = next(i for i, ln in enumerate(lines) if ln.split()[:4] == ["stage", "state", "elapsed", "reason"])
    block = []
    for ln in lines[first + 1:]:
        if not ln.strip():
            break
        block.append(ln.split())
    return {fields[0]: fields for fields in block}


def test_status_shows_slurm_state_progress_and_why_a_job_waits(capsys, running):
    plan, ids = running
    rc, out, err = status(capsys, plan)
    assert rc == 0, err
    rows = table_rows(out)
    assert list(rows) == [s.full_name for s in plan.stages] and all(rows[s][4] == ids[s] for s in rows)         # job column
    assert rows["prepare"][:4] == ["prepare", "COMPLETED", "00:03:11", "-"] and rows["prepare"][6] == "1/1"
    assert rows["train"][:4] == ["train", "RUNNING", "01:10:00", "-"] and rows["train"][6] == "0/1"
    assert rows["score"][:4] == ["score", "PENDING", "00:00:00", "Dependency"]
    assert "waiting:" in out and "score       Dependency: it waits for train to finish. Nothing to do." in out
    assert "state      submitted" in out
    assert "next:" in out and "mpd logs" in out and "train" in out.split("next:")[1]


def test_status_of_a_failed_stage_shows_the_scrubbed_end_of_its_error_log_and_the_next_step(capsys, cli_env, fake_slurm):
    plan = stub_plan(cli_env.tmp)
    ids = queue(plan, fake_slurm, {"train": ("FAILED", {"exitcode": "1:0", "elapsed": "00:05:00"}),
                                   "score": ("CANCELLED", {}), "evaluate": ("CANCELLED", {})})
    write_stub_campaign(plan, state="submitted", jobs=ids)
    write_logs(plan.campaign_dir, {f"mpd-train-esm2650m_{ids['train']}.err":
                                   "Traceback (most recent call last):\nMPD_NO_GPU: no usable GPU for tester01 account 123456789012\n"})
    rc, out, _ = status(capsys, plan)
    assert rc == 0
    assert "FAILED" in out and "last lines of the error log of train" in out and "MPD_NO_GPU: no usable GPU" in out
    assert "123456789012" not in out and "tester01" not in out.split("last lines")[1]
    assert "mpd logs" in out and "train --err" in out and "--force" in out and "TROUBLESHOOTING" in out


def test_status_when_everything_finished_points_at_results(capsys, cli_env, fake_slurm):
    plan = stub_plan(cli_env.tmp)
    write_stub_campaign(plan, state="submitted", jobs=queue(plan, fake_slurm, {}))
    write_summary(plan.campaign_dir, result_rows())
    rc, out, _ = status(capsys, plan)
    assert rc == 0 and f"mpd results {plan.campaign_dir}" in out


def test_status_names_the_campaign_by_id_when_mpd_home_finds_it(capsys, running, monkeypatch):
    plan, _ = running
    monkeypatch.setenv("MPD_HOME", str(plan.options.home))
    rc, out, _ = run_cli(capsys, "status", plan.campaign_id)
    assert rc == 0 and f"mpd logs {plan.campaign_id} train" in out and str(plan.campaign_dir) not in out.split("next:")[1]


def test_status_without_accounting_records_says_so(capsys, running, fake_slurm):
    plan, _ = running
    fake_slurm.remove("sacct")
    fake_slurm.remove("squeue")
    rc, out, _ = status(capsys, plan)
    assert rc == 0 and "no job states" in out


def test_status_of_a_campaign_that_was_never_submitted(capsys, cli_env, fake_slurm):
    plan = stub_plan(cli_env.tmp)
    write_stub_campaign(plan)
    rc, out, err = status(capsys, plan)
    assert rc == 1 and "never submitted" in err
    rc, out, _ = run_cli(capsys, "status", "--home", plan.options.home)
    assert "unsubmitted" in out


def test_status_names_a_partial_campaign_and_how_to_continue(capsys, cli_env, fake_slurm):
    plan = stub_plan(cli_env.tmp)
    ids = queue(plan, fake_slurm, {})
    write_stub_campaign(plan, state="partial", jobs={"prepare": ids["prepare"]}, message="controller down")
    rc, out, _ = status(capsys, plan)
    assert rc == 0 and "partial" in out and "controller down" in out and "--force" in out and "mpd cancel" in out


def test_status_error_for_an_unknown_or_ambiguous_campaign(capsys, running, cli_env):
    plan, _ = running
    rc, _, err = status(capsys, plan, ref="nothing-like-it")
    assert rc == 2 and "no campaign called 'nothing-like-it'" in err and plan.campaign_id in err
    clone = plan.campaign_dir.parent / (plan.campaign_id + "b")
    clone.mkdir()
    (clone / "spec.json").write_text("{}", encoding="utf-8")
    rc, _, err = status(capsys, plan, ref=plan.campaign_id[:20])
    assert rc == 2 and "matches 2 campaigns" in err


# ---------------------------------------------------------------- logs
def test_logs_lists_the_newest_job_of_each_stage(capsys, running):
    plan, ids = running
    write_logs(plan.campaign_dir, {f"mpd-prepare-esm2650m_{ids['prepare']}.out": "preparing\n", f"mpd-prepare-esm2650m_{ids['prepare']}.err": "",
                                   "mpd-score-base-esm2650m_2001_0.out": "a\n", "mpd-score-base-esm2650m_2001_1.out": "b\n",
                                   "mpd-score-base-esm2650m_2002_0.out": "newer\n", "other.txt": "ignored\n"})
    rc, out, _ = run_cli(capsys, "logs", plan.campaign_id, "--home", plan.options.home)
    assert rc == 0
    rows = {line.split()[0]: line.split() for line in out.splitlines() if line.startswith(("prepare", "score-base"))}
    assert rows["prepare"][1] == ids["prepare"] and rows["score-base"][1] == "2002" and rows["score-base"][2] == "1"
    assert "other.txt" not in out and "mpd logs <campaign> <stage>" in out


def test_logs_shows_the_tail_of_the_newest_file_of_a_stage(capsys, running):
    plan, ids = running
    text = "".join(f"step {i}\n" for i in range(100))
    write_logs(plan.campaign_dir, {f"mpd-train-esm2650m_{ids['train']}.out": text, f"mpd-train-esm2650m_{ids['train']}.err": "warning: x\n",
                                   "mpd-score-esm2650m_3001_0.out": "score task 0\n", "mpd-score-esm2650m_3001_1.out": "score task 1\n",
                                   "mpd-score-base-esm2650m_3002_0.out": "base\n"})
    rc, out, _ = run_cli(capsys, "logs", str(plan.campaign_dir), "train", "--tail", "3")
    assert rc == 0 and out.splitlines()[1:] == ["step 97", "step 98", "step 99"] and f"(job {ids['train']})" in out.splitlines()[0]
    rc, out, _ = run_cli(capsys, "logs", str(plan.campaign_dir), "train", "--err")
    assert rc == 0 and "warning: x" in out
    rc, out, _ = run_cli(capsys, "logs", str(plan.campaign_dir), "score")
    assert "score task 1" in out and "task 1; 1 other task log(s)" in out and "base" not in out         # score is not score-base


def test_logs_errors(capsys, running, cli_env):
    plan, _ = running
    rc, _, err = run_cli(capsys, "logs", str(plan.campaign_dir))
    assert rc == 1 and "no log files" in err
    write_logs(plan.campaign_dir, {"mpd-prepare-esm2650m_5_.out": "x\n", "mpd-prepare-esm2650m_5.out": "x\n"})
    rc, _, err = run_cli(capsys, "logs", str(plan.campaign_dir), "train")
    assert rc == 2 and "no logs for stage 'train'" in err and "prepare" in err
    rc, _, err = run_cli(capsys, "logs", str(plan.campaign_dir), "prepare", "--err")
    assert rc == 1 and "no .err file yet" in err
    assert run_cli(capsys, "logs", str(plan.campaign_dir), "prepare", "--tail", "0")[0] == 2


# ---------------------------------------------------------------- cancel
def test_cancel_cancels_the_jobs_that_are_alive_and_marks_the_campaign(capsys, running, fake_slurm):
    plan, ids = running
    rc, out, err = run_cli(capsys, "cancel", plan.campaign_id, "--home", plan.options.home)
    assert rc == 0, err
    assert f"cancelled 3 job(s) of {plan.campaign_id}" in out
    scancels = [c for c in fake_slurm.calls("scancel")]
    assert scancels and {ids["train"], ids["score"], ids["evaluate"]} <= {a for c in scancels for a in c.argv}
    assert json.loads((plan.campaign_dir / "SUBMITTED.json").read_text())["state"] == "cancelled"
    rc, out, _ = run_cli(capsys, "cancel", plan.campaign_id, "--home", plan.options.home)
    assert rc == 0 and "already cancelled" in out or "no job" in out


def test_cancel_reports_a_scancel_failure_as_an_error_with_the_ids(capsys, running, fake_slurm):
    plan, ids = running
    fake_slurm.scancel_fails()
    rc, out, err = run_cli(capsys, "cancel", str(plan.campaign_dir))
    assert rc == 2 and "scancel failed" in err and ids["train"] in err
    assert json.loads((plan.campaign_dir / "SUBMITTED.json").read_text())["state"] == "partial"


# ---------------------------------------------------------------- results
def make_results(plan, units=("all",), adapters=3, **kw):
    rows = []
    for unit in units:
        rows += result_rows(unit=unit, **kw)
    write_summary(plan.campaign_dir, rows)
    for unit in units[:adapters]:
        write_adapter(plan.campaign_dir, unit)
    return plan.campaign_dir


def test_results_prints_both_readouts_base_versus_distilled_and_where_the_adapter_is(capsys, cli_env):
    plan = stub_plan(cli_env.tmp)
    write_stub_campaign(plan)
    root = make_results(plan)
    rc, out, err = run_cli(capsys, "results", str(root))
    assert rc == 0, err
    lines = out.splitlines()
    assert any(ln.startswith("proteingym") and "delta " in ln and "0.4000" in ln for ln in lines)         # frozen student, delta
    assert any(ln.startswith("proteingym") and "dual *" in ln and "0.3000" in ln for ln in lines)         # primary marked
    tuned = [ln for ln in lines if ln.startswith("proteingym") and "all" in ln.split()]
    assert any("0.4500" in ln and "+0.0500" in ln for ln in tuned) and any("0.3800" in ln and "+0.0800" in ln for ln in tuned)
    assert "assumption of the maintainer" in out and "both readouts are always reported" in out
    assert str(root / "train" / "all" / "checkpoint_00100") in out
    assert "not backed up" in out and "rsync -av <netid>@grace.hprc.tamu.edu:" in out and "docs/RESULTS.md" in out


def test_results_by_id_and_when_the_evaluate_stage_has_not_run(capsys, cli_env):
    plan = stub_plan(cli_env.tmp)
    write_stub_campaign(plan)
    rc, out, _ = run_cli(capsys, "results", plan.campaign_id, "--home", plan.options.home)
    assert rc == 1 and "no results yet" in out and f"mpd status {plan.campaign_id}" in out
    make_results(plan)
    assert run_cli(capsys, "results", plan.campaign_id, "--home", plan.options.home)[0] == 0


def test_results_of_many_units_are_summarised(capsys, cli_env):
    plan = stub_plan(cli_env.tmp)
    write_stub_campaign(plan)
    units = [f"ASSAY_{i:02d}" for i in range(15)]
    root = make_results(plan, units=units, adapters=15, assays=1)
    rc, out, _ = run_cli(capsys, "results", str(root))
    assert rc == 0 and "mean of 15 units (15 improved)" in out
    assert "... and 10 more under" in out
    assert "ASSAY_14" not in out.split("distilled student")[1].split("files")[0]


def test_results_tolerate_a_summary_with_missing_or_extra_fields(tmp_path):
    root = tmp_path / "c"
    write_summary(root, [{"dataset": "venusmuthub", "unit": "all", "readout": "delta", "macro_spearman": float("nan"), "future_field": 1},
                         {"unit": "base"}])
    text, found = results.render(root)
    assert found and "nan" in text and "venusmuthub" in text
    assert results.format_results({"results": []}, root).count("summary.json has no results") == 1
    (root / "evaluate" / "summary.json").write_text("{broken", encoding="utf-8")
    assert results.render(root)[1] is False


def test_selected_adapters_skip_aside_and_partial_folders(tmp_path):
    root = tmp_path / "c"
    write_adapter(root, "A", 50)
    write_adapter(root, "B.incomplete-1700000000", 50)
    write_adapter(root, "C.partial", 50)
    (root / "train" / "D").mkdir()
    assert [u for u, _ in results.selected_adapters(root)] == ["A"]
    assert campaign.read_marker(root) is None                       # the helpers wrote no marker
