"""mpd.slurm.campaign: the marker, listing, Slurm state queries, progress on disk, log tails and cancel."""

import json
import os
import time

import pytest

from fx_slurm import clean_path, fake_slurm, stub_plan, write_stub_campaign  # noqa: F401
from mpd.slurm import campaign
from mpd.slurm.campaign import (CampaignError, SubmitError, campaign_state, cancel, create_marker, failure_log, failure_tail,
                                job_states, job_tag, list_campaigns, log_tail, parse_tag, pending_reasons, read_marker,
                                resolve_campaign, scrub, stage_log, stage_progress, write_marker)

CID = "esm2-3b__esm2-650m__proteingym-score-all__seed1__ab12cd34"


@pytest.fixture(autouse=True)
def no_real_slurm(monkeypatch):
    """Even a test that forgets `fake_slurm` cannot find the real Slurm clients on PATH."""
    monkeypatch.setenv("PATH", clean_path(os.environ.get("PATH", "")))


def make(tmp_path, state=None, jobs=None, **kw):
    plan = stub_plan(tmp_path, **kw)
    write_stub_campaign(plan, state=state, jobs=jobs)
    return plan


def tag(stage, nonce="0a1b2c3d4e5f", cid=CID):
    return job_tag(cid, nonce, stage)


# ---------------------------------------------------------------- marker and state
def test_tags_round_trip_and_reject_foreign_comments():
    assert parse_tag(job_tag("a__b", "n1", "score.1")) == ("a__b", "n1", "score.1")
    assert parse_tag(job_tag("odd:id", "n1", "train")) == ("odd:id", "n1", "train")
    assert parse_tag("") is None and parse_tag("other job") is None and parse_tag("mpd:only:two") is None


def test_no_marker_means_no_state_and_a_marker_without_state_reads_as_submitted(tmp_path):
    plan = make(tmp_path)
    assert campaign_state(plan.campaign_dir) is None and read_marker(plan.campaign_dir) is None
    # the marker of earlier versions: written only after the whole chain was queued, no state field
    (plan.campaign_dir / "SUBMITTED.json").write_text(json.dumps({"submitted_at": "t0", "jobs": {"prepare": "7"}, "campaign_id": CID}))
    st = campaign_state(plan.campaign_dir)
    assert (st.state, st.jobs, st.campaign_id, st.path) == ("submitted", {"prepare": "7"}, CID, plan.campaign_dir)


@pytest.mark.parametrize("content", ["{not json", "[1, 2]", ""])
def test_an_unreadable_marker_is_reported_and_blocks(tmp_path, content):
    plan = make(tmp_path)
    (plan.campaign_dir / "SUBMITTED.json").write_text(content)
    st = campaign_state(plan.campaign_dir)
    assert st.state == "unreadable" and "SUBMITTED.json" in st.message and st.state in campaign.BLOCKING


def test_marker_writes_are_atomic_and_creation_is_exclusive(tmp_path):
    root = tmp_path / "c"
    assert create_marker(root, {"campaign_id": "x", "state": "submitting", "jobs": {}}) is True
    assert create_marker(root, {"campaign_id": "x", "state": "other", "jobs": {}}) is False       # somebody else got there first
    assert read_marker(root)["state"] == "submitting"
    write_marker(root, {"campaign_id": "x", "state": "submitted", "jobs": {"a": "1"}})
    data = read_marker(root)
    assert data["schema"] == 1 and data["state"] == "submitted" and "updated_at" in data
    assert sorted(p.name for p in root.iterdir()) == ["SUBMITTED.json"]                           # no temp files left


def test_list_campaigns_is_newest_first_and_includes_unsubmitted_ones(tmp_path):
    home = tmp_path / "home"
    old = stub_plan(tmp_path, campaign_id="old__aaaa1111")
    new = stub_plan(tmp_path, campaign_id="new__bbbb2222")
    dry = stub_plan(tmp_path, campaign_id="dry__cccc3333")
    write_stub_campaign(old, state="submitted", jobs={"prepare": "1"})
    write_stub_campaign(new, state="partial", jobs={"prepare": "2"})
    write_stub_campaign(dry)
    write_marker(old.campaign_dir, {"campaign_id": old.campaign_id, "state": "submitted", "submitted_at": "2026-11-01T10:00:00-0600", "jobs": {}})
    write_marker(new.campaign_dir, {"campaign_id": new.campaign_id, "state": "partial", "submitted_at": "2026-11-08T10:00:00-0600", "jobs": {"prepare": "2"}})
    os.utime(dry.campaign_dir, (1.7e9, 1.7e9))              # written long ago, never submitted
    (home / "runs" / ".hidden").mkdir()
    (home / "runs" / "stray_file").write_text("x")
    (home / "runs" / "empty_dir").mkdir()
    got = [(c.campaign_id, c.state) for c in list_campaigns(home)]
    assert got[:2] == [("new__bbbb2222", "partial"), ("old__aaaa1111", "submitted")] and got[2] == ("dry__cccc3333", "unsubmitted")
    assert len(got) == 3
    assert list_campaigns(tmp_path / "nowhere") == []


def test_resolve_campaign_accepts_a_path_an_id_or_a_unique_prefix(tmp_path):
    home = tmp_path / "home"
    a = make(tmp_path, campaign_id="esm2-3b__esm2-650m__x__seed1__aaaa1111")
    b = make(tmp_path, campaign_id="esm2-3b__esm2-650m__x__seed2__bbbb2222")
    c = make(tmp_path, campaign_id="saprot-650m__esm2-35m__y__seed1__cccc3333")
    assert resolve_campaign(a.campaign_dir, home) == a.campaign_dir
    assert resolve_campaign(a.campaign_id, home) == a.campaign_dir
    assert resolve_campaign("saprot", home) == c.campaign_dir
    assert resolve_campaign(str(a.campaign_dir), None) == a.campaign_dir       # a path needs no home
    with pytest.raises(CampaignError, match="matches 2 campaigns") as exc:
        resolve_campaign("esm2-3b", home)
    assert a.campaign_id in str(exc.value) and b.campaign_id in str(exc.value)
    with pytest.raises(CampaignError, match="no campaign called 'nope'") as exc:
        resolve_campaign("nope", home)
    assert c.campaign_id in str(exc.value)                                      # lists what exists
    with pytest.raises(CampaignError, match="no campaigns under"):
        resolve_campaign("x", tmp_path / "empty")
    with pytest.raises(CampaignError, match="not a campaign directory"):
        resolve_campaign("x", None)
    assert issubclass(CampaignError, SubmitError)                                # the cli maps SubmitError to exit 2


# ---------------------------------------------------------------- Slurm state
def test_job_states_summarises_an_array_in_one_entry(fake_slurm, tmp_path):
    plan = make(tmp_path, units=("a", "b", "c", "d"))
    script = tmp_path / "home" / "runs" / plan.campaign_id / "scripts" / "train.sbatch"
    import subprocess
    out = subprocess.run(["sbatch", "--parsable", str(script)], capture_output=True, text=True).stdout.strip()
    assert out == "1001"
    assert job_states(["1001"]) == {"1001": campaign.JobState("PENDING", "00:00:00", "0:0", "mpd-train-esm2650m", "4 PENDING")}
    fake_slurm.set_state(1001, "RUNNING")
    st = job_states(["1001"])["1001"]
    assert st.state == "RUNNING" and st.detail == "2 RUNNING, 2 PENDING"
    fake_slurm.set_state(1001, "FAILED", exitcode="1:0", tasks=["COMPLETED", "FAILED", "TIMEOUT", "COMPLETED"])
    st = job_states(["1001"])["1001"]
    assert (st.state, st.exitcode) == ("FAILED", "0:0") or st.state == "FAILED"
    assert "2 COMPLETED" in st.detail and "1 FAILED" in st.detail and "1 TIMEOUT" in st.detail


def test_job_states_ignores_unknown_ids_and_empty_input(fake_slurm):
    assert job_states([]) == {} and job_states(["999"]) == {}


def test_job_states_survives_missing_or_broken_sacct_and_fills_in_from_squeue(fake_slurm, tmp_path):
    plan = make(tmp_path)
    import subprocess
    subprocess.run(["sbatch", "--parsable", str(plan.campaign_dir / "scripts" / "prepare.sbatch")], capture_output=True)
    fake_slurm.remove("sacct")
    got = job_states(["1001"])
    assert got["1001"].state == "PENDING" and got["1001"].name == "mpd-prepare-esm2650m"        # from squeue
    fake_slurm.restore("sacct")
    fake_slurm.sacct_fails()
    assert job_states(["1001"])["1001"].state == "PENDING"
    fake_slurm.remove("squeue")
    assert job_states(["1001"], squeue="squeue") == {}                                           # nothing answers: empty, no exception
    assert job_states(["1001"], squeue=None) == {}


def test_a_job_without_a_comment_reads_as_an_empty_comment_although_squeue_prints_null(fake_slurm):
    fake_slurm.add_job(name="plain", comment="", state="RUNNING")
    import subprocess
    raw = subprocess.run(["squeue", "-h", "-o", "%k"], capture_output=True, text=True).stdout.strip()
    assert raw == "(null)"                                             # what the real squeue prints for an unset comment
    rows, _ = campaign.read_queue("squeue")
    assert [r.comment for r in rows] == [""] and campaign.tagged(rows, "any") == []


def test_pending_reasons_name_why_a_job_waits(fake_slurm):
    a = fake_slurm.add_job(name="a", state="PENDING", reason="Dependency")
    b = fake_slurm.add_job(name="b", state="PENDING", reason="PartitionTimeLimit")
    c = fake_slurm.add_job(name="c", state="RUNNING")
    assert pending_reasons([a, b, c]) == {a: "Dependency", b: "PartitionTimeLimit"}
    assert pending_reasons([c]) == {}
    fake_slurm.squeue_fails()
    assert pending_reasons([a]) == {}


# ---------------------------------------------------------------- progress and logs
def test_stage_progress_counts_the_markers_the_stages_leave(tmp_path):
    plan = make(tmp_path, units=("u1", "u2"), datasets=("proteingym",), shards=2)
    root = plan.campaign_dir
    assert stage_progress(root) == {"prepare": (0, 1), "teacher": (0, 1), "train": (0, 2), "score-base": (0, 2), "score": (0, 4), "evaluate": (0, 1)}
    list(map(lambda p: (p.parent.mkdir(parents=True, exist_ok=True), p.write_text("{}")), [
        root / "contexts" / "MANIFEST.json", root / "teacher" / "COMPLETE.json", root / "train" / "u1" / "COMPLETE.json",
        root / "score" / "proteingym" / "base" / "shard_000_of_002.json", root / "score" / "proteingym" / "base" / "shard_001_of_002.json",
        root / "score" / "proteingym" / "u1" / "shard_000_of_002.json",
        # leftovers of an earlier --score-shards 3 must not count
        root / "score" / "proteingym" / "u1" / "shard_001_of_003.json", root / "score" / "proteingym" / "u2" / "shard_000_of_003.json"]))
    got = stage_progress(root)
    assert got["prepare"] == (1, 1) and got["teacher"] == (1, 1) and got["train"] == (1, 2)
    assert got["score-base"] == (2, 2) and got["score"] == (1, 4) and got["evaluate"] == (0, 1)
    assert list(got) == ["prepare", "teacher", "train", "score-base", "score", "evaluate"]       # pipeline order
    (root / "evaluate").mkdir()
    (root / "evaluate" / "summary.json").write_text("{}")
    assert stage_progress(root)["evaluate"] == (1, 1)


def test_stage_progress_uses_the_plans_shard_count_when_a_task_lacks_its_own(tmp_path):
    plan = make(tmp_path, shards=2)
    tasks = json.loads((plan.campaign_dir / "tasks" / "score.json").read_text())
    for t in tasks:
        t.pop("num_shards")
    (plan.campaign_dir / "tasks" / "score.json").write_text(json.dumps(tasks))
    d = plan.campaign_dir / "score" / "proteingym" / "all"
    d.mkdir(parents=True)
    (d / "shard_000_of_002.json").write_text("{}")
    (d / "shard_001_of_009.json").write_text("{}")
    assert stage_progress(plan.campaign_dir)["score"] == (1, 2)


def test_stage_progress_of_a_directory_without_tasks_is_empty(tmp_path):
    assert stage_progress(tmp_path) == {}


def write_log(plan, name, text, age=0):
    path = plan.campaign_dir / "logs" / name
    path.write_text(text)
    when = time.time() - age
    os.utime(path, (when, when))
    return path


def test_failure_tail_returns_the_newest_non_empty_log_of_the_stage(tmp_path):
    plan = make(tmp_path, state="submitted", jobs={"train": "500", "score": "600", "score-base": "700"})
    write_log(plan, "mpd-train-esm2650m_500_0.err", "old task\n", age=300)
    write_log(plan, "mpd-train-esm2650m_500_1.err", "Traceback\nValueError: boom\n", age=100)
    write_log(plan, "mpd-train-esm2650m_500_2.err", "", age=10)                # newest but empty: tasks that succeeded
    write_log(plan, "mpd-score-esm2650m_600.err", "score error\n")
    write_log(plan, "mpd-score-base-esm2650m_700.err", "base error\n")
    write_log(plan, "mpd-train-esm2650m_500_0.out", "stdout is not read\n")
    assert failure_tail(plan.campaign_dir, "train") == "Traceback\nValueError: boom"
    assert failure_tail(plan.campaign_dir, "score") == "score error"                # not score-base
    assert failure_tail(plan.campaign_dir, "score-base") == "base error"
    assert failure_log(plan.campaign_dir, "teacher") is None and failure_tail(plan.campaign_dir, "teacher") == ""
    assert failure_log(plan.campaign_dir, "train").name == "mpd-train-esm2650m_500_1.err"


def test_stdout_logs_are_found_the_same_way(tmp_path):
    plan = make(tmp_path, state="submitted", jobs={"train": "500", "score": "600", "score-base": "700"})
    write_log(plan, "mpd-train-esm2650m_500_0.out", "epoch 1\nepoch 2\n", age=50)
    write_log(plan, "mpd-train-esm2650m_500_0.err", "an error\n")
    write_log(plan, "mpd-score-esm2650m_600.out", "scored\n")
    write_log(plan, "mpd-score.1-esm2650m_601.out", "chunk one\n")
    write_log(plan, "mpd-score-base-esm2650m_700.out", "base\n")
    assert stage_log(plan.campaign_dir, "train", err=False).name == "mpd-train-esm2650m_500_0.out"
    assert log_tail(plan.campaign_dir, "train", 1, err=False) == "epoch 2" and log_tail(plan.campaign_dir, "train", err=True) == "an error"
    assert log_tail(plan.campaign_dir, "score", err=False) in ("scored", "chunk one")       # the base name covers its chunks
    assert log_tail(plan.campaign_dir, "score.1", err=False) == "chunk one" and log_tail(plan.campaign_dir, "score-base", err=False) == "base"
    assert stage_log(plan.campaign_dir, "evaluate", err=False) is None


def test_failure_tail_keeps_the_last_n_lines_and_what_a_progress_bar_left_visible(tmp_path):
    plan = make(tmp_path, state="submitted", jobs={"train": "500"})
    body = "".join(f"line {i}\n" for i in range(100)) + "10%\r50%\r100% done\n"
    write_log(plan, "mpd-train-esm2650m_500.err", body)
    lines = failure_tail(plan.campaign_dir, "train", n=3).splitlines()
    assert lines == ["line 98", "line 99", "100% done"]
    assert len(failure_tail(plan.campaign_dir, "train").splitlines()) == 20


def test_failure_tail_finds_logs_by_name_when_the_marker_is_gone(tmp_path):
    plan = make(tmp_path)
    write_log(plan, "mpd-evaluate-esm2650m_9.err", "no marker needed\n")
    assert failure_tail(plan.campaign_dir, "evaluate") == "no marker needed"


def test_failure_tail_scrubs_account_user_and_email(tmp_path, monkeypatch):
    monkeypatch.setenv("USER", "alice42")
    monkeypatch.setenv("LOGNAME", "alice42")
    plan = make(tmp_path, state="submitted", jobs={"train": "500"})
    scripts = plan.campaign_dir / "scripts" / "train.sbatch"
    scripts.write_text(scripts.read_text() + "#SBATCH --account=PROJ-77\n")
    write_log(plan, "mpd-train-esm2650m_500.err",
              "File /work/alice42/mpd/x.py failed\ncharged to account 123456789012 and PROJ-77\nmail alice42@example.edu\n"
              "node alice42-box is fine but alice420 is another name\n")
    text = failure_tail(plan.campaign_dir, "train")
    assert "alice42@" not in text and "123456789012" not in text and "PROJ-77" not in text
    assert "/work/<user>/mpd/x.py" in text and "<account>" in text and "<email>" in text
    assert "alice420" in text                                                    # only whole names are replaced


def test_scrub_is_conservative_about_what_it_hides(monkeypatch):
    monkeypatch.setenv("USER", "bob")
    monkeypatch.setenv("LOGNAME", "bob")
    assert scrub("job 12345 took 99 s; id 1234567890123456") == "job 12345 took 99 s; id 1234567890123456"    # not account-like
    assert scrub("acct 1234567890123 and 123456789012.") == "acct <account> and <account>."
    assert scrub("bobcat bob bob_x") == "bobcat <user> bob_x" or scrub("bobcat bob") == "bobcat <user>"
    assert scrub("xx PRJ9", extra=("PRJ9", "a")) == "xx <account>"


# ---------------------------------------------------------------- cancel
def submit_two(fake_slurm, plan, tmp_path):
    """Two jobs queued through the fake, as a real submit would, and a matching marker."""
    import subprocess
    jobs = {}
    for name in ("prepare", "teacher"):
        out = subprocess.run(["sbatch", "--parsable", f"--comment={tag(name)}", str(plan.campaign_dir / "scripts" / f"{name}.sbatch")],
                             capture_output=True, text=True).stdout.strip()
        jobs[name] = out
    write_marker(plan.campaign_dir, {"campaign_id": plan.campaign_id, "state": "submitted", "nonce": plan.nonce, "jobs": jobs})
    return jobs


def test_cancel_cancels_the_marker_jobs_and_marks_the_campaign(fake_slurm, tmp_path):
    plan = make(tmp_path)
    jobs = submit_two(fake_slurm, plan, tmp_path)
    fake_slurm.set_state(jobs["prepare"], "COMPLETED")                  # finished jobs are left alone
    said = []
    got = cancel(plan.campaign_dir, echo=said.append)
    assert got == [jobs["teacher"]]
    assert fake_slurm.state_of(jobs["teacher"]) == "CANCELLED" and fake_slurm.state_of(jobs["prepare"]) == "COMPLETED"
    st = campaign_state(plan.campaign_dir)
    assert st.state == "cancelled" and st.jobs == jobs and "mpd cancel" in st.message          # ids are never lost
    assert [c.argv for c in fake_slurm.calls("scancel")] == [[jobs["teacher"]]]
    assert any("cancelled 1 job" in s for s in said)


def test_cancel_finds_a_job_the_marker_does_not_know_by_its_tag(fake_slurm, tmp_path):
    plan = make(tmp_path)
    write_marker(plan.campaign_dir, {"campaign_id": plan.campaign_id, "state": "partial", "nonce": plan.nonce, "jobs": {"prepare": "1001"}})
    fake_slurm.add_job(name="p", comment=tag("prepare"), state="RUNNING")                       # 1001, recorded
    lost = fake_slurm.add_job(name="t", comment=tag("teacher"), state="PENDING")                # accepted but never recorded
    mine_old = fake_slurm.add_job(name="t0", comment=tag("train", nonce="old000000000"), state="PENDING")   # an earlier submission of it
    other_campaign = fake_slurm.add_job(name="x", comment=job_tag("another__id", "n", "train"), state="PENDING")
    theirs = fake_slurm.add_job(name="y", comment=tag("train"), user="someone-else", state="PENDING")
    plain = fake_slurm.add_job(name="z", comment="", state="RUNNING")
    got = cancel(plan.campaign_dir, echo=lambda *_: None)
    assert sorted(got, key=int) == ["1001", lost, mine_old]
    for untouched in (other_campaign, theirs, plain):
        assert fake_slurm.state_of(untouched) in ("PENDING", "RUNNING")
    assert campaign_state(plan.campaign_dir).state == "cancelled"


def test_cancel_works_without_a_marker_by_the_directory_name(fake_slurm, tmp_path):
    plan = make(tmp_path)
    j = fake_slurm.add_job(name="p", comment=tag("prepare"), state="PENDING")
    assert cancel(plan.campaign_dir, echo=lambda *_: None) == [j]
    assert campaign_state(plan.campaign_dir).state == "cancelled"


def test_cancel_reports_a_scancel_failure_keeps_the_ids_and_prints_the_command(fake_slurm, tmp_path):
    plan = make(tmp_path)
    jobs = submit_two(fake_slurm, plan, tmp_path)
    fake_slurm.scancel_fails()
    with pytest.raises(SubmitError) as exc:
        cancel(plan.campaign_dir, echo=lambda *_: None)
    line = f"scancel {jobs['prepare']} {jobs['teacher']}"
    assert line in str(exc.value) and "Unable to contact" in str(exc.value)
    st = campaign_state(plan.campaign_dir)
    assert st.state == "partial" and st.jobs == jobs and line in st.message
    assert fake_slurm.state_of(jobs["prepare"]) == "PENDING"


def test_cancel_with_squeue_down_still_cancels_the_recorded_jobs(fake_slurm, tmp_path):
    plan = make(tmp_path)
    jobs = submit_two(fake_slurm, plan, tmp_path)
    fake_slurm.squeue_fails()
    said = []
    got = cancel(plan.campaign_dir, echo=said.append)
    assert sorted(got) == sorted(jobs.values()) and campaign_state(plan.campaign_dir).state == "cancelled"
    assert any("squeue did not answer" in s for s in said)


def test_cancel_with_squeue_down_and_no_recorded_ids_says_so(fake_slurm, tmp_path):
    plan = make(tmp_path)
    fake_slurm.squeue_fails()
    with pytest.raises(SubmitError, match="no job ids are recorded"):
        cancel(plan.campaign_dir, echo=lambda *_: None)
    assert fake_slurm.calls("scancel") == []


def test_cancel_is_a_no_op_for_finished_or_cancelled_campaigns(fake_slurm, tmp_path):
    plan = make(tmp_path)
    jobs = submit_two(fake_slurm, plan, tmp_path)
    for j in jobs.values():
        fake_slurm.set_state(j, "COMPLETED")
    said = []
    assert cancel(plan.campaign_dir, echo=said.append) == []
    assert campaign_state(plan.campaign_dir).state == "submitted" and "finished" in said[0]          # a finished run is not "cancelled"
    write_marker(plan.campaign_dir, {"campaign_id": plan.campaign_id, "state": "cancelled", "jobs": jobs})
    assert cancel(plan.campaign_dir, echo=said.append) == [] and "already cancelled" in said[1]
    assert fake_slurm.calls("scancel") == []


def test_cancel_of_a_half_submitted_campaign_with_nothing_queued_frees_it(fake_slurm, tmp_path):
    plan = make(tmp_path, state="partial", jobs={"prepare": "4040"})                 # the job is long gone
    assert cancel(plan.campaign_dir, echo=lambda *_: None) == []
    assert campaign_state(plan.campaign_dir).state == "cancelled"


def test_scancel_complaints_about_jobs_that_are_gone_are_not_failures(fake_slurm, tmp_path):
    ok, text = campaign.scancel_ids(["9999"])           # unknown to the fake: "Invalid job id specified", exit 1
    assert ok is True and "Invalid job id" in text
    fake_slurm.scancel_fails()
    ok, text = campaign.scancel_ids(["9999"])
    assert ok is False and "Unable to contact" in text
    assert campaign.scancel_ids([]) == (True, "")


def test_ids_read_from_a_marker_can_never_become_scancel_or_sacct_options(fake_slurm, tmp_path):
    plan = make(tmp_path)
    write_marker(plan.campaign_dir, {"campaign_id": plan.campaign_id, "state": "partial", "jobs": {"prepare": "--user=somebody", "train": "x y"}})
    fake_slurm.squeue_fails()
    with pytest.raises(SubmitError, match="no job ids are recorded"):
        cancel(plan.campaign_dir, echo=lambda *_: None)
    assert fake_slurm.calls("scancel") == [] and job_states(["--all", "1; rm", "12"], sacct="sacct") == {}
    assert campaign.scancel_ids(["-u", "--name=x"]) == (True, "") and fake_slurm.calls("scancel") == []
    assert [c.argv for c in fake_slurm.calls("sacct")] == [["-X", "-n", "-P", "-j", "12", "--format=JobID,JobName,State,Elapsed,ExitCode"]]


def test_cancel_refuses_a_directory_that_is_not_a_campaign(tmp_path):
    with pytest.raises(CampaignError, match="not a campaign directory"):
        cancel(tmp_path / "missing", echo=lambda *_: None)


def test_clients_run_without_a_terminal_and_a_missing_binary_is_a_result_not_an_exception(tmp_path):
    proc = campaign.run_tool([str(tmp_path / "no-such-tool")])
    assert proc.returncode == 127 and "not found" in proc.stderr
    proc = campaign.run_tool(["sh", "-c", "read x; echo got=$x"])       # stdin is /dev/null: returns at once
    assert proc.returncode == 0 and proc.stdout == "got=\n"
    proc = campaign.run_tool(["sh", "-c", "sleep 5"], timeout=0.2)
    assert proc.returncode == 124 and "no answer" in proc.stderr


def test_marker_creation_falls_back_to_an_exclusive_create_without_hard_links(tmp_path, monkeypatch):
    def no_links(*_a, **_k):
        raise PermissionError("hard links are not supported here")

    monkeypatch.setattr(os, "link", no_links)
    root = tmp_path / "c"
    assert create_marker(root, {"campaign_id": "x", "state": "submitting", "jobs": {}}) is True
    assert create_marker(root, {"campaign_id": "x", "state": "other", "jobs": {}}) is False
    assert read_marker(root)["state"] == "submitting" and sorted(p.name for p in root.iterdir()) == ["SUBMITTED.json"]


def test_failure_tail_reads_only_the_end_of_a_huge_log(tmp_path):
    plan = make(tmp_path, state="submitted", jobs={"train": "500"})
    path = write_log(plan, "mpd-train-esm2650m_500.err", "x" * 5_000_000 + "\nlast line\n")
    assert failure_tail(plan.campaign_dir, "train", n=2).splitlines()[-1] == "last line"
    assert path.stat().st_size > 5_000_000


def test_the_campaign_and_submit_modules_need_nothing_but_the_standard_library():
    import subprocess
    import sys
    code = ("import sys; sys.path.insert(0, sys.argv[1]); import mpd.slurm.campaign, mpd.slurm.submit; "
            "print([m for m in ('torch', 'transformers', 'peft', 'numpy', 'scipy', 'pandas', 'yaml') if m in sys.modules])")
    src = str(__import__("pathlib").Path(__file__).parents[1] / "src")
    out = subprocess.run([sys.executable, "-I", "-c", code, src], capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.strip() == "[]", out.stderr
