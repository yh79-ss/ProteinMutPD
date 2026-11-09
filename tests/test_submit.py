"""mpd.slurm.submit against the fake Slurm toolbox: the marker, retries, the accepted-but-reported-failure case, cancel on failure,
continuation of a half-submitted chain, and write_campaign. Nothing here talks to a real cluster."""

import json
import types

import pytest

from fx_slurm import BAD_GRES, BAD_TIME, BALANCE, INVALID_ACCOUNT, clean_path, fake_slurm, stub_plan, stub_render  # noqa: F401
from mpd.slurm import submit as submit_mod
from mpd.slurm.campaign import SubmitError, job_tag, read_marker, write_marker
from mpd.slurm.submit import classify, submit, write_campaign


@pytest.fixture(autouse=True)
def no_real_slurm(monkeypatch):
    """Even a test that forgets `fake_slurm` cannot find the real sbatch/squeue/scancel/sacct on PATH."""
    import os
    monkeypatch.setenv("PATH", clean_path(os.environ.get("PATH", "")))


def prepared(tmp_path, **kw):
    plan = stub_plan(tmp_path, **kw)
    return plan, write_campaign(plan, render=stub_render)


def go(fs, plan, scripts, **kw):
    said = []
    kw.setdefault("sleep", fs.sleep)
    kw.setdefault("clock", fs.clock)
    kw.setdefault("echo", said.append)
    return submit(plan, scripts, **kw), said


def calls_for(fs, stage, tool="sbatch"):
    return [c for c in fs.calls(tool) if c.info.get("stage") == stage and not c.test_only]


def deps_of(call):
    return next((a.split(":", 1)[1] for a in call.argv if a.startswith("--dependency=afterok:")), "")


def marker(plan):
    return read_marker(plan.campaign_dir)


# ---------------------------------------------------------------- the ordinary case
def test_submit_chains_dependencies_tags_every_job_and_records_the_ids(fake_slurm, tmp_path):
    plan, scripts = prepared(tmp_path)
    jobs, said = go(fake_slurm, plan, scripts)
    assert list(jobs) == [s.full_name for s in plan.stages] and len(set(jobs.values())) == 6
    for stage in plan.stages:
        (call,) = calls_for(fake_slurm, stage.full_name)
        assert call.argv[0] == "--parsable" and call.argv[-1] == str(scripts[stage.full_name])
        assert f"--comment=mpd:{plan.campaign_id}:{plan.nonce}:{stage.full_name}" in call.argv
        assert deps_of(call) == ":".join(jobs[d] for d in stage.depends_on)
        assert ("--kill-on-invalid-dep=yes" in call.argv) == bool(stage.depends_on)
    m = marker(plan)
    assert (m["schema"], m["state"], m["campaign_id"], m["nonce"], m["cluster"], m["message"]) == (1, "submitted", plan.campaign_id, plan.nonce, "grace", "")
    assert m["jobs"] == jobs and m["submitted_at"] and m["updated_at"]
    assert fake_slurm.calls("squeue") == [] and fake_slurm.calls("scancel") == []        # a healthy submit never looks at the queue
    text = "\n".join(said)
    for command in ("mpd status " + plan.campaign_id, "mpd logs " + plan.campaign_id, "mpd cancel " + plan.campaign_id,
                    "mpd results " + plan.campaign_id):
        assert command in text


def test_the_marker_exists_before_the_first_job_and_grows_with_every_accepted_one(fake_slurm, tmp_path):
    plan, scripts = prepared(tmp_path, minimal=True)
    log = tmp_path / "snapshots.txt"
    spy = tmp_path / "sbatch_spy"
    spy.write_text(f'#!/bin/sh\ncat "{plan.campaign_dir / "SUBMITTED.json"}" >> "{log}" 2>/dev/null\necho "@@" >> "{log}"\n'
                   f'exec "{fake_slurm.bin / "sbatch"}" "$@"\n')
    spy.chmod(0o755)
    jobs, _ = go(fake_slurm, plan, scripts, sbatch=str(spy))
    snaps = [json.loads(chunk) for chunk in log.read_text().split("@@\n") if chunk.strip()]
    assert [(m["state"], list(m["jobs"])) for m in snaps] == [("submitting", []), ("submitting", ["prepare"]), ("submitting", ["prepare", "train"])]
    assert all(m["nonce"] == plan.nonce for m in snaps) and marker(plan)["state"] == "submitted"
    assert [m["jobs"] for m in snaps][-1] == {k: jobs[k] for k in ("prepare", "train")}


def test_progress_lines_are_flushed_so_a_log_file_shows_them_during_a_wait(fake_slurm, tmp_path, monkeypatch):
    class Recorder:
        def __init__(self):
            self.text, self.flushes = [], 0

        def write(self, s):
            self.text.append(s)

        def flush(self):
            self.flushes += 1

    out = Recorder()
    monkeypatch.setattr("sys.stdout", out)
    plan, scripts = prepared(tmp_path, minimal=True)
    submit(plan, scripts, sleep=fake_slurm.sleep, clock=fake_slurm.clock)          # echo=print, the default
    prints = len(out.text) // 2                                                     # print() writes the text, then the newline
    assert prints >= 3 and out.flushes >= prints                                    # every printed line was flushed


def test_the_job_submit_banner_is_shown_with_one_note_that_it_counts_cpus_only(fake_slurm, tmp_path):
    fake_slurm.configure(sbatch={"job_submit_lines": True})
    plan, scripts = prepared(tmp_path, minimal=True)
    _, said = go(fake_slurm, plan, scripts)
    assert sum("Requested SUs:" in s for s in said) == 3
    assert sum("counts CPUs only" in s and s.startswith("note:") for s in said) == 1


def test_old_keyword_names_and_a_plan_without_nonce_still_work(fake_slurm, tmp_path):
    new = stub_plan(tmp_path, minimal=True)
    old = types.SimpleNamespace(spec=new.spec, options=new.options, campaign_dir=new.campaign_dir, stages=new.stages,
                                resolved=new.resolved, warnings=new.warnings, total_su=new.total_su)       # an older Plan: no nonce, no campaign_id
    untagged = lambda plan, stage: stub_render(types.SimpleNamespace(campaign_id="x", nonce="y", campaign_dir=plan.campaign_dir), stage)  # noqa: E731
    scripts = write_campaign(old, render=untagged)                  # the old renderer wrote no --comment line either
    said = []
    jobs = submit(old, scripts, sbatch=str(fake_slurm.bin / "sbatch"), echo=said.append)         # the pre-existing call shape
    assert list(jobs) == ["prepare", "train", "evaluate"]
    nonce = json.loads((old.campaign_dir / "plan.json").read_text())["nonce"]
    assert nonce == old.nonce and marker(old)["nonce"] == nonce
    assert f"--comment=mpd:{new.campaign_id}:{nonce}:train" in calls_for(fake_slurm, "train")[0].argv


def test_squeue_is_asked_for_the_user_and_never_with_me(fake_slurm, tmp_path):
    fake_slurm.fail_sbatch(1, stage="train")
    plan, scripts = prepared(tmp_path, minimal=True)
    go(fake_slurm, plan, scripts)
    queue_calls = fake_slurm.calls("squeue")
    assert queue_calls
    for call in queue_calls:
        assert call.argv == ["-u", fake_slurm.user, "-h", "-t", "all", "-o", "%F|%j|%k|%T|%M|%r"]


# ---------------------------------------------------------------- controller errors
def test_controller_errors_are_retried_after_the_queue_was_checked(fake_slurm, tmp_path):
    fake_slurm.fail_sbatch(2, stage="teacher")
    plan, scripts = prepared(tmp_path)
    jobs, said = go(fake_slurm, plan, scripts)
    assert len(calls_for(fake_slurm, "teacher")) == 3 and fake_slurm.sleeps == [60, 90]
    assert len([j for j in fake_slurm.jobs().values() if j["stage"] == "teacher"]) == 1          # failures queued nothing
    assert deps_of(calls_for(fake_slurm, "train")[0]) == jobs["teacher"]
    assert marker(plan)["state"] == "submitted"
    assert any("waiting 60 s, then checking the queue" in s for s in said) and any("no job of this submission is queued" in s for s in said)


def test_a_job_accepted_while_sbatch_reported_an_error_is_adopted_not_duplicated(fake_slurm, tmp_path):
    fake_slurm.accept_but_fail("teacher")                 # the 2026-07-26 incident
    plan, scripts = prepared(tmp_path)
    jobs, said = go(fake_slurm, plan, scripts)
    teachers = [j for j in fake_slurm.jobs().values() if j["stage"] == "teacher"]
    assert len(teachers) == 1 and str(teachers[0]["id"]) == jobs["teacher"] and len(calls_for(fake_slurm, "teacher")) == 1
    assert deps_of(calls_for(fake_slurm, "train")[0]) == jobs["teacher"]            # the chain hangs on the adopted job
    assert fake_slurm.calls("scancel") == [] and marker(plan)["jobs"] == jobs
    assert any("queued after all" in s for s in said)


def test_squeue_down_during_the_wait_means_waiting_longer_never_submitting_blind(fake_slurm, tmp_path):
    fake_slurm.accept_but_fail("teacher")
    fake_slurm.squeue_fails({"lte": 2})
    plan, scripts = prepared(tmp_path)
    jobs, said = go(fake_slurm, plan, scripts)
    assert fake_slurm.sleeps == [60, 90, 135]
    assert len(calls_for(fake_slurm, "teacher")) == 1                                   # not one blind resubmission
    assert len([j for j in fake_slurm.jobs().values() if j["stage"] == "teacher"]) == 1
    assert sum("queue cannot be read yet" in s for s in said) == 2


def test_an_outage_that_hides_an_accepted_job_until_it_ends_is_waited_out(fake_slurm, tmp_path):
    fake_slurm.outage(until_s=200, accept=True)
    plan, scripts = prepared(tmp_path, minimal=True)
    jobs, _ = go(fake_slurm, plan, scripts)
    assert len([j for j in fake_slurm.jobs().values() if j["stage"] == "prepare"]) == 1
    assert fake_slurm.sleeps == [60, 90, 135] and len(calls_for(fake_slurm, "prepare")) == 1
    assert marker(plan)["state"] == "submitted" and len(jobs) == 3


def test_a_job_of_another_submission_or_another_user_is_never_adopted(fake_slurm, tmp_path):
    plan, scripts = prepared(tmp_path, minimal=True)
    cid = plan.campaign_id
    old = fake_slurm.add_job(name="old", comment=job_tag(cid, "ffffffffffff", "train"), state="RUNNING")      # an earlier submission
    theirs = fake_slurm.add_job(name="theirs", comment=job_tag(cid, plan.nonce, "train"), user="someone-else", state="RUNNING")
    other = fake_slurm.add_job(name="other", comment=job_tag("another__id", plan.nonce, "train"), state="RUNNING")
    fake_slurm.fail_sbatch(1, stage="train")
    jobs, _ = go(fake_slurm, plan, scripts)
    assert jobs["train"] not in (old, theirs, other) and len(calls_for(fake_slurm, "train")) == 2
    assert deps_of(calls_for(fake_slurm, "evaluate")[0]) == jobs["train"]
    assert all(fake_slurm.state_of(j) == "RUNNING" for j in (old, theirs, other)) and fake_slurm.calls("scancel") == []


def test_a_copy_that_became_visible_late_is_cancelled_keeping_the_first(fake_slurm, tmp_path):
    fake_slurm.accept_but_fail("teacher", hide_squeue_calls=1)       # invisible to the first look, so a second copy is queued
    plan, scripts = prepared(tmp_path)
    jobs, said = go(fake_slurm, plan, scripts)
    copies = sorted(int(j["id"]) for j in fake_slurm.jobs().values() if j["stage"] == "teacher")
    assert len(copies) == 2 and jobs["teacher"] == str(copies[0])                        # the first one is kept
    assert fake_slurm.state_of(copies[1]) == "CANCELLED" and fake_slurm.state_of(copies[0]) == "PENDING"
    assert deps_of(calls_for(fake_slurm, "train")[0]) == jobs["teacher"]                 # nothing waits on the cancelled copy
    assert any("queued twice" in s and f"kept job {copies[0]}" in s for s in said)


def test_a_copy_that_only_shows_up_after_the_chain_is_queued_is_cancelled_by_the_final_sweep(fake_slurm, tmp_path):
    fake_slurm.accept_but_fail("teacher", hide_squeue_calls=2)
    plan, scripts = prepared(tmp_path)
    jobs, said = go(fake_slurm, plan, scripts)
    first, second = sorted(int(j["id"]) for j in fake_slurm.jobs().values() if j["stage"] == "teacher")
    assert jobs["teacher"] == str(second) and fake_slurm.state_of(first) == "CANCELLED"           # never a job something waits for
    assert fake_slurm.state_of(second) == "PENDING" and deps_of(calls_for(fake_slurm, "train")[0]) == str(second)


def test_a_running_array_has_many_squeue_rows_but_is_one_job(fake_slurm, tmp_path):
    plan, scripts = prepared(tmp_path, minimal=True, units=("a", "b", "c"))
    fake_slurm.accept_but_fail("train")

    def sleep(s):
        fake_slurm.sleep(s)
        fake_slurm.set_state("train", "RUNNING")             # now squeue prints several rows with the same id

    jobs, said = go(fake_slurm, plan, scripts, sleep=sleep)
    assert len(fake_slurm.jobs()) == 3 and fake_slurm.calls("scancel") == [] and not any("queued twice" in s for s in said)
    assert jobs["train"] == str(fake_slurm.job("train")["id"])


def test_retries_and_the_budget_bound_the_waiting(fake_slurm, tmp_path):
    fake_slurm.fail_sbatch(None, stage="teacher")
    plan, scripts = prepared(tmp_path)
    with pytest.raises(SubmitError, match="could not queue stage teacher: Slurm did not answer after 3 attempt"):
        go(fake_slurm, plan, scripts, retries=2)
    assert len(calls_for(fake_slurm, "teacher")) == 3 and fake_slurm.sleeps == [60, 90]        # first try + 2 retries

    fake_slurm.sleeps.clear()
    plan2, scripts2 = prepared(tmp_path, campaign_id="other__budget")
    t0 = fake_slurm.clock()
    with pytest.raises(SubmitError):
        go(fake_slurm, plan2, scripts2, retries=50, wait_s=60, budget_s=300)
    assert fake_slurm.sleeps == [60, 90, 135, 15] and fake_slurm.clock() - t0 == 300           # the last wait is cut to what is left
    assert len(calls_for(fake_slurm, "teacher")) == 3 + 5


def test_the_budget_starts_at_the_first_error_not_at_the_start(fake_slurm, tmp_path):
    fake_slurm.configure(sbatch={"duration_s": 100})          # every call takes 100 virtual seconds
    fake_slurm.fail_sbatch(1, stage="teacher")
    plan, scripts = prepared(tmp_path)
    go(fake_slurm, plan, scripts, budget_s=150)               # prepare took 100 s, which must not count against the budget
    assert fake_slurm.sleeps == [60]


def test_no_retries_means_one_look_and_no_waiting(fake_slurm, tmp_path):
    fake_slurm.fail_sbatch(None, stage="prepare")
    plan, scripts = prepared(tmp_path, minimal=True)
    with pytest.raises(SubmitError, match="after 1 attempt"):
        go(fake_slurm, plan, scripts, retries=0)
    assert fake_slurm.sleeps == [] and len(fake_slurm.calls("squeue")) >= 1 and marker(plan)["state"] == "failed"


def test_a_persistent_outage_cancels_what_was_queued_and_says_how_to_start_over(fake_slurm, tmp_path):
    fake_slurm.fail_sbatch(None, stage="teacher")
    plan, scripts = prepared(tmp_path)
    with pytest.raises(SubmitError) as exc:
        go(fake_slurm, plan, scripts, retries=2, wait_s=10)
    prepare_id = marker(plan)["jobs"]["prepare"]
    assert fake_slurm.state_of(prepare_id) == "CANCELLED"
    assert f"Cancelled the jobs already queued by this submission: {prepare_id}" in str(exc.value) and "start over" in str(exc.value)
    m = marker(plan)
    assert m["state"] == "failed" and m["jobs"] == {"prepare": prepare_id} and "Cancelled" in m["message"]        # the id is not lost
    assert "Unable to contact slurm controller" in m["message"] or "did not answer" in m["message"]               # and neither is the reason
    assert [c.argv for c in fake_slurm.calls("scancel")] == [[prepare_id]]


def test_an_outage_that_also_blocks_squeue_and_scancel_keeps_every_id_and_prints_the_command(fake_slurm, tmp_path):
    fake_slurm.fail_sbatch(None, stage="teacher")
    fake_slurm.squeue_fails()
    fake_slurm.scancel_fails()
    plan, scripts = prepared(tmp_path)
    with pytest.raises(SubmitError) as exc:
        go(fake_slurm, plan, scripts, retries=3, wait_s=10)
    text = str(exc.value)
    prepare_id = marker(plan)["jobs"]["prepare"]
    assert len(calls_for(fake_slurm, "teacher")) == 1                       # squeue never answered: no blind resubmission
    assert len(fake_slurm.sleeps) == 3
    assert f"Run `scancel {prepare_id}`" in text and "Stage teacher may have been accepted" in text and "mpd cancel" in text
    m = marker(plan)
    assert m["state"] == "partial" and m["jobs"] == {"prepare": prepare_id} and f"scancel {prepare_id}" in m["message"]
    assert fake_slurm.state_of(prepare_id) == "PENDING"                    # still queued, and the marker knows it


def test_a_hidden_copy_is_found_and_cancelled_by_its_tag_when_a_later_stage_is_refused(fake_slurm, tmp_path):
    fake_slurm.accept_but_fail("teacher", hide_squeue_calls=2)         # invisible to the retry check and to the duplicate check
    fake_slurm.reject_stage("train", INVALID_ACCOUNT)
    plan, scripts = prepared(tmp_path)
    with pytest.raises(SubmitError, match="rejected stage train"):
        go(fake_slurm, plan, scripts)
    teachers = [j for j in fake_slurm.jobs().values() if j["stage"] == "teacher"]
    assert len(teachers) == 2 and all(j["state"] == "CANCELLED" for j in teachers)             # the one nobody recorded too
    assert fake_slurm.state_of(marker(plan)["jobs"]["prepare"]) == "CANCELLED" and marker(plan)["state"] == "failed"


# ---------------------------------------------------------------- permanent errors
@pytest.mark.parametrize("message,hint", [
    (INVALID_ACCOUNT, "myproject"), (BALANCE, "balance"), (BAD_GRES, "--gpu"), (BAD_TIME, "--train-time"),
    ("sbatch: error: invalid partition specified: nosuch", "--gpu"), ("sbatch: error: Job violates accounting/QOS policy (job submit limit)", "QOS"),
])
def test_permanent_refusals_stop_at_once_with_sbatchs_words_and_a_hint(fake_slurm, tmp_path, message, hint):
    fake_slurm.reject_stage("train", message)
    plan, scripts = prepared(tmp_path)
    with pytest.raises(SubmitError) as exc:
        go(fake_slurm, plan, scripts)
    text = str(exc.value)
    assert text.startswith("sbatch rejected stage train: ") and message.splitlines()[0] in text and hint in text
    assert message.splitlines()[0].split(": ", 2)[-1][:30] in marker(plan)["message"]       # `mpd status` can say why it failed
    assert fake_slurm.sleeps == [] and len(calls_for(fake_slurm, "train")) == 1
    queued = marker(plan)["jobs"]
    assert set(queued) == {"prepare", "teacher"} and all(fake_slurm.state_of(j) == "CANCELLED" for j in queued.values())
    assert marker(plan)["state"] == "failed" and fake_slurm.calls("squeue") == []       # a clear refusal needs no queue sweep
    assert not any(c.info.get("stage") == "score" for c in fake_slurm.calls("sbatch"))   # later stages are not even tried


def test_an_unrecognised_error_is_not_retried_either(fake_slurm, tmp_path):
    fake_slurm.reject_stage("prepare", "sbatch: error: something odd happened")
    plan, scripts = prepared(tmp_path, minimal=True)
    with pytest.raises(SubmitError, match="did not retry"):
        go(fake_slurm, plan, scripts)
    assert fake_slurm.sleeps == [] and marker(plan)["state"] == "failed" and marker(plan)["jobs"] == {}
    assert fake_slurm.calls("scancel") == []


def test_exit_zero_without_a_job_id_is_treated_as_unknown_and_checked_in_the_queue(fake_slurm, tmp_path):
    plan, scripts = prepared(tmp_path, minimal=True)
    with pytest.raises(SubmitError, match="printed no job id"):
        go(fake_slurm, plan, scripts, sbatch="true", retries=1, wait_s=1)
    assert fake_slurm.sleeps == [1] and marker(plan)["state"] == "failed"


@pytest.mark.parametrize("message,kind", [
    ("sbatch: error: Batch job submission failed: Unable to contact slurm controller (connect failure)", "transient"),
    ("sbatch: error: Batch job submission failed: Socket timed out on send/recv operation", "transient"),
    ("sbatch: error: slurm_persist_conn_open_without_init: failed to open persistent connection", "transient"),
    ("sbatch: error: Batch job submission failed: Zero Bytes were transmitted or received", "transient"),
    ("sbatch: error: Slurm temporarily unable to accept job, sleeping and retrying.", "transient"),
    ("connect: Connection refused", "transient"), ("connect: Connection timed out", "transient"),
    (BALANCE, "permanent"), (INVALID_ACCOUNT, "permanent"), (BAD_GRES, "permanent"), (BAD_TIME, "permanent"),
    ("sbatch: error: Batch job submission failed: Job dependency problem", "permanent"), ("sbatch: unrecognized option '--x'", "permanent"),
    ("sbatch: error: Unable to open file x", "permanent"), ("something never seen", "unknown"), ("", "unknown")])
def test_classification_of_sbatch_messages(message, kind):
    assert classify(message) == kind


def test_a_stage_that_finished_while_we_waited_is_still_adopted(fake_slurm, tmp_path):
    fake_slurm.accept_but_fail("prepare")
    plan, scripts = prepared(tmp_path, minimal=True)

    def sleep(s):
        fake_slurm.sleep(s)
        fake_slurm.set_state("prepare", "COMPLETED")        # it ran and ended while sbatch's error was being waited out

    jobs, _ = go(fake_slurm, plan, scripts, sleep=sleep)
    assert len(calls_for(fake_slurm, "prepare")) == 1 and jobs["prepare"] == str(fake_slurm.job("prepare")["id"])
    assert deps_of(calls_for(fake_slurm, "train")[0]) == jobs["prepare"]


def test_a_stage_that_died_while_we_waited_is_queued_again(fake_slurm, tmp_path):
    fake_slurm.accept_but_fail("train")
    plan, scripts = prepared(tmp_path, minimal=True)

    def sleep(s):
        fake_slurm.sleep(s)
        fake_slurm.set_state("train", "FAILED", exitcode="1:0")

    jobs, said = go(fake_slurm, plan, scripts, sleep=sleep)
    assert len(calls_for(fake_slurm, "train")) == 2 and any("already ended (FAILED); queueing it again" in s for s in said)
    assert fake_slurm.state_of(jobs["train"]) == "PENDING" and deps_of(calls_for(fake_slurm, "evaluate")[0]) == jobs["train"]


def test_a_cluster_that_rewrites_job_comments_is_never_retried_blind(fake_slurm, tmp_path):
    fake_slurm.configure(sbatch={"drop_comment": True})           # tags would prove nothing here
    fake_slurm.accept_but_fail("train")
    plan, scripts = prepared(tmp_path, minimal=True)
    with pytest.raises(SubmitError, match="could not queue stage train") as exc:
        go(fake_slurm, plan, scripts, retries=2, wait_s=5)
    assert len(calls_for(fake_slurm, "train")) == 1                  # the job it could not recognise was not submitted a second time
    assert "does not keep the --comment tag" in str(exc.value) and "squeue -u $USER" in str(exc.value)
    assert marker(plan)["state"] == "partial"                         # it may still be queued, and the marker says so
    assert fake_slurm.state_of(marker(plan)["jobs"]["prepare"]) == "CANCELLED"


def test_force_requeues_only_the_chunk_of_a_stage_that_is_missing(fake_slurm, tmp_path):
    plan, scripts = prepared(tmp_path, units=("a", "b"), shards=2, score_chunks=2)
    first, _ = go(fake_slurm, plan, scripts)
    root = plan.campaign_dir
    for stage in plan.stages:                                  # everything ran; the first score chunk left its shard files, the second did not
        fake_slurm.set_state(first[stage.full_name], "COMPLETED")
    for name, path in (("prepare", "contexts/MANIFEST.json"), ("teacher", "teacher/COMPLETE.json"), ("train/a", "train/a/COMPLETE.json"),
                       ("train/b", "train/b/COMPLETE.json")):
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text("{}")
    for t in plan.stages[4].tasks:                             # chunk 0 of score
        f = root / "score" / t["dataset"] / t["unit"] / f"shard_{t['shard']:03d}_of_{t['num_shards']:03d}.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("{}")
    for t in plan.stages[3].tasks:                             # score-base is complete too
        f = root / "score" / t["dataset"] / t["unit"] / f"shard_{t['shard']:03d}_of_{t['num_shards']:03d}.json"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("{}")
    fake_slurm.set_state(first["score.1"], "FAILED")
    before = len(fake_slurm.calls("sbatch"))
    again = stub_plan(tmp_path, units=("a", "b"), shards=2, score_chunks=2, nonce="secondnonce00")
    jobs, said = go(fake_slurm, again, write_campaign(again, force=True, render=stub_render), force=True)
    assert [c.info["stage"] for c in fake_slurm.calls("sbatch")[before:]] == ["score.1", "evaluate"]
    assert deps_of(calls_for(fake_slurm, "evaluate")[-1]) == jobs["score.1"]          # finished stages are not waited for


# ---------------------------------------------------------------- Ctrl-C and cleanup
def test_ctrl_c_between_two_stages_cancels_what_was_queued_and_re_raises(fake_slurm, tmp_path):
    plan, scripts = prepared(tmp_path)
    said = []

    def echo(line):
        said.append(line)
        if line.startswith("submitted teacher") and not getattr(echo, "fired", False):
            echo.fired = True
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        submit(plan, scripts, echo=echo, sleep=fake_slurm.sleep, clock=fake_slurm.clock)
    jobs = marker(plan)["jobs"]
    assert set(jobs) == {"prepare", "teacher"} and marker(plan)["state"] == "failed"
    assert all(fake_slurm.state_of(j) == "CANCELLED" for j in jobs.values())
    assert any("Cancelled the jobs already queued" in s for s in said)               # the user is told even though they interrupted


def test_ctrl_c_while_waiting_for_the_controller_also_cancels_the_job_nobody_recorded(fake_slurm, tmp_path):
    fake_slurm.accept_but_fail("train")
    plan, scripts = prepared(tmp_path, minimal=True)

    def sleep(_s):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        go(fake_slurm, plan, scripts, sleep=sleep)
    assert marker(plan)["state"] == "failed"
    assert {j["stage"]: j["state"] for j in fake_slurm.jobs().values()} == {"prepare": "CANCELLED", "train": "CANCELLED"}


def test_ctrl_c_when_nothing_can_be_cancelled_leaves_a_partial_marker_with_the_ids(fake_slurm, tmp_path):
    plan, scripts = prepared(tmp_path, minimal=True)
    fake_slurm.scancel_fails()

    def echo(line):
        if line.startswith("submitted train"):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        submit(plan, scripts, echo=echo, sleep=fake_slurm.sleep, clock=fake_slurm.clock)
    m = marker(plan)
    assert m["state"] == "partial" and set(m["jobs"]) == {"prepare", "train"} and "scancel" in m["message"]


def test_an_unexpected_exception_also_cleans_up_and_is_not_swallowed(fake_slurm, tmp_path, monkeypatch):
    plan, scripts = prepared(tmp_path, minimal=True)
    real = submit_mod._Submission._submit_stage

    def boom(self, stage, script):
        if stage.full_name == "evaluate":
            raise RuntimeError("a bug in mpd")
        return real(self, stage, script)

    monkeypatch.setattr(submit_mod._Submission, "_submit_stage", boom)
    said = []
    with pytest.raises(RuntimeError, match="a bug in mpd"):
        go(fake_slurm, plan, scripts, echo=said.append)
    assert marker(plan)["state"] == "failed" and all(fake_slurm.state_of(j) == "CANCELLED" for j in marker(plan)["jobs"].values())
    assert any("Cancelled the jobs already queued" in s for s in said)


# ---------------------------------------------------------------- markers and refusals
@pytest.mark.parametrize("state", ["submitted", "submitting", "partial"])
def test_a_second_submission_is_refused_in_these_states_and_nothing_is_written(fake_slurm, tmp_path, state):
    plan, scripts = prepared(tmp_path, minimal=True)
    write_marker(plan.campaign_dir, {"campaign_id": plan.campaign_id, "state": state, "submitted_at": "2026-11-08T10:00:00-0600",
                                     "nonce": "oldnonce0000", "jobs": {"prepare": "77"}, "message": "stage train: no answer"})
    before = marker(plan)
    other = stub_plan(tmp_path, minimal=True, nonce="newnonce0000")
    with pytest.raises(SubmitError) as exc:
        submit(other, scripts, echo=lambda *_: None, sleep=fake_slurm.sleep)
    assert plan.campaign_id in str(exc.value) and "prepare=77" in str(exc.value) and "mpd status" in str(exc.value) and "--force" in str(exc.value)
    assert fake_slurm.calls() == [] and marker(plan) == before
    with pytest.raises(SubmitError, match="Nothing was written"):
        write_campaign(other, render=stub_render)


def test_the_refusal_for_a_submitted_campaign_says_already_submitted(fake_slurm, tmp_path):
    plan, scripts = prepared(tmp_path, minimal=True)
    go(fake_slurm, plan, scripts)
    with pytest.raises(SubmitError, match="already submitted"):
        go(fake_slurm, plan, scripts)
    assert len(fake_slurm.submitted()) == 3


def test_an_unreadable_marker_blocks_because_something_may_be_queued(fake_slurm, tmp_path):
    plan, scripts = prepared(tmp_path, minimal=True)
    (plan.campaign_dir / "SUBMITTED.json").write_text("{broken")
    with pytest.raises(SubmitError, match="cannot be read"):
        go(fake_slurm, plan, scripts)
    with pytest.raises(SubmitError, match="cannot be read"):
        write_campaign(plan, render=stub_render)
    assert fake_slurm.calls() == []


@pytest.mark.parametrize("state", ["failed", "cancelled"])
def test_a_failed_or_cancelled_campaign_is_submitted_again_with_a_message(fake_slurm, tmp_path, state):
    plan, scripts = prepared(tmp_path, minimal=True)
    write_marker(plan.campaign_dir, {"campaign_id": plan.campaign_id, "state": state, "jobs": {"prepare": "5"}, "message": "an old problem"})
    fresh = stub_plan(tmp_path, minimal=True, nonce="freshnonce000")
    scripts = write_campaign(fresh, render=stub_render)                      # allowed again
    jobs, said = go(fake_slurm, fresh, scripts)
    assert any(f"the earlier submission of this campaign {state} (an old problem); submitting it again" in s for s in said)
    assert marker(fresh)["state"] == "submitted" and marker(fresh)["nonce"] == "freshnonce000" and jobs["prepare"] != "5"


def test_two_submissions_racing_for_the_marker_cannot_both_win(fake_slurm, tmp_path, monkeypatch):
    plan, scripts = prepared(tmp_path, minimal=True)

    def lose(root, data):                       # the other terminal created the marker between our read and our claim
        write_marker(root, {**data, "jobs": {"prepare": "9"}})
        return False

    monkeypatch.setattr(submit_mod, "create_marker", lose)
    with pytest.raises(SubmitError, match="did not finish"):
        go(fake_slurm, plan, scripts)
    assert fake_slurm.calls("sbatch") == []


def test_a_missing_sbatch_is_a_clear_error_and_leaves_no_marker(fake_slurm, tmp_path):
    fake_slurm.remove("sbatch")
    plan, scripts = prepared(tmp_path, minimal=True)
    with pytest.raises(SubmitError, match="sbatch was not found.*--dry-run"):
        go(fake_slurm, plan, scripts)
    assert marker(plan) is None


def test_a_missing_script_is_reported_before_anything_is_queued(fake_slurm, tmp_path):
    plan, scripts = prepared(tmp_path, minimal=True)
    scripts["train"].unlink()
    with pytest.raises(SubmitError, match="script of stage train is missing"):
        go(fake_slurm, plan, scripts)
    assert marker(plan) is None and fake_slurm.calls() == []


def test_scripts_default_to_the_files_in_the_campaign_directory(fake_slurm, tmp_path):
    plan, _ = prepared(tmp_path, minimal=True)
    jobs, _ = go(fake_slurm, plan, None)
    assert list(jobs) == ["prepare", "train", "evaluate"]


# ---------------------------------------------------------------- --force: continue a chain
def half_submitted(fs, tmp_path, **kw):
    """prepare is queued; teacher keeps failing and neither squeue nor scancel answers: a `partial` campaign."""
    fs.fail_sbatch(None, stage="teacher")
    fs.squeue_fails()
    fs.scancel_fails()
    plan, scripts = prepared(tmp_path, **kw)
    with pytest.raises(SubmitError):
        go(fs, plan, scripts, retries=1, wait_s=1)
    assert marker(plan)["state"] == "partial"
    fs.configure(sbatch={}, squeue={}, scancel={})              # the controller is back
    return plan


def test_force_continues_a_partial_chain_keeping_the_queued_stage(fake_slurm, tmp_path):
    plan = half_submitted(fake_slurm, tmp_path)
    first_marker = marker(plan)
    prepare_id = first_marker["jobs"]["prepare"]
    again = stub_plan(tmp_path, nonce="secondnonce00")
    with pytest.raises(SubmitError, match="partly submitted"):
        go(fake_slurm, again, None)                              # without --force nothing happens
    sbatch_calls_before = len(fake_slurm.calls("sbatch"))
    scripts = write_campaign(again, force=True, render=stub_render)
    jobs, said = go(fake_slurm, again, scripts, force=True)
    assert jobs["prepare"] == prepare_id and len(calls_for(fake_slurm, "prepare")) == 1            # not submitted twice
    assert len(fake_slurm.calls("sbatch")) - sbatch_calls_before == 5                                # teacher .. evaluate
    assert deps_of(calls_for(fake_slurm, "teacher")[-1]) == prepare_id
    assert deps_of(calls_for(fake_slurm, "score-base")[-1]) == prepare_id
    m = marker(again)
    assert m["state"] == "submitted" and m["nonce"] == "secondnonce00" and m["jobs"] == jobs and list(jobs) == [s.full_name for s in again.stages]
    assert any(s.startswith("kept      prepare") and prepare_id in s for s in said)
    assert m["submitted_at"] == first_marker["submitted_at"]                              # it is the same submission, continued


def test_force_does_not_queue_a_stage_whose_output_exists(fake_slurm, tmp_path):
    plan, scripts = prepared(tmp_path, minimal=True)
    write_marker(plan.campaign_dir, {"campaign_id": plan.campaign_id, "state": "partial", "jobs": {"prepare": "41"}, "message": "stage train failed"})
    (plan.campaign_dir / "contexts").mkdir()
    (plan.campaign_dir / "contexts" / "MANIFEST.json").write_text("{}")           # prepare finished, its job left the queue long ago
    jobs, said = go(fake_slurm, plan, scripts, force=True)
    assert list(jobs) == ["prepare", "train", "evaluate"] and jobs["prepare"] == "41"          # its old id stays on record for `mpd status`
    assert len(calls_for(fake_slurm, "prepare")) == 0 and deps_of(calls_for(fake_slurm, "train")[0]) == ""         # but nothing waits for it
    assert any(s.startswith("finished  prepare") for s in said)
    assert marker(plan)["state"] == "submitted" and marker(plan)["jobs"] == jobs


def test_force_requeues_a_dead_stage_and_everything_waiting_on_it(fake_slurm, tmp_path):
    plan, scripts = prepared(tmp_path, units=("all",), shards=1)
    first, _ = go(fake_slurm, plan, scripts)
    root = plan.campaign_dir
    for name, marker_file in (("prepare", "contexts/MANIFEST.json"), ("teacher", "teacher/COMPLETE.json")):
        (root / marker_file).parent.mkdir(parents=True, exist_ok=True)
        (root / marker_file).write_text("{}")
        fake_slurm.set_state(first[name], "COMPLETED")
    fake_slurm.set_state(first["train"], "TIMEOUT")
    fake_slurm.set_state(first["score-base"], "RUNNING")
    before = len(fake_slurm.calls("sbatch"))
    again = stub_plan(tmp_path, units=("all",), shards=1, nonce="secondnonce00")
    jobs, said = go(fake_slurm, again, write_campaign(again, force=True, render=stub_render), force=True)
    new = [c.info["stage"] for c in fake_slurm.calls("sbatch")[before:]]
    assert new == ["train", "score", "evaluate"]                                       # score-base keeps running
    assert jobs["score-base"] == first["score-base"] and jobs["train"] != first["train"]
    assert deps_of(calls_for(fake_slurm, "train")[-1]) == "" and deps_of(calls_for(fake_slurm, "score")[-1]) == jobs["train"]
    assert set(deps_of(calls_for(fake_slurm, "evaluate")[-1]).split(":")) == {jobs["score-base"], jobs["score"]}
    stale = [first["score"], first["evaluate"]]                                         # they waited for the dead train job
    assert all(fake_slurm.state_of(j) == "CANCELLED" for j in stale)
    assert marker(again)["replaced"] == {"train": [first["train"]], "score": [first["score"]], "evaluate": [first["evaluate"]]}


def test_the_ids_of_every_replaced_job_stay_on_record_across_several_repairs(fake_slurm, tmp_path):
    plan, scripts = prepared(tmp_path, minimal=True)
    first, _ = go(fake_slurm, plan, scripts)
    seen = [first["train"]]
    for n in range(2):
        fake_slurm.set_state(seen[-1], "TIMEOUT")
        again = stub_plan(tmp_path, minimal=True, nonce=f"repair{n}nonce0")
        jobs, _ = go(fake_slurm, again, write_campaign(again, force=True, render=stub_render), force=True)
        seen.append(jobs["train"])
        assert marker(again)["replaced"]["train"] == seen[:-1]
    assert len(set(seen)) == 3 and marker(plan)["jobs"]["train"] == seen[-1]


def test_force_on_a_chain_that_is_all_queued_changes_nothing(fake_slurm, tmp_path):
    plan, scripts = prepared(tmp_path, minimal=True)
    first, _ = go(fake_slurm, plan, scripts)
    before = len(fake_slurm.calls("sbatch"))
    jobs, said = go(fake_slurm, plan, write_campaign(plan, force=True, render=stub_render), force=True)
    assert jobs == first and len(fake_slurm.calls("sbatch")) == before and fake_slurm.calls("scancel") == []
    assert any("nothing to submit" in s for s in said) and marker(plan)["state"] == "submitted"


def test_force_needs_a_readable_queue_and_touches_nothing_otherwise(fake_slurm, tmp_path):
    plan, scripts = prepared(tmp_path, minimal=True)
    go(fake_slurm, plan, scripts)
    before, calls = marker(plan), len(fake_slurm.calls("sbatch"))
    fake_slurm.squeue_fails()
    with pytest.raises(SubmitError, match="squeue did not answer"):
        go(fake_slurm, plan, None, force=True)
    assert marker(plan) == before and len(fake_slurm.calls("sbatch")) == calls


def test_force_continues_a_submission_that_was_killed_half_way(fake_slurm, tmp_path):
    plan, scripts = prepared(tmp_path, minimal=True)
    prepare = fake_slurm.add_job(name="p", comment=job_tag(plan.campaign_id, "deadbeef0000", "prepare"), state="RUNNING")
    write_marker(plan.campaign_dir, {"campaign_id": plan.campaign_id, "state": "submitting", "nonce": "deadbeef0000", "jobs": {"prepare": prepare}})
    with pytest.raises(SubmitError, match="did not finish"):
        go(fake_slurm, plan, scripts)
    jobs, _ = go(fake_slurm, plan, scripts, force=True)
    assert jobs["prepare"] == prepare and deps_of(calls_for(fake_slurm, "train")[0]) == prepare


def test_force_adopts_a_job_that_was_accepted_but_never_recorded(fake_slurm, tmp_path):
    plan, scripts = prepared(tmp_path, minimal=True)
    write_marker(plan.campaign_dir, {"campaign_id": plan.campaign_id, "state": "partial", "nonce": "deadbeef0000", "jobs": {}})
    lost = fake_slurm.add_job(name="p", comment=job_tag(plan.campaign_id, "deadbeef0000", "prepare"), state="PENDING")
    jobs, _ = go(fake_slurm, plan, scripts, force=True)
    assert jobs["prepare"] == lost and calls_for(fake_slurm, "prepare") == []


def test_a_failure_while_continuing_leaves_the_earlier_jobs_alone(fake_slurm, tmp_path):
    plan, scripts = prepared(tmp_path, minimal=True)
    kept = fake_slurm.add_job(name="p", comment=job_tag(plan.campaign_id, "deadbeef0000", "prepare"), state="RUNNING")
    write_marker(plan.campaign_dir, {"campaign_id": plan.campaign_id, "state": "partial", "nonce": "deadbeef0000", "jobs": {"prepare": kept}})
    fake_slurm.reject_stage("evaluate", BALANCE)
    with pytest.raises(SubmitError, match="rejected stage evaluate"):
        go(fake_slurm, plan, scripts, force=True)
    train = fake_slurm.job("train")
    assert train["state"] == "CANCELLED" and fake_slurm.state_of(kept) == "RUNNING"           # only what this call queued is cancelled
    m = marker(plan)
    assert m["state"] == "partial" and kept in m["jobs"].values() and "from the earlier submission are still queued" in m["message"]


# ---------------------------------------------------------------- write_campaign
def snapshot(root):
    return {str(p.relative_to(root)): p.read_bytes() for p in sorted(root.rglob("*")) if p.is_file()}


def test_write_campaign_writes_the_recipe_scripts_and_a_plan_with_the_nonce(tmp_path):
    plan = stub_plan(tmp_path, units=("a", "b"), score_chunks=2)
    scripts = write_campaign(plan, render=stub_render)
    root = plan.campaign_dir
    assert list(scripts) == [s.full_name for s in plan.stages] and all(p.parent == root / "scripts" and p.is_file() for p in scripts.values())
    assert json.loads((root / "spec.json").read_text()) == {"campaign": plan.campaign_id}
    assert json.loads((root / "resolved.json").read_text())["score_shards"] == {"proteingym": 2}
    tasks = json.loads((root / "tasks" / "score.json").read_text())
    assert len(tasks) == sum(len(s.tasks) for s in plan.stages if s.name == "score")        # chunks of a stage share one task list
    summary = json.loads((root / "plan.json").read_text())
    assert summary["nonce"] == plan.nonce and summary["campaign_id"] == plan.campaign_id and summary["cluster"] == "grace"
    assert [s["name"] for s in summary["stages"]][:2] == ["prepare", "teacher"] and "tasks" not in summary["stages"][0] and summary["stages"][2]["n_tasks"] == 2
    assert summary["estimated_su_upper_bound"] == plan.total_su and summary["warnings"] == ["stub warning"]
    assert (root / "logs").is_dir()                                     # Slurm does not create the --output directory
    assert not [p for p in root.rglob("*") if p.name.endswith(".tmp")]
    assert all(p.read_bytes().endswith(b"\n") and b"\r" not in p.read_bytes() for p in scripts.values())
    assert f"#SBATCH --comment=mpd:{plan.campaign_id}:{plan.nonce}:" in scripts["train"].read_text()


@pytest.mark.parametrize("state", ["submitted", "submitting", "partial"])
def test_write_campaign_refusal_leaves_every_file_byte_identical(tmp_path, state):
    plan = stub_plan(tmp_path)
    write_campaign(plan, render=stub_render)
    write_marker(plan.campaign_dir, {"campaign_id": plan.campaign_id, "state": state, "jobs": {"prepare": "1"}})
    before = snapshot(plan.campaign_dir)
    changed = stub_plan(tmp_path, nonce="anothernonce0", units=("x", "y"), shards=5)         # a different recipe AND nonce
    with pytest.raises(SubmitError, match="Nothing was written"):
        write_campaign(changed, render=stub_render)
    assert snapshot(plan.campaign_dir) == before


def test_write_campaign_with_force_rewrites_only_scripts_and_only_for_the_same_recipe(tmp_path):
    plan = stub_plan(tmp_path)
    write_campaign(plan, render=stub_render)
    write_marker(plan.campaign_dir, {"campaign_id": plan.campaign_id, "state": "submitted", "jobs": {}})
    before = snapshot(plan.campaign_dir)
    again = stub_plan(tmp_path, nonce="anothernonce0")
    write_campaign(again, force=True, render=stub_render)
    after = snapshot(plan.campaign_dir)
    changed = sorted(k for k in after if after[k] != before.get(k))
    assert all(k.startswith("scripts/") or k == "plan.json" for k in changed) and "plan.json" in changed
    assert all(k in after for k in before) and "anothernonce0" in (plan.campaign_dir / "scripts" / "train.sbatch").read_text()
    different = stub_plan(tmp_path, units=("x", "y"))
    with pytest.raises(SubmitError, match="different recipe .*tasks/train.json"):
        write_campaign(different, force=True, render=stub_render)
    assert snapshot(plan.campaign_dir) == after


@pytest.mark.parametrize("state", ["failed", "cancelled"])
def test_write_campaign_is_allowed_again_after_a_failure_or_a_cancel(tmp_path, state):
    plan = stub_plan(tmp_path)
    write_campaign(plan, render=stub_render)
    write_marker(plan.campaign_dir, {"campaign_id": plan.campaign_id, "state": state, "jobs": {}})
    write_campaign(stub_plan(tmp_path, nonce="n2n2n2n2n2n2"), render=stub_render)
    assert "n2n2n2n2n2n2" in json.loads((plan.campaign_dir / "plan.json").read_text())["nonce"]


# ---------------------------------------------------------------- a full disk
def full_disk(monkeypatch, after=0):
    """write_marker works `after` times, then fails like a full disk or an exceeded quota."""
    real, calls = submit_mod.write_marker, []

    def write(root, data):
        calls.append(data["state"])
        if len(calls) > after:
            raise OSError(28, "No space left on device")
        return real(root, data)

    monkeypatch.setattr(submit_mod, "write_marker", write)
    return calls


def test_a_full_disk_while_recording_a_job_cancels_it_and_prints_the_ids_instead_of_a_traceback(fake_slurm, tmp_path, monkeypatch):
    plan, scripts = prepared(tmp_path, minimal=True)
    full_disk(monkeypatch)                                    # even the cleanup cannot write the marker
    said = []
    with pytest.raises(SubmitError, match="cannot write .*SUBMITTED.json .*No space left on device") as exc:
        go(fake_slurm, plan, scripts, echo=said.append)
    first = str(fake_slurm.job("prepare")["id"])
    assert fake_slurm.state_of(first) == "CANCELLED"          # the job that could not be recorded is not left running
    assert "The marker could not be written" in str(exc.value) and f"prepare={first}" in str(exc.value)
    assert len(fake_slurm.submitted()) == 1


def test_a_full_disk_when_the_marker_is_first_created_queues_nothing(fake_slurm, tmp_path, monkeypatch):
    plan, scripts = prepared(tmp_path, minimal=True)
    monkeypatch.setattr(submit_mod, "create_marker", lambda *a, **k: (_ for _ in ()).throw(OSError(122, "Disk quota exceeded")))
    with pytest.raises(SubmitError, match="Disk quota exceeded.*quota full"):
        go(fake_slurm, plan, scripts)
    assert fake_slurm.calls("sbatch") == []


def test_a_full_disk_while_writing_the_campaign_is_a_submit_error(tmp_path, monkeypatch):
    plan = stub_plan(tmp_path)

    def fail(path, text):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(submit_mod, "write_text_atomic", fail)
    with pytest.raises(SubmitError, match="cannot write the campaign directory .*No space left"):
        write_campaign(plan, render=stub_render)


def test_a_full_disk_after_the_last_job_is_queued_leaves_the_chain_alone(fake_slurm, tmp_path, monkeypatch):
    plan, scripts = prepared(tmp_path, minimal=True)
    calls = full_disk(monkeypatch, after=3)                    # the three "submitting" records succeed, the final "submitted" does not
    with pytest.raises(SubmitError, match="No space left on device") as exc:
        go(fake_slurm, plan, scripts)
    assert calls == ["submitting"] * 3 + ["submitted"]
    assert "All jobs are queued, nothing was cancelled: prepare=1001, train=1002, evaluate=1003" in str(exc.value)
    assert all(fake_slurm.state_of(j) == "PENDING" for j in ("1001", "1002", "1003")) and fake_slurm.calls("scancel") == []
    assert marker(plan)["state"] == "submitting" and len(marker(plan)["jobs"]) == 3          # every id was recorded as it was accepted
