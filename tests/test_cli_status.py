"""`mpd status` as a student reads it: a table with one stable line per stage, then what went wrong in plain words with the exact command
to continue, then why the waiting stages wait. The campaigns are made by the real CLI against the fake Slurm; the states Slurm would
report are set by hand. Nothing here is a claim about Grace.
"""

import json

import pytest
from fx_cli import FAKE_ACCOUNT, FAKE_USER, cli_env, ready_home, recipe, run_cli, write_logs  # noqa: F401
from fx_slurm import fake_slurm  # noqa: F401

from mpd import cli
from mpd.slurm import campaign
from mpd.slurm.submit import submit as real_submit

pytestmark = pytest.mark.usefixtures("cli_env")

STAGES = ["prepare", "teacher", "train", "score-base", "score", "evaluate"]
HEADER = ["stage", "state", "elapsed", "reason", "job", "exit", "tasks", "note"]


@pytest.fixture
def world(cli_env, monkeypatch, fake_slurm):
    ready_home(cli_env, monkeypatch)
    monkeypatch.setenv("MPD_HOME", str(cli_env.home))                  # commands are then named by the campaign id, as in the docs
    monkeypatch.setattr(cli, "submit", lambda plan, scripts, **kw: real_submit(plan, scripts, sleep=fake_slurm.sleep, clock=fake_slurm.clock, **kw))
    return cli_env, fake_slurm


def submitted(capsys, env, *extra):
    rc, out, err = run_cli(capsys, "submit", *recipe(env.home, "--account", FAKE_ACCOUNT, *extra), "--yes")
    assert rc == 0, (out, err)
    (root,) = (env.home / "runs").iterdir()
    return root, json.loads((root / "SUBMITTED.json").read_text())["jobs"]


def status(capsys, root):
    rc, out, err = run_cli(capsys, "status", root.name)
    assert rc == 0, err
    return out


def table(out: str) -> dict[str, list[str]]:
    lines = out.splitlines()
    first = next(i for i, ln in enumerate(lines) if ln.split() == HEADER)
    rows = {}
    for ln in lines[first + 1:]:
        if not ln.strip():
            break
        rows[ln.split()[0]] = ln.split()
    return rows


def section(out: str, title: str) -> str:
    """The lines under `title:` up to the next blank line."""
    lines = out.splitlines()
    start = lines.index(f"{title}:")
    block = []
    for ln in lines[start + 1:]:
        if not ln.strip():
            break
        block.append(ln)
    return "\n".join(block)


def log(root, job, text, *, stage="train", err=True):
    write_logs(root, {f"mpd-{stage}-esm2650m_{job}.{'err' if err else 'out'}": text})


# ---------------------------------------------------------------- the table
def test_the_first_four_fields_of_every_stage_line_are_stage_state_elapsed_and_reason(capsys, world):
    env, fs = world
    root, jobs = submitted(capsys, env)
    fs.set_state("prepare", "COMPLETED", elapsed="00:03:11")
    fs.set_state("teacher", "RUNNING", elapsed="00:10:00")
    fs.set_state("train", "PENDING", reason="user env retrieval failed requeued held")
    out = status(capsys, root)
    rows = table(out)
    assert list(rows) == STAGES
    assert rows["prepare"][:5] == ["prepare", "COMPLETED", "00:03:11", "-", jobs["prepare"]]
    assert rows["teacher"][:4] == ["teacher", "RUNNING", "00:10:00", "-"]
    assert rows["train"][:4] == ["train", "PENDING", "00:00:00", "user_env_retrieval_failed_requeued_held"]        # one word, whatever squeue says
    assert rows["score"][3] == "Dependency" and rows["evaluate"][3] == "Dependency"
    for fields in rows.values():
        assert len(fields) >= 7 and fields[5] in ("-", "0:0") or ":" in fields[5]                                   # exit code
    assert out.splitlines()[0].startswith("campaign   ") and "state      submitted" in out


def test_the_tasks_column_counts_finished_tasks_and_an_array_adds_a_note(capsys, world):
    env, fs = world
    root, _ = submitted(capsys, env)
    for i in range(3):
        (root / "score" / "proteingym" / "base").mkdir(parents=True, exist_ok=True)
        (root / "score" / "proteingym" / "base" / f"shard_{i:03d}_of_004.json").write_text("{}")
    fs.set_state("score-base", "RUNNING", tasks=["COMPLETED", "COMPLETED", "COMPLETED", "RUNNING"])
    row = table(status(capsys, root))["score-base"]
    assert row[1] == "RUNNING" and row[6] == "3/4" and " ".join(row[7:]) == "3 COMPLETED, 1 RUNNING"


def test_without_sacct_and_squeue_the_table_has_dashes_and_no_advice(capsys, world):
    env, fs = world
    root, _ = submitted(capsys, env)
    fs.remove("sacct")
    fs.remove("squeue")
    out = status(capsys, root)
    assert "no job states" in out and all(f[1] == "-" for f in table(out).values())
    assert "problems:" not in out and "waiting:" not in out


# ---------------------------------------------------------------- a stage that ran out of time
def test_a_timeout_says_how_long_it_ran_what_it_asked_for_and_the_command_to_continue(capsys, world):
    env, fs = world
    root, jobs = submitted(capsys, env, "--train-time", "08:00:00")
    fs.set_state("prepare", "COMPLETED")
    fs.set_state("teacher", "COMPLETED")
    fs.set_state("train", "TIMEOUT", elapsed="08:00:12", exitcode="0:15", node="g003")
    log(root, jobs["train"], "step 1500 of 2103 loss 0.31\nslurmstepd: error: *** JOB 1003 ON g003 CANCELLED DUE TO TIME LIMIT ***\n")
    out = status(capsys, root)
    assert table(out)["train"][:3] == ["train", "TIMEOUT", "08:00:12"]
    problems = section(out, "problems")
    flat = " ".join(problems.split())
    assert f"train (job {jobs['train']}) reached its time limit: it ran 08:00:12 of the 08:00:00 it asked for." in flat
    assert "continues from the latest save, so the work done so far is kept" in flat
    assert ("Run your original `mpd submit ...` command again with a larger --train-time (for example --train-time 16:00:00; "
            "this partition allows up to 4-00:00:00) and --force.") in flat
    assert f"last lines of the error log of train (job {jobs['train']}):" in out and "DUE TO TIME LIMIT" in out
    assert f"mpd logs {root.name} train --err" in out and "docs/TROUBLESHOOTING.md" in out


def test_the_maximum_comes_from_the_profile_of_the_cluster_the_campaign_was_planned_for(capsys, world):
    env, fs = world
    root, _ = submitted(capsys, env, "--train-time", "4-00:00:00", "--max-su", "50000")      # 4 days of an A100 is above the default cost guard
    fs.set_state("train", "TIMEOUT", elapsed="4-00:00:05")
    flat = " ".join(section(status(capsys, root), "problems").split())
    assert "It already asked for the longest time this partition allows (4-00:00:00)" in flat and "every attempt continues from the latest save" in flat
    plan = json.loads((root / "plan.json").read_text())
    assert {s["name"]: s["partition_limit_s"] for s in plan["stages"]} == {"prepare": 7 * 86400, "teacher": 4 * 86400, "train": 4 * 86400,
                                                                           "score-base": 4 * 86400, "score": 4 * 86400, "evaluate": 7 * 86400}


def test_a_timeout_of_a_stage_without_a_time_option_does_not_invent_one(capsys, world):
    env, fs = world
    root, _ = submitted(capsys, env)
    fs.set_state("prepare", "TIMEOUT", elapsed="04:00:03")
    flat = " ".join(section(status(capsys, root), "problems").split())
    assert "This stage has no time option of its own" in flat and "--train-time" not in flat


# ---------------------------------------------------------------- memory and nodes
def test_out_of_memory_by_state_and_by_exit_code_137(capsys, world):
    env, fs = world
    root, jobs = submitted(capsys, env)
    fs.set_state("train", "OUT_OF_MEMORY", elapsed="00:12:00", exitcode="0:125")
    fs.set_state("score-base", "FAILED", elapsed="00:02:00", exitcode="137:0")
    flat = " ".join(section(status(capsys, root), "problems").split())
    assert f"train (job {jobs['train']}) ran out of memory." in flat and f"score-base (job {jobs['score-base']}) ran out of memory (exit 137)." in flat
    assert "smaller student or a larger GPU (--gpu" in flat and "More GPUs do not help" in flat and "the job asked for 180G" in flat


def test_a_failed_node_names_the_node_sacct_reported(capsys, world):
    env, fs = world
    root, jobs = submitted(capsys, env)
    fs.set_state("train", "NODE_FAIL", elapsed="00:20:00", node="g011")
    flat = " ".join(section(status(capsys, root), "problems").split())
    assert "ended with NODE_FAIL: the node it ran on g011 failed" in flat
    assert "again with --force. If it lands on g011 again, add --sbatch-arg=--exclude=g011." in flat


# ---------------------------------------------------------------- FAILED and the markers of the log
@pytest.mark.parametrize("err,out,expect", [
    ("MPD_NO_GPU: no usable GPU on g007; resubmit with --sbatch-arg=--exclude=g007\n", "", "again with --sbatch-arg=--exclude=g007 and --force."),
    ("MPD_NO_GPU: torch.cuda.is_available() is False CUDA_VISIBLE_DEVICES=<unset> torch=2.10.0 cuda=12.8\n",
     "MPD_JOB_START host=g009.grace.hprc.tamu.edu job=1 stage=train\n", "again with --sbatch-arg=--exclude=g009 and --force."),
    ("MPD_MODULE_LOAD_FAILED Anaconda3/2024.02-1\n", "", "`bash scripts/setup_env.sh --cluster grace` writes it again"),
    ("weights for esm2-3b are missing (/x/models/esm2-3b). Run on a login node: mpd fetch model esm2-3b --cluster grace\n", "",
     "On a login node run `mpd fetch model esm2-3b --cluster grace`"),
    ("huggingface_hub.errors.OfflineModeIsEnabled: Cannot reach https://huggingface.co/api/models: offline mode is enabled.\n", "",
     "on a login node run `mpd fetch model <KEY>`"),
    ("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB\n", "", "the GPU's memory is too small for this model"),
    ("ValueError: nobody foresaw this\n", "", "shows all of it"),
])
def test_a_failed_stage_is_explained_from_the_markers_in_its_log(capsys, world, err, out, expect):
    env, fs = world
    root, jobs = submitted(capsys, env)
    fs.set_state("train", "FAILED", elapsed="00:00:31", exitcode="31:0" if "MPD_NO_GPU" in err else "1:0", node="")
    log(root, jobs["train"], err)
    if out:
        log(root, jobs["train"], out, err=False)
    text = status(capsys, root)
    flat = " ".join(section(text, "problems").split())
    assert expect.lower() in flat.lower(), flat
    assert err.splitlines()[0][:60] in text                                              # and the end of the log itself is shown
    assert "--force" in flat


def test_the_log_of_the_job_in_the_marker_is_used_not_the_newest_log_of_the_stage(capsys, world):
    env, fs = world
    root, jobs = submitted(capsys, env)
    fs.set_state("train", "CANCELLED")
    log(root, "4242", "MPD_NO_GPU: no usable GPU on g001; an earlier attempt\n")             # a log of another job of the same stage
    flat = " ".join(section(status(capsys, root), "problems").split())
    assert "was cancelled" in flat and "g001" not in flat and "earlier attempt" not in flat


def test_what_the_status_prints_of_a_log_hides_the_account_and_the_user(capsys, world):
    env, fs = world
    root, jobs = submitted(capsys, env)
    fs.set_state("train", "FAILED", elapsed="00:00:09", exitcode="1:0")
    log(root, jobs["train"], f"Traceback\nValueError: charged to {FAKE_ACCOUNT} by {FAKE_USER} ({FAKE_USER}@example.org)\n")
    out = status(capsys, root)
    assert FAKE_ACCOUNT not in out.split("last lines")[1] and "tester01" not in out.split("last lines")[1] and "<account>" in out


# ---------------------------------------------------------------- cancelled
def test_stages_cancelled_after_a_failure_get_one_line_and_the_failure_gets_the_advice(capsys, world):
    env, fs = world
    root, jobs = submitted(capsys, env)
    fs.set_state("prepare", "COMPLETED")
    fs.set_state("teacher", "COMPLETED")
    fs.set_state("train", "FAILED", exitcode="1:0", elapsed="00:01:00")
    for stage in ("score", "evaluate"):
        fs.set_state(stage, "CANCELLED", reason="DependencyNeverSatisfied")
    out = status(capsys, root)
    lines = section(out, "problems").splitlines()
    assert any(ln.strip().startswith(f"score (job {jobs['score']}) was cancelled because a stage it waited for did not finish.") for ln in lines)
    assert not any("Nothing to do for this stage" in ln for ln in lines)                      # the advice of the failed stage is the one to follow
    assert table(out)["score"][3] == "DependencyNeverSatisfied"


def test_after_mpd_cancel_status_says_once_that_the_same_command_submits_it_again(capsys, world):
    env, fs = world
    root, jobs = submitted(capsys, env)
    assert run_cli(capsys, "cancel", root.name)[0] == 0
    out = status(capsys, root)
    assert "state      cancelled" in out and all(r[1] == "CANCELLED" for r in table(out).values())
    flat = " ".join(section(out, "problems").split())
    assert flat == ("prepare, teacher, train, score-base, score, evaluate: cancelled with `mpd cancel`. "
                    "Run your original `mpd submit ...` command again to queue the campaign again.")
    assert "this campaign was cancelled: repeat your mpd submit command, with any option you want to change, to queue it again; stages that already finished are kept" in " ".join(out.split())


def test_a_failure_before_the_cancel_still_gets_its_own_advice(capsys, world):
    env, fs = world
    root, jobs = submitted(capsys, env)
    fs.set_state("prepare", "COMPLETED")
    fs.set_state("teacher", "FAILED", exitcode="1:0", elapsed="00:01:00")
    assert run_cli(capsys, "cancel", root.name)[0] == 0
    flat = " ".join(section(status(capsys, root), "problems").split())
    assert flat.startswith("train, score-base, score, evaluate: cancelled with `mpd cancel`.") and f"teacher (job {jobs['teacher']}) failed with exit code 1." in flat


# ---------------------------------------------------------------- waiting
def test_waiting_stages_get_one_line_each_with_slurms_reason_in_words(capsys, world):
    env, fs = world
    root, jobs = submitted(capsys, env)
    fs.set_state("prepare", "COMPLETED")
    fs.set_state("teacher", "PENDING", reason="Resources")
    fs.set_state("train", "PENDING", reason="QOSMaxJobsPerUserLimit")
    fs.set_state("score-base", "PENDING", reason="AssocGrpBillingMinutes")
    fs.set_state("score", "PENDING", reason="ReqNodeNotAvail, UnavailableNodes:g001")
    fs.set_state("evaluate", "PENDING", reason="SomethingNewInSlurm")
    out = status(capsys, root)
    waiting = section(out, "waiting").splitlines()
    text = "\n".join(waiting)
    assert len([ln for ln in waiting if ln.startswith("  ") and not ln.startswith("   ")]) == 5        # a long sentence wraps under its stage
    assert "teacher     Resources: it is next in line" in text and "train       QOSMaxJobsPerUserLimit: a limit of its QOS" in text
    assert "score-base  AssocGrpBillingMinutes: the account's SU balance" in text and "myproject" in text
    assert "score       ReqNodeNotAvail_UnavailableNodes:g001: a node it needs is not available" in text
    assert "(Slurm: ReqNodeNotAvail, UnavailableNodes:g001)" in " ".join(text.split())
    assert "evaluate    SomethingNewInSlurm: Slurm says: SomethingNewInSlurm" in text                        # unknown: printed as it is
    assert "problems:" not in out


def test_a_stage_that_waits_for_others_names_them(capsys, world):
    env, fs = world
    root, _ = submitted(capsys, env)
    waiting = section(status(capsys, root), "waiting")
    assert "score       Dependency: it waits for train to finish. Nothing to do." in waiting
    assert "evaluate    Dependency: it waits for score-base, score to finish. Nothing to do." in waiting
    assert "train       Dependency: it waits for teacher to finish." in waiting


def test_a_stage_that_waits_for_a_failed_one_does_not_say_nothing_to_do(capsys, world):
    env, fs = world
    root, _ = submitted(capsys, env)
    fs.set_state("prepare", "COMPLETED")
    fs.set_state("teacher", "COMPLETED")
    fs.set_state("train", "OUT_OF_MEMORY", exitcode="0:125", elapsed="00:12:00")       # Slurm has not yet changed its dependents' reason
    out = status(capsys, root)
    waiting = " ".join(section(out, "waiting").split())
    assert "score Dependency: it waits for train, which did not end well (see problems above); it cannot start before that is fixed." in waiting
    assert "evaluate Dependency: it waits for score-base, score to finish. Nothing to do." in waiting          # these two are fine
    assert "train (job" in section(out, "problems")


def test_a_partition_time_limit_says_to_cancel_and_submit_with_a_shorter_walltime(capsys, world):
    env, fs = world
    root, _ = submitted(capsys, env)
    fs.set_state("prepare", "PENDING", reason="PartitionTimeLimit")
    flat = " ".join(section(status(capsys, root), "waiting").split())
    assert f"`mpd cancel {root.name}`, then submit again with a shorter walltime" in flat


# ---------------------------------------------------------------- after a recovery
def test_after_a_recovery_status_lists_the_jobs_that_force_replaced(capsys, world):
    env, fs = world
    root, first = submitted(capsys, env, "--train-time", "01:00:00")
    for stage in ("prepare", "teacher"):
        (root / ("contexts/MANIFEST.json" if stage == "prepare" else "teacher/COMPLETE.json")).parent.mkdir(parents=True, exist_ok=True)
        (root / ("contexts/MANIFEST.json" if stage == "prepare" else "teacher/COMPLETE.json")).write_text("{}")
        fs.set_state(stage, "COMPLETED")
    fs.set_state("train", "TIMEOUT", elapsed="01:00:07")
    for stage in ("score", "evaluate", "score-base"):
        fs.set_state(stage, "CANCELLED", reason="DependencyNeverSatisfied")
    rc, out, err = run_cli(capsys, "submit", *recipe(env.home, "--account", FAKE_ACCOUNT, "--train-time", "10:00:00"), "--yes", "--force")
    assert rc == 0, err
    text = status(capsys, root)
    assert f"earlier    train {first['train']}; score-base {first['score-base']}; score {first['score']}; evaluate {first['evaluate']}" in text
    assert "(jobs that --force replaced; their logs stay in logs/)" in text
    assert table(text)["train"][1] == "PENDING" and "problems:" not in text


def test_a_campaign_with_no_job_recorded_yet_still_has_a_status(capsys, world):
    """A submission that died before sbatch accepted the first job leaves a marker without ids."""
    env, fs = world
    root, jobs = submitted(capsys, env)
    campaign.write_marker(root, {"campaign_id": root.name, "state": "partial", "jobs": {}, "message": "the controller never answered"})
    rc, out, err = run_cli(capsys, "status", root.name)
    assert rc == 0, err
    assert "state      partial" in out and "the controller never answered" in out and "stage  state  elapsed  reason  job  exit  tasks  note" in out
    assert "repeat your mpd submit command with --force to continue the chain" in out


def test_the_status_of_a_campaign_with_all_jobs_finished_points_at_the_results(capsys, world):
    env, fs = world
    root, jobs = submitted(capsys, env)
    for stage in jobs:
        fs.set_state(stage, "COMPLETED", elapsed="00:10:00")
    (root / "evaluate").mkdir()
    (root / "evaluate" / "summary.json").write_text("{}")
    out = status(capsys, root)
    assert f"mpd results {root.name}" in out and "problems:" not in out and "waiting:" not in out
    assert {f[1] for f in table(out).values()} == {"COMPLETED"} and campaign.campaign_state(root).state == "submitted"
