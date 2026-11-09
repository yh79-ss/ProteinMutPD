"""Recovery flows, end to end: the real planner and renderer, the real CLI and the fake Slurm toolbox.

A student whose training hit its time limit, or whose job landed on a node without a working GPU, repeats the original
`mpd submit` command with the one option they changed and `--force`. The campaign must stay the same, the stages that finished
must not be queued again, and the stages that must run again must carry the new option. Nothing here talks to a real cluster; the
stage output files that mark a finished stage are written by hand from the list in the slurm-submit hand-off.
"""

import hashlib
import json
from pathlib import Path

import pytest
from fx_cli import FAKE_ACCOUNT, cli_env, ready_home, recipe, run_cli, write_logs  # noqa: F401
from fx_slurm import fake_slurm  # noqa: F401

from mpd import cli
from mpd.slurm import campaign
from mpd.slurm.submit import submit as real_submit

pytestmark = pytest.mark.usefixtures("cli_env")

STAGES = ["prepare", "teacher", "train", "score-base", "score", "evaluate"]


@pytest.fixture
def world(cli_env, monkeypatch, fake_slurm):
    """A ready Grace home, the fake Slurm, and a `submit` whose waits are virtual (the CLI would sleep for real)."""
    ready_home(cli_env, monkeypatch)
    monkeypatch.setattr(cli, "submit", lambda plan, scripts, **kw: real_submit(plan, scripts, sleep=fake_slurm.sleep, clock=fake_slurm.clock, **kw))
    return cli_env, fake_slurm


def command(env, *extra) -> list[str]:
    """The student's `mpd submit ...` command; `extra` are the options they add or change."""
    return ["submit", *recipe(env.home, "--account", FAKE_ACCOUNT, *extra), "--yes"]


def campaigns(env) -> list[Path]:
    runs = env.home / "runs"
    return sorted(p for p in runs.iterdir() if p.is_dir()) if runs.is_dir() else []


def finish(root: Path, *stages: str, shards: int = 4) -> None:
    """The files a finished stage leaves behind (handoff/slurm-submit/NOTES.md, section 3), for the one unit `all`."""
    files = {"prepare": ["contexts/MANIFEST.json"], "teacher": ["teacher/COMPLETE.json"], "train": ["train/all/COMPLETE.json"],
             "evaluate": ["evaluate/summary.json"],
             "score-base": [f"score/proteingym/base/shard_{i:03d}_of_{shards:03d}.json" for i in range(shards)],
             "score": [f"score/proteingym/all/shard_{i:03d}_of_{shards:03d}.json" for i in range(shards)]}
    for stage in stages:
        for rel in files[stage]:
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            (root / rel).write_text("{}", encoding="utf-8")


def snapshot(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


def queued(fs, stage: str) -> list[dict]:
    return [j for j in fs.submitted() if j["stage"] == stage]


def deps(job: dict) -> list[int]:
    return job["deps"]


def alive_tagged(fs, cid: str) -> dict[str, list[str]]:
    """stage -> ids of the jobs squeue shows as queued or running with this campaign's tag: what a duplicate would look like."""
    rows, why = campaign.read_queue("squeue", user=fs.user)
    assert rows is not None, why
    found: dict[str, list[str]] = {}
    for row in campaign.tagged(rows, cid, alive_only=True):
        ids = found.setdefault(campaign.parse_tag(row.comment)[2], [])
        if row.job_id not in ids:                                     # a running array is one row per task, all with the id sbatch printed
            ids.append(row.job_id)
    return found


def script_of(root: Path, stage: str) -> str:
    return (root / "scripts" / f"{stage}.sbatch").read_text(encoding="utf-8")


def time_limit(text: str) -> str:
    return next(line.split("=", 1)[1] for line in text.splitlines() if line.startswith("#SBATCH --time="))


def nonce_of(text: str) -> str:
    return next(line for line in text.splitlines() if line.startswith("#SBATCH --comment=")).split(":")[-2]


def submit_first_round(capsys, env, *extra):
    rc, out, err = run_cli(capsys, *command(env, *extra))
    assert rc == 0, (out, err)
    (root,) = campaigns(env)
    return root, json.loads((root / "SUBMITTED.json").read_text())["jobs"]


# ---------------------------------------------------------------- (a) TIMEOUT: a longer --train-time and --force
def timed_out_world(fs, root: Path, *, score_base: str = "COMPLETED") -> None:
    """prepare and teacher finished; train hit its limit; what waited for it was cancelled by Slurm."""
    finish(root, "prepare", "teacher")
    fs.set_state("prepare", "COMPLETED", elapsed="00:03:11")
    fs.set_state("teacher", "COMPLETED", elapsed="00:40:12")
    fs.set_state("train", "TIMEOUT", elapsed="01:00:07", exitcode="0:15", node="g003")
    if score_base == "COMPLETED":
        finish(root, "score-base")
    fs.set_state("score-base", score_base, elapsed="00:25:00")
    for stage in ("score", "evaluate"):
        fs.set_state(stage, "CANCELLED", reason="DependencyNeverSatisfied")


def test_a_timeout_is_recovered_in_the_same_campaign_with_a_longer_time_and_force(capsys, world):
    env, fs = world
    root, first = submit_first_round(capsys, env, "--train-time", "01:00:00")
    assert time_limit(script_of(root, "train")) == "01:00:00"
    old_nonce = nonce_of(script_of(root, "train"))
    timed_out_world(fs, root)
    before = len(fs.calls("sbatch"))

    rc, out, err = run_cli(capsys, *command(env, "--train-time", "10:00:00"), "--force")
    assert rc == 0, (out, err)

    assert campaigns(env) == [root]                                                        # the same campaign, no second directory
    new = [c.info["stage"] for c in fs.calls("sbatch")[before:] if not c.test_only]
    assert new == ["train", "score", "evaluate"]                                           # prepare, teacher and score-base are not queued again
    assert [len(queued(fs, s)) for s in STAGES] == [1, 1, 2, 1, 2, 2]
    now = json.loads((root / "SUBMITTED.json").read_text())
    assert now["state"] == "submitted" and now["jobs"]["prepare"] == first["prepare"] and now["jobs"]["teacher"] == first["teacher"]
    train, score, evaluate = (queued(fs, s)[-1] for s in ("train", "score", "evaluate"))
    assert deps(train) == [] and deps(score) == [train["id"]]                              # teacher is finished: nothing to wait for
    assert set(deps(evaluate)) == {score["id"]}                                            # score-base is finished too
    assert now["jobs"]["train"] == str(train["id"]) and now["replaced"]["train"] == [first["train"]]
    assert time_limit(script_of(root, "train")) == "10:00:00" and time_limit(script_of(root, "prepare")) != "10:00:00"
    assert nonce_of(script_of(root, "train")) != old_nonce                                 # a new submission, a new tag
    assert train["comment"] == f"mpd:{root.name}:{nonce_of(script_of(root, 'train'))}:train"
    assert alive_tagged(fs, root.name) == {s: [str(queued(fs, s)[-1]["id"])] for s in ("train", "score", "evaluate")}
    assert all(f"finished  {s}" in out for s in ("prepare", "teacher", "score-base")) and "kept      train" not in out


def test_a_timeout_with_score_base_still_running_keeps_it_and_evaluate_waits_for_both(capsys, world):
    env, fs = world
    root, first = submit_first_round(capsys, env, "--train-time", "01:00:00")
    timed_out_world(fs, root, score_base="RUNNING")
    rc, out, err = run_cli(capsys, *command(env, "--train-time", "10:00:00"), "--force")
    assert rc == 0, (out, err)
    assert len(queued(fs, "score-base")) == 1 and [len(queued(fs, s)) for s in ("train", "score", "evaluate")] == [2, 2, 2]
    score, evaluate = queued(fs, "score")[-1], queued(fs, "evaluate")[-1]
    assert set(deps(evaluate)) == {score["id"], int(first["score-base"])}                  # the chain stays intact through the kept job
    assert "kept      score-base" in out and "note: kept stages (score-base) stay queued or running" in out
    assert fs.state_of(first["score-base"]) == "RUNNING"                                   # the running job was not touched
    assert alive_tagged(fs, root.name)["score-base"] == [first["score-base"]]


def test_a_timeout_with_score_base_cancelled_too_queues_every_score_stage_again(capsys, world):
    env, fs = world
    root, first = submit_first_round(capsys, env, "--train-time", "01:00:00")
    timed_out_world(fs, root, score_base="CANCELLED")                    # an administrator cancelled it: no shard files, no live job
    rc, out, err = run_cli(capsys, *command(env, "--train-time", "10:00:00"), "--force")
    assert rc == 0, (out, err)
    assert [len(queued(fs, s)) for s in STAGES] == [1, 1, 2, 2, 2, 2]
    base, train, score, evaluate = (queued(fs, s)[-1] for s in ("score-base", "train", "score", "evaluate"))
    assert deps(base) == [] and deps(train) == [] and deps(score) == [train["id"]]               # score-base needs only prepare, which is finished
    assert set(deps(evaluate)) == {base["id"], score["id"]}
    assert json.loads((root / "SUBMITTED.json").read_text())["replaced"].keys() == {"train", "score-base", "score", "evaluate"}
    assert all(len(v) == 1 for v in alive_tagged(fs, root.name).values())


def test_waiting_jobs_that_lost_their_dependency_are_cancelled_not_duplicated(capsys, world):
    env, fs = world
    root, first = submit_first_round(capsys, env, "--train-time", "01:00:00")
    timed_out_world(fs, root)
    for stage in ("score", "evaluate"):                      # a cluster that ignores --kill-on-invalid-dep leaves them pending for ever
        fs.set_state(stage, "PENDING", reason="DependencyNeverSatisfied")
    rc, out, err = run_cli(capsys, *command(env, "--train-time", "10:00:00"), "--force")
    assert rc == 0, (out, err)
    assert fs.state_of(first["score"]) == "CANCELLED" and fs.state_of(first["evaluate"]) == "CANCELLED"
    assert all(len(v) == 1 for v in alive_tagged(fs, root.name).values())


# ---------------------------------------------------------------- (b) a node without a working GPU: --sbatch-arg=--exclude and --force
def test_a_bad_node_is_recovered_with_exclude_and_force(capsys, world):
    env, fs = world
    root, first = submit_first_round(capsys, env)
    finish(root, "prepare", "teacher")
    fs.set_state("prepare", "COMPLETED")
    fs.set_state("teacher", "COMPLETED")
    fs.set_state("train", "FAILED", elapsed="00:00:09", exitcode="31:0", node="g007")
    fs.set_state("score-base", "RUNNING")
    for stage in ("score", "evaluate"):
        fs.set_state(stage, "CANCELLED", reason="DependencyNeverSatisfied")
    write_logs(root, {f"mpd-train-esm2650m_{first['train']}.err":
                      "MPD_NO_GPU: no usable GPU on g007; resubmit with --sbatch-arg=--exclude=g007\n"})
    assert "#SBATCH --exclude" not in script_of(root, "train")

    rc, out, _ = run_cli(capsys, "status", root.name, "--home", env.home)                  # what the student reads first
    assert rc == 0 and "--sbatch-arg=--exclude=g007" in out and "--force" in out

    before = len(fs.calls("sbatch"))
    rc, out, err = run_cli(capsys, *command(env, "--sbatch-arg=--exclude=g007"), "--force")
    assert rc == 0, (out, err)
    assert campaigns(env) == [root]
    assert [c.info["stage"] for c in fs.calls("sbatch")[before:] if not c.test_only] == ["train", "score", "evaluate"]
    assert "#SBATCH --exclude=g007" in script_of(root, "train") and "#SBATCH --exclude=g007" in script_of(root, "prepare")
    assert fs.state_of(first["score-base"]) == "RUNNING" and len(queued(fs, "score-base")) == 1
    assert all(len(v) == 1 for v in alive_tagged(fs, root.name).values())


# ---------------------------------------------------------------- (c) without --force the re-run refuses and says what --force does
def test_a_re_run_without_force_refuses_names_the_three_options_and_changes_nothing(capsys, world):
    env, fs = world
    root, first = submit_first_round(capsys, env, "--train-time", "01:00:00")
    timed_out_world(fs, root)
    before, calls = snapshot(root), len(fs.calls("sbatch"))
    for extra in ([], ["--dry-run"]):
        rc, out, err = run_cli(capsys, *command(env, "--train-time", "10:00:00"), *extra)
        assert rc == 2 and "was already submitted" in err and root.name in err
        assert f"mpd status {root.name}" in err and f"mpd cancel {root.name}" in err and "--force" in err         # the three options
        assert "--train-time" in err and "--sbatch-arg" in err and "applies it to the stages it queues again" in " ".join(err.split())
        assert "different campaign" in " ".join(err.split()) and "--seed" in err
    assert snapshot(root) == before and len(fs.calls("sbatch")) == calls and campaigns(env) == [root]


def test_the_library_refusal_says_the_same(world, tmp_path):
    env, fs = world
    from mpd.slurm.submit import SubmitError, write_campaign
    from fx_render import make
    plan = make("grace", home=env.home)
    write_campaign(plan)
    campaign.write_marker(plan.campaign_dir, {"campaign_id": plan.campaign_id, "state": "submitted", "jobs": {"prepare": "7"}})
    with pytest.raises(SubmitError) as exc:
        write_campaign(make("grace", home=env.home, train_time="10:00:00"))
    text = " ".join(str(exc.value).split())
    assert "--force applies it to the stages it queues again" in text and "Nothing was written" in text


# ---------------------------------------------------------------- (d) after `mpd cancel` a plain re-run is allowed
def test_after_cancel_a_plain_re_run_is_allowed_and_queues_the_campaign_again(capsys, world):
    env, fs = world
    root, first = submit_first_round(capsys, env, "--train-time", "01:00:00")
    rc, out, err = run_cli(capsys, "cancel", root.name, "--home", env.home)
    assert rc == 0, err
    assert json.loads((root / "SUBMITTED.json").read_text())["state"] == "cancelled"
    assert all(fs.state_of(j) == "CANCELLED" for j in first.values())

    rc, out, err = run_cli(capsys, *command(env, "--train-time", "10:00:00"))              # no --force
    assert rc == 0, (out, err)
    assert "ended as cancelled; submitting it again" in out and campaigns(env) == [root]
    now = json.loads((root / "SUBMITTED.json").read_text())
    assert now["state"] == "submitted" and set(now["jobs"]) == set(STAGES)
    assert all(now["jobs"][s] != first[s] for s in STAGES)                                  # every stage is queued again
    assert time_limit(script_of(root, "train")) == "10:00:00"
    assert all(len(v) == 1 for v in alive_tagged(fs, root.name).values())


@pytest.mark.parametrize("extra", [[], ["--force"]], ids=["plain", "force"])
def test_after_cancel_the_stages_that_finished_are_not_queued_again(capsys, world, extra):
    """A finished teacher that is queued again waits for a GPU only to print "already computed", and train waits for it. After
    `mpd cancel` nothing is alive, so the output files decide, with or without --force."""
    env, fs = world
    root, first = submit_first_round(capsys, env)
    finish(root, "prepare", "teacher")
    fs.set_state("prepare", "COMPLETED")
    fs.set_state("teacher", "COMPLETED")
    assert run_cli(capsys, "cancel", root.name, "--home", env.home)[0] == 0
    before = len(fs.calls("sbatch"))
    rc, out, err = run_cli(capsys, *command(env, "--train-time", "10:00:00"), *extra)
    assert rc == 0, (out, err)
    assert [c.info["stage"] for c in fs.calls("sbatch")[before:] if not c.test_only] == ["train", "score-base", "score", "evaluate"]
    assert "finished  prepare" in out and "finished  teacher" in out and "ended as cancelled; submitting it again" in out
    now = json.loads((root / "SUBMITTED.json").read_text())
    assert now["state"] == "submitted" and list(now["jobs"]) == STAGES
    assert now["jobs"]["prepare"] == first["prepare"] and now["jobs"]["teacher"] == first["teacher"]
    assert all(now["jobs"][s] != first[s] for s in ("train", "score-base", "score", "evaluate"))
    assert deps(queued(fs, "train")[-1]) == []                                              # the teacher is finished: train starts at once
    assert time_limit(script_of(root, "train")) == "10:00:00"
    assert alive_tagged(fs, root.name).keys() == {"train", "score-base", "score", "evaluate"}


# ---------------------------------------------------------------- (h) a stage whose marker file was deleted runs again, the rest is kept
def finished_world(capsys, env, fs):
    root, first = submit_first_round(capsys, env)
    finish(root, *STAGES)
    for stage in STAGES:
        fs.set_state(stage, "COMPLETED")
    return root, first


@pytest.mark.parametrize("deleted, again", [("teacher/COMPLETE.json", "teacher"), ("contexts/MANIFEST.json", "prepare")])
def test_a_deleted_marker_queues_that_stage_again_and_keeps_the_rest(capsys, world, deleted, again):
    """The messages of the stages say: delete the marker file and repeat the `mpd submit` command with --force."""
    env, fs = world
    root, first = finished_world(capsys, env, fs)
    (root / deleted).unlink()
    before = len(fs.calls("sbatch"))
    rc, out, err = run_cli(capsys, *command(env), "--force")
    assert rc == 0, (out, err)
    assert [c.info["stage"] for c in fs.calls("sbatch")[before:] if not c.test_only] == [again]
    assert all(f"finished  {s}" in out for s in STAGES if s != again) and f"submitted {again}" in out
    now = json.loads((root / "SUBMITTED.json").read_text())
    assert now["jobs"][again] != first[again] and now["replaced"] == {again: [first[again]]}
    assert all(now["jobs"][s] == first[s] for s in STAGES if s != again)
    assert alive_tagged(fs, root.name) == {again: [now["jobs"][again]]}


def test_a_deleted_teacher_marker_also_queues_the_unfinished_stages_that_wait_for_it(capsys, world):
    env, fs = world
    root, first = submit_first_round(capsys, env)                    # train, score-base, score and evaluate are still queued
    finish(root, "prepare", "teacher")
    fs.set_state("prepare", "COMPLETED")
    fs.set_state("teacher", "COMPLETED")
    (root / "teacher" / "COMPLETE.json").unlink()                   # the student wants the teacher table computed again
    before = len(fs.calls("sbatch"))
    rc, out, err = run_cli(capsys, *command(env), "--force")
    assert rc == 0, (out, err)
    assert [c.info["stage"] for c in fs.calls("sbatch")[before:] if not c.test_only] == ["teacher", "train", "score", "evaluate"]
    teacher, train, evaluate = queued(fs, "teacher")[-1], queued(fs, "train")[-1], queued(fs, "evaluate")[-1]
    assert deps(teacher) == [] and deps(train) == [teacher["id"]]         # the chain is intact through the new teacher job
    assert all(fs.state_of(first[s]) == "CANCELLED" for s in ("train", "score", "evaluate"))        # they waited for a job that is gone
    assert fs.state_of(first["score-base"]) == "PENDING" and len(queued(fs, "score-base")) == 1     # it needs only prepare: kept
    assert set(deps(evaluate)) == {int(first["score-base"]), queued(fs, "score")[-1]["id"]}
    assert all(len(v) == 1 for v in alive_tagged(fs, root.name).values())


# ---------------------------------------------------------------- (e) a controller outage in the middle of the chain, then --force
def test_a_partial_submit_is_completed_by_force_without_duplicates(capsys, world, monkeypatch):
    env, fs = world
    fs.outage(from_s=100, until_s=5000, accept=True)             # sbatch queues a job and reports failure; squeue, scancel and sacct are down too

    def submit_then_lose_the_controller(plan, scripts, **kw):
        def echo(line):
            print(line, flush=True)
            if line.startswith("submitted teacher"):
                fs.advance(100)                                  # the outage starts after the second job
        return real_submit(plan, scripts, sleep=fs.sleep, clock=fs.clock, echo=echo, **kw)

    monkeypatch.setattr(cli, "submit", submit_then_lose_the_controller)
    rc, out, err = run_cli(capsys, *command(env))
    assert rc == 2 and "Slurm did not answer" in err
    (root,) = campaigns(env)
    half = json.loads((root / "SUBMITTED.json").read_text())
    assert half["state"] == "partial" and set(half["jobs"]) == {"prepare", "teacher"}
    fs.advance(6000)                                             # the outage is over
    monkeypatch.setattr(cli, "submit", lambda plan, scripts, **kw: real_submit(plan, scripts, sleep=fs.sleep, clock=fs.clock, **kw))

    rc, out, err = run_cli(capsys, *command(env), "--force")
    assert rc == 0, (out, err)
    done = json.loads((root / "SUBMITTED.json").read_text())
    assert done["state"] == "submitted" and list(done["jobs"]) == STAGES
    assert done["jobs"]["prepare"] == half["jobs"]["prepare"] and done["jobs"]["teacher"] == half["jobs"]["teacher"]
    found = alive_tagged(fs, root.name)
    assert set(found) == set(STAGES) and all(len(v) == 1 for v in found.values()), found    # one live job per stage, no duplicates
    assert {s: found[s][0] for s in STAGES} == done["jobs"]
    assert len(queued(fs, "train")) == 1, "the train job that sbatch accepted while reporting an error must be adopted, not queued again"


# ---------------------------------------------------------------- (f) another --gpu is another campaign
def test_another_gpu_is_a_new_campaign_and_the_old_one_is_untouched(capsys, world):
    env, fs = world
    root, first = submit_first_round(capsys, env)
    before, calls = snapshot(root), len(fs.calls("scancel"))
    rc, out, err = run_cli(capsys, *command(env, "--gpu", "a40"))
    assert rc == 0, (out, err)
    other = [p for p in campaigns(env) if p != root]
    assert len(other) == 1 and other[0].name.split("__")[:-1] == root.name.split("__")[:-1] and other[0].name != root.name
    assert snapshot(root) == before and len(fs.calls("scancel")) == calls
    assert json.loads((root / "SUBMITTED.json").read_text())["jobs"] == first
    assert all(fs.state_of(j) == "PENDING" for j in first.values())
    assert "#SBATCH --gres=gpu:a40:1" in script_of(other[0], "train") and "#SBATCH --gres=gpu:a100:1" in script_of(root, "train")
    assert len(fs.submitted()) == 12                                                         # a whole second chain: it trains again


def test_scheduling_options_do_not_start_a_new_campaign(capsys, world):
    env, fs = world
    root, _ = submit_first_round(capsys, env)
    for extra in (["--train-time", "10:00:00"], ["--score-time", "03:00:00"], ["--teacher-time", "05:00:00"], ["--array-throttle", "3"],
                  ["--sbatch-arg=--exclude=g001"], ["--module", "Anaconda3/2024.02-1"], ["--env", "FOO=1"], ["--account", "654321"]):
        rc, out, err = run_cli(capsys, "plan", *recipe(env.home, *extra))
        assert rc == 0, (extra, out, err)
        assert f"campaign   {root.name}\n" in out, extra
    rc, out, _ = run_cli(capsys, "plan", *recipe(env.home, "--gpu", "a40"))
    assert f"campaign   {root.name}\n" not in out
