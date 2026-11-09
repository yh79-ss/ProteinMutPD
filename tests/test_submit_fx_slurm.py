"""The fake Slurm toolbox behaves as its docstring says: other areas' tests depend on it."""

import shutil
import subprocess

import pytest

import fx_slurm
from fx_slurm import CONTROLLER_DOWN, fake_slurm, stub_plan, stub_render, write_stub_campaign  # noqa: F401


def run(fs, *argv, **kw):
    return subprocess.run(list(argv), capture_output=True, text=True, env=fs.env(), stdin=subprocess.DEVNULL, **kw)


@pytest.fixture
def script(tmp_path):
    path = tmp_path / "job.sbatch"
    path.write_text("#!/bin/bash\n#SBATCH --job-name=demo\n#SBATCH --time=01:00:00\n#SBATCH --gres=gpu:a100:1\n"
                    "#SBATCH --partition=gpu\n#SBATCH --array=0-3%2\n\necho hi\n", encoding="utf-8")
    return path


def test_fake_binaries_shadow_the_real_ones_and_every_call_is_logged(fake_slurm, script):
    assert shutil.which("sbatch").startswith(str(fake_slurm.bin))
    assert shutil.which("sbatch") == str(fake_slurm.bin / "sbatch")
    out = run(fake_slurm, "sbatch", "--parsable", str(script))
    assert (out.returncode, out.stdout) == (0, "1001\n")
    out = run(fake_slurm, "sbatch", str(script))
    assert out.stdout == "Submitted batch job 1002\n"
    calls = fake_slurm.calls("sbatch")
    assert [c.rc for c in calls] == [0, 0] and calls[0].argv == ["--parsable", str(script)]
    assert [j["id"] for j in fake_slurm.submitted()] == [1001, 1002]


def test_sbatch_records_directives_comment_and_dependencies(fake_slurm, script):
    run(fake_slurm, "sbatch", "--parsable", "--comment=mpd:cid:nonce1:prepare", str(script))
    run(fake_slurm, "sbatch", "--parsable", "--comment=mpd:cid:nonce1:train", "--dependency=afterok:1001", "--kill-on-invalid-dep=yes", str(script))
    first, second = fake_slurm.job(1001), fake_slurm.job("train")
    assert (first["name"], first["stage"], first["comment"], first["gres"], first["n_array"]) == ("demo", "prepare", "mpd:cid:nonce1:prepare", "gpu:a100:1", 4)
    assert second["deps"] == [1001] and second["kill_invalid"] and second["reason"] == "Dependency" and first["reason"] == "Priority"


@pytest.mark.parametrize("args,text", [
    (["--get-user-env=L"], "option '--get-user-env' doesn't allow an argument"),     # verified on Vision, Slurm 25.05.6
    (["--bogus"], "unrecognized option '--bogus'"),
    (["--dependency=afterok:999"], "Job dependency problem"),
])
def test_sbatch_refuses_what_real_slurm_refuses(fake_slurm, script, args, text):
    out = run(fake_slurm, "sbatch", "--parsable", *args, str(script))
    assert out.returncode == 1 and text in out.stderr and not fake_slurm.jobs()


def test_sbatch_refuses_a_bad_script(fake_slurm, tmp_path):
    crlf, plain = tmp_path / "crlf.sbatch", tmp_path / "plain.sbatch"
    crlf.write_bytes(b"#!/bin/bash\r\necho hi\r\n")
    plain.write_text("echo hi\n")
    assert "DOS line breaks" in run(fake_slurm, "sbatch", str(crlf)).stderr
    assert "does not look like a batch script" in run(fake_slurm, "sbatch", str(plain)).stderr
    bad_directive = tmp_path / "bad.sbatch"
    bad_directive.write_text("#!/bin/bash\n#SBATCH --get-user-env=L\necho hi\n")
    assert "doesn't allow an argument" in run(fake_slurm, "sbatch", str(bad_directive)).stderr


def test_test_only_queues_nothing_and_is_logged_as_such(fake_slurm, script):
    out = run(fake_slurm, "sbatch", "--test-only", str(script))
    assert out.returncode == 0 and "to start at" in out.stderr and not fake_slurm.jobs()
    assert [c.test_only for c in fake_slurm.calls("sbatch")] == [True] and fake_slurm.submitted() == []


def test_rules_fail_first_calls_then_accept(fake_slurm, script):
    fake_slurm.fail_sbatch(2)
    results = [run(fake_slurm, "sbatch", "--parsable", str(script)) for _ in range(3)]
    assert [r.returncode for r in results] == [1, 1, 0]
    assert CONTROLLER_DOWN in results[0].stderr and results[2].stdout == "1001\n"


def test_accept_but_fail_queues_the_job_and_reports_an_error(fake_slurm, script):
    fake_slurm.accept_but_fail("job", times=1)
    out = run(fake_slurm, "sbatch", "--parsable", str(script))      # stage = file stem when there is no tag
    assert out.returncode == 1 and "Unable to contact" in out.stderr and out.stdout == ""
    assert list(fake_slurm.jobs()) == ["1001"]
    assert run(fake_slurm, "squeue", "-h", "-u", fake_slurm.user, "-o", "%F|%T").stdout == "1001|PENDING\n"


def test_hide_squeue_calls_delays_visibility_of_an_accepted_job(fake_slurm, script):
    fake_slurm.accept_but_fail("job", hide_squeue_calls=2)
    run(fake_slurm, "sbatch", "--parsable", str(script))
    seen = [run(fake_slurm, "squeue", "-h", "-o", "%F").stdout for _ in range(3)]
    assert seen == ["", "", "1001\n"]


def test_outage_window_follows_the_virtual_clock(fake_slurm, script):
    fake_slurm.outage(until_s=120)
    assert run(fake_slurm, "sbatch", "--parsable", str(script)).returncode == 1
    assert "Unable to contact" in run(fake_slurm, "squeue", "-h").stderr
    assert "Unable to contact" in run(fake_slurm, "scancel", "1").stderr
    fake_slurm.sleep(60)
    assert run(fake_slurm, "squeue", "-h").returncode == 1
    fake_slurm.sleep(61)
    assert fake_slurm.clock() == 121 and fake_slurm.sleeps == [60, 61]
    assert run(fake_slurm, "squeue", "-h").returncode == 0
    assert run(fake_slurm, "sbatch", "--parsable", str(script)).stdout == "1001\n"


def test_outage_that_still_accepts_hides_the_job_until_it_ends(fake_slurm, script):
    fake_slurm.outage(until_s=100, accept=True)
    assert run(fake_slurm, "sbatch", "--parsable", str(script)).returncode == 1
    assert list(fake_slurm.jobs()) == ["1001"]          # queued although sbatch said no
    fake_slurm.sleep(100)
    assert run(fake_slurm, "squeue", "-h", "-o", "%F").stdout == "1001\n"


def test_squeue_hides_finished_jobs_unless_asked_and_formats_fields(fake_slurm, script):
    run(fake_slurm, "sbatch", "--parsable", "--comment=mpd:c:n:train", str(script))
    other = fake_slurm.add_job(name="theirs", user="someone-else", comment="x")
    fake_slurm.set_state(1001, "COMPLETED")
    assert run(fake_slurm, "squeue", "-h", "-u", fake_slurm.user).stdout == ""
    out = run(fake_slurm, "squeue", "-h", "-t", "all", "-u", fake_slurm.user, "-o", "%F|%j|%k|%T|%r").stdout
    assert out == "1001|demo|mpd:c:n:train|COMPLETED|None\n" and other not in out
    assert run(fake_slurm, "squeue", "-h", "-u", "someone-else", "-o", "%i %T").stdout == f"{other} RUNNING\n"


def test_running_array_has_several_squeue_rows_with_one_base_id(fake_slurm, script):
    run(fake_slurm, "sbatch", "--parsable", str(script))
    fake_slurm.set_state(1001, "RUNNING")
    rows = run(fake_slurm, "squeue", "-h", "-o", "%F %T").stdout.splitlines()
    assert rows == ["1001 RUNNING", "1001 RUNNING", "1001 PENDING"]


def test_squeue_failure_injection(fake_slurm):
    fake_slurm.squeue_fails({"lte": 2})
    assert [run(fake_slurm, "squeue", "-h").returncode for _ in range(3)] == [1, 1, 0]


def test_scancel_cancels_and_complains_like_slurm(fake_slurm, script):
    run(fake_slurm, "sbatch", "--parsable", str(script))
    run(fake_slurm, "sbatch", "--parsable", str(script))
    fake_slurm.set_state(1002, "COMPLETED")
    out = run(fake_slurm, "scancel", "1001", "1002", "4040")
    assert out.returncode == 1
    assert "1002: Job/step already completed" in out.stderr and "4040: Invalid job id specified" in out.stderr
    assert fake_slurm.state_of(1001) == "CANCELLED"
    fake_slurm.scancel_fails()
    assert "Unable to contact" in run(fake_slurm, "scancel", "1001").stderr


def test_sacct_prints_array_tasks_and_pending_ranges(fake_slurm, script):
    run(fake_slurm, "sbatch", "--parsable", str(script))
    fmt = "--format=JobID,JobName,State,Elapsed,ExitCode"
    assert run(fake_slurm, "sacct", "-X", "-n", "-P", "-j", "1001", fmt).stdout == "1001_[0-3]|demo|PENDING|00:00:00|0:0\n"
    fake_slurm.set_state(1001, "RUNNING")
    assert run(fake_slurm, "sacct", "-X", "-n", "-P", "-j", "1001", fmt).stdout.splitlines() == [
        "1001_0|demo|RUNNING|00:10:00|0:0", "1001_1|demo|RUNNING|00:10:00|0:0", "1001_[2-3]|demo|PENDING|00:00:00|0:0"]
    fake_slurm.set_state(1001, "FAILED", exitcode="1:0")
    assert run(fake_slurm, "sacct", "-X", "-n", "-P", "-j", "1001", fmt).stdout.splitlines()[0].endswith("FAILED|00:30:00|1:0")


def test_sacct_lags_behind_the_queue_for_its_first_calls(fake_slurm, script):
    fake_slurm.configure(sacct={"lag_calls": 1})
    run(fake_slurm, "sbatch", "--parsable", str(script))
    assert run(fake_slurm, "sacct", "-n", "-P", "-j", "1001").stdout == ""
    assert run(fake_slurm, "sacct", "-n", "-P", "-j", "1001").stdout.startswith("1001_[0-3]|demo|PENDING")


def test_progress_mode_walks_a_chain_and_cancels_dependents_of_a_failure(fake_slurm, tmp_path):
    plan = stub_plan(tmp_path, units=("all",), shards=1, teacher=False)
    scripts = write_stub_campaign(plan)
    ids = {}
    for stage in plan.stages:
        dep = [ids[d] for d in stage.depends_on]
        extra = ["--dependency=afterok:" + ":".join(dep), "--kill-on-invalid-dep=yes"] if dep else []
        ids[stage.full_name] = run(fake_slurm, "sbatch", "--parsable", *extra, str(scripts[stage.full_name])).stdout.strip()
    fake_slurm.configure(progress={"enabled": True, "end_state": {"train": "FAILED"}})
    states = []
    for _ in range(4):
        run(fake_slurm, "squeue", "-h")
        states.append({s: fake_slurm.state_of(i) for s, i in ids.items()})
    assert states[0]["prepare"] == "RUNNING" and states[0]["train"] == "PENDING"
    assert states[-1]["prepare"] == "COMPLETED" and states[-1]["train"] == "FAILED"
    assert states[-1]["score"] == "CANCELLED" and states[-1]["evaluate"] == "CANCELLED"     # --kill-on-invalid-dep
    assert states[-1]["score-base"] == "COMPLETED"


def test_module_is_a_shell_function_that_changes_loadedmodules(fake_slurm):
    script = ('type module >/dev/null 2>&1 || { echo NO_MODULE; exit 3; }\n'
              'module purge\nmodule load GCC/11.3.0 Python/3.11.5 || exit 4\necho "loaded=$LOADEDMODULES"\n'
              'module load Nonexistent/1.0 2>/dev/null; echo "unknown_rc=$?"\nmodule purge; echo "after_purge=[${LOADEDMODULES}]"\n')
    out = run(fake_slurm, "bash", "-c", script)
    assert out.stdout.splitlines() == ["loaded=GCC/11.3.0:Python/3.11.5", "unknown_rc=1", "after_purge=[]"], out.stderr
    assert [c.argv for c in fake_slurm.calls("module")][:2] == [["purge"], ["load", "GCC/11.3.0", "Python/3.11.5"]]


def test_available_modules_can_be_changed(fake_slurm):
    fake_slurm.set_modules(["Only/1.0"])
    out = run(fake_slurm, "bash", "-c", "module load GCC/11.3.0; echo rc=$?")
    assert "rc=1" in out.stdout and "unknown" in out.stderr


def test_canned_site_commands_and_nvidia_smi(fake_slurm):
    assert "123456789012" in run(fake_slurm, "myproject").stdout
    assert "files" in run(fake_slurm, "showquota").stdout
    assert "A100" in run(fake_slurm, "nvidia-smi", "-L").stdout
    fake_slurm.configure(nvidia_smi={"fail": True}, myproject={"stdout": "no table\n", "exit": 2})
    assert run(fake_slurm, "nvidia-smi", "-L").returncode == 9
    out = run(fake_slurm, "myproject")
    assert (out.returncode, out.stdout) == (2, "no table\n")
    assert "MaxArraySize" in run(fake_slurm, "scontrol", "show", "config").stdout


def test_tripwire_tools_fail_loudly_and_are_reported_at_teardown(fake_slurm):
    out = run(fake_slurm, "srun", "hostname")
    assert out.returncode == 99 and "never" not in out.stdout
    with pytest.raises(AssertionError, match="never simulated"):
        fake_slurm.assert_no_tripwire()
    fake_slurm.clear_tripwires()            # this test called it on purpose
    fake_slurm.assert_no_tripwire()


def test_removed_binary_is_absent_from_path_even_when_a_real_one_exists(fake_slurm):
    assert shutil.which("sacct") == str(fake_slurm.bin / "sacct")
    fake_slurm.remove("sacct")
    assert shutil.which("sacct") is None
    assert run(fake_slurm, "sh", "-c", "command -v sacct").stdout == ""
    assert run(fake_slurm, "sh", "-c", "sacct -j 1").returncode == 127
    fake_slurm.restore("sacct")
    assert shutil.which("sacct") == str(fake_slurm.bin / "sacct")


def test_real_slurm_clients_that_are_not_faked_cannot_be_reached(fake_slurm):
    for tool in ("sinfo", "sstat", "sprio", "sshare", "sreport", "sdiag"):
        assert shutil.which(tool) is None, tool
    assert shutil.which("bash") and shutil.which("python3") or shutil.which("sh")        # ordinary tools stay reachable
    fake_slurm.remove("showquota")
    assert shutil.which("showquota") is None and shutil.which("ls")


def test_stub_plan_and_campaign_files_have_the_real_layout(fake_slurm, tmp_path):
    plan = stub_plan(tmp_path, units=("a", "b"), datasets=("proteingym", "venusmuthub"), shards=2)
    scripts = write_stub_campaign(plan, state="submitted", jobs={"prepare": "1"})
    assert list(scripts) == [s.full_name for s in plan.stages]
    text = stub_render(plan, plan.stages[3])
    assert text.startswith("#!/bin/bash\n") and f"#SBATCH --comment=mpd:{plan.campaign_id}:{plan.nonce}:" in text and "\r" not in text
    assert {s.name: len(s.tasks) for s in plan.stages}["score"] == 2 * 2 * 2
    chunked = stub_plan(tmp_path, units=("a",), shards=3, score_chunks=2, campaign_id="chunks__x")
    names = [s.full_name for s in chunked.stages]
    assert names == ["prepare", "teacher", "train", "score-base", "score", "score.1", "evaluate"]
    assert chunked.stages[-1].depends_on == ["score-base", "score", "score.1"] and chunked.stages[5].task_offset == 2
    assert [s.full_name for s in stub_plan(tmp_path, minimal=True).stages] == ["prepare", "train", "evaluate"]
    assert (plan.campaign_dir / "SUBMITTED.json").is_file() and (plan.campaign_dir / "tasks" / "score.json").is_file()
    assert fx_slurm.TERMINAL >= {"COMPLETED", "FAILED"}


def test_partitions_refuse_unknown_names_and_either_refuse_or_pend_an_over_long_request(fake_slurm, script):
    fake_slurm.configure(sbatch={"partitions": {"gpu": {"max_time_s": 4 * 86400}, "short": {"max_time_s": 7200, "exceed": "pend"}}})
    text = script.read_text()
    script.write_text(text.replace("--partition=gpu", "--partition=nosuch"))
    assert "invalid partition specified: nosuch" in run(fake_slurm, "sbatch", str(script)).stderr
    script.write_text(text.replace("--time=01:00:00", "--time=5-00:00:00"))
    assert "Requested time limit is invalid" in run(fake_slurm, "sbatch", str(script)).stderr
    script.write_text(text.replace("--partition=gpu", "--partition=short").replace("--time=01:00:00", "--time=12:00:00"))
    assert run(fake_slurm, "sbatch", "--parsable", str(script)).stdout == "1001\n"           # Grace accepts it ...
    assert fake_slurm.job(1001)["reason"] == "PartitionTimeLimit"                           # ... and the job waits for ever
    assert run(fake_slurm, "squeue", "-h", "-o", "%i %r").stdout.strip().endswith("PartitionTimeLimit")


def test_gres_types_are_case_sensitive_and_flags_can_be_refused(fake_slurm, script):
    fake_slurm.configure(sbatch={"gres_types": ["H200"], "reject_flags": {"--partition": "sbatch: error: no partitions on this cluster"}})
    out = run(fake_slurm, "sbatch", "--test-only", str(script))
    assert out.returncode == 1 and "no partitions on this cluster" in out.stderr
    fake_slurm.configure(sbatch={"gres_types": ["H200"]})
    assert "Requested node configuration is not available" in run(fake_slurm, "sbatch", str(script)).stderr
    script.write_text(script.read_text().replace("gpu:a100:1", "gpu:H200:1"))
    assert run(fake_slurm, "sbatch", "--parsable", str(script)).returncode == 0


def test_rules_can_be_limited_to_a_time_window_and_to_test_only_calls(fake_slurm, script):
    fake_slurm.rule(do="fail", message="window", after_s=100, until_s=200, test_only=False)
    fake_slurm.rule(do="fail", message="only test-only", test_only=True)
    assert "only test-only" in run(fake_slurm, "sbatch", "--test-only", str(script)).stderr
    assert run(fake_slurm, "sbatch", "--parsable", str(script)).returncode == 0
    fake_slurm.advance(150)
    assert "window" in run(fake_slurm, "sbatch", "--parsable", str(script)).stderr
    fake_slurm.advance(100)
    assert run(fake_slurm, "sbatch", "--parsable", str(script)).returncode == 0


def test_finished_jobs_can_vanish_from_squeue_like_purged_ones_and_scontrol_describes_a_job(fake_slurm, script):
    run(fake_slurm, "sbatch", "--parsable", "--comment=mpd:c:n:train", str(script))
    fake_slurm.set_state(1001, "COMPLETED")
    assert run(fake_slurm, "squeue", "-h", "-t", "all", "-o", "%F").stdout == "1001\n"
    fake_slurm.configure(squeue={"hide_finished": True})
    assert run(fake_slurm, "squeue", "-h", "-t", "all", "-o", "%F").stdout == ""
    shown = run(fake_slurm, "scontrol", "show", "job", "1001").stdout
    assert "JobState=COMPLETED" in shown and "Comment=mpd:c:n:train" in shown
    assert run(fake_slurm, "scontrol", "show", "nothing").returncode == 1
