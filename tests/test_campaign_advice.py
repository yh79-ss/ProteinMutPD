"""The words `mpd status` uses when a stage did not end well or has not started: one table (the dicts in mpd.slurm.campaign), pinned here.

The functions are pure, so every case is built by hand from what sacct and squeue would give: a state, an exit code, a reason, a time
limit, a node, and the end of a log. Nothing here is a claim about Grace; the texts say what mpd does with what Slurm reported.
"""

import re

import pytest

from mpd.slurm import campaign
from mpd.slurm.campaign import (ADVICE, CAUSE_ADVICE, CAUSES, PENDING_FAMILIES, PENDING_REASONS, TIME_FLAGS, JobState, advise, exit_info,
                                find_cause, node_of, pending_help)

REF = "esm2-3b__esm2-650m__proteingym-score-all__seed1__live__ab12cd34"


def job(state, elapsed="01:00:07", exitcode="0:0", limit="01:00:00", node="g003") -> JobState:
    return JobState(state, elapsed, exitcode, "mpd-train-esm2650m", "", limit, node)


def say(stage, state, **kw):
    kw.setdefault("ref", REF)
    return advise(stage, "1003", state, **kw)


# ---------------------------------------------------------------- TIMEOUT
def test_a_timeout_names_elapsed_and_limit_the_flag_an_example_and_what_the_partition_allows():
    lines = say("train", job("TIMEOUT"), plan={"walltime_s": 3600, "partition_limit_s": 4 * 86400, "walltime_basis": "user"})
    assert lines[0] == "train (job 1003) reached its time limit: it ran 01:00:07 of the 01:00:00 it asked for."
    assert "continues from the latest save" in lines[1]
    assert lines[2] == ("Run your original `mpd submit ...` command again with a larger --train-time (for example --train-time 02:00:00; "
                        "this partition allows up to 4-00:00:00) and --force.")
    assert len(lines) == 3


def test_the_example_is_twice_the_limit_in_whole_hours_and_never_above_the_partition():
    big = say("train", job("TIMEOUT", elapsed="2-12:00:05", limit="2-12:00:00"), plan={"partition_limit_s": 4 * 86400})
    assert "for example --train-time 4-00:00:00;" in big[-1]                          # 5 days would be refused: the cap is the partition
    small = say("train", job("TIMEOUT", limit="00:20:00", elapsed="00:20:03"), plan={"partition_limit_s": 4 * 86400})
    assert "for example --train-time 01:00:00;" in small[-1]


def test_a_timeout_at_the_longest_time_the_partition_allows_says_so():
    train = say("train", job("TIMEOUT", limit="4-00:00:00", elapsed="4-00:00:03"), plan={"partition_limit_s": 4 * 86400})
    assert "It already asked for the longest time this partition allows (4-00:00:00)" in train[-1]
    assert "Run the same command with --force again: every attempt continues from the latest save." in train[-1]
    score = say("score", job("TIMEOUT", limit="4-00:00:00"), plan={"partition_limit_s": 4 * 86400})
    assert "Use more shards (--score-shards) or a faster GPU (--gpu)" in score[-1]


def test_the_time_limit_comes_from_sacct_else_from_plan_json_and_estimates_are_labelled():
    no_limit = say("train", job("TIMEOUT", limit=""), plan={"walltime_s": 7200, "walltime_basis": "estimate"})
    assert "of the 02:00:00 it asked for. That limit was mpd's estimate, not a measurement." in no_limit[0]
    nothing = say("train", job("TIMEOUT", limit=""))
    assert nothing[0] == "train (job 1003) reached its time limit after 01:00:07."
    assert nothing[-1].endswith("larger --train-time (for example --train-time 08:00:00) and --force.")      # no maximum known: none claimed


@pytest.mark.parametrize("stage,flag,resume", [("teacher", "--teacher-time", "starts from the beginning"), ("score", "--score-time", "keeps the shards"),
                                               ("score-base", "--score-time", "keeps the shards"), ("score.1", "--score-time", "keeps the shards")])
def test_each_stage_names_its_own_time_option_and_what_a_new_attempt_repeats(stage, flag, resume):
    lines = say(stage, job("TIMEOUT"), plan={"partition_limit_s": 4 * 86400})
    assert resume in lines[1] and f"larger {flag} " in lines[2]


@pytest.mark.parametrize("stage", ["prepare", "evaluate"])
def test_stages_without_a_time_option_say_so_instead_of_inventing_one(stage):
    lines = say(stage, job("TIMEOUT", limit="04:00:00"), plan={"partition_limit_s": 7 * 86400})
    assert lines[-1] == "This stage has no time option of its own. Report it with `mpd doctor --report` and the end of its log."
    assert not any("--train-time" in x or "--score-time" in x for x in lines)


# ---------------------------------------------------------------- memory
def test_out_of_memory_states_and_exit_137_give_the_gpu_and_host_advice():
    for state in (job("OUT_OF_MEMORY", exitcode="0:125"), job("FAILED", exitcode="137:0"), job("FAILED", exitcode="0:9")):
        lines = say("train", state, plan={"mem_gb": 180})
        assert lines[0].startswith("train (job 1003) ran out of memory")
        text = " ".join(lines)
        assert "smaller student or a larger GPU (--gpu" in text and "More GPUs do not help" in text and "host memory (the job asked for 180G)" in text
        assert "--sbatch-arg=--mem=<more>G" in text and "estimates, not measurements" in text and text.endswith("again with --force.")


def test_a_cuda_out_of_memory_in_the_log_is_the_gpu_and_only_the_gpu():
    lines = say("train", job("FAILED", exitcode="1:0"), err="torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB")
    assert lines[0] == "train (job 1003) ran out of memory." and "host memory" not in lines[1] and "More GPUs do not help" in lines[1]


# ---------------------------------------------------------------- the node
def test_a_failed_node_is_not_the_users_fault_and_is_excluded_if_it_comes_back():
    lines = say("train", job("NODE_FAIL", node="g011"))
    assert lines[0] == "train (job 1003) ended with NODE_FAIL: the node it ran on g011 failed, which is not a problem with your settings."
    assert lines[1] == ("Run your original `mpd submit ...` command again with --force. If it lands on g011 again, add "
                        "--sbatch-arg=--exclude=g011.")
    anonymous = say("train", job("BOOT_FAIL", node=""))
    assert "--exclude" not in " ".join(anonymous) and anonymous[0].endswith("the node it ran failed, which is not a problem with your settings.")


# ---------------------------------------------------------------- FAILED and what the log says
def test_no_usable_gpu_names_the_node_from_sacct_the_script_line_or_the_job_start_line():
    expect = "again with --sbatch-arg=--exclude=g007 and --force."
    from_sacct = say("train", job("FAILED", exitcode="31:0", node="g007"), err="MPD_NO_GPU: torch.cuda.is_available() is False")
    from_script = say("train", job("FAILED", exitcode="31:0", node=""), err="MPD_NO_GPU: no usable GPU on g007; resubmit with --sbatch-arg=--exclude=g007")
    from_start = say("train", job("FAILED", exitcode="31:0", node=""), err="MPD_NO_GPU: torch.cuda.is_available() is False",
                     out="MPD_JOB_START host=g007.grace.hprc.tamu.edu job=1003 stage=train")
    for lines in (from_sacct, from_script, from_start):
        assert lines[0] == "train (job 1003) failed with exit code 31." and lines[1].endswith(expect)
        assert "stopped instead of running on the CPU" in lines[1]
    assert "--exclude=<node>" in say("train", job("FAILED", exitcode="31:0", node=""))[1]                # exit 31 alone is enough to know the cause


def test_a_module_that_did_not_load_points_at_the_environment_record():
    lines = say("prepare", job("FAILED", exitcode="1:0"), err="MPD_MODULE_LOAD_FAILED Anaconda3/2024.02-1", cluster="grace")
    assert "`mpd doctor` shows it (row env-record)" in lines[1] and "bash scripts/setup_env.sh --cluster grace" in lines[1]
    assert lines[1].endswith("Then run your original `mpd submit ...` command again with --force.")


@pytest.mark.parametrize("err", ["weights for esm2-3b are missing (/x/models/esm2-3b). Run on a login node: mpd fetch model esm2-3b --cluster grace",
                                 "OSError: We couldn't connect to 'https://huggingface.co' to load this file; Run on a login node: mpd fetch model esm2-3b --cluster grace",
                                 "huggingface_hub.errors.OfflineModeIsEnabled: Cannot reach https://huggingface.co/api/models: offline mode is enabled."])
def test_missing_weights_and_the_offline_error_point_at_mpd_fetch_model(err):
    lines = say("train", job("FAILED", exitcode="1:0"), err=err)
    assert "mpd fetch model" in lines[1] and "login node" in lines[1] and lines[1].endswith("again with --force.")
    if "mpd fetch model esm2-3b --cluster grace" in err:
        assert "`mpd fetch model esm2-3b --cluster grace`" in lines[1]                    # the command the job printed, not a guess
    else:
        assert "`mpd fetch model <KEY>`" in lines[1]


@pytest.mark.parametrize("hostile", ["mpd fetch model esm2-3b; rm -rf ~", "mpd fetch model $(touch x)", "mpd fetch model a`id`", "mpd fetch model a && b",
                                     "mpd fetch model esm2-3b --home /a b"])
def test_a_command_taken_from_a_log_is_shown_only_if_it_is_plain(hostile):
    lines = say("train", job("FAILED", exitcode="1:0"), err=f"weights for esm2-3b are missing (/x). Run on a login node: {hostile}")
    assert "`mpd fetch model <KEY>`" in lines[1] and "rm -rf" not in " ".join(lines) and "touch" not in " ".join(lines)


def test_a_node_name_is_shown_only_if_it_is_a_plain_name():
    assert node_of(job("FAILED", node="g[001-002]")) == "g[001-002]"
    assert node_of(job("FAILED", node="g1;reboot")) == "" and node_of(None, "MPD_NO_GPU: no usable GPU on $(id);") == ""
    assert "--exclude=<node>" in say("train", job("FAILED", exitcode="31:0", node="a b;c"))[1]


def test_an_unknown_failure_shows_where_to_read_more_and_how_to_continue():
    lines = say("train", job("FAILED", exitcode="1:0"), err="ValueError: something nobody anticipated")
    assert lines[0] == "train (job 1003) failed with exit code 1."
    assert f"`mpd logs {REF} train --err` shows all of it" in lines[1] and "finished stages are kept" in lines[1]


def test_the_cause_comes_from_the_end_of_the_log_of_that_job_only():
    assert find_cause("", "") is None and find_cause("MPD_NO_GPU: x") == "no_gpu" and find_cause("", "MPD_MODULE_LOAD_FAILED x") == "module"
    assert find_cause("CUDA out of memory", "MPD_NO_GPU") == "cuda_oom"                      # stderr first
    assert find_cause("MPD_NO_GPU: a CUDA out of memory in a later line") == "no_gpu"        # then the table's order


# ---------------------------------------------------------------- cancelled and the rest
def test_a_cancelled_campaign_is_summarised_in_two_lines():
    assert campaign.advise_cancelled(["train", "score"]) == ["train, score: cancelled with `mpd cancel`.",
                                                             "Run your original `mpd submit ...` command again to queue the campaign again."]


def test_a_stage_cancelled_by_a_failed_dependency_waits_for_the_other_stage_to_be_fixed():
    lines = say("score", job("CANCELLED"), reason="DependencyNeverSatisfied")
    assert lines == ["score (job 1003) was cancelled because a stage it waited for did not finish.",
                     "Nothing to do for this stage until that one is fixed; then run your original `mpd submit ...` command again with --force."]


def test_a_cancelled_stage_after_mpd_cancel_needs_no_force_otherwise_it_does():
    after_cancel = say("train", job("CANCELLED"), cancelled=True)
    assert after_cancel[1] == "Run your original `mpd submit ...` command again to queue the campaign again."
    by_someone_else = say("train", job("CANCELLED"), cancelled=False)
    assert by_someone_else[1] == "Run your original `mpd submit ...` command again with --force to queue what is missing."
    assert by_someone_else[0].endswith("by `mpd cancel`, by an administrator, or by a limit of the account.")


def test_a_state_without_advice_says_so_and_a_good_state_gets_none():
    assert say("train", job("PREEMPTED"))[0] == "train (job 1003) ended with state PREEMPTED, which mpd has no advice for."
    assert say("train", job("PREEMPTED"))[1].startswith(f"`mpd logs {REF} train --err`")
    for good in ("COMPLETED", "RUNNING", "PENDING", "COMPLETING", "CONFIGURING"):
        assert say("train", job(good)) == [], good
    assert say("train", None) == []


# ---------------------------------------------------------------- pending reasons
@pytest.mark.parametrize("reason", sorted(PENDING_REASONS))
def test_every_known_pending_reason_has_one_sentence_and_fills_its_placeholders(reason):
    text = pending_help(reason, REF)
    assert text == PENDING_REASONS[reason].format(ref=REF) and "\n" not in text and text.endswith(".") and "{" not in text


def test_the_reasons_a_student_meets_first_say_to_wait_or_what_to_do():
    assert pending_help("Priority").endswith("Nothing to do.") and pending_help("Resources").endswith("Nothing to do.")
    assert pending_help("Dependency").endswith("Nothing to do.")
    assert "run your original `mpd submit ...` command again with --force" in pending_help("DependencyNeverSatisfied")
    assert f"`mpd cancel {REF}`, then submit again with a shorter walltime" in pending_help("PartitionTimeLimit", REF)
    assert "--sbatch-arg=--exclude" in pending_help("ReqNodeNotAvail")


def test_limits_of_the_account_and_the_qos_and_the_balance_have_their_own_lines():
    assert "SU balance" in pending_help("AssocGrpBillingMinutes") and "myproject" in pending_help("AssocGrpCPUMinutesLimit")
    assert "balance" in pending_help("the account balance is too low").lower() and "myproject" in pending_help("balance")
    assert pending_help("AssocMaxJobsLimit").startswith("a limit of your account")
    assert pending_help("QOSMaxJobsPerUserLimit").startswith("a limit of its QOS") and pending_help("QOSGrpGRES").startswith("a limit of its QOS")


def test_a_reason_with_a_detail_keeps_the_detail_and_an_unknown_reason_is_printed_as_it_is():
    detailed = pending_help("ReqNodeNotAvail, UnavailableNodes:g001")
    assert detailed.startswith(PENDING_REASONS["ReqNodeNotAvail"]) and detailed.endswith("(Slurm: ReqNodeNotAvail, UnavailableNodes:g001)")
    assert pending_help("(Resources)") == PENDING_REASONS["Resources"]                          # squeue's own parentheses
    assert pending_help("user env retrieval failed requeued held") == "Slurm says: user env retrieval failed requeued held"
    assert pending_help("SomeFutureReason") == "Slurm says: SomeFutureReason"


# ---------------------------------------------------------------- the table is one table
FIXTURE_LOGS = {"no_gpu": "MPD_NO_GPU: torch.cuda.is_available() is False", "module": "MPD_MODULE_LOAD_FAILED X/1",
                "weights": "weights for esm2-3b are missing (/x). Run on a login node: mpd fetch model esm2-3b",
                "offline": "OfflineModeIsEnabled: Cannot reach https://huggingface.co"}


def test_every_sentence_is_used_by_some_case_and_no_case_leaves_a_placeholder(monkeypatch):
    used: set[str] = set()

    class Spy(dict):
        def __getitem__(self, key):
            used.add(key)
            return dict.__getitem__(self, key)

    monkeypatch.setattr(campaign, "ADVICE", Spy(ADVICE))
    big = {"walltime_s": 3600, "partition_limit_s": 4 * 86400, "walltime_basis": "estimate"}
    cases = [("train", job("TIMEOUT"), {"plan": big}), ("train", job("TIMEOUT", limit=""), {}), ("prepare", job("TIMEOUT"), {}),
             ("train", job("TIMEOUT", limit="4-00:00:00"), {"plan": big}), ("score", job("TIMEOUT", limit="4-00:00:00"), {"plan": big}),
             ("teacher", job("TIMEOUT"), {}), ("score", job("TIMEOUT"), {}),
             ("train", job("OUT_OF_MEMORY"), {}), ("train", job("FAILED", exitcode="1:0"), {"err": "CUDA out of memory"}),
             ("train", job("NODE_FAIL"), {}), ("train", job("NODE_FAIL", node=""), {}), ("train", job("FAILED", exitcode="1:0"), {}),
             ("train", job("CANCELLED"), {"reason": "DependencyNeverSatisfied"}), ("train", job("CANCELLED"), {"cancelled": True}),
             ("train", job("CANCELLED"), {}), ("train", job("PREEMPTED"), {})]
    cases += [("train", job("FAILED", exitcode="1:0"), {"err": log}) for log in FIXTURE_LOGS.values()]
    for stage, state, kw in cases:
        for line in say(stage, state, **kw):
            assert "{" not in line and "}" not in line, line
    assert campaign.advise_cancelled(["train"])
    assert set(ADVICE) <= used, f"sentences no case reaches: {sorted(set(ADVICE) - used)}"


def test_every_cause_has_advice_and_every_advice_a_cause():
    assert [name for name, _ in CAUSES if name != "cuda_oom"] == list(CAUSE_ADVICE) and set(FIXTURE_LOGS) == set(CAUSE_ADVICE)
    for name, text in FIXTURE_LOGS.items():
        assert find_cause(text) == name


def test_every_option_the_texts_suggest_exists_in_the_command_line():
    import argparse

    from mpd.cli import _parser
    commands = next(a for a in _parser()._actions if isinstance(a, argparse._SubParsersAction)).choices
    known = {opt for action in commands["submit"]._actions for opt in action.option_strings}
    other_commands = {"--err", "--cluster", "--dry-run", "--report"}               # mpd logs / setup_env.sh / mpd doctor
    texts = [*ADVICE.values(), *CAUSE_ADVICE.values(), *PENDING_REASONS.values(), *(t for _, t in PENDING_FAMILIES)]
    mentioned = {opt for text in texts for opt in re.findall(r"(?<![\w=-])--[a-z][a-z-]*", text)}
    assert mentioned - known - other_commands == set(), mentioned - known
    assert {"--force", "--sbatch-arg", "--score-shards", "--gpu", "--account"} <= mentioned
    assert set(TIME_FLAGS.values()) == {"--train-time", "--teacher-time", "--score-time"} <= known           # filled in as {flag}
    assert set(TIME_FLAGS) == {"train", "teacher", "score", "score-base"}


def test_the_no_gpu_exit_code_is_the_one_the_stages_use():
    from mpd.stages.common import NO_GPU_EXIT
    assert campaign.NO_GPU_EXIT == NO_GPU_EXIT == 31


def test_exit_info_reads_sacct_and_survives_garbage():
    assert exit_info("137:0") == (137, 0) and exit_info("0:15") == (0, 15) and exit_info("") == (0, 0) and exit_info("x:y") == (0, 0)
    assert exit_info("1") == (1, 0)


def test_node_of_prefers_sacct_then_the_script_line_then_the_start_line():
    assert node_of(job("FAILED", node="g1"), "MPD_NO_GPU: no usable GPU on g2;", "MPD_JOB_START host=g3") == "g1"
    assert node_of(job("FAILED", node=""), "MPD_NO_GPU: no usable GPU on g2; resubmit", "MPD_JOB_START host=g3") == "g2"
    assert node_of(None, "", "MPD_JOB_START host=g3.grace.hprc.tamu.edu job=1 stage=train") == "g3" and node_of(None, "", "") == ""
