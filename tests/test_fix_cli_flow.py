"""Repairs in the command-line flow, each with the case that failed: the confirmation prompt of `mpd submit`, a working root from
$MPD_HOME that sits under /home, a submit that was cut off, the copy-out command of `mpd results`, and the row that stands for many
units. Everything runs on the CPU against the fake Slurm; nothing here is a claim about Grace.
"""

import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fx_cli import (FAKE_ACCOUNT, FAKE_USER, cli_env, ready_home, recipe, result_rows, run_cli, write_adapter,  # noqa: F401
                    write_summary)
from fx_slurm import fake_slurm, stub_plan, write_stub_campaign  # noqa: F401

from mpd import cli, results
from mpd.slurm.submit import submit as real_submit

pytestmark = pytest.mark.usefixtures("cli_env")

SRC = str(Path(__file__).parents[1] / "src")
RSYNC = shutil.which("rsync")                 # looked up before the tests narrow PATH
STAGES = ["prepare", "teacher", "train", "score-base", "score", "evaluate"]


@pytest.fixture
def world(cli_env, monkeypatch, fake_slurm):
    """A ready Grace home, the fake Slurm, and a `submit` whose waits are virtual (the CLI would sleep for real)."""
    ready_home(cli_env, monkeypatch)
    monkeypatch.setattr(cli, "submit", lambda plan, scripts, **kw: real_submit(plan, scripts, sleep=fake_slurm.sleep, clock=fake_slurm.clock, **kw))
    return cli_env, fake_slurm


def command(env, *extra) -> list[str]:
    """The student's `mpd submit ...` command."""
    return ["submit", *recipe(env.home, "--account", FAKE_ACCOUNT, *extra), "--yes"]


def snapshot(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


# ---------------------------------------------------------------- F8: the prompt of `mpd submit`
def at_a_terminal(monkeypatch, answer):
    """stdin is a terminal and the student types `answer`; None is end of input (Ctrl-D)."""
    def ask(prompt=""):
        sys.stdout.write(prompt)                    # what input() does before it reads
        if answer is None:
            raise EOFError
        return answer
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("builtins.input", ask)


@pytest.mark.parametrize("answer", ["y", "Y", "yes", "YES", " Yes ", "yEs"])
def test_the_prompt_accepts_y_and_yes_in_any_case(capsys, world, monkeypatch, answer):
    env, fs = world
    at_a_terminal(monkeypatch, answer)
    rc, out, err = run_cli(capsys, "submit", *recipe(env.home))
    assert rc == 0, (out, err)
    assert len(fs.submitted()) == 6


@pytest.mark.parametrize("answer", ["n", "N", "no", "", "ye", "yep", "sure", "y es"])
def test_any_other_answer_declines_and_says_why(capsys, world, monkeypatch, answer):
    env, fs = world
    at_a_terminal(monkeypatch, answer)
    rc, out, err = run_cli(capsys, "submit", *recipe(env.home))
    (root,) = (env.home / "runs").iterdir()
    assert rc == 1 and err == ""
    assert "not submitted: the answer was not y or yes." in out
    assert f"The scripts stay in {root / 'scripts'}." in out and "Run the same command to be asked again." in out
    assert fs.submitted() == []


def test_end_of_input_at_the_prompt_is_a_plain_refusal_and_asking_again_works(capsys, world, monkeypatch):
    env, fs = world
    at_a_terminal(monkeypatch, None)
    rc, out, err = run_cli(capsys, "submit", *recipe(env.home))
    assert rc == 1
    assert "[y/N] \nnot submitted: no answer was given." in out          # the open prompt line is ended first
    assert err == "" and "EOFError" not in out + err and "report" not in out + err         # not the 'unexpected internal error' text
    assert fs.submitted() == []
    at_a_terminal(monkeypatch, "yes")                                      # the message says the same command asks again
    rc, out, err = run_cli(capsys, "submit", *recipe(env.home))
    assert rc == 0, (out, err)
    assert len(fs.submitted()) == 6


# ---------------------------------------------------------------- F5: a root from $MPD_HOME under /home
UNDER_HOME = f"/home/{FAKE_USER}/mpd"
PLAN = ["plan", "--teacher", "esm2-3b", "--student", "esm2-650m", "--train-data", "proteingym", "--method", "score-all",
        "--eval-data", "proteingym"]
COMMANDS = [["fetch", "model", "esm2-8m"], ["fetch", "model", "--list"], ["fetch", "proteingym"], ["status"], ["status", "some-id"],
            ["logs", "some-id"], ["results", "some-id"], ["cancel", "some-id"], ["assays"], ["models"]]


@pytest.mark.parametrize("argv", COMMANDS, ids=lambda a: " ".join(a))
def test_a_root_from_mpd_home_under_slash_home_gets_the_message_the_planner_gives(capsys, monkeypatch, argv):
    monkeypatch.setenv("MPD_CLUSTER", "grace")
    monkeypatch.setenv("MPD_HOME", UNDER_HOME)
    rc, _, wanted = run_cli(capsys, *PLAN)
    assert rc == 2 and f"{UNDER_HOME} is under /home" in wanted and "Use scratch: export MPD_HOME=$SCRATCH/mpd" in wanted
    rc, out, err = run_cli(capsys, *argv)
    assert rc == 2 and err == wanted, (out, err)
    assert "give --home" not in err


def test_without_a_named_root_the_hint_still_says_to_give_one(capsys, cli_env, monkeypatch):
    monkeypatch.setenv("MPD_CLUSTER", "grace")                  # $SCRATCH is unset here: the profile cannot name a root, and nobody did
    rc, out, _ = run_cli(capsys, "models")
    assert rc == 0 and "add --cluster or --home" in out
    rc, _, err = run_cli(capsys, "status")
    assert rc == 2 and "give --home" in err
    _, _, err = run_cli(capsys, "fetch", "model", "esm2-8m")
    assert "give --home" in err and "MPD_MODELS_DIR" in err


def test_a_models_dir_keeps_fetch_model_working_when_mpd_home_is_under_slash_home(capsys, cli_env, monkeypatch):
    monkeypatch.setenv("MPD_CLUSTER", "grace")
    monkeypatch.setenv("MPD_HOME", UNDER_HOME)
    monkeypatch.setenv("MPD_MODELS_DIR", str(cli_env.tmp / "models"))
    rc, out, err = run_cli(capsys, "fetch", "model", "--list")
    assert rc == 0, err
    assert out.startswith("key")


# ---------------------------------------------------------------- G2: a submit that was cut off keeps the marker `submitting`
# The child process queues two jobs and is then killed the way a dropped ssh connection kills it: no exception, no clean-up.
KILL_AFTER_TWO = """
import os, signal, sys
from mpd import cli
from mpd.slurm.submit import submit as real_submit

def cut_off(plan, scripts, **kw):
    accepted = []
    def echo(line):
        print(line, flush=True)
        if line.startswith("submitted "):
            accepted.append(line)
        if len(accepted) == 2:
            os.kill(os.getpid(), signal.SIGKILL)
    return real_submit(plan, scripts, echo=echo, **kw)

cli.submit = cut_off
sys.exit(cli.main(sys.argv[1:]))
"""


@pytest.fixture
def cut_off(world):
    env, fs = world
    done = subprocess.run([sys.executable, "-s", "-c", KILL_AFTER_TWO, *command(env)], capture_output=True, text=True, timeout=180,
                          env={**os.environ, "PYTHONPATH": SRC, "PYTHONDONTWRITEBYTECODE": "1"})
    assert done.returncode == -signal.SIGKILL, (done.returncode, done.stdout, done.stderr)
    (root,) = (env.home / "runs").iterdir()
    marker = json.loads((root / "SUBMITTED.json").read_text())
    assert marker["state"] == "submitting" and list(marker["jobs"]) == ["prepare", "teacher"]
    return env, fs, root, marker["jobs"]


def test_a_cut_off_submit_is_not_called_already_submitted(capsys, cut_off):
    env, fs, root, jobs = cut_off
    before, calls = snapshot(root), len(fs.calls("sbatch"))
    for extra in ([], ["--dry-run"]):
        rc, out, err = run_cli(capsys, *command(env), *extra)
        flat = " ".join(err.split())
        assert rc == 2 and "did not finish" in flat and "was already submitted" not in flat
        assert f"jobs queued so far: prepare {jobs['prepare']}, teacher {jobs['teacher']}" in flat
        assert "wait for it" in flat and "ssh connection dropped" in flat
        assert f"mpd status {root.name}" in flat and f"mpd cancel {root.name}" in flat
        assert "add --force to this command to finish it: the stages not yet queued are submitted and the others are kept" in flat
    assert snapshot(root) == before and len(fs.calls("sbatch")) == calls         # nothing was written and nothing was queued


def test_status_of_a_cut_off_submit_says_to_repeat_the_command_with_force(capsys, cut_off, monkeypatch):
    env, fs, root, jobs = cut_off
    monkeypatch.setenv("MPD_HOME", str(env.home))
    rc, out, err = run_cli(capsys, "status", root.name)
    assert rc == 0, err
    steps = " ".join(out.split("next:")[1].split())
    assert "state      submitting" in out
    assert "again later" not in steps                                        # nobody should wait for stages that were never queued
    assert "has not finished" in steps and "the stages missing from the table are not queued yet" in steps
    assert "repeat your mpd submit command with --force to queue them; the stages already queued are kept" in steps
    assert f"mpd cancel {root.name}" in steps


def test_the_same_command_with_force_queues_the_missing_stages_and_keeps_the_others(capsys, cut_off):
    env, fs, root, jobs = cut_off
    rc, out, err = run_cli(capsys, *command(env), "--force")
    assert rc == 0, (out, err)
    now = json.loads((root / "SUBMITTED.json").read_text())
    assert now["state"] == "submitted" and list(now["jobs"]) == STAGES
    assert {stage: now["jobs"][stage] for stage in jobs} == jobs               # prepare and teacher are the jobs of the cut-off run
    assert [job["stage"] for job in fs.submitted()] == STAGES                  # every stage was queued exactly once


# ---------------------------------------------------------------- G3 and F7: the copy-out command of `mpd results` on Grace
def units_summary(units, *, nan_units=(), dataset="venusmuthub") -> list[dict]:
    """The base rows once, then the trained rows of each unit (base 0.25, distilled 0.30); `nan_units` have no defined Spearman."""
    rows = [r for r in result_rows(dataset=dataset, unit="x", assays=1, base=(0.25, 0.25), tuned=(0.30, 0.30)) if r["unit"] == "base"]
    for unit in units:
        for r in result_rows(dataset=dataset, unit=unit, assays=1, base=(0.25, 0.25), tuned=(0.30, 0.30)):
            if r["unit"] == unit:
                if unit in nan_units:
                    r["macro_spearman"] = r["delta_vs_base"] = math.nan
                rows.append(r)
    return rows


def grace_campaign(tmp, units=("all",), *, step=2103, adapters=True, **kw):
    plan = stub_plan(tmp)
    write_stub_campaign(plan)
    write_summary(plan.campaign_dir, units_summary(units, **kw))
    if adapters:
        for unit in units:
            write_adapter(plan.campaign_dir, unit, step)
    return plan.campaign_dir


def printed_rsync(out: str) -> tuple[list[str], str]:
    """(the names inside the braces, the line): the one rsync line `mpd results` prints."""
    lines = [ln.strip() for ln in out.splitlines() if ln.lstrip().startswith("rsync ")]
    assert len(lines) == 1, out
    m = re.fullmatch(r"rsync -av <netid>@grace\.hprc\.tamu\.edu:(\S+)/\{([^{}]*)\} \./(\S+)/", lines[0])
    assert m, lines[0]
    return m[2].split(","), lines[0]


def covered(names: list[str], relative: str) -> bool:
    return any(relative == n or relative.startswith(n + "/") for n in names)


def test_the_printed_command_includes_the_adapter_it_lists(capsys, cli_env):
    root = grace_campaign(cli_env.tmp)
    rc, out, err = run_cli(capsys, "results", str(root))
    assert rc == 0, err
    names, line = printed_rsync(out)
    assert {"evaluate", "spec.json", "resolved.json"} <= set(names)
    assert covered(names, "train/all/checkpoint_02103")
    assert f"{root}/train/all/checkpoint_02103" in out                         # the adapter the same output lists
    assert "train/ holds the adapter of every unit" in out


def test_many_units_keep_the_copy_command_to_one_short_line_that_still_covers_every_adapter(capsys, cli_env):
    units = [f"ASSAY_{i:02d}" for i in range(40)]
    root = grace_campaign(cli_env.tmp, units, step=50)
    rc, out, err = run_cli(capsys, "results", str(root))
    assert rc == 0, err
    names, line = printed_rsync(out)
    assert len(names) <= 4 and "ASSAY_" not in line
    assert all(covered(names, f"train/{unit}/checkpoint_00050") for unit in units)
    assert "... and 35 more under" in out                                       # the list of adapters is still cut at five
    assert len(out.split("not backed up")[1].strip().splitlines()) <= 3


def test_without_an_adapter_the_command_copies_only_the_numbers_and_the_recipe(capsys, cli_env):
    root = grace_campaign(cli_env.tmp, adapters=False)
    rc, out, err = run_cli(capsys, "results", str(root))
    assert rc == 0, err
    names, _ = printed_rsync(out)
    assert names == ["evaluate", "spec.json", "resolved.json"] and "train/ holds" not in out and "adapters: none found" in out


@pytest.mark.skipif(RSYNC is None, reason="rsync is not installed")
def test_the_printed_command_copies_the_numbers_the_recipe_and_the_adapter(capsys, cli_env):
    """Run the printed line for real, with the host part dropped so that rsync copies from the local campaign folder."""
    root = grace_campaign(cli_env.tmp, ["all", "other"])
    (root / "logs").mkdir(exist_ok=True)
    (root / "logs" / "mpd-train_1.out").write_text("the bulk stays behind\n", encoding="utf-8")
    rc, out, err = run_cli(capsys, "results", str(root))
    assert rc == 0, err
    _, line = printed_rsync(out)
    laptop = cli_env.tmp / "laptop"
    laptop.mkdir()
    local = line.replace("<netid>@grace.hprc.tamu.edu:", "").replace("rsync", RSYNC, 1)
    done = subprocess.run(["bash", "-c", local], cwd=laptop, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    got = sorted(str(p.relative_to(laptop / root.name)) for p in (laptop / root.name).rglob("*") if p.is_file())
    assert {"evaluate/summary.json", "spec.json", "resolved.json"} <= set(got)
    assert {"train/all/checkpoint_02103/adapter_config.json", "train/other/checkpoint_02103/adapter_config.json"} <= set(got)
    assert not any(p.startswith(("logs/", "scripts/", "tasks/")) for p in got)


# ---------------------------------------------------------------- ml-05: the row for many units
def many_rows(capsys, tmp, units, **kw) -> tuple[str, list[str]]:
    root = grace_campaign(tmp, units, step=100, **kw)
    rc, out, err = run_cli(capsys, "results", str(root))
    assert rc == 0, err
    return out, [ln for ln in out.splitlines() if "mean of" in ln]


def test_one_unit_without_a_spearman_does_not_turn_the_row_for_many_units_into_nan(capsys, cli_env):
    units = [f"ASSAY_{i:02d}" for i in range(13)]
    out, rows = many_rows(capsys, cli_env.tmp, units, nan_units={"ASSAY_04"})
    assert len(rows) == 2                                                        # delta and dual
    for row in rows:
        assert "mean of 12 of 13 units (12 improved; 1 left out)" in row
        assert row.split()[-3:] == ["0.2500", "0.3000", "+0.0500"] and "nan" not in row
    assert "A unit is left out of a mean when its base or distilled Spearman is nan" in out
    assert "per_assay.csv has every unit" in out


def test_with_every_unit_defined_the_row_is_as_before_and_says_nothing_is_left_out(capsys, cli_env):
    units = [f"ASSAY_{i:02d}" for i in range(15)]
    out, rows = many_rows(capsys, cli_env.tmp, units)
    assert all("mean of 15 units (15 improved)" in row and row.split()[-3:] == ["0.2500", "0.3000", "+0.0500"] for row in rows)
    assert "left out" not in out


def test_units_with_odd_numbers_are_left_out_from_all_three_means_alike(tmp_path):
    units = [f"U{i:02d}" for i in range(14)]
    rows = units_summary(units)
    for r in rows:
        if r["unit"] == "U00":
            r["macro_spearman"] = None                                           # a null in the file
        elif r["unit"] == "U01":
            del r["delta_vs_base"]
        elif r["unit"] == "U02":
            r["base_macro_same_assays"] = math.nan
            r["delta_vs_base"] = math.nan
        elif r["unit"] == "U03":
            r["macro_spearman"] = math.inf
        elif r["unit"] == "U04":
            r["base_macro_same_assays"] = 0.45                                   # defined and kept: it must change the base mean
            r["macro_spearman"], r["delta_vs_base"] = 0.40, -0.05
    text = results.format_results({"results": rows, "primary_readout": "dual"}, tmp_path / "c")
    row = next(ln for ln in text.splitlines() if "mean of" in ln)
    assert "mean of 10 of 14 units (9 improved; 4 left out)" in row
    base, distilled, delta = (float(x) for x in row.split()[-3:])
    assert base == pytest.approx((9 * 0.25 + 0.45) / 10, abs=1e-4) and distilled == pytest.approx((9 * 0.30 + 0.40) / 10, abs=1e-4)
    assert delta == pytest.approx(distilled - base, abs=1e-4)                    # one set of units, so the columns agree


def test_when_no_unit_is_defined_the_row_says_nan_and_how_many_were_left_out(tmp_path):
    rows = units_summary([f"U{i:02d}" for i in range(13)], nan_units={f"U{i:02d}" for i in range(13)})
    text = results.format_results({"results": rows}, tmp_path / "c")
    assert "mean of 0 of 13 units (0 improved; 13 left out)" in text
    assert next(ln for ln in text.splitlines() if "mean of" in ln).split()[-3:] == ["nan", "nan", "nan"]


def test_twelve_units_or_fewer_still_get_a_row_each(tmp_path):
    text = results.format_results({"results": units_summary([f"U{i:02d}" for i in range(12)], nan_units={"U03"})}, tmp_path / "c")
    assert "mean of" not in text and "left out" not in text
    assert next(ln for ln in text.splitlines() if ln.split()[:3] == ["venusmuthub", "delta", "U03"]).split()[-3:] == ["0.2500", "nan", "nan"]
