"""The `mpd` command line: every documented shape parses, bad input is a one-line error, plan and submit follow the contract."""

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from fx_cli import (FAKE_ACCOUNT, cli_env, override_weights, ready_home, recipe, run_cli, write_proteingym,  # noqa: F401
                    write_venus)
from fx_slurm import INVALID_ACCOUNT, fake_slurm  # noqa: F401

from mpd import cli, registry
from mpd.cli import _parser, main
from mpd.data import models
from mpd.data.fetch import TimeBudgetReached
from mpd.slurm import campaign

pytestmark = pytest.mark.usefixtures("cli_env")

RECIPE = ["--teacher", "esm2-3b", "--student", "esm2-650m", "--train-data", "proteingym", "--method", "score-all",
          "--eval-data", "proteingym", "venusmuthub"]


def digest(root: Path) -> dict[str, str]:
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.rglob("*")) if p.is_file()}


# ---------------------------------------------------------------- every documented command shape parses
SHAPES = [
    ["plan", *RECIPE, "--cluster", "grace", "--account", FAKE_ACCOUNT],
    ["plan", *RECIPE, "--cluster", "grace", "--env", "A=1", "--env", "B=2", "--module", "GCC/12.3.0", "--sbatch-arg=--exclude=g006",
     "--allow-unsupported-gpu", "--max-su", "50000", "--train-gpus", "1", "--score-shards", "proteingym=2", "--array-throttle", "4"],
    ["submit", *RECIPE, "--cluster", "grace", "--dry-run", "--yes", "--force", "--max-su", "500"],
    ["status"], ["status", "ID"], ["status", "ID", "--home", "/x"], ["logs", "ID"], ["logs", "ID", "train", "--err", "--tail", "20"],
    ["cancel", "ID"], ["results", "ID"],
    ["doctor"], ["doctor", "--cluster", "grace", "--home", "/x", "--deep", "--verify", "--report"],
    ["fetch"], ["fetch", "--list"], ["fetch", "proteingym", "--cluster", "grace"], ["fetch", "venusmuthub", "--home", "/x"],
    ["fetch", "venus-structures", "--from-dir", "/x", "--no-unpack", "--zenodo-record", "1"],
    ["fetch", "foldseek"], ["fetch", "foldseek", "--from-file", "/x.tar.gz"],
    ["fetch", "model", "esm2-650m", "esm2-3b", "--cluster", "grace"], ["fetch", "--cluster", "grace", "model", "esm2-650m"],
    ["fetch", "model", "--all"], ["fetch", "model", "--list", "--cluster", "grace"],
    ["fetch", "model", "esm2-8m", "--models-dir", "/x", "--from-dir", "/y", "--verify", "--force", "--dry-run", "--time-budget", "30"],
    ["models"], ["models", "--cluster", "grace"], ["clusters"], ["assays"], ["assays", "--dataset", "proteingym", "--grep", "ENV"],
    ["--debug", "models"], ["models", "--debug"],
]


@pytest.mark.parametrize("argv", SHAPES, ids=lambda a: " ".join(a)[:70])
def test_every_documented_command_shape_parses(argv):
    _parser().parse_args(argv)


def test_keys_on_both_sides_of_an_option_do_not_parse():
    with pytest.raises(SystemExit):
        _parser().parse_args(["fetch", "model", "esm2-8m", "--cluster", "grace", "esm2-3b"])


def test_fetch_model_keys_and_source_end_up_in_the_right_fields():
    a = _parser().parse_args(["fetch", "--cluster", "grace", "model", "esm2-650m", "esm2-3b"])
    assert (a.source, a.keys, a.cluster) == ("model", ["esm2-650m", "esm2-3b"], "grace")
    a = _parser().parse_args(["fetch", "proteingym", "--cluster", "grace"])
    assert (a.source, a.keys) == ("proteingym", [])


def test_the_login_node_commands_never_import_torch(cli_env):
    src = str(Path(__file__).parents[1] / "src")
    home = cli_env.home
    code = f"""
import io, sys, contextlib
sys.path.insert(0, {src!r})
from mpd.cli import main
runs = [["models"], ["clusters"], ["fetch", "--list"], ["fetch", "model", "--list", "--home", {str(home)!r}],
        ["doctor", "--cluster", "grace", "--home", {str(home)!r}], ["status", "--home", {str(home)!r}], ["assays", "--home", {str(home)!r}],
        ["plan", "--cluster", "grace", "--home", {str(home)!r}, "--teacher", "esm2-3b", "--student", "esm2-650m", "--train-data",
         "proteingym", "--method", "score-all", "--eval-data", "proteingym"],
        ["submit", "--dry-run", "--cluster", "grace", "--home", {str(home)!r}, "--teacher", "esm2-3b", "--student", "esm2-650m",
         "--train-data", "proteingym", "--method", "score-all", "--eval-data", "proteingym"]]
for argv in runs:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        main(argv)
assert "torch" not in sys.modules, "torch was imported by a login-node command"
"""
    env = {"PATH": str(cli_env.bin), "HOME": str(cli_env.userhome), "USER": "jdoe42"}
    out = subprocess.run([sys.executable, "-s", "-c", code], env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


def test_a_closed_pipe_is_not_an_error(cli_env):
    src = str(Path(__file__).parents[1] / "src")
    proc = subprocess.Popen([sys.executable, "-s", "-m", "mpd", "models"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            env={"PYTHONPATH": src, "PATH": str(cli_env.bin)})
    proc.stdout.close()                                          # the reader is gone before the first line is written (`mpd models | head`)
    err = proc.stderr.read()
    assert proc.wait() == 0 and err == b""


def test_version_and_python_dash_m(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0 and capsys.readouterr().out.startswith("mpd ")
    src = str(Path(__file__).parents[1] / "src")
    out = subprocess.run([sys.executable, "-s", "-m", "mpd", "--version"], capture_output=True, text=True,
                         env={"PYTHONPATH": src, "PATH": "/usr/bin:/bin"})
    assert out.returncode == 0 and out.stdout.startswith("mpd "), out.stderr


def test_help_texts_say_what_is_true(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    top = capsys.readouterr().out
    assert "mpd doctor" in top and "tested on CPU with fake Slurm tools" in top
    for command in ("plan", "submit", "status", "logs", "cancel", "results", "fetch", "models", "clusters", "assays", "doctor"):
        assert command in top
    with pytest.raises(SystemExit):
        main(["plan", "--help"])
    plan = capsys.readouterr().out
    assert "12-digit" not in plan and "reproduce the paper" not in plan
    assert "myproject" in plan and "MPD_ACCOUNT" in plan and "--sbatch-arg=--exclude" in plan and "--max-su" in plan


# ---------------------------------------------------------------- bad input is a one-line error
@pytest.mark.parametrize("extra,fragment", [
    (["--cluster", "nope"], "unknown cluster 'nope'"),
    (["--train-time", "8h"], "--train-time"),
    (["--train-time", "0"], "--train-time"),
    (["--score-shards", "bad"], "--score-shards"),
    (["--score-shards", "proteingym=0"], "--score-shards"),
    (["--assays-file", "missing.txt"], "cannot read --assays-file"),
    (["--env", "NOEQUALS"], "NAME=VALUE"),
    (["--env", "A=1", "--env", "A=2"], "twice"),
    (["--sbatch-arg=--output=elsewhere"], "--sbatch-arg"),
    (["--sbatch-arg", "exclude=g006"], "--sbatch-arg"),
    (["--module", "bad module"], "--module"),
    (["--account", "a b"], "--account"),
    (["--gpu", "h100"], "no GPU type"),
    (["--gpu", "t4"], "bfloat16"),
])
def test_bad_input_prints_error_and_exits_2_without_a_traceback(capsys, cli_env, extra, fragment):
    argv = ["plan", *recipe(cli_env.home)]
    argv = [a for a in argv if a != "--cluster" and a != "grace"] if "--cluster" in extra else argv
    rc, out, err = run_cli(capsys, *argv, *extra)
    assert rc == 2, (out, err)
    assert err.startswith("error: ") and fragment in err and "Traceback" not in err
    assert "was not expected" not in err


@pytest.mark.parametrize("flags", [["--lr", "0"], ["--lr", "-1"], ["--kl-weight", "-1"], ["--array-throttle", "0"],
                                   ["--updates", "0"], ["--global-batch", "x"], ["--max-su", "0"]])
def test_numbers_are_checked_by_the_parser(capsys, cli_env, flags):
    rc, _, err = run_cli(capsys, "plan", *recipe(cli_env.home), *flags)
    assert rc == 2 and "error: argument" in err


def test_missing_required_options_and_unknown_commands_are_usage_errors(capsys, cli_env):
    assert run_cli(capsys, "plan")[0] == 2
    assert run_cli(capsys, "frobnicate")[0] == 2
    rc, _, err = run_cli(capsys, "plan", *RECIPE)
    assert rc == 2 and "--cluster is required" in err and "grace" in err


def test_debug_shows_the_traceback_and_an_unexpected_error_asks_for_it(capsys, cli_env, monkeypatch):
    with pytest.raises(Exception, match="unknown cluster"):
        main(["plan", *recipe(cli_env.home, cluster="nope"), "--debug"])
    monkeypatch.setattr(cli, "make_plan", lambda *a, **k: {}["boom"])
    rc, _, err = run_cli(capsys, "plan", *recipe(cli_env.home))
    assert rc == 2 and "KeyError was not expected" in err and "--debug" in err


def test_ctrl_c_is_a_quiet_exit_130(capsys, cli_env, monkeypatch):
    def interrupt(*a, **k):
        raise KeyboardInterrupt
    monkeypatch.setattr(cli, "make_plan", interrupt)
    rc, _, err = run_cli(capsys, "plan", *recipe(cli_env.home))
    assert rc == 130 and "Traceback" not in err


def test_a_home_under_slash_home_is_refused_with_the_canonical_text(capsys, cli_env):
    rc, _, err = run_cli(capsys, "plan", *recipe("/home/jdoe42/mpd"))
    assert rc == 2
    assert "/home/jdoe42/mpd is under /home (10 GB / 10,000 files on Grace). Use scratch: export MPD_HOME=$SCRATCH/mpd" in err


# ---------------------------------------------------------------- plan
def test_plan_on_an_empty_home_exits_0_names_every_gap_and_writes_nothing(capsys, cli_env):
    rc, out, err = run_cli(capsys, "plan", *recipe(cli_env.home, "--account", FAKE_ACCOUNT))
    assert rc == 0, err
    assert "estimated upper bound" in out and "NOT READY" in out.splitlines()[-1]
    assert f"no Python environment at {cli_env.home / 'env'}. Run: bash scripts/setup_env.sh --cluster grace" in out
    assert f"ProteinGym data not found under {cli_env.home / 'data' / 'proteingym-v1.3'}. Run on a login node: mpd fetch proteingym --cluster grace" in out
    assert f"weights for esm2-3b are missing ({cli_env.home / 'models' / 'esm2-3b'})" in out
    assert not cli_env.home.exists() and not list(cli_env.userhome.iterdir())


def test_plan_on_a_ready_home_says_ready(capsys, cli_env, monkeypatch):
    ready_home(cli_env, monkeypatch)
    rc, out, _ = run_cli(capsys, "plan", *recipe(cli_env.home, "--account", FAKE_ACCOUNT))
    assert rc == 0 and "[FAIL]" not in out and out.splitlines()[-1].startswith("READY")
    assert not (cli_env.home / "runs").exists()


def test_plan_warns_when_the_estimate_is_above_the_cost_guard(capsys, cli_env, monkeypatch):
    ready_home(cli_env, monkeypatch)
    rc, out, _ = run_cli(capsys, "plan", *recipe(cli_env.home), "--max-su", "100")
    assert rc == 0 and "[warn] cost" in out and "above --max-su 100" in out


def test_plan_reads_the_account_from_the_environment(capsys, cli_env, monkeypatch):
    monkeypatch.setenv("MPD_ACCOUNT", FAKE_ACCOUNT)
    ready_home(cli_env, monkeypatch)
    rc, out, _ = run_cli(capsys, "submit", *recipe(cli_env.home), "--dry-run")
    assert rc == 0
    script = next((cli_env.home / "runs").glob("*/scripts/train.sbatch")).read_text()
    assert f"#SBATCH --account={FAKE_ACCOUNT}" in script


# ---------------------------------------------------------------- submit --dry-run
def test_dry_run_on_an_empty_home_exits_0_says_not_ready_and_writes_the_scripts(capsys, cli_env):
    rc, out, err = run_cli(capsys, "submit", *recipe(cli_env.home), "--dry-run")
    assert rc == 0, err
    assert "NOT READY" in out and "dry run: nothing submitted" in out
    campaign_dir = next((cli_env.home / "runs").iterdir())
    assert (campaign_dir / "scripts" / "prepare.sbatch").is_file() and not (campaign_dir / "SUBMITTED.json").exists()


def test_job_script_options_reach_the_scripts(capsys, cli_env, monkeypatch):
    ready_home(cli_env, monkeypatch)
    rc, out, err = run_cli(capsys, "submit", *recipe(cli_env.home), "--dry-run", "--account", FAKE_ACCOUNT, "--env", "MY_FLAG=on",
                           "--module", "GCC/12.3.0", "--sbatch-arg=--exclude=g006")
    assert rc == 0, err
    script = next((cli_env.home / "runs").glob("*/scripts/train.sbatch")).read_text()
    assert "MY_FLAG=on" in script and "GCC/12.3.0" in script and "#SBATCH --exclude=g006" in script


def test_an_unsupported_gpu_is_refused_unless_allowed(capsys, cli_env):
    small = ["--cluster", "grace", "--home", str(cli_env.home), "--teacher", "esm2-35m", "--student", "esm2-35m", "--train-data",
             "proteingym", "--method", "score-all", "--eval-data", "proteingym", "--gpu", "t4"]
    rc, _, err = run_cli(capsys, "plan", *small)
    assert rc == 2 and "--gpu a100" in err and "bfloat16" in err
    rc, out, err = run_cli(capsys, "submit", *small, "--dry-run", "--allow-unsupported-gpu")
    assert rc == 0, err
    assert "MPD_DTYPE=float32" in next((cli_env.home / "runs").glob("*/scripts/train.sbatch")).read_text()


def test_relative_paths_become_absolute_before_they_reach_a_script(capsys, cli_env, monkeypatch):
    monkeypatch.chdir(cli_env.tmp)
    rc, _, err = run_cli(capsys, "submit", *recipe("rel/mpd"), "--dry-run")
    assert rc == 0, err
    text = next((cli_env.tmp / "rel" / "mpd" / "runs").glob("*/scripts/prepare.sbatch")).read_text()
    assert f"cd {cli_env.tmp / 'rel' / 'mpd' / 'runs'}" in text and "cd rel/" not in text


# ---------------------------------------------------------------- real submit
def test_real_submit_refuses_when_environment_data_and_weights_are_missing(capsys, cli_env):
    rc, out, err = run_cli(capsys, "submit", *recipe(cli_env.home), "--yes")
    assert rc == 2
    assert f"no Python environment at {cli_env.home / 'env'}. Run: bash scripts/setup_env.sh --cluster grace" in err
    assert "ProteinGym data not found under" in err and "weights for esm2-3b are missing" in err
    assert "sbatch not found on PATH" in err                                    # no Slurm in this shell: also a blocker
    assert "NOT READY" in out
    assert not (cli_env.home / "runs").exists()                                  # refused before anything was written


def test_real_submit_without_yes_in_a_non_interactive_shell_is_refused(capsys, cli_env, monkeypatch, fake_slurm):
    ready_home(cli_env, monkeypatch)
    rc, out, err = run_cli(capsys, "submit", *recipe(cli_env.home))
    assert rc == 2 and "--yes" in err
    assert fake_slurm.submitted() == []
    assert not next((cli_env.home / "runs").iterdir()).joinpath("SUBMITTED.json").exists()


def test_the_cost_guard_stops_a_real_submit_and_not_a_dry_run(capsys, cli_env, monkeypatch, fake_slurm):
    ready_home(cli_env, monkeypatch)
    rc, out, err = run_cli(capsys, "submit", *recipe(cli_env.home), "--yes", "--max-su", "100")
    assert rc == 2 and "above --max-su 100" in err and "Requested SUs" in err and "--max-su" in err
    assert fake_slurm.submitted() == [] and not (cli_env.home / "runs").exists()
    rc, out, _ = run_cli(capsys, "submit", *recipe(cli_env.home), "--dry-run", "--max-su", "100")
    assert rc == 0 and "[FAIL] cost" in out and "NOT READY" in out


def test_declining_the_prompt_submits_nothing_and_accepting_submits(capsys, cli_env, monkeypatch, fake_slurm):
    ready_home(cli_env, monkeypatch)
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    rc, out, _ = run_cli(capsys, "submit", *recipe(cli_env.home))
    assert rc == 1 and "not submitted" in out and fake_slurm.submitted() == []
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")
    rc, out, err = run_cli(capsys, "submit", *recipe(cli_env.home))
    assert rc == 0, err
    assert len(fake_slurm.submitted()) == 6


def test_submit_end_to_end_against_fake_slurm(capsys, cli_env, monkeypatch, fake_slurm):
    ready_home(cli_env, monkeypatch)
    rc, out, err = run_cli(capsys, "submit", *recipe(cli_env.home, "--account", FAKE_ACCOUNT), "--yes")
    assert rc == 0, (out, err)
    jobs = fake_slurm.submitted()
    assert [j["stage"] for j in jobs] == ["prepare", "teacher", "train", "score-base", "score", "evaluate"]
    assert fake_slurm.test_only_calls(), "the scripts were not test-submitted before queueing"
    assert "mpd status" in out and "mpd cancel" in out and "mpd results" in out
    campaign_dir = next((cli_env.home / "runs").iterdir())
    assert json.loads((campaign_dir / "SUBMITTED.json").read_text())["state"] == "submitted"
    assert any(ln.startswith("[ok]") and "sbatch --test-only" in ln for ln in out.splitlines())


def test_submitting_twice_is_refused_before_anything_is_written(capsys, cli_env, monkeypatch, fake_slurm):
    ready_home(cli_env, monkeypatch)
    args = ["submit", *recipe(cli_env.home), "--yes"]
    assert run_cli(capsys, *args)[0] == 0
    campaign_dir = next((cli_env.home / "runs").iterdir())
    before, calls = digest(campaign_dir), len(fake_slurm.calls("sbatch"))
    for extra in ([], ["--dry-run"]):
        rc, out, err = run_cli(capsys, *args, *extra)
        assert rc == 2 and "was already submitted" in err and campaign_dir.name in err
        assert f"mpd status {campaign_dir.name}" in err and "--force" in err
    assert digest(campaign_dir) == before and len(fake_slurm.calls("sbatch")) == calls


def test_a_partly_submitted_campaign_is_refused_and_force_continues_it(capsys, cli_env, monkeypatch, fake_slurm):
    ready_home(cli_env, monkeypatch)
    args = ["submit", *recipe(cli_env.home), "--yes"]
    assert run_cli(capsys, *args, "--dry-run")[0] == 0
    campaign_dir = next((cli_env.home / "runs").iterdir())
    campaign.write_marker(campaign_dir, {"campaign_id": campaign_dir.name, "state": "partial", "jobs": {"prepare": "77"},
                                         "nonce": "abc123", "cluster": "grace", "message": "controller down"})
    rc, _, err = run_cli(capsys, *args)
    assert rc == 2 and "only partly submitted" in err and "prepare 77" in err and "--force" in err
    calls = []
    monkeypatch.setattr(cli, "write_campaign", lambda plan, **kw: calls.append(("write", kw)) or {"prepare": campaign_dir})
    monkeypatch.setattr(cli, "submit", lambda plan, scripts, **kw: calls.append(("submit", kw)) or {})
    rc, out, err = run_cli(capsys, *args, "--force")
    assert rc == 0, err
    assert calls == [("write", {"force": True}), ("submit", {"force": True})]


def test_dry_run_with_force_on_a_submitted_campaign_writes_nothing(capsys, cli_env, monkeypatch, fake_slurm):
    ready_home(cli_env, monkeypatch)
    args = ["submit", *recipe(cli_env.home), "--yes"]
    assert run_cli(capsys, *args)[0] == 0
    campaign_dir = next((cli_env.home / "runs").iterdir())
    before = digest(campaign_dir)
    rc, out, _ = run_cli(capsys, *args, "--dry-run", "--force")
    assert rc == 0 and "so nothing was written" in out and digest(campaign_dir) == before


def test_a_slurm_rejection_is_a_one_line_error_with_the_hint(capsys, cli_env, monkeypatch, fake_slurm):
    ready_home(cli_env, monkeypatch)
    fake_slurm.reject_stage("prepare", INVALID_ACCOUNT)
    rc, out, err = run_cli(capsys, "submit", *recipe(cli_env.home, "--account", FAKE_ACCOUNT), "--yes")
    assert rc == 2 and err.startswith("error: ") and "myproject" in err and "Traceback" not in err
    assert fake_slurm.submitted() == []


# ---------------------------------------------------------------- models, clusters, assays
def test_models_lists_everything_without_cutting_notes(capsys):
    rc, out, _ = run_cli(capsys, "models")
    assert rc == 0
    for key in registry.MODELS:
        assert key in out
    aido = registry.get("aido-rag-16b")
    flat = " ".join(out.split())
    assert " ".join(aido.license_note.split()) in flat and " ".join(aido.note.split()) in flat      # whole texts, wrapped not cut
    assert "\u2026" not in out
    assert "2.61 GB" in out and "11.37 GB" in out and "MIT" in out
    assert "add --cluster or --home" in out


def test_models_shows_which_weights_are_fetched_when_a_home_is_known(capsys, cli_env, monkeypatch):
    override_weights(cli_env, monkeypatch, "esm2-650m")
    rc, out, _ = run_cli(capsys, "models", "--cluster", "grace", "--home", cli_env.home)
    rows = {line.split()[0]: line.split()[-1] for line in out.splitlines() if line.split() and line.split()[0] in registry.MODELS}
    assert rc == 0 and rows["esm2-650m"] == "override" and rows["esm2-3b"] == "missing" and rows["aido-rag-16b"] == "-"


def test_clusters_lists_the_profiles_and_the_gpus_that_cannot_train(capsys):
    rc, out, _ = run_cli(capsys, "clusters")
    assert rc == 0 and "grace" in out and "vision" in out and "a100" in out and "h200" in out
    assert "no native bf16" in out


def test_assays_lists_filters_and_signals_no_match(capsys, cli_env):
    write_proteingym(cli_env.home)
    write_venus(cli_env.home, ids=("ACT_one", "BIND_two"))
    rc, out, _ = run_cli(capsys, "assays", "--home", cli_env.home)
    assert rc == 0 and "proteingym: 3 of 3" in out and "ASSAY_B" in out and "venusmuthub: 2 of 2" in out and "BIND_two" in out
    rc, out, _ = run_cli(capsys, "assays", "--home", cli_env.home, "--dataset", "proteingym", "--grep", "assay_b")
    assert rc == 0 and "1 of 3" in out and "ASSAY_A" not in out
    assert run_cli(capsys, "assays", "--home", cli_env.home, "--grep", "zzz")[0] == 1


def test_assays_without_data_or_home_is_a_clear_error(capsys, cli_env):
    rc, _, err = run_cli(capsys, "assays", "--cluster", "grace", "--home", cli_env.home)
    assert rc == 2 and "ProteinGym data not found under" in err and "mpd fetch proteingym --cluster grace" in err
    rc, _, err = run_cli(capsys, "assays")
    assert rc == 2 and "give --home" in err


# ---------------------------------------------------------------- fetch
def test_fetch_list_describes_every_source_and_points_at_models_and_foldseek(capsys):
    rc, out, _ = run_cli(capsys, "fetch", "--list")
    assert rc == 0 and "15293562" in out
    assert "huggingface.co/datasets/AI4Protein/VenusMutHub" in out and "not yet published (use --from-dir" in out
    venus = out.split("venusmuthub")[1].split("venus-structures")[0]
    assert "not yet published" not in venus                      # the old text claimed this of a source that is fetched
    assert "model KEY" in out and "foldseek" in out


def test_fetch_without_a_home_stays_a_system_exit_with_the_hint():
    with pytest.raises(SystemExit, match="--home"):
        main(["fetch", "proteingym"])


def test_fetch_with_a_cluster_that_needs_scratch_says_so(capsys, cli_env):
    rc, _, err = run_cli(capsys, "fetch", "proteingym", "--cluster", "grace")
    assert rc == 2 and "$SCRATCH" in err


def test_fetch_data_source_with_model_keys_is_an_error(capsys, cli_env):
    rc, _, err = run_cli(capsys, "fetch", "proteingym", "esm2-8m", "--home", cli_env.home)
    assert rc == 2 and "only go with `mpd fetch model`" in err
    rc, _, err = run_cli(capsys, "fetch", "proteingym", "--all", "--home", cli_env.home)
    assert rc == 2 and "--all only goes with" in err


@pytest.fixture
def fetched(monkeypatch):
    """Replace the network part of `mpd fetch model` and record what it was asked."""
    calls = []

    def fetch_model(key, home, **kw):
        calls.append({"key": key, "home": home, "models_dir": __import__("os").environ.get("MPD_MODELS_DIR"), **kw})
        return SimpleNamespace(state="ready")
    monkeypatch.setattr(models, "fetch_model", fetch_model)
    return calls


def test_fetch_model_passes_keys_flags_and_a_deadline_to_the_library(capsys, cli_env, fetched):
    rc, out, err = run_cli(capsys, "fetch", "model", "esm2-650m", "esm2-3b", "--cluster", "grace", "--home", cli_env.home,
                           "--verify", "--force", "--dry-run", "--from-dir", cli_env.tmp)
    assert rc == 0, err
    assert [c["key"] for c in fetched] == ["esm2-650m", "esm2-3b"]
    first = fetched[0]
    assert first["home"] == cli_env.home and first["verify"] and first["force"] and first["dry_run"] and first["from_dir"] == cli_env.tmp
    assert 54 * 60 < first["deadline"] - time.monotonic() <= 55 * 60              # the profile's 60 login minutes minus 5


@pytest.mark.parametrize("flag,expect", [(["--time-budget", "10"], 600), (["--time-budget", "0"], None)])
def test_fetch_model_time_budget(capsys, cli_env, fetched, flag, expect):
    assert run_cli(capsys, "fetch", "model", "esm2-8m", "--cluster", "grace", "--home", cli_env.home, *flag)[0] == 0
    deadline = fetched[0]["deadline"]
    assert deadline is None if expect is None else expect - 5 < deadline - time.monotonic() <= expect


def test_fetch_model_without_a_cluster_has_no_deadline(capsys, cli_env, fetched):
    assert run_cli(capsys, "fetch", "model", "esm2-8m", "--home", cli_env.home)[0] == 0
    assert fetched[0]["deadline"] is None


def test_fetch_model_all_fetches_what_the_pipeline_can_use(capsys, cli_env, fetched):
    assert run_cli(capsys, "fetch", "model", "--all", "--home", cli_env.home)[0] == 0
    assert [c["key"] for c in fetched] == models.fetchable_keys()
    assert "saprot-650m" not in [c["key"] for c in fetched]


def test_fetch_model_stopped_by_the_time_budget_exits_3(capsys, cli_env, monkeypatch):
    def fetch_model(key, home, **kw):
        raise TimeBudgetReached(f"{key}: stopped before the login-node time limit with 2 file(s) left; run the same command again to resume")
    monkeypatch.setattr(models, "fetch_model", fetch_model)
    rc, out, err = run_cli(capsys, "fetch", "model", "esm2-3b", "--cluster", "grace", "--home", cli_env.home)
    assert rc == 3 and "run the same command again to resume" in err and "Traceback" not in err


def test_fetch_model_usage_errors(capsys, cli_env, fetched):
    rc, _, err = run_cli(capsys, "fetch", "model", "--home", cli_env.home)
    assert rc == 2 and "name the models to fetch" in err
    rc, _, err = run_cli(capsys, "fetch", "model", "esm2-650M", "--home", cli_env.home)
    assert rc == 2 and "unknown model 'esm2-650M'" in err and "esm2-650m" in err and "aido" not in err.split("choose from:")[1]
    rc, _, err = run_cli(capsys, "fetch", "model", "esm2-8m", "--all", "--home", cli_env.home)
    assert rc == 2 and "not both" in err
    assert fetched == []
    with pytest.raises(SystemExit, match="MPD_MODELS_DIR"):
        main(["fetch", "model", "esm2-8m"])


def test_fetch_model_accepts_a_models_dir_without_a_home_and_restores_the_environment(capsys, cli_env, fetched, monkeypatch):
    target = cli_env.tmp / "shared"
    assert run_cli(capsys, "fetch", "model", "esm2-8m", "--models-dir", target)[0] == 0
    assert fetched[0]["home"] is None and fetched[0]["models_dir"] == str(target)
    assert "MPD_MODELS_DIR" not in __import__("os").environ
    monkeypatch.setenv("MPD_MODELS_DIR", str(target))
    assert run_cli(capsys, "fetch", "model", "esm2-3b")[0] == 0              # the variable alone is enough


def test_aido_is_never_fetched(capsys, cli_env):
    rc, _, err = run_cli(capsys, "fetch", "model", "aido-rag-16b", "--home", cli_env.home)
    assert rc == 2 and "GenBio" in err and "Traceback" not in err


def test_fetch_model_list_shows_pins_sizes_and_state(capsys, cli_env, monkeypatch):
    override_weights(cli_env, monkeypatch, "esm2-8m")
    rc, out, _ = run_cli(capsys, "fetch", "model", "--list", "--cluster", "grace", "--home", cli_env.home)
    row = next(line for line in out.splitlines() if line.startswith("esm2-8m"))
    assert rc == 0 and "facebook/esm2_t6_8M_UR50D @ c731040fcd" in row and "31.4 MB" in row and row.split()[-1] == "override"
    assert "never fetched" in next(line for line in out.splitlines() if line.startswith("aido-rag-16b"))


def test_fetch_foldseek_calls_the_structure_tools(capsys, cli_env, monkeypatch):
    from mpd.structure import tools
    calls = []
    monkeypatch.setattr(tools, "fetch_foldseek", lambda home, *, from_file=None, log=print: calls.append((home, from_file)) or home / "tools" / "fs")
    tarball = cli_env.tmp / "foldseek-linux-avx2.tar.gz"
    rc, out, _ = run_cli(capsys, "fetch", "foldseek", "--home", cli_env.home, "--from-file", tarball)
    assert rc == 0 and calls == [(cli_env.home, tarball)] and f"foldseek: {cli_env.home / 'tools' / 'fs'}" in out
    rc, _, err = run_cli(capsys, "fetch", "foldseek", "--home", cli_env.home, "--from-dir", cli_env.tmp)
    assert rc == 2 and "--from-file" in err
