"""scripts/setup_env.sh: profile names in any letter case, a Python newer than the tested ones, and the message for a shell
that is not bash. The fake Python, fake `module` and repository copy are those of test_setup_script.py."""

import subprocess

import pytest
from fx_cli import cli_env  # noqa: F401
from test_setup_script import REAL_SCRIPT, repo, setup  # noqa: F401

from mpd import envrecord

DRY = {"MPD_SETUP_DRY_RUN": "1"}
HOME_REFUSAL = "/home/jdoe42/mpd is under /home (10 GB / 10,000 files on Grace). Use scratch: export MPD_HOME=$SCRATCH/mpd"


# ---------------------------------------------------------------- Grace, GRACE
@pytest.mark.parametrize("spelling", ["Grace", "GRACE", "gRaCe"])
def test_grace_in_any_letter_case_gets_the_grace_defaults(setup, spelling):
    out = setup("--cluster", spelling, "--yes", env=DRY)
    assert out.returncode == 0, out.stderr
    assert f"cluster grace, home {setup.home}" in out.stdout
    assert "module load Anaconda3/2024.02-1" in setup.modules()                 # the Grace default module
    assert any("mpd.envrecord write" in ln and "--cluster grace " in ln for ln in out.stdout.splitlines())
    assert any(ln.endswith(f"--home {setup.home}") and "mpd doctor --cluster grace" in ln for ln in out.stdout.splitlines())


@pytest.mark.parametrize("spelling", ["Grace", "GRACE"])
def test_a_home_under_slash_home_is_refused_for_grace_in_any_letter_case(setup, cli_env, spelling):
    out = setup("--cluster", spelling, "--yes", "--home", "/home/jdoe42/mpd")
    assert out.returncode == 2 and HOME_REFUSAL in out.stderr
    assert setup.log() == [] and list(cli_env.userhome.iterdir()) == []


def test_grace_without_scratch_asks_for_a_home_whatever_the_letter_case(setup):
    out = setup("--cluster", "Grace", "--yes", env={"SCRATCH": ""})
    assert out.returncode == 2 and "$SCRATCH is not set" in out.stderr


def test_a_full_run_with_a_capital_letter_records_and_activates_the_lower_case_name(setup):
    out = setup("--cluster", "Grace", "--yes")
    assert out.returncode == 0, out.stderr + out.stdout
    assert envrecord.read(setup.home)["cluster"] == "grace"                      # the planner compares this with the profile name
    activate = (setup.home / "env" / "bin" / "activate").read_text()
    assert "export MPD_CLUSTER=grace\n" in activate and "Grace" not in activate
    assert any(ln.startswith("mpd doctor --cluster grace --home") for ln in setup.log())


def test_mpd_cluster_in_the_environment_gets_the_same_treatment(setup):
    out = setup("--yes", env={**DRY, "MPD_CLUSTER": "GRACE"})
    assert out.returncode == 0 and f"cluster grace, home {setup.home}" in out.stdout


def test_a_path_to_a_profile_file_keeps_its_spelling(setup, tmp_path):
    profile = tmp_path / "Mine.yaml"
    profile.write_text("name: Mine\n")
    out = setup("--cluster", str(profile), "--yes", "--modules", "none", "--home", str(tmp_path / "h"))
    assert out.returncode == 0, out.stderr
    assert f"export MPD_CLUSTER={profile}\n" in (tmp_path / "h" / "env" / "bin" / "activate").read_text()
    assert envrecord.read(tmp_path / "h")["cluster"] == "Mine"                   # the file's own name, as before


# ---------------------------------------------------------------- Python newer than 3.13
@pytest.mark.parametrize("version", ["3.14.0", "3.14.6", "3.15.1", "3.20.0", "4.0.0"])
def test_a_python_newer_than_the_tested_ones_is_a_warning_and_the_script_goes_on(setup, version):
    out = setup("--cluster", "grace", "--yes", "--modules", "none", env={**DRY, "FAKE_PY_VERSION": version})
    assert out.returncode == 0, out.stderr
    assert f"Python {version}, which is newer than the Python 3.13 that mpd was tested with." in out.stderr
    assert "build again with Python 3.11, 3.12 or 3.13" in out.stderr and "error" not in out.stderr
    assert f"python {version} (" in out.stdout and any(" -m pip " in ln for ln in out.stdout.splitlines())


@pytest.mark.parametrize("version", ["3.11.0", "3.11.7", "3.12.3", "3.13.0", "3.13.12"])
def test_the_tested_pythons_get_no_warning(setup, version):
    out = setup("--cluster", "grace", "--yes", "--modules", "none", env={**DRY, "FAKE_PY_VERSION": version})
    assert out.returncode == 0 and "warning" not in out.stderr


def test_an_untested_python_still_builds_the_environment(setup):
    out = setup("--cluster", "grace", "--yes", "--modules", "none", env={"FAKE_PY_VERSION": "3.14.2"})
    assert out.returncode == 0, out.stderr + out.stdout
    assert "newer than the Python 3.13" in out.stderr and any(ln.startswith("pip install") for ln in setup.log())


def test_the_missing_module_hint_names_the_python_versions_that_were_tested(setup):
    out = setup("--cluster", "grace", "--yes", "--modules", "Nonexistent/9.9")
    assert out.returncode == 1 and "Use Python 3.11, 3.12 or 3.13: newer versions were not tested." in out.stderr


# ---------------------------------------------------------------- sh instead of bash
def test_the_message_for_a_shell_that_is_not_bash_gives_the_command_to_run(setup):
    shell = subprocess.run([str(setup.sh), "-c", "echo ${BASH_VERSION:+bash}"], capture_output=True, text=True).stdout.strip()
    if shell == "bash":
        pytest.skip("/bin/sh is bash here, so the guard cannot fire")
    out = subprocess.run([str(setup.sh), str(REAL_SCRIPT), "--cluster", "grace", "--yes"], capture_output=True, text=True,
                         stdin=subprocess.DEVNULL)
    assert out.returncode == 2 and out.stdout == ""
    assert out.stderr == "setup: error: run this script with bash:  bash scripts/setup_env.sh --cluster grace\n"
