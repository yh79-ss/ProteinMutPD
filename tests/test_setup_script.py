"""scripts/setup_env.sh against a fake Python, a fake `module` and a copy of the repository layout. No network, no pip, no torch."""

import hashlib
import json
import os
import pty
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fx_cli import cli_env, make_setup_python, write_module_env  # noqa: F401

from mpd import envrecord

REAL_SCRIPT = Path(__file__).parents[1] / "scripts" / "setup_env.sh"
SRC = str(Path(__file__).parents[1] / "src")


@pytest.fixture
def repo(tmp_path):
    """A copy of the layout the script expects around itself, so these tests do not depend on the rest of the tree."""
    root = tmp_path / "clone"
    (root / "scripts").mkdir(parents=True)
    (root / "constraints").mkdir()
    shutil.copy(REAL_SCRIPT, root / "scripts" / "setup_env.sh")
    (root / "pyproject.toml").write_text("[project]\nname = 'stand-in'\n", encoding="utf-8")
    (root / "README.md").write_text("# stand-in\n", encoding="utf-8")
    (root / "constraints" / "grace.txt").write_text("torch==2.10.0\ntransformers==4.51.0\n", encoding="utf-8")
    (root / "src" / "mpd" / "__pycache__").mkdir(parents=True)
    (root / "src" / "mpd" / "__init__.py").write_text("__version__ = '0'\n", encoding="utf-8")
    (root / "src" / "mpd" / "__pycache__" / "x.pyc").write_text("x", encoding="utf-8")
    (root / "src" / "stale.egg-info").mkdir()
    (root / "src" / "stale.egg-info" / "PKG-INFO").write_text("x", encoding="utf-8")
    for junk in (".git/HEAD", "tests/test_x.py", "build/lib/old_module.py", "docs/GRACE.md"):
        (root / junk).parent.mkdir(parents=True, exist_ok=True)
        (root / junk).write_text("x", encoding="utf-8")
    return root


@pytest.fixture
def setup(cli_env, repo, tmp_path):
    """run(*args, **extra_env) -> CompletedProcess; with a fake python3 first on PATH and `module` defined through BASH_ENV."""
    python = make_setup_python(tmp_path / "fakepy")
    cli_env.link("python3", python)
    bash_env, available = write_module_env(tmp_path / "modules")
    log = tmp_path / "calls.log"
    log.write_text("", encoding="utf-8")
    scratch = tmp_path / "scratch"

    def run(*args, env=None, stdin=subprocess.DEVNULL, script=None):
        environ = {"PATH": str(cli_env.bin), "HOME": str(cli_env.userhome), "BASH_ENV": str(bash_env), "MODULES_AVAILABLE": str(available),
                   "MODULE_LOG": str(tmp_path / "module.log"), "FAKE_LOG": str(log), "FAKE_SRC": SRC, "FAKE_REAL_PYTHON": sys.executable,
                   "SCRATCH": str(scratch), **(env or {})}
        return subprocess.run(["bash", str(script or repo / "scripts" / "setup_env.sh"), *args], cwd=tmp_path, env=environ,
                              capture_output=True, text=True, stdin=stdin)

    run.log = lambda: log.read_text(encoding="utf-8").splitlines()
    run.modules = lambda: (tmp_path / "module.log").read_text(encoding="utf-8").splitlines() if (tmp_path / "module.log").exists() else []
    run.home = scratch / "mpd"
    run.python = python
    run.sh = cli_env.bin / "sh"
    return run


def userhome_is_untouched(cli_env):
    return list(cli_env.userhome.iterdir()) == []


# ---------------------------------------------------------------- the file itself
def test_the_script_is_valid_bash_and_shellcheck_clean_when_shellcheck_exists():
    assert subprocess.run(["bash", "-n", str(REAL_SCRIPT)], capture_output=True).returncode == 0
    if shutil.which("shellcheck"):
        out = subprocess.run(["shellcheck", "-x", str(REAL_SCRIPT)], capture_output=True, text=True)
        assert out.returncode == 0, out.stdout
    else:
        pytest.skip("shellcheck is not installed; only `bash -n` ran")


def test_help_documents_every_option_and_says_it_was_not_run_on_grace(setup):
    out = setup("--help")
    assert out.returncode == 0
    for text in ("--cluster", "--home", "--python", "--modules", "--dev", "--no-torch", "--constraints", "--rebuild", "--yes",
                 "MPD_SETUP_DRY_RUN", "Not run on Grace", "$MPD_HOME", "$MPD_SETUP_MODULES", "Anaconda3/2024.02-1"):
        assert text in out.stdout, text
    assert "10,000 files" in out.stdout


def test_running_it_with_a_shell_that_is_not_bash_says_so(setup):
    out = subprocess.run([str(setup.sh), str(REAL_SCRIPT), "--help"], capture_output=True, text=True, stdin=subprocess.DEVNULL)
    if "BASH_VERSION" in subprocess.run([str(setup.sh), "-c", "echo ${BASH_VERSION:+BASH_VERSION}"], capture_output=True, text=True).stdout:
        pytest.skip("/bin/sh is bash here, so the guard cannot fire")
    assert out.returncode == 2 and "run this script with bash" in out.stderr


def test_the_script_has_no_command_that_installs_into_the_users_home():
    text = REAL_SCRIPT.read_text(encoding="utf-8")
    for forbidden in ("~/.", "$HOME/.", "conda init", "pip install --user", "sudo ", ".bashrc", ".profile"):
        assert forbidden not in text, forbidden


@pytest.mark.parametrize("args,fragment", [(["--bogus"], "unknown option --bogus"), (["--home"], "needs a value"),
                                           ([], "give --cluster"), (["--cluster", "grace", "--home", "x", "--constraints", "/nope.txt"], "not found")])
def test_usage_errors_exit_2_with_a_message(setup, args, fragment):
    out = setup(*args, env={"MPD_CLUSTER": ""})
    assert out.returncode == 2 and fragment in out.stderr and "Traceback" not in out.stderr


# ---------------------------------------------------------------- refusals
@pytest.mark.parametrize("how", ["flag", "equals", "env"])
def test_a_home_under_slash_home_is_refused_on_grace_and_nothing_is_created(setup, cli_env, how):
    args = {"flag": ["--home", "/home/jdoe42/mpd"], "equals": ["--home=/home/jdoe42/mpd"], "env": []}[how]
    out = setup("--cluster", "grace", "--yes", *args, env={"MPD_HOME": "/home/jdoe42/mpd"} if how == "env" else None)
    assert out.returncode == 2
    assert "/home/jdoe42/mpd is under /home (10 GB / 10,000 files on Grace). Use scratch: export MPD_HOME=$SCRATCH/mpd" in out.stderr
    assert setup.log() == [] and userhome_is_untouched(cli_env)


def test_a_dotted_path_into_slash_home_is_still_refused(setup):
    out = setup("--cluster", "grace", "--yes", "--home", "/scratch/../home/jdoe42/mpd")
    assert out.returncode == 2 and "is under /home" in out.stderr


def test_grace_without_scratch_or_home_says_what_to_pass(setup):
    out = setup("--cluster", "grace", "--yes", env={"SCRATCH": ""})
    assert out.returncode == 2 and "$SCRATCH is not set" in out.stderr


def test_other_clusters_need_an_explicit_home(setup):
    out = setup("--cluster", "vision", "--yes")
    assert out.returncode == 2 and "no home" in out.stderr


def test_a_missing_module_fails_loudly_with_the_way_to_find_the_right_one(setup):
    out = setup("--cluster", "grace", "--yes", "--modules", "Nonexistent/9.9")
    assert out.returncode == 1
    assert "module Nonexistent/9.9 did not load" in out.stderr and "module spider Python" in out.stderr
    assert "--modules none --python" in out.stderr and "not been run on Grace" in out.stderr
    assert setup.modules()[0] == "module purge"                   # a clean slate before anything is loaded
    assert not setup.home.exists() and setup.log() == []


def test_the_default_module_comes_from_the_cluster_and_can_be_overridden(setup):
    assert setup("--cluster", "grace", "--yes", env={"MPD_SETUP_DRY_RUN": "1"}).returncode == 0
    assert "module load Anaconda3/2024.02-1" in setup.modules()
    assert setup("--cluster", "grace", "--yes", env={"MPD_SETUP_DRY_RUN": "1", "MPD_SETUP_MODULES": "Python/3.11.5"}).returncode == 0
    assert setup.modules()[-1] == "module load Python/3.11.5"
    assert setup("--cluster", "grace", "--yes", "--modules", "Python/3.11.5 Anaconda3/2024.02-1", env={"MPD_SETUP_DRY_RUN": "1"}).returncode == 0
    assert setup.modules()[-1] == "module load Python/3.11.5 Anaconda3/2024.02-1"


def test_modules_none_loads_nothing_even_when_module_exists(setup, tmp_path):
    out = setup("--cluster", "grace", "--yes", "--modules", "none", env={"MPD_SETUP_DRY_RUN": "1"})
    assert out.returncode == 0 and "modules: none" in out.stdout and setup.modules() == []


def test_no_module_command_and_modules_wanted_is_an_error(setup):
    out = setup("--cluster", "grace", "--yes", env={"MPD_SETUP_DRY_RUN": "1", "BASH_ENV": "/dev/null", "MPD_SETUP_PROFILE": "/dev/null"})
    assert out.returncode == 1 and "'module' command is not available" in out.stderr and "--modules none" in out.stderr


def test_an_old_python_is_refused(setup):
    out = setup("--cluster", "grace", "--yes", "--modules", "none", env={"FAKE_PY_VERSION": "3.10.14"})
    assert out.returncode == 1 and "Python 3.10.14" in out.stderr and "3.11 or newer" in out.stderr
    assert not setup.home.exists()


def test_a_python_without_venv_is_refused(setup):
    out = setup("--cluster", "grace", "--yes", "--modules", "none", env={"FAKE_NO_VENV": "1"})
    assert out.returncode == 1 and "no venv or ensurepip" in out.stderr


def test_missing_constraints_are_refused_unless_none_is_asked_for(setup, repo):
    (repo / "constraints" / "grace.txt").unlink()
    out = setup("--cluster", "grace", "--yes", "--modules", "none")
    assert out.returncode == 2 and "constraints file" in out.stderr and "--constraints none" in out.stderr
    out = setup("--cluster", "grace", "--yes", "--modules", "none", "--constraints", "none", env={"MPD_SETUP_DRY_RUN": "1"})
    assert out.returncode == 0 and "not the tested" in out.stderr and " -c " not in out.stdout


@pytest.mark.parametrize("answer,code,fragment", [("n", 2, "not confirmed"), ("y", 0, "setup: done")])
def test_on_a_terminal_it_asks_before_it_builds(setup, answer, code, fragment):
    master, slave = pty.openpty()
    os.write(master, f"{answer}\n".encode())                         # waits in the terminal's buffer until the script reads it
    try:
        out = setup("--cluster", "grace", stdin=slave)
    finally:
        os.close(master)
        os.close(slave)
    assert out.returncode == code and fragment in out.stdout + out.stderr
    assert "about to build" in out.stdout and "Continue? [y/N]" in out.stdout
    assert (setup.home / "env").exists() == (answer == "y")


def test_a_relative_python_is_made_absolute(setup, tmp_path):
    rel = os.path.relpath(setup.python, tmp_path)
    out = setup("--cluster", "grace", "--yes", "--modules", "none", "--python", rel, env={"MPD_SETUP_DRY_RUN": "1"})
    assert out.returncode == 0 and f"({setup.python})" in out.stdout


def test_without_yes_and_without_a_terminal_it_refuses_to_start(setup):
    out = setup("--cluster", "grace", "--modules", "none")
    assert out.returncode == 2 and "pass --yes" in out.stderr and not setup.home.exists()


# ---------------------------------------------------------------- dry run
def test_dry_run_prints_the_commands_and_writes_nothing(setup, repo, cli_env):
    out = setup("--cluster", "grace", "--modules", "none", env={"MPD_SETUP_DRY_RUN": "1"})
    assert out.returncode == 0, out.stderr
    lines = out.stdout.splitlines()
    pip = next(ln for ln in lines if " -m pip " in ln)
    assert f"-c {repo}/constraints/grace.txt {setup.home}/cache/install-src\\[train\\]" in pip and " -e " not in pip
    assert any(ln.startswith("+ mkdir -p ") and f"{setup.home}/cache/pip" in ln for ln in lines)
    assert any(" -m venv " in ln and ln.endswith(f"{setup.home}/env") for ln in lines)
    assert any("mpd.envrecord write" in ln and "--cluster grace" in ln and "--modules none" in ln for ln in lines)
    assert any("mpd doctor --cluster grace" in ln for ln in lines) and "dry run: nothing was written" in out.stdout
    assert not setup.home.exists() and userhome_is_untouched(cli_env) and setup.log() == ["python -c import sys; print(\".\".join(map(str, sys.version_info[:3])))",
                                                                                         "python -c import venv, ensurepip"]


@pytest.mark.parametrize("flags,target,editable", [(["--dev"], "[train,test]", True), (["--no-torch"], "", False),
                                                  (["--dev", "--no-torch"], "[test]", True)])
def test_dev_and_no_torch_choose_the_install_target(setup, repo, flags, target, editable):
    out = setup("--cluster", "grace", "--modules", "none", *flags, env={"MPD_SETUP_DRY_RUN": "1"})
    argv = shlex.split(next(ln for ln in out.stdout.splitlines() if " -m pip " in ln)[2:])      # undo the shell quoting of the printout
    where = repo if editable else setup.home / "cache" / "install-src"                      # a normal install builds from a copy
    assert argv[-1] == f"{where}{target}" and (argv[-2] == "-e") == editable
    assert argv[argv.index("-c") + 1] == f"{repo}/constraints/grace.txt"


def test_the_relative_home_and_tilde_become_absolute(setup, tmp_path, cli_env):
    out = setup("--cluster", "vision", "--modules", "none", "--home", "work/../mpdhome", env={"MPD_SETUP_DRY_RUN": "1"})
    assert out.returncode == 0 and f"home {tmp_path}/mpdhome" in out.stdout
    out = setup("--cluster", "vision", "--modules", "none", "--home", "~/elsewhere", env={"MPD_SETUP_DRY_RUN": "1"})
    assert f"home {cli_env.userhome}/elsewhere" in out.stdout and "inside your home directory" in out.stderr


# ---------------------------------------------------------------- a full run with the fake python
def test_full_run_builds_installs_records_and_activates(setup, repo, cli_env):
    out = setup("--cluster", "grace", "--yes")
    assert out.returncode == 0, out.stderr + out.stdout
    home = setup.home
    log = setup.log()
    pip = next(ln for ln in log if ln.startswith("pip install"))
    assert pip == f"pip install -c {repo}/constraints/grace.txt {home}/cache/install-src[train]"
    assert f"pip env PIP_CACHE_DIR={home}/cache/pip TMPDIR={home}/cache/tmp PYTHONNOUSERSITE=1" in log      # nothing cached in $HOME
    assert (home / "env" / "pyvenv.cfg").is_file() and (home / "cache" / "pip").is_dir()
    assert "environment: 6 files" in out.stdout and "250,000 files" in out.stdout
    record = envrecord.read(home)                                   # written by the real mpd.envrecord, through the fake python
    assert record["cluster"] == "grace" and record["modules"] == ["Anaconda3/2024.02-1"] and record["home"] == str(home)
    sha = hashlib.sha256((repo / "constraints" / "grace.txt").read_bytes()).hexdigest()
    assert record["constraints_sha256"] == sha and record["constraints"] == str(repo / "constraints" / "grace.txt")
    activate = (home / "env" / "bin" / "activate").read_text()
    assert f"export MPD_HOME={home}" in activate and "export MPD_CLUSTER=grace" in activate and "# <<< mpd <<<" in activate
    assert any(ln.startswith("mpd doctor --cluster grace --home") for ln in log)
    assert "fake mpd doctor output" in out.stdout and "mpd fetch proteingym" in out.stdout
    assert userhome_is_untouched(cli_env)


def test_a_normal_install_builds_from_a_copy_and_leaves_the_clone_alone(setup, repo):
    def listing():
        return sorted(str(p.relative_to(repo)) for p in repo.rglob("*"))
    before = listing()
    assert setup("--cluster", "grace", "--yes").returncode == 0
    assert listing() == before                                     # nothing written into the clone
    copy = setup.home / "cache" / "install-src"
    names = sorted(str(p.relative_to(copy)) for p in copy.rglob("*") if p.is_file())
    assert names == ["README.md", "pyproject.toml", "src/mpd/__init__.py"]            # sources and top-level files, no junk
    (copy / "src" / "mpd" / "gone_after_pull.py").write_text("stale")
    assert setup("--cluster", "grace", "--yes").returncode == 0
    assert not (copy / "src" / "mpd" / "gone_after_pull.py").exists()                # every run starts from a fresh copy


def test_running_again_reuses_the_environment_and_does_not_duplicate_the_activation_block(setup):
    assert setup("--cluster", "grace", "--yes").returncode == 0
    activate = setup.home / "env" / "bin" / "activate"
    first = activate.read_text()
    out = setup("--cluster", "grace", "--yes")
    assert out.returncode == 0 and "reusing the environment" in out.stdout
    assert activate.read_text() == first and first.count("# >>> mpd") == 1
    log = setup.log()
    assert sum(ln.startswith("pip install") for ln in log) == 2 and sum(ln == f"python -m venv {setup.home}/env" for ln in log) == 1


def test_activation_block_is_replaced_when_the_cluster_changes_and_other_lines_survive(setup):
    assert setup("--cluster", "grace", "--yes").returncode == 0
    activate = setup.home / "env" / "bin" / "activate"
    activate.write_text("# my own line\n" + activate.read_text() + "# trailing line\n")
    assert setup("--cluster", "grace", "--yes", "--home", str(setup.home)).returncode == 0
    text = activate.read_text()
    assert "# my own line" in text and "# trailing line" in text and text.count("# >>> mpd") == 1


def test_rebuild_removes_only_a_real_venv(setup):
    assert setup("--cluster", "grace", "--yes").returncode == 0
    marker = setup.home / "env" / "leftover.txt"
    marker.write_text("x")
    out = setup("--cluster", "grace", "--yes", "--rebuild")
    assert out.returncode == 0 and not marker.exists() and "removing the old environment" in out.stdout
    (setup.home / "env" / "pyvenv.cfg").unlink()
    (setup.home / "env" / "keep.txt").write_text("precious")
    out = setup("--cluster", "grace", "--yes", "--rebuild")
    assert out.returncode == 2 and "not a venv" in out.stderr and (setup.home / "env" / "keep.txt").exists()


def test_a_non_empty_folder_that_is_not_a_venv_is_not_touched(setup):
    (setup.home / "env").mkdir(parents=True)
    (setup.home / "env" / "keep.txt").write_text("precious")
    out = setup("--cluster", "grace", "--yes")
    assert out.returncode == 2 and "not a working venv" in out.stderr and (setup.home / "env" / "keep.txt").exists()


def test_a_failing_pip_stops_the_script_before_the_record_is_written(setup):
    out = setup("--cluster", "grace", "--yes", env={"FAKE_PIP_FAIL": "1"})
    assert out.returncode != 0 and "no matching distribution" in out.stderr
    assert envrecord.read(setup.home) is None and "# >>> mpd" not in (setup.home / "env" / "bin" / "activate").read_text()


def test_an_mpd_without_envrecord_gets_the_same_record_from_the_script(setup, repo):
    out = setup("--cluster", "grace", "--yes", env={"FAKE_NO_ENVRECORD": "1"})
    assert out.returncode == 0 and "has no mpd.envrecord module" in out.stderr
    record = envrecord.read(setup.home)                              # the real reader validates the schema
    assert record is not None and record["schema"] == 1 and record["modules"] == ["Anaconda3/2024.02-1"]
    assert record["constraints_sha256"] == hashlib.sha256((repo / "constraints" / "grace.txt").read_bytes()).hexdigest()
    assert set(record) >= {"created", "cluster", "home", "python", "python_version", "modules", "mpd_version", "constraints", "packages"}


def test_the_record_names_a_yaml_profile_by_its_file_stem(setup, tmp_path):
    profile = tmp_path / "my.yaml"
    profile.write_text("name: my\n")
    out = setup("--cluster", str(profile), "--yes", "--modules", "none", "--home", str(tmp_path / "h"))
    assert out.returncode == 0, out.stderr
    assert envrecord.read(tmp_path / "h")["cluster"] == "my"
    assert f"export MPD_CLUSTER={profile}" in (tmp_path / "h" / "env" / "bin" / "activate").read_text()


def test_a_doctor_failure_fails_the_setup_except_with_no_torch(setup):
    out = setup("--cluster", "grace", "--yes", env={"FAKE_DOCTOR_RC": "1"})
    assert out.returncode == 1 and "mpd doctor reported problems" in out.stderr
    out = setup("--cluster", "grace", "--yes", "--no-torch", env={"FAKE_DOCTOR_RC": "1"})
    assert out.returncode == 0 and "a failing 'packages' row is expected" in out.stdout
    assert [ln for ln in setup.log() if ln.startswith("pip install")][-1].endswith("install-src")


def test_a_dev_install_is_editable_with_the_test_extra(setup, repo):
    out = setup("--cluster", "grace", "--yes", "--dev")
    assert out.returncode == 0
    assert f"pip install -c {repo}/constraints/grace.txt -e {repo}[train,test]" in setup.log()


def test_the_default_constraints_are_the_ones_next_to_the_script(setup, repo, tmp_path):
    other = tmp_path / "mine.txt"
    other.write_text("torch==2.9.0\n")
    assert setup("--cluster", "grace", "--yes", "--constraints", str(other)).returncode == 0
    assert any(ln.startswith(f"pip install -c {other} ") for ln in setup.log())
    assert json.loads((setup.home / "env" / "mpd_env.json").read_text())["constraints"] == str(other)
