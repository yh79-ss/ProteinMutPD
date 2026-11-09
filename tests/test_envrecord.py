"""The environment record (<home>/env/mpd_env.json): its writer, its tolerant reader and what the planner makes of it."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import pytest

from fx_render import clean_mpd_env, make, render_all, stage  # noqa: F401  (clean_mpd_env is autouse)
from mpd import envrecord
from mpd.slurm.plan import PlanError, default_python, resolve_python

SRC = str(Path(__file__).resolve().parents[1] / "src")
NOW = datetime(2026, 11, 9, 12, 30, 5, tzinfo=timezone.utc)


def record_in(tmp_path, **kw) -> tuple[Path, dict]:
    home = tmp_path / "home"
    kw.setdefault("cluster", "grace")
    kw.setdefault("python", str(home / "env" / "bin" / "python"))
    kw.setdefault("python_version", "3.11.7")
    return home, envrecord.write(home, now=NOW, **kw)


# ---------------------------------------------------------------- writer and reader
def test_round_trip(tmp_path):
    constraints = tmp_path / "grace.txt"
    constraints.write_text("torch==2.10.0\ntransformers==4.51.0\n")
    home, written = record_in(tmp_path, modules=["Anaconda3/2024.02-1"], constraints=str(constraints))
    assert envrecord.read(home) == written
    assert written["schema"] == 1 and written["created"] == "2026-11-09T12:30:05Z"
    assert written["cluster"] == "grace" and written["home"] == str(home) and written["modules"] == ["Anaconda3/2024.02-1"]
    assert written["python"] == str(home / "env" / "bin" / "python") and written["python_version"] == "3.11.7"
    assert written["constraints"] == str(constraints)
    assert written["constraints_sha256"] == hashlib.sha256(constraints.read_bytes()).hexdigest()
    assert written["mpd_version"]
    assert written["packages"]["numpy"] == metadata.version("numpy")           # versions come from importlib.metadata
    assert set(written["packages"]) <= set(envrecord.PACKAGES)
    assert list((home / "env").iterdir()) == [home / "env" / "mpd_env.json"]    # the temporary file was renamed away


def test_no_constraints_and_no_modules(tmp_path):
    home, written = record_in(tmp_path)
    assert written["constraints"] is None and written["constraints_sha256"] is None and written["modules"] == []
    assert envrecord.problem(home) is None


def test_rewriting_replaces_the_record(tmp_path):
    home, _ = record_in(tmp_path, modules=["A/1"])
    record_in(tmp_path, modules=["B/2"])
    assert envrecord.read(home)["modules"] == ["B/2"]


@pytest.mark.parametrize("modules", [["bad name"], ["$(touch x)"], ["a;b"], ["x\ny"], [""], ["-x"]])
def test_module_names_are_validated_on_write(tmp_path, modules):
    with pytest.raises(ValueError, match="not a module name"):
        envrecord.write(tmp_path / "home", cluster="grace", modules=modules)


def test_cluster_name_is_validated_on_write(tmp_path):
    with pytest.raises(ValueError, match="cluster"):
        envrecord.write(tmp_path / "home", cluster="grace; rm -rf /")


@pytest.mark.parametrize("content", [
    "", "not json", "[]", "null", '{"schema": 2}', '{"schema": 1, "modules": "A/1"}', '{"schema": 1, "modules": ["a b"]}',
    '{"schema": 1, "modules": [1]}', '{"schema": 1, "python": "env/bin/python"}', '{"schema": 1, "python": 3}',
    '{"schema": 1, "modules": ["$(x)"]}', "\x00\x01"])
def test_a_damaged_record_reads_as_none_with_a_reason(tmp_path, content):
    home = tmp_path / "home"
    (home / "env").mkdir(parents=True)
    (home / "env" / "mpd_env.json").write_text(content)
    assert envrecord.read(home) is None
    assert envrecord.problem(home)


def test_missing_and_odd_records(tmp_path):
    home = tmp_path / "home"
    assert envrecord.read(home) is None and envrecord.problem(home) is None          # absent is not a problem
    (home / "env" / "mpd_env.json").mkdir(parents=True)                                 # a directory where the file should be
    assert envrecord.read(home) is None
    shutil.rmtree(home / "env" / "mpd_env.json")
    (home / "env" / "mpd_env.json").write_bytes(b"\xff\xfe not utf-8")
    assert envrecord.read(home) is None and "cannot be read" in envrecord.problem(home)
    (home / "env" / "mpd_env.json").write_text("x" * (envrecord.MAX_BYTES + 1))
    assert envrecord.read(home) is None and "larger than" in envrecord.problem(home)


# ---------------------------------------------------------------- python -m mpd.envrecord
def cli(*args, cwd=None):
    env = {**os.environ, "PYTHONPATH": SRC, "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": ""}
    return subprocess.run([sys.executable, "-m", "mpd.envrecord", *args], env=env, capture_output=True, text=True, cwd=cwd, timeout=120)


def test_command_line_writes_the_record_for_the_running_python(tmp_path):
    constraints = tmp_path / "c.txt"
    constraints.write_text("numpy==1.26.4\n")
    home = tmp_path / "home"
    done = cli("write", "--home", str(home), "--cluster", "grace", "--modules", "Anaconda3/2024.02-1 GCC/11.3.0",
               "--constraints", str(constraints))
    assert done.returncode == 0, done.stderr
    record = json.loads((home / "env" / "mpd_env.json").read_text())
    assert record["modules"] == ["Anaconda3/2024.02-1", "GCC/11.3.0"] and record["cluster"] == "grace"
    assert record["python"] == sys.executable and record["home"] == str(home)
    assert record["constraints_sha256"] == hashlib.sha256(constraints.read_bytes()).hexdigest()
    assert "wrote" in done.stdout and str(home) not in done.stderr


def test_command_line_accepts_none_and_commas(tmp_path):
    home = tmp_path / "home"
    assert cli("write", "--home", str(home), "--cluster", "vision", "--modules", "none").returncode == 0
    assert envrecord.read(home)["modules"] == []
    assert cli("write", "--home", str(home), "--cluster", "vision", "--modules", "A/1,B/2").returncode == 0
    assert envrecord.read(home)["modules"] == ["A/1", "B/2"]


def test_command_line_errors_are_messages(tmp_path):
    home = tmp_path / "home"
    bad = cli("write", "--home", str(home), "--cluster", "grace", "--modules", "bad;name")
    assert bad.returncode == 2 and "not a module name" in bad.stderr and "Traceback" not in bad.stderr
    gone = cli("write", "--home", str(home), "--cluster", "grace", "--constraints", str(tmp_path / "missing.txt"))
    assert gone.returncode == 2 and "Traceback" not in gone.stderr and not (home / "env" / "mpd_env.json").exists()
    shown = cli("show", "--home", str(home))
    assert shown.returncode == 1 and "no environment record" in shown.stderr
    cli("write", "--home", str(home), "--cluster", "grace")
    shown = cli("show", "--home", str(home))
    assert shown.returncode == 0 and json.loads(shown.stdout)["cluster"] == "grace"


def test_the_module_imports_nothing_but_the_standard_library():
    code = ("import sys; sys.path.insert(0, %r); import mpd.envrecord; "
            "bad = [m for m in ('yaml', 'numpy', 'torch', 'scipy', 'pandas') if m in sys.modules]; print(bad)" % SRC)
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, env={"PYTHONDONTWRITEBYTECODE": "1"})
    assert out.returncode == 0 and out.stdout.strip() == "[]", out.stderr


# ---------------------------------------------------------------- what the planner makes of it
def test_recorded_modules_are_loaded_by_the_job_scripts(tmp_path):
    home, _ = record_in(tmp_path, modules=["Anaconda3/2024.02-1"])
    plan = make("grace", home=home, python=None)
    assert plan.env_record["modules"] == ["Anaconda3/2024.02-1"] and plan.modules == ("Anaconda3/2024.02-1",)
    text = render_all(plan)["train"]
    assert "module purge\nmodule load Anaconda3/2024.02-1 ||" in text
    assert plan.options.python == str(home / "env" / "bin" / "python")           # no --python: the record's interpreter


def test_modules_are_profile_then_record_then_flag_without_repeats(tmp_path):
    home, _ = record_in(tmp_path, cluster="vision", modules=["Anaconda3/2024.02-1", "CUDA/12.8.0"])
    plan = make("vision", home=home, extra_modules=("WebProxy", "Anaconda3/2024.02-1"))
    assert plan.modules == ("CUDA/12.8.0", "Anaconda3/2024.02-1", "WebProxy")
    assert "module load CUDA/12.8.0 Anaconda3/2024.02-1 WebProxy ||" in render_all(plan)["prepare"]


def test_python_precedence_is_flag_then_environment_variable_then_record_then_default(tmp_path, monkeypatch):
    home, record = record_in(tmp_path, python=str(tmp_path / "recorded" / "bin" / "python"))
    assert make("grace", home=home, python=None).options.python == record["python"]
    monkeypatch.setenv("MPD_PYTHON", "/opt/env/bin/python")
    assert make("grace", home=home, python=None).options.python == "/opt/env/bin/python"
    assert make("grace", home=home, python="/flag/bin/python").options.python == "/flag/bin/python"
    monkeypatch.delenv("MPD_PYTHON")
    bare = tmp_path / "bare"
    assert make("grace", home=bare, python=None).options.python == str(bare / "env" / "bin" / "python")
    assert default_python(bare) == str(bare / "env" / "bin" / "python") and default_python(home) == record["python"]


def test_default_python_ignores_a_record_written_for_another_home(tmp_path):
    home, record = record_in(tmp_path)
    other = tmp_path / "other"
    (other / "env").mkdir(parents=True)
    shutil.copy(home / "env" / "mpd_env.json", other / "env" / "mpd_env.json")
    assert default_python(home) == record["python"]
    assert default_python(other) == str(other / "env" / "bin" / "python")


def test_resolve_python_helper_keeps_old_callers_working(tmp_path):
    from fx_render import opts
    assert resolve_python(opts(tmp_path, python="/given/bin/python")) == "/given/bin/python"
    assert resolve_python(opts(tmp_path, python=None)) == str(tmp_path / "env" / "bin" / "python")


def test_a_missing_record_is_not_an_error_or_a_warning(tmp_path):
    plan = make("grace", home=tmp_path / "empty")
    assert plan.env_record is None and not any("environment record" in w for w in plan.warnings)


def test_a_corrupt_record_is_ignored_with_a_warning(tmp_path):
    home = tmp_path / "home"
    (home / "env").mkdir(parents=True)
    (home / "env" / "mpd_env.json").write_text("{ broken")
    plan = make("grace", home=home)
    assert plan.env_record is None and plan.modules == ()
    assert any("ignoring the environment record" in w and "setup_env.sh --cluster grace" in w for w in plan.warnings)


def test_a_record_from_another_cluster_is_ignored(tmp_path):
    home, _ = record_in(tmp_path, cluster="vision", modules=["CUDA/12.8.0"])
    plan = make("grace", home=home)
    assert plan.env_record is None and plan.modules == ()
    assert any("built for cluster 'vision'" in w for w in plan.warnings)


def test_a_record_copied_to_another_home_is_ignored(tmp_path):
    home, _ = record_in(tmp_path, modules=["A/1"])
    other = tmp_path / "other"
    (other / "env").mkdir(parents=True)
    shutil.copy(home / "env" / "mpd_env.json", other / "env" / "mpd_env.json")
    plan = make("grace", home=other)
    assert plan.env_record is None and any("written for" in w for w in plan.warnings)


def test_the_record_is_matched_by_real_path(tmp_path):
    home, _ = record_in(tmp_path, modules=["A/1"])
    link = tmp_path / "link"
    link.symlink_to(home)
    assert make("grace", home=link).env_record is not None


def test_stage_scripts_of_one_plan_agree_on_the_modules(tmp_path):
    home, _ = record_in(tmp_path, modules=["Anaconda3/2024.02-1"])
    plan = make("grace", home=home)
    for name, text in render_all(plan).items():
        assert text.count("module load Anaconda3/2024.02-1") == 1, name
    assert stage(plan, "train")


def test_a_relative_home_is_refused_before_the_record_is_read(tmp_path):
    with pytest.raises(PlanError, match="relative path"):
        make("grace", home=Path("relhome"))
