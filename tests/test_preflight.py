"""The preflight checks behind `mpd plan`, `mpd submit` and `mpd doctor`, their table, and the paste-safe report."""

import hashlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fx_cli import (FAKE_ACCOUNT, FAKE_USER, cli_env, make_fake_python, override_weights, ready_home,  # noqa: F401
                    run_cli, write_proteingym, write_record, write_shim, write_venus)

from mpd import preflight
from mpd.checks import Check
from mpd.preflight import Context
from mpd.slurm.cluster import Cluster
from mpd.spec import RunSpec

pytestmark = pytest.mark.usefixtures("cli_env")


def grace():
    return Cluster.load("grace")


def by_name(checks, name):
    return [c for c in checks if c.name == name]


def one(checks, name):
    found = by_name(checks, name)
    assert len(found) == 1, [c.name for c in checks]
    return found[0]


def spec(**kw):
    base = dict(teacher="esm2-3b", student="esm2-650m", train_data="proteingym", method="score-all", eval_data=("proteingym",))
    return RunSpec(**{**base, **kw})


# ---------------------------------------------------------------- imports
def test_cli_modules_import_without_torch():
    src = str(Path(__file__).parents[1] / "src")
    code = (f"import sys; sys.path.insert(0, {src!r}); import mpd.cli, mpd.preflight, mpd.results, mpd.__main__; "
            "assert 'torch' not in sys.modules, 'torch was imported'")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    out = subprocess.run([sys.executable, "-s", "-c", code], env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


# ---------------------------------------------------------------- home
def test_home_under_the_forbidden_prefix_uses_the_canonical_text():
    c = one(preflight.check_home(Context(grace(), Path("/home/jdoe42/mpd"))), "home")
    assert c.level == "error"
    assert c.message == "/home/jdoe42/mpd is under /home (10 GB / 10,000 files on Grace). Use scratch: export MPD_HOME=$SCRATCH/mpd"


def test_home_that_will_be_created_is_ok_and_not_created(cli_env):
    c = one(preflight.check_home(Context(grace(), cli_env.home)), "home")
    assert c.level == "ok" and "will be created" in c.message and "writable" in c.message
    assert not cli_env.home.exists()                         # the preflight writes nothing


def test_unwritable_home_is_an_error(cli_env, monkeypatch):
    monkeypatch.setattr(preflight.os, "access", lambda *a, **k: False)
    c = one(preflight.check_home(Context(grace(), cli_env.home)), "home")
    assert c.level == "error" and "not writable" in c.message and c.fix


def test_low_free_space_is_a_warning(cli_env, monkeypatch):
    monkeypatch.setattr(preflight.shutil, "disk_usage", lambda p: SimpleNamespace(total=1e12, used=9.9e11, free=5e9))
    c = one(preflight.check_home(Context(grace(), cli_env.home)), "home")
    assert c.level == "warn" and "Low" in c.message


def test_home_inside_the_users_home_directory_is_a_warning_on_any_profile(cli_env):
    c = one(preflight.check_home(Context(Cluster.load("vision"), cli_env.userhome / "mpd")), "home")
    assert c.level == "warn" and "inside your home directory" in c.message and c.fix == "export MPD_HOME=$SCRATCH/mpd"
    assert one(preflight.check_home(Context(Cluster.load("vision"), cli_env.home)), "home").level == "ok"


# ---------------------------------------------------------------- showquota and myproject
def test_showquota_parser_reads_percentages_and_used_over_limit_pairs_and_ignores_the_rest():
    text = """Disk quotas for jdoe42
    Filesystem   Used   Limit  Files  Limit
    /home/jdoe42   2.5 GB  10 GB  9.8K  10K
    /scratch/user/jdoe42  100 GB  1 TB  40,000  250,000
    /scratch/user/other  12%
    nothing useful here"""
    rows = dict((line.split()[0], pct) for line, pct in preflight.parse_showquota(text))
    assert rows["/home/jdoe42"] == pytest.approx(98.0)       # 9.8K of 10K files is the highest of the two ratios
    assert rows["/scratch/user/jdoe42"] == pytest.approx(16.0)
    assert rows["/scratch/user/other"] == 12.0
    assert preflight.parse_showquota("garbage\n\n???") == []


def test_quota_without_showquota_is_skipped(cli_env):
    assert one(preflight.check_quota(Context(grace(), cli_env.home)), "quota").level == "skip"


@pytest.mark.parametrize("text,level,fragment", [
    ("/home/jdoe42   9.5 GB  10 GB  100  10000", "warn", "95%"),
    ("/scratch/user/jdoe42  100 GB  1 TB  40,000  250,000", "ok", "16%"),
    ("an output format nobody has seen", "ok", "not recognised"),
])
def test_quota_levels_follow_the_highest_usage(cli_env, text, level, fragment):
    write_shim(cli_env, "showquota", text)
    c = one(preflight.check_quota(Context(grace(), cli_env.home)), "quota")
    assert c.level == level and fragment in c.message


def test_failing_showquota_never_fails_the_checks(cli_env):
    write_shim(cli_env, "showquota", "", rc=1, stderr="showquota: cannot reach the quota server")
    c = one(preflight.check_quota(Context(grace(), cli_env.home)), "quota")
    assert c.level == "skip" and "cannot reach" in c.message


def test_doctor_level_shows_the_raw_showquota_lines(cli_env):
    write_shim(cli_env, "showquota", "/scratch/user/jdoe42  100 GB  1 TB  40,000  250,000")
    c = one(preflight.check_quota(Context(grace(), cli_env.home), verbose=True), "quota")
    assert "/scratch/user/jdoe42  100 GB" in c.message


# ---------------------------------------------------------------- account
def test_account_is_ok_unset_and_a_bad_string_is_an_error(cli_env):
    assert one(preflight.check_account(Context(grace(), cli_env.home)), "account").level == "ok"
    c = one(preflight.check_account(Context(grace(), cli_env.home, account="12 34")), "account")
    assert c.level == "error" and "myproject" in c.fix


def test_account_is_checked_against_myproject_when_it_exists(cli_env):
    ctx = Context(grace(), cli_env.home, account=FAKE_ACCOUNT)
    assert "command not found" in one(preflight.check_account(ctx), "account").message
    write_shim(cli_env, "myproject", f"Account  Balance\n{FAKE_ACCOUNT}  20000.0")
    assert "is listed" in one(preflight.check_account(ctx), "account").message
    write_shim(cli_env, "myproject", "Account  Balance\n987654321  20000.0")
    c = one(preflight.check_account(ctx), "account")
    assert c.level == "warn" and "not found" in c.message


# ---------------------------------------------------------------- the job environment
def test_missing_environment_uses_the_canonical_text_with_the_right_home_flag(cli_env):
    c = one(preflight.check_environment(Context(grace(), cli_env.home)), "python")
    assert c.level == "error"
    assert c.message.startswith(f"no Python environment at {cli_env.home / 'env'}. Run: bash scripts/setup_env.sh --cluster grace")
    assert f"--home {cli_env.home}" in c.message                 # not the profile's default home, so the fix repeats it


def test_fix_commands_leave_out_home_when_it_is_the_profile_default(cli_env, monkeypatch):
    monkeypatch.setenv("SCRATCH", str(cli_env.tmp / "scratch"))
    home = cli_env.tmp / "scratch" / "mpd"
    c = one(preflight.check_environment(Context(grace(), home)), "python")
    assert c.message.endswith("Run: bash scripts/setup_env.sh --cluster grace")


def test_user_given_python_that_does_not_exist_is_a_different_error(cli_env):
    c = one(preflight.check_environment(Context(grace(), cli_env.home, python=str(cli_env.tmp / "nope" / "python"))), "python")
    assert c.level == "error" and "does not exist" in c.message and "--python" in c.fix


def test_a_healthy_environment_reports_python_packages_and_record(cli_env):
    python = make_fake_python(cli_env.home / "env")
    write_record(cli_env.home, python)
    checks = preflight.check_environment(Context(grace(), cli_env.home, python=str(python)))
    assert [c.level for c in checks] == ["ok", "ok", "ok"]
    assert "Python 3.11.7" in one(checks, "python").message
    assert "torch 2.10.0, transformers 4.51.0, peft 0.13.2" in one(checks, "packages").message
    assert "Anaconda3/2024.02-1" in one(checks, "env-record").message


@pytest.mark.parametrize("kw,name,level,fragment", [
    (dict(version="3.10.14"), "python", "error", "needs 3.11"),
    (dict(packages={"proteinmutpd": "0.1.0.dev0"}), "packages", "error", "torch, transformers, peft not installed"),
    (dict(mpd_importable=False), "packages", "error", "mpd is not installed"),
    (dict(packages={"proteinmutpd": "0.1.0.dev0", "torch": "2.5.1", "transformers": "4.51.0", "peft": "0.13.2"}), "packages", "warn", "older than 2.7"),
    (dict(packages={"proteinmutpd": "0.0.1", "torch": "2.10.0", "transformers": "4.51.0", "peft": "0.13.2"}), "packages", "warn", "the jobs use mpd 0.0.1"),
])
def test_environment_problems(cli_env, kw, name, level, fragment):
    python = make_fake_python(cli_env.home / "env", **kw)
    write_record(cli_env.home, python)
    found = [c for c in preflight.check_environment(Context(grace(), cli_env.home, python=str(python))) if c.name == name]
    assert any(c.level == level and fragment in c.message for c in found), found


def test_a_python_that_cannot_run_the_probe_is_an_error_with_a_rebuild_hint(cli_env):
    python = make_fake_python(cli_env.home / "env", probe_error="ImportError: libpython3.11.so: cannot open shared object file")
    c = one(preflight.check_environment(Context(grace(), cli_env.home, python=str(python))), "python")
    assert c.level == "error" and "libpython3.11.so" in c.message and "--rebuild" in c.fix


def test_deep_imports_torch_in_the_job_python(cli_env):
    python = make_fake_python(cli_env.home / "env")
    write_record(cli_env.home, python)
    ctx = Context(grace(), cli_env.home, python=str(python))
    assert "no GPU visible here" in one(preflight.check_environment(ctx, deep=True), "torch-import").message
    assert not by_name(preflight.check_environment(ctx), "torch-import")           # not without --deep
    broken = make_fake_python(cli_env.tmp / "broken", torch_error="OSError: libcudnn.so.9: cannot open shared object file")
    c = one(preflight.check_environment(Context(grace(), cli_env.home, python=str(broken)), deep=True), "torch-import")
    assert c.level == "error" and "libcudnn" in c.message


def test_record_problems_are_warnings(cli_env):
    python = make_fake_python(cli_env.home / "env")
    ctx = Context(grace(), cli_env.home, python=str(python))
    assert one(preflight.check_environment(ctx), "env-record").level == "warn"          # missing
    (cli_env.home / "env" / "mpd_env.json").write_text("{not json", encoding="utf-8")
    assert "cannot be read" in one(preflight.check_environment(ctx), "env-record").message
    write_record(cli_env.home, python, cluster="vision", packages={"torch": "2.9.0", "transformers": "4.51.0", "peft": "0.13.2"})
    message = one(preflight.check_environment(ctx), "env-record").message
    assert "cluster 'vision'" in message and "torch is 2.10.0 now but was 2.9.0" in message


def test_record_notices_a_changed_constraints_file(cli_env):
    python = make_fake_python(cli_env.home / "env")
    constraints = cli_env.tmp / "grace.txt"
    constraints.write_text("torch==2.10.0\n", encoding="utf-8")
    write_record(cli_env.home, python, constraints=str(constraints), sha256=hashlib.sha256(constraints.read_bytes()).hexdigest())
    ctx = Context(grace(), cli_env.home, python=str(python))
    assert one(preflight.check_environment(ctx), "env-record").level == "ok"
    constraints.write_text("torch==2.11.0\n", encoding="utf-8")
    c = one(preflight.check_environment(ctx), "env-record")
    assert c.level == "warn" and "changed since the environment was built" in c.message


# ---------------------------------------------------------------- data and assay ids
def test_missing_data_uses_the_canonical_text(cli_env):
    ctx = Context(grace(), cli_env.home)
    c = preflight.check_dataset("proteingym", ctx)
    assert c.level == "error"
    assert c.message.startswith(f"ProteinGym data not found under {cli_env.home / 'data' / 'proteingym-v1.3'}. "
                                "Run on a login node: mpd fetch proteingym --cluster grace")
    assert preflight.check_dataset("venusmuthub", ctx).message.startswith("VenusMutHub data not found under ")
    assert preflight.check_dataset("venusmuthub", ctx, required=False).level == "warn"


def test_present_data_is_ok_and_a_half_fetched_venus_folder_is_a_warning(cli_env):
    write_proteingym(cli_env.home)
    write_venus(cli_env.home, fetched=False)
    ctx = Context(grace(), cli_env.home)
    assert preflight.check_dataset("proteingym", ctx).level == "ok"
    half = preflight.check_dataset("venusmuthub", ctx)
    assert half.level == "warn" and "no FETCHED.json" in half.message and half.fix.startswith("mpd fetch venusmuthub")
    write_venus(cli_env.home, fetched=True)
    assert preflight.check_dataset("venusmuthub", ctx).level == "ok"


def test_hand_placed_proteingym_without_a_marker_is_accepted_with_a_note(cli_env):
    write_proteingym(cli_env.home, fetched=False)
    c = preflight.check_dataset("proteingym", Context(grace(), cli_env.home))
    assert c.level == "ok" and "not hash-checked" in c.message


def test_data_checks_cover_train_and_eval_data_once_each(cli_env):
    names = [c.name for c in preflight.check_data(spec(eval_data=("proteingym", "venusmuthub")), Context(grace(), cli_env.home))]
    assert names == ["data:proteingym", "data:venusmuthub"]


def test_assay_ids_come_from_the_fetched_data(cli_env):
    write_proteingym(cli_env.home)
    write_venus(cli_env.home, ids=("ACT_one", "BIND_two"))
    assert preflight.assay_ids("proteingym", cli_env.home) == ["ASSAY_A", "ASSAY_B", "ASSAY_C"]
    assert preflight.assay_ids("venusmuthub", cli_env.home) == ["ACT_one", "BIND_two"]


def test_unknown_assay_ids_name_the_closest_and_the_command_that_lists_them(cli_env):
    write_proteingym(cli_env.home)
    ctx = Context(grace(), cli_env.home)
    ok = preflight.check_assays(spec(method="score-one", assays=("ASSAY_A",)), ctx)
    assert ok[0].level == "ok"
    bad = preflight.check_assays(spec(method="score-one", assays=("ASSAY_A", "ASSAY_Z", "NOT_AN_ASSAY")), ctx)[0]
    assert bad.level == "error" and "'ASSAY_Z' (closest: " in bad.message and "'NOT_AN_ASSAY'" in bad.message
    assert bad.fix == "mpd assays --dataset proteingym --grep <text>"


def test_assay_ids_are_not_checked_until_the_data_exists(cli_env):
    out = preflight.check_assays(spec(method="score-one", assays=("ASSAY_A",)), Context(grace(), cli_env.home))
    assert [c.level for c in out] == ["skip"]
    assert preflight.check_assays(spec(), Context(grace(), cli_env.home)) == []             # an *-all method names none


# ---------------------------------------------------------------- slurm tools and cost
def test_slurm_tools_by_level(cli_env):
    assert preflight.check_slurm_tools("doctor")[0].level == "skip"
    assert preflight.check_slurm_tools("submit")[0].level == "error"
    for tool in ("sbatch", "squeue", "sacct"):
        cli_env.link(tool, "/bin/true")
    c = preflight.check_slurm_tools("doctor")[0]
    assert c.level == "warn" and "scancel" in c.message
    cli_env.link("scancel", "/bin/true")
    assert preflight.check_slurm_tools("doctor")[0].level == "ok"


def test_a_dry_run_does_not_need_sbatch(cli_env):
    """`mpd submit --dry-run` works anywhere: a machine without sbatch is reported, but it is not an error there."""
    row = preflight.check_slurm_tools("submit", dry_run=True)[0]
    assert row.name == "slurm" and row.level == "skip" and "work anywhere" in row.message
    assert preflight.check_slurm_tools("submit", dry_run=False)[0].level == "error"
    assert not preflight.has_errors(preflight.check_slurm_tools("submit", dry_run=True))


def test_cost_guard_is_a_warning_in_a_plan_and_an_error_in_a_submit():
    plan = SimpleNamespace(total_su=348_702.0)
    assert preflight.check_cost(plan, 10_000, "plan")[0].level == "warn"
    c = preflight.check_cost(plan, 10_000, "submit")[0]
    assert c.level == "error" and "348,702 SU" in c.message and "Requested SUs" in c.message and "myproject" in c.message
    assert c.fix == "re-run with --max-su 350000 if you mean it"
    assert preflight.check_cost(plan, 1_000_000, "submit")[0].level == "ok"
    assert preflight.check_cost(None, 10_000, "submit") == []
    unbilled = SimpleNamespace(total_su=348_702.0, options=SimpleNamespace(cluster=Cluster.load("vision")))
    assert preflight.check_cost(unbilled, 100, "submit") == []                          # Vision has no SU rates


# ---------------------------------------------------------------- contributed checks
def test_a_missing_contributor_is_skipped_and_a_crashing_one_is_a_warning():
    assert preflight._contributed("x", "mpd.no_such_module", "f")[0].level == "skip"
    assert preflight._contributed("x", "mpd.checks", "no_such_function")[0].level == "skip"

    def boom():
        raise RuntimeError("kaput")
    mod = SimpleNamespace(boom=boom)
    sys.modules["mpd_fake_contributor"] = mod
    try:
        c = preflight._contributed("x", "mpd_fake_contributor", "boom")[0]
    finally:
        del sys.modules["mpd_fake_contributor"]
    assert c.level == "warn" and "kaput" in c.message


def test_the_models_module_contributes_the_canonical_weights_text(cli_env):
    checks = preflight.run_checks(spec(), Context(grace(), cli_env.home), level="plan")
    weights = [c for c in checks if c.name.startswith("weights ")]
    assert [c.name for c in weights] == ["weights esm2-3b", "weights esm2-650m"]
    assert all(c.level == "error" for c in weights)
    assert weights[0].message.startswith(f"weights for esm2-3b are missing ({cli_env.home / 'models' / 'esm2-3b'})")
    assert weights[0].fix == f"mpd fetch model esm2-3b --cluster grace --home {cli_env.home}"


def test_run_checks_by_level(cli_env, monkeypatch):
    ready_home(cli_env, monkeypatch)
    opts = Context(grace(), cli_env.home)
    plan_names = {c.name for c in preflight.run_checks(spec(), opts, level="plan")}
    assert {"home", "quota", "account", "python", "packages", "env-record", "data:proteingym"} <= plan_names
    assert "slurm" not in plan_names
    assert "slurm" in {c.name for c in preflight.run_checks(spec(), opts, level="submit")}
    doctor = preflight.run_checks(None, opts, level="doctor")
    assert {"weights", "foldseek", "slurm", "data:venusmuthub"} <= {c.name for c in doctor}
    with pytest.raises(ValueError, match="level"):
        preflight.run_checks(None, opts, level="bogus")


def test_doctor_weights_row_counts_overrides_as_ready_and_names_the_rest(cli_env, monkeypatch):
    override_weights(cli_env, monkeypatch, "esm2-650m")
    row = one(preflight.run_checks(None, Context(grace(), cli_env.home), level="doctor"), "weights")
    assert row.level == "ok" and "ready: esm2-650m" in row.message and "not fetched: esm2-8m" in row.message
    assert row.fix.startswith("mpd fetch model esm2-8m --cluster grace")


def test_a_slurm_job_context_is_called_out(cli_env, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    host = one(preflight.run_checks(None, Context(grace(), cli_env.home), level="doctor"), "host")
    assert host.level == "warn" and "no internet" in host.message


# ---------------------------------------------------------------- the table
def test_render_aligns_rows_puts_fixes_under_findings_and_ends_with_a_verdict():
    checks = [Check("home", "ok", "fine"), Check("python", "error", "no Python environment at /x/env. Run: bash scripts/setup_env.sh",
                                                  "bash scripts/setup_env.sh"),
              Check("weights esm2-3b", "warn", "missing", "mpd fetch model esm2-3b"), Check("quota", "skip", "showquota not found"),
              Check("torch for pickle weights", "ok", "torch 2.10.0"), Check("multi", "ok", "first line\nsecond line")]
    text = preflight.render(checks, mode="plan")
    lines = text.splitlines()
    assert lines[0].startswith("[ok]   home") and lines[1].startswith("[FAIL] python")
    assert "fix: bash scripts/setup_env.sh" not in text                       # already inside the message: not repeated
    assert any(ln.strip() == "fix: mpd fetch model esm2-3b" for ln in lines)
    long = lines.index("[ok]   torch for pickle weights")                      # a long name keeps the columns aligned
    assert lines[long + 1].lstrip() == "torch 2.10.0" and lines[long + 1].startswith(" " * 10)
    assert lines[-1].startswith("NOT READY: 1 error, 1 warning.") and "exit 0" in lines[-1]
    assert "second line" in text


@pytest.mark.parametrize("mode,fragment", [("plan", "exit 0"), ("dry-run", "A dry run only reports"),
                                           ("submit", "Nothing was submitted"), ("doctor", "mpd doctor")])
def test_not_ready_footer_says_what_the_mode_does(mode, fragment):
    assert fragment in preflight.render([Check("a", "error", "bad")], mode=mode).splitlines()[-1]


def test_ready_footer_is_careful_about_what_it_claims():
    text = preflight.render([Check("a", "ok", "fine"), Check("b", "warn", "meh"), Check("c", "warn", "meh")])
    assert text.splitlines()[-1].startswith("READY (2 warnings): no blocking problem found.")
    assert "do not prove that a job will run" in text
    assert preflight.render([Check("a", "ok", "fine")]).splitlines()[-1].startswith("READY: ")


# ---------------------------------------------------------------- scrubbing and --report
def test_scrub_removes_accounts_user_names_home_paths_tokens_and_addresses(cli_env, monkeypatch):
    monkeypatch.setenv("SCRATCH", "/scratch/user/jdoe42")
    github, hugging = "gh" + "p_" + "a" * 30, "hf_" + "b" * 30
    text = (f"account {FAKE_ACCOUNT} user {FAKE_USER} home {cli_env.userhome}/x scratch /scratch/user/jdoe42/mpd/env "
            f"mail jdoe42@example.edu tokens {github} {hugging} group /scratch/group/labname/data other 99999999999")
    clean = preflight.scrub(text, accounts=[FAKE_ACCOUNT], paths=[(cli_env.home, "<mpd-home>")])
    for secret in (FAKE_ACCOUNT, FAKE_USER, "labname", "example.edu", github, hugging, str(cli_env.userhome), "99999999999"):
        assert secret not in clean, (secret, clean)
    assert "<account>" in clean and "<netid>" in clean and "<token>" in clean and "<email>" in clean


def test_scrub_never_replaces_a_one_character_home(cli_env, monkeypatch):
    monkeypatch.setenv("HOME", "/")
    assert preflight.scrub("/usr/lib/python3") == "/usr/lib/python3"


def test_report_is_paste_safe(cli_env, monkeypatch):
    monkeypatch.setenv("MPD_ACCOUNT", FAKE_ACCOUNT)
    checks = [Check("account", "ok", f"{FAKE_ACCOUNT} is listed"),
              Check("home", "ok", f"{cli_env.home} (exists) of {FAKE_USER} at /home/{FAKE_USER}/x")]
    text = preflight.report(checks, Context(grace(), cli_env.home, account=FAKE_ACCOUNT))
    assert text.startswith("mpd ") and "doctor report" in text.splitlines()[0]
    assert "cluster grace" in text and "<mpd-home>" in text
    for secret in (FAKE_ACCOUNT, FAKE_USER, str(cli_env.home)):
        assert secret not in text


# ---------------------------------------------------------------- mpd doctor
def test_doctor_on_an_empty_home_exits_1_and_says_how_to_fix(capsys, cli_env):
    rc, out, err = run_cli(capsys, "doctor", "--cluster", "grace", "--home", cli_env.home)
    assert rc == 1
    assert "[FAIL] python" in out and "bash scripts/setup_env.sh --cluster grace" in out
    assert out.splitlines()[-1].startswith("NOT READY: 1 error")
    assert not cli_env.home.exists()


def test_doctor_on_a_ready_home_exits_0(capsys, cli_env, monkeypatch):
    ready_home(cli_env, monkeypatch)
    rc, out, _ = run_cli(capsys, "doctor", "--cluster", "grace", "--home", cli_env.home)
    assert rc == 0, out
    assert "[ok]   python" in out and "ready: esm2-650m, esm2-3b." in out
    assert out.splitlines()[-1].startswith("READY")


def test_doctor_uses_mpd_home_and_mpd_cluster_from_the_environment(capsys, cli_env, monkeypatch):
    monkeypatch.setenv("MPD_HOME", str(cli_env.home))
    monkeypatch.setenv("MPD_CLUSTER", "grace")
    assert run_cli(capsys, "doctor")[0] == 1
    rc, _, err = run_cli(capsys, "doctor", "--cluster", "grace", "--home", "/home/jdoe42/mpd")
    assert rc == 1                                           # a bad home is a finding, so exit 1 and a FAIL row, not a usage error


def test_doctor_reports_a_home_under_slash_home_as_a_failure(capsys, cli_env):
    rc, out, _ = run_cli(capsys, "doctor", "--cluster", "grace", "--home", "/home/jdoe42/mpd")
    assert rc == 1 and "[FAIL] home" in out and "Use scratch: export MPD_HOME=$SCRATCH/mpd" in out
    rc, out, _ = run_cli(capsys, "doctor", "--cluster", "grace", "--home", "/home/jdoe42/mpd", "--report")
    assert rc == 1 and "[FAIL] home" in out and FAKE_USER not in out and "/home/<name>/mpd" in out


def test_doctor_names_the_profile_variable_that_is_missing(capsys, cli_env):
    rc, out, _ = run_cli(capsys, "doctor", "--cluster", "grace")                     # no --home, no MPD_HOME, no SCRATCH
    assert rc == 1 and "[FAIL] home" in out and "$SCRATCH" in out and "Pass --home" in out


def test_doctor_needs_to_know_where_to_look(capsys, cli_env):
    rc, _, err = run_cli(capsys, "doctor")
    assert rc == 2 and "give --cluster" in err


def test_doctor_report_scrubs_the_account_user_and_paths(capsys, cli_env, monkeypatch):
    ready_home(cli_env, monkeypatch)
    monkeypatch.setenv("MPD_ACCOUNT", FAKE_ACCOUNT)
    write_shim(cli_env, "myproject", f"{FAKE_ACCOUNT} 20000")
    write_shim(cli_env, "showquota", f"/home/{FAKE_USER}  1 GB  10 GB  10  10000")
    rc, out, _ = run_cli(capsys, "doctor", "--cluster", "grace", "--home", cli_env.home, "--report")
    assert rc == 0 and "doctor report" in out
    for secret in (FAKE_ACCOUNT, FAKE_USER, str(cli_env.home), str(cli_env.userhome)):
        assert secret not in out, secret
    assert "<account>" in out or "listed by" not in out


def test_doctor_deep_checks_torch_in_the_job_python(capsys, cli_env, monkeypatch):
    ready_home(cli_env, monkeypatch)
    rc, out, _ = run_cli(capsys, "doctor", "--cluster", "grace", "--home", cli_env.home, "--deep")
    assert rc == 0 and "torch-import" in out
