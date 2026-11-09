"""What `mpd logs` and `mpd doctor --report` print is safe to paste into a public issue unless the user asks for the raw text."""

import getpass
import os
import re
import stat
from types import SimpleNamespace

import pytest

from fx_cli import cli_env, run_cli  # noqa: F401  (cli_env is a pytest fixture)
from fx_slurm import stub_plan, write_stub_campaign

from mpd import preflight

ACCOUNT = "123456789012"


def log_with_secrets(plan, stage="train", ext="err"):
    user = getpass.getuser()
    logs = plan.campaign_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / f"mpd-{stage}-esm2650m_1001.{ext}").write_text(
        f"File \"/scratch/user/{user}/mpd/env/lib/python3.11/x.py\", line 3\n"
        f"charged to account {ACCOUNT} by {user}@example.org\nthe job of {user}-env ran\nplain line\n", encoding="utf-8")


def test_mpd_logs_hides_account_user_and_email_by_default(capsys, cli_env):
    plan = stub_plan(cli_env.tmp)
    write_stub_campaign(plan, state="submitted", jobs={"train": "1001"})
    log_with_secrets(plan)
    rc, out, err = run_cli(capsys, "logs", str(plan.campaign_dir), "train", "--err")
    user = getpass.getuser()
    assert rc == 0 and "plain line" in out
    assert ACCOUNT not in out and f"{user}@" not in out and "<account>" in out and "<email>" in out
    if len(user) > 2:
        assert not re.search(rf"(?<![A-Za-z0-9]){re.escape(user)}(?![A-Za-z0-9])", out)


def test_mpd_logs_raw_prints_the_file_as_written(capsys, cli_env):
    plan = stub_plan(cli_env.tmp)
    write_stub_campaign(plan, state="submitted", jobs={"train": "1001"})
    log_with_secrets(plan)
    rc, out, err = run_cli(capsys, "logs", str(plan.campaign_dir), "train", "--err", "--raw")
    assert rc == 0 and ACCOUNT in out and getpass.getuser() in out


def fake_showquota(tmp_path, text):
    bindir = tmp_path / "bin"
    bindir.mkdir()
    tool = bindir / "showquota"
    tool.write_text("#!/bin/bash\ncat <<'EOF'\n" + text + "\nEOF\n")
    tool.chmod(tool.stat().st_mode | stat.S_IXUSR)
    return bindir


def test_the_doctor_report_does_not_copy_the_raw_showquota_lines(tmp_path, monkeypatch):
    bindir = fake_showquota(tmp_path, "Disk quota for Alice Example (lab: mylab)\n/scratch/user/someone  120 GB / 1 TB  12%\nGroup mylab-hpc")
    monkeypatch.setenv("PATH", f"{bindir}:{os.environ['PATH']}")
    checks = preflight.check_quota(SimpleNamespace(), verbose=True)
    assert "Alice Example" in checks[0].message                         # on your own screen the raw lines help
    opts = SimpleNamespace(cluster=None, account=None, home=tmp_path / "home")
    text = preflight.report(checks, opts)
    assert "Alice Example" not in text and "mylab" not in text and "showquota: highest usage 12%" in text


@pytest.mark.parametrize("text,gone", [
    ("see account 999999999999. It was listed", "999999999999"),
    ("the env /x/envs/{u}-env/bin/python", "{u}"),
    ("jobs {u}_job and user-{u}", "{u}"),
])
def test_the_report_scrubber_has_the_same_boundaries_as_the_log_scrubber(text, gone):
    user = getpass.getuser()
    if len(user) <= 2:
        pytest.skip("user name too short to scrub")
    out = preflight.scrub(text.replace("{u}", user))
    assert gone.replace("{u}", user) not in out
