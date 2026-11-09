"""What the command line says about itself: no promise it cannot keep. Nothing in this repository has run on Grace through Slurm, so
help texts, messages and advice must not say that something works, is guaranteed, or was verified there (CONTRACTS 1.7).
"""

import argparse
import re

import pytest
from fx_cli import FAKE_ACCOUNT, cli_env, ready_home, recipe, run_cli  # noqa: F401
from fx_slurm import fake_slurm  # noqa: F401

from mpd import cli
from mpd.slurm import campaign, plan as planner, submit as submit_mod

pytestmark = pytest.mark.usefixtures("cli_env")

# "Works" with a capital is the licence term "Derivative Works" (the AIDO licence note), not a claim
PROMISES = re.compile(r"\bworks\b|\b[Jj]ust\b|[Gg]uarantee|12-digit|(?i:(?:verified|tested|works|runs?|ran) on grace)|will work|always works")


def promises(text: str) -> list[str]:
    return sorted({m.group(0) for m in PROMISES.finditer(text)})


def all_help() -> dict[str, str]:
    parser = cli._parser()
    texts = {"mpd": parser.format_help()}
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    texts.update({f"mpd {name}": p.format_help() for name, p in sub.choices.items()})
    return texts


@pytest.mark.parametrize("name,text", sorted(all_help().items()))
def test_no_help_text_promises_anything(name, text):
    assert promises(text) == [], name


def test_the_advice_and_the_notes_promise_nothing():
    texts = [*campaign.ADVICE.values(), *campaign.CAUSE_ADVICE.values(), *campaign.PENDING_REASONS.values(),
             *(t for _, t in campaign.PENDING_FAMILIES), submit_mod.FORCE_APPLIES, *planner.ID_NOTE, cli.EPILOG]
    texts += [value for name, value in vars(submit_mod).items() if name.startswith("_HINT_") and isinstance(value, str)]
    assert [(t[:40], promises(t)) for t in texts if promises(t)] == []


def test_what_plan_doctor_and_a_dry_run_print_on_an_empty_home_promise_nothing(capsys, cli_env):
    for argv in (["plan", *recipe(cli_env.home)], ["submit", *recipe(cli_env.home), "--dry-run"], ["doctor", "--cluster", "grace", "--home", cli_env.home],
                 ["clusters"]):
        rc, out, err = run_cli(capsys, *argv)
        assert promises(out + err) == [], (argv[0], promises(out + err))


def test_what_the_recovery_commands_print_promises_nothing(capsys, cli_env, monkeypatch, fake_slurm):
    ready_home(cli_env, monkeypatch)
    args = ["submit", *recipe(cli_env.home, "--account", FAKE_ACCOUNT), "--yes"]
    first, refused, forced = run_cli(capsys, *args), run_cli(capsys, *args), run_cli(capsys, *args, "--force")
    (root,) = (cli_env.home / "runs").iterdir()
    status = run_cli(capsys, "status", root.name, "--home", cli_env.home)
    for rc, out, err in (first, refused, forced, status):
        assert promises(out + err) == []
    assert refused[0] == 2 and "was already submitted" in refused[2] and "nothing to submit" in forced[1] and status[0] == 0


def test_the_scheduling_options_say_in_their_help_that_they_keep_the_campaign(capsys):
    with pytest.raises(SystemExit):
        cli.main(["submit", "--help"])
    flat = " ".join(capsys.readouterr().out.split())
    assert "The walltimes, --array-throttle, --account, --home and --python only decide where and when the jobs run: they keep the campaign" in flat
    assert "Another --cluster, --gpu, --train-gpus or --score-shards (and another --eval-data) is another campaign" in flat
    assert "a longer --train-time after a TIMEOUT, --sbatch-arg=--exclude=NODE after a bad node" in flat
    assert "except MPD_DTYPE, MPD_TF32 and MPD_STRUCTURE_LEGACY_MASK" in flat


def test_the_plan_says_what_the_campaign_id_depends_on(capsys, cli_env):
    rc, out, _ = run_cli(capsys, "plan", *recipe(cli_env.home))
    assert rc == 0
    flat = " ".join(out.split())
    assert "campaign id: a new --eval-data, --gpu, --train-gpus or --score-shards" in flat and "is a new campaign and trains again from the start" in flat
    assert "Walltimes, --array-throttle, --sbatch-arg, --module, other --env pairs and --account keep the id" in flat
