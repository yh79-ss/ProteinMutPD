"""`prepare` looks at the training/development split before any GPU stage runs: a unit that `train` would reject is rejected here, on a CPU node."""

import csv
import json
import re

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from fx_stages import env, plan_for, run_stages  # noqa: E402,F401

from mpd.contexts import ContextTable  # noqa: E402
from mpd.data.sources import PROTEINGYM  # noqa: E402


def keep_one_position(home, assay: str) -> None:
    """Cut an assay of the tiny ProteinGym down to the variants of its first measured position."""
    path = home / "data" / PROTEINGYM.dirname / "DMS_ProteinGym_substitutions" / f"{assay}.csv"
    rows = list(csv.DictReader(open(path)))
    first = re.fullmatch(r"[A-Z](\d+)[A-Z]", rows[0]["mutant"]).group(1)
    kept = [r for r in rows if re.fullmatch(rf"[A-Z]{first}[A-Z]", r["mutant"])]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(kept)


def test_a_one_position_assay_stops_prepare_before_the_gpu_stages(env, capsys):
    keep_one_position(env, "ASSAY_B")
    plan = plan_for(env, method="score-one", assays=("ASSAY_A", "ASSAY_B"))
    with pytest.raises(SystemExit) as stop:
        run_stages(plan, only={"prepare"})
    message = str(stop.value)
    assert "ASSAY_B" in message and "ASSAY_A" not in message and "at least two" in message and "new campaign" in message
    assert not (plan.campaign_dir / "contexts" / "MANIFEST.json").exists()           # the stage did not finish, so nothing downstream starts


def test_prepare_records_the_split_that_train_will_use_and_says_how_big_the_development_share_is(env, capsys):
    plan = plan_for(env)
    run_stages(plan, only={"prepare"})
    manifest = json.loads((plan.campaign_dir / "contexts" / "MANIFEST.json").read_text())
    train, dev = manifest["unit_split"]["all"]
    assert train + dev == manifest["unit_contexts"]["all"] and train > 0 and dev > 0 and manifest["dev_fraction"] == 0.3
    mask = ContextTable.load(plan.campaign_dir / "contexts" / "unit__all").dev_mask(manifest["dev_fraction"])      # what `train` computes
    assert [int((~mask).sum()), int(mask.sum())] == [train, dev]
    out = capsys.readouterr().out
    assert re.search(r"development split: \d+ of \d+ contexts \(\d+\.\d%\)", out) and "min(5, n-1)" in out


def test_a_unit_with_few_training_contexts_is_noted_but_still_prepared(env, capsys):
    plan = plan_for(env, method="score-one", assays=("ASSAY_A",), dev_fraction=0.05)
    run_stages(plan, only={"prepare"})
    out = capsys.readouterr().out
    manifest = json.loads((plan.campaign_dir / "contexts" / "MANIFEST.json").read_text())
    train, dev = manifest["unit_split"]["ASSAY_A"]
    assert dev >= 5 and train + dev == 12                    # twelve measured positions: the floor of five development contexts applies
    assert train < 10 and "ASSAY_A (%d training, %d development)" % (train, dev) in out and "fewer than 10 contexts" in out
