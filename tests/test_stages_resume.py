"""Resuming a training run killed by its walltime (M3): exact continuation, atomic state, safe fallbacks."""

import csv
import hashlib
import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from fx_stages import (Killed, assert_same_training, env, finish_ddp_train, kill_after, plan_for,  # noqa: F401
                       run_stages, start_ddp_train)
from mpd.data.sources import PROTEINGYM
from mpd.stages import common, train
from tinydata import write_tiny_proteingym

MONITORS = {2, 4, 6}


# -- the saved state itself (no model needed) -------------------------------------------------------

def tiny_optimizer(steps=3):
    torch.manual_seed(0)
    params = [torch.nn.Parameter(torch.randn(3, 2)), torch.nn.Parameter(torch.randn(4))]
    optimizer = torch.optim.AdamW(params, lr=1e-2, weight_decay=0.01)
    for _ in range(steps):
        optimizer.zero_grad()
        sum((p ** 2).sum() for p in params).backward()
        optimizer.step()
    return params, ["a.weight", "b.bias"], optimizer


def save(out, step=3, total=10, digest="d1", resumed_from=()):
    params, names, optimizer = tiny_optimizer()
    train.save_partial(out, step=step, total=total, digest=digest, names=names, trainable=params, optimizer=optimizer,
                       elapsed=1.5, resumed_from=list(resumed_from))
    return params, names, optimizer


def test_saved_state_restores_parameters_and_optimizer_exactly(tmp_path):
    out = tmp_path / "all"
    params, names, optimizer = save(out)
    state = train.check_partial(train.partial_dir(out), "d1", 10)
    assert state["step"] == 3 and state["elapsed_seconds"] == 1.5 and set(state["files"]) == {"trainable.pt", "optimizer.pt"}
    fresh = [torch.nn.Parameter(torch.zeros_like(p)) for p in params]
    restored = torch.optim.AdamW(fresh, lr=1e-2, weight_decay=0.01)
    train.load_partial(train.partial_dir(out), fresh, names, restored)
    for a, b in zip(params, fresh):
        assert torch.equal(a, b)
    for a, b in zip(params, fresh):
        for key in ("exp_avg", "exp_avg_sq", "step"):
            assert torch.equal(optimizer.state[a][key], restored.state[b][key]), key
    for ps, opt in ((params, optimizer), (fresh, restored)):          # the next update is identical too
        opt.zero_grad()
        sum((p ** 3).sum() for p in ps).backward()
        opt.step()
    assert all(torch.equal(a, b) for a, b in zip(params, fresh))


def test_a_state_for_other_parameters_is_refused(tmp_path):
    out = tmp_path / "all"
    params, names, _ = save(out)
    other = torch.optim.AdamW(params, lr=1e-2)
    with pytest.raises(RuntimeError, match="not this model's trainable parameters"):
        train.load_partial(train.partial_dir(out), params, ["x", "y"], other)
    with pytest.raises(RuntimeError, match="shape"):
        train.load_partial(train.partial_dir(out), [torch.nn.Parameter(torch.zeros(5)), params[1]], names, other)


def test_only_the_latest_state_is_kept(tmp_path):
    out = tmp_path / "all"
    save(out, step=3)
    save(out, step=5, resumed_from=[2])
    assert sorted(p.name for p in tmp_path.iterdir()) == ["all.partial"]
    state = train.check_partial(train.partial_dir(out), "d1", 10)
    assert state["step"] == 5 and state["resumed_from"] == [2]
    assert sorted(p.name for p in train.partial_dir(out).iterdir()) == ["STATE.json", "optimizer.pt", "trainable.pt"]


def damage(folder: Path, how: str) -> str:
    if how == "flip":
        data = bytearray((folder / "trainable.pt").read_bytes())
        data[len(data) // 2] ^= 0xFF
        (folder / "trainable.pt").write_bytes(bytes(data))
        return "trainable.pt is missing or damaged"
    if how == "truncate":
        data = (folder / "optimizer.pt").read_bytes()
        (folder / "optimizer.pt").write_bytes(data[:len(data) // 2])
        return "optimizer.pt is missing or damaged"
    if how == "missing":
        (folder / "optimizer.pt").unlink()
        return "optimizer.pt is missing or damaged"
    if how == "unreadable":                                            # intact as far as the recorded sha256 goes
        junk = b"not a torch file"
        (folder / "trainable.pt").write_bytes(junk)
        state = json.loads((folder / "STATE.json").read_text())
        state["files"]["trainable.pt"] = {"bytes": len(junk), "sha256": hashlib.sha256(junk).hexdigest()}
        (folder / "STATE.json").write_text(json.dumps(state))
        return "trainable.pt cannot be loaded"
    if how == "no state":
        (folder / "STATE.json").unlink()
        return "STATE.json is unreadable"
    if how == "bad json":
        (folder / "STATE.json").write_text("{")
        return "STATE.json is unreadable"
    state = json.loads((folder / "STATE.json").read_text())
    if how == "schema":
        state["schema"] = 2
    if how == "no files":
        state["files"] = []
    (folder / "STATE.json").write_text(json.dumps(state))
    return {"schema": "is not the one this version writes", "no files": "lists no files"}[how]


@pytest.mark.parametrize("how", ["flip", "truncate", "missing", "unreadable", "no state", "bad json", "schema", "no files"])
def test_a_damaged_state_is_refused_with_the_reason(tmp_path, how):
    out = tmp_path / "all"
    save(out)
    reason = damage(train.partial_dir(out), how)
    with pytest.raises(ValueError, match=reason):
        train.check_partial(train.partial_dir(out), "d1", 10)


def test_a_state_of_another_configuration_or_a_finished_run_is_refused(tmp_path):
    save(tmp_path / "all", step=3)
    with pytest.raises(ValueError, match="different configuration"):
        train.check_partial(train.partial_dir(tmp_path / "all"), "other", 10)
    save(tmp_path / "done", step=10)
    with pytest.raises(ValueError, match=r"update 10 is outside 1\.\.9"):
        train.check_partial(train.partial_dir(tmp_path / "done"), "d1", 10)


def test_the_configuration_digest_ignores_provenance_and_changes_with_anything_else():
    config = {"unit": "all", "peak_lr": 3e-5, "updates": 6, "provenance": {"host": "a"}}
    digest = train.config_digest(config)
    assert train.config_digest({**config, "provenance": {"host": "b"}, "nonce": "x", "config_digest": "y"}) == digest
    assert train.config_digest({**config, "peak_lr": 6e-5}) != digest and train.config_digest({**config, "extra": 1}) != digest


def test_checkpoint_interval_defaults_to_200_and_takes_argument_then_environment(monkeypatch):
    monkeypatch.delenv("MPD_CKPT_EVERY", raising=False)
    assert train.checkpoint_interval(None) == 200
    monkeypatch.setenv("MPD_CKPT_EVERY", "50")
    assert train.checkpoint_interval(None) == 50 and train.checkpoint_interval(7) == 7
    monkeypatch.setenv("MPD_CKPT_EVERY", "0")
    assert train.checkpoint_interval(None) == 0
    monkeypatch.setenv("MPD_CKPT_EVERY", "")
    assert train.checkpoint_interval(None) == 200


@pytest.mark.parametrize("bad", ["abc", "-3", "1.5"])
def test_a_bad_checkpoint_interval_is_a_clear_error(monkeypatch, bad):
    monkeypatch.setenv("MPD_CKPT_EVERY", bad)
    with pytest.raises(SystemExit, match="MPD_CKPT_EVERY must be a whole number of updates"):
        train.checkpoint_interval(None)


# -- claiming the output folder -------------------------------------------------------------------

def fake_run(out: Path, logged: int) -> None:
    """A train/<unit> folder as an interrupted run leaves it, logged up to update `logged` of a 6-update run."""
    out.mkdir(parents=True)
    (out / "CONFIG.json").write_text("{}")
    lines = [{"step": s, "dev_loss": 1.0 / (1 + s)} for s in [0, *sorted(MONITORS)] if s <= logged]
    (out / "dev_curve.jsonl").write_text("".join(json.dumps(row) + "\n" for row in lines))
    with open(out / "train_steps.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "loss", "huber", "lr", "grad_norm", "elapsed_seconds"])
        w.writerows([s, 0.5, 0.5, 1e-5, 1.0, s] for s in range(1, logged + 1))
    for m in sorted(MONITORS):
        if m <= logged:
            (out / f"checkpoint_{m:05d}").mkdir()
            (out / f"checkpoint_{m:05d}" / "adapter_model.safetensors").write_text("x")


def steps_of(out: Path) -> tuple[list[int], list[int], list[str]]:
    curve = [json.loads(line)["step"] for line in (out / "dev_curve.jsonl").read_text().splitlines()]
    rows = [int(r["step"]) for r in csv.DictReader(open(out / "train_steps.csv"))]
    return curve, rows, sorted(p.name for p in out.glob("checkpoint_*"))


def test_claiming_cuts_the_logs_and_adapters_back_to_the_checkpoint(tmp_path):
    out = tmp_path / "all"
    fake_run(out, logged=6)                                            # the run got to the end, then was killed
    save(out, step=4, total=6)
    state = train.claim_output(out, "d1", 6, MONITORS)
    assert state["step"] == 4
    assert steps_of(out) == ([0, 2, 4], [1, 2, 3, 4], ["checkpoint_00002", "checkpoint_00004"])
    assert not list(tmp_path.glob("*incomplete*"))


def test_claiming_survives_lines_cut_in_half_by_a_kill(tmp_path):
    out = tmp_path / "all"
    fake_run(out, logged=5)
    save(out, step=4, total=6)
    with open(out / "train_steps.csv", "a") as fh:
        fh.write("5,0.12")
    with open(out / "dev_curve.jsonl", "a") as fh:
        fh.write('{"step": 6, "dev_l')
    assert train.claim_output(out, "d1", 6, MONITORS)["step"] == 4
    assert steps_of(out)[:2] == ([0, 2, 4], [1, 2, 3, 4])


def test_an_attempt_without_a_checkpoint_is_moved_aside(tmp_path, capsys):
    out = tmp_path / "all"
    fake_run(out, logged=3)
    assert train.claim_output(out, "d1", 6, MONITORS) is None
    assert out.is_dir() and not list(out.iterdir())
    aside = list(tmp_path.glob("all.incomplete-*"))
    assert len(aside) == 1 and (aside[0] / "CONFIG.json").is_file()
    assert f"moved an incomplete earlier attempt to {aside[0].name}" in capsys.readouterr().out


def test_a_first_attempt_just_gets_a_folder(tmp_path, capsys):
    out = tmp_path / "train" / "all"
    assert train.claim_output(out, "d1", 6, MONITORS) is None
    assert out.is_dir() and capsys.readouterr().out == ""


@pytest.mark.parametrize("fault,reason", [
    (lambda out: shutil.rmtree(out / "checkpoint_00002"), "checkpoint_00002 is missing"),
    (lambda out: (out / "train_steps.csv").write_text("step\n1\n2\n"), "train_steps.csv lacks rows up to update 4"),
    (lambda out: (out / "dev_curve.jsonl").write_text('{"step": 0}\n'), r"dev_curve.jsonl lacks update\(s\) \[2, 4\]"),
    (lambda out: (out / "train_steps.csv").unlink(), "the logs cannot be read"),
    (lambda out: (out / "CONFIG.json").unlink(), "CONFIG.json is missing"),
])
def test_a_checkpoint_whose_logs_do_not_cover_it_is_not_resumed(tmp_path, capsys, fault, reason):
    out = tmp_path / "all"
    fake_run(out, logged=4)
    save(out, step=4, total=6)
    fault(out)
    assert train.claim_output(out, "d1", 6, MONITORS) is None
    said = capsys.readouterr().out
    assert "not resuming: all.partial: " in said and re.search(reason, said), said
    assert len(list(tmp_path.glob("all.incomplete-*"))) == 1 and len(list(tmp_path.glob("all.partial.incomplete-*"))) == 1


def test_a_crash_between_the_two_renames_of_a_checkpoint_is_recovered(tmp_path):
    out = tmp_path / "all"
    fake_run(out, logged=4)
    save(out, step=4, total=6)
    train.partial_dir(out).rename(tmp_path / "all.partial.old")        # save_partial died after moving the old copy away
    (tmp_path / "all.partial.tmp").mkdir()
    (tmp_path / "all.partial.tmp" / "trainable.pt").write_text("half")
    assert train.claim_output(out, "d1", 6, MONITORS)["step"] == 4
    assert sorted(p.name for p in tmp_path.iterdir()) == ["all", "all.partial"]


def test_a_damaged_newest_copy_falls_back_to_the_previous_one(tmp_path):
    out = tmp_path / "all"
    fake_run(out, logged=4)
    save(out, step=2, total=6)
    shutil.copytree(train.partial_dir(out), tmp_path / "older")
    save(out, step=4, total=6)
    (tmp_path / "older").rename(tmp_path / "all.partial.old")          # as if save_partial had not yet deleted it
    damage(train.partial_dir(out), "flip")
    state = train.claim_output(out, "d1", 6, MONITORS)
    assert state["step"] == 2 and steps_of(out) == ([0, 2], [1, 2], ["checkpoint_00002"])
    assert train.check_partial(train.partial_dir(out), "d1", 6)["step"] == 2


# -- killing and resuming real runs --------------------------------------------------------------

@pytest.fixture(scope="module")
def base(tmp_path_factory, tiny_models):
    """A prepared campaign with its teacher table, and the uninterrupted run (every monitored adapter kept) to compare with."""
    root = tmp_path_factory.mktemp("resume")
    home = root / "home"
    data = home / "data" / PROTEINGYM.dirname
    data.mkdir(parents=True)
    write_tiny_proteingym(data)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("MPD_HOME", str(home))
        mp.setenv("MPD_DTYPE", "float32")
        mp.setenv("MPD_MODEL_ESM2_8M", str(tiny_models["student"]))
        mp.setenv("MPD_MODEL_ESM2_35M", str(tiny_models["teacher"]))
        mp.setenv("MPD_KEEP_ALL_CHECKPOINTS", "1")
        plan = plan_for(home)
        run_stages(plan, only={"prepare", "teacher"})
        train.run(common.Campaign.load(plan.campaign_dir), {"unit": "all"}, ckpt_every=0)
        shutil.move(plan.campaign_dir / "train" / "all", root / "reference")
    return SimpleNamespace(home=home, campaign=plan.campaign_dir, reference=root / "reference", models=tiny_models)


@pytest.fixture
def run_in(base, monkeypatch, tmp_path):
    """A private copy of the prepared campaign; `go()` runs its train stage."""
    monkeypatch.setenv("MPD_HOME", str(base.home))
    monkeypatch.setenv("MPD_DTYPE", "float32")
    monkeypatch.setenv("MPD_MODEL_ESM2_8M", str(base.models["student"]))
    monkeypatch.setenv("MPD_MODEL_ESM2_35M", str(base.models["teacher"]))
    monkeypatch.setenv("MPD_KEEP_ALL_CHECKPOINTS", "1")
    monkeypatch.delenv("MPD_CKPT_EVERY", raising=False)
    root = tmp_path / "runs" / "campaign"
    shutil.copytree(base.campaign, root)
    out = root / "train" / "all"

    def go(**kw):
        train.run(common.Campaign.load(root), {"unit": "all"}, **kw)

    return SimpleNamespace(root=root, out=out, ref=base.reference, go=go,
                           saved_step=lambda: json.loads((train.partial_dir(out) / "STATE.json").read_text())["step"],
                           complete=lambda: json.loads((out / "COMPLETE.json").read_text()))


def test_the_uninterrupted_reference_is_a_complete_run(base):
    assert [json.loads(line)["step"] for line in (base.reference / "dev_curve.jsonl").read_text().splitlines()] == [0, 2, 4, 6]
    assert sorted(p.name for p in base.reference.glob("checkpoint_*")) == [f"checkpoint_{m:05d}" for m in sorted(MONITORS)]


@pytest.mark.parametrize("kill,saved", [(3, 2), (4, 4), (5, 4)])
def test_a_killed_run_resumes_to_exactly_the_uninterrupted_result(run_in, capsys, kill, saved):
    with pytest.raises(Killed):
        run_in.go(ckpt_every=2, on_update=kill_after(kill))
    assert run_in.saved_step() == saved and not (run_in.out / "COMPLETE.json").exists()
    run_in.go(ckpt_every=2)
    assert f"resuming from update {saved} of 6" in capsys.readouterr().out
    assert_same_training(run_in.ref, run_in.out)
    assert run_in.complete()["resumed_from"] == [saved] and run_in.complete()["updates"] == 6
    assert json.loads((run_in.out / "SELECTED.json").read_text())["step"] == json.loads((run_in.ref / "SELECTED.json").read_text())["step"]
    assert not list((run_in.root / "train").glob("*incomplete*")) and not train.partial_dir(run_in.out).exists()


def test_two_kills_in_a_row_still_end_at_the_same_result(run_in):
    with pytest.raises(Killed):
        run_in.go(ckpt_every=2, on_update=kill_after(3))
    with pytest.raises(Killed):
        run_in.go(ckpt_every=2, on_update=kill_after(5))
    run_in.go(ckpt_every=2)
    assert_same_training(run_in.ref, run_in.out)
    assert run_in.complete()["resumed_from"] == [2, 4]


def test_the_resumed_run_keeps_counting_time_and_records_where_it_ran(run_in):
    with pytest.raises(Killed):
        run_in.go(ckpt_every=2, on_update=kill_after(5))
    elapsed = json.loads((train.partial_dir(run_in.out) / "STATE.json").read_text())["elapsed_seconds"]
    run_in.go(ckpt_every=2)
    done = run_in.complete()
    assert elapsed > 0 and done["seconds"] >= round(elapsed, 1)         # `seconds` is rounded and counts every attempt
    assert done["provenance"]["torch"] and done["config_digest"]
    assert json.loads((run_in.out / "CONFIG.json").read_text())["config_digest"] == done["config_digest"]


def test_a_run_killed_before_its_first_checkpoint_starts_again(run_in, capsys):
    with pytest.raises(Killed):
        run_in.go(ckpt_every=2, on_update=kill_after(1))
    assert not train.partial_dir(run_in.out).exists()
    run_in.go(ckpt_every=2)
    assert "moved an incomplete earlier attempt to all.incomplete-" in capsys.readouterr().out
    assert len(list((run_in.root / "train").glob("all.incomplete-*"))) == 1
    assert_same_training(run_in.ref, run_in.out)
    assert run_in.complete()["resumed_from"] == []


def test_logs_cut_off_by_a_kill_do_not_stop_the_resume(run_in):
    with pytest.raises(Killed):
        run_in.go(ckpt_every=2, on_update=kill_after(5))
    with open(run_in.out / "train_steps.csv", "a") as fh:
        fh.write("5,0.12")
    with open(run_in.out / "dev_curve.jsonl", "a") as fh:
        fh.write('{"step": 6, "dev_l')
    run_in.go(ckpt_every=2)
    assert_same_training(run_in.ref, run_in.out)


def test_checkpoints_can_be_turned_off_and_default_to_every_200_updates(run_in, monkeypatch):
    seen = []
    run_in.go(ckpt_every=0, on_update=lambda step: seen.append(train.partial_dir(run_in.out).exists()))
    assert seen == [False] * 6
    shutil.rmtree(run_in.out)
    seen.clear()
    run_in.go(on_update=lambda step: seen.append(train.partial_dir(run_in.out).exists()))      # default: none in 6 updates
    assert seen == [False] * 6
    shutil.rmtree(run_in.out)
    seen.clear()
    monkeypatch.setenv("MPD_CKPT_EVERY", "3")                           # the environment variable sets the interval
    run_in.go(on_update=lambda step: seen.append(train.partial_dir(run_in.out).exists()))
    assert seen == [False, False, True, True, True, True]
    assert not train.partial_dir(run_in.out).exists()                   # gone once the unit is complete


def test_a_different_configuration_is_not_resumed(run_in, capsys):
    with pytest.raises(Killed):
        run_in.go(ckpt_every=2, on_update=kill_after(3))
    resolved = json.loads((run_in.root / "resolved.json").read_text())
    resolved["peak_lr"] *= 2
    (run_in.root / "resolved.json").write_text(json.dumps(resolved))
    run_in.go(ckpt_every=2)
    said = capsys.readouterr().out
    assert "not resuming: all.partial: it was written by a run with a different configuration" in said
    assert len(list((run_in.root / "train").glob("all.incomplete-*"))) == 1
    assert len(list((run_in.root / "train").glob("all.partial.incomplete-*"))) == 1
    assert run_in.complete()["resumed_from"] == [] and run_in.complete()["status"] == "PASS"


def test_a_damaged_checkpoint_is_not_resumed(run_in, capsys):
    with pytest.raises(Killed):
        run_in.go(ckpt_every=2, on_update=kill_after(3))
    damage(train.partial_dir(run_in.out), "flip")
    run_in.go(ckpt_every=2)
    assert "not resuming: all.partial: trainable.pt is missing or damaged" in capsys.readouterr().out
    assert_same_training(run_in.ref, run_in.out)                        # a fresh run is deterministic too


def test_a_crash_between_the_two_renames_of_a_checkpoint_is_resumed_from_the_old_copy(run_in):
    with pytest.raises(Killed):
        run_in.go(ckpt_every=2, on_update=kill_after(5))
    train.partial_dir(run_in.out).rename(run_in.root / "train" / "all.partial.old")
    (run_in.root / "train" / "all.partial.tmp").mkdir()
    run_in.go(ckpt_every=2)
    assert_same_training(run_in.ref, run_in.out)
    assert run_in.complete()["resumed_from"] == [4]
    assert sorted(p.name for p in (run_in.root / "train").iterdir()) == ["all"]


def test_a_complete_unit_discards_a_leftover_checkpoint(run_in, capsys):
    run_in.go(ckpt_every=0)
    leftover = train.partial_dir(run_in.out)
    leftover.mkdir()
    (leftover / "STATE.json").write_text("{}")
    capsys.readouterr()
    run_in.go(ckpt_every=2)
    assert "already complete" in capsys.readouterr().out and not leftover.exists()


def test_a_checkpoint_that_cannot_be_written_does_not_stop_training(run_in, monkeypatch, capsys):
    def disk_full(*_a, **_k):
        raise OSError("No space left on device")

    monkeypatch.setattr(train, "save_partial", disk_full)
    run_in.go(ckpt_every=2)
    said = capsys.readouterr().out
    assert said.count("could not write the resumable checkpoint") == 2 and "No space left on device" in said
    assert_same_training(run_in.ref, run_in.out)


def test_a_score_objective_run_resumes_exactly_too(env, monkeypatch, tmp_path):
    monkeypatch.setenv("MPD_KEEP_ALL_CHECKPOINTS", "1")
    plan = plan_for(env, method="score-all")
    run_stages(plan, only={"prepare", "teacher"})
    first, second = plan.campaign_dir, tmp_path / "runs" / "second"
    shutil.copytree(first, second)
    train.run(common.Campaign.load(first), {"unit": "all"}, ckpt_every=0)
    with pytest.raises(Killed):
        train.run(common.Campaign.load(second), {"unit": "all"}, ckpt_every=2, on_update=kill_after(5))
    train.run(common.Campaign.load(second), {"unit": "all"}, ckpt_every=2)
    assert_same_training(first / "train" / "all", second / "train" / "all")


# -- two processes ------------------------------------------------------------------------------------

def test_a_two_process_run_killed_part_way_resumes_to_the_uninterrupted_result(env, monkeypatch, tmp_path):
    """Gloo on CPU, global batch 8 over 2 processes = 2 micro-batches each (the no_sync accumulation path)."""
    monkeypatch.setenv("MPD_KEEP_ALL_CHECKPOINTS", "1")
    monkeypatch.setenv("MPD_CKPT_EVERY", "2")
    plan = plan_for(env, train_gpus=2, global_batch=8, peak_lr=3e-3)
    assert plan.resolved["world_size"] == 2 and plan.resolved["accumulation"] == 2
    run_stages(plan, only={"prepare", "teacher"})
    second = tmp_path / "runs" / "second"
    shutil.copytree(plan.campaign_dir, second)

    reference = start_ddp_train(plan, tmp_path, "reference")           # uninterrupted, at the same time as the killed one
    killed = start_ddp_train(second, tmp_path, "killed", stop_after=5)
    code, log = finish_ddp_train(killed)
    assert code != 0 and "simulated kill after update 5" in log, log[-3000:]
    assert json.loads((second / "train" / "all.partial" / "STATE.json").read_text())["step"] == 4
    code, log = finish_ddp_train(start_ddp_train(second, tmp_path, "resumed"))
    assert code == 0 and "resuming from update 4 of 6" in log, log[-3000:]
    code, log = finish_ddp_train(reference)
    assert code == 0, log[-3000:]

    assert_same_training(plan.campaign_dir / "train" / "all", second / "train" / "all")
    done = json.loads((second / "train" / "all" / "COMPLETE.json").read_text())
    assert done["resumed_from"] == [4] and not (second / "train" / "all.partial").exists()
    assert not list((second / "train").glob("*incomplete*"))


def test_one_rank_failing_alone_ends_the_whole_launch_long_before_the_group_timeout(env, tmp_path):
    """Rank 0 dies after update 3 while rank 1 is already inside the next update's all-reduce."""
    plan = plan_for(env, train_gpus=2, global_batch=8)
    run_stages(plan, only={"prepare", "teacher"})
    code, log = finish_ddp_train(start_ddp_train(plan, tmp_path, "rank0", stop_after=3, only_rank=0), timeout=300)
    assert code != 0 and "simulated kill after update 3" in log, log[-3000:]
    assert not (plan.campaign_dir / "train" / "all" / "COMPLETE.json").exists()
