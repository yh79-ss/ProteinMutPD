"""Teacher stage: skip when finished, recompute when the contexts differ, atomic table; train checks the table (M6)."""

import json

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from fx_stages import env, plan_for, run_stages  # noqa: F401
from mpd.contexts import ContextTable
from mpd.stages import teacher


def finished(env, **kw):
    plan = plan_for(env, **kw)
    run_stages(plan, only={"prepare", "teacher"})
    return plan, plan.campaign_dir / "teacher"


def edit_record(folder, **changes):
    path = folder / "COMPLETE.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), **changes}))


def never(*_a, **_k):
    raise AssertionError("the teacher was run again")


def test_a_finished_teacher_is_skipped_with_one_line(env, monkeypatch, capsys):
    plan, folder = finished(env)
    before = (folder / "logits.npy").stat().st_mtime_ns, (folder / "COMPLETE.json").read_text()
    capsys.readouterr()
    monkeypatch.setattr(teacher, "aa20_logits", never)
    run_stages(plan, only={"teacher"})
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and "already computed" in out[0] and "esm2-35m" in out[0]
    assert ((folder / "logits.npy").stat().st_mtime_ns, (folder / "COMPLETE.json").read_text()) == before


def test_the_record_carries_provenance_and_the_table_hash(env):
    _, folder = finished(env)
    info = json.loads((folder / "COMPLETE.json").read_text())
    assert info["status"] == "PASS" and info["teacher"] == "esm2-35m" and info["dtype"] == "torch.float32"
    assert info["provenance"]["torch"] and info["provenance"]["mpd"] and info["peak_gpu_memory_gb"] is None
    assert info["context_digest"] == ContextTable.load(folder.parent / "contexts" / "union").digest()


@pytest.mark.parametrize("change,reason", [
    ({"context_digest": "ab" * 32}, "computed for contexts abababababab"),
    ({"teacher": "esm2-650m"}, "computed by esm2-650m"),
    ({"logits_sha256": "0" * 64}, "does not match the sha256"),
    ({"status": "FAIL"}, "does not say PASS"),
])
def test_a_table_for_other_contexts_or_another_teacher_is_recomputed(env, capsys, change, reason):
    plan, folder = finished(env)
    good = np.load(folder / "logits.npy")
    edit_record(folder, **change)
    capsys.readouterr()
    run_stages(plan, only={"teacher"})
    out = capsys.readouterr().out
    assert "recomputing" in out and reason in out
    assert json.loads((folder / "COMPLETE.json").read_text())["status"] == "PASS"
    np.testing.assert_array_equal(np.load(folder / "logits.npy"), good)


def test_a_changed_logits_file_is_not_trusted(env, capsys):
    plan, folder = finished(env)
    np.save(folder / "logits.npy", np.zeros((69, 20), dtype=np.float32))
    run_stages(plan, only={"teacher"})
    assert "does not match the sha256" in capsys.readouterr().out
    assert np.load(folder / "logits.npy").any()


def test_a_crash_while_writing_leaves_no_table_and_no_completion_record(env, monkeypatch):
    plan = plan_for(env)
    run_stages(plan, only={"prepare"})

    def half_written(fh, array):
        fh.write(b"\x93NUMPY partial")
        raise OSError("disk full")

    with monkeypatch.context() as patched:
        patched.setattr(teacher.np, "save", half_written)
        with pytest.raises(OSError, match="disk full"):
            run_stages(plan, only={"teacher"})
    folder = plan.campaign_dir / "teacher"
    assert not (folder / "logits.npy").exists() and not (folder / "COMPLETE.json").exists()
    run_stages(plan, only={"teacher"})                                 # the leftover temporary file does not get in the way
    assert (folder / "logits.npy").is_file() and (folder / "COMPLETE.json").is_file()


def test_a_crash_during_a_recomputation_keeps_the_old_table_but_not_its_pass(env, monkeypatch):
    plan, folder = finished(env)
    edit_record(folder, context_digest="0" * 64)
    old = (folder / "logits.npy").read_bytes()

    def killed(*_a, **_k):
        raise RuntimeError("killed")

    monkeypatch.setattr(teacher, "aa20_logits", killed)
    with pytest.raises(RuntimeError, match="killed"):
        run_stages(plan, only={"teacher"})
    assert (folder / "logits.npy").read_bytes() == old and not (folder / "COMPLETE.json").exists()


# -- train refuses a table that does not belong to the campaign ---------------------------------

@pytest.mark.parametrize("change,fragment", [
    ({"context_digest": "f" * 64}, "computed for contexts ffffffffffff"),
    ({"teacher": "esm2-3b"}, "computed by esm2-3b"),
])
def test_train_names_the_fix_when_the_table_is_for_other_contexts(env, change, fragment):
    plan, folder = finished(env)
    edit_record(folder, **change)
    with pytest.raises(RuntimeError) as exc:
        run_stages(plan, only={"train"})
    message = str(exc.value)
    assert fragment in message and "Run the teacher stage again" in message and "--force" in message
    assert not (plan.campaign_dir / "train" / "all").exists()          # nothing was started


def test_train_refuses_when_there_is_no_teacher_record(env):
    plan, folder = finished(env)
    (folder / "COMPLETE.json").unlink()
    with pytest.raises(RuntimeError, match=r"teacher/COMPLETE.json does not exist"):
        run_stages(plan, only={"train"})
