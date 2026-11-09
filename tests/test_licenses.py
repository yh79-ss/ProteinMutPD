"""The licence gate: a licence-locked teacher's outputs only train models of its own family, and nothing of it ships."""

import dataclasses

import pytest

from mpd import registry
from mpd.data import models
from mpd.spec import RunSpec, SpecError

BLOCKED = (r"blocked.*only be distilled into aido models.*1\.2\.2.*mpd has no override for this.*"
           r"`mpd models`.*docs/LICENSES\.md")


def spec(**kw):
    base = dict(teacher="esm2-3b", student="esm2-650m", train_data="proteingym", method="score-all",
                eval_data=("proteingym", "venusmuthub"))
    return RunSpec(**{**base, **kw})


def test_aido_teacher_for_a_non_aido_student_is_blocked_by_licence():
    for student in ("esm2-650m", "saprot-650m"):
        with pytest.raises(SpecError, match=BLOCKED):
            spec(teacher="aido-rag-16b", student=student).validate()


def test_the_licence_block_fires_before_the_planned_message():
    # AIDO is "planned" today; that status must not hide why a request is refused, now or after it is implemented
    with pytest.raises(SpecError) as err:
        spec(teacher="aido-rag-16b", student="esm2-650m").validate()
    assert "is planned" not in str(err.value) and str(err.value).startswith("aido-rag-16b -> esm2-650m is blocked")


def test_the_block_message_is_the_documented_one():
    with pytest.raises(SpecError) as err:
        spec(teacher="aido-rag-16b", student="esm2-650m").validate()
    assert str(err.value) == (
        "aido-rag-16b -> esm2-650m is blocked: outputs of aido-rag-16b may only be distilled into aido models. "
        "GenBio AI Community License, clause 1.2.2: Outputs (any output of the GenBio Materials, including "
        "intermediate activations and embeddings) may not be used to train, optimize or improve machine-learning "
        "models similar to the GenBio Materials, nor for model distillation toward that end, except to train the "
        "GenBio Materials themselves and their downstream Derivative Works. Non-commercial use only. "
        "mpd has no override for this; choose a teacher without an output restriction (see `mpd models`). "
        "See docs/LICENSES.md.")


def test_other_aido_requests_say_not_in_v01():
    for kw in (dict(teacher="aido-rag-16b", student="aido-rag-16b"), dict(student="aido-rag-16b")):
        with pytest.raises(SpecError, match=r"planned.*Not in v0\.1.*docs/LICENSES\.md") as err:
            spec(**kw).validate()
        assert str(err.value) == ("aido-rag-16b is planned and cannot be used yet. "
                                  "Not in v0.1. Outputs may only teach AIDO models: docs/LICENSES.md")


def test_aido_row_keeps_its_lock_and_licence_note():
    row = registry.get("aido-rag-16b")
    assert row.output_family_lock == "aido" and "1.2.2" in row.license_note and row.status == "planned"
    assert row.teacher_backends == ("live",)               # a cache of AIDO outputs can never be shipped
    assert row.license == "GenBio AI Community License" and row.revision is None
    assert len(row.note) <= 70                              # `mpd models` prints 70 characters of it


def test_shipped_aido_entry_is_locked_to_its_own_family(monkeypatch):
    """The synthetic lock test (test_spec_plan) proves the mechanism; this pins the real GenBio entry (licence clause
    1.2.2), so flipping AIDO from 'planned' to 'available' cannot silently allow distillation into other families."""
    aido = registry.get("aido-rag-16b")
    assert aido.output_family_lock == "aido" and "1.2.2" in aido.license_note
    monkeypatch.setitem(registry.MODELS, "aido-rag-16b", dataclasses.replace(aido, status="available"))
    with pytest.raises(SpecError, match="may only be distilled into aido models"):
        spec(teacher="aido-rag-16b", student="esm2-650m").validate()


def test_only_the_genbio_model_is_locked():
    assert [k for k, m in registry.MODELS.items() if m.output_family_lock] == ["aido-rag-16b"]
    for key, m in registry.MODELS.items():
        assert m.license, key                               # every model states its licence (`mpd models` shows it)
        if m.status == "available":
            spec(teacher=key, student="esm2-650m").validate()        # no other teacher is restricted


def test_aido_is_never_fetched_and_nothing_of_it_is_pinned(tmp_path):
    assert "aido-rag-16b" not in models.pinned_keys() and "aido-rag-16b" not in models.fetchable_keys()
    assert models.status("aido-rag-16b", tmp_path).state == "unpinned"
    with pytest.raises(models.ModelError, match=r"mpd never downloads aido-rag-16b.*clause 1\.2\.2"):
        models.fetch_model("aido-rag-16b", tmp_path, log=lambda _m: None)
    assert not (tmp_path / "models").exists()
