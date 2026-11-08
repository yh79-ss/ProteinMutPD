"""Shared fixtures: a tiny random ESM2 and a miniature ProteinGym-format dataset, so whole pipelines run on CPU."""

import pytest

from tinydata import make_tiny_esm, write_tiny_proteingym

@pytest.fixture(scope="session")
def tiny_models(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny_esm")
    return {"student": make_tiny_esm(root / "student", seed=1), "teacher": make_tiny_esm(root / "teacher", seed=2)}


@pytest.fixture
def tiny_home(tmp_path):
    from mpd.data.sources import PROTEINGYM
    root = tmp_path / "home" / "data" / PROTEINGYM.dirname
    root.mkdir(parents=True)
    write_tiny_proteingym(root)
    return tmp_path / "home"
