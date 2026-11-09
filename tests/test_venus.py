import csv
import hashlib
import io
import os
import urllib.error
from pathlib import Path

import pytest

from mpd.data import fetch as fetch_mod, sources, venus_structures as vs
from mpd.data.venusmuthub import EXCLUDED_ASSAYS, VenusMutHub, protein_id

WT = "MKLVAGDEWQ"


def write_assay(root: Path, prop: str, name: str, rows: list[tuple[str, str, str]]):
    folder = root / "single_mutant" / prop
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / f"{name}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["mutant", "mutated_sequence", "fitness_score"])
        w.writerows(rows)


def mutate(seq: str, pos1: int, aa: str) -> str:
    return seq[:pos1 - 1] + aa + seq[pos1:]


@pytest.fixture
def venus(tmp_path):
    root = tmp_path / "venus"
    write_assay(root, "stability", "S1", [
        ("M1M", WT, "23.0"),                                  # identity row: the wild type measured
        ("K2A", mutate(WT, 2, "A"), "-1.5"),
        ("L3W", mutate(WT, 3, "W"), "NDA"),                   # non-numeric label: dropped
        ("V4I", mutate(WT, 4, "I"), "2.25"),
        ("A5G", mutate(WT, 5, "G"), "1.25 ± 0.04"),           # not a number either
    ])
    write_assay(root, "selectivity", "S2", [("M1A", mutate(WT, 1, "A"), "0.1"), ("K2R", mutate(WT, 2, "R"), "0.2")])
    write_assay(root, "ppi_binding", "P14779_ee", [("M1A", mutate(WT, 1, "A"), "9")])   # excluded by the cohort rule
    return VenusMutHub(root)


def test_cohort_rules_and_label_alignment(venus):
    assert {a.assay_id for a in venus.assays()} == {"S1", "S2"}
    assert "P14779_ee" in EXCLUDED_ASSAYS and all(a.assay_id != "P14779_ee" for a in venus.assays())
    v, y = venus.variants("S1"), venus.labels("S1")
    assert v.mutants == ("M1M", "K2A", "V4I") and list(y) == [23.0, -1.5, 2.25]
    assert v.wt[0] == v.mt[0]                                   # identity row survives
    assert venus.task("S1") == "Stability" and venus.task("S2") == "Selectivity"
    assert venus.assay("S1").sequence == WT and venus.assay("S1").protein_id == protein_id(WT)
    assert venus.assay("S1").protein_id == venus.assay("S2").protein_id     # assays on one protein share an id


def test_inconsistent_wild_types_are_an_error(tmp_path):
    root = tmp_path / "v"
    write_assay(root, "stability", "BAD", [("K2A", mutate(WT, 2, "A"), "1"), ("L3W", mutate("MKLVAGDEWA", 3, "W"), "2")])
    with pytest.raises(ValueError, match="different wild-type"):
        VenusMutHub(root).assays()


def test_structure_manifest_attaches_structures_and_confidence_scale(venus, tmp_path):
    structures = tmp_path / "structures"
    (structures / "structures" / "esmfold").mkdir(parents=True)
    pid = protein_id(WT)
    vs.write_manifest([{"protein_id": pid, "source": "esmfold", "confidence_scale": "fraction_0_to_1",
                        "file": f"structures/esmfold/{pid}.pdb", "file_sha256": "0" * 64}], structures / vs.MANIFEST_NAME)
    with_struct = VenusMutHub(venus.root, structures)
    a = with_struct.assay("S1")
    assert a.structure == structures / f"structures/esmfold/{pid}.pdb" and a.structure_scale == "fraction_0_to_1"
    assert venus.assay("S1").structure is None


def test_variants_do_not_depend_on_label_values(tmp_path):
    root = tmp_path / "v"
    write_assay(root, "stability", "A", [("K2A", mutate(WT, 2, "A"), "1.0"), ("L3W", mutate(WT, 3, "W"), "7.5")])
    first = VenusMutHub(root).variants("A")
    write_assay(root, "stability", "A", [("K2A", mutate(WT, 2, "A"), "-100"), ("L3W", mutate(WT, 3, "W"), "0.001")])
    second = VenusMutHub(root).variants("A")
    assert first.mutants == second.mutants and (first.pos == second.pos).all()


# -- pinned Hugging Face fetch --------------------------------------------------

@pytest.fixture
def hf_source(tmp_path, monkeypatch):
    server = tmp_path / "server"
    files = {"single_mutant/stability/A.csv": b"mutant,mutated_sequence,fitness_score\n", "single_mutant/activity/B.csv": b"x\n"}
    for rel, data in files.items():
        (server / rel).parent.mkdir(parents=True, exist_ok=True)
        (server / rel).write_bytes(data)
    entries = [(rel, len(d), hashlib.sha256(d).hexdigest()) for rel, d in sorted(files.items())]
    monkeypatch.setattr(fetch_mod, "_hf_entries", lambda source: entries)
    src = sources.Source("toyhf", "toy", "toyhf", None, None, "MIT", (), hf_repo="o/r", hf_revision="r", manifest="x.tsv")
    return server, src


def test_hf_fetch_verifies_every_file_and_only_refetches_what_is_missing(tmp_path, hf_source):
    server, src = hf_source
    out, logs = tmp_path / "out", []
    fetch_mod.fetch_source(src, out, base_url=server.as_uri(), pause=0, log=logs.append)
    assert (out / "single_mutant/stability/A.csv").is_file() and (out / "FETCHED.json").is_file()
    (out / "single_mutant/activity/B.csv").unlink()
    logs.clear()
    fetch_mod.fetch_source(src, out, base_url=server.as_uri(), pause=0, log=logs.append)
    assert any("1/2 files already present" in line for line in logs)


def test_hf_fetch_rejects_a_changed_file(tmp_path, hf_source):
    server, src = hf_source
    (server / "single_mutant/activity/B.csv").write_bytes(b"y\n")      # same size, different content
    with pytest.raises(fetch_mod.FetchError, match="sha256 mismatch"):
        fetch_mod.fetch_source(src, tmp_path / "out", base_url=server.as_uri(), pause=0, log=lambda _m: None)
    assert not (tmp_path / "out/single_mutant/activity/B.csv").exists()


def test_rate_limiting_is_waited_out_not_hammered(tmp_path, monkeypatch):
    calls, sleeps = [], []

    def flaky(url, timeout=0):
        calls.append(url)
        if len(calls) == 1:
            raise urllib.error.HTTPError(url, 429, "Too Many Requests", {"Retry-After": "7"}, io.BytesIO(b""))
        return io.BytesIO(b"payload")

    monkeypatch.setattr(fetch_mod.urllib.request, "urlopen", flaky)
    monkeypatch.setattr(fetch_mod.time, "sleep", sleeps.append)
    target = tmp_path / "f.bin"
    fetch_mod._download("https://example.org/f", target, retries=3, log=lambda _m: None)
    assert target.read_bytes() == b"payload" and sleeps == [7] and len(calls) == 2


def test_the_shipped_manifest_pins_all_796_public_files():
    entries = fetch_mod._hf_entries(sources.VENUSMUTHUB)
    assert len(entries) == 796 and len({p for p, _, _ in entries}) == 796
    assert all(p.startswith("single_mutant/") and len(h) == 64 for p, _, h in entries)


@pytest.mark.skipif(not os.environ.get("MPD_TEST_VENUS_HOME"), reason="set MPD_TEST_VENUS_HOME to a home holding fetched Venus data")
def test_real_cohort_matches_the_published_numbers():
    bm = VenusMutHub.open(Path(os.environ["MPD_TEST_VENUS_HOME"]))
    assays = bm.assays()
    assert len(assays) == 793 and len({a.protein_id for a in assays}) == 489
    assert sum(len(bm.variants(a.assay_id)) for a in assays) == 25044
