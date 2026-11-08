import hashlib
import importlib.util
import io
import json
import re
import tarfile
import zipfile
from pathlib import Path

import pytest

from mpd.cli import main
from mpd.data import sources, venus_structures as vs
from mpd.data.fetch import FetchError, fetch_source

BUILDER = Path(__file__).resolve().parents[1] / "scripts" / "maintainers" / "build_venus_structure_package.py"


def md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# -- pinned sources -----------------------------------------------------------

def test_proteingym_pins_are_well_formed():
    pg = sources.PROTEINGYM
    assert pg.doi == f"10.5281/zenodo.{pg.zenodo_record}"
    names = [f.name for f in pg.files]
    assert len(names) == len(set(names)) and set(pg.unpack) <= set(names)
    for f in pg.files:
        assert re.fullmatch(r"[0-9a-f]{32}", f.md5) and re.fullmatch(r"[0-9a-f]{64}", f.sha256) and f.size > 0
    assert "MSA" not in " ".join(names)          # no student uses MSAs; the 1.5 GB download is not pinned


def test_unpublished_source_demands_a_location():
    with pytest.raises(sources.SourceError, match="not.*published|no published"):
        sources.VENUS_STRUCTURES.base_url()


# -- fetch --------------------------------------------------------------------

@pytest.fixture
def served(tmp_path):
    """A 'server' folder with a csv and a zip, plus a Source pinned to them; file:// stands in for Zenodo."""
    server = tmp_path / "server"
    server.mkdir()
    csv_bytes = b"a,b\n1,2\n"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("pkg/x.csv", "m,s\nA1C,0.5\n")
    zip_bytes = buf.getvalue()
    (server / "ref.csv").write_bytes(csv_bytes)
    (server / "pkg.zip").write_bytes(zip_bytes)
    src = sources.Source(
        key="toy", title="toy", dirname="toy", zenodo_record="1", doi=None, license="MIT", unpack=("pkg.zip",),
        files=(sources.RemoteFile("ref.csv", len(csv_bytes), md5(csv_bytes), sha(csv_bytes)),
               sources.RemoteFile("pkg.zip", len(zip_bytes), md5(zip_bytes), sha(zip_bytes))))
    return server, src


def test_fetch_downloads_verifies_and_unpacks(tmp_path, served):
    server, src = served
    dest = fetch_source(src, tmp_path / "out", base_url=server.as_uri(), log=lambda _m: None)
    assert (dest / "ref.csv").read_text() == "a,b\n1,2\n"
    assert (dest / "pkg" / "x.csv").is_file()
    assert json.loads((dest / "FETCHED.json").read_text())["files"]["ref.csv"] == sha(b"a,b\n1,2\n")


def test_fetch_is_idempotent_and_needs_no_server_when_files_are_present(tmp_path, served):
    server, src = served
    fetch_source(src, tmp_path / "out", base_url=server.as_uri(), log=lambda _m: None)
    for f in server.iterdir():
        f.unlink()                                   # a second run must not touch the network
    fetch_source(src, tmp_path / "out", base_url=server.as_uri(), log=lambda _m: None)


def test_fetch_rejects_and_removes_a_corrupted_download(tmp_path, served):
    server, src = served
    (server / "ref.csv").write_bytes(b"a,b\n1,3\n")  # same size, different content
    with pytest.raises(FetchError, match="md5 mismatch"):
        fetch_source(src, tmp_path / "out", base_url=server.as_uri(), log=lambda _m: None)
    assert not (tmp_path / "out" / "ref.csv").exists()


def test_fetch_replaces_a_corrupted_local_copy(tmp_path, served):
    server, src = served
    out = tmp_path / "out"
    out.mkdir()
    (out / "ref.csv").write_bytes(b"garbage")
    fetch_source(src, out, base_url=server.as_uri(), log=lambda _m: None)
    assert (out / "ref.csv").read_bytes() == b"a,b\n1,2\n"


def test_fetch_from_local_dir(tmp_path, served):
    server, src = served
    fetch_source(src, tmp_path / "out", from_dir=server, log=lambda _m: None)
    assert (tmp_path / "out" / "pkg" / "x.csv").is_file()
    with pytest.raises(FetchError, match="not found"):
        fetch_source(src, tmp_path / "out2", from_dir=tmp_path / "empty", log=lambda _m: None)


def test_zip_with_path_traversal_is_refused(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("../evil.txt", "x")
    data = buf.getvalue()
    server = tmp_path / "server"
    server.mkdir()
    (server / "bad.zip").write_bytes(data)
    src = sources.Source("toy", "toy", "toy", "1", None, "MIT", (sources.RemoteFile("bad.zip", len(data), md5(data), sha(data)),),
                         unpack=("bad.zip",))
    with pytest.raises(FetchError, match="unsafe"):
        fetch_source(src, tmp_path / "out" / "d", base_url=server.as_uri(), log=lambda _m: None)
    assert not (tmp_path / "out" / "evil.txt").exists()


# -- the Venus structure set --------------------------------------------------

def make_pdb(chain="A", residues=3, extra=()):
    lines = ["MODEL     1"]
    for i in range(residues):
        lines.append(f"ATOM  {i + 1:5d}  CA  ALA {chain}{i + 1:4d}    {i * 3.8:8.3f}{0.0:8.3f}{0.0:8.3f}{1.0:6.2f}{90.0:6.2f}           C  ")
    return "\n".join(lines + list(extra) + ["TER", "ENDMDL", "END"]) + "\n"


@pytest.fixture(scope="module")
def builder():
    spec = importlib.util.spec_from_file_location("build_venus_structure_package", BUILDER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def curation(tmp_path):
    """Three proteins, one per source tier, in the shape of the study's curation records."""
    root = tmp_path / "curation"
    (root / "structures").mkdir(parents=True)
    seqs = {"AAA": "exact_experimental_pdb", "CCC": "exact_sequence_alphafold_db", "DDD": "de_novo_esmfold_fallback"}
    selection, records = [], []
    for seq, tier in seqs.items():
        pid = vs.protein_id_for(seq)
        rel = f"structures/{tier}/{pid}.pdb"
        (root / rel).parent.mkdir(exist_ok=True)
        (root / rel).write_text(make_pdb())
        digest = vs.sha256_file(root / rel)
        sel = {"curated_relative_path": rel, "curated_sha256": digest}
        if tier == "exact_experimental_pdb":
            sel |= {"candidate_pdb_id": "1ABC", "selected_chain": "B", "source_sha256": "e" * 64}
        elif tier == "exact_sequence_alphafold_db":
            sel |= {"entry_id": "AF-P00001-F1", "accession": "P00001", "pdb_url": "https://example.org/AF-P00001-F1-model_v6.pdb",
                    "version": 6, "mean_plddt": 91.5, "pdb_bfactor_confidence_scale": "percent_0_to_100", "raw_pdb_sha256": "a" * 64}
        else:
            sel |= {"model_id": "facebook/esmfold_v1", "model_revision": "r" * 40, "mean_plddt": 88.0,
                    "pdb_bfactor_confidence_scale": "fraction_0_to_1", "raw_sha256": "b" * 64}
        selection.append({"protein_id": pid, "sequence_sha256": vs.sequence_sha256(seq), "source_tier": tier, "selected": sel})
        records.append({"protein_id": pid, "sequence": seq})
    selection.append({"protein_id": "venus_seq_000000000000", "source_tier": None, "resolution_status": "UNSUPPORTED_LENGTH"})
    (root / "selection.jsonl").write_text("\n".join(json.dumps(r) for r in selection) + "\n")
    (root / "plan.json").write_text(json.dumps({"records": records}))
    return root


def build(builder, curation, out):
    rows, members = builder.collect(curation / "selection.jsonl", curation, curation / "plan.json")
    out.mkdir(parents=True, exist_ok=True)
    vs.write_archive(members, out / vs.ARCHIVE_NAME)
    vs.write_manifest(rows, out / vs.MANIFEST_NAME)
    (out / "README.md").write_text(builder.readme(rows, "1.0.0", None, "https://example.org/repo"))
    names = (vs.ARCHIVE_NAME, vs.MANIFEST_NAME, "README.md")
    (out / vs.SUMS_NAME).write_text("".join(f"{vs.sha256_file(out / n)}  {n}\n" for n in names))
    builder.verify_output(out, rows)
    return rows


def test_protein_id_is_the_sha1_prefix_of_the_sequence():
    # the rule the benchmark ids follow (checked against all 489 real ids when the package was built)
    assert vs.protein_id_for("MKV") == "venus_seq_" + hashlib.sha1(b"MKV").hexdigest()[:12]


def test_package_records_each_tiers_provenance(builder, curation, tmp_path):
    rows = {r["source"]: r for r in build(builder, curation, tmp_path / "pkg")}
    assert set(rows) == {"experimental", "alphafold_db", "esmfold"}
    assert (rows["experimental"]["source_id"], rows["experimental"]["chain"], rows["experimental"]["confidence_scale"]) == ("1ABC", "B", "bfactor")
    assert (rows["alphafold_db"]["uniprot_accession"], rows["alphafold_db"]["source_version"]) == ("P00001", "v6")
    assert (rows["esmfold"]["source_version"], rows["esmfold"]["confidence_scale"]) == ("r" * 40, "fraction_0_to_1")
    manifest = vs.read_manifest(tmp_path / "pkg" / vs.MANIFEST_NAME)
    assert len(manifest) == 3                                      # the unsupported-length record is not packaged


def test_archive_is_byte_reproducible(builder, curation, tmp_path):
    build(builder, curation, tmp_path / "a")
    build(builder, curation, tmp_path / "b")
    assert vs.sha256_file(tmp_path / "a" / vs.ARCHIVE_NAME) == vs.sha256_file(tmp_path / "b" / vs.ARCHIVE_NAME)


def test_builder_refuses_inputs_that_break_the_documented_format(builder, curation):
    (curation / "plan.json").write_text(json.dumps({"records": [{"protein_id": vs.protein_id_for("AAA"), "sequence": "AAB"}]}))
    with pytest.raises(ValueError, match="does not match"):
        builder.collect(curation / "selection.jsonl", curation, curation / "plan.json")
    with pytest.raises(ValueError, match="chains"):
        p = curation / "two_chains.pdb"
        p.write_text(make_pdb(chain="B"))
        builder.check_pdb(p)
    with pytest.raises(ValueError, match="HETATM"):
        p.write_text(make_pdb(extra=["HETATM    1  O   HOH A 100       1.000   1.000   1.000  1.00 10.00           O  "]))
        builder.check_pdb(p)


def test_published_files_are_checked_end_to_end(builder, curation, tmp_path):
    out = tmp_path / "pkg"
    build(builder, curation, out)
    dest = fetch_source(sources.VENUS_STRUCTURES, tmp_path / "home" / "data" / "v", from_dir=out, log=lambda _m: None)
    assert len(list((dest / "structures").rglob("*.pdb"))) == 3
    assert vs.verify_structures(dest, vs.read_manifest(dest / vs.MANIFEST_NAME)) == []


def test_a_tampered_archive_is_caught(builder, curation, tmp_path):
    out = tmp_path / "pkg"
    build(builder, curation, out)
    blob = bytearray((out / vs.ARCHIVE_NAME).read_bytes())
    blob[len(blob) // 2] ^= 0xFF
    (out / vs.ARCHIVE_NAME).write_bytes(bytes(blob))
    assert any("sha256 mismatch" in p for p in vs.verify_sums(out))
    with pytest.raises(FetchError, match="sha256 mismatch"):
        fetch_source(sources.VENUS_STRUCTURES, tmp_path / "dest", from_dir=out, log=lambda _m: None)


def test_extraction_refuses_traversal_and_links(tmp_path):
    evil = tmp_path / "evil.tar"
    with tarfile.open(evil, "w") as tar:
        info = tarfile.TarInfo("../outside.pdb")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="unsafe"):
        vs.safe_extract_tar(evil, tmp_path / "dest")
    link = tmp_path / "link.tar"
    with tarfile.open(link, "w") as tar:
        info = tarfile.TarInfo("structures/a.pdb")
        info.type, info.linkname = tarfile.SYMTYPE, "/etc/passwd"
        tar.addfile(info)
    with pytest.raises(ValueError, match="unsupported"):
        vs.safe_extract_tar(link, tmp_path / "dest2")


def test_confidence_scales_convert_to_percent_or_refuse():
    assert vs.plddt_percent(0.93, "fraction_0_to_1") == pytest.approx(93.0)
    assert vs.plddt_percent(93.0, "percent_0_to_100") == 93.0
    assert vs.plddt_percent(54.9, "bfactor") is None              # crystallographic B-factors are not confidence


# -- command line -------------------------------------------------------------

def test_fetch_cli_lists_sources_and_needs_a_home(capsys, monkeypatch):
    monkeypatch.delenv("MPD_HOME", raising=False)
    monkeypatch.delenv("MPD_CLUSTER", raising=False)
    assert main(["fetch", "--list"]) == 0
    assert "15293562" in capsys.readouterr().out
    with pytest.raises(SystemExit, match="--home"):
        main(["fetch", "proteingym"])


def test_fetch_cli_reports_an_unpublished_source_as_a_normal_error(capsys, tmp_path, monkeypatch):
    monkeypatch.delenv("MPD_CLUSTER", raising=False)
    assert main(["fetch", "venus-structures", "--home", str(tmp_path)]) == 2
    assert "--from-dir" in capsys.readouterr().err
