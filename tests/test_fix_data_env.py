"""Data and environment checks: file names in a deposit, downloads that end early, a ProteinGym fetch cut off while unpacking, and
profile names in any letter case. No network and no sockets: a `file://` folder stands in for the server, and a real
`http.client.HTTPResponse` is fed bytes from memory."""

import hashlib
import http.client
import io
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from fx_cli import cli_env, ready_home, recipe, run_cli, write_proteingym  # noqa: F401

from mpd import preflight
from mpd.data import fetch as fetch_mod, sources, venus_structures as vs
from mpd.data.fetch import fetch_source
from mpd.slurm.cluster import Cluster, ClusterError

NOT_PLAIN = "not a plain file name"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# -- a name in the deposit's own files must stay inside the folder ---------------------------------------------------

@pytest.mark.parametrize("line", [
    f"{'0' * 64}  ../outside.txt",
    f"{'0' * 64}  /etc/outside.txt",
    f"{'0' * 64}  //etc/outside.txt",
    f"{'0' * 64}  sub/../../outside.txt",
    f"{'0' * 64}  sub\\outside.txt",
    f"{'0' * 64}  .",
    f"{'0' * 64} *",
    "0" * 64,
])
def test_a_sums_file_cannot_name_a_path_that_leaves_the_folder(tmp_path, line):
    sums = tmp_path / vs.SUMS_NAME
    sums.write_text(f"{'1' * 64}  README.md\n{line}\n")
    with pytest.raises(ValueError, match=NOT_PLAIN) as err:
        vs.parse_sums(sums)
    assert str(err.value).startswith("SHA256SUMS: ") and "Download them again" in str(err.value)


def test_a_sums_file_with_the_real_names_still_parses(tmp_path):
    sums = tmp_path / vs.SUMS_NAME
    sums.write_text(f"{'1' * 64}  {vs.ARCHIVE_NAME}\n\n{'2' * 64} *{vs.MANIFEST_NAME}\n{'3' * 64}\tREADME.md\n")
    assert vs.parse_sums(sums) == {vs.ARCHIVE_NAME: "1" * 64, vs.MANIFEST_NAME: "2" * 64, "README.md": "3" * 64}


def test_a_sums_file_made_with_a_leading_dot_slash_is_read_as_plain_names(tmp_path):
    sums = tmp_path / vs.SUMS_NAME
    sums.write_text(f"{'1' * 64}  ./README.md\n{'2' * 64}  sub//notes.txt\n")         # `sha256sum ./*` writes this; it stays in the folder
    assert vs.parse_sums(sums) == {"README.md": "1" * 64, "sub/notes.txt": "2" * 64}


@pytest.mark.parametrize("name", ["../outside.pdb", "/etc/passwd", "structures\\x.pdb", "structures/../../x.pdb", ".", ""])
def test_a_manifest_cannot_name_a_structure_outside_the_folder(tmp_path, name):
    path = tmp_path / vs.MANIFEST_NAME
    good = {"protein_id": "venus_seq_aaaaaaaaaaaa", "source": "esmfold", "confidence_scale": "fraction_0_to_1",
            "file": "structures/esmfold/venus_seq_aaaaaaaaaaaa.pdb", "file_sha256": "0" * 64}
    vs.write_manifest([good], path)
    assert [r["file"] for r in vs.read_manifest(path)] == [good["file"]]
    vs.write_manifest([good, {**good, "protein_id": "venus_seq_bbbbbbbbbbbb", "file": name}], path)
    with pytest.raises(ValueError, match=NOT_PLAIN) as err:
        vs.read_manifest(path)
    assert str(err.value).startswith("structures_manifest.tsv: ")


def write_deposit(folder: Path, name: str, data: bytes = b"attacker bytes\n") -> Path:
    """A deposit folder whose SHA256SUMS lists `name` with the right hash for `data` (and no other file)."""
    folder.mkdir()
    (folder / vs.SUMS_NAME).write_text(f"{sha(data)}  {name}\n")
    return folder


@pytest.mark.parametrize("how", ["from_dir", "download"])
@pytest.mark.parametrize("kind", ["parent", "absolute"])
def test_fetching_a_deposit_cannot_write_outside_the_destination(tmp_path, how, kind):
    data = b"attacker bytes\n"
    victim = tmp_path / "victim" / "pwned.txt"
    victim.parent.mkdir()
    deposit = write_deposit(tmp_path / "deposit", "../outside.txt" if kind == "parent" else str(victim), data)
    dest = tmp_path / "home" / "data" / sources.VENUS_STRUCTURES.dirname
    (tmp_path / "outside.txt").write_bytes(data)                      # what ../outside.txt points at, from the deposit and from dest
    mirror = deposit / str(victim).lstrip("/")                        # where a server would find the absolute name
    mirror.parent.mkdir(parents=True)
    mirror.write_bytes(data)
    where = {"from_dir": dict(from_dir=deposit), "download": dict(base_url=deposit.as_uri())}[how]
    with pytest.raises(ValueError, match=NOT_PLAIN):
        fetch_source(sources.VENUS_STRUCTURES, dest, unpack=False, log=lambda _m: None, **where)
    assert not victim.exists() and not (dest.parent / "outside.txt").exists()
    assert [p.name for p in dest.iterdir()] == [vs.SUMS_NAME]         # nothing but the list that was refused


def test_the_fetch_command_refuses_such_a_deposit_with_a_plain_message(capsys, cli_env):
    deposit = write_deposit(cli_env.tmp / "deposit", "../outside.txt")
    (cli_env.tmp / "outside.txt").write_bytes(b"attacker bytes\n")
    rc, _, err = run_cli(capsys, "fetch", "venus-structures", "--home", cli_env.home, "--from-dir", deposit, "--no-unpack")
    assert rc == 2 and err.startswith("error: SHA256SUMS: '../outside.txt' is not a plain file name.")
    assert "Traceback" not in err and "was not expected" not in err
    written = sorted(str(p.relative_to(cli_env.home)) for p in cli_env.home.rglob("*") if p.is_file())
    assert written == [f"data/{sources.VENUS_STRUCTURES.dirname}/SHA256SUMS"]


def test_a_deposit_with_plain_names_is_still_fetched(tmp_path):
    deposit = tmp_path / "deposit"
    deposit.mkdir()
    (deposit / "README.md").write_text("hello\n")
    (deposit / vs.SUMS_NAME).write_text(f"{vs.sha256_file(deposit / 'README.md')}  README.md\n")
    dest = fetch_source(sources.VENUS_STRUCTURES, tmp_path / "dest", from_dir=deposit, unpack=False, log=lambda _m: None)
    assert (dest / "README.md").read_text() == "hello\n"


# -- a body that ends before Content-Length is a network error, so it is retried -----------------------------------

class _Sock:
    def __init__(self, raw: bytes):
        self.raw = raw

    def makefile(self, mode="rb", bufsize=None):
        return io.BytesIO(self.raw)


def served(body: bytes, announced: int | None = None):
    """What `urlopen` returns for a 200 that announces `announced` bytes while `body` is all that arrives (a real HTTPResponse)."""
    head = b"HTTP/1.1 200 OK\r\n" + (b"Content-Length: %d\r\n" % announced if announced is not None else b"") + b"\r\n"
    response = http.client.HTTPResponse(_Sock(head + body))
    response.begin()
    return response


def test_a_body_that_ends_before_its_content_length_is_retried(tmp_path, monkeypatch):
    full, logs, sleeps = bytes(range(250)) * 4, [], []
    answers = iter([served(full[:400], len(full)), served(full, len(full))])
    monkeypatch.setattr(fetch_mod.urllib.request, "urlopen", lambda request, timeout=0: next(answers))
    monkeypatch.setattr(fetch_mod.time, "sleep", sleeps.append)
    target = tmp_path / "f.bin"
    fetch_mod._download("https://example.org/f", target, retries=3, log=logs.append)
    assert target.read_bytes() == full and sleeps == [2] and not (tmp_path / "f.bin.part").exists()
    assert len(logs) == 1 and logs[0].startswith("  retry 1/2 in 2s after: ") and "400" in logs[0]


def test_a_body_that_keeps_ending_early_fails_in_plain_words_and_leaves_nothing(tmp_path, monkeypatch):
    sleeps = []
    monkeypatch.setattr(fetch_mod.urllib.request, "urlopen", lambda request, timeout=0: served(b"x" * 400, 1000))
    monkeypatch.setattr(fetch_mod.time, "sleep", sleeps.append)
    target = tmp_path / "f.bin"
    with pytest.raises(fetch_mod.NetworkError, match=r"^could not download https://example\.org/f: the server ended the body "
                                                      r"after 400 of 1000 bytes$"):
        fetch_mod._download("https://example.org/f", target, retries=3, log=lambda _m: None)
    assert sleeps == [2, 4] and not target.exists() and not (tmp_path / "f.bin.part").exists()


def test_a_pinned_file_cut_short_is_downloaded_again_not_blamed_on_the_file(tmp_path, monkeypatch):
    data = b"pinned bytes " * 100
    answers = iter([served(data[:400], len(data)), served(data, len(data))])
    monkeypatch.setattr(fetch_mod.urllib.request, "urlopen", lambda request, timeout=0: next(answers))
    monkeypatch.setattr(fetch_mod.time, "sleep", lambda s: None)
    spec = sources.RemoteFile("pinned.bin", len(data), None, sha(data))
    path = fetch_mod._obtain(spec, tmp_path, base_url="https://example.org/files", from_dir=None, log=lambda _m: None)
    assert path.read_bytes() == data


def test_a_response_without_a_usable_length_is_taken_as_it_comes(tmp_path, monkeypatch):
    for response in (served(b"payload"), io.BytesIO(b"payload")):
        monkeypatch.setattr(fetch_mod.urllib.request, "urlopen", lambda request, timeout=0, r=response: r)
        target = tmp_path / "f.bin"
        fetch_mod._download("https://example.org/f", target, log=lambda _m: None)
        assert target.read_bytes() == b"payload"
        target.unlink()
    announced = fetch_mod._announced_length
    assert announced(SimpleNamespace(headers={"Content-Length": " 12 "})) == 12
    assert announced(SimpleNamespace(headers={"Content-Length": "twelve"})) is None
    assert announced(SimpleNamespace(headers={})) is None and announced(object()) is None
    assert announced(SimpleNamespace(headers={"Content-Length": "12"}, chunked=True)) is None      # the chunks end it, not the header


# -- a ProteinGym fetch that was cut off must not read READY -------------------------------------------------------

def context(cli_env):
    return preflight.Context(Cluster.load("grace"), cli_env.home)


def proteingym_dir(cli_env) -> Path:
    return cli_env.home / "data" / sources.PROTEINGYM.dirname


def test_a_proteingym_folder_missing_assay_files_without_a_marker_is_an_error(cli_env):
    write_proteingym(cli_env.home, fetched=False)
    (proteingym_dir(cli_env) / "DMS_ProteinGym_substitutions" / "ASSAY_C.csv").unlink()
    fix = f"mpd fetch proteingym --cluster grace --home {cli_env.home}"
    expected = (f"ProteinGym data is incomplete under {proteingym_dir(cli_env)}. It lacks 1 of 3 assay files, such as ASSAY_C.csv. "
                f"The fetch was probably cut off. Run it again on a login node: {fix}")
    for required in (True, False):                                    # `mpd doctor` asks with required=False
        row = preflight.check_dataset("proteingym", context(cli_env), required=required)
        assert (row.name, row.level, row.message, row.fix) == ("data:proteingym", "error", expected, fix)


def test_missing_structure_files_are_found_the_same_way(cli_env):
    write_proteingym(cli_env.home, fetched=False)
    folder = proteingym_dir(cli_env) / "ProteinGym_AF2_structures"
    assert preflight.check_dataset("proteingym", context(cli_env)).level == "ok"          # no structure folder: ESM2 students need none
    folder.mkdir()
    (folder / "P1.pdb").write_text("x\n")                                                 # the unpacking stopped before P2.pdb
    row = preflight.check_dataset("proteingym", context(cli_env))
    assert row.level == "error" and "It lacks 1 of 2 structure files, such as P2.pdb." in row.message
    assert row.message.endswith(f"Run it again on a login node: {row.fix}")
    (folder / "P2.pdb").write_text("x\n")
    assert preflight.check_dataset("proteingym", context(cli_env)).level == "ok"


def test_a_complete_folder_without_a_marker_and_a_fetched_one_keep_their_rows(cli_env):
    write_proteingym(cli_env.home, fetched=False)
    row = preflight.check_dataset("proteingym", context(cli_env))
    assert row.level == "ok" and "not hash-checked" in row.message
    (proteingym_dir(cli_env) / "FETCHED.json").write_text("{}")
    assert "complete: FETCHED.json present" in preflight.check_dataset("proteingym", context(cli_env)).message


def test_doctor_and_plan_do_not_say_ready_for_a_cut_off_fetch(capsys, cli_env, monkeypatch):
    ready_home(cli_env, monkeypatch)
    (proteingym_dir(cli_env) / "FETCHED.json").unlink()
    (proteingym_dir(cli_env) / "DMS_ProteinGym_substitutions" / "ASSAY_B.csv").unlink()
    rc, out, _ = run_cli(capsys, "doctor", "--cluster", "grace", "--home", cli_env.home)
    assert rc == 1 and "[FAIL] data:proteingym" in out and "Run it again on a login node: mpd fetch proteingym" in out
    assert out.splitlines()[-1].startswith("NOT READY: 1 error")
    rc, out, _ = run_cli(capsys, "plan", *recipe(cli_env.home))
    assert rc == 0 and "[FAIL] data:proteingym" in out and out.splitlines()[-1].startswith("NOT READY: 1 error")


@pytest.fixture
def toy_release(tmp_path):
    """A server folder with a ProteinGym-shaped release (reference table, assay zip, structure zip) and the Source pinned to it."""
    server = tmp_path / "server"
    server.mkdir()
    ids = ("ASSAY_A", "ASSAY_B", "ASSAY_C")
    ref = "DMS_id,DMS_filename,UniProt_ID,target_seq,pdb_file\n" + "".join(f"{i},{i}.csv,P,MKV,{i}.pdb\n" for i in ids)
    (server / "DMS_substitutions.csv").write_text(ref)
    for archive, folder, ext in (("DMS_ProteinGym_substitutions.zip", "DMS_ProteinGym_substitutions", "csv"),
                                 ("ProteinGym_AF2_structures.zip", "ProteinGym_AF2_structures", "pdb")):
        with zipfile.ZipFile(server / archive, "w") as zf:
            for i in ids:
                zf.writestr(f"{folder}/{i}.{ext}", "mutant,DMS_score\nM1A,0.5\n" if ext == "csv" else "ATOM\n")
    files = tuple(sources.RemoteFile(p.name, p.stat().st_size, hashlib.md5(p.read_bytes()).hexdigest(), sha(p.read_bytes()))
                  for p in sorted(server.iterdir()))
    release = sources.Source(key="proteingym", title="toy ProteinGym", dirname=sources.PROTEINGYM.dirname, zenodo_record="1",
                             doi=None, license="MIT", files=files,
                             unpack=("DMS_ProteinGym_substitutions.zip", "ProteinGym_AF2_structures.zip"))
    return server, release


@pytest.mark.parametrize("archive,kind", [("DMS_ProteinGym_substitutions.zip", "assay"), ("ProteinGym_AF2_structures.zip", "structure")])
def test_a_fetch_killed_while_unpacking_is_flagged_and_the_same_command_repairs_it(cli_env, toy_release, monkeypatch, archive, kind):
    server, release = toy_release
    dest = proteingym_dir(cli_env)
    real = fetch_mod._unpack_zip

    def killed(path, folder):
        if path.name != archive:
            return real(path, folder)
        with zipfile.ZipFile(path) as zf:
            zf.extract(zf.namelist()[0], folder)                     # one file is out when the connection to the login node drops
        raise KeyboardInterrupt

    monkeypatch.setattr(fetch_mod, "_unpack_zip", killed)
    with pytest.raises(KeyboardInterrupt):
        fetch_source(release, dest, base_url=server.as_uri(), log=lambda _m: None)
    assert not (dest / "FETCHED.json").exists()
    row = preflight.check_dataset("proteingym", context(cli_env))
    assert row.level == "error" and f"It lacks 2 of 3 {kind} files" in row.message and "Run it again" in row.message
    monkeypatch.setattr(fetch_mod, "_unpack_zip", real)
    fetch_source(release, dest, base_url=server.as_uri(), log=lambda _m: None)               # what the message told the student to do
    assert "complete: FETCHED.json present" in preflight.check_dataset("proteingym", context(cli_env)).message


# -- the built-in profile names in any letter case -----------------------------------------------------------------

@pytest.mark.parametrize("spelling,name", [("Grace", "grace"), ("GRACE", "grace"), ("gRaCe", "grace"),
                                           ("Vision", "vision"), ("VISION", "vision")])
def test_a_builtin_profile_name_is_accepted_in_any_letter_case(spelling, name):
    assert Cluster.load(spelling) == Cluster.load(name) and Cluster.load(spelling).name == name


def test_an_unknown_name_is_reported_as_typed_and_a_path_keeps_its_spelling(tmp_path, monkeypatch):
    with pytest.raises(ClusterError, match=r"unknown cluster 'Nope'; use one of grace, vision or a path to a profile YAML"):
        Cluster.load("Nope")
    clusters = Path(sources.__file__).parents[1] / "clusters"
    mine = tmp_path / "Grace.yaml"                                    # a file called Grace.yaml is a file, not the grace profile
    mine.write_text((clusters / "grace.yaml").read_text().replace("name: grace", "name: mine", 1))
    monkeypatch.chdir(tmp_path)
    assert Cluster.load("Grace.yaml").name == Cluster.load(str(mine)).name == "mine"


def test_the_commands_print_the_lower_case_profile_name_in_their_fixes(capsys, cli_env):
    rc, out, _ = run_cli(capsys, "doctor", "--cluster", "GRACE", "--home", cli_env.home)
    assert rc == 1 and "bash scripts/setup_env.sh --cluster grace" in out and "--cluster GRACE" not in out
