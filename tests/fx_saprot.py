"""Fixtures for the SaProt tests: tiny random SaProt models, tiny PDB files, a fake Foldseek, a local file server.

Import what a test needs, e.g. `from fx_saprot import tiny_saprot_models, fake_foldseek  # noqa: F401`.
"""

from __future__ import annotations

import contextlib
import csv
import functools
import http.server
import io
import os
import stat
import sys
import tarfile
import threading
from pathlib import Path

import pytest

from tinydata import write_tiny_proteingym

THREE = {"A": "ALA", "C": "CYS", "D": "ASP", "E": "GLU", "F": "PHE", "G": "GLY", "H": "HIS", "I": "ILE", "K": "LYS",
         "L": "LEU", "M": "MET", "N": "ASN", "P": "PRO", "Q": "GLN", "R": "ARG", "S": "SER", "T": "THR", "V": "VAL",
         "W": "TRP", "Y": "TYR"}


# -- PDB files ---------------------------------------------------------------------------------------------

def atom_line(serial: int, name: str, res: str, chain: str, number: int, b: float, *, altloc: str = " ", icode: str = " ",
              record: str = "ATOM  ") -> str:
    x = 1.5 * number
    temp = f"{b:6.3f}" if abs(b) < 10 else f"{b:6.2f}"      # ESMFold writes 0-1 values; keep 3 decimals to test the 0.70 edge
    return (f"{record}{serial:5d} {name:^4s}{altloc}{res:>3s} {chain}{number:4d}{icode}   {x:8.3f}{0.0:8.3f}{0.0:8.3f}"
            f"{1.0:6.2f}{temp}          {name.strip()[0]:>2s}  ")


def pdb_text(seq: str, bfactors, chain: str = "A", start: int = 1) -> str:
    """A one-model PDB with N, CA and C atoms per residue; `bfactors` is a number or one per residue."""
    values = [bfactors] * len(seq) if isinstance(bfactors, (int, float)) else list(bfactors)
    assert len(values) == len(seq)
    lines, serial = ["MODEL        1"], 0
    for i, (a, b) in enumerate(zip(seq, values)):
        for name in ("N", "CA", "C"):
            serial += 1
            lines.append(atom_line(serial, name, THREE[a], chain, start + i, b))
    lines += ["TER", "ENDMDL", "END"]
    return "\n".join(lines) + "\n"


def pdb_text_chains(chains: list[tuple[str, str]], b: float = 90.0) -> str:
    """One model holding several chains: [(chain id, sequence), ...]."""
    lines, serial = ["MODEL        1"], 0
    for chain, seq in chains:
        for i, a in enumerate(seq):
            for name in ("N", "CA", "C"):
                serial += 1
                lines.append(atom_line(serial, name, THREE[a], chain, 1 + i, b))
        lines.append("TER")
    return "\n".join(lines + ["ENDMDL", "END"]) + "\n"


def helix_pdb_text(seq: str, b: float = 90.0, chain: str = "A") -> str:
    """N, CA, C atoms on an alpha-helix-like spiral (100 degrees and 1.5 A per residue): non-degenerate geometry for
    code that really computes 3Di. Not a physical structure."""
    import math
    lines, serial = ["MODEL        1"], 0
    for i, a in enumerate(seq):
        theta = math.radians(100.0 * i)
        for name, dtheta, dz in (("N", -0.55, -0.9), ("CA", 0.0, 0.0), ("C", 0.5, 0.8)):
            serial += 1
            x, y, z = 2.3 * math.cos(theta + dtheta), 2.3 * math.sin(theta + dtheta), 1.5 * i + dz
            temp = f"{b:6.2f}"
            lines.append(f"ATOM  {serial:5d} {name:^4s} {THREE[a]:>3s} {chain}{i + 1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{temp}"
                         f"          {name[0]:>2s}  ")
    return "\n".join(lines + ["TER", "ENDMDL", "END"]) + "\n"


def write_pdb(path: Path, seq: str, bfactors, chain: str = "A", start: int = 1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pdb_text(seq, bfactors, chain, start))
    return path


# -- a fake Foldseek honouring `structureto3didescriptor -v 0 --threads 1 --chain-name-mode 1 <in> <out>` ----

FAKE_SOURCE = r'''
import os, sys
args = sys.argv[1:]
log = os.environ.get("FAKE_FOLDSEEK_LOG")
if log:
    with open(log, "a") as fh:
        fh.write(" ".join(args) + "\n")
mode = os.environ.get("FAKE_FOLDSEEK_MODE", "")
if args[:1] == ["version"]:
    print(os.environ.get("FAKE_FOLDSEEK_VERSION", "fake-941cd33"))
    sys.exit(0)
if args[:1] != ["structureto3didescriptor"] or args[1:7] != ["-v", "0", "--threads", "1", "--chain-name-mode", "1"] or len(args) != 9:
    sys.exit("fake foldseek: unexpected command line: " + " ".join(args))
src, out = args[7], args[8]
if mode == "fail":
    print("Error: could not parse the structure", file=sys.stderr)
    sys.exit(7)
if mode == "silent":
    sys.exit(0)
chains, seen = {}, set()
for line in open(src):
    if line.startswith("ENDMDL"):
        break
    if line.startswith("ATOM") and line[12:16].strip() == "CA" and line[16] in " A":
        key = (line[21], line[22:26], line[26])
        if key in seen:
            continue
        seen.add(key)
        chains.setdefault(line[21], []).append(line[17:20])
THREE = %(three)r
LETTERS = "ACDEFGHIKLMNPQRSTVWY"
rows = []
for chain, residues in chains.items():
    aa = "".join(THREE.get(r, "X") for r in residues)
    states = "".join(LETTERS[((LETTERS.index(a) if a in LETTERS else 0) * 3 + i) %% 20] for i, a in enumerate(aa))
    if mode == "unknown":
        states = states[:1] + "X" + states[2:]
    name = os.path.basename(src) + "_" + chain + (" some description" if mode == "describe" else "")
    rows.append("\t".join([name, aa, states, "0.0 0.1 0.2"]))
if mode == "badline":
    rows = ["only-one-column"]
with open(out, "w") as fh:
    fh.write("\n".join(rows) + "\n")
with open(out + ".dbtype", "wb") as fh:
    fh.write(b"\x0b\x00\x00\x00")
''' % {"three": {v: k for k, v in THREE.items()}}


def fake_states(seq: str) -> str:
    """The lower-case state string the fake produces for a chain with this amino-acid sequence."""
    letters = "ACDEFGHIKLMNPQRSTVWY"
    return "".join(letters[(letters.index(a) * 3 + i) % 20] for i, a in enumerate(seq)).lower()


def install_fake_foldseek(folder: Path, name: str = "foldseek") -> Path:
    """An executable wrapper around the fake (a real Python run, so the subprocess contract is exercised)."""
    folder.mkdir(parents=True, exist_ok=True)
    script = folder / (name + "_impl.py")
    script.write_text(FAKE_SOURCE)
    wrapper = folder / name
    wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" -I "{script}" "$@"\n')
    wrapper.chmod(0o755)
    return wrapper


@pytest.fixture
def fake_foldseek(tmp_path, monkeypatch):
    """Path of a fake Foldseek; also exported as MPD_FOLDSEEK. FAKE_FOLDSEEK_LOG records every call."""
    binary = install_fake_foldseek(tmp_path / "fakebin")
    monkeypatch.setenv("MPD_FOLDSEEK", str(binary))
    monkeypatch.setenv("FAKE_FOLDSEEK_LOG", str(tmp_path / "fakebin" / "calls.log"))
    monkeypatch.delenv("FAKE_FOLDSEEK_MODE", raising=False)
    monkeypatch.delenv("MPD_3DI_ENGINE", raising=False)
    monkeypatch.delenv("MPD_STRUCTURE_LEGACY_MASK", raising=False)
    return binary


def fake_calls(binary: Path) -> list[str]:
    log = binary.parent / "calls.log"
    return [l for l in log.read_text().splitlines() if l.startswith("structureto3didescriptor")] if log.is_file() else []


# -- tiny data ------------------------------------------------------------------------------------------------

def write_tiny_structure_home(home: Path, *, b: float = 90.0, ranges: bool = True) -> dict[str, str]:
    """The tiny ProteinGym of tests/tinydata.py plus a `pdb_range` column and one AF2-style PDB per protein.

    P2 is covered only from residue 5 to 30 when `ranges` is set. Returns {assay id: sequence}.
    """
    from mpd.data.sources import PROTEINGYM
    root = home / "data" / PROTEINGYM.dirname
    root.mkdir(parents=True, exist_ok=True)
    seqs = write_tiny_proteingym(root)
    rows = list(csv.DictReader(open(root / "DMS_substitutions.csv", newline="")))
    for r in rows:
        seq = r["target_seq"]
        a, c = (5, 30) if (ranges and r["UniProt_ID"] == "P2") else (1, len(seq))
        if ranges:
            r["pdb_range"] = f"{a}-{c}"
        write_pdb(root / "ProteinGym_AF2_structures" / r["pdb_file"], seq[a - 1:c], b)
    with open(root / "DMS_substitutions.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return seqs


# -- tiny SaProt models -----------------------------------------------------------------------------------------

def make_tiny_saprot(folder: Path, seed: int, dropout: float = 0.1) -> Path:
    """A random SaProt-shaped model with the real 446-token vocabulary (the hub config of the 650M model has dropout 0.1)."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from mpd.structure import vocab
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "vocab.txt").write_text(vocab.vocab_text())
    cfg = transformers.EsmConfig(vocab_size=446, mask_token_id=4, pad_token_id=1, hidden_size=32, num_hidden_layers=2,
                                 num_attention_heads=2, intermediate_size=64, max_position_embeddings=1026,
                                 position_embedding_type="rotary", token_dropout=True, emb_layer_norm_before=False,
                                 hidden_dropout_prob=dropout, attention_probs_dropout_prob=dropout)
    torch.manual_seed(seed)
    transformers.EsmForMaskedLM(cfg).save_pretrained(folder)
    return folder


@pytest.fixture(scope="session")
def tiny_saprot_models(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny_saprot")
    return {"student": make_tiny_saprot(root / "student", seed=1), "teacher": make_tiny_saprot(root / "teacher", seed=2)}


# -- a local HTTP server for download tests ----------------------------------------------------------------------

class _Handler(http.server.SimpleHTTPRequestHandler):
    throttle: dict = {}

    def log_message(self, *args):
        pass

    def do_GET(self):
        left = self.throttle.get("count", 0)
        if left > 0:
            self.throttle["count"] = left - 1
            self.send_response(429)
            self.send_header("Retry-After", "7")
            self.end_headers()
            return
        super().do_GET()


@contextlib.contextmanager
def serve(folder: Path, throttle: int = 0):
    """Serve `folder` on 127.0.0.1; the first `throttle` requests get HTTP 429 with Retry-After: 7."""
    handler = type("Handler", (_Handler,), {"throttle": {"count": throttle}})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(handler, directory=str(folder)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def fake_release(folder: Path, *, marker: Path | None = None, binary_mode: int = 0o755, extra=(),
                 body: str = 'echo "fake-941cd33"\n'):
    """A tiny stand-in for the Foldseek release archive. Returns (archive path, a Tool pin that matches it).

    The fake binary touches `marker` whenever it runs, so a test can prove an unverified binary was never executed."""
    import dataclasses
    import hashlib
    from mpd.data.sources import FOLDSEEK
    script = "#!/bin/sh\n" + (f'touch "{marker}"\n' if marker else "") + body
    members = [("foldseek/bin/foldseek", script.encode(), binary_mode), ("foldseek/README.md", b"readme\n", 0o644), *extra]
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data, mode in members:
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), mode
            tar.addfile(info, io.BytesIO(data))
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / FOLDSEEK.archive
    archive.write_bytes(buf.getvalue())
    pin = dataclasses.replace(
        FOLDSEEK, archive_size=archive.stat().st_size, archive_sha256=hashlib.sha256(buf.getvalue()).hexdigest(),
        binary_size=len(script), binary_sha256=hashlib.sha256(script.encode()).hexdigest())
    return archive, pin


def is_executable(path: Path) -> bool:
    return bool(path.stat().st_mode & stat.S_IXUSR) and os.access(path, os.X_OK)
