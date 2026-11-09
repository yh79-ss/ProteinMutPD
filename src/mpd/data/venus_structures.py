"""The VenusMutHub structure set: manifest schema, deterministic archive, integrity checks.

One structure per VenusMutHub protein (489 of them), keyed by the sequence hash. The set is distributed as a
Zenodo deposit: `ARCHIVE_NAME`, `MANIFEST_NAME`, `SUMS_NAME` and a README.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import tarfile
from pathlib import Path, PurePosixPath

ARCHIVE_NAME = "proteinmutpd-venusmuthub-structures-v1.tar.gz"
MANIFEST_NAME = "structures_manifest.tsv"
SUMS_NAME = "SHA256SUMS"

SOURCES = ("experimental", "alphafold_db", "esmfold")

# How to read the B-factor column. AlphaFold DB writes pLDDT in 0-100, ESMFold (as run here) in 0-1, and
# experimental files hold crystallographic B-factors that are NOT confidence values.
SCALE_TO_PERCENT = {"percent_0_to_100": 1.0, "fraction_0_to_1": 100.0, "bfactor": None}

COLUMNS = (
    "protein_id", "sequence_sha256", "length", "source", "source_id", "chain", "uniprot_accession",
    "source_url", "source_version", "confidence_scale", "mean_plddt", "file", "file_sha256", "upstream_sha256",
)


def protein_id_for(sequence: str) -> str:
    """The benchmark's protein id: `venus_seq_` + first 12 hex digits of sha1(sequence)."""
    return "venus_seq_" + hashlib.sha1(sequence.encode()).hexdigest()[:12]


def sequence_sha256(sequence: str) -> str:
    return hashlib.sha256(sequence.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def plddt_percent(value: float, scale: str) -> float | None:
    """A confidence column value on the 0-100 scale; None for experimental B-factors."""
    factor = SCALE_TO_PERCENT[scale]
    return None if factor is None else value * factor


def write_manifest(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in sorted(rows, key=lambda r: r["protein_id"]):
            writer.writerow({c: row.get(c, "") for c in COLUMNS})


def check_name(name: str, where: str) -> str:
    """`name` as a clean relative path if joining it onto a folder cannot leave the folder, else a ValueError.

    File names in a deposit's SHA256SUMS and manifest are written by whoever published it, so they are checked before use."""
    path = PurePosixPath(name)
    if not path.parts or path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"{where}: {name!r} is not a plain file name. A name may not be empty, start with '/', "
                         "contain a backslash, or have a '..' part. Do not use these files. "
                         "Download them again, or ask whoever gave them to you.")
    return str(path)


def read_manifest(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if tuple(reader.fieldnames or ()) != COLUMNS:
            raise ValueError(f"{path}: unexpected columns {reader.fieldnames}")
        rows = list(reader)
    for row in rows:
        if row["source"] not in SOURCES:
            raise ValueError(f"{path}: {row['protein_id']} has unknown source {row['source']!r}")
        if row["confidence_scale"] not in SCALE_TO_PERCENT:
            raise ValueError(f"{path}: {row['protein_id']} has unknown confidence_scale {row['confidence_scale']!r}")
        check_name(row["file"] or "", path.name)
    return rows


def write_archive(members: list[tuple[str, Path]], out: Path) -> None:
    """A tar.gz whose bytes depend only on the member names and contents (no mtimes, owners or host info)."""
    raw = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as gz:
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.USTAR_FORMAT) as tar:
            for arcname, path in sorted(members):
                data = Path(path).read_bytes()
                info = tarfile.TarInfo(arcname)
                info.size, info.mtime, info.mode = len(data), 0, 0o644
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                tar.addfile(info, io.BytesIO(data))
    out.write_bytes(raw.getvalue())


def safe_extract_tar(archive: Path, dest: Path) -> list[Path]:
    """Extract regular files only; refuse absolute paths, `..`, links and devices. A member that is executable in
    the archive is executable on disk (mode 0755, never setuid or group-writable)."""
    dest = dest.resolve()
    written = []
    with tarfile.open(archive, "r:*") as tar:
        for member in tar.getmembers():
            target = (dest / member.name).resolve()
            if dest != target and dest not in target.parents:
                raise ValueError(f"{archive}: unsafe member path {member.name!r}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isreg():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(tar.extractfile(member).read())
                if member.mode & 0o111:
                    target.chmod(0o755)
                written.append(target)
            else:
                raise ValueError(f"{archive}: unsupported member type for {member.name!r}")
    return written


def parse_sums(path: Path) -> dict[str, str]:
    """`sha256sum`-format file -> {name: sha256}. A name that would leave the folder it is joined onto is a ValueError."""
    sums = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, *rest = line.split(maxsplit=1)
            sums[check_name(rest[0].lstrip("*").strip() if rest else "", path.name)] = digest
    return sums


def verify_sums(directory: Path) -> list[str]:
    """Problems with the files named in `directory/SHA256SUMS`; empty means all present and matching."""
    sums_path = directory / SUMS_NAME
    if not sums_path.is_file():
        return [f"{sums_path} is missing"]
    problems = []
    for name, digest in parse_sums(sums_path).items():
        path = directory / name
        if not path.is_file():
            problems.append(f"{name}: missing")
        elif sha256_file(path) != digest:
            problems.append(f"{name}: sha256 mismatch")
    return problems


def verify_structures(root: Path, rows: list[dict]) -> list[str]:
    """Check every manifest row's file under `root` exists and has the recorded hash; flag stray files."""
    problems = []
    expected = set()
    for row in rows:
        path = root / row["file"]
        expected.add(path.resolve())
        if not path.is_file():
            problems.append(f"{row['protein_id']}: {row['file']} missing")
        elif sha256_file(path) != row["file_sha256"]:
            problems.append(f"{row['protein_id']}: {row['file']} sha256 mismatch")
    structures = root / "structures"
    if structures.is_dir():
        for path in structures.rglob("*"):
            if path.is_file() and path.resolve() not in expected:
                problems.append(f"unexpected file {path.relative_to(root)}")
    return problems
