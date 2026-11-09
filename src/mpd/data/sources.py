"""Pinned external data: what `mpd fetch` downloads, from where, and the checksums it must match."""

from __future__ import annotations

from dataclasses import dataclass

ZENODO_FILES = "https://zenodo.org/records/{record}/files"


class SourceError(RuntimeError):
    pass


@dataclass(frozen=True)
class RemoteFile:
    name: str
    size: int | None
    md5: str | None              # the checksum Zenodo publishes for the file
    sha256: str | None = None    # computed from a copy whose md5 matched; checked as well when present


@dataclass(frozen=True)
class Source:
    key: str
    title: str
    dirname: str                 # directory under <home>/data
    zenodo_record: str | None    # None until the record is published
    doi: str | None
    license: str
    files: tuple[RemoteFile, ...]
    unpack: tuple[str, ...] = ()  # archives to extract after download
    note: str = ""
    # A Hugging Face dataset pinned to a revision and to per-file hashes (a package-data TSV: path, size, sha256).
    hf_repo: str | None = None
    hf_revision: str | None = None
    manifest: str | None = None

    def base_url(self) -> str:
        if not self.zenodo_record:
            raise SourceError(
                f"{self.key} has no published Zenodo record yet. Pass --from-dir <folder with the files> "
                f"or --zenodo-record <id>.")
        return ZENODO_FILES.format(record=self.zenodo_record)


# ProteinGym v1.3 from the official Zenodo archive (MIT). Only what the pipeline reads: the single/multiple
# substitution assays (labels), the reference table, and the AF2 structures. No MSAs are used by any student.
PROTEINGYM = Source(
    key="proteingym",
    title="ProteinGym v1.3 (official release)",
    dirname="proteingym-v1.3",
    zenodo_record="15293562",
    doi="10.5281/zenodo.15293562",
    license="MIT",
    files=(
        RemoteFile("DMS_substitutions.csv", 208_734, "c434631737013fceb56efc98056151e0",
                   "a8f498011532a74aa9fe556a50555a75e928c5837d19c06a87592ae04049b308"),
        RemoteFile("DMS_ProteinGym_substitutions.zip", 43_021_128, "ca1a4d46941ef33cc972245347118c7d",
                   "3a83766254ac9ac9984ec25cb73c6e010ea4418f5e35f143933e6b6e6473b921"),
        RemoteFile("ProteinGym_AF2_structures.zip", 19_173_248, "442881616899de02cd1c1b0a57badd37",
                   "ed8611bf12cfb840ea1c2561be36254901008610a49b1be475e917b6c5d6d56a"),
    ),
    unpack=("DMS_ProteinGym_substitutions.zip", "ProteinGym_AF2_structures.zip"),
    note="Mutations are numbered 1-based on the full `target_seq` in every assay; the structures are the official AF2 models.",
)

# Structures for the 489 VenusMutHub proteins the benchmark uses (built by scripts/maintainers/build_venus_structure_package.py).
# Fill in zenodo_record / doi / md5 after the deposit is published; until then use `--from-dir`. The deposit is in
# preparation: do not invent a record id. Once it is published, these are the files to list (sha256 from the deposit's
# SHA256SUMS; add the md5 Zenodo shows for each):
#   RemoteFile("proteinmutpd-venusmuthub-structures-v1.tar.gz", 18_955_417, None,
#              "c1331f9adf53c51cee2bb34da3548d216145ea948025e51703f67be8e9c92070")
#   RemoteFile("structures_manifest.tsv", None, None, "e1bfb88581a9495f344f4244f3039bad0dae2daf8cffab1315d0bbf16e8633f3")
#   RemoteFile("README.md", None, None, "bd8ede74ff44ea5148099d7f3f9937c5bceb596426f10e2b3babaa9eb90836a3")
VENUS_STRUCTURES = Source(
    key="venus-structures",
    title="VenusMutHub structure set v1 (experimental / AlphaFold DB / ESMFold)",
    dirname="venusmuthub-structures-v1",
    zenodo_record=None,
    doi=None,
    license="CC-BY-4.0",
    files=(),
    unpack=("proteinmutpd-venusmuthub-structures-v1.tar.gz",),
    note="Needed only for structure-aware students (SaProt). ESM2 students use sequence only.",
)

# VenusMutHub single-mutant assays at the revision the benchmark was built from. All 796 files are pinned by
# sha256 and are byte-identical to the files of the original study.
VENUSMUTHUB = Source(
    key="venusmuthub",
    title="VenusMutHub single_mutant (AI4Protein/VenusMutHub)",
    dirname="venusmuthub-v1",
    zenodo_record=None,
    doi=None,
    license="MIT",
    files=(),
    hf_repo="AI4Protein/VenusMutHub",
    hf_revision="08a27d07b5764af58f99d26734d096ebe2a5f9cf",
    manifest="venusmuthub_single_mutant.tsv",
    note="796 small CSV files fetched one at a time (Hugging Face rate-limits bursts); labels and sequences only.",
)

SOURCES = {s.key: s for s in (PROTEINGYM, VENUSMUTHUB, VENUS_STRUCTURES)}


def get(key: str) -> Source:
    try:
        return SOURCES[key]
    except KeyError:
        raise ValueError(f"unknown data source {key!r}; choose from {', '.join(SOURCES)}") from None


@dataclass(frozen=True)
class Tool:
    """An external program fetched as a release archive, pinned by the archive's and the binary's sha256."""
    key: str
    title: str
    release: str
    commit: str
    url: str                     # the immutable release asset
    archive: str                 # file name of the archive
    archive_size: int
    archive_sha256: str
    binary: str                  # path of the executable inside the archive
    binary_size: int
    binary_sha256: str
    license: str
    dirname: str                 # directory under <home>/tools
    note: str = ""


# Foldseek computes the 3Di structure alphabet SaProt reads. GPL-3.0: it is downloaded from its owners and run as a
# separate process, never bundled (a copy in a container image, release asset or deposit would carry source and licence
# obligations). Release 10 is pinned because it is an immutable upstream asset; the earlier study ran a rolling build of
# 2026-09-09 (FOLDSEEK_REFERENCE_BUILD below). The 3Di encoder source and weights are unchanged between the two according
# to the upstream commit history, but identical output has not been checked: see docs/SAPROT.md.
FOLDSEEK = Tool(
    key="foldseek",
    title="Foldseek release 10-941cd33 (linux, AVX2)",
    release="10-941cd33",
    commit="941cd33ff0771cd2e3f144e3293e22a2b87e9fda",
    url="https://github.com/steineggerlab/foldseek/releases/download/10-941cd33/foldseek-linux-avx2.tar.gz",
    archive="foldseek-linux-avx2.tar.gz",
    archive_size=11_941_946,
    archive_sha256="e6a3e3860e76b0b7b858736f627fe680d4d7f0fc7298301d8a92b9f711d74ac7",
    binary="foldseek/bin/foldseek",
    binary_size=29_402_960,
    binary_sha256="4db94a7825d409e416643772860ddf7cfe328d582679b47e6dffda7df4d41ea0",
    license="GPL-3.0",
    dirname="foldseek-10-941cd33",
    note="Static x86-64 binary built for CPUs with AVX2. Run as a separate process; never bundled.",
)

# The build the earlier study's recorded SaProt numbers came from (a mutable "latest" download of 2026-09-09).
FOLDSEEK_REFERENCE_BUILD = {
    "version": "463739e0014a1549a527de589102cde98f802f37",
    "binary_sha256": "56ec365f98227b07b7de4eba97aa1ae5f3cbf70fd9819311fab98eec42ada6ae",
}

TOOLS = {t.key: t for t in (FOLDSEEK,)}


def get_tool(key: str) -> Tool:
    try:
        return TOOLS[key]
    except KeyError:
        raise ValueError(f"unknown tool {key!r}; choose from {', '.join(TOOLS)}") from None
