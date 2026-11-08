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
# Fill in zenodo_record / doi / md5 after the deposit is published; until then use `--from-dir`.
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
