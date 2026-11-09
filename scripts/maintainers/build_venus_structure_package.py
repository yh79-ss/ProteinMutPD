#!/usr/bin/env python3
r"""Build the Zenodo deposit for the VenusMutHub structure set (maintainers only).

Inputs are the curation records of the original study (a `final_selection.jsonl` of per-protein source
records, the curated PDB files they point to, and a JSON with the benchmark sequences). Output is a folder to
upload to Zenodo: the archive, `structures_manifest.tsv`, `SHA256SUMS`, `README.md` and `zenodo_metadata.json`.
Nothing is uploaded.

    python scripts/maintainers/build_venus_structure_package.py \
        --selection .../structures/final_selection.jsonl --root .../venus_rag_inputs_v1 \
        --sequences .../PLAN.json --out zenodo_out --creator "Surname, Given" --affiliation "University"
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import tempfile
from pathlib import Path

from mpd.data import venus_structures as vs

EXPECTED_PDB_RECORDS = {"MODEL", "ATOM", "TER", "ENDMDL", "END"}
LICENSE_ID = "cc-by-4.0"


def check_pdb(path: Path) -> None:
    """The README promises single-model, single-chain, ATOM-only files; refuse anything else."""
    kinds, chains, models = set(), set(), 0
    for line in path.read_text().splitlines():
        kind = line[:6].strip()
        kinds.add(kind)
        models += kind == "MODEL"
        if kind == "ATOM":
            chains.add(line[21])
    if not kinds <= EXPECTED_PDB_RECORDS or chains != {"A"} or models != 1:
        raise ValueError(f"{path.name}: records {sorted(kinds - EXPECTED_PDB_RECORDS)}, chains {sorted(chains)}, models {models}")


def source_fields(tier: str, sel: dict) -> dict:
    if tier == "exact_experimental_pdb":
        return dict(source="experimental", source_id=sel["candidate_pdb_id"], chain=sel["selected_chain"],
                    source_url=f"https://www.rcsb.org/structure/{sel['candidate_pdb_id']}", confidence_scale="bfactor",
                    upstream_sha256=sel["source_sha256"])
    if tier == "exact_sequence_alphafold_db":
        return dict(source="alphafold_db", source_id=sel["entry_id"], chain="A", uniprot_accession=sel["accession"],
                    source_url=sel["pdb_url"], source_version=f"v{sel['version']}", mean_plddt=f"{sel['mean_plddt']:.2f}",
                    confidence_scale=sel["pdb_bfactor_confidence_scale"], upstream_sha256=sel["raw_pdb_sha256"])
    if tier == "de_novo_esmfold_fallback":
        return dict(source="esmfold", source_id=sel["model_id"], chain="A", source_version=sel["model_revision"],
                    mean_plddt=f"{sel['mean_plddt']:.2f}", confidence_scale=sel["pdb_bfactor_confidence_scale"],
                    upstream_sha256=sel["raw_sha256"])
    raise ValueError(f"unknown structure tier {tier!r}")


def collect(selection: Path, root: Path, sequences: Path) -> tuple[list[dict], list[tuple[str, Path]]]:
    seqs = {r["protein_id"]: r["sequence"] for r in json.loads(sequences.read_text())["records"]}
    rows, members = [], []
    for line in selection.read_text().splitlines():
        rec = json.loads(line)
        if not rec.get("source_tier"):
            continue                                   # proteins with no usable structure are not part of the set
        pid, sel = rec["protein_id"], rec["selected"]
        seq = seqs[pid]
        if vs.protein_id_for(seq) != pid or vs.sequence_sha256(seq) != rec["sequence_sha256"]:
            raise ValueError(f"{pid}: sequence does not match its id / recorded hash")
        path = root / sel["curated_relative_path"]
        digest = vs.sha256_file(path)
        if digest != sel["curated_sha256"]:
            raise ValueError(f"{pid}: {path} changed since curation")
        check_pdb(path)
        fields = source_fields(rec["source_tier"], sel)
        arc = f"structures/{fields['source']}/{pid}.pdb"
        rows.append({"protein_id": pid, "sequence_sha256": rec["sequence_sha256"], "length": len(seq),
                     "file": arc, "file_sha256": digest, **fields})
        members.append((arc, path))
    if len({r["protein_id"] for r in rows}) != len(rows):
        raise ValueError("duplicate protein ids")
    return rows, members


def readme(rows: list[dict], version: str, doi: str | None, repo: str) -> str:
    n = collections.Counter(r["source"] for r in rows)
    cite = f"doi:{doi}" if doi else "the DOI shown on this Zenodo record"
    return f"""# ProteinMutPD: VenusMutHub structure set v{version}

One protein structure for each of the {len(rows)} VenusMutHub proteins used by the ProteinMutPD benchmark
({repo}). Structure-aware students (for example SaProt, which reads Foldseek 3Di tokens) need a structure for
every evaluated sequence; sequence-only students (ESM2) do not need this set.

| source | proteins | how it was chosen |
|---|---|---|
| experimental (PDB) | {n['experimental']} | a deposited entry whose chain matches the sequence exactly |
| AlphaFold DB | {n['alphafold_db']} | the AlphaFold DB entry for the exact sequence (model version in the manifest) |
| ESMFold | {n['esmfold']} | `facebook/esmfold_v1`, used only when neither of the above existed |

The selection was made from sequences alone; no DMS labels or scores were read. Two further proteins were
excluded because their length is unsupported; they have no structure here and are not part of the benchmark.

## Files

- `{vs.ARCHIVE_NAME}`: `structures/<source>/<protein_id>.pdb`
- `{vs.MANIFEST_NAME}`: one row per protein (source ids, accession, version, confidence scale, checksums)
- `{vs.SUMS_NAME}`: sha256 of the files above

`protein_id` is `venus_seq_` + the first 12 hex digits of `sha1(sequence)`, so it can be recomputed from the
VenusMutHub sequence. `sequence_sha256` is the full sha256 of the sequence.

## Use

    mpd fetch venus-structures --from-dir <folder with these files> --home <work dir>

verifies every checksum, unpacks the archive and checks each structure against the manifest.

## What the PDB files are

They are the copies used in the original study, not the upstream files. Header, REMARK and HETATM records were
removed; each file holds one model and one chain (written as chain A) of ATOM records. `upstream_sha256` is
the hash of the upstream file the curation used.

- **AlphaFold DB and ESMFold:** the same atoms, coordinates and confidence values as the upstream model file.
- **Experimental:** derived from the selected chain of the deposited entry. Residue numbering, atom order,
  hydrogens and alternate locations can differ from the deposited file, so these are not byte-for-byte copies.

**The B-factor column means different things:**

| `confidence_scale` | source | meaning |
|---|---|---|
| `percent_0_to_100` | AlphaFold DB | pLDDT, 0-100 |
| `fraction_0_to_1` | ESMFold (as run here) | pLDDT as a fraction, 0-1; multiply by 100 before applying a 70 cutoff |
| `bfactor` | experimental | crystallographic B-factor; not a confidence value |

## Licence and attribution

This deposit is released under CC BY 4.0. It contains material from:

- **wwPDB / RCSB PDB** (CC0 1.0): credit the original depositors; the entry ids are in the manifest.
- **AlphaFold Protein Structure Database** (CC BY 4.0, EMBL-EBI / Google DeepMind): please cite Jumper et al.,
  Nature (2021) and the current citation on the AlphaFold DB download page.
- **ESMFold** predictions made with `facebook/esmfold_v1` (MIT licence), revision
  `{next((r['source_version'] for r in rows if r['source'] == 'esmfold'), 'n/a')}`; cite Lin et al., Science (2023).
- Protein sequences come from **VenusMutHub** (MIT licence, `AI4Protein/VenusMutHub` on Hugging Face).

## Cite

If you use this set, cite {cite} and the ProteinMutPD repository ({repo}).
"""


def zenodo_metadata(rows: list[dict], version: str, creator: str, affiliation: str, repo: str) -> dict:
    desc = (f"<p>One structure for each of the {len(rows)} VenusMutHub proteins used by the ProteinMutPD benchmark: "
            "experimental PDB entries, AlphaFold DB models and ESMFold predictions, with a manifest recording the source, "
            "version, confidence scale and checksum of every file. Used by structure-aware students such as SaProt. "
            "See README.md for processing details, licences and attribution.</p>")
    return {"metadata": {
        "upload_type": "dataset", "title": f"ProteinMutPD: VenusMutHub structure set v{version}", "description": desc,
        "creators": [{"name": creator, "affiliation": affiliation}], "license": LICENSE_ID, "access_right": "open",
        "version": version,
        "keywords": ["protein structure", "VenusMutHub", "protein language models", "mutation effect prediction",
                     "knowledge distillation"],
        "related_identifiers": [
            {"identifier": repo, "relation": "isSupplementTo", "resource_type": "software"},
            {"identifier": "https://huggingface.co/datasets/AI4Protein/VenusMutHub", "relation": "references",
             "resource_type": "dataset"},
        ]}}


def verify_output(out: Path, rows: list[dict]) -> None:
    problems = vs.verify_sums(out)
    with tempfile.TemporaryDirectory() as tmp:
        vs.safe_extract_tar(out / vs.ARCHIVE_NAME, Path(tmp))
        problems += vs.verify_structures(Path(tmp), vs.read_manifest(out / vs.MANIFEST_NAME))
    if problems:
        raise SystemExit("verification failed:\n  " + "\n  ".join(problems[:10]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selection", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True, help="folder the selection's curated_relative_path is relative to")
    ap.add_argument("--sequences", type=Path, required=True, help="JSON with records[].protein_id/sequence")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--creator", required=True, help='"Surname, Given"')
    ap.add_argument("--affiliation", required=True)
    ap.add_argument("--version", default="1.0.0")
    ap.add_argument("--doi", help="reserved Zenodo DOI, if you have one (goes into README only)")
    ap.add_argument("--repo", default="https://github.com/yh79-ss/ProteinMutPD")
    args = ap.parse_args()

    if args.out.exists() and any(args.out.iterdir()):
        sys.exit(f"{args.out} is not empty")
    rows, members = collect(args.selection, args.root, args.sequences)
    args.out.mkdir(parents=True, exist_ok=True)
    vs.write_archive(members, args.out / vs.ARCHIVE_NAME)
    vs.write_manifest(rows, args.out / vs.MANIFEST_NAME)
    (args.out / "README.md").write_text(readme(rows, args.version, args.doi, args.repo), encoding="utf-8")
    (args.out / "zenodo_metadata.json").write_text(
        json.dumps(zenodo_metadata(rows, args.version, args.creator, args.affiliation, args.repo), indent=2) + "\n")
    names = (vs.ARCHIVE_NAME, vs.MANIFEST_NAME, "README.md")
    (args.out / vs.SUMS_NAME).write_text("".join(f"{vs.sha256_file(args.out / n)}  {n}\n" for n in names))
    verify_output(args.out, rows)
    counts = collections.Counter(r["source"] for r in rows)
    size = (args.out / vs.ARCHIVE_NAME).stat().st_size
    print(f"{len(rows)} structures {dict(counts)}; archive {size / 1e6:.1f} MB; verified (checksums, re-extraction, manifest)")
    print(f"upload these from {args.out}: {', '.join(names)}  (zenodo_metadata.json is for the deposit form / API)")


if __name__ == "__main__":
    main()
