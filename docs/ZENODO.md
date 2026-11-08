# Publishing the Venus structure set on Zenodo (maintainers)

Publishing creates a permanent, citable record. A DOI cannot be deleted, so check the files first.

## 1. Build the deposit

```bash
python scripts/maintainers/build_venus_structure_package.py \
    --selection <venus_rag_inputs>/structures/final_selection.jsonl \
    --root <venus_rag_inputs> --sequences <PLAN.json with the 489 sequences> \
    --out <staging>/proteinmutpd-venusmuthub-structures-v1 \
    --creator "Surname, Given" --affiliation "Institution"
```

The script checks that every PDB is one model, one chain, ATOM records only; that each protein id and sequence hash
match; and that the finished archive re-extracts and verifies. The archive is byte-reproducible. Nothing is uploaded.

Output (about 19 MB): `proteinmutpd-venusmuthub-structures-v1.tar.gz`, `structures_manifest.tsv`, `SHA256SUMS`,
`README.md`, and `zenodo_metadata.json` (the deposit form fields; not uploaded as a file).

## 2. Deposit

1. Zenodo → New upload. Upload the archive, `structures_manifest.tsv`, `SHA256SUMS` and `README.md`.
2. Fill the form from `zenodo_metadata.json`: type *Dataset*, licence *CC BY 4.0*, creators, keywords, related
   identifiers. Add co-authors here if they should be credited.
3. Click *Reserve DOI*. If you want the DOI printed in the README, rebuild with `--doi <doi>` and upload that
   README instead (the archive is unchanged).
4. Publish.

## 3. Pin it in the code

In `src/mpd/data/sources.py`, set `VENUS_STRUCTURES.zenodo_record` and `doi`, and list the three files with their
md5 (shown on the Zenodo record) and sha256 (`SHA256SUMS`). Until then
`mpd fetch venus-structures --from-dir <folder>` verifies against the deposit's own `SHA256SUMS`.

## Licensing of the content

CC BY 4.0 for the deposit as a whole. It contains wwPDB entries (CC0), AlphaFold DB models (CC BY 4.0, cite the
AlphaFold papers per the current EMBL-EBI guidance) and ESMFold predictions (`facebook/esmfold_v1`, MIT). The README
inside the deposit carries the per-source attribution.
