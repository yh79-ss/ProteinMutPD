# Publishing the Venus structure set on Zenodo (maintainers)

Publishing creates a permanent, citable record. A DOI cannot be deleted, so check the files first.

**Status: in preparation.** The deposit (489 structures: 159 experimental, 232 AlphaFold DB, 98 ESMFold; about 19 MB) has been built and
verified locally. Nothing has been uploaded, there is no record id, DOI or md5, and `mpd fetch venus-structures` therefore works only from
a local copy (`--from-dir <folder>`). Until it is published, nobody except a holder of the files can run SaProt on VenusMutHub; ESM2
students and everything on ProteinGym are unaffected ([DATA.md](DATA.md#venusmuthub), [SAPROT.md](SAPROT.md)). The checksums that will pin
it are already recorded, so publishing only needs the record id, the DOI and the md5 values.

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
md5 (shown on the Zenodo record) and sha256 (`SHA256SUMS`). The comment above `VENUS_STRUCTURES` already holds the sizes and sha256 of the
built files (the archive is 18,955,417 bytes); if you rebuilt the README with `--doi`, its sha256 changes, so copy the values from the
`SHA256SUMS` you uploaded. Until then `mpd fetch venus-structures --from-dir <folder>` verifies against the deposit's own `SHA256SUMS`.

## Licensing of the content

CC BY 4.0 for the deposit as a whole. It contains wwPDB entries (CC0), AlphaFold DB models (CC BY 4.0, cite the
AlphaFold papers per the current EMBL-EBI guidance) and ESMFold predictions (`facebook/esmfold_v1`, MIT). The README
inside the deposit carries the per-source attribution.

## 4. After publishing

1. Test it from a clean home: `mpd fetch venus-structures --home <empty-folder>` must download, verify and unpack, and a second run must
   make no download.
2. Change the statements that say "in preparation": the data table and the Venus structure sentence in [DATA.md](DATA.md), the README
   (Models, data and licences; Status), the "Venus structures: availability" section of [SAPROT.md](SAPROT.md), the Venus row of
   [LICENSES.md](LICENSES.md), and the line in `mpd fetch --list` (it reads from `sources.py`).
3. Say in the release notes ([CHANGELOG.md](../CHANGELOG.md)) that SaProt on VenusMutHub no longer needs a local copy.
