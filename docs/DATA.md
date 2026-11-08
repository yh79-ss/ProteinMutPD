# Data notes

## ProteinGym: official release, not the GenBio copy

`mpd fetch proteingym` pins ProteinGym v1.3 from its Zenodo record
([10.5281/zenodo.15293562](https://doi.org/10.5281/zenodo.15293562)): 217 assays, 696,311 single substitutions.

**Mutation numbering.** In every assay the mutation `X123Y` indexes the full `target_seq` as `target_seq[123 - 1]`,
and the `mutated_sequence` column agrees with that for all 696,311 rows. `MSA_start` is not needed; subtracting it
instead picks residues that do not match the wild type in 4 assays (the long proteins).

**Differences from GenBio's `ProteinGYM-DMS-RAG-zeroshot` copy** (revision `7ea80896`), checked file by file:

- Assay ids are identical (217). For 212 assays the sequence is identical; for 5 long proteins (ZIKV, KCNH2, POLG
  CXB3N, POLG HCVJF, SCN5A) GenBio's query is a cropped region of the official full-length `target_seq`.
- Structures are **different predictions**: none of the 217 files is byte-identical, and for each of the 214 whose
  chain can be compared (same or cropped sequence), the CA-CA distances and the per-residue confidence values both
  differ. The official release ships 197 structure files shared across the 217 assays, one per protein.

Consequence: base-model numbers for structure-aware students (SaProt) computed with the official structures need not
equal numbers computed with GenBio's. Treat earlier values as a reference to be compared, not reproduced exactly.
Sequence-only students differ only for the 5 cropped assays.

## VenusMutHub

Sequences and labels come from `AI4Protein/VenusMutHub` (MIT). The benchmark uses 489 proteins, 793 assays and
25,044 rows. The Venus loader (not written yet) will record how these follow from the public files.

Structures for those 489 proteins are not in VenusMutHub. The structure set (159 experimental, 232 AlphaFold DB,
98 ESMFold) is distributed as a Zenodo deposit with a manifest recording the source, version, confidence scale and
checksum of every file. The B-factor column means different things by source:

| source | column | pLDDT <70 mask |
|---|---|---|
| AlphaFold DB | pLDDT, 0-100 | use as is |
| ESMFold (as run here) | pLDDT, 0-1 | multiply by 100 first |
| experimental | crystallographic B-factor | do not mask by confidence |

Applying `value >= 70` to the ESMFold column without scaling would mask every residue.

## Labels

Labels are read only by the evaluation stage.
