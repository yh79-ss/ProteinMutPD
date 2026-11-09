# Data notes

## What `mpd fetch` downloads

Every file is pinned by size and checksum and refused if it does not match. Run these on a login node: compute nodes have no internet and
the jobs run offline ([GRACE.md](GRACE.md#why-jobs-are-offline)). Sizes are from the pinned manifests and a real fetch.

| command | what | download | on disk | licence |
|---|---|---|---|---|
| `mpd fetch proteingym` | ProteinGym v1.3 substitution assays, reference table, AF2 structures; official [Zenodo record 15293562](https://doi.org/10.5281/zenodo.15293562) | 62.4 MB | 1.2 GB, 421 files | MIT |
| `mpd fetch venusmuthub` | the 796 `single_mutant` files of `AI4Protein/VenusMutHub` at revision `08a27d07`, fetched one at a time | 6.3 MB | 797 files | MIT |
| `mpd fetch venus-structures` | one structure for each of the 489 VenusMutHub proteins; **Zenodo deposit in preparation** (below) | 19 MB | | CC BY 4.0 |
| `mpd fetch model KEY ...` | pinned model weights, for example `esm2-3b esm2-650m` (13.98 GB); see [MODELS.md](MODELS.md) | | | MIT (ESM2, SaProt) |
| `mpd fetch foldseek` | the Foldseek release SaProt needs to make structure strings; see [SAPROT.md](SAPROT.md) | 11.9 MB (the archive is removed after unpacking) | 29.4 MB | GPL-3.0, never bundled |

`mpd assays [--dataset proteingym|venusmuthub] [--grep TEXT]` lists the assay ids of the fetched data (what `--assay` takes). Real output on the
fetched data (CPU, a login node of another cluster, not Grace):

```text
$ mpd assays --dataset proteingym --grep BLAT
proteingym: 4 of 217 assay ids containing 'BLAT'
  BLAT_ECOLX_Deng_2012
  BLAT_ECOLX_Firnberg_2014
  BLAT_ECOLX_Jacquier_2013
  BLAT_ECOLX_Stiffler_2015
$ mpd assays --dataset venusmuthub --grep 3FCR
venusmuthub: 2 of 793 assay ids containing '3FCR'
  3FCR_activity-substrate-1
  3FCR_activity-substrate-2
```

Sequence-only students (ESM2) need neither the Venus structures nor Foldseek. Licences: [LICENSES.md](LICENSES.md).

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

Sequences and labels come from `AI4Protein/VenusMutHub` (MIT), directory `single_mutant`, pinned to revision
`08a27d07` with all 796 files hashed. These files are byte-identical to the ones the original study recorded, and the
loader reproduces its cohort exactly: 793 assays, 489 proteins, 25,044 rows, every label equal.

From the 796 public files:

- 3 assays on 2 proteins (1,048 and 1,267 residues) are left out; the original input curation had marked them
  unsupported, so they have no structure. This is an explicit list, not a length cutoff: included proteins reach 1,450.
- 2 rows with a non-numeric label (`NDA`, `1.25 ± 0.04`) are dropped.
- 198 identity rows (`M1M`, the wild type measured) are kept.

Hugging Face rate-limits bursts of requests from one address (a parallel `snapshot_download` of these files got
HTTP 429 from a cluster node), so `mpd fetch venusmuthub` fetches one file at a time and waits out `Retry-After`.

Structures for those 489 proteins are not in VenusMutHub. The structure set (159 experimental, 232 AlphaFold DB,
98 ESMFold) is to be distributed as a Zenodo deposit with a manifest recording the source, version, confidence scale and
checksum of every file. **Status: in preparation.** The deposit is built and verified locally but not published, so there is no record
id, DOI or md5 yet; `mpd fetch venus-structures` works only for someone who holds the deposit files (`--from-dir <folder>`), and
until it is published nobody else can run SaProt on VenusMutHub. The sha256 values that will pin it are recorded in
`src/mpd/data/sources.py`; [ZENODO.md](ZENODO.md) lists the steps that finish it. ProteinGym needs nothing unpublished. The B-factor
column means different things by source:

| source | column | pLDDT <70 mask |
|---|---|---|
| AlphaFold DB | pLDDT, 0-100 | use as is |
| ESMFold (as run here) | pLDDT, 0-1 | multiply by 100 first |
| experimental | crystallographic B-factor | do not mask by confidence |

Applying `value >= 70` to the ESMFold column without scaling would mask every residue.

## Structures for SaProt

SaProt reads a Foldseek 3Di letter per residue, computed from a structure and masked where the model's confidence is below 70 pLDDT.
Sequence-only students need none of this. SaProt is `available`, tested on CPU only (nothing has run on Grace or a GPU). Facts below were
measured with the SaProt library of this repository on CPU; the full account is in [SAPROT.md](SAPROT.md).

| benchmark | structures | confidence column | rule |
|---|---|---|---|
| ProteinGym v1.3 | the official AF2 models (`mpd fetch proteingym`); `pdb_range` (1-based, inclusive, over `target_seq`) says which part of the sequence each covers | pLDDT, 0-100 | mask below 70 |
| VenusMutHub | the Venus structure set (not yet published) | AlphaFold DB: pLDDT 0-100; ESMFold: pLDDT 0-1, multiplied by 100 first; experimental: crystallographic B-factor, never masked | as left |

ProteinGym's structures are part of the ordinary download: `mpd fetch proteingym` fetches and unpacks `ProteinGym_AF2_structures.zip` (19,173,248 bytes,
pinned by md5 and sha256 in `src/mpd/data/sources.py`) into `<home>/data/proteingym-v1.3/ProteinGym_AF2_structures/` (197 files; 186 of them are referenced by the 217
assays). Nothing else has to be fetched for SaProt on ProteinGym except Foldseek and the weights. `mpd plan` reports the structures as missing if that folder holds no
`.pdb` file.

What a SaProt campaign adds to its directory:

| file | written by | content |
|---|---|---|
| `contexts/structures.json` | `prepare`, only when the teacher (run live) or the student reads structure | one state string per sequence (keyed by the sha256 of the sequence), the engine and its binary hash, the masking policy, per-sequence provenance (file name and hash, offset, substitutions, masked residues), `dms_labels_opened: false`; strings that two datasets give differently for one sequence are stored under that dataset |
| `<home>/cache/3di/<engine id>/<sha256 of the structure file>.json` | `prepare` | Foldseek's answer for one structure file, shared by every campaign of the home; a second campaign on the same data does not start Foldseek again |

`contexts/MANIFEST.json` gains a `structures` block (digest, number of sequences, fraction masked, engine, policy, datasets, number of conflicts, `dms_labels_opened: false`)
exactly when `structures.json` exists; campaigns that read no structure keep their files and digests unchanged. `teacher/COMPLETE.json` (SaProt teacher),
`train/<unit>/CONFIG.json` (SaProt student) and every `score/.../shard_*.json` (SaProt student) carry `structures_digest`, the digest of `structures.json`. A teacher table or a
resumable checkpoint made with other strings is recomputed or not resumed.

- Three ProteinGym assays are cropped (ZIKV, BRCA2, POLG HCVJF): residues outside the model get no structure state. The Cas9 model has a
  22-residue tag that is ignored, and three P53 assays carry one residue that differs from the shared model (`R72P`, recorded).
- Ten ProteinGym sequences also occur in VenusMutHub with different structures; each dataset is scored with its own.
- The earlier study's recorded SaProt-on-Venus numbers contain a scale error for the 98 ESMFold proteins (all their residues were masked).
  `mpd` masks correctly, so those cells will differ; [SAPROT.md](SAPROT.md) has the numbers.

## Training contexts

A context is one masked position of one sequence. Direct-logit distillation uses **dense** contexts: every residue of
each assay's region (`MSA_start`..`MSA_end` for ProteinGym, the whole sequence for Venus), which gives the 62,671
ProteinGym contexts of the original study. Score distillation uses **observed** contexts: the assayed positions, each
with the mutants that were measured (37,753 positions and 592,774 variants for ProteinGym). Evaluation always scores
the assayed positions (ProteinGym 37,753; Venus 8,945).

## Labels

Labels are read only by the evaluation stage; `tests/test_stages.py` fails if any other stage touches them.

## Regression against the earlier study

The earlier study's stored ESM2-650M base-model outputs (mutant-minus-wild-type logits at each scored position)
were replayed through the loaders and evaluator here, with no model run:

| benchmark | assay-macro Spearman, this code | recorded |
|---|---|---|
| VenusMutHub, 793 assays, all rows | 0.2095016689 | 0.2095016689 |
| ProteinGym, 217 assays | 0.4279547 (float64 labels) | 0.4279488 |
| ProteinGym with labels rounded to float32 | 0.4279487815, all 217 assays equal | 0.4279487815 |

The earlier evaluator stored the labels as `float32`, which creates a few ties between close label values. This code
keeps float64 labels, so ProteinGym macro Spearman is higher by 6e-6 (largest single assay: 1.2e-3,
`ENVZ_ECOLI_Ghose_2023`). The ProteinGym labels themselves are identical between GenBio's archive and the official
release for all 217 assays.

Separately, running ESM2-650M on CPU (fp32) on six short assays reproduces the recorded per-assay Spearman to within
0.003 (the earlier run used bf16 on GPU). Not yet checked against recorded numbers: the model forward pass on long
sequences, and the five assays whose official sequence is longer than the cropped query the earlier study used.
