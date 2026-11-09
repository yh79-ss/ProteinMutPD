# SaProt

[SaProt](https://github.com/westlake-repl/SaProt) reads a protein as one token per residue: the amino acid and a
structural letter from [Foldseek](https://github.com/steineggerlab/foldseek)'s 3Di alphabet. `mpd` can use SaProt-35M and
SaProt-650M as a student or a teacher.

> **Status.** SaProt-35M and SaProt-650M are `available` in `mpd models`. The library (structure strings, the Foldseek
> wrapper, the model class, the benchmark changes) and the stages that use it (`prepare` builds the strings, `teacher`,
> `train` and `score` read them) are written and tested on CPU, with tiny random models and a fake Foldseek; the library
> was also compared with the earlier study's stored outputs. Nothing here has run on Grace, on a GPU, or through Slurm.
> The pinned Foldseek release has never been executed, and SaProt on VenusMutHub needs a structure set that is not
> published yet.

Every statement below is one of: *tested here* (on CPU, with this code), *from a document* (cited), *from the earlier analysis* (see
next paragraph), or *not verified* (said so). The *earlier study* is the maintainers' previous SaProt and ESM2 distillation work, whose
stored outputs are the "recorded" numbers. The *earlier analysis* is a separate review of that study made before this code was
written; what it found is repeated only where "tested here" or "measured here" says it was checked again.

## What the model sees

| | |
|---|---|
| token | `AA` + `state`, for example `Ev` (glutamate in 3Di state `v`); 446 tokens; `#` is a masked amino acid or an unknown state |
| scoring a position | the residue's token becomes `#` + its own state (`#d`); the `<mask>` token is never used |
| output | the 21 logits of one amino acid (one per state) are combined by log-sum-exp, giving the 20 amino-acid log-marginals in `ACDEFGHIKLMNPQRSTVWY` order; differences of these equal SaProt's own `log(p_mut / p_wt)` score |
| long sequences | more than 1,024 tokens: the 1,024-token window centred on the position, the same function as for ESM2 |
| dropout | forced to 0. The SaProt-650M configuration on Hugging Face has 0.1; the 35M one has 0.0 |
| training | LoRA rank 8, alpha 16 on query, key, value and attention output, the same recipe as ESM2: 368,640 trainable parameters on SaProt-35M (measured here), 2,703,360 on 650M (from its configuration; the earlier study reported the same figure) |

*Tested here:* the vocabulary (sha256 `d4289b3f...`, ids `Ap`=5, `A#`=25, `Md`=223, `#p`=425, `##`=445), SaProt's README
tokenization example, the focal-token rule, the window for a 1,100-residue sequence, the marginal against the softmax-ratio
formula (to 1e-5), dropout 0, the LoRA parameter count, an adapter round trip, and agreement with the earlier study's recipe
written out independently (to 1e-5 in float32 on a random model). Loading the SaProt-35M `pytorch_model.bin` from a local folder
works with transformers 4.51.0, torch 2.10.0 and peft 0.13.2.

Loading prints "Some weights of EsmForMaskedLM were not initialized ... `esm.contact_head.regression.*`,
`esm.embeddings.position_embeddings.weight`". This is expected: the model uses rotary positions and a masked-language-model
head, and neither of those layers is used.

## Structure inputs

3Di is computed once, in the CPU `prepare` stage, for every sequence a structure-aware model reads, and frozen as
`contexts/structures.json` in the campaign. A structure-aware student needs the structures of the training data and of every
evaluation dataset; a structure-aware teacher (run live) needs those of the training data. Sequence-only students (ESM2)
need nothing and keep their file formats and digests.

| benchmark | structure files | B-factor column | masking |
|---|---|---|---|
| ProteinGym v1.3 | the official AF2 models: 187 distinct sequences in 217 assays use 186 of the 197 files. They are part of the `mpd fetch proteingym` download (`ProteinGym_AF2_structures.zip`, 19.2 MB, unpacked to `<home>/data/proteingym-v1.3/ProteinGym_AF2_structures/`; a fetch made for this work holds those 197 files) | pLDDT, 0-100 | pLDDT < 70 |
| VenusMutHub | one structure per protein from the Venus structure set: 159 experimental, 232 AlphaFold DB, 98 ESMFold | experimental: crystallographic B-factor; AlphaFold DB: pLDDT 0-100; ESMFold: pLDDT 0-1 | experimental never masked; AlphaFold DB pLDDT < 70; ESMFold pLDDT x 100 < 70 |

### Masking policy

* A residue gets its Foldseek letter if its confidence is at least 70.0 pLDDT, and `#` otherwise. 69.99 is masked, 70.0 is kept.
* Confidence is the B-factor of the residue's CA atom, scaled to 0-100 by the structure's declared scale (`Assay.structure_scale`).
  ESMFold files hold 0-1 values and are multiplied by 100 first. Experimental B-factors are not confidences and never mask.
* The string always has the length of the full assay sequence. Residues the structure does not cover (see below) and residues
  Foldseek calls unknown (`X`) are `#`. The amino-acid channel of the input is always the assay's own sequence.
* The CA rule is the earlier study's. SaProt's official code averages over all atoms of a residue instead. For AlphaFold DB and
  AF2 files the two are identical (the column is constant within a residue); for the 98 ESMFold files they differ for 816 of
  27,455 residues, 3.0% (from the earlier analysis, not re-counted here). The rule is recorded in `structures.json`.
* Tested here: the 69.99 / 70.0 and 0.699 / 0.70 edges for each source and that experimental files are never masked.

Measured with this code on the real files (Foldseek build of 2026-09-09, see below): the 187 ProteinGym sequences have 72,933
residues, of which 66,639 lie inside a structure; 16,074 of those (24.1%) are masked for pLDDT < 70. The 489 Venus structures have
135,060 residues, of which 14,293 (10.6%) are masked: AlphaFold DB 10,371 of 81,968, ESMFold 3,922 of 27,455, experimental 0 of 25,637.

### The ESMFold scale artefact in the earlier study's recorded SaProt numbers

The earlier study applied `value >= 70` to the raw B-factor column of every non-experimental structure. For ESMFold models that
column is a fraction between 0 and 1, so every residue of the 98 ESMFold proteins (154 of 793 assays, 5,756 of 25,044 rows) was
masked: SaProt saw those proteins with no structure at all. *Measured here:* the earlier study's frozen strings hold 27,455 `#`
for 27,455 ESMFold residues. The recorded SaProt-650M Venus base number (assay-macro Spearman 0.2238 over all 793 assays) therefore
contains this artefact; the 639 AlphaFold DB and experimental assays (0.2421, from the earlier study's report) do not.

`mpd` masks correctly by default. How much the correction matters was measured on a subset only: on the 36 ESMFold assays of the
CPU subset below, macro Spearman moves from 0.2510 (all residues masked, the recorded input) to 0.3722 (scale-aware masking), a mean
change of +0.121 per assay with single assays moving by up to 0.9 (assays with 6 to 9 variants flip ranks easily). The effect on
all 154 ESMFold assays, and so on the full 793-assay figure, is not measured: it needs about 1,960 forward passes of up to 955
residues. Cells of earlier SaProt-on-Venus tables computed with the old inputs should be re-run or annotated.

To reproduce the earlier inputs for a regression, set `MPD_STRUCTURE_LEGACY_MASK=1` (for a job: `--env MPD_STRUCTURE_LEGACY_MASK=1`).
It makes the raw column be compared with 70 whatever its scale. It is recorded in `structures.json` and the campaign manifest and has
no command-line flag; do not use it for new results.

### Partial structures and constructs

* ProteinGym's `pdb_range` (`a-b`, 1-based inclusive over `target_seq`) says which part of the sequence a model covers. Three
  assays are cropped: `A0A140D2T1_ZIKV_Sourisseau_2019` (504 of 3,423 residues), `BRCA2_HUMAN_Erwood_2022_HEK293T` (2,832 of
  3,418) and `POLG_HCVJF_Qi_2014` (244 of 3,033). Residues outside the model are `#` (6,294 residues in all, measured here), so the
  full-sequence context tables and the 1,024-token windows are unchanged. ProteinGym itself feeds the cropped slice, so windows near
  the edges of these three assays differ slightly from its.
* `CAS9_STRP1_Spencer_2017_positive`: the model has a 22-residue tag after the 1,368 residues of the sequence; the tag is ignored.
* `P53_HUMAN_Giacomelli_2018_*`: three assays carry arginine at position 72 where the shared model has proline. Up to three
  single-residue differences between sequence and model are accepted, the assay's residue stays in the amino-acid channel, the state
  comes from the model, and the difference is recorded (`R72P`). Anything else (an insertion, a deletion, more than three
  differences, a structure that occurs twice in the sequence) stops `prepare` with a message naming the first mismatch.
* An assay with several structures (`|` in `pdb_file` or `pdb_range`) does not occur in v1.3 and is not supported: it gets no
  structure, and `prepare` says so if a structure-aware model needs it.
* *Tested here:* each of these cases with the fake Foldseek and, on the real files, that the ProteinGym strings are exactly the
  recorded digest (below) with `R72P` as the only substitution.

### One sequence, two benchmarks

Ten ProteinGym sequences also occur in VenusMutHub, and for all ten the two datasets' structures give different strings (203 of
their 2,954 residues differ; measured here). A set keeps one string per sequence from the first dataset (the training data); a
different string from another dataset is stored separately and used when that dataset is scored
(`StructureSet.scoped(dataset)`), so every dataset is scored with its own structures, exactly as in a single-dataset campaign. A
lookup that does not name the dataset raises an error when the table holds one of these sequences, instead of choosing one.
*Tested here* with fake structures and, for the strings above, on the real files; through the stages in
`tests/test_saprot_pipeline.py`, which trains on each benchmark in turn and checks the teacher table and the scores against the
model run directly on that dataset's own strings.

### Which stage reads what

| stage | reads | what it keeps |
|---|---|---|
| `prepare` | the structure files, through Foldseek | `contexts/structures.json`, written before `contexts/MANIFEST.json` (the file that marks the stage as finished); `MANIFEST.json` gets a `structures` block (digest, number of sequences, fraction masked, engine, masking policy, datasets, conflicts, `dms_labels_opened: false`). It also checks that the weights of the models the run loads are fetched, so a missing or purged weights folder fails in this CPU job minutes after submission instead of in the GPU queue |
| `teacher` | the strings of the training data, if the teacher is SaProt | `structures_digest` in `teacher/COMPLETE.json`; a table made with other strings is recomputed, and `train` refuses it |
| `train` | the strings of the training data, if the student is SaProt | `structures_digest` in `CONFIG.json`, which is part of the configuration digest: a resumable checkpoint made with other strings is not resumed, the unit starts again and the old attempt is kept aside |
| `score` | the strings of the dataset being scored, if the student is SaProt | `structures_digest` in every shard's JSON |
| `evaluate` | no structure | |

The digest is that of the whole `structures.json`, so a change in a dataset a stage does not read also makes a cached teacher table or
a checkpoint count as stale. That costs a recomputation, never a wrong result. `prepare` is the only stage that writes the file, and
running it again on unchanged inputs writes identical strings. No stage but `evaluate` opens a label; a test puts a tripwire on both
benchmarks' label readers and runs every other stage with a SaProt recipe on both datasets.

## Foldseek

`mpd` runs `foldseek structureto3didescriptor -v 0 --threads 1 --chain-name-mode 1 <structure.pdb> <out.tsv>` in a private
temporary folder per structure and reads the amino-acid and 3Di columns. Structures are PDB files with ATOM records; a file with
several chains needs `Assay.structure_chain`, otherwise it is an error. Results are cached per structure file and per Foldseek
binary in `<home>/cache/3di/<engine id>/`, so a second campaign on the same data does not call Foldseek again.

| | |
|---|---|
| pinned release | `10-941cd33` (2025-01-19), the immutable GitHub release asset `foldseek-linux-avx2.tar.gz`, pinned by the sha256 of the archive and of the binary inside it (`src/mpd/data/sources.py`) |
| install | `mpd fetch foldseek --cluster grace` on a login node, into `<home>/tools/foldseek-10-941cd33/`, then `foldseek version` is run and recorded in `FETCHED.json`. On a machine that cannot download, put the archive in reach and pass it: `mpd fetch foldseek --cluster grace --from-file <foldseek-linux-avx2.tar.gz>` (checked against the pinned sha256 first) |
| another binary | `MPD_FOLDSEEK=/path/to/foldseek` (there is no command-line flag; the variable is forwarded to jobs). It is never second-guessed: if it is set and not an executable file there is no fallback. The binary's sha256 is recorded in `structures.json`; one that is not the pin only produces a warning |
| no binary | `MPD_3DI_ENGINE=biotite` uses biotite's NumPy port (`pip install biotite`, 1.6.0 tested). Measured here against the 2026 Foldseek build on all 675 ProteinGym and Venus structure files: identical on 674; in the other (the 2,832-residue BRCA2 model) one residue differs. The engine is recorded in `structures.json`. Foldseek stays the default because it defines 3Di and the recorded results |
| licence | Foldseek is GPL-3.0. `mpd` downloads it from its owners and runs it as a separate program; it is never bundled, committed, put in a container image or a deposit. This is not legal advice |

`fetch_foldseek` verifies the archive's size and sha256 before opening it, extracts to a staging folder, checks the binary's sha256
against the pin before making it executable, and only then runs it. *Tested here* against a fake release served by a local HTTP
server: a wrong archive, a wrong binary (never executed), an unsafe archive member, a download with rate limiting (`Retry-After`
honoured), a binary that does not run, and a local archive (`--from-file`). *Not verified:* a real download of the pinned release and
running it. The build was not allowed to download and execute an external binary.

### Is the pinned release the build the earlier results came from? Not verified

The earlier study ran a rolling "latest" build of 2026-09-09 (`foldseek version` prints
`463739e0014a1549a527de589102cde98f802f37`; sha256 of the binary `56ec365f98227b07b7de4eba97aa1ae5f3cbf70fd9819311fab98eec42ada6ae`).
It is 374 commits newer than release 10. The earlier analysis found no upstream commit in between that touches the 3Di encoder, its
weights or the `structureto3didescriptor` command (the structure-file parser has 17 commits, none seen to change how ATOM records are
read). That is an inference from commit history; **identical output has not been checked**. To check it, run the pinned binary over
the same structures:

```bash
MPD_FOLDSEEK=<home>/tools/foldseek-10-941cd33/foldseek/bin/foldseek \
MPD_TEST_VENUS_HOME=<home with data/venusmuthub-v1 and data/venusmuthub-structures-v1> \
MPD_TEST_PG_HOME=<home with data/proteingym-v1.3> \
python -m pytest -q tests/test_saprot_slow.py
```

or `scripts/maintainers/regress_saprot_venus.py --part structures --foldseek <pinned binary> ...`, which also reports how many
residues differ if the strings do not match. If they differ, either mirror the 2026 build (with the corresponding source, as the
GPL requires for redistribution) or re-baseline the golden strings. The reference values, produced by the 2026 build:

| structure | 3Di (Foldseek upper case) |
|---|---|
| 1PGA chain A (Venus `venus_seq_016be5b116b4`, experimental, 56 residues) | `DKAKEAEAEPPDGDMDIDDDPDLVVVVVVVVVVCVVVPHDADWDADPVRRYIYGHD` |
| AlphaFold DB (Venus `venus_seq_00da05aa2d2c`, 79 residues) | `DPLPPWDKDDWDDDPWKIKIKIAADPVLLVVDDPPDFDDPVVVVVSVDDDDPFWGQDDQPVVHSRITIIIGTDDPDPDD` |
| ESMFold (Venus `venus_seq_b775285baac5`, 44 residues) | `DPPVLVVLLVVLVPPPPDDPVRSVVLVVVCVVPVVCSVVVSVVD` |

Whole-set digests (sha256 of the sorted `id<TAB>string` lines), all reproduced here with the 2026 build:

| set | digest |
|---|---|
| Venus, earlier study's frozen strings (= this code with `MPD_STRUCTURE_LEGACY_MASK=1`) | `2a99932edcc2fda67bce95fb036342efc28580fa6f54869e8c824507beb6a96d` |
| Venus, scale-aware masking (the default) | `a885bf3c093c724109802405cb224a0fa2a60fa8a18bcdf54008e6850b6af7ff` |
| ProteinGym v1.3, 187 sequences keyed by the sha256 of the sequence (`StructureSet.digest()`) | `baea4585ceef32eb1b9b965b8bdc55b0698b3f34ed0359d3c3a8b4258aba6cbf` |

*Foldseek on Grace: not verified.* HPRC's Grace primer (Fall 2025) says "The 940 nodes are based on the Intel Cascade Lake
processor"; Cascade Lake has AVX2, the instruction set the `linux-avx2` build needs. The CPU models of individual partitions (for
example the A40 nodes) are not documented there, and nobody has run Foldseek on Grace. Install it on a file system that is not
mounted `noexec`; `mpd fetch foldseek` and `mpd doctor` run `foldseek version` to find out.

## Venus structures: availability

The structure set for the 489 VenusMutHub proteins (`venusmuthub-structures-v1`: 159 experimental, 232 AlphaFold DB and 98 ESMFold
files with a manifest) is **not published yet**; the Zenodo deposit is in preparation. Until it is, `mpd fetch venus-structures`
only works for someone who already holds the deposit files (`--from-dir <folder>`), and SaProt cannot be run on VenusMutHub by anyone
else. ProteinGym needs nothing unpublished. When the record exists, its id, DOI and md5 go into `src/mpd/data/sources.py`; the three
sha256 values are already recorded there as a comment.

## Licences

| | licence |
|---|---|
| SaProt code and both model cards (`westlake-repl/SaProt_35M_AF2`, `SaProt_650M_AF2`) | MIT (checked on the GitHub repository and the model cards on 2026-11-09) |
| Foldseek | GPL-3.0; fetched from its owners and run as a separate program, never bundled |
| Venus structure set, once published | CC BY 4.0, with wwPDB (CC0), AlphaFold DB (CC BY 4.0) and ESMFold (MIT) content |

## Regression against the earlier study

Maintainer tools (not part of the test suite). All numbers were produced for this document on a CPU login node: torch 2.10.0,
transformers 4.51.0, bf16, 4 threads, SaProt-650M weights from the earlier study's converted `model.safetensors` (the tensors equal
the pinned `pytorch_model.bin` according to the earlier analysis; the pinned `.bin` of 650M was not loaded here), the
Foldseek build of 2026-09-09, and the earlier study's stored outputs of the same positions.

**Venus structure strings** (`scripts/maintainers/regress_saprot_venus.py --part structures`, 2 s):

| | strings equal to the earlier study's | residues masked |
|---|---|---|
| earlier study (frozen) | 489 of 489 | 37,826 of 135,060 (28.0%) |
| this code, `MPD_STRUCTURE_LEGACY_MASK=1` | **489 of 489** (232 AlphaFold DB, 159 experimental, 98 ESMFold) | 37,826 |
| this code, default | 391 of 489 (232 AlphaFold DB, 159 experimental; 0 of 98 ESMFold) | 14,293 (10.6%) |

So the 489 proteins are re-derived exactly, and the default differs from the earlier inputs only for the ESMFold proteins
(23,533 of their 27,455 residues change from `#` to a state).

**SaProt-650M on CPU against the recorded Venus base outputs** (`--part scores`, a subset of 56 proteins of at most 400 residues,
899 positions, 17,980 values, 103 assays; 102 s of compute; the inputs are the earlier study's, i.e. the legacy strings):

| | recorded | here | difference |
|---|---|---|---|
| largest absolute difference of any value | | | 0.250 (bf16 logits of a GPU against bf16 on a CPU; mean 0.024) |
| per-position Spearman of the 19 mutants | | mean 0.99903, minimum 0.9842 | |
| 28 assays with at least 30 variants, macro Spearman | 0.251247 | 0.253805 | +0.0026 (largest single assay 0.0138) |
| all 103 assays | 0.335070 | 0.336196 | +0.0011 |
| AlphaFold DB (40) / experimental (27) / ESMFold, legacy input (36) | 0.4645 / 0.2557 / 0.2508 | 0.4702 / 0.2512 / 0.2510 | +0.0057 / -0.0044 / +0.0002 |
| protein of 1,130 residues, positions 0, 298, 395 (1,024-token window) | | | 0.031, 0.094, 0.094 |

Differences of this size are bf16 rounding: the 6-9 variant assays flip ranks, which is why the gate uses the assays with at least
30 variants. The gates (mean per-position rho at least 0.995, largest difference at most 0.25, macro difference at most 3e-3) all pass; the
largest difference sits exactly on its limit.

**ProteinGym's published Spearman** (`scripts/maintainers/regress_saprot_proteingym.py`, official v1.3 structures; SaProt-35M on 14 assays in
86 s, SaProt-650M on six in 361 s). On six assays this code reproduces the values ProteinGym publishes for both models to within 0.002.
SaProt-35M (largest difference 0.0018):
`RS15_GEOSE_Tsuboyama_2023_1A32` 0.6477 (published 0.6460), `CCDB_ECOLI_Adkar_2012` 0.1948 (0.1940),
`KCNE1_HUMAN_Muhammad_2023_function` 0.2389 (0.2380), `NUD15_HUMAN_Suiter_2020` 0.5635 (0.5640), `P84126_THETH_Chan_2017` 0.6082
(0.6100), `MET_HUMAN_Estevam_2023` 0.5290 (0.5290). SaProt-650M (largest difference 0.0008): 0.4327 (0.4330), 0.4372 (0.4380), 0.6413
(0.6420), 0.6788 (0.6790), 0.6212 (0.6220), 0.5749 (0.5750), in the same order. This checks the `pdb_range` handling, the masking, the
masked token and the readout end to end. **On eight other assays the SaProt-35M value does not reproduce**, and the cause is not
established:

| assay | here | published | difference |
|---|---|---|---|
| `TCRG1_MOUSE_Tsuboyama_2023_1E0L` | 0.7243 | 0.7740 | -0.050 |
| `SAV1_MOUSE_Tsuboyama_2023_2YSB` | 0.6205 | 0.6070 | +0.014 |
| `PITX2_HUMAN_Tsuboyama_2023_2L7M` | 0.6580 | 0.5930 | +0.065 |
| `NUSG_MYCTU_Tsuboyama_2023_2MI6` | 0.6146 | 0.5880 | +0.027 |
| `RL20_AQUAE_Tsuboyama_2023_1GYZ` | 0.4976 | 0.7100 | -0.212 |
| `SRBS1_HUMAN_Tsuboyama_2023_2O2W` | 0.6595 | 0.6890 | -0.030 |
| `CATR_CHLRE_Tsuboyama_2023_2AMI` | 0.4194 | 0.6860 | -0.267 |
| `Q6WV12_9MAXI_Somermeyer_2022` | 0.0759 | 0.0310 | +0.045 |

A separate implementation written for the earlier analysis (not this code) gave the same numbers to four decimals on all 14 assays, so
this is not a bug in one code path.
These are short Tsuboyama domains (and one fluorescent protein) with 0 to 8 masked residues; a likely reason is that the published
values used different structures than the v1.3 package, but that is a guess. Do not treat ProteinGym's published SaProt values as a gate
for a whole run. The earlier analysis found the same eight disagreeing for SaProt-650M too (differences of up to 0.32); the 650M
values for these eight were not re-run here.

**Earlier recorded ProteinGym SaProt numbers** (macro 0.4635) used GenBio's structures, which differ from the official ones (3Di letters
agree on 84.6% of positions on average, per the earlier analysis); they are a reference to compare in spirit, never to reproduce.
SaProt-35M has no outputs recorded anywhere in the earlier study.

To repeat all of this (about 4 minutes on a loaded login node for the Venus run):

```bash
PYTHONPATH=src python scripts/maintainers/regress_saprot_venus.py --home <home with the Venus data and structures> \
    --plan <earlier PLAN.json> --foldseek <foldseek binary> --predictions <earlier predictions/saprot_base> --model <SaProt-650M folder>
PYTHONPATH=src python scripts/maintainers/regress_saprot_proteingym.py --home <home with data/proteingym-v1.3> \
    --foldseek <foldseek binary> --model <SaProt-35M folder> --table <ProteinGym DMS-level Spearman CSV>
```

## Using SaProt

On a login node (`--cluster grace`; the other profile works the same way):

```bash
bash scripts/setup_env.sh --cluster grace
mpd fetch proteingym --cluster grace                 # the data and the AF2 structures SaProt reads (62.4 MB, 19.2 MB of them structures)
mpd fetch foldseek --cluster grace
mpd fetch model saprot-650m esm2-3b --cluster grace
mpd doctor --cluster grace

# SaProt-650M student distilled from ESM2-3B, trained on ProteinGym, evaluated on ProteinGym
mpd plan --cluster grace --account <account> --teacher esm2-3b --student saprot-650m \
    --train-data proteingym --method score-all --eval-data proteingym

# SaProt as the teacher of an ESM2 student
mpd plan --cluster grace --account <account> --teacher saprot-650m --student esm2-650m \
    --train-data proteingym --method score-all --eval-data proteingym
```

`mpd submit` takes the same flags. For VenusMutHub add `mpd fetch venusmuthub --cluster grace` and, because the structure set is not
published yet, `mpd fetch venus-structures --cluster grace --from-dir <folder with the deposit files>`. `mpd doctor` and `mpd plan`
report a missing Foldseek or missing structure data with the command that fixes it.

### A teacher and a student from different families

SaProt's 20 values at a position are, for each amino acid, a log-sum over the 21 structure states of its tokens; ESM2's are raw
head outputs. The two sit on different offsets, and the offset can change from position to position. The `score-*` methods train on differences of log-softmax scores, in which any constant per position cancels, so they do
not depend on the offset. The `direct-*` methods apply a Huber loss to the raw values (plus a KL term, which is offset-free), so a
student trained on a teacher of the other family inherits the teacher's offset. The earlier study trained SaProt with the same
direct objective on raw AIDO log-marginals (from the earlier analysis, not re-checked here). `tests/test_saprot_offsets.py` pins the property on random
numbers: shifting the teacher's values by a constant per position leaves the score loss unchanged and changes the direct loss.

**This was reasoned and unit-tested, not studied.** Nobody has measured what the offset does to a real SaProt and ESM2 pair, or which
method is better for one. Pairs inside one family (ESM2 to ESM2, SaProt to SaProt) have no such offset.

The memory estimates for SaProt (`act_gb`) are placeholders equal to the same-size ESM2 model and have not been measured; the
walltimes `mpd plan` prints for SaProt stages are the ESM2 ones extrapolated by size and are marked `est`. The `prepare` stage asks
for a fixed 4 hours on 8 CPUs (a CPU job, no GPU service units); Foldseek itself needed about 15 seconds for the 675 benchmark
structures on one core in the earlier analysis (not measured on Grace).

| variable | meaning |
|---|---|
| `MPD_FOLDSEEK` | path of the Foldseek binary (overrides `<home>/tools`) |
| `MPD_3DI_ENGINE` | `foldseek` (default) or `biotite` |
| `MPD_MODEL_SAPROT_35M`, `MPD_MODEL_SAPROT_650M` | a local folder of weights instead of the fetched copy |
| `MPD_STRUCTURE_LEGACY_MASK` | `1` reproduces the earlier study's ESMFold masking; for regression only |

## What is and is not verified

| | status |
|---|---|
| vocabulary, tokens, focal masking, windows, log-marginal, dropout 0, LoRA, adapters | tested on CPU here (tiny random model; real SaProt-650M and 35M weights in the regression) |
| masking per structure source, alignment cases, Foldseek wrapper contract, 3Di cache, structure sets, benchmark ranges, preflight messages | tested on CPU here |
| the Venus and ProteinGym structure strings against the recorded digests | tested here with the 2026-09-09 Foldseek build (opt-in `tests/test_saprot_slow.py`) |
| SaProt-650M and 35M numbers against recorded and published values | measured here on CPU (section above); eight published ProteinGym values are not reproduced |
| `fetch_foldseek` with hash checks, exec bit, rate limiting | tested here against a fake release; a real download and run: not verified |
| the pinned release 10 gives the same 3Di as the 2026 build | not verified |
| Foldseek on Grace | not verified |
| the stage hooks: strings built by `prepare`, read by `teacher`, `train` and `score` with each dataset's own structure, part of the teacher-table and checkpoint digests, no label opened before `evaluate`; the weights check at the start of `prepare` | tested on CPU here with tiny random SaProt and ESM2 models and a fake Foldseek (`tests/test_saprot_pipeline.py`) |
| the messages for a missing Foldseek, ProteinGym structures or Venus structure set, through `mpd plan` and `mpd submit` | tested on CPU here through the real command line (`tests/test_saprot_cli.py`) |
| the opt-in real-data tests (`tests/test_saprot_slow.py`, six tests) | passed on 2026-11-09 with the 2026-09-09 Foldseek build, the earlier study's data and the SaProt-35M weights: 6 passed in 58 s |
| the `prepare` stage itself on the real files (run by hand once, CPU, not through Slurm, with the 2026-09-09 Foldseek build) | ProteinGym alone (direct-all, 62,671 training and 37,753 evaluation contexts, 186 structure files): 20 s, and its `structures.json` digest is the recorded ProteinGym digest `baea4585...`. ProteinGym and VenusMutHub together (score-all, 675 structure files run through Foldseek for the first time): 23 s, 666 sequences, 10 with a different structure in the second dataset, 17.6% of residues masked. Times are from a loaded login node |
| the planner's walltime and memory for SaProt jobs | estimates extrapolated from ESM2, not measured |
| a GPU run, bf16 on a GPU, two-GPU training with SaProt | not verified |
