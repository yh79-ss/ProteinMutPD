# ProteinMutPD

Distill protein mutation-effect predictors into smaller students, and evaluate them on
[ProteinGym](https://proteingym.org) and [VenusMutHub](https://huggingface.co/datasets/AI4Protein/VenusMutHub).
You choose the teacher, the student, the training data, the evaluation data and the method on the command line;
`mpd` plans the work and submits it as GPU Slurm jobs that follow the cluster's policy (TAMU Grace and Vision
profiles are included).

> **Status: alpha.** The planner, Slurm submission and data download work and are tested. The training, scoring
> and evaluation stages that the submitted jobs call are not written yet, so `mpd submit` cannot produce
> results today. See [Status](#status).

## Methods

`--method` is an objective and a training scope:

| | all-assay | one-assay |
|---|---|---|
| **direct logits** (Huber + KL on amino-acid logits at masked positions) | `direct-all` | `direct-one` |
| **score** (Huber on the mutant-minus-wild-type score, dual temperature) | `score-all` | `score-one` |

One-assay methods take `--assay <id> [...]` and train one student per assay.

## Use

```bash
pip install -e '.[test]'              # login node: planning and downloads need no GPU or torch
mpd models                            # teachers and students
mpd fetch proteingym --cluster grace  # official ProteinGym v1.3 (62 MB), checksum-verified

mpd plan --cluster grace --account <12-digit project> \
    --teacher esm2-3b --student esm2-650m \
    --train-data proteingym --method score-all --eval-data proteingym venusmuthub
```

`mpd plan` prints each stage, its GPU/CPU/memory request, walltime and an upper bound on the service units. `mpd submit`
writes the scripts and submits them with dependencies (add `--dry-run` to write without submitting).

On Grace, GPU jobs ask for a proportional share of a node (for example 24 cores and 180 GB per A100) and the
global batch is kept constant by adjusting gradient accumulation.

## Data

`mpd fetch <source>` downloads pinned files and refuses anything that does not match the recorded checksum.

| source | what | from |
|---|---|---|
| `proteingym` | ProteinGym v1.3 substitution assays, reference table, AF2 structures | official Zenodo record [10.5281/zenodo.15293562](https://doi.org/10.5281/zenodo.15293562) (MIT) |
| `venus-structures` | one structure for each of the 489 VenusMutHub proteins (CC BY 4.0) | Zenodo deposit, in preparation ([docs/ZENODO.md](docs/ZENODO.md)) |

Sequence-only students (ESM2) do not need the Venus structures. Details and caveats are in
[docs/DATA.md](docs/DATA.md).

## Models and licences

Students: ESM2, SaProt. Teachers: ESM2, SaProt. Not yet available: the AIDO models.

The AIDO.Protein-RAG teacher is covered by the GenBio AI Community License, whose clause 1.2.2 bars using model
outputs for distillation except into GenBio's own models and their derivative works. `mpd` therefore blocks AIDO
teachers for non-AIDO students and ships no AIDO outputs. Read the licence before you rely on any AIDO use.

The ProteinMutPD code is MIT-licensed ([LICENSE](LICENSE)). Data and models keep their own licences.

## Status

| | |
|---|---|
| Cluster profiles, planner, Slurm submission, `mpd fetch`, pinned data | done, tested |
| ProteinGym / VenusMutHub loaders, label firewall | next |
| Zero-shot scorer and evaluator (reproduce the base-model Spearman values) | next |
| ESM2 trainer, then SaProt | planned |
| AIDO student | planned, after the licence question is settled for each use |

Nothing has been run on Grace yet. Per-user job limits, scratch purge rules, `gpu-a40` access and the CUDA module
names are unverified and listed in `src/mpd/clusters/grace.yaml`.

## Development

```bash
PYTHONPATH=src python -m pytest -q
```
