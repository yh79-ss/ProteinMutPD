# ProteinMutPD

Distill protein mutation-effect predictors into smaller students, and evaluate them on
[ProteinGym](https://proteingym.org) and [VenusMutHub](https://huggingface.co/datasets/AI4Protein/VenusMutHub).
You choose the teacher, the student, the training data, the evaluation data and the method on the command line;
`mpd` plans the work, checks that the cluster has what the jobs need, and submits it as dependent GPU Slurm jobs.
A student is a small LoRA adapter on a pretrained protein language model (ESM2 or SaProt). Profiles for TAMU HPRC Grace
and Vision are included; [another cluster](docs/CLUSTERS.md) needs one YAML file.

> **Status: alpha. Nothing has run on a Grace GPU node.** The planner, the script writer, the submit and recovery logic are tested
> on CPU against fake Slurm tools, and the scripts written for the Vision cluster are accepted by its `sbatch --test-only`.
> No job has run through Slurm on any GPU. [Status](#status) says what is and is not verified.

## Methods

`--method` is an objective and a training scope:

| | all-assay | one-assay |
|---|---|---|
| **direct logits** (Huber + KL on amino-acid logits at masked positions) | `direct-all` | `direct-one` |
| **score** (Huber on the mutant-minus-wild-type score, dual temperature) | `score-all` | `score-one` |

One-assay methods take `--assay <id> [...]` (`mpd assays` lists the ids) and train one student per assay, so they cost one
training per assay: use them for a few assays. When teacher and student are from different families (SaProt with ESM2, either
way), prefer `score-*`: `direct-*` copies the teacher's raw values, which carry a per-position offset that differs between
families, and `score-*` cancels it. That is reasoned and unit-tested on random numbers, never measured on a real pair
([docs/SAPROT.md](docs/SAPROT.md#a-teacher-and-a-student-from-different-families)).

## Quick start on Grace

On a login node, with a Grace account that has an allocation ([Prerequisites](#prerequisites)). Change the three marked lines, then paste
the block. `setup_env.sh --yes` skips its "Continue? [y/N]" question, which would otherwise swallow the next pasted line; `mpd submit`
asks one too, so paste the lines after it only once you have answered.

```bash
git clone https://github.com/yh79-ss/ProteinMutPD && cd ProteinMutPD
bash scripts/setup_env.sh --cluster grace --yes   # builds $SCRATCH/mpd/env: 3 to 4 GB to download, 28,000 files, 7.7 GB
source $SCRATCH/mpd/env/bin/activate              # sets MPD_HOME and MPD_CLUSTER; do this in every new shell

export MPD_ACCOUNT=<ACCOUNT>                      # change 1 of 3, account: the number `myproject` prints
TEACHER=esm2-3b STUDENT=esm2-650m                 # change 2 of 3, models: `mpd models` lists them
RECIPE="--train-data proteingym --method score-all --eval-data proteingym venusmuthub"   # change 3 of 3, recipe

mpd fetch proteingym                              # 62 MB download, 1.2 GB on disk
mpd fetch venusmuthub                             # 796 small files, 6 MB
mpd fetch model $TEACHER $STUDENT                 # 13.98 GB for this pair; resumable, so run it again if it stops at the login-node limit
mpd doctor                                        # environment, data, weights, quota, Slurm tools
mpd plan   --teacher $TEACHER --student $STUDENT $RECIPE      # stages, resources, estimated cost; writes nothing
mpd submit --teacher $TEACHER --student $STUDENT $RECIPE      # shows the plan, asks "Submit N job(s), up to X SU? [y/N]"

mpd status                                        # campaigns; with a campaign id, the state of its jobs
mpd logs <campaign> train --err                   # a stage's log; without a stage it lists the logs
mpd results <campaign>                            # after the evaluate stage; scratch is not backed up, copy what you keep
mpd cancel <campaign>
```

Compute nodes have no internet, so the weights and data are fetched first and the jobs run offline. `mpd plan` prints each stage,
what it asks Slurm for, its walltime and an upper bound in service units, and says what is still missing and which command fixes it.
`mpd submit` shows the same, asks before queueing, and refuses (exit status 2) while something the jobs need is missing; add
`--dry-run` to write and check the scripts without queueing anything, and `--yes` in a script. Plan and dry run only warn.

First time on Grace? Run the [cheap first run](docs/GRACE.md#your-first-run-cheap) (133 SU by the planner's upper bound) before an
expensive campaign. Something failed? [docs/GRACE.md](docs/GRACE.md#when-a-job-does-not-finish) has the recovery recipes and
[docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) maps messages to fixes.

## Prerequisites

- **A Grace account and an allocation.** `myproject` lists your accounts and balances; the default Basic allocation is 20,000 SU. Off
  campus, connect to the TAMU VPN first. [docs/GRACE.md](docs/GRACE.md) covers accounts, quotas, partitions and what a login node may do.
- **Scratch space, not `$HOME`.** `$HOME` is 10 GB and 10,000 files; `$SCRATCH` is 1 TB and 250,000 files and is not backed up.
  `mpd` refuses a working root under `/home`.
- **A Python module, Python 3.11 or newer.** `setup_env.sh` loads `Anaconda3/2024.02-1` by default (the module earlier Grace jobs
  used; which Python it provides now is not verified). HPRC's `GCCcore/13.2.0 Python/3.11.5` is an alternative that was never run
  ([docs/GRACE.md](docs/GRACE.md#the-environment)).
- **Login-node manners.** At most 8 cores across all login nodes and no process longer than an hour. Everything above runs there. The
  `mpd` commands never import torch (only `mpd doctor --deep` does).

## Space and service-unit budget

Disk, for the example above (the environment size is measured for the pinned package versions in a reference environment; the
download size is an estimate; the others are the pinned manifests and a real fetch):

| what | download | on disk | files |
|---|---|---|---|
| Python environment (`$MPD_HOME/env`): torch 2.10 with CUDA 12.8 libraries, transformers, peft | 3 to 4 GB, cached in `cache/pip` | 7.7 GB | about 28,000 |
| ESM2-3B and ESM2-650M weights (all five ESM2 models: 14.74 GB; all seven models: 17.48 GB) | 13.98 GB | 13.98 GB | 12 |
| ProteinGym v1.3 | 62.4 MB | 1.2 GB | 421 |
| VenusMutHub | 6.3 MB | 6.3 MB | 797 |
| one campaign (`runs/<id>`) | | well under 100 MB (an estimate; the tiny CPU test campaign is 79 KB) | tens to hundreds |

That is about 23 GB (about 27 GB while the pip cache is kept) and 29,600 files of the 1 TB and 250,000 files of `$SCRATCH`. A `*-one`
campaign over many assays writes a few thousand small files.

Service units: the plan above (`esm2-3b` to `esm2-650m`, `score-all`, evaluated on both benchmarks) prints an **upper bound of
1,806 SU, 9% of a 20,000 SU Basic allocation** (simulated: the Grace profile on a machine that is not Grace; the planner's estimate, not a
measurement; the allocation is from HPRC's *Introduction to HPRC Computing Resources*, 2026-01-23, slide 45, not verified for
November 2026). `score-one` over all 217 ProteinGym assays plans to 184,700 SU. sbatch's own "Requested SUs" line counts CPUs only,
so budget with the plan's figure. A submit above `--max-su` (default 10,000) is refused. Details and worked examples:
[docs/GRACE.md](docs/GRACE.md#what-it-costs).

## What happens after submit

`mpd submit` writes `<home>/runs/<campaign id>/` and queues one job per stage, each waiting (`afterok`) for the stages it needs.
The stage table of the plan above (simulated; the two `campaign id:` lines under it in the `mpd plan` output say what the campaign id depends on):

```text
stage          tasks  gpu       cpus   mem  walltime          SU (max)  after
prepare            1  -            8   32G  04:00:00                32  -
teacher            1  1xa100      24  180G  03:00:00 est           291  prepare
train              1  1xa100      24  180G  10:15:00 est           994  teacher
score-base         5  1xa100      24  180G  00:30:00 est           242  prepare
score              5  1xa100      24  180G  00:30:00 est           242  train
evaluate           1  -            4   16G  01:00:00                 4  score-base, score
```

`prepare` (CPU) builds the masked-position tables, the structure strings for SaProt, and checks that the weights are on disk; `teacher`,
`train`, `score-base` (the frozen student) and `score` (the trained one) run on GPUs; `evaluate` (CPU) is the only stage that reads labels.
`prepare` asks for 04:00:00, 8 CPUs and 32 GB, so its 32 SU is an upper bound; on real data it took 20 to 23 seconds on a loaded login
node of another cluster, not on Grace. Training saves a resumable state every 200 updates.

`mpd status <campaign>` prints one line per stage (Slurm's state, run time, why a job waits, job id, exit code, tasks finished), then what
failed with the exact command that continues it, why jobs wait, and the next step. Results land in `evaluate/` and
the adapters in `train/<unit>/checkpoint_<update>/`; [docs/RESULTS.md](docs/RESULTS.md) lists every file and metric. If a stage
times out or lands on a bad node, repeat your `mpd submit` command with the changed option and `--force`: finished stages are kept
([recipes](docs/GRACE.md#when-a-job-does-not-finish)). Changing `--gpu`, `--train-gpus`, `--eval-data` or `--score-shards` starts a new
campaign that trains again; walltimes and `--sbatch-arg` do not.

## Models, data and licences

| | what | in this release |
|---|---|---|
| ESM2, 8M to 3B | teacher or student; MIT | available |
| SaProt, 35M and 650M | teacher or student; needs Foldseek structure strings; MIT | available, tested on CPU only; never run on a GPU, on Grace or through Slurm; VenusMutHub needs a structure set that is not published yet ([docs/SAPROT.md](docs/SAPROT.md)) |
| AIDO.Protein-RAG-16B | GenBio AI Community License | **not included** (below) |

Data: ProteinGym v1.3 (official Zenodo record [10.5281/zenodo.15293562](https://doi.org/10.5281/zenodo.15293562), MIT) and VenusMutHub
(Hugging Face `AI4Protein/VenusMutHub` at revision `08a27d07`, MIT), each file pinned by checksum; the structure set that SaProt needs for
VenusMutHub is a Zenodo deposit in preparation ([docs/ZENODO.md](docs/ZENODO.md)). Weights are pinned to one Hugging Face commit and
checked by sha256 ([docs/MODELS.md](docs/MODELS.md)). SaProt also needs Foldseek: `mpd fetch foldseek` installs a pinned release and
`mpd plan` names whatever is missing. Details and caveats: [docs/DATA.md](docs/DATA.md).

**AIDO is not part of this release.** `genbio-ai/GB.Protein-RAG-16B-proteingym-dms-zeroshot` appears in `mpd models` only so that `mpd`
can tell you why it is not offered: `mpd` cannot plan, download or run it, and this repository contains no AIDO weights, code or outputs.
AIDO is under the GenBio AI Community License: non-commercial use only, and clause 1.2.2 does not allow using its outputs to train other,
similar models or for distillation toward that end, except for training the GenBio Materials themselves and their derivative works.
`mpd` therefore refuses every run in which AIDO is the teacher of a non-AIDO student, and has no switch to override that; it would need
written permission from GenBio and a change to the code. The AIDO-teacher results of the maintainers' earlier study cannot be reproduced
with `mpd` for this reason. If you want to use AIDO yourself, read [docs/LICENSES.md](docs/LICENSES.md) and the licence text first. That
page is a plain-language summary, not legal advice.

The ProteinMutPD code is MIT-licensed ([LICENSE](LICENSE)). Data and models keep their own licences, listed in
[docs/LICENSES.md](docs/LICENSES.md).

## Status

Tags: **tested on CPU** (this repository's tests, with fake Slurm tools and tiny random models), **Vision** (accepted by `sbatch
--test-only` on the Vision cluster, Slurm 25.05.6, nothing queued), **not verified** (nobody has run it).

| | status |
|---|---|
| Planner, resource model, script writer, `submit`, `status`, `logs`, `cancel`, `results`, `doctor`, `fetch` | tested on CPU against fake `sbatch`, `squeue`, `sacct`, `scancel`, `module`, `nvidia-smi` and a local Hugging Face stand-in |
| Recovery: a TIMEOUT, a bad node, an interrupted submit, a cancel, a deleted stage marker, each by repeating the command (with `--force`, except after a cancel) | tested on CPU against the fake Slurm; never against a real queue |
| Scripts for the Vision profile (24 plans, 132 scripts, 12 distinct `#SBATCH` header shapes) | every shape accepted by `sbatch --test-only` on Vision; this says nothing about Grace (A100, Slurm version unknown) |
| Scripts for the Grace profile | rendered and `bash -n` clean; **never given to a Grace Slurm** |
| The job scripts that `mpd submit` writes, run end to end with `bash` on a CPU: `tests/test_rehearsal.py` | tested on CPU: all four methods on the Grace and Vision profiles (every array task, two-process training on Vision), an ESM2 pair and a SaProt student, tiny random models, a fake `module`, `nvidia-smi` and `SLURM_*` variables; a training job stopped by SIGTERM and started again gave the same adapter. **Not run on Grace, on a GPU or by a real scheduler**: module names, drivers, NCCL, memory and queue are not covered |
| `scripts/setup_env.sh` | run for real once on the Vision login node without torch; the torch install and the Grace defaults (module, home) are **not verified** |
| Any job on a Grace GPU node (first `import torch`, memory of a real student, every walltime estimate) | **not verified**: all walltimes and GPU memory figures are labelled estimates |
| Multi-GPU (DDP) training | tested on CPU: two processes equal one; **never run with NCCL on a GPU**; Grace defaults to one GPU |
| ESM2 stages for all four methods; resume after a walltime kill | tested on CPU with tiny random models |
| Base-model numbers of the earlier study | its stored ESM2-650M outputs, replayed through these loaders and evaluator, reproduce its macro Spearman (0.2095 VenusMutHub, 0.4280 ProteinGym; [docs/DATA.md](docs/DATA.md)). **No distilled number has been reproduced on a GPU** |
| SaProt (stages, `prepare` structure strings, weights, adapters) | tested on CPU with tiny random SaProt and ESM2 models and a fake Foldseek, and `prepare` once by hand on the real files; the pinned Foldseek release 10-941cd33 was **never executed**; bf16, GPU memory and walltime for SaProt are **estimates** ([docs/SAPROT.md](docs/SAPROT.md)) |
| SaProt on VenusMutHub for other people | waits for the Venus structure deposit, which is in preparation; the pins are ready, nothing is published |
| A teacher and a student from different families with `direct-*` | reasoned and unit-tested on random numbers; **not studied** on a real pair |
| AIDO | not in this release ([docs/LICENSES.md](docs/LICENSES.md)) |
| GitHub Actions workflow (ruff; the test suite and the CPU rehearsal on Python 3.11 with CPU torch; a wheel installed without torch that runs `plan`, `submit --dry-run`, `status` and `doctor`; the documentation tests) | written; each job was run step by step on a Linux machine, in fresh virtual environments with a scrubbed environment and no Slurm client on PATH, and the YAML was checked against GitHub's published schemas; **it has not run on GitHub**, so runner timing, the pip cache and the PyTorch CPU index are untested |
| `pip install` and `pytest` | run on CPU on 2026-11-09 from a copy of the files a clone holds: with the pinned set on Python 3.11 (the CI steps, one by one), 3.12 (Ubuntu's own build) and 3.13; with torch 2.7.0 (the floor) and with the newest versions the ranges allow (torch 2.15, transformers 4.57, peft 0.21); in a virtual environment without torch (the tests that need torch skip). Python 3.10 is not supported |
| Which readout is "primary" | `delta` for direct runs and `dual` for score runs is an **unconfirmed assumption of the maintainer**; both are always reported |

## Use the student

A trained student is a PEFT LoRA adapter (`adapter_config.json` and `adapter_model.safetensors`, rank 8 on the attention projections)
in `<campaign>/train/<unit>/checkpoint_<update>/`; `mpd results <campaign>` prints the path. Load it on the base model from `mpd fetch model`
(run on CPU against a tiny model; it reproduces `mpd`'s own logits exactly in float32):

```python
import torch
from peft import PeftModel
from transformers import AutoModelForMaskedLM, AutoTokenizer

base = "/scratch/user/<netid>/mpd/models/esm2-650m"        # the folder `mpd fetch model` made
tok = AutoTokenizer.from_pretrained(base)
model = AutoModelForMaskedLM.from_pretrained(base, torch_dtype=torch.float32, attn_implementation="eager")
model = PeftModel.from_pretrained(model, "<campaign>/train/all/checkpoint_02103").eval()   # bfloat16 on a GPU, as the jobs do
```

[docs/RESULTS.md](docs/RESULTS.md) scores a variant the way `mpd` does and explains the result files and metrics.

## Other clusters

`mpd plan --cluster ./mycluster.yaml ...` reads a profile you write; the field reference, a commented template and how to check a new
profile are in [docs/CLUSTERS.md](docs/CLUSTERS.md). To add a model, see [docs/ADDING_A_MODEL.md](docs/ADDING_A_MODEL.md).

## Development

```bash
python3 -m venv .venv && . .venv/bin/activate
# a CPU build of torch is enough for the tests; the versions are pinned in constraints/
python -m pip install -c constraints/grace.txt --extra-index-url https://download.pytorch.org/whl/cpu torch
python -m pip install -c constraints/grace.txt -c constraints/ci.txt -e '.[train,structure,test,dev]'
python -m pytest -q -m "not slow and not gpu and not rehearsal"
python -m pytest -q -m rehearsal
ruff check src tests scripts
```

The first `pytest` line is the fast suite; the second runs the rest of the CPU rehearsal of the rendered job scripts (the other recipes, the Vision profile with two-process training, chunked arrays and a SaProt student; three to six minutes on a busy login node).
The suite needs no GPU, no Slurm and no network. It passed on 2026-11-09 with the pinned versions on Python 3.11, 3.12 and 3.13, with torch 2.7.0 and with the newest compatible versions. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Documentation

[GRACE](docs/GRACE.md) | [TROUBLESHOOTING](docs/TROUBLESHOOTING.md) | [RESULTS](docs/RESULTS.md) | [MODELS](docs/MODELS.md) |
[SAPROT](docs/SAPROT.md) | [DATA](docs/DATA.md) | [CLUSTERS](docs/CLUSTERS.md) | [ADDING_A_MODEL](docs/ADDING_A_MODEL.md) |
[LICENSES](docs/LICENSES.md) | [ZENODO](docs/ZENODO.md)

## Citation, contributing, support

Cite the software with [CITATION.cff](CITATION.cff) (its author list is still a placeholder); the data and models have their own citations ([docs/LICENSES.md](docs/LICENSES.md)).
Contributions: [CONTRIBUTING.md](CONTRIBUTING.md); changes: [CHANGELOG.md](CHANGELOG.md); security: [SECURITY.md](SECURITY.md). For an
issue, paste the output of `mpd doctor --report`, which removes account numbers, user names and home paths.
