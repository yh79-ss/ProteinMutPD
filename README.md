# ProteinMutPD

Distill protein mutation-effect predictors into smaller students, and evaluate them on
[ProteinGym](https://proteingym.org) and [VenusMutHub](https://huggingface.co/datasets/AI4Protein/VenusMutHub).
You choose the teacher, the student, the training data, the evaluation data and the method on the command line;
`mpd` plans the work, checks that the cluster has what the jobs need, and submits it as dependent GPU Slurm jobs.
A student is a small LoRA adapter on a pretrained protein language model (ESM2 today). Profiles for TAMU HPRC Grace
and Vision are included; [another cluster](docs/CLUSTERS.md) needs one YAML file.

> **Status: alpha. Nothing has run on a Grace GPU node.** The planner, the script writer and the submit logic are tested on
> CPU against fake Slurm tools, and the scripts written for the Vision cluster are accepted by its `sbatch --test-only`.
> No job has run through Slurm on any GPU. [Status](#status) says what is and is not verified.

## Methods

`--method` is an objective and a training scope:

| | all-assay | one-assay |
|---|---|---|
| **direct logits** (Huber + KL on amino-acid logits at masked positions) | `direct-all` | `direct-one` |
| **score** (Huber on the mutant-minus-wild-type score, dual temperature) | `score-all` | `score-one` |

One-assay methods take `--assay <id> [...]` (`mpd assays` lists the ids) and train one student per assay, so they cost one
training per assay: use them for a few assays.

## Quick start on Grace

On a login node, with a Grace account that has an allocation ([Prerequisites](#prerequisites)). Every step is a command you can copy;
`<ACCOUNT>` is the number `myproject` prints.

```bash
git clone https://github.com/yh79-ss/ProteinMutPD && cd ProteinMutPD
bash scripts/setup_env.sh --cluster grace        # builds $SCRATCH/mpd/env: about 3 GB to download, 28,000 files, 7.7 GB
source $SCRATCH/mpd/env/bin/activate             # sets MPD_HOME and MPD_CLUSTER; do this in every new shell
mpd fetch proteingym                             # 62 MB download, 1.2 GB on disk
mpd fetch venusmuthub                            # 796 small files, 6 MB
mpd fetch model esm2-3b esm2-650m                # 13.98 GB, resumable; run it again if it stops at the login-node time limit
mpd doctor                                       # what a run needs: environment, data, weights, quota, Slurm tools
mpd plan   --account <ACCOUNT> --teacher esm2-3b --student esm2-650m --train-data proteingym --method score-all --eval-data proteingym venusmuthub
mpd submit --account <ACCOUNT> --teacher esm2-3b --student esm2-650m --train-data proteingym --method score-all --eval-data proteingym venusmuthub
mpd status                                       # lists campaigns; add a campaign id to see its jobs
mpd logs <campaign> train --err                  # a stage's log (.err); without a stage, lists the logs
mpd results <campaign>                           # after the evaluate stage; scratch is not backed up, copy what you keep
mpd cancel <campaign>
```

Compute nodes have no internet, so the weights and data are fetched first and the jobs run offline. `mpd plan` prints each stage,
what it asks Slurm for, its walltime and an upper bound in service units, and says what is still missing and which command fixes it.
`mpd submit` shows the same, asks before queueing, and refuses (exit status 2) while something the jobs need is missing; add
`--dry-run` to write and check the scripts without queueing anything, and `--yes` in a script. Plan and dry-run only warn.

First time on Grace? Run the [cheap first run](docs/GRACE.md#your-first-run-cheap) (about 133 SU by the planner's upper bound) before
an expensive campaign. Something failed? [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) maps messages to fixes.

## Prerequisites

- **A Grace account and an allocation.** `myproject` lists your accounts and balances; the default Basic allocation is 20,000 SU. Off
  campus, connect to the TAMU VPN first. [docs/GRACE.md](docs/GRACE.md) covers accounts, quotas, partitions and what a login node may do.
- **Scratch space, not `$HOME`.** `$HOME` is 10 GB and 10,000 files; `$SCRATCH` is 1 TB and 250,000 files and is not backed up.
  `mpd` refuses a working root under `/home`.
- **A Python module.** `setup_env.sh` loads `Anaconda3/2024.02-1` by default (the module earlier Grace jobs used; which Python it
  provides now is not verified). Python 3.11 or newer is required.
- **Login-node manners.** At most 8 cores across all login nodes and no process longer than an hour. Everything above runs there. The
  `mpd` commands never import torch (only `mpd doctor --deep` does).

## Space and service-unit budget

Disk, for the example above (the environment size is measured for the pinned package versions in a reference environment; the
download size is an estimate; the others are the pinned manifests and a real fetch):

| what | download | on disk | files |
|---|---|---|---|
| Python environment (`$MPD_HOME/env`): torch 2.10 with CUDA 12.8 libraries, transformers, peft | about 3 GB, cached in `cache/pip` | 7.7 GB | about 28,000 |
| ESM2-3B and ESM2-650M weights (all five ESM2 models: 14.74 GB) | 13.98 GB | 13.98 GB | 12 |
| ProteinGym v1.3 | 62.4 MB | 1.2 GB | 421 |
| VenusMutHub | 6.3 MB | 6.3 MB | 797 |
| one campaign (`runs/<id>`) | | well under 100 MB (an estimate; the tiny test campaign is 188 KB) | tens to hundreds |

That is about 23 GB (26 GB while the pip cache is kept) and 29,500 files of the 1 TB and 250,000 files of `$SCRATCH`. A `*-one`
campaign over many assays writes a few thousand small files.

Service units: the plan above (`esm2-3b` to `esm2-650m`, `score-all`, evaluated on both benchmarks) prints an **upper bound of
1,806 SU, 9% of a 20,000 SU Basic allocation** (simulated: the Grace profile on a machine that is not Grace; the planner's estimate, not a
measurement; the allocation is from HPRC's *Introduction to HPRC Computing Resources*, 2026-01-23, slide 45, not verified for
November 2026). `score-one` over all 217 ProteinGym assays plans to 184,700 SU. sbatch's own "Requested SUs" line counts CPUs only,
so budget with the plan's figure. A submit above `--max-su` (default 10,000) is refused. Details and worked examples:
[docs/GRACE.md](docs/GRACE.md#what-it-costs).

## Models, data and licences

| | what | in this release |
|---|---|---|
| ESM2, 8M to 3B | teacher or student; MIT | available |
| SaProt, 35M and 650M | teacher or student; needs Foldseek structure strings; MIT | planned: the library is written and tested, the stages are not connected yet ([docs/SAPROT.md](docs/SAPROT.md)) |
| AIDO.Protein-RAG-16B | GenBio AI Community License | **not included** (below) |

Data: ProteinGym v1.3 (official Zenodo record [10.5281/zenodo.15293562](https://doi.org/10.5281/zenodo.15293562), MIT) and VenusMutHub
(Hugging Face `AI4Protein/VenusMutHub` at revision `08a27d07`, MIT), each file pinned by checksum; the structure set that SaProt needs for
VenusMutHub is a Zenodo deposit in preparation ([docs/ZENODO.md](docs/ZENODO.md)). Weights are pinned to one Hugging Face commit and
checked by sha256 ([docs/MODELS.md](docs/MODELS.md)). Details and caveats: [docs/DATA.md](docs/DATA.md).

**AIDO is not part of this release.** `genbio-ai/GB.Protein-RAG-16B-proteingym-dms-zeroshot` appears in `mpd models` only so that `mpd`
can tell you why it is not offered: `mpd` cannot plan, download or run it, and this repository contains no AIDO weights, code or outputs.
AIDO is under the GenBio AI Community License: non-commercial use only, and clause 1.2.2 does not allow using its outputs to train other,
similar models or for distillation toward that end, except for training the GenBio Materials themselves and their derivative works.
`mpd` therefore refuses every run in which AIDO is the teacher of a non-AIDO student, and has no switch to override that; it would need
written permission from GenBio and a change to the code. The AIDO-teacher results of the maintainers' earlier study (the distillation runs whose stored outputs the tests replay) cannot be
reproduced with `mpd` for this reason. If you want to use AIDO yourself, read [docs/LICENSES.md](docs/LICENSES.md) and the licence text first. That page is a
plain-language summary, not legal advice.

The ProteinMutPD code is MIT-licensed ([LICENSE](LICENSE)). Data and models keep their own licences, listed in
[docs/LICENSES.md](docs/LICENSES.md).

## Status

Tags: **tested on CPU** (this repository's tests, with fake Slurm tools and tiny random models), **Vision** (accepted by `sbatch
--test-only` on the Vision cluster, Slurm 25.05.6, nothing queued), **not verified** (nobody has run it).

| | status |
|---|---|
| Planner, resource model, script writer, `submit`, `status`, `logs`, `cancel`, `results`, `doctor`, `fetch` | tested on CPU against fake `sbatch`, `squeue`, `sacct`, `scancel`, `module`, `nvidia-smi` and a local Hugging Face stand-in |
| Scripts for the Vision profile (50 scripts: 4 methods, every stage, arrays) | accepted by `sbatch --test-only` on Vision |
| Scripts for the Grace profile | rendered and `bash -n` clean; **never given to a Grace Slurm** |
| `scripts/setup_env.sh` | run for real once on the Vision login node without torch; the torch install and the Grace defaults (module, home) are **not verified** |
| Any job on a Grace GPU node (first `import torch`, memory of a real student, every walltime estimate) | **not verified**: all walltimes and GPU memory figures are labelled estimates |
| Multi-GPU (DDP) training | tested on CPU: two processes equal one; **never run with NCCL on a GPU**; Grace defaults to one GPU |
| ESM2 stages for all four methods; resume after a walltime kill | tested on CPU with tiny random models |
| Base-model numbers of the earlier study | its stored ESM2-650M outputs, replayed through these loaders and evaluator, reproduce its macro Spearman (0.2095 VenusMutHub, 0.4280 ProteinGym; [docs/DATA.md](docs/DATA.md)). **No distilled number has been reproduced on a GPU** |
| SaProt | library tested on CPU; it reproduces the earlier study's structure strings with that study's Foldseek build, but the pinned Foldseek release has never been run; not connected to the stages, so still `planned` |
| AIDO | not in this release ([docs/LICENSES.md](docs/LICENSES.md)) |
| Venus structure deposit (SaProt on VenusMutHub for others) | in preparation; the pins are ready, nothing is published |
| Which readout is "primary" | `delta` for direct runs and `dual` for score runs is an **unconfirmed assumption of the maintainer**; both are always reported |

## Use the student

A trained student is a PEFT LoRA adapter (`adapter_config.json` and `adapter_model.safetensors`, rank 8 on the attention projections)
in `<campaign>/train/<unit>/checkpoint_<update>/`; `mpd results <campaign>` prints the path. Load it on the base model from `mpd fetch model`:

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

## Documentation

[GRACE](docs/GRACE.md) | [TROUBLESHOOTING](docs/TROUBLESHOOTING.md) | [RESULTS](docs/RESULTS.md) | [MODELS](docs/MODELS.md) |
[SAPROT](docs/SAPROT.md) | [DATA](docs/DATA.md) | [CLUSTERS](docs/CLUSTERS.md) | [ADDING_A_MODEL](docs/ADDING_A_MODEL.md) |
[LICENSES](docs/LICENSES.md) | [ZENODO](docs/ZENODO.md)

## Citation, contributing, support

Cite the software with [CITATION.cff](CITATION.cff) (its author list is still a placeholder); the data and models have their own citations ([docs/LICENSES.md](docs/LICENSES.md)).
Contributions: [CONTRIBUTING.md](CONTRIBUTING.md); changes: [CHANGELOG.md](CHANGELOG.md); security: [SECURITY.md](SECURITY.md). For an
issue, paste the output of `mpd doctor --report`, which removes account numbers, user names and home paths.
