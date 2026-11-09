# Running on TAMU HPRC Grace

What a student with a Grace account needs to know before and while using `mpd`: accounts, quotas, the environment,
what the jobs ask Slurm for, what it costs, and what has not been verified.

**Nothing in this repository has been run on Grace.** Every claim about Grace or about what `mpd` does carries one of these tags:

| tag | meaning |
|---|---|
| **[CPU]** | tested on CPU in this repository (the planner, the script writer and the submit logic, against fake `sbatch`, `squeue`, `sacct`, `scancel`, `module` and `nvidia-smi` programs; tiny random models) |
| **[Vision]** | accepted by `sbatch --test-only` on the Vision cluster (Slurm 25.05.6, nothing was queued) |
| **[HPRC]** | taken from an HPRC document, named in the text (see [Sources](#sources)) |
| **[earlier jobs]** | observed in Grace jobs that the profile's author ran in February-July 2026, before `mpd` existed; `mpd` has not been run |
| **[NOT VERIFIED]** | nobody has run it; read it as a plan, not as a fact |

Output pasted here is marked **simulated** when it came from a machine that is not Grace.

Contents: [Before you start](#before-you-start) | [Set up once](#set-up-once) | [Your first run](#your-first-run-cheap) |
[What mpd asks Slurm for](#what-mpd-asks-slurm-for) | [What it costs](#what-it-costs) | [Why jobs are offline](#why-jobs-are-offline) |
[Other ways to build the environment](#other-ways-to-build-the-environment) | [What is not verified](#what-has-not-been-verified-on-grace)

## Before you start

### Account and allocation

- Jobs are charged to a project account in service units (SU). `myproject` lists your accounts and balances, and
  `myproject -d <number>` changes the default **[HPRC]** (Introduction to HPRC Computing Resources, 2026-01-23, slides 45-46). A job
  with no `--account` is charged to the default account **[HPRC]** (Grace primer, Spring 2026, slide 15).
- `mpd` takes the account from `--account`, else `$MPD_ACCOUNT`, else the cluster profile, else Slurm's default **[CPU]**. It checks
  only the characters (letters, digits, `_`, `.`, `-`; there is no length rule, because HPRC's examples show 6, 12 and 13 digits).
  When `myproject` exists, `mpd plan` and `mpd doctor` look for your account in its output and warn if it is missing; the format of
  that output was not available when this was written, so the check is a plain text search **[CPU]** with a fake `myproject`.
- A default (Basic) allocation starts at 20,000 SU **[HPRC]** (same deck, slide 45; **[NOT VERIFIED]** for November 2026). SUs are
  per cluster and expire at the end of the fiscal year, 1 September to 31 August **[HPRC]** (knowledge base, AMS page).
- A request the balance cannot cover is refused at submission with
  `sbatch: error: (from job_submit) your account's balance is not sufficient to submit your job` **[HPRC]** (Grace primer, 2024).
  The balance is checked against the amount computed from the resources and the walltime you **request**; after the job ends the
  actual usage is deducted **[HPRC]** (AMS page). So a plan whose upper bound is above your balance can be refused even if the run
  would have been cheaper.

### Getting in, and what a login node may do

`ssh <netid>@grace.hprc.tamu.edu` lands on one of five login nodes, chosen round robin; off campus connect to the TAMU VPN first, and
have Duo ready **[HPRC]** (knowledge base, Accessing Grace; deck 2026-01, slide 15).

The login banner says "Please restrict usage to 8 CORES across ALL login nodes", and the rules are: no process longer than 1 hour,
idle sessions are killed after 1 hour, no `sudo` **[HPRC]** (deck 2026-01, slides 23-24).

| fine on a login node | not on a login node |
|---|---|
| `bash scripts/setup_env.sh` (pip downloads), `mpd fetch ...`, `mpd doctor`, `mpd plan`, `mpd submit`, `mpd status`, `mpd logs`, `mpd cancel`, `mpd results` | running a model; `mpd doctor --deep` (it imports torch, which took minutes on a cold node in the author's jobs); conda installs, which HPRC wants inside an `srun` session or a job |

None of the `mpd` commands in the left column imports torch **[CPU]** (a test enforces it for the planner and the modules it imports; `--deep` is
the exception). `mpd fetch model` stops by itself after
55 minutes, keeps what it has and exits with status 3; run the same command again to continue **[CPU]**.

### Storage

| | `$HOME` (`/home/<netid>`) | `$SCRATCH` (`/scratch/user/<netid>`) |
|---|---|---|
| space | 10 GB | 1 TB |
| files | 10,000 | 250,000 |
| backup | nightly | none |
| notes | never expanded; a full file quota can lock you out of your account | meant for temporary work files; **the purge rule is not stated in any HPRC document read: [NOT VERIFIED]** |

**[HPRC]** (deck 2026-01, slide 20; deck 2026-04, slides 2 and 34). `showquota` prints your usage.

`mpd` keeps everything under one working root, `$MPD_HOME`, which is `$SCRATCH/mpd` unless you say otherwise. A root under `/home`
is refused with `<path> is under /home (10 GB / 10,000 files on Grace). Use scratch: export MPD_HOME=$SCRATCH/mpd` **[CPU]**. The
setup script keeps pip's cache out of `$HOME`, and every job script points the Hugging Face, Torch, Triton and matplotlib caches into
`$MPD_HOME/cache` **[CPU]**.

```
$MPD_HOME/
  env/          the Python venv and mpd_env.json (which modules the jobs load)
  cache/        pip wheels, tmp, build copy of the sources, job caches (xdg, torch, triton, mpl), Foldseek 3Di cache
  data/         proteingym-v1.3/  venusmuthub-v1/  (and venusmuthub-structures-v1/ for SaProt)
  models/       one plain folder per model key (esm2-650m/, esm2-3b/, ...)
  tools/        foldseek-10-941cd33/ (SaProt only)
  runs/         one folder per campaign (see RESULTS.md)
```

How much of the quota a run takes is in the [README](../README.md#space-and-service-unit-budget).

## Set up once

Run these on a login node, in a clone of the repository.

```bash
git clone https://github.com/yh79-ss/ProteinMutPD && cd ProteinMutPD
bash scripts/setup_env.sh --cluster grace
source $SCRATCH/mpd/env/bin/activate
mpd fetch proteingym
mpd fetch venusmuthub
mpd fetch model esm2-3b esm2-650m
mpd doctor
```

`mpd doctor` reports `[ok]`, `[warn]` (not blocking), `[FAIL]` (a real `mpd submit` will refuse) and `[skip]` (could not be checked), prints the
command that fixes each problem, and exits with status 1 on any `[FAIL]`. What it prints before anything is set up (**simulated**: run on a
machine that is not Grace, with an empty `$SCRATCH/mpd`, so `showquota` and Slurm are absent):

```text
[ok]   home              /scratch/user/<netid>/mpd (exists; writable; 149 GB free on this filesystem)
[skip] quota             showquota not found (it exists on HPRC clusters); check your quota yourself
[ok]   account           not set: jobs are charged to your default Slurm account (`myproject` lists accounts; set --account or MPD_ACCOUNT)
[FAIL] python            no Python environment at /scratch/user/<netid>/mpd/env. Run: bash scripts/setup_env.sh --cluster grace
[warn] data:proteingym   ProteinGym data not found under /scratch/user/<netid>/mpd/data/proteingym-v1.3. Run on a login node: mpd fetch proteingym --cluster grace
[warn] data:venusmuthub  VenusMutHub data not found under /scratch/user/<netid>/mpd/data/venusmuthub-v1. Run on a login node: mpd fetch venusmuthub --cluster grace
[warn] weights           none fetched yet. not fetched: esm2-8m, esm2-35m, esm2-150m, esm2-650m, esm2-3b (a plan needs only its teacher and student; `mpd models` has sizes)
                         fix: mpd fetch model esm2-650m --cluster grace
[ok]   foldseek          not installed (only SaProt needs it): mpd fetch foldseek --cluster grace
[skip] slurm             sbatch not found on PATH: this is not a Slurm login node (plan and --dry-run work anywhere; submit needs it)

NOT READY: 1 error, 3 warnings. Fix the [FAIL] lines (the command to run is under each), then run `mpd doctor` again.
```

When nothing blocks, the footer reads `READY`, followed by "These are static checks; they do not prove that a job will run on the cluster".
`mpd plan` and `mpd submit --dry-run` print the same table for the run you described (with `weights` and `data` as `[FAIL]` when that run needs
them) and still exit 0.

### The environment

`scripts/setup_env.sh` is a bash script because `mpd` does not exist before the environment does. In order, it **[CPU]** (fake
`module` and fake Python; run for real once on the Vision login node with `PIP_NO_DEPS=1` and `--no-torch`, which installs no
torch):

1. refuses a home under `/home`;
2. loads the modules (`module purge` first). The Grace default is `Anaconda3/2024.02-1`, the module the author's earlier jobs
   loaded **[earlier jobs]**; which Python it provides today is **[NOT VERIFIED]**. If it is missing the script says so and points at
   `module spider Python`. HPRC's own examples use `GCCcore/13.2.0 Python/3.11.5` **[HPRC]** (deck 2026-01, slide 40; never run
   here): `bash scripts/setup_env.sh --cluster grace --modules "GCCcore/13.2.0 Python/3.11.5"`;
3. needs Python 3.11 or newer, then creates the venv `$MPD_HOME/env`;
4. runs `pip install -c constraints/grace.txt` for `mpd` with the training extra. The pinned set is torch 2.10.0, transformers
   4.51.0 and peft 0.13.2, the combination the test suite ran with on CPU. `--constraints none` installs the looser ranges of
   `pyproject.toml`, which is **not** the tested set;
5. writes `$MPD_HOME/env/mpd_env.json`, which records the modules, the interpreter and the package versions. The job scripts load the
   same modules, so there is no profile to edit **[CPU]**;
6. adds `MPD_HOME` and `MPD_CLUSTER` to `env/bin/activate` (so `source .../activate` is all a new shell needs), and runs `mpd doctor`.

pip's cache and `TMPDIR` are under `$MPD_HOME/cache`; nothing is written under `$HOME`. The wheels are about 3 GB to download (an
estimate from a pip dry run for a newer torch; not measured for the pinned set) and the result is about 28,000 files and 7.7 GB
(measured on the reference environment with the same package versions). If the login node ends the script after its hour, run it
again: the wheels are cached and it continues.

Other options: `--home DIR` (then `source DIR/env/bin/activate`), `--python PATH`, `--modules none`, `--no-torch` (planner only, for a machine that runs no job), `--dev`
(editable install with the test extra, for working on `mpd`), `--rebuild`, `--yes`. `bash scripts/setup_env.sh --help` lists them.

**After `git pull`**, run the script again. A normal install is a snapshot taken from a copy of the sources, so pulling never changes
the code under a running campaign; `mpd doctor` warns when the installed `mpd` or `constraints/grace.txt` differs from what the
environment was built with **[CPU]**.

Whether a venv built from a module's Python runs on a compute node with only the recorded modules loaded is **[NOT VERIFIED]**. The
only recipe with a record on Grace is a conda environment called by its absolute interpreter path **[earlier jobs]**; see
[Other ways to build the environment](#other-ways-to-build-the-environment).

### Data and weights

Compute nodes have no internet (below), so these run on a login node:

| command | download | on disk |
|---|---|---|
| `mpd fetch proteingym` | 62.4 MB (Zenodo record 15293562, checksum-verified) | 1.2 GB, 421 files |
| `mpd fetch venusmuthub` | 796 small files, fetched one at a time (Hugging Face rate-limits bursts) | 6.3 MB, 797 files |
| `mpd fetch model esm2-3b esm2-650m` | 13.98 GB (11.37 GB + 2.61 GB), resumable | the same, 12 files |
| `mpd fetch model --list` | what is pinned, its size, and what you already have | |

Sizes are from the pinned manifests and from a real fetch of the data **[CPU]**; the download of the large model files from a Grace
login node has not been timed **[NOT VERIFIED]**. If a login node cannot reach the Hugging Face Hub, or you are rate-limited
(HTTP 429), see [MODELS.md](MODELS.md#no-internet-on-the-cluster) and [TROUBLESHOOTING.md](TROUBLESHOOTING.md).

## Your first run (cheap)

Before an expensive campaign, check the whole chain once with the smallest model. This uses existing flags only: ESM2-8M as both
teacher and student, 50 updates, one GPU, one scoring shard. It needs `proteingym` and `esm2-8m` (31 MB) fetched.

```bash
mpd fetch model esm2-8m
mpd plan   --account <ACCOUNT> --teacher esm2-8m --student esm2-8m --train-data proteingym --method score-all --eval-data proteingym --updates 50 --train-gpus 1 --score-shards proteingym=1
mpd submit --account <ACCOUNT> --teacher esm2-8m --student esm2-8m --train-data proteingym --method score-all --eval-data proteingym --updates 50 --train-gpus 1 --score-shards proteingym=1 --dry-run
mpd submit --account <ACCOUNT> --teacher esm2-8m --student esm2-8m --train-data proteingym --method score-all --eval-data proteingym --updates 50 --train-gpus 1 --score-shards proteingym=1
```

The plan (**simulated**: the Grace profile on a machine that is not Grace, with a made-up account; the numbers are the planner's
estimates, not measurements):

```text
stage          tasks  gpu       cpus   mem  walltime          SU (max)  after
prepare            1  -            8   32G  04:00:00                32  -
teacher            1  1xa100      24  180G  00:15:00 est            24  prepare
train              1  1xa100      24  180G  00:15:00 est            24  teacher
score-base         1  1xa100      24  180G  00:15:00 est            24  prepare
score              1  1xa100      24  180G  00:15:00 est            24  train
evaluate           1  -            4   16G  01:00:00                 4  score-base, score

estimated upper bound: 133 SU (every task runs to its requested walltime and the node share it asks for; the charge follows actual runtime). Estimate, not measured.
```

133 SU is 0.7% of a 20,000 SU allocation. It is an upper bound: 15 minutes is the floor of the walltime formula (a 10-minute start-up
allowance plus rounding), not a prediction, and the real charge follows the run time. `--dry-run` writes and checks the scripts
(including `sbatch --test-only` on each distinct shape) and queues nothing.

What to look at afterwards:

- `mpd status` shows each stage's Slurm state; `mpd logs <campaign> prepare` shows a stage's output (`MPD_JOB_START host=... job=...`
  is the first line of every job). A job with an empty log failed before that line: read its `.err` file with `--err`.
- `mpd results <campaign>` after the `evaluate` stage. Because the teacher is the student, its development loss starts at (about)
  zero and `delta_vs_base` is about zero (on CPU in float32 the loss is exactly zero **[CPU]**; on a GPU in bf16 it will not be). The run
  checks the plumbing and costs little; it is not an experiment. Use `--teacher esm2-35m` (136 MB more to fetch) to see something move.
- No `mpd` job has ever run on a GPU. The first `import torch` took 88 to 127 s, and once more than 3 minutes, in the author's earlier Grace
  jobs **[earlier jobs]**; the 10-minute start-up allowance is meant to cover that **[NOT VERIFIED]**.

## What mpd asks Slurm for

| stage | GPUs | cores / memory | default walltime |
|---|---|---|---|
| `prepare` | none | 8 / 32 GB | 4 h, fixed |
| `teacher` | 1 | 24 / 180 GB | from the token count of the training table (estimate) |
| `train` | 1 (`--train-gpus 2` for DDP) | 24 / 180 GB for one A100, 48 / 360 GB for two | from the updates, the model size and the GPU (estimate) |
| `score-base`, `score` | 1 | 24 / 180 GB | from the tokens of the largest task (estimate) |
| `evaluate` | none | 4 / 16 GB | 1 h, fixed |

Half a node (24 cores, 180 GB) per A100 is HPRC's `maxconfig` recommendation **[HPRC]** (Introduction to AlphaFold, Fall 2024).
Leaner requests (4-16 cores, 64-160 GB per A100) ran on Grace and cost fewer SUs, because memory counts as cores
**[earlier jobs]**; `mpd` has no flag for them, use a [custom profile](CLUSTERS.md). The host memory the stages really need has not
been measured **[NOT VERIFIED]**.

**Training uses one GPU by default.** DDP with NCCL has never run on a real GPU, a two-GPU request takes a whole node (48 cores,
360 GB), and one GPU costs about the same SUs for twice the time **[NOT VERIFIED]** for the two-GPU request. `--train-gpus 2` gives the
DDP shape (tested on CPU with gloo: two processes equal one **[CPU]**); more GPUs than a node holds is an error rather than a silent
clamp **[CPU]**.

### Partitions and the walltime trap

| partition | longest request | what it is |
|---|---|---|
| `short` | 2:00:00 | CPU |
| `medium` | 1-00:00:00 | CPU |
| `long` | 7-00:00:00 | CPU |
| `xlong` | 21-00:00:00 | CPU |
| `gpu` | 4-00:00:00 | A100, RTX 6000, T4 |
| `gpu-a40` | 4-00:00:00 | A40 (the 2024 and Fall-2025 decks said 10 days) |
| `bigmem` | 2-00:00:00 | 3 TB nodes |

**[HPRC]** (deck 2026-01, slide 62, `sinfo`). Jobs are assigned to partitions "based on the resources requested" (slide 60). The two CPU
stages name no partition, so Slurm routes them by walltime: a six-hour request with no partition went to `medium` **[earlier jobs]**.

**The trap.** Slurm accepts a request that is longer than its partition allows and then leaves it pending forever with the reason
`PartitionTimeLimit`; an explicit `--partition=short` with 12 hours did exactly that **[earlier jobs]**. So `mpd` refuses a walltime
above the GPU partition's limit (`[train] requested 5-00:00:00 exceeds the a100 partition limit 4-00:00:00; Slurm would hold the
job with PartitionTimeLimit ...`) and never names a CPU partition **[CPU]**. `mpd status` prints the pending reason of waiting jobs.

### GPUs

| `--gpu` | card | per node | memory | SU per GPU-hour | partition | trained by `mpd` |
|---|---|---|---|---|---|---|
| `a100` (default) | NVIDIA A100-PCIE-40GB, no NVLink | 2 (100 nodes) | 40 GB | 72 | `gpu` | yes |
| `a40` | A40 | 2 (15 nodes); see the note | 48 GB | 48 | `gpu-a40` | yes, with a warning |
| `rtx` | Quadro RTX 6000 | 2 (9 nodes) | 24 GB | 48 | `gpu` | refused |
| `t4` | T4 | 4 (8 nodes) | 16 GB | 24 | `gpu` | refused |

Rates, memory, partitions and node counts: **[HPRC]** (deck 2026-01, slides 9, 58, 62). The card name of the A100 was printed by
`nvidia-smi` in the author's jobs **[earlier jobs]**. `--gres` is always typed (`gpu:a100:1`), because an untyped request landed on a
T4 node in the author's jobs **[earlier jobs]** and Vision rejects a type with the wrong case **[Vision]**.

- **Why `rtx` and `t4` are refused.** Both have compute capability 7.5 in HPRC's `gpuinfo` table (A100 8.0, A40 8.6), and `mpd` trains
  in bfloat16, which needs native support (8.0 and up). The same table labels the RTX 6000 "Ada", which does not fit a compute capability
  of 7.5 or a memory of 24 GB; `mpd` follows the numbers. The planner stops with an error that names `--gpu a100` and `--gpu a40`.
  `--allow-unsupported-gpu` overrides it and runs every stage in float32 (`MPD_DTYPE=float32`), which is slow, needs about twice the
  memory and has never been run on such a card (only the planning is tested **[CPU]**); bf16 on these cards is **[NOT VERIFIED]**.
- **The A40 tip.** When the A100 queue is long, `--gpu a40` is the alternative. In July 2026 the author moved 60 single-GPU jobs from the
  A100 to the A40 partition because the A100 jobs were projected to wait much longer while an A40 restart could begin at once; they ran
  20 at a time without failures **[earlier jobs]** (one observation, not a guarantee). The A40 profile prints a warning because access
  and limits were not confirmed. HPRC's sources disagree on A40s per node (two on the
  hardware slide, three in a `maxconfig` capture in the same deck); `mpd` assumes two, which only limits `--train-gpus`
  **[NOT VERIFIED]**. Choosing `--gpu a40` for a recipe you already planned with the A100 makes a new campaign, because the GPU is part of
  the campaign id **[CPU]**.
- **A bad node.** In May 2026 a node (g006) listed its A100 in `nvidia-smi` while `torch.cuda.is_available()` was false
  **[earlier jobs]**. A GPU stage now stops with exit status 31 and `MPD_NO_GPU: ...` instead of running on the CPU **[CPU]**. Run the same `mpd submit`
  command again with `--force`; if the same node fails again, add `--sbatch-arg=--exclude=<node>` with the node name from the log. Extra sbatch
  options are part of the campaign id, so that starts a new campaign ([TROUBLESHOOTING.md](TROUBLESHOOTING.md#a-job-failed-or-stopped)).

### Walltimes and GPU memory are estimates

Every walltime `mpd` computes is printed with `est` (estimate, not measured), or `meas` (scaled from timed runs: only esm2-650m on the
Vision H200), and one you gave is `set`:

```
training minutes = (63.8 all-assay | 53.2 one-assay) per 1,000 updates on one H200 for esm2-650m      [timed, earlier study]
                   x (student parameters / 655)^0.9                                                    [extrapolated: no other student was timed]
                   x updates / 1000  /  data-parallel GPUs  x GPU slowdown (A100 3.0, A40 6.0)        [ASSUMED, not measured]
walltime         = round up to 15 minutes (training minutes x 1.5 slack + 10 minutes start-up)
teacher / score  = tokens / 36,000 per second on an H200 (esm2-650m) x the same size and GPU factors x 1.5 + 10 minutes
```

Every Grace walltime therefore depends on an assumed A100 slowdown. Override with `--train-time`, `--score-time` and `--teacher-time`
(`HH:MM:SS` or `D-HH:MM:SS`; `8h`, `0` and the ambiguous `08:00` are refused with a message **[CPU]**).

Memory: weights, plus estimated activations (counted from the saved tensors on the meta device, not measured on a GPU), plus 2.5 GB,
must fit 85% of the card. `mpd` trains at micro-batch 2 when that fits and at 1 otherwise, keeping the global batch of 64 through more
accumulation steps: on a 40 GB A100 the `esm2-3b` student trains at micro-batch 1 **[CPU]**. If the first job runs out of memory, use a
larger GPU (`--gpu a40` has 48 GB) or a smaller student; more GPUs do not help, because every GPU holds a full copy **[NOT VERIFIED]**.

### The job script

`mpd submit` writes one script per stage under `<campaign>/scripts/`; you can read them (and `bash -n` them) after a `--dry-run`. The
`train` script for `--train-gpus 2` with the module `Anaconda3/2024.02-1` recorded by `setup_env.sh`, as the code renders it (paths
and the list of cache variables shortened; **simulated**: rendered on a machine that is not Grace with a made-up account; the script itself is never given to Grace's Slurm
in any test **[NOT VERIFIED]**, while the same renderer's Vision scripts were accepted by `sbatch --test-only` **[Vision]**):

```bash
#!/bin/bash
#SBATCH --job-name=mpd-train-esm28m
#SBATCH --comment=mpd:<campaign>:<nonce>:train
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=360G
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=gpu
#SBATCH --account=123456789012
#SBATCH --output=/scratch/user/<netid>/mpd/runs/<campaign>/logs/%x_%j.out
#SBATCH --error=/scratch/user/<netid>/mpd/runs/<campaign>/logs/%x_%j.err
#SBATCH --open-mode=append

set -eo pipefail
type module >/dev/null 2>&1 || source /etc/profile 2>/dev/null || true
type module >/dev/null 2>&1 || { echo "MPD_MODULE_LOAD_FAILED the module command is not available (Anaconda3/2024.02-1)" >&2; exit 1; }
module purge
module load Anaconda3/2024.02-1 || { echo "MPD_MODULE_LOAD_FAILED Anaconda3/2024.02-1" >&2; exit 1; }
unset PYTHONPATH PYTHONHOME
set -u
export MPD_HOME=... HF_HOME=... XDG_CACHE_HOME=... TORCH_HOME=... TRITON_CACHE_DIR=... MPLCONFIGDIR=...
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=24
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MPD_REQUIRE_CUDA=1
nvidia-smi -L || { echo "MPD_NO_GPU: no usable GPU on $(hostname -s); resubmit with --sbatch-arg=--exclude=$(hostname -s)" >&2; exit 31; }
echo "MPD_JOB_START host=$(hostname) job=${SLURM_JOB_ID:-} stage=train"
cd /scratch/user/<netid>/mpd/runs/<campaign>

<python> -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 -m mpd.stages.train --campaign-dir /scratch/user/<netid>/mpd/runs/<campaign> --stage train --task-index 0
echo "MPD_STAGE_DONE train"
```

- **The environment is cleaned, not inherited.** A job starts with the environment of the shell that ran `sbatch`, so the script runs `module
  purge` (when `module` exists), loads only the recorded modules, and unsets `PYTHONPATH` and `PYTHONHOME`. With no module recorded the
  three module lines become `if type module >/dev/null 2>&1; then module purge; fi`.
- **There is no `--export` and no `--get-user-env`.** Slurm 25.05 rejects `--get-user-env=L` ("doesn't allow an argument") **[Vision]**, and
  `--export=NONE` implies `--get-user-env`; none of the author's Grace job scripts (about a thousand) used either flag **[earlier jobs]**. What
  Grace's Slurm does with them is **[NOT VERIFIED]**, so `mpd` leaves them out.
- **`set -u` comes after the modules.** `/etc/bashrc` and the module scripts read unset variables; `set -euo pipefail` before them gave
  `BASHRCSOURCED: unbound variable` on Grace **[earlier jobs]**.
- **Everything writable is under `$MPD_HOME`**: the caches, and `HF_HOME` for stray Hugging Face writes. `PYTHONNOUSERSITE=1` keeps
  `~/.local` out of the job.
- **Offline.** `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` (above). A `MPD_MODEL_<KEY>`, `MPD_MODELS_DIR` or `MPD_FOLDSEEK` set in your shell when
  you submit is copied into the script as an `export`, and so is every `--env NAME=VALUE`.
- **GPU stages check the GPU.** `nvidia-smi -L` must work, and the stage itself stops with `MPD_NO_GPU` (exit 31) if CUDA is unusable.
  `MPD_JOB_START` is the first line a job prints; `MPD_STAGE_DONE` the last.
- **`--comment` carries a per-submission tag**, which `mpd status`, `mpd cancel` and the retry after a controller error use to find this
  submission's jobs in `squeue`.
- A DDP training script (above) is launched with `torch.distributed.run --standalone`; with one GPU the line is `<python> -m mpd.stages.train
  ...`. Array stages add `--array=0-N%T` and pass `--task-index $((SLURM_ARRAY_TASK_ID + offset))`.

## What it costs

### The SU rule

```
effective cores = max(cores, min(48, ceil(memory_GB / 351.5625 * 48)))
SU              = (effective cores + GPU rate x GPUs) x hours        GPU rate per hour: A100 72, A40 48, RTX 6000 48, T4 24
```

**[HPRC]** (knowledge base, AMS page, archived 2026-02-09; deck 2026-01, slide 58). Memory counts as cores, and the CPU part is a
maximum, not a sum. The code reproduces HPRC's `maxconfig` example (1 A100 at half a node for 96 hours = 9,312 SU), the two-GPU example of
the deck (2 A100, 250 GB, 1 hour = 179 SU, slide 79) and the three memory examples of the AMS page **[CPU]**.

| request (per hour) | effective cores | SU per hour |
|---|---|---|
| 1 A100, 24 cores, 180 GB (what `mpd` asks, and `maxconfig`'s shape) | 25 | 97 |
| 1 A100, 8 cores, 64 GB | 9 | 81 |
| 1 A100, 4 cores, 160 GB | 22 | 94 |
| 2 A100, 48 cores, 360 GB (a whole node) | 48 | 192 |
| 2 A100, 16 cores, 128 GB | 18 | 162 |
| 1 A40, 24 cores, 180 GB | 25 | 73 |
| 1 A40, 4 cores, 32 GB | 5 | 53 |

Computed with the code (`Cluster.estimate_su`) **[CPU]**.

### What sbatch's own SU line means

After a successful submission Grace prints a block like this (copied from a real job log of 2026):

```text
Submitted batch job <id>
(from job_submit) your job is charged as below
              Project Account: <account>
              Account Balance: <balance>
              Requested SUs:   280.0
  Requested SUs is based on CPU only and does not include SU charges for
  requested GPUs. A negative balance is possible if the difference is
  greater than the remaining balance.
```

**[earlier jobs]** The figure counts CPUs only and covers all tasks of an array at once, so it is always lower than the plan's
total, which includes the GPUs. Budget with the plan's number. `maxconfig -f <script>` prints HPRC's own figure for a script, and
`mpd submit ... --dry-run` writes the scripts you can give it **[HPRC]** (deck 2026-01, slide 64).

### Budget

- The plan's total is an upper bound (every task for its whole requested walltime) and an estimate. For the README example
  (`esm2-3b` to `esm2-650m`, `score-all`, evaluated on both benchmarks) it is **1,806 SU, 9% of 20,000** (**simulated** plan; **[HPRC]**
  for the allocation). With `esm2-3b` as the student (teacher `esm2-650m`) the plan is 4,983 SU, 25%, with 1-16:00:00 of training at
  micro-batch 1 (**simulated**).
- `*-one` methods train one student per assay, so the training line is multiplied by the number of assays: `score-one` over all 217
  ProteinGym assays plans to **184,700 SU**, about nine times the Basic allocation (**simulated**). `mpd submit` refuses a plan above
  `--max-su` (default 10,000) and says why; `plan` and `--dry-run` only warn. Use `*-one` for a few assays.
- Changing only `--eval-data`, `--gpu`, `--cluster`, `--score-shards`, a walltime override, `--sbatch-arg`, `--module` or `--env` makes
  a **new campaign**, which retrains. `--home`, `--python` and `--account` do not **[CPU]**.

## Why jobs are offline

Compute nodes have no internet **[HPRC]** (Introduction to NGS, Fall 2025, slide 97; knowledge base, File Transfer), and the author's
jobs failed with `OSError: [Errno 101] Network is unreachable` when they tried to reach `huggingface.co` **[earlier jobs]**. So the
weights and data are fetched first, and every job script sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. A model missing at job
time is an error naming the fetch command, never a download **[CPU]**. A job loaded a model from a local folder with those flags and
no network **[earlier jobs]**.

`module load WebProxy` sets the proxy variables so that a compute node can reach the internet; HPRC's own job templates use it
**[HPRC]** (deck 2026-04, slide 30). You need it only if you fetch inside a job or an `srun` session. Whether Hugging Face works through
it is **[NOT VERIFIED]**: the author loaded it in many scripts, but no log shows a download through it. Do not copy proxy addresses.

## Other ways to build the environment

The default is a venv made by `scripts/setup_env.sh`. HPRC advises against conda "if it can be set up with pip install" (about 30,000
files per conda environment against a `$HOME` limit of 10,000) and gives rules for the case where it is the only way **[HPRC]**
(deck 2026-04, slides 34-35 and 37).

**Conda, the recipe with a track record.** The author's Grace jobs (170 or more, February to July 2026) ran a conda environment's interpreter
by its absolute path; some loaded no module at all, others loaded `Anaconda3/2024.02-1` first and still called the interpreter by its
path, and a CUDA wheel worked either way with no CUDA module **[earlier jobs]**. To use that with `mpd`:

- build it inside an `srun` session or a one-core batch job, never on a login node, with `module load WebProxy` for the internet
  **[HPRC]** (deck 2026-04, slide 37). HPRC recommends its Mamba module over Anaconda because it is much faster (slide 36); the author's jobs
  used `Anaconda3/2024.02-1` **[earlier jobs]**;
- keep it under `$SCRATCH`, never `$HOME`: create it with `-p $SCRATCH/...`, and point `CONDA_PKGS_DIRS` and `CONDA_ENVS_PATH` at
  `$SCRATCH` too, because conda's default package cache and environment folder are under `~/.conda` (this advice is the author's, not
  HPRC's; **[NOT VERIFIED]** on Grace);
- never run `conda init`; use `source activate`; load only the one conda module and nothing else, because other modules "will likely
  break" a conda environment **[HPRC]** (slide 35);
- install `mpd` into it with `pip install -c constraints/grace.txt '.[train]'` from the clone (**[NOT VERIFIED]**: the pins were tested
  on CPU in a venv, not in conda);
- tell `mpd` to run that interpreter and to load no modules: `export MPD_PYTHON=/scratch/user/<netid>/conda/envs/mpd/bin/python` (or
  `--python`), and do not run `setup_env.sh` for it. With no `mpd_env.json` the job scripts only `module purge`, then run the
  interpreter **[CPU]**, and `mpd doctor` warns that there is no environment record: expected here.

**HPRC's ModuLair route** builds a venv on top of the `PyTorch/2.7.0` module (`GCC/13.2.0 OpenMPI/4.1.6`) **[HPRC]** (deck 2026-04,
slides 15-23). `mpd` asks for `torch>=2.7`, so it could work, but the pinned and tested set is torch 2.10; this route is untried
**[NOT VERIFIED]**.

Whatever you use, do not install `deepspeed` into the environment: in the author's jobs it broke `from transformers import
EsmForMaskedLM` with `MissingCUDAException: CUDA_HOME does not exist` **[earlier jobs]**.

## What has not been verified on Grace

**Nothing has run on Grace**, so everything above is a plan. Specifically:

1. that a script rendered by `mpd` is accepted and runs; DDP with NCCL on two A100-PCIe cards and a two-GPU request (48 cores, 360
   GB); job arrays above 212 tasks (`mpd` splits arrays at 1,000 tasks, which is safe only if `MaxArraySize` is at least 1,001);
2. Grace's Slurm version (so whether it accepts `--comment` and `--open-mode=append`, which `mpd` writes and Vision accepts), per-user
   job and GPU limits, QOS limits;
3. the `$SCRATCH` purge age;
4. the A40 partition: GPUs per node, the walltime limit (4 or 10 days) and whether every account may use it;
5. what `module load WebProxy` sets, and whether Hugging Face works through it;
6. that a venv built from a module's Python runs on compute nodes with only the recorded modules; that `Anaconda3/2024.02-1` still
   exists and which Python it provides; HPRC's module-based recipes;
7. bfloat16 on the T4 and the RTX 6000;
8. every `time_factor` and walltime, the host memory the stages need, and the GPU memory of real training runs;
9. the SU rate of 72 per A100-hour and the 20,000 SU Basic allocation as of November 2026;
10. what Slurm does when `--output` points at a missing directory (`mpd` creates `<campaign>/logs` before it submits);
11. the output formats of `showquota` and `myproject` on Grace (`mpd` parses them tolerantly and never fails on them);
12. how long the first `import torch` takes, and whether the 10-minute start-up allowance covers it plus a 3B model load.

To settle most of these once, run on a Grace login node (all are read-only; `sbatch --test-only` queues nothing):

```bash
sbatch --version; scontrol show config | grep -E 'SLURM_VERSION|MaxArraySize|MaxJobCount|SchedulerParameters'
sinfo -o '%P %l %D %G' | sort
gpuinfo; gpuavail; maxconfig -f <a rendered .sbatch from mpd submit --dry-run>
sacctmgr -n -P show qos format=Name,MaxJobsPU,MaxSubmitPU,MaxTRESPU,MaxWall
showquota; myproject; module spider Python; module spider Anaconda3
sbatch --test-only <a rendered prepare.sbatch>
```

If you run them, the output (without account numbers or your user name) is the most useful thing you can add to an issue; see
[TROUBLESHOOTING.md](TROUBLESHOOTING.md#what-to-paste-in-an-issue).

## Sources

HPRC documents used for the **[HPRC]** tags. The live knowledge base answers scripted clients with a bot challenge, so the knowledge
base pages were read from archived copies (dates given); the decks are public PDFs.

- Introduction to HPRC Computing Resources, 2026-01-23: `https://hprc.tamu.edu/files/training/2026/Spring/Intro_to_HPRC_clusters_2026_spring.pdf`
  ("deck 2026-01"; slide numbers as printed in the PDF).
- Setting up Environments for AI Research and Software Development on TAMU HPRC, 2026-04-24:
  `https://hprc.tamu.edu/files/training/2026/Spring/Setting_Up_Coding_Environments_for_HPC.pdf` ("deck 2026-04").
- Knowledge base pages, archived copies: Account Management System (AMS, 2026-02-09), Grace Batch System (2026-03-14), Accessing Grace
  (2026-02-07), and the File Transfer page as summarised by a search engine (secondary).
- Introduction to AlphaFold (Fall 2024), Introduction to NGS (Fall 2025), and the Grace primers of 2024 and Fall 2025 (training decks).

Not read: the live knowledge base pages for Filesystems and Files, WebProxy, Hardware and Key Policies, which is why the purge rule and
the proxy settings are listed above as not verified.
