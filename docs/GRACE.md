# Running on TAMU HPRC Grace

What a student with a Grace account needs to know before and while using `mpd`: accounts, quotas, the environment,
what the jobs ask Slurm for, what it costs, what to do when a job does not finish, and what has not been verified.

**Nothing in this repository has been run on Grace.** Every claim about Grace or about what `mpd` does carries one of these tags:

| tag | meaning |
|---|---|
| **[CPU]** | tested on CPU in this repository (the planner, the script writer, the submit and recovery logic, against fake `sbatch`, `squeue`, `sacct`, `scancel`, `module` and `nvidia-smi` programs; tiny random models) |
| **[Vision]** | accepted by `sbatch --test-only` on the Vision cluster (Slurm 25.05.6, nothing was queued) |
| **[HPRC]** | taken from an HPRC document, named in the text (see [Sources](#sources)) |
| **[earlier jobs]** | observed in Grace jobs that the profile's author ran in February-July 2026, before `mpd` existed; `mpd` has not been run |
| **[NOT VERIFIED]** | nobody has run it; read it as a plan, not as a fact |

Output pasted here is marked **simulated** when it came from a machine that is not Grace or from a fake Slurm; job ids, nodes and times in
such output are invented.

Contents: [Before you start](#before-you-start) | [Set up once](#set-up-once) | [Your first run](#your-first-run-cheap) |
[When a job does not finish](#when-a-job-does-not-finish) | [What mpd asks Slurm for](#what-mpd-asks-slurm-for) |
[What it costs](#what-it-costs) | [SaProt on Grace](#saprot-on-grace) | [Why jobs are offline](#why-jobs-are-offline) |
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
  hf_cache/     only for stray Hugging Face writes: the jobs are offline and read the folders under models/
  data/         proteingym-v1.3/  venusmuthub-v1/  (and venusmuthub-structures-v1/ for SaProt)
  models/       one plain folder per model key (esm2-650m/, esm2-3b/, ...)
  tools/        foldseek-10-941cd33/ (SaProt only)
  runs/         one folder per campaign (see RESULTS.md)
```

### How much space a run takes

Disk, for the README's example (`esm2-3b` to `esm2-650m`, `score-all`, evaluated on both benchmarks; the environment size is measured for the pinned package versions in a reference environment; the
download size is an estimate; the others are the pinned manifests and a real fetch):

| what | download | on disk | files |
|---|---|---|---|
| Python environment (`$MPD_HOME/env`): torch 2.10 with CUDA 12.8 libraries, transformers, peft | about 4 GB, cached in `cache/pip` | 7.7 GB | about 28,000 |
| ESM2-3B and ESM2-650M weights (all five ESM2 models: 14.74 GB; all seven models: 17.48 GB) | 13.98 GB | 13.98 GB | 12 |
| ProteinGym v1.3 | 62.4 MB | 1.2 GB | 421 |
| VenusMutHub | 6.3 MB | 6.3 MB | 797 |
| one campaign (`runs/<id>`) | | well under 100 MB (an estimate; the tiny CPU test campaign is 79 KB) | tens to hundreds |

That is about 23 GB (about 27 GB while the pip cache is kept) and 29,600 files of the 1 TB and 250,000 files of `$SCRATCH`. A `*-one`
campaign over many assays writes a few thousand small files.

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

`setup_env.sh` asks "Continue? [y/N]" before it builds; add `--yes` when you paste several lines at once. `mpd doctor` reports `[ok]`, `[warn]` (not
blocking), `[FAIL]` (a real `mpd submit` will refuse) and `[skip]` (could not be checked), prints the command that fixes each problem, and exits
with status 1 on any `[FAIL]`. What it prints before anything is set up (**simulated**: run on a machine that is not Grace, with an empty
`$SCRATCH/mpd`, so `showquota` and Slurm are absent):

```text
[ok]   home              /scratch/user/<netid>/mpd (exists; writable; 154 GB free on this filesystem)
[skip] quota             showquota not found (it exists on HPRC clusters); check your quota yourself
[ok]   account           not set: jobs are charged to your default Slurm account (`myproject` lists accounts; set --account or MPD_ACCOUNT)
[FAIL] python            no Python environment at /scratch/user/<netid>/mpd/env. Run: bash scripts/setup_env.sh --cluster grace
[warn] data:proteingym   ProteinGym data not found under /scratch/user/<netid>/mpd/data/proteingym-v1.3. Run on a login node: mpd fetch proteingym --cluster grace
[warn] data:venusmuthub  VenusMutHub data not found under /scratch/user/<netid>/mpd/data/venusmuthub-v1. Run on a login node: mpd fetch venusmuthub --cluster grace
[warn] weights           none fetched yet. not fetched: esm2-8m, esm2-35m, esm2-150m, esm2-650m, esm2-3b, saprot-35m, saprot-650m (a plan needs only its teacher and student; `mpd models` has sizes)
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
3. needs Python 3.11 or newer (3.11, 3.12 and 3.13 are the tested ones; a newer one only gets a warning), then creates the venv `$MPD_HOME/env`;
4. runs `pip install -c constraints/grace.txt` for `mpd` with the training extra. The pinned set is torch 2.10.0, transformers
   4.51.0 and peft 0.13.2, the combination the test suite ran with on CPU. `--constraints none` installs the looser ranges of
   `pyproject.toml`, which is **not** the tested set;
5. writes `$MPD_HOME/env/mpd_env.json`, which records the modules, the interpreter and the package versions. The job scripts load the
   same modules, so there is no profile to edit **[CPU]**;
6. adds `MPD_HOME` and `MPD_CLUSTER` to `env/bin/activate` (so `source .../activate` is all a new shell needs), and runs `mpd doctor`.

pip's cache and `TMPDIR` are under `$MPD_HOME/cache`; nothing is written under `$HOME`. The wheels are about 4 GB to download (installing the pinned set
filled a pip cache with 4.0 GB of wheels; the download was not timed on Grace) and the result is about 28,000 files and 7.7 GB
(measured on the reference environment with the same package versions). If the login node ends the script after its hour, run it
again: the wheels are cached and it continues.

Other options: `--home DIR` (then `source DIR/env/bin/activate`), `--python PATH`, `--modules none`, `--no-torch` (planner only, for a machine that runs no job), `--dev`
(editable install with the test extra, for working on `mpd`), `--rebuild`, `--yes`. `bash scripts/setup_env.sh --help` lists them.

**After `git pull`**, run the script again, but only when no job of a campaign is queued or running (`mpd status` shows it): the script installs
the new `mpd` into the environment those jobs execute. A normal install is a snapshot taken from a copy of the sources, so pulling alone never changes
the code under a running campaign; `mpd doctor` warns when the installed `mpd` or `constraints/grace.txt` differs from what the
environment was built with **[CPU]**. A campaign directory is also fixed once it has been submitted: if a newer `mpd` would plan a different
recipe for the same campaign id, it refuses to mix the two ([below](#a-different-recipe-is-refused)).

Whether a venv built from a module's Python runs on a compute node with only the recorded modules loaded is **[NOT VERIFIED]**. The
only recipe with a record on Grace is a conda environment called by its absolute interpreter path **[earlier jobs]**; see
[Other ways to build the environment](#other-ways-to-build-the-environment).

### Data and weights

Compute nodes have no internet (below), so these run on a login node:

| command | download | on disk |
|---|---|---|
| `mpd fetch proteingym` | 62.4 MB (Zenodo record 15293562, checksum-verified; includes the AF2 structures SaProt reads) | 1.2 GB, 421 files |
| `mpd fetch venusmuthub` | 796 small files, fetched one at a time (Hugging Face rate-limits bursts) | 6.3 MB, 797 files |
| `mpd fetch model esm2-3b esm2-650m` | 13.98 GB (11.37 GB + 2.61 GB), resumable | the same, 12 files |
| `mpd fetch model --list` | what is pinned, its size, and what you already have | |
| `mpd fetch foldseek` (SaProt only) | 11.9 MB, from github.com | 29.4 MB |

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

The plan (**simulated**: the Grace profile on a machine that is not Grace, with a made-up account, ProteinGym fetched and nothing else; the
numbers are the planner's estimates, not measurements; the five `warning:` lines that follow the notes, each saying that a walltime or the GPU
memory is an estimate, are left out here):

```text
campaign   esm2-8m__esm2-8m__proteingym-score-all__seed20260922__cbeda1e7__6bebd418
directory  /scratch/user/<netid>/mpd/runs/esm2-8m__esm2-8m__proteingym-score-all__seed20260922__cbeda1e7__6bebd418
cluster    grace  gpu=a100  home=/scratch/user/<netid>/mpd
python     /scratch/user/<netid>/mpd/env/bin/python
teacher    esm2-8m (live)   student esm2-8m
train      proteingym  method=score-all  units=1
eval       proteingym
recipe     50 updates, global batch 64 (= 1 GPU x 2 micro x 32 accum), lr 3e-05, kl 0.0, support observed

stage          tasks  gpu       cpus   mem  walltime          SU (max)  after
prepare            1  -            8   32G  04:00:00                32  -
teacher            1  1xa100      24  180G  00:15:00 est            24  prepare
train              1  1xa100      24  180G  00:15:00 est            24  teacher
score-base         1  1xa100      24  180G  00:15:00 est            24  prepare
score              1  1xa100      24  180G  00:15:00 est            24  train
evaluate           1  -            4   16G  01:00:00                 4  score-base, score

walltime: est = estimate, not measured; meas = scaled from timed runs; set = given by you
estimated upper bound: 133 SU (every task runs to its requested walltime and the node share it asks for; the charge follows actual runtime). Estimate, not measured.
SU rule: (effective cores + GPU rate x GPUs) x hours; memory counts as cores. Grace's sbatch prints a "Requested SUs" line that counts CPUs only, so it is lower than the total above; `myproject` shows your balance.
campaign id: a new --eval-data, --gpu, --train-gpus or --score-shards (or MPD_DTYPE, MPD_TF32, --env MPD_STRUCTURE_LEGACY_MASK=1) is a new campaign and trains again from the start.
  Walltimes, --array-throttle, --sbatch-arg, --module, other --env pairs and --account keep the id: `mpd submit --force` applies them to the stages it queues again.

[ok]   home             /scratch/user/<netid>/mpd (exists; writable; 154 GB free on this filesystem)
[skip] quota            showquota not found (it exists on HPRC clusters); check your quota yourself
[ok]   account          123456789012 (not checked against `myproject`: command not found)
[FAIL] python           no Python environment at /scratch/user/<netid>/mpd/env. Run: bash scripts/setup_env.sh --cluster grace
[ok]   data:proteingym  /scratch/user/<netid>/mpd/data/proteingym-v1.3 (complete: FETCHED.json present)
[FAIL] weights esm2-8m  weights for esm2-8m are missing (/scratch/user/<netid>/mpd/models/esm2-8m)
                        fix: mpd fetch model esm2-8m --cluster grace
[ok]   cost             estimated upper bound 133 SU (an estimate, not measured)

NOT READY: 2 errors, 0 warnings. Plan and --dry-run only report these (exit 0); `mpd submit` stops at the first error.
```

133 SU is 0.7% of a 20,000 SU allocation. It is an upper bound: 15 minutes is the floor of the walltime formula (a 10-minute start-up
allowance plus rounding), not a prediction, and the real charge follows the run time. `prepare` is a CPU job that asks for 04:00:00, 8 CPUs and
32 GB whatever the recipe, which accounts for 32 of the 133 SU; on the real ProteinGym and VenusMutHub files its work (tables and, for a SaProt run, Foldseek) took 20 to 23
seconds on a loaded login node of another cluster **[CPU]** (not on Grace, not through Slurm).

`--dry-run` writes the scripts under `<campaign>/scripts/` and runs the checks above, which on a Slurm login node include
`sbatch --test-only` once for each distinct job shape (read-only: it asks Slurm whether it would accept the request and queues nothing). Without
`sbatch` on the `PATH` the dry run reports (**simulated**, a machine that is not Grace; neither line counts as an error):

```text
[skip] slurm               sbatch not found on PATH: this is not a Slurm login node (plan and --dry-run work anywhere; submit needs it)
[skip] sbatch --test-only  sbatch is not on PATH (not a Slurm login node): the generated scripts were not test-submitted
```

`mpd submit` without `--dry-run` stops on such a machine with `[FAIL] slurm` and `sbatch not found on PATH: submit from a Slurm login node`.

What to look at afterwards:

- `mpd status` shows each stage's Slurm state; `mpd logs <campaign> prepare` shows a stage's output (`MPD_JOB_START host=... job=...`
  is the first line of every job). A job with an empty log failed before that line: read its `.err` file with `--err`.
- `mpd results <campaign>` after the `evaluate` stage. Because the teacher is the student, its development loss starts at (about)
  zero and `delta_vs_base` is about zero (on CPU in float32 the loss is exactly zero **[CPU]**; on a GPU in bf16 it will not be). The run
  checks the plumbing and costs little; it is not an experiment. Use `--teacher esm2-35m` (136 MB more to fetch) to see something move.
- No `mpd` job has ever run on a GPU. The first `import torch` took 88 to 127 s, and once more than 3 minutes, in the author's earlier Grace
  jobs **[earlier jobs]**; the 10-minute start-up allowance is meant to cover that **[NOT VERIFIED]**.

## When a job does not finish

The rule is one sentence: **run the `mpd submit` command you used the first time, change only the option the problem calls for, and add
`--force`.** `--force` means "continue or repair". Stages that are queued, running or finished are kept; the stages that died, and everything
that waited for them, are queued again, with the walltimes and sbatch options of the command you typed last. Training continues from its last
save (every `MPD_CKPT_EVERY` updates, 200 unless you changed it). `--force` never queues a second copy of a chain that is still alive; for an
independent run change `--seed`. Everything in this section was run against a fake Slurm **[CPU]**; none of it has been seen on Grace.

### What starts a new campaign and what does not

A campaign is a folder `<home>/runs/<campaign id>/` with its stages, checkpoints and results. The id ends in eight hex digits that depend on:

| Changes the id (a new campaign: new folder, everything is computed again) | Keeps the id (the same campaign; `--force` applies it to the stages it queues again) |
|---|---|
| the recipe: teacher, student, data, method, assays, `--seed`, `--updates`, `--global-batch`, `--lr`, `--kl-weight`, `--dev-fraction` | `--train-time`, `--score-time`, `--teacher-time` |
| `--eval-data` | `--array-throttle` |
| `--cluster`, `--gpu`, `--train-gpus` | `--sbatch-arg` (for example `--sbatch-arg=--exclude=g007`), `--module` |
| `--score-shards` | `--env` pairs other than the three on the left |
| `MPD_DTYPE`, `MPD_TF32` (from the shell or `--env`), `--env MPD_STRUCTURE_LEGACY_MASK=1` | `--account`, `--python` |

`--home` is not part of the id, but a different home is a different folder: nothing is shared between two homes. `mpd plan` and `mpd submit`
print this rule under the stage table (the two `campaign id:` lines above). The reason for the split: after a TIMEOUT the natural fix is a longer
`--train-time`, and after a bad node it is `--sbatch-arg=--exclude=NODE`; both must find the finished stages and the training checkpoints again.
A stage that is still queued or running keeps the script it was submitted with, because Slurm cannot change a queued job; `mpd submit --force`
says so. To change a job that is still waiting, `mpd cancel <campaign>` and submit again.

### Recipes

The commands of the recipes below, in one place. `$TEACHER`, `$STUDENT` and `$RECIPE` stand for the choices of your first `mpd submit` command (the example here is the
one in the [README](../README.md#run-a-distillation)); the first line is the command you ran the first time.

```bash
TEACHER=esm2-3b STUDENT=esm2-650m
RECIPE="--train-data proteingym --method score-all --eval-data proteingym venusmuthub"

mpd submit --teacher $TEACHER --student $STUDENT $RECIPE
mpd submit --teacher $TEACHER --student $STUDENT $RECIPE --train-time 16:00:00 --force                # a TIMEOUT
mpd submit --teacher $TEACHER --student $STUDENT $RECIPE --sbatch-arg=--exclude=g007 --force         # a node without a working GPU
mpd submit --teacher $TEACHER --student $STUDENT $RECIPE --force                                      # an interrupted submission, a NODE_FAIL, a failed stage
mpd cancel <campaign>
mpd submit --teacher $TEACHER --student $STUDENT $RECIPE                                              # after the cancel: no --force
mpd submit --teacher $TEACHER --student $STUDENT $RECIPE --seed 2                                     # an independent run
```

Each recipe starts from what `mpd status <campaign>` prints under `problems:`.

| What `mpd status` shows | Do this |
|---|---|
| `train ... TIMEOUT`, "reached its time limit" | the original command with a larger `--train-time` (it suggests twice the old limit, never above what the partition allows) and `--force`. Training continues from its last save. If it already asked for the partition maximum, run the same command with `--force` again: every attempt continues from the last save. A `score` or `teacher` timeout names `--score-time` or `--teacher-time`; the teacher starts from the beginning because it saves its table only at the end. `prepare` and `evaluate` have no time option. |
| `FAILED`, exit code 31, `MPD_NO_GPU` in the log | the node has no usable GPU. The original command plus `--sbatch-arg=--exclude=<node>` (the node is named in the message) and `--force`. |
| `OUT_OF_MEMORY`, or `FAILED` with exit 137, or `CUDA out of memory` in the log | the log decides: a CUDA message means the GPU is too small for this model at this size, so use a smaller student or a larger GPU (`--gpu`; that is a new campaign). More GPUs do not help: training is data-parallel and every GPU holds a full copy of the model. Without a CUDA message the node ran out of host memory: add `--sbatch-arg=--mem=<more>G` and `--force`. mpd's memory figures are estimates, not measurements. |
| `NODE_FAIL` | the node failed, not your settings: the original command with `--force`; if it lands on the same node, add `--sbatch-arg=--exclude=<node>`. |
| `FAILED`, `MPD_MODULE_LOAD_FAILED` in the log | a module of the environment record could not be loaded. `mpd doctor` (row `env-record`), then `bash scripts/setup_env.sh --cluster grace` writes the record again; then the original command with `--force`. |
| `FAILED`, `weights for ... are missing`, or `huggingface.co` / offline mode in the log | the weights are not on disk and compute nodes are offline. On a login node run the `mpd fetch model ...` command the message prints; then the original command with `--force`. |
| `CANCELLED` with reason `DependencyNeverSatisfied` | a stage before it failed. Fix that one first; then the original command with `--force`. |
| `submit` ended with `Slurm did not answer ...`, state `partial` | the controller went away while the chain was being queued. When `squeue -u "$USER"` answers again, run the same command with `--force`: jobs that were queued (even one that sbatch accepted while reporting an error) are found by their tag and kept, the rest are queued. No duplicates. |
| you ran `mpd cancel`, or the last submission ended `failed` | the same command, with any option you want to change; no `--force` needed. Stages whose output files exist are not queued again. |
| a stage ran, but you want it to run again | delete its marker file and repeat the command with `--force` ([below](#running-a-finished-stage-again)). |

All the words `mpd status` uses, state by state and reason by reason, are in [TROUBLESHOOTING.md](TROUBLESHOOTING.md#what-mpd-status-says).

### A TIMEOUT

Simulated (the real CLI against a fake Slurm; job ids, nodes and times invented; nothing ran on Grace). `train` asked for 8 hours and
hit the limit; what waited for it was cancelled by Slurm. Without `--force` the same command is refused and nothing is written; with it, only
the missing stages are queued, and `train` and the stages behind it carry the new time. In the `mpd submit` transcripts of this section the plan, the checks
and the line `scripts written to <campaign>/scripts`, which come before the queueing lines, are left out:

```text
$ mpd status <campaign>
campaign   esm2-3b__esm2-650m__proteingym-score-all__seed20260922__054935cb
directory  /scratch/user/<netid>/mpd/runs/esm2-3b__esm2-650m__proteingym-score-all__seed20260922__054935cb
state      submitted   (submitted 2026-11-09T07:09:27-0600)

stage       state      elapsed   reason                    job   exit  tasks  note
prepare     COMPLETED  00:03:11  -                         1001  0:0   1/1
teacher     COMPLETED  00:40:12  -                         1002  0:0   1/1
train       TIMEOUT    08:00:12  -                         1003  0:15  0/1
score-base  COMPLETED  00:25:00  -                         1004  0:0   4/4    4 COMPLETED
score       CANCELLED  00:30:00  DependencyNeverSatisfied  1005  0:0   0/4    4 CANCELLED
evaluate    CANCELLED  00:30:00  DependencyNeverSatisfied  1006  0:0   0/1

problems:
  train (job 1003) reached its time limit: it ran 08:00:12 of the 08:00:00 it asked for.
    Training saves its progress every MPD_CKPT_EVERY updates (200 unless you changed it) and continues from the latest
      save, so the work done so far is kept.
    Run your original `mpd submit ...` command again with a larger --train-time (for example --train-time 16:00:00; this
      partition allows up to 4-00:00:00) and --force.
  last lines of the error log of train (job 1003):
    step 1500 loss 0.31
    slurmstepd: error: *** JOB 1003 ON g003 CANCELLED AT 2026-11-09T12:05:00 DUE TO TIME LIMIT ***
  score (job 1005) was cancelled because a stage it waited for did not finish.
  evaluate (job 1006) was cancelled because a stage it waited for did not finish.

next:
  mpd logs esm2-3b__esm2-650m__proteingym-score-all__seed20260922__054935cb train --err    the whole error log
  docs/TROUBLESHOOTING.md explains the usual failures
  repeat your mpd submit command with --force to continue: stages that are queued, running or finished are kept, the others are queued again, with any --train-time, --sbatch-arg or similar option you changed

$ mpd submit <original options> --train-time 16:00:00 --force
finished  prepare      (its output files exist)
finished  teacher      (its output files exist)
submitted train        job 1007
finished  score-base   (its output files exist)
submitted score        job 1008
submitted evaluate     job 1009

submitted 3 job(s) for esm2-3b__esm2-650m__proteingym-score-all__seed20260922__054935cb
  follow it  mpd status esm2-3b__esm2-650m__proteingym-score-all__seed20260922__054935cb
  read logs  mpd logs esm2-3b__esm2-650m__proteingym-score-all__seed20260922__054935cb [STAGE] [--err]
  cancel     mpd cancel esm2-3b__esm2-650m__proteingym-score-all__seed20260922__054935cb
  results    mpd results esm2-3b__esm2-650m__proteingym-score-all__seed20260922__054935cb   (after the evaluate stage)
```

The refusal without `--force` (exit status 2) reads:

```text
error: campaign esm2-3b__esm2-650m__proteingym-score-all__seed20260922__054935cb was already submitted (jobs: prepare 1001, teacher 1002, train 1003, score-base 1004, score 1005, evaluate 1006). Nothing was written.
  mpd status esm2-3b__esm2-650m__proteingym-score-all__seed20260922__054935cb    follow it
  mpd cancel esm2-3b__esm2-650m__proteingym-score-all__seed20260922__054935cb    stop its queued and running jobs; after that the same command submits it again (finished stages are kept)
  add --force to this command to continue or repair it: stages that are queued, running or finished are kept and the others are queued
      A changed --train-time, --score-time, --teacher-time, --array-throttle, --sbatch-arg, --module
      or --env (except MPD_DTYPE, MPD_TF32 and MPD_STRUCTURE_LEGACY_MASK) keeps the campaign (its id
      does not depend on them), and --force applies it to the stages it queues again; a stage that
      is still queued or running keeps the options it was submitted with. A changed --gpu,
      --train-gpus, --eval-data or --score-shards is a different campaign that trains again from the
      start.
  or change --seed for an independent run
```

### A bad node

In May 2026 one Grace node listed its A100 in `nvidia-smi` while `torch.cuda.is_available()` was false **[earlier jobs]**. A GPU stage now stops with
exit status 31 and `MPD_NO_GPU: ...` instead of running on the CPU **[CPU]**. `mpd status` names the node and the exact option (simulated):

```text
problems:
  train (job 1003) failed with exit code 31.
    The node has no usable GPU (the job stopped instead of running on the CPU). Run your original `mpd submit ...`
      command again with --sbatch-arg=--exclude=g007 and --force.
  last lines of the error log of train (job 1003):
    MPD_NO_GPU: no usable GPU on g007; resubmit with --sbatch-arg=--exclude=g007

$ mpd submit <original options> --sbatch-arg=--exclude=g007 --force
finished  prepare      (its output files exist)
finished  teacher      (its output files exist)
submitted train        job 1007
kept      score-base   job 1004 (already queued or running)
submitted score        job 1008
submitted evaluate     job 1009

submitted 3 job(s) for esm2-3b__esm2-650m__proteingym-score-all__seed20260922__054935cb
note: kept stages (score-base) stay queued or running with the options they were submitted with; a changed walltime, --sbatch-arg or --module applies to the stages queued now. To change a job that is still waiting, `mpd cancel` the campaign and submit it again
```

The `kept` stage is still running and is not touched. `--sbatch-arg` takes an equals sign (`--sbatch-arg=--exclude=g007`) because the value starts with
`--`. Extra sbatch options keep the campaign, so the finished `prepare` and `teacher` are not computed again **[CPU]**.

### A submission that was interrupted

Grace's controller went away for minutes to hours on 2026-06-11, 06-26, 07-24 and 07-26, and on 07-26 `sbatch` printed
`Unable to contact slurm controller` although it had accepted the whole chain **[earlier jobs]**. So a failed `sbatch` is not proof that nothing was
queued. `mpd submit` gives every job a tag in its `--comment`, writes `SUBMITTED.json` after each accepted job, waits and looks in `squeue -u "$USER"`
for a job with this submission's tag before it tries again (up to 6 tries and about 15 minutes), and cancels what it queued if it gives up. Simulated:
the controller stops answering after the second job and `squeue` and `scancel` are down too.

```text
$ mpd submit <options>
submitted prepare      job 1001
submitted teacher      job 1002
Slurm did not answer for stage train (Unable to contact slurm controller (connect failure)); waiting 60 s, then checking the queue before trying again (try 1 of 6)
...
Slurm did not answer for stage train (Unable to contact slurm controller (connect failure)); waiting 172 s, then checking the queue before trying again (try 6 of 6)
error: sbatch could not queue stage train: Slurm did not answer after 7 attempt(s) over 900 s (last message: Unable to contact slurm controller (connect failure)). The job may have been accepted anyway; the queue could not be read to check.
Could not cancel the jobs already queued (Unable to contact slurm controller (connect failure)). Run `scancel 1001 1002` when Slurm answers, or `mpd cancel <campaign>`; their ids are saved in <campaign>/SUBMITTED.json.
Stage train may have been accepted although sbatch reported an error (the queue could not be read to check). `mpd cancel <campaign>` finds and cancels it by its tag once Slurm answers.
Run the same command with --force to continue from the first missing stage, or `mpd cancel <campaign>`.

$ mpd submit <options> --force          (after the outage; a job that sbatch had accepted while reporting an error is found by its tag)
kept      prepare      job 1001 (already queued or running)
kept      teacher      job 1002 (already queued or running)
kept      train        job 1003 (already queued or running)
submitted score-base   job 1004
submitted score        job 1005
submitted evaluate     job 1006
```

Run `mpd status <campaign>` before anything else: the message says whether a stage may have been accepted anyway. Do not delete `SUBMITTED.json`. If
`mpd status` itself prints `slurm      no job states: ...`, the controller is not answering yet; wait and ask again.

### After `mpd cancel`, or a submission that ended `failed`

`mpd cancel <campaign>` removes the queued and running jobs and marks the campaign `cancelled`. A submission that Slurm refused ends `failed` (what that
run had queued is cancelled). In both cases the same `mpd submit` command, with or without `--force`, queues **only the stages whose output files are
missing**; a stage whose output files exist is printed as `finished  <stage>  (its output files exist)` and not queued, and a job that is still alive is
kept, never duplicated. Simulated: `prepare` and `teacher` had finished, the student cancelled the rest, then repeated the command with a different `--train-time`:

```text
$ mpd cancel <campaign>
cancelled 4 job(s) of esm2-3b__esm2-650m__proteingym-score-all__seed20260922__054935cb: 1003 1004 1005 1006

$ mpd submit <options> --train-time 12:00:00
...                      (the plan and the checks, as for the first submit)

scripts written to <home>/runs/esm2-3b__esm2-650m__proteingym-score-all__seed20260922__054935cb/scripts
the earlier submission of this campaign cancelled (cancelled 4 job(s) with mpd cancel at 2026-11-09T07:09:33-0600); submitting it again
finished  prepare      (its output files exist)
finished  teacher      (its output files exist)
submitted train        job 1007
submitted score-base   job 1008
submitted score        job 1009
submitted evaluate     job 1010
```

#### A different recipe is refused

A campaign directory's recipe files (`spec.json`, `resolved.json`, `tasks/*.json`) are fixed once it has a `SUBMITTED.json`, in any state. If a command would
write a different recipe into it (the usual cause is a newer `mpd` with other presets or registry), the command is refused and the message names the file and
the keys that differ (simulated: `resolved.json` was edited by hand to stand for another version):

```text
error: <home>/runs/<campaign> holds a campaign that ended as cancelled and has a different recipe (resolved.json), so it cannot be submitted again; its finished stages would be mixed with the new recipe:
  resolved.json: microbatch: 1 on disk, 2 now
A submitted campaign is never rewritten. Its recipe follows from the options in its id (--eval-data, --gpu, --train-gpus, --score-shards, the recipe flags and --seed) and from the mpd version, presets and model registry that wrote it, so the usual cause is another mpd version or an edited file. Run the command with the mpd version that created the campaign, or change --seed to start a new campaign, or remove the directory (`mpd cancel` its jobs first).
```

### Running a finished stage again

A stage counts as finished when its output files exist. To run one again, delete its marker file and repeat the command with `--force`. The stage is queued again, with every
unfinished stage that waits on it (a queued job that waited for the old one is cancelled and queued again); stages that are finished or still alive are kept. The
stage messages say the same (`grep -rn "COMPLETE.json" src/mpd/stages`).

| stage | marker to delete |
|---|---|
| `prepare` | `contexts/MANIFEST.json` |
| `teacher` | `teacher/COMPLETE.json` |
| `train` | `train/<unit>/COMPLETE.json` |
| `score-base`, `score` | the `score/<dataset>/<unit>/shard_*.json` files |
| `evaluate` | `evaluate/summary.json` |

A stage whose own output exists is never queued, even when an earlier stage runs again: delete the markers of every stage you want to run again. The tests delete
the markers of `prepare` and `teacher`; the other rows follow the same rule in the code but were not exercised. Simulated, with everything finished and
`teacher/COMPLETE.json` deleted:

```text
$ mpd submit <original options> --force
finished  prepare      (its output files exist)
submitted teacher      job 1007
finished  train        (its output files exist)
finished  score-base   (its output files exist)
finished  score        (its output files exist)
finished  evaluate     (its output files exist)

submitted 1 job(s) for esm2-3b__esm2-650m__proteingym-score-all__seed20260922__054935cb
```

### Reading the table of `mpd status`

One line per stage. The first four fields are single words and stay in this order:

```text
stage  state  elapsed  reason  job  exit  tasks  note
```

`state` is Slurm's (`sacct`), `elapsed` its run time, `reason` the reason `squeue` still has for the job (`Priority`, `Dependency`, `TimeLimit`, ...; `-` when there is none), `job` the
Slurm id, `exit` the exit status and signal, `tasks` how many tasks of the stage wrote their output files, `note` the per-state counts of an array. Sections after the table:
`problems:` (a stage that failed, timed out or was cancelled, in plain words with the exact command), `waiting:` (one line for each stage that has not started), `next:`. A line
`earlier ...` lists the job ids that `--force` replaced; their logs stay in `<campaign>/logs/`. If `sacct` and `squeue` are missing or silent the state column is `-` and there
is no advice: `mpd` says so instead of guessing.

### Not verified

All of this was built and tested against a fake Slurm on CPU **[CPU]**. Not verified on Grace: the wording of `sacct` and `squeue` there (the `Timelimit` and `NodeList` columns; if `sacct` refuses them
`mpd` asks again without them), the `ExitCode` of a job that hit its limit or was killed, whether `squeue -t all` still lists a finished job (`MinJobAge`) and what reason it keeps; that Grace
honours `--kill-on-invalid-dep=yes` and keeps the `--comment` tag `mpd` uses to recognise its jobs; that a training run that hit its limit really continues from its save on a GPU node; the partition
maximum printed in the TIMEOUT advice is the figure in the cluster profile, taken from HPRC documents, not read from the cluster.

## What mpd asks Slurm for

| stage | GPUs | cores / memory | default walltime |
|---|---|---|---|
| `prepare` | none | 8 / 32 GB | 4 h, fixed (an upper bound: see above) |
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
DDP shape (tested on CPU with gloo: two processes equal one for the `direct-*` methods **[CPU]**; for `score-*` the loss is the average over
micro-batches of the per-micro-batch average over variants, so the 1-GPU and 2-GPU runs follow slightly different paths, see
[the note below](#micro-batch-and-the-score-loss)); more GPUs than a node holds is an error rather than a silent clamp **[CPU]**.

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
- **A bad node** is a recovery recipe: [above](#a-bad-node).

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

#### Micro-batch and the score loss

The global batch stays 64 whatever the micro-batch. For `direct-*` methods that makes a run independent of the micro-batch and of the number of
GPUs. For `score-*` methods the loss of an update is the average of the loss of each micro-batch, and the loss of a micro-batch is the average
over the variants it holds. A position with few measured variants therefore weighs more at micro-batch 1 than at micro-batch 2 (measured on
the ProteinGym score table: contexts with at most five variants carry 9.4% of the loss at micro-batch 1 and 4.1% at micro-batch 2). Compare
`score-*` results across students or GPU types with that in mind; the `esm2-3b` student on a 40 GB A100 trains at micro-batch 1, the others at 2.

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
#SBATCH --output=<campaign>/logs/%x_%j.out
#SBATCH --error=<campaign>/logs/%x_%j.err
#SBATCH --open-mode=append

set -eo pipefail
type module >/dev/null 2>&1 || source /etc/profile 2>/dev/null || true
type module >/dev/null 2>&1 || { echo "MPD_MODULE_LOAD_FAILED the module command is not available (Anaconda3/2024.02-1)" >&2; exit 1; }
module purge
module load Anaconda3/2024.02-1 || { echo "MPD_MODULE_LOAD_FAILED Anaconda3/2024.02-1" >&2; exit 1; }
for _v in $(compgen -e); do case $_v in MPD_*) unset "$_v";; esac; done; unset _v PYTHONPATH PYTHONHOME
set -u
export MPD_HOME=... MPD_CLUSTER=grace HF_HOME=... XDG_CACHE_HOME=... TORCH_HOME=... TRITON_CACHE_DIR=... MPLCONFIGDIR=...
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=24
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MPD_REQUIRE_CUDA=1
nvidia-smi -L || { echo "MPD_NO_GPU: no usable GPU on $(hostname -s); resubmit with --sbatch-arg=--exclude=$(hostname -s)" >&2; exit 31; }
echo "MPD_JOB_START host=$(hostname) job=${SLURM_JOB_ID:-} stage=train"
cd <campaign>

<python> -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 -m mpd.stages.train --campaign-dir <campaign> --stage train --task-index 0
echo "MPD_STAGE_DONE train"
```

- **Part of the environment is cleaned, the rest is inherited.** A job starts with the environment of the shell that ran `sbatch` (Slurm's
  default). The script runs `module purge` (when `module` exists), loads only the recorded modules, and drops `PYTHONPATH`, `PYTHONHOME`
  and every `MPD_*` variable of your shell, then exports the ones the plan chose (`MPD_ALLOW_HUB` or `MPD_STRUCTURE_LEGACY_MASK` in your
  shell therefore never reach a job; the second only through `--env`). `PATH`, `LD_LIBRARY_PATH`, `VIRTUAL_ENV`, `CONDA_*` and whatever
  else you exported do arrive: the Grace job scripts that ran did the same **[earlier jobs]**. If a job fails to import `torch` with a
  library error and your shell has an old `LD_LIBRARY_PATH`, `unset` it before `mpd submit`. With no module recorded the three module
  lines become `if type module >/dev/null 2>&1; then module purge; fi`.
- **There is no `--export` and no `--get-user-env`.** Slurm 25.05 rejects `--get-user-env=L` ("doesn't allow an argument") **[Vision]**, and
  `--export=NONE` implies `--get-user-env`; none of the author's Grace job scripts (about a thousand) used either flag **[earlier jobs]**. What
  Grace's Slurm does with them is **[NOT VERIFIED]**, so `mpd` leaves them out.
- **`set -u` comes after the modules.** `/etc/bashrc` and the module scripts read unset variables; `set -euo pipefail` before them gave
  `BASHRCSOURCED: unbound variable` on Grace **[earlier jobs]**.
- **Everything writable is under `$MPD_HOME`**: the caches, and `HF_HOME` for stray Hugging Face writes. `PYTHONNOUSERSITE=1` keeps
  `~/.local` out of the job.
- **Offline.** `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` (above). These shell variables are copied into the script as an `export` when you submit:
  `MPD_MODEL_<KEY>`, `MPD_MODELS_DIR`, `MPD_FOLDSEEK` (always as absolute paths), `MPD_3DI_ENGINE`, `MPD_DTYPE`, `MPD_TF32`, `MPD_KEEP_ALL_CHECKPOINTS` and
  `MPD_CKPT_EVERY`; so is every `--env NAME=VALUE`. `MPD_STRUCTURE_LEGACY_MASK` reaches a job only through `--env`, never from the shell (a shell that exports it
  gets a plan warning), and a relative path in `--env MPD_FOLDSEEK`, `MPD_MODELS_DIR` or `MPD_MODEL_<KEY>` is a plan error **[CPU]**.
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
  (`esm2-3b` to `esm2-650m`, `score-all`, evaluated on both benchmarks) it is **1,806 SU, 9% of 20,000** (**simulated** plan: teacher 03:00:00,
  train 10:15:00, score 00:30:00; **[HPRC]** for the allocation). With `esm2-3b` as the student (teacher `esm2-650m`) the plan is 4,983 SU, 25%, with 1-16:00:00 of training at
  micro-batch 1 (**simulated**).
- `*-one` methods train one student per assay, so the training line is multiplied by the number of assays: `score-one` over all 217
  ProteinGym assays, evaluated on ProteinGym only, plans to **184,700 SU**, about nine times the Basic allocation (**simulated**). `mpd submit` refuses a plan above
  `--max-su` (default 10,000) and says why; `plan` and `--dry-run` only warn. Use `*-one` for a few assays.
- Changing only `--eval-data`, `--gpu`, `--train-gpus`, `--cluster`, `--score-shards`, `MPD_DTYPE` or `MPD_TF32` makes a **new campaign**, which
  retrains. Walltime overrides, `--array-throttle`, `--sbatch-arg`, `--module`, other `--env` pairs, `--account`, `--home` and `--python` do not
  ([recipes](#what-starts-a-new-campaign-and-what-does-not)) **[CPU]**.

## SaProt on Grace

SaProt-35M and SaProt-650M are `available` in `mpd models`, with these labels wherever the status is stated: **tested on CPU only** (tiny random models, a fake
Foldseek, one by-hand `prepare` run on real data); nothing has run on Grace or on a GPU; the pinned Foldseek release 10-941cd33 was never executed;
SaProt on VenusMutHub needs the structure deposit that is not published yet (`mpd fetch venus-structures --from-dir`); bf16, GPU memory and walltime
for SaProt are estimates. [SAPROT.md](SAPROT.md) has the details.

- On a login node: `mpd fetch proteingym` (62.4 MB; it includes the AF2 structures SaProt reads), `mpd fetch foldseek` (about 12 MB, from github.com),
  `mpd fetch model saprot-650m` (2.6 GB). Compute nodes have no internet. `mpd plan` and `mpd doctor` print what is still missing, with the command for each. If a login node cannot reach github.com, download `foldseek-linux-avx2.tar.gz` of release 10-941cd33 elsewhere and run `mpd fetch foldseek --from-file <archive>` (checked against the pinned sha256 before anything runs); `MPD_FOLDSEEK=/path/to/foldseek` uses a Foldseek you already have.
- The 3Di strings are built by the `prepare` stage, a CPU job (8 CPUs, 32 GB, a fixed 4 hour request; no GPU service units). On the real ProteinGym and
  VenusMutHub files (675 structures) `prepare` took 23 seconds on a loaded login node of another cluster, CPU only, with the earlier
  study's Foldseek build **[CPU]**; not measured on Grace, not run through Slurm.
- `prepare` also checks that the weights of every model in the run are fetched, so a purged or never-fetched weights folder stops the CPU job within
  minutes instead of the GPU job hours later.
- Foldseek is a static `linux-avx2` binary. HPRC's Grace primer (Fall 2025) says the 940 nodes are based on the Intel Cascade Lake processor, which has
  AVX2; the CPU of each partition is not documented there. **[NOT VERIFIED]** Run `mpd doctor` once; it runs `foldseek version`. A `noexec` mount would
  stop it from running.
- SaProt jobs ask for the same GPU shapes as the ESM2 model of the same size. The memory and walltime figures for SaProt are the ESM2 ones extrapolated
  by size, marked as estimates, and have not been measured.
- A teacher and a student from different families (SaProt with ESM2, either way): prefer `score-*`. `direct-*` trains the student's raw logits toward
  the teacher's raw logits, which carry a per-position offset that differs between families; `score-*` subtracts it. Reasoned and unit-tested on random
  numbers (`tests/test_saprot_offsets.py`), never measured on a real pair.

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
12. how long the first `import torch` takes, and whether the 10-minute start-up allowance covers it plus a 3B model load;
13. everything under [Not verified](#not-verified) in the recovery section: `sacct` and `squeue` output on Grace, `--kill-on-invalid-dep`, the `--comment` tag, resume after a TIMEOUT on a GPU;
14. for SaProt: that Foldseek runs on Grace (AVX2 on every partition, no `noexec` mount), that release 10-941cd33 gives the same 3Di strings as the build behind the
    recorded SaProt numbers, and everything about SaProt on a GPU.

To settle most of these once, run on a Grace login node (all are read-only; `sbatch --test-only` queues nothing):

```bash
sbatch --version; scontrol show config | grep -E 'SLURM_VERSION|MaxArraySize|MaxJobCount|SchedulerParameters|KillOnInvalidDep'
sinfo -o '%P %l %D %G' | sort
gpuinfo; gpuavail; maxconfig -f <a rendered .sbatch from mpd submit --dry-run>
sacctmgr -n -P show qos format=Name,MaxJobsPU,MaxSubmitPU,MaxTRESPU,MaxWall
sacctmgr -n -P show assoc user=$USER format=Account,Partition,QOS,MaxJobs,MaxSubmit,GrpTRES
showquota; myproject; module spider Python; module spider Anaconda3; module spider PyTorch
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
