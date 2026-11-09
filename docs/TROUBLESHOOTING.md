# Troubleshooting

Start with `mpd doctor`. It checks the environment, data, weights, quota, account and Slurm tools, and prints the command that fixes
each problem under the `[FAIL]` and `[warn]` lines. `mpd plan` and `mpd submit --dry-run` print the same table for the run you
described. A `READY` footer means the static checks passed; it does not prove that a job will run on the cluster
([what has not been verified](GRACE.md#what-has-not-been-verified-on-grace)).

Tags as in [GRACE.md](GRACE.md): **[CPU]** tested on CPU here, **[Vision]** accepted by `sbatch --test-only` on Vision, **[HPRC]** from an
HPRC document, **[earlier jobs]** seen in the profile author's Grace jobs before `mpd`, **[NOT VERIFIED]** nobody has run it. The
messages below are the ones the code prints (`<x>` marks a part that varies); nothing here was seen on Grace.

Contents: [Setting up](#setting-up) | [Fetching data and weights](#fetching-data-and-weights) |
[Planning and submitting](#planning-and-submitting) | [In the queue](#in-the-queue) | [A job failed or stopped](#a-job-failed-or-stopped) |
[Cancelling, logs and results](#cancelling-logs-and-results) | [SaProt and Foldseek](#saprot-and-foldseek) | [What to paste in an issue](#what-to-paste-in-an-issue)

## Setting up

| you see | it means | do |
|---|---|---|
| `mpd: command not found` | the environment is not active in this shell | `source $SCRATCH/mpd/env/bin/activate`, or run `$SCRATCH/mpd/env/bin/python -m mpd ...` |
| `error: --cluster is required (or set MPD_CLUSTER). Available: grace, vision` | `MPD_CLUSTER` is not set; `activate` sets it | activate the environment, or add `--cluster grace` |
| `grace: storage root '{env:SCRATCH}/mpd' needs $SCRATCH, which is not set. Pass --home ... or export MPD_HOME.` | `$SCRATCH` is empty in this shell | `export MPD_HOME=/scratch/user/<netid>/mpd` |
| `<path> is under /home (10 GB / 10,000 files on Grace). Use scratch: export MPD_HOME=$SCRATCH/mpd` | the working root would fill `$HOME`, whose quota can lock you out | use a path under `$SCRATCH` |
| `no Python environment at <home>/env. Run: bash scripts/setup_env.sh --cluster grace` | the venv does not exist (or `MPD_PYTHON` points elsewhere) | run the command shown |
| `setup: error: module Anaconda3/2024.02-1 did not load. On Grace look for a Python module with: module spider Python ...` | the default module is not on this system; it was never checked on Grace again (**[NOT VERIFIED]**) | `module spider Python`, then `bash scripts/setup_env.sh --cluster grace --modules "<module names>"`; or `--modules none --python /path/to/python3` |
| `setup: error: ... is Python 3.10...; mpd needs 3.11 or newer` | the module's Python is too old | load a newer Python module |
| `setup_env.sh` ends without finishing after about an hour | the login-node limit (no process longer than an hour) **[HPRC]** | run the same command again: the wheels are cached in `<home>/cache/pip` and it continues |
| `Disk quota exceeded` during pip, or `OSError: [Errno 122]` | `$SCRATCH` is out of space or files (1 TB, 250,000) | `showquota`; delete old runs, caches and unused models; then run the command again |
| `packages ... torch not installed` right after `--no-torch` | expected: the planner needs no torch | none; a job environment is built without `--no-torch` |
| `the jobs use mpd X but this command is mpd Y`, or a warning that `constraints/grace.txt` changed | you pulled new code; the environment is a snapshot | `bash scripts/setup_env.sh --cluster grace` again |
| `ignoring the environment record: ...; re-run bash scripts/setup_env.sh --cluster grace` (a plan warning), or `env-record ... it was written for cluster 'x', not 'y', so the planner ignores it` (`mpd doctor`) | `<home>/env/mpd_env.json` was written for another cluster or another home, so the jobs would not load the recorded modules | run `setup_env.sh` again for this cluster and home |

## Fetching data and weights

Run these on a login node: compute nodes have no internet.

| you see | it means | do |
|---|---|---|
| `ProteinGym data not found under <path>. Run on a login node: mpd fetch proteingym --cluster grace` | not fetched yet (or `--home` differs from the one you fetched into) | run the command shown |
| `VenusMutHub data not found under <path>. Run on a login node: mpd fetch venusmuthub --cluster grace` | same | same; it fetches one file at a time and takes a few minutes |
| `weights for esm2-3b are missing (<path>). Run on a login node: mpd fetch model esm2-3b --cluster grace` | the weights are not there, or were deleted after submission (**scratch purge rules are [NOT VERIFIED]**) | run the command, then resubmit |
| `HTTP Error 429` or `503` from Hugging Face | rate limiting; the fetch already waited the `Retry-After` time | run the same command again later: it resumes. Do not set `HF_TOKEN`; `mpd` never sends one. If it keeps failing, fetch elsewhere and copy ([MODELS.md](MODELS.md#no-internet-on-the-cluster): `rsync`, or `--from-dir`) |
| `stopped before the login-node time limit` (exit status 3) | the fetch used its 55-minute budget **[CPU]** | run the same command again; it continues from the `.part` files |
| `could not download <url>: ...` with a pointer to `--from-dir` | this machine cannot reach the Hub | fetch on a machine that can and copy the folder (MODELS.md) |
| `not enough free space under ...`, `disk full or quota exceeded while writing ...` | free space is below the missing bytes plus 1 GiB, or a quota was hit | `showquota`; another `--home` on `$SCRATCH`; delete old runs |
| `sha256 mismatch; removed the file` | the bytes differ from the pin | run again; if it repeats, open an issue with the message |
| `another mpd fetch model ... is already writing` | a fetch of that model is running (maybe in another session) | wait for it |
| `ValueError ... torch.load ... upgrade torch to at least v2.6` while loading `esm2-3b` or SaProt | torch is older than the pickle-weights hardening needs | `bash scripts/setup_env.sh --cluster grace` (the pins are torch 2.10) |
| `mpd fetch model aido-rag-16b` exits 2 with the licence text | AIDO is never fetched | see [LICENSES.md](LICENSES.md) |

The full list for weights is in [MODELS.md](MODELS.md#if-something-goes-wrong).

## Planning and submitting

| you see | it means | do |
|---|---|---|
| `NOT READY: N errors ...` at the end of `plan` or `submit --dry-run`, exit status 0 | the recipe is fine but something it needs is missing; plan and dry-run only report it | fix the `[FAIL]` lines; each has its command under it |
| `error: not submitting: N problem(s) to fix first` (exit status 2) | a real `submit` found the same problems and stopped before writing anything | fix them, run the same command again |
| `campaign <id> was already submitted (jobs: ...). Nothing was written.` | the same options give the same campaign id, and it is queued or finished | `mpd status <id>`; `mpd cancel <id>`; `--force` to queue only the stages whose job died or is missing; or `--seed` for an independent run |
| `campaign <id> was only partly submitted (queued: ...)` | a submission was interrupted (Ctrl-C, outage, killed shell) | `mpd status <id>`, then `mpd cancel <id>`, or the same command with `--force` to continue from the first missing stage |
| `estimated upper bound N SU is above --max-su M` | the cost guard: a real `submit` stops (`plan` and `--dry-run` only warn). sbatch's own "Requested SUs" would not have shown it, because it counts CPUs only | read the plan; shrink it (fewer `--updates` or assays, `--gpu`), or pass `--max-su` on purpose |
| `[train] requested 5-00:00:00 exceeds the a100 partition limit 4-00:00:00; Slurm would hold the job with PartitionTimeLimit ...` | a walltime above the partition's limit | a shorter `--train-time`, more `--train-gpus`, or another GPU |
| `<gpu> cannot train here: it has no native bfloat16 (Turing) ...` | `rtx` and `t4` are refused for training | `--gpu a100` or `--gpu a40`; `--allow-unsupported-gpu` runs float32 (never tested) |
| `[train] 3 x a100 does not fit one node (2 per node on grace) ...` | more GPUs than a node holds; multi-node jobs are not generated | `--train-gpus 1` or `2` |
| `<model> needs about N GB (...) but <gpu> offers U GB usable per GPU.` | the memory estimate says it does not fit | a larger GPU (`--gpu a40` has 48 GB) or a smaller model |
| `global batch 63 is not divisible by GPUs x microbatch = 1 x 2 ...` | the global batch must divide | follow the message |
| `--sbatch-arg '...' must be one long option written --name or --name=value` | argparse split the value | write it with an equals sign: `--sbatch-arg=--exclude=g006` |
| `error: aido-rag-16b -> esm2-650m is blocked: outputs of aido-rag-16b may only be distilled into aido models ...` | the licence lock; there is no override | choose another teacher; [LICENSES.md](LICENSES.md) |
| `sbatch rejected stage <stage>: ...` followed by a sentence of advice | Slurm refused the script; the sentence says why | [below](#if-sbatch-refuses-a-stage) |
| `sbatch was not found` | this is not a Slurm login node | `mpd submit --dry-run` writes and checks the scripts anywhere; submit from the login node |
| `a submission of <id> started <time> and did not finish` | an earlier `mpd submit` was killed, or is still running in another terminal | `mpd status <id>`, then the same command with `--force` ([below](#the-same-command-again-and---force)) |
| `squeue shows a job of this submission without its --comment tag ...` | the cluster rewrites job comments, so `mpd` cannot recognise a job that `sbatch` accepted while reporting an error, and stops instead of guessing | look in `squeue -u "$USER"` for that stage by its name and cancel a stray copy by hand |

### The same command again, and `--force`

Running the same `mpd submit` command twice never queues a second copy of a chain. What happens depends on the state in
`<campaign>/SUBMITTED.json`, which `mpd status` shows **[CPU]** (fake Slurm):

| state | meaning | the same command again |
|---|---|---|
| `submitting` | a submission started and did not finish | refused; `mpd status`, then `--force` continues it |
| `submitted` | the whole chain was queued | refused; `--force` queues only the stages whose job died (`TIMEOUT`, `FAILED`, cancelled by hand) or vanished, plus the stages that wait for them; queued, running and finished stages are kept |
| `partial` | something may still be queued although a stage failed | refused; `--force` continues from the first missing stage |
| `failed` | Slurm refused a stage; what that run had queued was cancelled | runs again from the start, no `--force` needed |
| `cancelled` | `mpd cancel` removed the jobs | runs again from the start |

A stage counts as finished when its output file exists: `contexts/MANIFEST.json` (`prepare`), `teacher/COMPLETE.json`,
`train/<unit>/COMPLETE.json`, the `score/<dataset>/<unit>/shard_<i>_of_<n>.json` files, `evaluate/summary.json`. Delete the file if you want
that stage to run again. Make sure no other `mpd submit` of the campaign is running before you use `--force`. For a second, independent
run of the same recipe change `--seed`.

### If `sbatch` refuses a stage

A refusal, as opposed to silence, is not retried: `mpd` prints sbatch's words and a hint, cancels the jobs it queued earlier in that run,
and marks the campaign `failed`. Fix the cause and run the same command again.

| sbatch says (part) | meaning | do |
|---|---|---|
| `your account's balance is not sufficient` **[HPRC]** | the amount computed from the request (not the real usage) is above the balance | `myproject`; the plan's figure includes GPUs, sbatch's counts CPUs only; fewer `--updates` or assays, a cheaper `--gpu`, or ask HPRC for SUs |
| `Invalid account or account/partition combination` | Slurm does not accept the billing account | `myproject` lists yours; pass `--account` (or `MPD_ACCOUNT`), or leave it out for the default |
| `Requested node configuration is not available`, `invalid partition`, `invalid generic resource` | the partition or GPU type does not exist in that shape | another `--gpu`; `mpd submit --dry-run` test-submits each distinct stage shape before you queue anything; `mpd clusters` lists the types |
| `Requested time limit is invalid`, `PartitionTimeLimit` | a walltime longer than the partition allows | lower `--train-time`, `--score-time`, `--teacher-time`, or a GPU type with a longer limit |
| `Job violates accounting/QOS policy`, `AssocMax...`, `QOSMax...` | your account or QOS limits (jobs, resources) | fewer assays at once, a lower `--array-throttle` |
| `unrecognized option`, `doesn't allow an argument` | a flag in the script or in `--sbatch-arg` is not known to this Slurm | remove it; report it with `mpd doctor --report` |
| `Job dependency problem` | a job a stage waits for is unknown to Slurm (it ended long ago) | `mpd status`, then `--force` |
| anything else | `mpd` does not recognise it as temporary, so it does not retry | the message is sbatch's; `mpd doctor --report` for an issue |

### `Unable to contact slurm controller` during submit

Grace's controller went away for minutes to hours on 2026-06-11, 06-26, 07-24 and 07-26, and on 07-26 `sbatch` printed
`Unable to contact slurm controller` although it had accepted the whole chain **[earlier jobs]**. So a failed `sbatch` is not proof that
nothing was queued. `mpd submit` therefore:

1. gives every job a tag in its `--comment`, writes `SUBMITTED.json` after each accepted job, and on a controller error waits (the
   messages say `Slurm did not answer for stage <stage> ...; waiting N s, then checking the queue ... (try k of 6)`);
2. looks in `squeue -u "$USER"` for a job with this submission's tag before it tries again, and keeps only one copy if a lost reply left two:
   `stage <stage> was queued after all (job <id>): the controller accepted it while reporting an error`;
3. gives up after about 15 minutes in total and cancels what it queued, or says what it could not cancel **[CPU]** (fake Slurm tools).

If it gave up, **run `mpd status <campaign>` before anything else**: the message says whether a stage may have been accepted anyway. Then
run the same `mpd submit` command with `--force` to continue from the first missing stage, or `mpd cancel <campaign>`. Do not delete
`SUBMITTED.json`. If `mpd status` itself prints `slurm no job states: ...`, the controller is not answering yet; wait and ask again.

## In the queue

`mpd status <campaign>` shows each stage's state and, for waiting jobs, `waiting: <reason>`.

| you see | it means | do |
|---|---|---|
| `waiting: Dependency` | the stage waits for an earlier stage | nothing |
| `waiting: Priority` or `Resources` | the queue; the A100 queue can be long | wait, or plan a new campaign with `--gpu a40` (an A40 request started sooner in the author's July 2026 jobs; one observation) |
| `waiting: PartitionTimeLimit` | the requested time is longer than the partition allows (Slurm accepts such a job and holds it forever). `mpd` refuses this for its own estimates and for `--train-time` and the like, so it can only come from `--sbatch-arg=--time=...` or a custom profile | `mpd cancel`, plan again with a shorter time |
| a stage `FAILED` and its dependants `CANCELLED` | `mpd` submits with `--kill-on-invalid-dep=yes`, so jobs that waited for a failed stage are cancelled | fix the cause (next section), then the same `mpd submit` command with `--force` |
| nothing is queued but `mpd status` lists the campaign as `unsubmitted` | it was planned or dry-run, never submitted | run the `mpd submit` command without `--dry-run` |

## A job failed or stopped

Read the end of the stage's error log first: `mpd logs <campaign> <stage> --err` (without `--err` for the output). `mpd status` already
prints the last 20 lines for the first failed stage, with account numbers and your user name removed.

| you see | it means | do |
|---|---|---|
| `MPD_NO_GPU: no usable GPU on <node>; resubmit with --sbatch-arg=--exclude=<node>` or `MPD_NO_GPU: torch.cuda.is_available() is False ... CUDA_VISIBLE_DEVICES=...` (exit 31) | the node has no usable GPU. In May 2026 one node listed its A100 and could not use it **[earlier jobs]**; the stage stops instead of running on the CPU | first the same `mpd submit` command with `--force`, which queues the failed stage again; it may land on another node. If the same node fails again, add `--sbatch-arg=--exclude=<node>`; extra sbatch options are part of the campaign id, so that starts a **new campaign** from `prepare` and redoes the earlier stages, the teacher included |
| `MPD_MODULE_LOAD_FAILED <modules>` (exit 1) | a module of the profile, of `<home>/env/mpd_env.json` or of `--module` did not load | `module spider <name>` on a login node; run `setup_env.sh` again with other `--modules`; or `--module` for one more |
| `MPD_MODULE_LOAD_FAILED the module command is not available (...)` | the script could not find Lmod's `module` function | report it with `mpd doctor --report` |
| `/etc/bashrc: ...: unbound variable` or another `unbound variable` | a variable used after `set -u`. The script switches `set -u` on only after the modules loaded, because it broke Grace's bashrc in the author's jobs **[earlier jobs]** | report the line; do not add `source ~/.bashrc` to scripts |
| `OSError: We couldn't connect to 'https://huggingface.co' ...` or `Network is unreachable` in a job | something tried to download. Compute nodes have no internet and `mpd` jobs are offline on purpose; a missing model normally gives the `weights for ... are missing` message instead | `mpd fetch model --list` shows what is fetched; check that no `MPD_MODEL_*` or `--env` points at a Hub id; do not set `MPD_ALLOW_HUB` (it never reaches a job) |
| `torch.OutOfMemoryError` or `CUDA out of memory` | the memory estimate was too low: activation sizes are estimated, not measured **[NOT VERIFIED]** | a larger GPU (`--gpu a40` has 48 GB) or a smaller student; more GPUs do not help (every GPU holds a full copy); the micro-batch is chosen by the planner (2, else 1) and has no flag. Please report the model, GPU and the `peak_gpu_memory_gb` in the stage's `COMPLETE.json` |
| state `TIMEOUT` in `mpd status`, and the log has no `MPD_STAGE_DONE` | the walltime was too short; every Grace walltime is an estimate | see [TIMEOUT](#timeout-the-walltime-was-too-short) below |
| `shards incomplete (have [0, 1] of 3; missing [2]) ...` from `evaluate` | some `score` tasks failed or were cancelled | `mpd status <campaign>`, fix the cause, the same command with `--force`; leftover shard files of another count are ignored |
| `the teacher output cannot be used for this campaign: <reason> ...` from `train` | the teacher table on disk was computed for other contexts or another teacher | the same command with `--force` re-runs the teacher |
| `could not write the resumable checkpoint at update <k> (...); continuing without it` | the disk or your quota refused the write; training goes on but cannot resume | `showquota`, free space, rerun if the job later dies |
| `not resuming: <reason>` | the saved state did not belong to this run; the unit starts at update 0 and the old attempt is kept as `*.incomplete-<epoch>` | read the reason; delete the `*.incomplete-*` folders when you no longer need them |
| `Some weights of EsmForMaskedLM were not initialized ... contact_head ...` | harmless for SaProt (unused layers) | none |

### `TIMEOUT`: the walltime was too short

Training saves a resumable state every 200 updates (`MPD_CKPT_EVERY`) **[CPU]**: killed and resumed runs matched uninterrupted ones in
float32 on CPU; a GPU run resumed this way is equivalent, not bit-identical, and none has been tried **[NOT VERIFIED]**.

- **Run the identical `mpd submit` command again, with `--force`.** Stages that finished skip themselves, and the unfinished training continues from its
  last saved update (the log says `resuming from update <k> of <n>`). Each rerun gets the same walltime, so a stage that times out
  after the 200th update still makes progress.
- **Do not change any option for the rerun.** Walltime flags, `--eval-data`, `--gpu`, `--score-shards`, `--sbatch-arg`, `--env` and
  `--module` are part of the campaign id: change one and you get a **new campaign directory that trains from update 0**. That is also
  what happens if you answer a timeout with a larger `--train-time`; do it only if you want a clean single run.
- The `teacher` and `score` stages have no partial progress: a killed teacher recomputes (estimated 10 to 15 minutes for a 650M teacher,
  45 to 60 minutes for a 3B one, not measured) and a killed score task redoes its shard. A finished teacher or shard is skipped.
- Nothing prevents two live jobs from training the same unit: `mpd cancel <campaign>` before you rerun.

## Cancelling, logs and results

**Cancel.** `mpd cancel <campaign>` cancels the queued and running jobs of a campaign: the job ids in `SUBMITTED.json` that are still
alive, plus any job in `squeue -u "$USER"` that carries the campaign's `--comment` tag (a job that `sbatch` accepted while reporting an
error is not in the marker). Afterwards the campaign is marked `cancelled`, so the same `mpd submit` command can be run again (if every job had already finished it
says so and leaves the state alone). If
`squeue` or `scancel` does not answer (the controller is down) it says so, keeps the ids, marks the campaign `partial` and tells you to run
`mpd cancel` again later or `scancel <ids>` by hand. `<campaign>` is the directory, the id, or a unique prefix of the id (the id needs
`MPD_HOME` or `--home` to be found). Do not use `scancel -u <netid>`: it cancels all your jobs.

**Logs.** `mpd logs <campaign>` lists the newest log of every stage, `mpd logs <campaign> <stage> [--err] [--tail N]` shows one. The files
are `<campaign>/logs/mpd-<stage>-<student>_<job>[_<task>].out` and `.err`, appended to if a job is requeued. Lines to look for:

| line | meaning |
|---|---|
| `MPD_JOB_START host=<node> job=<id> stage=<stage>` | the script got past environment setup; a job with an empty log failed before this line, so read its `.err` file |
| `GPU 0: NVIDIA A100-PCIE-40GB (UUID: ...)` | `nvidia-smi -L` in a GPU stage |
| `MPD_STAGE_DONE <stage>` | the stage's Python process exited 0; missing after a crash or `TIMEOUT` |
| `teacher <key>: ... already computed ...; skipping` / `recomputing, <reason>` | a finished teacher table was reused / replaced |
| `<unit>: resumable checkpoint at update <k>` / `resuming from update <k> of <n>` | the resumable state was written / a rerun continues |
| `scored <n> contexts (<unit>, <dataset>, shard <i>/<num>)`, `<file>.npz: already done` | a score shard finished / was found finished |
| one JSON line per monitored update | development metrics (`dev_loss`, `dev_rho_ST`, ...) |
| `<unit>: selected update <k> (dev loss ..., start ...)` | the end of training |

Every record the stages write (`teacher/COMPLETE.json`, `train/<unit>/CONFIG.json` and `COMPLETE.json`, each score shard `.json`,
`evaluate/summary.json`) has a `provenance` block: mpd, torch, transformers and peft versions, Python, platform, host, Slurm job id,
UTC time, and on a GPU node the GPU name, count and driver. It holds no paths, so it can go into an issue.

**Results.** `mpd results <campaign>` prints the tables and where the adapters are ([RESULTS.md](RESULTS.md)). `$SCRATCH` is not
backed up. Copy what you want to keep from your own computer (the command is printed by `mpd results`):

```bash
rsync -av <netid>@grace.hprc.tamu.edu:/scratch/user/<netid>/mpd/runs/<campaign>/{evaluate,spec.json,resolved.json} ./<campaign>/
mkdir -p ./<campaign>/train/all
rsync -av <netid>@grace.hprc.tamu.edu:/scratch/user/<netid>/mpd/runs/<campaign>/train/all/checkpoint_02103 ./<campaign>/train/all/
```

HPRC's portal and Globus also work for copying files off Grace.

## SaProt and Foldseek

SaProt is `planned`, so these appear only once the stages are connected ([SAPROT.md](SAPROT.md)).

| you see | it means | do |
|---|---|---|
| `Foldseek was not found (<home>/tools/foldseek-10-941cd33). Run on a login node: mpd fetch foldseek --cluster grace` | Foldseek is installed on a login node, because compute nodes have no internet | run the command; with no route, download `foldseek-linux-avx2.tar.gz` of release 10-941cd33 elsewhere and run `mpd fetch foldseek --cluster grace --from-file <tarball>` (checked against the pinned sha256 before anything runs) |
| `MPD_FOLDSEEK=<path> is not an executable file (and no fallback is used while it is set)` | the variable points nowhere | fix the path or `unset MPD_FOLDSEEK` |
| `the pinned Foldseek binary (sha256 verified) does not run here` or `cannot execute ...` | a `noexec` mount, or a CPU without AVX2 | install under another `--home`; try `<binary> version` by hand (**[NOT VERIFIED]** on Grace) |
| warning `Foldseek ... is not the pinned release 10-941cd33` | a different build works, but its strings were not compared with the recorded results | none; its sha256 is recorded in `contexts/structures.json` |
| `VenusMutHub structures not found ... The structure set is not published yet` | SaProt on VenusMutHub needs the structure deposit, which is in preparation | with a copy of the deposit: `mpd fetch venus-structures --cluster grace --from-dir <folder>`; ProteinGym needs nothing extra |
| `<assay>: N of M structure residues differ from the sequence`, `... does not contain the sequence`, `... occurs 2 times` | a structure must match its sequence residue for residue (at most three single differences) | that assay cannot be used with SaProt |
| `<file> has 2 chains (A, B)` | structure files must hold one chain, or the assay must name it | use another structure |

## What to paste in an issue

Paste the output of

```bash
mpd doctor --report
```

It prints the `mpd` and Python versions and the check table with account numbers, your user name, your home and scratch paths,
e-mail addresses and tokens removed (**[CPU]**; look it over before you post). Add:

- the exact command you ran, with `<ACCOUNT>` in place of your account number, and the campaign id;
- the last lines of the failing stage's `.err` (`mpd logs <campaign> <stage> --err`) and the `provenance` block of its `COMPLETE.json`;
- for a Slurm problem, the output of `sbatch --version` and the rejected line.

Do not paste account numbers, user names, tokens, or raw `squeue`, `sacct`, `showquota` or `myproject` output (they contain your user
name and project names).
