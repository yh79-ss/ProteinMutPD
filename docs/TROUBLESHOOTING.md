# Troubleshooting

Start with `mpd doctor`. It checks the environment, data, weights, quota, account and Slurm tools, and prints the command that fixes
each problem under the `[FAIL]` and `[warn]` lines. `mpd plan` and `mpd submit --dry-run` print the same table for the run you
described. A `READY` footer means the static checks passed; it does not prove that a job will run on the cluster
([what has not been verified](GRACE.md#what-has-not-been-verified-on-grace)).

Tags as in [GRACE.md](GRACE.md): **[CPU]** tested on CPU here, **[Vision]** accepted by `sbatch --test-only` on Vision, **[HPRC]** from an
HPRC document, **[earlier jobs]** seen in the profile author's Grace jobs before `mpd`, **[NOT VERIFIED]** nobody has run it. The
messages below are the ones the code prints; in a quoted message `<x>` marks a part that varies, and `<command>` is the exact command to run, which
the message prints. Nothing here was seen on Grace.

Exit statuses: 0 done; 1 `mpd doctor` found an error, or `logs`/`results` had nothing to show, or `submit` was declined at the prompt; 2 a usage or configuration
error (the message starts with `error:`), or a real `submit` stopped by a check or by `--max-su`; 3 `mpd fetch model` stopped at the login-node time budget; 31 a GPU stage
found no usable GPU (`MPD_NO_GPU`); 130 Ctrl-C.

Contents: [Setting up](#setting-up) | [What doctor and the plan table say](#what-mpd-doctor-and-the-plan-table-say) |
[Fetching data and weights](#fetching-data-and-weights) | [Planning and submitting](#planning-and-submitting) | [In the queue](#in-the-queue) | [A job failed or stopped](#a-job-failed-or-stopped) |
[What mpd status says](#what-mpd-status-says) | [Cancelling, logs and results](#cancelling-logs-and-results) |
[SaProt and Foldseek](#saprot-and-foldseek) | [What to paste in an issue](#what-to-paste-in-an-issue)

## Setting up

| you see | it means | do |
|---|---|---|
| `mpd: command not found` | the environment is not active in this shell | `source $SCRATCH/mpd/env/bin/activate`, or run `$SCRATCH/mpd/env/bin/python -m mpd ...` |
| `--cluster is required (or set MPD_CLUSTER). Available: <profiles>` | `MPD_CLUSTER` is not set; `activate` sets it | activate the environment, or add `--cluster grace` |
| `storage root <root> needs $<variable>, which is not set. Pass --home /path/to/mpd-home or export MPD_HOME.` | `$SCRATCH` is empty in this shell | `export MPD_HOME=/scratch/user/<netid>/mpd` |
| `<path> is under <prefix> (10 GB / 10,000 files on Grace). Use scratch: export MPD_HOME=$SCRATCH/mpd` | the working root would fill `$HOME`, whose quota can lock you out | use a path under `$SCRATCH` |
| `no Python environment at <home>/env. Run: <command>` | the venv does not exist (or `MPD_PYTHON` points elsewhere) | run the command shown: `bash scripts/setup_env.sh --cluster grace` |
| `module <module> did not load. On Grace look for a Python module with:` | the default module is not on this system; it was never checked on Grace again (**[NOT VERIFIED]**) | `module spider Python`, then `bash scripts/setup_env.sh --cluster grace --modules "<module names>"`; or `--modules none --python /path/to/python3` |
| `<python> is Python <version>; mpd needs 3.11 or newer. Load a newer Python module (module spider Python) or pass --python` | the module's Python is too old | load a newer Python module |
| `is newer than the Python 3.13 that mpd was tested with` (a warning of `setup_env.sh`) | the module or `--python` gives Python 3.14 or newer; `mpd` was tested on 3.11, 3.12 and 3.13 only, and the pinned numpy and scipy have no 3.14 wheel | go on if you like; if the install or a job fails, build again with a 3.11 to 3.13 Python: `module spider Python`, then `bash scripts/setup_env.sh --cluster grace --modules "<module names>" --rebuild` |
| `run this script with bash:  bash scripts/setup_env.sh --cluster grace` | you started the script with `sh` (or `dash`) | type the command shown: it starts with `bash` |
| `unknown option <flag> (bash scripts/setup_env.sh --help)` | a flag the script does not have | `bash scripts/setup_env.sh --help` lists them |
| `no terminal to ask on: pass --yes to go ahead` | the script asks "Continue? [y/N]" and has no terminal (a batch job, a pipe) | add `--yes` |
| `setup_env.sh` ends without finishing after about an hour | the login-node limit (no process longer than an hour) **[HPRC]** | run the same command again: the wheels are cached in `<home>/cache/pip` and it continues |
| `Disk quota exceeded` during pip, or `OSError: [Errno 122]` | `$SCRATCH` is out of space or files (1 TB, 250,000) | `showquota`; delete old runs, caches and unused models; then run the command again |
| `<packages> not installed in <python> (the GPU stages import them)` right after `--no-torch` | expected: the planner needs no torch | none; a job environment is built without `--no-torch` |
| `mpd is not installed in <python>; the jobs run` | the interpreter is not the venv `setup_env.sh` built | run `setup_env.sh` again, or pass the right `--python` |
| `the jobs use mpd <installed> but this command is mpd <version> (the environment is a snapshot of the code at setup time)` | you pulled new code; the environment is a snapshot | `bash scripts/setup_env.sh --cluster grace` again |
| `ignoring the environment record:` (a plan warning), or `so the planner ignores it` (`mpd doctor`, row `env-record`) | `<home>/env/mpd_env.json` was written for another cluster or another home, so the jobs would not load the recorded modules | run `setup_env.sh` again for this cluster and home |
| `pip install` fails with ``configuration error: `project.license` must be valid exactly by one definition`` | the build ran with `--no-build-isolation` and a setuptools older than 77.0.3 (`pyproject.toml` uses an SPDX licence expression) | install without `--no-build-isolation`, which fetches a new enough setuptools, or run `python -m pip install -U 'setuptools>=77.0.3'` first |

## What `mpd doctor` and the plan table say

The rows of the check table (`[ok]`, `[warn]`, `[FAIL]`, `[skip]`). The ones that need a word of explanation:

| you see | it means | do |
|---|---|---|
| `is not writable (checked <path>)` | `[FAIL] home`: you cannot write the working root | `export MPD_HOME=$SCRATCH/mpd` |
| `is inside your home directory. Ignore this where that has no small quota; on HPRC clusters it does (10 GB / 10,000 files)` | `[warn] home` on a machine that is not covered by the Grace rule | on Grace use `$SCRATCH` |
| `Low: the environment, ESM2 weights and data need about <n> GB` | `[warn] home`: little free space on that filesystem | a larger filesystem with `--home` |
| `showquota reports <n>% used:` | `[warn] quota`: a quota is nearly full | delete old runs, caches, unused models |
| `showquota ran but its output was not recognised; run it yourself and look at the file counts` | `mpd` reads only percentages or used/limit pairs; the format on Grace is not known **[NOT VERIFIED]** | read `showquota` yourself |
| `account <x> has characters Slurm accounts do not use (letters, digits, '_', '.', '-')` | a typo in `--account` | the number `myproject` prints |
| ``<account> was not found in the output of `myproject` (its format is not parsed; look at it yourself)`` | `[warn] account`: the account is not in the text `myproject` printed | check with `myproject`; a wrong account is refused by Slurm at submit |
| `interpreter <x> does not exist or is not executable` | `--python` or `MPD_PYTHON` names a missing file | the right path, or `bash scripts/setup_env.sh --cluster grace` |
| `<path> has <n> files but no FETCHED.json, so the fetch may have been interrupted (it resumes)` | `[warn] data:...`: a fetch stopped half way | run the fetch command again |
| `ProteinGym data is incomplete under <path>. It lacks <n> of <m> assay files, such as <file>. The fetch was probably cut off.` | `[FAIL] data:proteingym`: `mpd fetch proteingym` stopped while it was unpacking (the connection dropped, or the login node ended it), so `FETCHED.json`, which is written last, is missing; the row says `structure files` when the second zip was the one that stopped | run the command shown under the row again on a login node; it keeps what it downloaded and unpacks the rest |
| ``running inside a Slurm job: compute nodes have no internet, so `mpd fetch` and `pip` will fail here; use a login node`` | `[warn] host`: `mpd` was started in a batch job or an `srun` session | use a login node |
| ``<tools> not found on PATH (`mpd status`, `logs` and `cancel` need them)`` | `[warn] slurm`: `sbatch` exists but one of the others does not | a normal Grace login node has all four |
| ``sbatch not found on PATH: submit from a Slurm login node (`mpd plan` and `mpd submit --dry-run` work anywhere)`` | `[FAIL] slurm` in `submit` on a machine without Slurm (`--dry-run` shows `[skip] slurm` there) | submit from the login node |
| `sbatch --test-only refused these lines: <lines>  Slurm said: <message>` | `[FAIL] sbatch --test-only`: Slurm would not accept that job shape | [the table below](#if-sbatch-refuses-a-stage) |
| `the Slurm controller did not answer (<reason>); the scripts were not tested` | `[skip] sbatch --test-only`: the controller was busy | `mpd doctor` again when `squeue -u "$USER"` answers |
| ``<module> could not be imported:`` | `[warn]`: a part of `mpd` the check needs is broken in this install | `bash scripts/setup_env.sh --cluster grace --rebuild` |
| `` `import torch` failed in <python>: <reason> `` | `mpd doctor --deep` imported torch in the job Python and it failed (the first import can take minutes on a cold node) | read the reason; `bash scripts/setup_env.sh --cluster grace` again |

## Fetching data and weights

Run these on a login node: compute nodes have no internet.

| you see | it means | do |
|---|---|---|
| `ProteinGym data not found under <path>. Run on a login node: <command>` | not fetched yet (or `--home` differs from the one you fetched into) | run the command shown: `mpd fetch proteingym --cluster grace` |
| `VenusMutHub data not found under <path>. Run on a login node: <command>` | same | `mpd fetch venusmuthub --cluster grace`; it fetches one file at a time and takes a few minutes |
| `weights for <key> are missing (<path>). Run on a login node: <command>` | the weights are not there, or were deleted after submission (**scratch purge rules are [NOT VERIFIED]**) | run the command shown, for example `mpd fetch model esm2-3b --cluster grace`, then resubmit |
| `HTTP Error 429` or `503` from Hugging Face | rate limiting; the fetch already waited the `Retry-After` time | run the same command again later: it resumes. `mpd` fetches anonymously and never sends a Hugging Face token. If it keeps failing, fetch elsewhere and copy ([MODELS.md](MODELS.md#no-internet-on-the-cluster): `rsync`, or `--from-dir`) |
| `stopped before the login-node time limit with <n> file(s) left; run the same command again to resume` (exit status 3) | `mpd fetch model` used its time budget, 55 minutes on Grace (`--time-budget MIN` changes it; the two data fetches have no budget) **[CPU]** | run the same command again; it continues from the `.part` files |
| `could not download <url>: HTTP <code> <reason>` | this machine cannot reach the server, or the server refused | fetch on a machine that can and copy the folder with `--from-dir` ([MODELS.md](MODELS.md)) |
| `not enough free space under <path>: need <size>` | free space is below the missing bytes plus 1 GiB | `showquota`; another `--home` on `$SCRATCH`; delete old runs |
| `<key>/<file>: <problem>; removed the file` | the file differs from the pin: `<problem>` reads `size <n> != expected <n>`, `md5 mismatch` or `sha256 mismatch`; the file was deleted | run again; if it repeats, open an issue with the message |
| `is already writing to <path>; wait for it to finish, then run this again` | the message starts "another `mpd fetch model <key>`": a fetch of that model is running (maybe in another session) | wait for it |
| `--from-dir <path> is not a folder` | the path given to `--from-dir` does not exist | give the folder that holds the files |
| `no folder with its files under <path> (looked for` | `--from-dir` found no folder for that model: it looks for `<key>/`, the repository name, a Hugging Face cache snapshot, and the folder itself | [MODELS.md](MODELS.md#no-internet-on-the-cluster) lists the layouts it accepts |
| `cannot locate the models folder: pass --home or --cluster, or set MPD_HOME (or MPD_MODELS_DIR)` | no working root is known | `source .../activate`, or `--cluster grace` |
| `incomplete download; the partial file is kept, run the command again` | the connection ended early | run the same command again; it resumes |
| `could not download <url>: the server ended the body after <n> of <m> bytes` | the connection dropped before the whole file arrived; `mpd` already asked again a few times | run the same command again; if it keeps happening, fetch from another network and copy the folder with `--from-dir` |
| `<file>: <name> is not a plain file name.` | `mpd fetch venus-structures` found a file name in the deposit's `SHA256SUMS` or manifest that would point outside the download folder (`../x`, a leading `/`, a backslash, an empty name); nothing was written for it | do not use that copy of the deposit: download it again, or ask whoever gave it to you |
| `has no published Zenodo record yet. Pass --from-dir <folder with the files> or --zenodo-record <id>.` | `mpd fetch venus-structures` before the deposit is published | with a copy of the deposit: `--from-dir <folder>` ([ZENODO.md](ZENODO.md)) |
| `structure set failed verification:` | the files of a Venus structure deposit do not match its `SHA256SUMS` | get the deposit again |
| `unknown data source <x>; choose from <list>` | a misspelt source | `mpd fetch --list` |
| `mpd never downloads <key>:` | AIDO is never fetched | see [LICENSES.md](LICENSES.md) |
| `ValueError ... torch.load ... upgrade torch to at least v2.6` while loading `esm2-3b` or SaProt | torch is older than the pickle-weights hardening needs | `bash scripts/setup_env.sh --cluster grace` (the pins are torch 2.10) |
| `ship PyTorch pickle weights only, and current transformers refuses to load those with torch` | the same, found by the preflight | install torch 2.7 or newer |

The full list for weights is in [MODELS.md](MODELS.md#if-something-goes-wrong).

## Planning and submitting

| you see | it means | do |
|---|---|---|
| `NOT READY: <n> errors` at the end of `plan` or `submit --dry-run`, exit status 0 | the recipe is fine but something it needs is missing; plan and dry-run only report it | fix the `[FAIL]` lines; each has its command under it |
| `not submitting: <n> problem(s) to fix first (the table above has the rest)` (exit status 2) | a real `submit` found the same problems and stopped before writing anything | fix them, run the same command again |
| `campaign <id> was already submitted (jobs: <list>). Nothing was written.` | the same options give the same campaign id, and it is queued or finished | `mpd status <id>`; `mpd cancel <id>`; `--force` to continue or repair it (queued, running and finished stages are kept); or `--seed` for an independent run |
| `campaign <id> was only partly submitted (queued: <list>). Nothing was written.` | a submission was interrupted (Ctrl-C, or an outage that lasted longer than the retries) | `mpd status <id>`, then `mpd cancel <id>`, or the same command with `--force` to continue from the first missing stage |
| `a submission of <id> started <when> and did not finish` | an earlier `mpd submit` was cut off (a dropped ssh connection, a closed terminal, a killed shell), or is still running in another terminal | `mpd status <id>`, then the same command with `--force` ([below](#the-same-command-again-and---force)) |
| `holds a campaign that ended as <state> and has a different recipe` / `already holds a submitted campaign with a different recipe` | a submitted campaign's recipe files are never rewritten, and this command would write different ones; the message names the file and the keys | run the `mpd` version that created the campaign, change `--seed` for a new campaign, or remove the directory (`mpd cancel` its jobs first) |
| `estimated upper bound <n> SU is above --max-su <n>.` | the cost guard: a real `submit` stops (`plan` and `--dry-run` only warn). sbatch's own "Requested SUs" would not have shown it, because it counts CPUs only | read the plan; shrink it (fewer `--updates` or assays, `--gpu`), or pass `--max-su` on purpose |
| `[train] requested <time> exceeds the <gpu> partition limit <limit>; Slurm would hold the job with PartitionTimeLimit.` | a walltime above the partition's limit | a shorter `--train-time`, more `--train-gpus`, or another GPU |
| `<gpu> cannot train here: it has no native bfloat16 (Turing) and mpd trains in bfloat16.` | `rtx` and `t4` are refused for training | `--gpu a100` or `--gpu a40`; `--allow-unsupported-gpu` runs float32 (never tested) |
| `[train] <n> x <gpu> does not fit one node (<k> per node on <cluster>); multi-node jobs are not generated.` | more GPUs than a node holds | `--train-gpus 1` or `2` |
| `<model> needs about <n> GB (` and `but <gpu> offers <n> GB usable per GPU.` | the memory estimate says it does not fit | a larger GPU (`--gpu a40` has 48 GB) or a smaller model |
| `global batch <n> is not divisible by GPUs x microbatch = <n> x <n>; pick --global-batch as a multiple of <n> or change --train-gpus` | the global batch must divide | follow the message |
| `<teacher> -> <student> is blocked: outputs of <teacher> may only be distilled into <family> models.` | the licence lock; there is no override | choose another teacher; [LICENSES.md](LICENSES.md) |
| `<model> is <status> and cannot be used yet.` | a model that is `planned` (AIDO) is not offered | `mpd models` lists what is `available` |
| `sbatch rejected stage <stage>: <message>` followed by a sentence of advice | Slurm refused the script; the sentence says why | [below](#if-sbatch-refuses-a-stage) |
| `<program> was not found. Submit from a Slurm login node` | this is not a Slurm login node | `mpd submit --dry-run` writes and checks the scripts anywhere; submit from the login node |
| `squeue did not answer (<reason>), so mpd cannot tell which stages are still queued` | `--force` could not read the queue | try again when `squeue -u "$USER"` answers |
| `squeue shows a job of this submission without its --comment tag` | the cluster rewrites job comments, so `mpd` cannot recognise a job that `sbatch` accepted while reporting an error, and stops instead of guessing | look in `squeue -u "$USER"` for that stage by its name and cancel a stray copy by hand |

### Flags the planner refuses

Each message names what is wrong. The ones people hit:

| you see | it means | do |
|---|---|---|
| `is not a walltime. Write HH:MM:SS (08:00:00 is eight hours) or D-HH:MM:SS (1-12:00:00 is a day and a half)` | `--train-time 8h` and the like | write `08:00:00` or `1-12:00:00` |
| `is ambiguous: Slurm reads it as` | `08:00` and a bare `90` are minutes for Slurm | write all three fields, `08:00:00` |
| `is zero; Slurm treats a zero time limit as unlimited. Give a positive walltime` | `--train-time 0` | give a real time |
| `argument --sbatch-arg: expected one argument` | argparse takes `--exclude=g006` written as a separate word for an option of its own | write it with an equals sign: `--sbatch-arg=--exclude=g006` |
| `must be one long option written --name or --name=value, without spaces (for example --sbatch-arg=--exclude=g001)` | the value has a space in it, or does not start with `--` | `--sbatch-arg=--exclude=g006`, one option per `--sbatch-arg` |
| `mpd sets <name> itself and relies on it; remove it` | `--sbatch-arg` named `--job-name`, `--output`, `--array`, `--dependency` or another option `mpd` needs | remove it |
| `--env expects NAME=VALUE, got <x>` | the `--env` value has no `=` | `--env MPD_DTYPE=float32` |
| `is not a module name (letters, digits, '_', '.', '+', '/', '-'; one module per --module)` | two modules in one `--module` | repeat `--module` |
| `--score-shards expects DATASET=N pairs such as proteingym=4,venusmuthub=1, got <x>` | a malformed `--score-shards` | `--score-shards proteingym=4` |
| `--train-gpus must be at least 1, got <n>; leave it out to use the profile's default` | `--train-gpus 0` | leave it out, or 1 or 2 |
| `needs --assay <assay id> [...]` | a `*-one` method without `--assay` | `mpd assays` lists the ids |
| `--assay is only valid with a *-one method, not` | `--assay` with an `*-all` method | drop `--assay`, or use a `*-one` method |
| `supports backends` | `--teacher-backend cache` for a model that only runs live | leave `--teacher-backend` out |
| `unknown model <key>; choose from:` | a misspelt model key (keys are lower case) | `mpd models` |
| `unknown cluster <name>; use one of` | a misspelt profile | `mpd clusters`; or give the path of a YAML file |
| `has no GPU type <name>; choose from` | `--gpu h100` and the like | `mpd clusters` lists the types |
| `cannot read --assays-file <path>:` | the assay list file does not exist | give an existing file |
| `--assay lists an assay twice` / `--eval-data lists a dataset twice` | the same id or dataset given twice | give each once |
| `--dev-fraction must be in (0, 0.5)` | the development split is a fraction of the contexts | for example `--dev-fraction 0.05` (the preset) |
| `--train-gpus applies to *-all methods; one-assay runs use one device` | `--train-gpus` with a `*-one` method | drop `--train-gpus` |
| `must be at least 1; got <n>` / `must be greater than 0; got <x>` / `must be 0 or more; got <x>` | a number that must be positive (`--updates`, `--global-batch`, `--lr`, `--array-throttle`, `--max-su`, ...) | give a positive number |
| `--score-shards: <x> is not a dataset; use <list>` / `must be a whole number from 1 to <n>` / `lists <x> twice` | a malformed `--score-shards` | `--score-shards proteingym=4,venusmuthub=1` |
| `give --cluster (or set MPD_CLUSTER) or --home (or set MPD_HOME)` | `mpd doctor` has no profile and no home | `source .../activate`, or add `--cluster grace` |
| `give --home, or --cluster (or set MPD_HOME / MPD_CLUSTER / MPD_MODELS_DIR)` | `mpd fetch` or `mpd models` has no idea where the working root is | the same |
| `name the models to fetch, or use --all or --list.` | `mpd fetch model` without a key | `mpd fetch model esm2-3b esm2-650m`, or `--all`, or `--list` |
| `give model keys or --all, not both` | both were given | one of them |
| `--all only goes with` | `--all` on a data fetch | `mpd fetch model --all` |
| ``model keys (<keys>) only go with `mpd fetch model`; `mpd fetch <source>` takes none`` | model keys after `proteingym`, `venusmuthub`, `venus-structures` or `foldseek` | `mpd fetch model <keys>` |
| `--env sets <name> twice with different values` | the same variable twice in `--env` | once |
| `is not an environment variable name` | an `--env` name with a space, a dash or a leading digit | letters, digits and `_` |
| `cannot be set for jobs:` followed by `it is for login nodes and laptops; jobs run offline (HF_HUB_OFFLINE=1)` | `--env MPD_ALLOW_HUB=1` | jobs never download; fetch on a login node |
| `must be one line of at most <n> characters without control characters` | an `--env` value with a newline | one line |
| `is a relative path; jobs change directory, so it would not resolve. Give an absolute path` | a relative path in `--env` for `MPD_FOLDSEEK`, `MPD_MODELS_DIR` or `MPD_MODEL_<KEY>` (a relative `--home` or `--python` is made absolute against the current directory) | an absolute path |
| `has characters that Slurm directives cannot carry (spaces, quotes, '%', '$', '#' ...). Use a path made of letters, digits and _ . + @ : , = / -` | the working root or a log path has such a character | a plain path under `$SCRATCH` |
| `may only contain letters, digits, '_', '.' and '-'` | an account with other characters | the number `myproject` prints |
| `has characters outside letters, digits and _ . + -; check the assay ids` | an assay id with an unusual character | `mpd assays` lists the ids |

### The same command again, and `--force`

Running the same `mpd submit` command twice never queues a second copy of a chain. What happens depends on the state in
`<campaign>/SUBMITTED.json`, which `mpd status` shows **[CPU]** (fake Slurm):

| state | meaning | the same command again |
|---|---|---|
| `submitting` | a submission started and did not finish | refused; `mpd status`, then `--force` continues it |
| `submitted` | the whole chain was queued | refused; `--force` queues only the stages whose job died (`TIMEOUT`, `FAILED`, cancelled by hand) or vanished, plus the stages that wait for them; queued, running and finished stages are kept |
| `partial` | something may still be queued although a stage failed | refused; `--force` continues from the first missing stage |
| `failed` | Slurm refused a stage; what that run had queued was cancelled | runs again, no `--force` needed; stages whose output files exist are not queued again |
| `cancelled` | `mpd cancel` removed the jobs | runs again, no `--force` needed; stages whose output files exist are not queued again |

A stage counts as finished when its output file exists: `contexts/MANIFEST.json` (`prepare`), `teacher/COMPLETE.json`,
`train/<unit>/COMPLETE.json`, the `score/<dataset>/<unit>/shard_<i>_of_<n>.json` files, `evaluate/summary.json`. **To run a finished stage again,
delete its marker file and repeat the command with `--force`**: that stage is queued again, with every unfinished stage that waits on it; finished and still-alive stages are kept
([example](GRACE.md#running-a-finished-stage-again)). Make sure no other `mpd submit` of the campaign is running before you use `--force`. For a second,
independent run of the same recipe change `--seed`.

A campaign's recipe files (`spec.json`, `resolved.json`, `tasks/*.json`) are fixed once a marker exists, in any state. What the options change:
`--eval-data`, `--gpu`, `--train-gpus`, `--score-shards`, `MPD_DTYPE`, `MPD_TF32` and the recipe flags start a **new** campaign; walltimes,
`--array-throttle`, `--sbatch-arg`, `--module` and other `--env` pairs keep it ([table](GRACE.md#what-starts-a-new-campaign-and-what-does-not)).

### If `sbatch` refuses a stage

A refusal, as opposed to silence, is not retried: `mpd` prints sbatch's words and a hint, cancels the jobs it queued earlier in that run,
and marks the campaign `failed`. Fix the cause and run the same command again (no `--force`; finished stages are kept).

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
   messages say `Slurm did not answer for stage <stage> (<reason>); waiting <n> s, then checking the queue before trying again (try <k> of 6)`);
2. looks in `squeue -u "$USER"` for a job with this submission's tag before it tries again, and keeps only one copy if a lost reply left two:
   `stage <stage> was queued after all (job <id>): the controller accepted it while reporting an error`;
3. gives up after about 15 minutes in total and cancels what it queued, or says what it could not cancel **[CPU]** (fake Slurm tools).

If it gave up, **run `mpd status <campaign>` before anything else**: the message says whether a stage may have been accepted anyway. Then
run the same `mpd submit` command with `--force` to continue from the first missing stage, or `mpd cancel <campaign>`. Do not delete
`SUBMITTED.json`. If `mpd status` itself prints `no job states:`, the controller is not answering yet; wait and ask again. A transcript is in
[GRACE.md](GRACE.md#a-submission-that-was-interrupted).

## In the queue

`mpd status <campaign>` shows each stage's state and, for waiting jobs, a `waiting:` section with the reason in words.

| you see | it means | do |
|---|---|---|
| `Dependency` | the stage waits for an earlier stage | nothing |
| `Priority` or `Resources` | the queue; the A100 queue can be long | wait, or plan a new campaign with `--gpu a40` (an A40 request started sooner in the author's July 2026 jobs; one observation) |
| `PartitionTimeLimit` | the requested time is longer than the partition allows (Slurm accepts such a job and holds it forever). `mpd` refuses this for its own estimates and for `--train-time` and the like, so it can only come from `--sbatch-arg=--time=...` or a custom profile | `mpd cancel`, plan again with a shorter time |
| a stage `FAILED` and its dependants `CANCELLED` | `mpd` submits with `--kill-on-invalid-dep=yes`, so jobs that waited for a failed stage are cancelled | fix the cause (next section), then the same `mpd submit` command with `--force` |
| nothing is queued but `mpd status` lists the campaign as `unsubmitted` | it was planned or dry-run, never submitted | run the `mpd submit` command without `--dry-run` |

Every reason `mpd status` explains is in [the table below](#why-a-stage-waits-waiting).

## A job failed or stopped

Read the end of the stage's error log first: `mpd logs <campaign> <stage> --err` (without `--err` for the output). `mpd status` already
prints the last 20 lines for the first failed stage, with account numbers and your user name removed, and the command that continues.

| you see | it means | do |
|---|---|---|
| `MPD_NO_GPU: no usable GPU on <node>; resubmit with --sbatch-arg=--exclude=<node>` or `MPD_NO_GPU:` with `torch.cuda.is_available() is False` (exit 31) | the node has no usable GPU. In May 2026 one node listed its A100 and could not use it **[earlier jobs]**; the stage stops instead of running on the CPU | the same `mpd submit` command with `--sbatch-arg=--exclude=<node>` and `--force`: the campaign stays the same and only the failed stage and what waits for it are queued again ([recipe](GRACE.md#a-bad-node)) |
| `MPD_MODULE_LOAD_FAILED <modules>` (exit 1) | a module of the profile, of `<home>/env/mpd_env.json` or of `--module` did not load | `module spider <name>` on a login node; run `setup_env.sh` again with other `--modules`; or `--module` for one more |
| `MPD_MODULE_LOAD_FAILED the module command is not available` | the script could not find Lmod's `module` function | report it with `mpd doctor --report` |
| `/etc/bashrc: ...: unbound variable` or another `unbound variable` | a variable used after `set -u`. The script switches `set -u` on only after the modules loaded, because it broke Grace's bashrc in the author's jobs **[earlier jobs]** | report the line; do not add `source ~/.bashrc` to scripts |
| `OSError: We couldn't connect to 'https://huggingface.co' ...` or `Network is unreachable` in a job | something tried to download. Compute nodes have no internet and `mpd` jobs are offline on purpose; a missing model normally gives the `weights for <key> are missing` message instead | `mpd fetch model --list` shows what is fetched; check that no `MPD_MODEL_*` or `--env` points at a Hub id; do not set `MPD_ALLOW_HUB` (it never reaches a job) |
| `ImportError: ... version 'GLIBCXX_3.4.29' not found` (or another `GLIBCXX` or `libstdc++.so.6` error) when a job imports `torch` | the job found an older C++ runtime than the Python packages need. The Grace jobs that ran used pip wheels and did not need a GCC module, so this is a precaution **[NOT VERIFIED]** | check on a login node that `module load GCC/11.3.0` works, then repeat your `mpd submit` command with `--module GCC/11.3.0 --force`; that module puts a newer `libstdc++` on `LD_LIBRARY_PATH` |
| `torch.OutOfMemoryError` or `CUDA out of memory` | the memory estimate was too low: activation sizes are estimated, not measured **[NOT VERIFIED]** | a larger GPU (`--gpu a40` has 48 GB; that is a new campaign) or a smaller student; more GPUs do not help (every GPU holds a full copy); the micro-batch is chosen by the planner (2, else 1) and has no flag. Please report the model, GPU and the `peak_gpu_memory_gb` in the stage's `COMPLETE.json` |
| state `TIMEOUT` in `mpd status`, and the log has no `MPD_STAGE_DONE` | the walltime was too short; every Grace walltime is an estimate | see [TIMEOUT](#timeout-the-walltime-was-too-short) below |
| `shards incomplete (have <list> of <n>; missing <list>)` from `evaluate` | some `score` tasks failed or were cancelled | `mpd status <campaign>`, fix the cause, the same command with `--force`; leftover shard files of another count are ignored |
| `the teacher output cannot be used for this campaign:` from `train` | the teacher table on disk was computed for other contexts, another teacher or other structure strings | delete `teacher/COMPLETE.json` and repeat your `mpd submit` command with `--force`: the teacher stage is queued again and the finished ones are kept |
| `could not write the resumable checkpoint at update <k> (` | the disk or your quota refused the write; training goes on but cannot resume | `showquota`, free space, rerun if the job later dies |
| `not resuming: <reason>`, with a reason such as `it was written by a run with a different configuration`, `STATE.json is unreadable`, `the logs cannot be read` or `the saved parameters are not this model's trainable parameters` | the saved state did not belong to this run or is damaged; the unit starts at update 0 and the old attempt is kept as `*.incomplete-<epoch>` | read the reason; delete the `*.incomplete-*` folders when you no longer need them |
| `Some weights of EsmForMaskedLM were not initialized ... contact_head ...` | harmless for SaProt (unused layers) | none |
| `non-finite loss at update <n>` / `non-finite gradient norm at update <n>` / `the teacher produced non-finite logits` / `no finite development checkpoint` | the numbers broke down; the stage stops instead of saving garbage **[NOT VERIFIED]**: none of these has been seen | run the command again with `--force` (training resumes from its last save) and, if it repeats, report the log and the recipe |
| `training unit(s) have too few measured positions to be split into training and development contexts` from `prepare` | a `*-one` assay has fewer than two measured positions (17 of the 793 VenusMutHub assays are like that), so it cannot give both a training and a development context. `prepare` stops before any GPU stage runs | leave the assays it names out of `--assay` or `--assays-file` and submit again; that is a new campaign |
| `training and <n> development contexts; need both` | the same problem found by `train`, which only happens for a campaign whose `prepare` ran with an older `mpd`. The `--dev-fraction` does not decide this: every sequence holds out at least `min(5, n-1)` positions | leave that assay out and submit again |
| `launched with <n> process(es) but the plan has` | a training script was started by hand with another number of processes | submit with `mpd submit`, which writes the launcher line |
| `MPD_CKPT_EVERY must be a whole number of updates, 0 or more (0 turns resumable checkpoints off); got <x>` | a bad value in the environment | a whole number, or unset it |
| `shard files exist for counts <list>, expected exactly one` / `was written for <n> shards` / `some evaluation contexts were never scored` | `evaluate` found score shards of a plan with another `--score-shards` | repeat your original command with `--force`; leftover shard files of another count are ignored |

### `TIMEOUT`: the walltime was too short

Training saves a resumable state every 200 updates (`MPD_CKPT_EVERY`) **[CPU]**: killed and resumed runs matched uninterrupted ones in
float32 on CPU; a GPU run resumed this way is equivalent, not bit-identical, and none has been tried **[NOT VERIFIED]**.

- **Run the identical `mpd submit` command again, with a larger `--train-time` and `--force`.** Stages that finished are not queued again, and the
  unfinished training continues from its last saved update (the log says `resuming from update <k> of <n>`). `mpd status` suggests twice the old
  limit, never above what the partition allows. Each rerun gets the walltime you give, so a stage that times out after the 200th update still makes progress.
- **Walltime flags, `--sbatch-arg`, `--module` and other `--env` pairs keep the campaign**; the rerun finds the finished stages and the checkpoints.
  `--eval-data`, `--gpu`, `--train-gpus`, `--score-shards`, `MPD_DTYPE` and `MPD_TF32` start a **new campaign** that trains from update 0.
- The `teacher` and `score` stages have no partial progress: a killed teacher recomputes (estimated 10 to 15 minutes for a 650M teacher,
  45 to 60 minutes for a 3B one, not measured) and a killed score task redoes its shard. A finished teacher or shard is skipped.
- A stage that is still queued or running keeps the options it was submitted with; to change a job that is still waiting, `mpd cancel <campaign>`
  and repeat the command. Nothing else prevents two live jobs from training the same unit, so cancel before you rerun if one may still be alive.

## What `mpd status` says

`mpd status <campaign>` explains every stage that did not end well, in these words (generated from the code, with invented job ids, nodes and times).
The sentence under `problems:` ends with the command to run: always your original `mpd submit` command, with the option named and `--force`.

### What `mpd status` says about a stage that did not end well

| Slurm state (and the log) | what `mpd status` prints under `problems:` |
|---|---|
| `TIMEOUT` (train) | train (job 1003) reached its time limit: it ran 08:00:12 of the 08:00:00 it asked for. Training saves its progress every MPD_CKPT_EVERY updates (200 unless you changed it) and continues from the latest save, so the work done so far is kept. Run your original `mpd submit ...` command again with a larger --train-time (for example --train-time 16:00:00; this partition allows up to 4-00:00:00) and --force. |
| `TIMEOUT` (score) | score (job 1005) reached its time limit: it ran 00:30:04 of the 00:30:00 it asked for. Scoring keeps the shards it finished; only the unfinished ones run again. Run your original `mpd submit ...` command again with a larger --score-time (for example --score-time 01:00:00; this partition allows up to 4-00:00:00) and --force. |
| `TIMEOUT` (prepare or evaluate) | prepare (job 1001) reached its time limit: it ran 04:00:02 of the 04:00:00 it asked for. This stage has no time option of its own. Report it with `mpd doctor --report` and the end of its log. |
| `OUT_OF_MEMORY`, or `FAILED` with exit 137 | train (job 1003) ran out of memory. If its log mentions CUDA, the GPU is too small for this model at this size: use a smaller student or a larger GPU (--gpu; `mpd clusters` lists them). More GPUs do not help, because every GPU holds a full copy of the model. If it does not, the node ran out of host memory (the job asked for 180G): add --sbatch-arg=--mem=<more>G. mpd's memory figures are estimates, not measurements. Then run your original `mpd submit ...` command again with --force. |
| `FAILED`, log says `CUDA out of memory` | train (job 1003) ran out of memory. The GPU's memory is too small for this model at this size: use a smaller student or a larger GPU (--gpu; `mpd clusters` lists them). More GPUs do not help, because every GPU holds a full copy of the model. mpd's memory figures are estimates, not measurements. Then run your original `mpd submit ...` command again with --force. |
| `NODE_FAIL` | train (job 1003) ended with NODE_FAIL: the node it ran on g003 failed, which is not a problem with your settings. Run your original `mpd submit ...` command again with --force. If it lands on g003 again, add --sbatch-arg=--exclude=g003. |
| `FAILED`, log says `MPD_NO_GPU` (exit 31) | train (job 1003) failed with exit code 31. The node has no usable GPU (the job stopped instead of running on the CPU). Run your original `mpd submit ...` command again with --sbatch-arg=--exclude=g007 and --force. |
| `FAILED`, log says `MPD_MODULE_LOAD_FAILED` | prepare (job 1001) failed with exit code 1. A module the job needs could not be loaded. The modules come from the environment record: `mpd doctor` shows it (row env-record) and `bash scripts/setup_env.sh --cluster grace` writes it again. Then run your original `mpd submit ...` command again with --force. |
| `FAILED`, the log says the weights of a model are missing | train (job 1003) failed with exit code 1. The job needed model weights that are not on disk, and compute nodes are offline. On a login node run `mpd fetch model esm2-3b --cluster grace`, then run your original `mpd submit ...` command again with --force. |
| `FAILED`, log says `huggingface.co` or offline mode | train (job 1003) failed with exit code 1. The job tried to reach huggingface.co, which compute nodes cannot do. The weights have to be on disk first: on a login node run `mpd fetch model <KEY>`, then run your original `mpd submit ...` command again with --force. |
| `FAILED`, anything else | train (job 1003) failed with exit code 1. The end of its error log is above; `mpd logs <campaign> train --err` shows all of it. When the cause is fixed, run your original `mpd submit ...` command again with --force: finished stages are kept. |
| `CANCELLED` after a failed stage (`DependencyNeverSatisfied`) | score (job 1005) was cancelled because a stage it waited for did not finish. Nothing to do for this stage until that one is fixed; then run your original `mpd submit ...` command again with --force. |
| `CANCELLED` by someone else | train (job 1003) was cancelled: by `mpd cancel`, by an administrator, or by a limit of the account. Run your original `mpd submit ...` command again with --force to queue what is missing. |
| any other bad state, for example `PREEMPTED` | train (job 1003) ended with state PREEMPTED, which mpd has no advice for. `mpd logs <campaign> train --err` shows the end of its log; then run your original `mpd submit ...` command again with --force. |
| campaign cancelled with `mpd cancel` | prepare, train, evaluate: cancelled with `mpd cancel`. Run your original `mpd submit ...` command again to queue the campaign again. |

### Why a stage waits (`waiting:`)

| reason in the `reason` column | what `mpd status` says |
|---|---|
| `Priority` | other jobs are ahead of it; this is the usual wait for a GPU. Nothing to do. |
| `Resources` | it is next in line and waits for a node with the GPU it asked for. Nothing to do. |
| `Dependency` | it waits for the stage or stages before it to finish. Nothing to do. |
| `JobArrayTaskLimit` | the array already runs as many tasks at once as its throttle allows (--array-throttle); the rest start as tasks finish. Nothing to do. |
| `DependencyNeverSatisfied` | a stage it waited for failed or was cancelled, so it will never start. Fix that stage, then run your original `mpd submit ...` command again with --force. |
| `PartitionTimeLimit` | its time limit is longer than the partition allows, so Slurm will never start it. `mpd cancel <campaign>`, then submit again with a shorter walltime. |
| `ReqNodeNotAvail` | a node it needs is not available (down, drained or reserved). If you excluded nodes with --sbatch-arg=--exclude=..., check the names; otherwise wait or ask the cluster's support. |
| `BeginTime` | its start time has not been reached. Nothing to do. |
| `JobHeldUser` | it is held; `scontrol release <job id>` frees it. |
| `JobHeldAdmin` | an administrator holds it; ask the cluster's support. |
| `InvalidAccount` | Slurm does not accept the account. `myproject` lists yours; `mpd cancel <campaign>`, then submit again with --account. |
| `InvalidQOS` | Slurm does not accept the QOS. `mpd cancel <campaign>`, then submit again without --sbatch-arg=--qos=... |
| `PartitionDown` | its partition is down. Wait, or ask the cluster's support. |
| `PartitionNodeLimit` | its node request is outside what the partition allows, or nodes are down. Ask the cluster's support. |
| `NodeDown` | a node it needs is down. Wait, or ask the cluster's support. |
| `Reservation` | it waits for an advance reservation. Nothing to do. |
| `Licenses` | it waits for a software license. Nothing to do. |
| `AssocGrp...Minutes`, anything with `balance` | the account's SU balance, or a limit on its minutes, cannot cover the job. `myproject` shows the balance; reduce the plan or ask the cluster's support for more SUs. |
| `Assoc...` (other) | a limit of your account (jobs or resources at once) holds it. It starts when your other jobs end. |
| `QOS...` | a limit of its QOS (jobs or GPUs per user at once) holds it back. It starts when your other jobs end. |
| anything else | `Slurm says: <the reason as Slurm printed it>` |

The wording of `sacct` and `squeue` on Grace, the exit codes of a job that hit its limit or ran out of memory, and the reason `squeue` keeps for a finished job are
**[NOT VERIFIED]** (see [GRACE.md](GRACE.md#not-verified)); a reason or a state that `mpd` does not know is shown as Slurm prints it.

## Cancelling, logs and results

**Cancel.** `mpd cancel <campaign>` cancels the queued and running jobs of a campaign: the job ids in `SUBMITTED.json` that are still
alive, plus any job in `squeue -u "$USER"` that carries the campaign's `--comment` tag (a job that `sbatch` accepted while reporting an
error is not in the marker). Afterwards the campaign is marked `cancelled`, and the same `mpd submit` command queues it again, **leaving out the stages whose output
files exist** (if every job had already finished it says so and leaves the state alone). If
`squeue` or `scancel` does not answer (the controller is down) it says so, keeps the ids, marks the campaign `partial` and tells you to run
`mpd cancel` again later or `scancel <ids>` by hand. `<campaign>` is the directory, the id, or a unique prefix of the id (the id needs
`MPD_HOME` or `--home` to be found). Do not use `scancel -u <netid>`: it cancels all your jobs.

| you see | it means | do |
|---|---|---|
| `no campaign called <name> under <path>. Campaigns there:` | the id is misspelt, or `MPD_HOME` is another home | `mpd status` lists the campaigns of the current home |
| `matches <n> campaigns under <path>; give more of the id:` | the prefix is not unique | give more of the id |
| `is not a campaign directory. Give the directory, or set --home / MPD_HOME to look the campaign up by id` | the argument is neither a campaign folder nor an id of this home | `source .../activate`, or pass `--home` |
| `has no SUBMITTED.json: it was planned or dry-run but never submitted.` | `mpd status` on a campaign that only has scripts | run the `mpd submit` command without `--dry-run` |
| `no results yet:` (exit status 1) | `mpd results` before the `evaluate` stage finished | wait; `mpd status <campaign>` |
| `no log files under <path> yet` / `no logs for stage <stage>; stages with logs:` | the jobs have not started, or the stage name is wrong | `mpd logs <campaign>` lists the stages with logs |

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
backed up. Copy what you want to keep from your own computer (the command is printed by `mpd results`; `train` holds the adapter of every unit and its training logs):

```bash
rsync -av <netid>@grace.hprc.tamu.edu:/scratch/user/<netid>/mpd/runs/<campaign>/{evaluate,spec.json,resolved.json,train} ./<campaign>/
```

HPRC's portal and Globus also work for copying files off Grace.

## SaProt and Foldseek

SaProt is `available`, tested on CPU only (tiny random models, a fake Foldseek, one by-hand `prepare` on real data); nothing has run on Grace or a GPU, and the pinned Foldseek
release 10-941cd33 was never executed ([SAPROT.md](SAPROT.md)). The messages below are the ones the code prints; `<home>` stands for the working root.

| you see | it means | do |
|---|---|---|
| `Foldseek was not found (<path>). Run on a login node: <command> (or set MPD_FOLDSEEK to a Foldseek you already have)` (`mpd plan`, `mpd doctor`, the `prepare` log) | Foldseek is installed on a login node, because compute nodes have no internet | run the command: `mpd fetch foldseek --cluster grace`. With no route to github.com, put `foldseek-linux-avx2.tar.gz` of release 10-941cd33 in reach and run `mpd fetch foldseek --cluster grace --from-file <archive>`; it is checked against the pinned sha256 before anything runs. To use your own Foldseek (for example from conda) `export MPD_FOLDSEEK=/path/to/foldseek` |
| `MPD_FOLDSEEK=<path> is not an executable file (and no fallback is used while it is set)` | the variable points nowhere | fix the path or `unset MPD_FOLDSEEK` |
| `cannot execute <binary>:` or `the pinned Foldseek binary (sha256 verified) does not run here` | a file system mounted `noexec`, or a CPU without AVX2 | install under another `--home`; try `<binary> version` by hand (**[NOT VERIFIED]** on Grace) |
| warning `Foldseek ... is not the pinned release 10-941cd33` | a different build works, but its 3Di strings were not compared with the recorded results | none; its sha256 is recorded in `contexts/structures.json` |
| `for foldseek use --from-file TARBALL (the downloaded foldseek-linux-avx2.tar.gz), not --from-dir` | `--from-dir` is for data and models | use `--from-file` |
| `ProteinGym structures not found under <path>. They are the official AF2 models (<file>, <size> MB), part of the ProteinGym download. Run on a login node: <command>` | the structures come with the ProteinGym data; the folder is missing or holds no `.pdb` file (a fetch with `--no-unpack` leaves the zip packed) | run the command shown: `mpd fetch proteingym --cluster grace` (files that are already there with the right checksum are not downloaded again) |
| `VenusMutHub structures not found under <path>. The structure set is not published yet, so SaProt can use VenusMutHub only with a copy of the deposit:` | SaProt on VenusMutHub needs the structure deposit, which is in preparation | with a copy of the deposit: `mpd fetch venus-structures --cluster grace --from-dir <folder with the deposit files>`; ESM2 students and everything on ProteinGym need nothing extra |
| `structure(s) could not be turned into state strings:` followed by up to eight lines (the `prepare` log) | every unusable structure is listed at once: `structure file <path> not found`, `<assay>: ...` | fix the listed files; nothing was written, so running `prepare` again starts clean |
| `structure residues differ from the sequence (more than`, `the <n>-residue structure does not contain the <m>-residue sequence; insertions and deletions are not aligned.`, `the sequence occurs <n> times in the <m>-residue structure (ambiguous)`, `the structure has <n> residues but the sequence has <m>; without a structure range the structure must cover the whole sequence` | a structure must match its assay sequence residue for residue (at most three single-residue differences, which are recorded); insertions and deletions are not aligned | give the assay a structure range (ProteinGym `pdb_range`) or another structure, or leave the assay out |
| `no chains (does the file have protein ATOM records with N, CA and C atoms?)`, `no CA atoms in ATOM records`, `no usable B-factor in columns 61-66` | the structure file is not a usable protein PDB | use another structure file |
| `foldseek exited with status <n> on <file>:`, `foldseek wrote no output for <file>` | Foldseek could not read that structure | try `<binary> structureto3didescriptor <file> /tmp/out.tsv` by hand, and use another structure |
| `of <file> disagree, first at residue <n>` | the residues in the PDB and in Foldseek's output differ | use another structure file, or `MPD_3DI_ENGINE=biotite` (a NumPy port, [SAPROT.md](SAPROT.md)) |
| `share a sequence but name different structures; a sequence has one structure per dataset` | two assays of one dataset give one sequence two structures | leave one of the assays out |
| `sequence(s) of the context tables are not in the <dataset> benchmark` | the contexts and the benchmark data do not belong together (a data folder from another version) | `mpd fetch <dataset>` again, in a new campaign |
| `vocab.txt has sha256 <x>..., expected <y>... (not a SaProt AA+3Di vocabulary)` | the folder given as SaProt weights is not a SaProt model | `mpd fetch model saprot-650m`, or fix `MPD_MODEL_SAPROT_650M` |
| `MPD_3DI_ENGINE=biotite needs biotite: pip install biotite` / `<variable>=biotite but biotite is not installed` | the optional 3Di engine is not installed | `pip install biotite` in the environment, or unset `MPD_3DI_ENGINE` |
| `must be 'foldseek' or 'biotite'; got <x>` | a bad `MPD_3DI_ENGINE` | `foldseek` (the default) or `biotite` |
| `not found (--from-file); expected the <archive> of release <release>` | the archive given to `mpd fetch foldseek --from-file` does not exist | the downloaded `foldseek-linux-avx2.tar.gz` |
| `could not download <url>: <reason>. On a cluster without internet access download the file elsewhere and pass it with --from-file <archive>.` | no route to github.com from this node | download the archive elsewhere, copy it over, and run `mpd fetch foldseek --from-file <archive>` |
| `; it is not release <release>` / `does not match the pinned sha256; nothing was run` | the archive or the binary in it is not the pinned release 10-941cd33; nothing was executed | download the archive again |
| `set the assay's structure_chain to choose one` | structure files must hold one chain, or the assay must name it | use another structure |
| `weights for <key> are missing (<path>). Run on a login node: <command>` in the `prepare` log | `prepare` checks every model of the run first, so the CPU job fails within minutes instead of the GPU job hours later | run the command on a login node, then `mpd submit` the same flags with `--force` |
| `reads protein structure, but` `does not exist. The prepare stage writes it: delete` | `prepare` did not finish, or the file was removed | follow the sentence: delete `contexts/MANIFEST.json` and repeat your `mpd submit` command with `--force` |
| `not resuming: all.partial: it was written by a run with a different configuration` after a TIMEOUT | something the run's configuration records changed, for SaProt also the structure strings | nothing to do: the unit starts again from update 0 and the old attempt is kept as `*.incomplete-<time>` |
| `Some weights of EsmForMaskedLM were not initialized from the model checkpoint ... esm.contact_head.regression ... position_embeddings` (job log) | expected for SaProt: it uses rotary positions and a masked-language-model head, and those layers are unused | none |
| SaProt results on VenusMutHub differ from the earlier study's ESMFold cells | the earlier code masked every residue of ESMFold models (a scale error); `mpd` masks correctly | `MPD_STRUCTURE_LEGACY_MASK=1` (job: `--env MPD_STRUCTURE_LEGACY_MASK=1`) reproduces the old inputs for a regression only; it makes a new campaign. See [SAPROT.md](SAPROT.md) |

## What to paste in an issue

Paste the output of

```bash
mpd doctor --report
```

It prints the `mpd` and Python versions and the check table with account numbers, your user name, your home and scratch paths,
e-mail addresses and tokens removed, and without the lines `showquota` printed (**[CPU]**; the removal is a best effort, so look it over before
you post). Add:

- the exact command you ran, with `<ACCOUNT>` in place of your account number, and the campaign id;
- the last lines of the failing stage's `.err` (`mpd logs <campaign> <stage> --err`; account numbers, e-mail addresses and your user name are hidden, and `--raw` shows the file as written, so do not paste that) and the `provenance` block of its `COMPLETE.json`;
- for a Slurm problem, the output of `sbatch --version` and the rejected line.

Do not paste account numbers, user names, tokens, or raw `squeue`, `sacct`, `showquota` or `myproject` output (they contain your user
name and project names).
