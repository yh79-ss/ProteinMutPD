# Changelog

All notable changes are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
the project will follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) from its first tagged release.

## [Unreleased]

First public version (`0.1.0.dev0`). How far it has been checked:

- Tested on CPU with tiny random models and fake `sbatch`, `squeue`, `scancel`, `sacct`, `module` and `nvidia-smi` tools. On
  2026-11-09 the suite passed with the versions pinned in `constraints/grace.txt` on Python 3.11 (the declared minimum and the
  version CI uses), 3.12 (Ubuntu's own build) and 3.13; with torch 2.7.0 (the floor) and with the newest versions the declared
  ranges allow (torch 2.15, transformers 4.57, peft 0.21); in an environment without torch (the tests that need it skip); and from
  the unpacked sdist.
- The job scripts of the Vision profile were accepted by `sbatch --test-only` on the Vision cluster (Slurm 25.05.6, H200
  GPUs, 2026-11-09): 24 plans, 132 scripts, 12 distinct `#SBATCH` header shapes, nothing queued. That says nothing about
  Grace, which has A100 GPUs, a Slurm version nobody has checked, and a typed `gpu:a100:N` request taken from HPRC documents.
- All rendered job scripts, for both profiles, pass `bash -n` and a list of invariants (`tests/test_render_matrix.py`).
- **Never run on TAMU Grace, never run on a GPU through Slurm, and the GitHub Actions workflow has not run on GitHub yet.**
  `docs/GRACE.md` lists what is still unverified.

### Added

- Cluster profiles for TAMU Grace and Vision, and `mpd plan`, `mpd submit`, `mpd status`, `mpd logs`, `mpd cancel`,
  `mpd results`, `mpd assays`, `mpd models` and `mpd clusters`. Submission writes a `SUBMITTED.json` marker after every accepted
  job, retries only after `squeue` has answered, and `mpd cancel` reads the marker. A cost guard refuses a submission whose
  estimated upper bound is above `--max-su` (default 10,000 service units). All resource figures are estimates, not measurements.
- `mpd submit --dry-run` writes the scripts, runs the preflight and, when `sbatch` exists, `sbatch --test-only` (it queues
  nothing); without `sbatch` that check is reported as skipped.
- `mpd status` prints a table of the stages (state, elapsed time, Slurm's reason, job id, exit code, finished tasks) and, for
  every stage that failed, timed out, ran out of memory, was cancelled or waits, says what happened and gives the next command.
- Recovery: the campaign id is `<recipe>__<8 hex>`, and the hash covers only what changes results (evaluation data, cluster,
  GPU type, number of training GPUs, score shards, and `MPD_DTYPE`, `MPD_TF32`, `MPD_STRUCTURE_LEGACY_MASK`). Walltimes,
  `--array-throttle`, `--sbatch-arg`, `--module`, other `--env` pairs and the account are not part of it, so recovery from a
  TIMEOUT or a bad node is the original command plus the changed option plus `--force`, in the same campaign directory.
  `--force` keeps queued, running and finished stages and queues the missing ones. After `mpd cancel`, or after a submission
  that ended `failed`, the same command queues only the stages whose output files are missing.
- `scripts/setup_env.sh`: builds a virtual environment from a module-provided Python, pinned by `constraints/grace.txt`,
  and records the modules it used so that job scripts load the same ones.
- `mpd doctor` (with `--report` for a paste-safe summary) and a preflight that `plan`, `submit` and `doctor` share.
- `mpd fetch proteingym | venusmuthub | venus-structures | foldseek` and `mpd fetch model KEY ...`: pinned revisions,
  sha256 checks, resumable downloads. Compute nodes have no internet, so jobs read weights from `<home>/models/<key>`.
- ProteinGym v1.3 and VenusMutHub loaders with a label firewall (only the evaluate stage reads labels).
- ESM2 prepare, teacher, train, score and evaluate stages for `direct-all`, `direct-one`, `score-all` and `score-one`:
  training resumes after a timeout, a GPU stage without a usable GPU stops with exit code 31, and the stage records carry
  the versions of mpd, torch, transformers and peft.
- SaProt-35M and SaProt-650M are available as teacher and student (`mpd models` shows `available`): `prepare` builds the
  Foldseek 3Di structure strings, `mpd fetch foldseek` installs a pinned Foldseek release. CPU-tested only: tiny random
  models and a fake Foldseek, plus opt-in checks on real data with the maintainer's own Foldseek build. The pinned
  Foldseek release has never been executed; SaProt on VenusMutHub needs a structure set that is not published yet
  (`mpd fetch venus-structures --from-dir`); bf16, GPU memory and walltime for SaProt are estimates. `docs/SAPROT.md`
  has the details, including the caution about cross-family pairs (SaProt with ESM2), which were reasoned and unit-tested
  on random numbers and never measured.
- AIDO is not in this release: the licence lock refuses an AIDO teacher for a non-AIDO student and no AIDO weights or
  outputs are shipped ([docs/LICENSES.md](docs/LICENSES.md)).
- CPU rehearsal of the rendered job scripts: `tests/test_rehearsal.py` and `scripts/maintainers/rehearse_campaign.py` run the
  scripts that `mpd submit` writes, with `bash`, in dependency order and for every array task, on the CPU with tiny random
  models, a fake `module`, `nvidia-smi` and `SLURM_*` variables and a tripwire that refuses any network connection (Grace and
  Vision profiles, all four methods, ESM2 and SaProt students, a training job stopped and started again). It says nothing about
  Grace's modules, drivers, NCCL, memory, queue or Slurm version.
- Render-matrix tests: `tests/test_render_matrix.py` renders 96 plans (Grace A100 and A40, Vision H200; the four methods;
  cached and live teachers; three student sizes; two-GPU training on Grace) into 528 job scripts and checks each with
  `bash -n` and a list of invariants (no `--export` or `--get-user-env`, the typed GPU request of the profile, `set -u` after
  the module block, the exit-31 GPU guard, array bounds). It needs no torch.
- Packaging, tests and community files: pinned dependency sets in `constraints/`, ruff configuration, a GitHub Actions
  workflow (lint, tests, the rehearsal of rendered job scripts, CLI without torch, docs checks), issue and pull request
  templates, `CONTRIBUTING.md`, `SECURITY.md`, `CITATION.cff` and `tests/test_packaging.py` (builds the wheel and the sdist,
  pins their exact contents, and runs the login-node commands from the unpacked wheel; the `test` extra pulls the setuptools
  that build needs).

### Changed

- Python 3.11 or newer (the first drafts allowed 3.10).
- Dependencies: `pandas`, `accelerate` and `safetensors` are no longer declared (nothing imports them directly;
  `transformers` and `peft` bring the last two). `torch>=2.7`, `transformers>=4.51,<5`, `peft>=0.13`.
- The Grace profile sets neither `--export` nor `--get-user-env`. A job inherits the environment of the shell that ran `sbatch`;
  its script drops `PYTHONPATH`, `PYTHONHOME` and every `MPD_*` variable and exports the planned ones, so `MPD_ALLOW_HUB` or
  `MPD_STRUCTURE_LEGACY_MASK` in a login shell never reach a job. `LD_LIBRARY_PATH`, `VIRTUAL_ENV` and the like are inherited, as in
  the Grace job scripts that ran. (The Vision cluster's Slurm rejects `--get-user-env=L`; whether Grace's does was not checked.)
- Grace trains on one GPU by default: two-GPU training has only been run on the CPU, never with NCCL on a GPU.
- `rtx` and `t4` GPUs have no native bf16, so the planner refuses them for training unless `--allow-unsupported-gpu` is given.

### Known gaps and follow-ups

- Not verified on Grace: the environment script, module names, header flags, GPU behaviour (including two-GPU NCCL runs),
  walltime and memory estimates, `sacct` and `squeue` output formats, and whether jobs are better off with `LD_LIBRARY_PATH`
  unset after `module purge` (the scripts leave it as the submitting shell has it).
- CI: no Python-version matrix, no minimum-versions job, and no scheduled job against the newest compatible dependencies
  (the workflow runs Python 3.11 with the pinned set only). Actions are pinned to major tags, not to commit SHAs.
- Author list and paper reference in `CITATION.cff` are placeholders (TODO). The repository URL in `pyproject.toml` and
  `CITATION.cff` returned 404 to an anonymous request on 2026-11-09 (not created yet, or private).
- No release process: nothing is published to PyPI and no version has been tagged.
