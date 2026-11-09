# Changelog

All notable changes are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
the project will follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) from its first tagged release.

## [Unreleased]

First public version (`0.1.0.dev0`). How far it has been checked:

- Tested on CPU (Python 3.11, 3.12 and 3.13, with the versions pinned in `constraints/grace.txt`), with tiny random models
  and fake `sbatch`, `squeue`, `scancel`, `sacct` and `module` tools. Python 3.11 is the declared minimum; older
  versions were not tried.
- Rendered job scripts pass `bash -n`; the Vision profile's scripts were accepted by `sbatch --test-only` on the Vision
  cluster (2026-11-09).
- **Never run on TAMU Grace, never run on a GPU through Slurm, and the GitHub Actions workflow has not run on GitHub yet.**
  The README status table and `docs/GRACE.md` list what is still unverified.

### Added

- Cluster profiles for TAMU Grace and Vision, and `mpd plan`, `mpd submit` (with `--dry-run`), `mpd status`,
  `mpd logs`, `mpd cancel` and `mpd results`. Submission writes a `SUBMITTED.json` marker after every accepted job, retries
  only after `squeue` has answered, and `mpd cancel` reads the marker.
- `scripts/setup_env.sh`: builds a virtual environment from a module-provided Python, pinned by `constraints/grace.txt`,
  and records the modules it used so that job scripts load the same ones.
- `mpd doctor` (with `--report` for a paste-safe summary) and a preflight that `plan`, `submit` and `doctor` share.
- `mpd fetch proteingym | venusmuthub | venus-structures | foldseek` and `mpd fetch model KEY ...`: pinned revisions,
  sha256 checks, resumable downloads. Compute nodes have no internet, so jobs read weights from `<home>/models/<key>`.
- ProteinGym v1.3 and VenusMutHub loaders with a label firewall (only the evaluate stage reads labels).
- ESM2 prepare, teacher, train, score and evaluate stages for `direct-all`, `direct-one`, `score-all` and `score-one`:
  training resumes after a timeout, a GPU stage without a usable GPU stops with exit code 31, and the stage records carry
  the versions of mpd, torch, transformers and peft.
- SaProt: library code for Foldseek 3Di structure inputs and a student wrapper, tested on CPU with a fake Foldseek.
  `mpd models` shows whether the `saprot-*` models can be selected, and `docs/SAPROT.md` says what is and is not verified.
- Licence handling: AIDO teachers are refused for non-AIDO students and no AIDO outputs are shipped
  ([docs/LICENSES.md](docs/LICENSES.md)).
- Packaging, tests and community files: pinned dependency sets in `constraints/`, ruff configuration, a GitHub Actions
  workflow (lint, tests, CLI without torch, docs checks), issue and pull request templates, `CONTRIBUTING.md`,
  `SECURITY.md`, `CITATION.cff` and `tests/test_packaging.py`.

### Changed

- Dependencies: `pandas`, `accelerate` and `safetensors` are no longer declared (nothing imports them directly;
  `transformers` and `peft` bring the last two). `torch>=2.7`, `transformers>=4.51,<5`, `peft>=0.13`.
- The Grace profile sets neither `--export` nor `--get-user-env`; job scripts clean their inherited environment themselves.
- Grace trains on one GPU by default.
- A campaign id now ends with a short hash of every plan option that changes tasks or scripts, so changing only
  `--eval-data`, `--gpu` or `--cluster` makes a new campaign (and retrains).

### Known gaps and follow-ups

- Not verified on Grace: the environment script, module names, header flags, GPU behaviour (including two-GPU NCCL runs),
  walltime and memory estimates.
- CI: no Python-version matrix, no minimum-versions job, and no scheduled job against the newest compatible dependencies
  (the workflow runs Python 3.11 with the pinned set only). Actions are pinned to major tags, not to commit SHAs.
- Author list and paper reference in `CITATION.cff` are placeholders (TODO). The repository URL in `pyproject.toml` and
  `CITATION.cff` returned 404 to an anonymous request on 2026-11-09 (not created yet, or private).
- No release process: nothing is published to PyPI and no version has been tagged.
