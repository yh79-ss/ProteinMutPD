# Contributing

Thanks for helping. ProteinMutPD is alpha research software: small, well-tested changes are the easiest to review.

## Rules that every change follows

- **Tests use no GPU, no Slurm and no network.** They run tiny random models and fake `sbatch`, `squeue`, `scancel`,
  `sacct` and `module` tools (`tests/fx_slurm.py`). A test must never call a real `sbatch`, and nobody needs to submit a
  job to review a change.
- **No secrets, no identity, no bulk files.** Never commit a Slurm account number, a user name, a real home or project
  path, a token, a key, data, model weights, run directories or logs. Examples use the placeholder account
  `123456789012`, the user `<netid>` and the path `/scratch/user/<netid>/mpd`. `tests/test_packaging.py` scans the tree
  for the common leaks and for files over 1 MB.
- **Label firewall.** Only `src/mpd/stages/evaluate.py` may open labels; `tests/test_stages.py` scans `stages/*.py`
  for `.labels(`. Keep structure and model code out of that pattern.
- **The planner stays torch-free.** `mpd plan`, `submit`, `doctor`, `status`, `fetch` and the modules they import must
  not import torch (a test enforces it).
- **Licence-restricted models stay gated.** AIDO teachers are refused for non-AIDO students in `RunSpec.validate`
  ([docs/LICENSES.md](docs/LICENSES.md)). Do not relax that check or add model outputs to the repository.
- **Say what was verified.** Claim only what you ran: CPU tests, `sbatch --test-only` on a named cluster, a real job, or
  "not verified". Nothing in this repository has been run on Grace or on a GPU through Slurm yet, so do not write
  "works on Grace".

## Set up

```bash
git clone https://github.com/yh79-ss/ProteinMutPD.git && cd ProteinMutPD
python3 -m venv .venv && . .venv/bin/activate
# a CPU build of torch is enough for the tests; the pins are in constraints/
python -m pip install -c constraints/grace.txt --extra-index-url https://download.pytorch.org/whl/cpu torch
python -m pip install -c constraints/grace.txt -c constraints/ci.txt -e '.[train,structure,test,dev]'
python -m pytest -q -m "not slow and not gpu"
ruff check src tests scripts
```

The suite has been run on Python 3.11, 3.12 and 3.13 (CI uses 3.11); 3.11 is the declared minimum. Without `[train]` the planner, `mpd fetch` and the data tests still
run, and the tests that need torch are skipped.

## Tests

- `tests/fx_<area>.py` hold the shared fixtures (fake Slurm, tiny models, tiny structure homes). A test module imports
  a fixture by name with `# noqa: F401`; keep the comment, or `ruff check --fix` deletes the import and pytest no
  longer finds the fixture.
- The rendered Slurm scripts are compared with `tests/golden_render/*.sbatch`. After a deliberate change run
  `UPDATE_GOLDEN=1 python -m pytest tests/test_render.py -k goldens` and review the diff.
- Opt-in tests skip by default: `MPD_TEST_NETWORK=1` (downloads a 31 MB model), `MPD_TEST_VENUS_HOME`,
  `MPD_TEST_PG_HOME`, `MPD_FOLDSEEK` and `MPD_MODEL_SAPROT_35M` (see the docstring of `tests/test_saprot_slow.py`), and the
  `slow` and `gpu` markers.

## Lint, dependencies, packaging

- `ruff check src tests scripts`: the configuration is in `pyproject.toml`. It enables bug checks only (syntax, imports,
  undefined names, unused variables) and no formatter.
- Dependency ranges in `pyproject.toml` are loose; `constraints/grace.txt` pins the set the suite was run with. If you
  raise a pin, run the whole suite on the new set and say so in the pull request.
- New data files inside `src/mpd` (YAML, TSV) need an entry under `[tool.setuptools.package-data]`; the packaging test
  fails if one is missing from the wheel.

## Adding things

- A teacher or student model: [docs/ADDING_A_MODEL.md](docs/ADDING_A_MODEL.md).
- A cluster profile: [docs/CLUSTERS.md](docs/CLUSTERS.md).
- A change to a recipe default needs a note in `CHANGELOG.md` and, if reported numbers move, in
  [docs/DATA.md](docs/DATA.md).

## Commits and pull requests

- One logical change per commit. Subject in the imperative, at most about 70 characters ("Refuse a home under /home");
  say why in the body.
- Add a line under `[Unreleased]` in `CHANGELOG.md` for anything a user can see.
- Fill in the pull request template: what changed, what you ran, and where it was verified.

## What CI does

The workflow in `.github/workflows/ci.yml` runs ruff, the test suite on Python 3.11 with CPU torch and the pinned set, a
check that the built wheel works on a login-node-style environment without torch (`mpd plan` and `mpd submit --dry-run`
write scripts but submit nothing), and a check of the Markdown links and shell scripts. Runners have no GPU and no Slurm,
so CI cannot show that a job runs on a cluster.

By contributing you agree that your contribution is licensed under the MIT licence of this repository. Data and model
weights keep their own licences ([docs/LICENSES.md](docs/LICENSES.md)).
