## What and why

## Checked

- [ ] `ruff check src tests scripts`
- [ ] `python -m pytest -q -m "not slow and not gpu and not rehearsal"` (CPU; tiny random models, no network, no Slurm)
- [ ] `python -m pytest -q -m rehearsal` (runs the rendered job scripts on the CPU; needed when you touch `src/mpd/slurm`, `src/mpd/stages` or the setup script)
- [ ] No data, model weights, logs, account numbers, user names or personal paths added
- [ ] Tests do not call a real `sbatch` or need a GPU (use the fake Slurm tools in `tests/fx_slurm.py`)
- [ ] User-visible behaviour changed: `CHANGELOG.md` and the docs are updated
- [ ] Anything about licence-restricted models (for example AIDO) keeps the check in `RunSpec.validate`

## Verified where?

Say what you ran and where: CPU tests only, `sbatch --test-only` on a cluster, or a real job. Do not claim more than that.
