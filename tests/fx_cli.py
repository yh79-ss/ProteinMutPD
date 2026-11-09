"""Shared pieces for the command-line tests: a hermetic environment, fake interpreters, tiny data, campaign builders.

    from fx_cli import cli_env, run_cli          # noqa: F401  (cli_env is a pytest fixture)

`cli_env(tmp_path, monkeypatch)` makes the process look like a login shell with nothing installed: no MPD_*, SCRATCH or
SLURM_* variables, HOME and PATH pointing into tmp_path (PATH holds only symlinks to a few coreutils, so no real Slurm,
showquota or myproject can be reached), and a made-up user name. A test that wants Slurm requests `fake_slurm` after it.
Nothing here calls the network, sbatch or a GPU.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path

import pytest

from mpd.cli import main

FAKE_USER = "jdoe42"                 # a made-up user name: scrub tests assert it never survives a report
FAKE_ACCOUNT = "123456789012"
TOOLS = ("bash", "sh", "env", "dirname", "basename", "mkdir", "cat", "ls", "find", "wc", "tr", "du", "cut", "awk", "rm", "cp",
         "mv", "sed", "grep", "head", "tail", "date", "uname", "chmod", "touch", "sort", "readlink", "id", "true", "false")


@dataclass
class Env:
    tmp: Path
    home: Path                       # the mpd home: tmp/mpd
    userhome: Path                   # $HOME: tmp/userhome (must stay empty: nothing may write there)
    bin: Path                        # the whole PATH

    def link(self, name: str, target) -> Path:
        """Put `target` on PATH under `name`."""
        path = self.bin / name
        path.unlink(missing_ok=True)
        path.symlink_to(target)
        return path


@pytest.fixture
def cli_env(tmp_path, monkeypatch):
    bin_dir = tmp_path / "hermetic_bin"
    bin_dir.mkdir()
    for tool in TOOLS:
        found = shutil.which(tool)
        if found:
            (bin_dir / tool).symlink_to(found)
    userhome = tmp_path / "userhome"
    userhome.mkdir()
    for var in list(os.environ):
        if var.startswith(("MPD_", "SLURM_")) or var in ("SCRATCH", "BASH_ENV", "LOADEDMODULES", "PYTHONPATH", "PYTHONHOME",
                                                         "HF_TOKEN", "TMPDIR", "XDG_CACHE_HOME"):
            monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.setenv("HOME", str(userhome))
    monkeypatch.setenv("USER", FAKE_USER)
    monkeypatch.setenv("LOGNAME", FAKE_USER)
    monkeypatch.setenv("COLUMNS", "120")
    return Env(tmp_path, tmp_path / "mpd", userhome, bin_dir)


def run_cli(capsys, *argv):
    """Run `mpd <argv>` in this process: (exit status, stdout, stderr). SystemExit is turned into a status."""
    capsys.readouterr()
    try:
        rc = main([str(a) for a in argv])
    except SystemExit as exc:
        rc = exc.code if isinstance(exc.code, int) else 1
        if isinstance(exc.code, str):
            print(exc.code, file=sys.stderr)
    out = capsys.readouterr()
    return rc, out.out, out.err


def write_shim(env: Env, name: str, stdout: str = "", *, rc: int = 0, stderr: str = "") -> Path:
    """A program on the test PATH that prints fixed text, e.g. showquota or myproject (their real output format is not known)."""
    path = env.bin / name
    path.write_text(f"#!/bin/sh\ncat <<'EOF_OUT'\n{stdout}\nEOF_OUT\ncat >&2 <<'EOF_ERR'\n{stderr}\nEOF_ERR\nexit {rc}\n", encoding="utf-8")
    path.chmod(0o755)
    return path


# ---------------------------------------------------------------- a fake job interpreter
_FAKE_PY = '''
import json, os, sys
here = os.path.dirname(os.path.abspath(__file__))
cfg = json.load(open(os.path.join(here, "fake_python.json")))
args = sys.argv[1:]
code = args[args.index("-c") + 1] if "-c" in args else ""
if "importlib.metadata" in code:
    print(json.dumps(cfg["probe"]))
elif "torch.cuda" in code:
    if cfg.get("torch_error"):
        print(cfg["torch_error"], file=sys.stderr)
        sys.exit(1)
    print(json.dumps(cfg["torch"]))
'''


def make_fake_python(root: Path, *, version: str = "3.11.7", packages: dict | None = None, mpd_importable: bool = True,
                     torch: dict | None = None, torch_error: str | None = None, probe_error: str | None = None) -> Path:
    """`root/bin/python`: answers the version probe of `mpd doctor` (and the `--deep` torch import) without any real package."""
    packages = {"proteinmutpd": "0.1.0.dev0", "torch": "2.10.0", "transformers": "4.51.0", "peft": "0.13.2"} \
        if packages is None else packages
    (root / "bin").mkdir(parents=True, exist_ok=True)
    (root / "fake_python.py").write_text(_FAKE_PY, encoding="utf-8")
    (root / "fake_python.json").write_text(json.dumps({
        "probe": {"python_version": version, "mpd_importable": mpd_importable, "packages": packages},
        "torch": torch or {"torch": packages.get("torch", "2.10.0"), "cuda_build": "12.8", "cuda_available": False, "device": None},
        "torch_error": torch_error}), encoding="utf-8")
    body = f'#!/bin/sh\nexec "{sys.executable}" -I -S "{root / "fake_python.py"}" "$@"\n'
    if probe_error:
        body = f'#!/bin/sh\necho "{probe_error}" >&2\nexit 3\n'
    path = root / "bin" / "python"
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)
    return path


def write_record(home: Path, python: Path, *, cluster: str = "grace", modules=("Anaconda3/2024.02-1",), packages: dict | None = None,
                 constraints: str | None = None, sha256: str | None = None, created: str = "2026-11-09T12:00:00Z") -> Path:
    """`<home>/env/mpd_env.json` in the schema-1 layout of CONTRACTS.md section 3."""
    packages = {"torch": "2.10.0", "transformers": "4.51.0", "peft": "0.13.2"} if packages is None else packages
    path = home / "env" / "mpd_env.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema": 1, "created": created, "cluster": cluster, "home": str(home), "python": str(python),
                                "python_version": "3.11.7", "modules": list(modules), "mpd_version": "0.1.0.dev0",
                                "constraints": constraints, "constraints_sha256": sha256, "packages": packages}, indent=2) + "\n",
                    encoding="utf-8")
    return path


# ---------------------------------------------------------------- tools for scripts/setup_env.sh
_SETUP_PYTHON = r"""#!/bin/bash
# Stands in for the Python that scripts/setup_env.sh builds the venv from; the venv's own python is a copy of this file.
printf 'python %s\n' "$*" >> "${FAKE_LOG:?}"
here=$(cd "$(dirname "$0")/.." && pwd)
case "$1" in
  -c)
    case "$2" in
      *version_info*) echo "${FAKE_PY_VERSION:-3.11.7}" ;;
      *ensurepip*) [ "${FAKE_NO_VENV:-0}" = 1 ] && exit 1 ;;
      *mpd.envrecord*) [ "${FAKE_NO_ENVRECORD:-0}" = 1 ] && exit 1 ;;
    esac
    exit 0 ;;
  -m)
    mod=$2; shift 2
    case "$mod" in
      venv)
        mkdir -p "$1/bin" && cp "$0" "$1/bin/python" && chmod +x "$1/bin/python"
        echo "home = /fake" > "$1/pyvenv.cfg"; printf '# activate\nexport VIRTUAL_ENV=%s\n' "$1" > "$1/bin/activate" ;;
      pip)
        printf 'pip %s\n' "$*" >> "$FAKE_LOG"
        printf 'pip env PIP_CACHE_DIR=%s TMPDIR=%s PYTHONNOUSERSITE=%s\n' "$PIP_CACHE_DIR" "$TMPDIR" "$PYTHONNOUSERSITE" >> "$FAKE_LOG"
        [ "${FAKE_PIP_FAIL:-0}" = 1 ] && { echo "ERROR: no matching distribution" >&2; exit 1; }
        mkdir -p "$here/lib/site-packages/pkg" && for i in 1 2 3; do echo x > "$here/lib/site-packages/pkg/f$i.py"; done ;;
      mpd.envrecord)
        printf 'envrecord %s\n' "$*" >> "$FAKE_LOG"
        PYTHONPATH="$FAKE_SRC" exec "$FAKE_REAL_PYTHON" -s -m mpd.envrecord "$@" ;;
      mpd)
        printf 'mpd %s\n' "$*" >> "$FAKE_LOG"; echo "fake mpd doctor output"; exit "${FAKE_DOCTOR_RC:-0}" ;;
    esac
    exit 0 ;;
  -) exec "$FAKE_REAL_PYTHON" -I - "${@:2}" ;;
esac
"""


def make_setup_python(root: Path) -> Path:
    """A bash stand-in for python: logs every call to $FAKE_LOG, builds a fake venv, and runs the real mpd.envrecord."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / "python3"
    path.write_text(_SETUP_PYTHON, encoding="utf-8")
    path.chmod(0o755)
    return path


def write_module_env(root: Path, available=("Anaconda3/2024.02-1", "Python/3.11.5")) -> tuple[Path, Path]:
    """A BASH_ENV file that defines `module` like Lmod does (purge, load; an unknown module only prints an error and returns 0)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "modules.txt").write_text("".join(f"{m}\n" for m in available), encoding="utf-8")
    env_file = root / "bash_env.sh"
    env_file.write_text("""module() {
  echo "module $*" >> "${MODULE_LOG:-/dev/null}"
  case "$1" in
    purge) unset LOADEDMODULES ;;
    load) shift
      for m in "$@"; do
        if grep -qx -- "$m" "${MODULES_AVAILABLE:?}"; then export LOADEDMODULES="${LOADEDMODULES:+$LOADEDMODULES:}$m"
        else echo "Lmod has detected the following error: The following module(s) are unknown: \"$m\"" >&2; fi
      done ;;
  esac
  return 0
}
export -f module
""", encoding="utf-8")
    return env_file, root / "modules.txt"


# ---------------------------------------------------------------- data and weights
def write_proteingym(home: Path, *, fetched: bool = True) -> dict[str, str]:
    """The three-assay ProteinGym stand-in of tests/tinydata.py in the layout `mpd fetch proteingym` produces."""
    from mpd.data.sources import PROTEINGYM
    from tinydata import write_tiny_proteingym
    root = home / "data" / PROTEINGYM.dirname
    root.mkdir(parents=True, exist_ok=True)
    seqs = write_tiny_proteingym(root)
    if fetched:
        (root / "FETCHED.json").write_text(json.dumps({"source": "proteingym"}), encoding="utf-8")
    return seqs


def write_venus(home: Path, ids=("ACT_one", "BIND_two"), *, fetched: bool = True) -> None:
    """A VenusMutHub folder with empty CSVs: enough for presence checks and the assay list."""
    from mpd.data.sources import VENUSMUTHUB
    root = home / "data" / VENUSMUTHUB.dirname
    for i, aid in enumerate(ids):
        folder = root / "single_mutant" / ("activity" if i % 2 == 0 else "ppi_binding")
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{aid}.csv").write_text("mutant,mutated_sequence,fitness_score\n", encoding="utf-8")
    if fetched:
        (root / "FETCHED.json").write_text(json.dumps({"source": "venusmuthub", "files": len(ids)}), encoding="utf-8")


def override_weights(env: Env, monkeypatch, *keys: str) -> None:
    """Make `mpd` treat these models as ready, the way a user does: MPD_MODEL_<KEY> pointing at a folder with a config."""
    for key in keys:
        folder = env.tmp / "weights" / key
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "config.json").write_text("{}", encoding="utf-8")
        monkeypatch.setenv("MPD_MODEL_" + key.upper().replace("-", "_"), str(folder))


def ready_home(env: Env, monkeypatch, *, venus: bool = True) -> Path:
    """Everything `plan`/`submit` check is satisfied: job interpreter and record, data, weights of esm2-3b and esm2-650m."""
    python = make_fake_python(env.home / "env")
    write_record(env.home, python)
    write_proteingym(env.home)
    if venus:
        write_venus(env.home)
    override_weights(env, monkeypatch, "esm2-3b", "esm2-650m")
    return env.home


def recipe(home, *extra, cluster: str = "grace", eval_data=("proteingym",), method: str = "score-all") -> list[str]:
    """The arguments every plan/submit test starts from."""
    return ["--cluster", cluster, "--home", str(home), "--teacher", "esm2-3b", "--student", "esm2-650m", "--train-data", "proteingym",
            "--method", method, "--eval-data", *eval_data, *map(str, extra)]


# ---------------------------------------------------------------- campaign directories
def write_summary(campaign_dir: Path, results: list[dict], *, objective: str = "score", primary: str = "dual") -> Path:
    """`evaluate/summary.json` in the layout stages/evaluate.py writes."""
    path = campaign_dir / "evaluate" / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"campaign": campaign_dir.name, "objective": objective, "primary_readout": primary, "results": results,
                                "dms_labels_opened": True}, indent=2), encoding="utf-8")
    return path


def result_rows(dataset: str = "proteingym", unit: str = "all", assays: int = 3, base=(0.40, 0.30), tuned=(0.45, 0.38),
                primary: str = "dual") -> list[dict]:
    """base-student rows plus one trained unit, for both readouts (delta first)."""
    rows = []
    for readout, b, t in zip(("delta", "dual"), base, tuned):
        rows.append({"dataset": dataset, "unit": "base", "readout": readout, "assays": assays, "macro_spearman": b,
                     "primary": readout == primary})
        rows.append({"dataset": dataset, "unit": unit, "readout": readout, "assays": assays, "macro_spearman": t,
                     "primary": readout == primary, "base_macro_same_assays": b, "delta_vs_base": t - b})
    return rows


def write_adapter(campaign_dir: Path, unit: str = "all", step: int = 100) -> Path:
    unit_dir = campaign_dir / "train" / unit
    ckpt = unit_dir / f"checkpoint_{step:05d}"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "adapter_config.json").write_text("{}", encoding="utf-8")
    (unit_dir / "SELECTED.json").write_text(json.dumps({"step": step}), encoding="utf-8")
    return ckpt


def write_logs(campaign_dir: Path, files: dict[str, str]) -> None:
    """{file name: text} under <campaign>/logs, with mtimes increasing in the order given."""
    logs = campaign_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for i, (name, text) in enumerate(files.items()):
        path = logs / name
        path.write_text(textwrap.dedent(text), encoding="utf-8")
        os.utime(path, (1_700_000_000 + i, 1_700_000_000 + i))
