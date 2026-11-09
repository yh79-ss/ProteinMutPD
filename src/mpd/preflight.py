"""One preflight for `mpd plan`, `mpd submit` and `mpd doctor`. Standard library plus the planner; never imports torch.

`run_checks` returns `Check` rows, `render` turns them into the table the commands print, and `report` makes a
version that is safe to paste into an issue. The checks owned here are the environment, home, quota, account, data,
assay ids and cost; weights, structures and `sbatch --test-only` come from the modules that know about them.
"""

from __future__ import annotations

import csv
import difflib
import getpass
import hashlib
import importlib
import json
import math
import os
import platform
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

from . import __version__, envrecord
from .checks import Check, counts, has_errors
from .data import sources as data_sources

__all__ = ["Check", "Context", "assay_ids", "check_dataset", "fix_flags", "has_errors", "parse_showquota", "render",
           "report", "run_checks", "scrub"]

ACCOUNT_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
MIN_PYTHON = (3, 11)
MIN_TORCH = (2, 7)             # pyproject: torch>=2.7 (ESM2-3B and SaProt ship pickle weights)
PROBE_TIMEOUT = 60.0
TORCH_TIMEOUT = 900.0          # a cold `import torch` took minutes on Grace (grace_truth section 3)
LOW_FREE_GB = 25.0             # environment (several GB) + ESM2 family (14.7 GB) + data (1.3 GB)
QUOTA_WARN_PERCENT = 80.0
TAGS = {"ok": "[ok]", "warn": "[warn]", "error": "[FAIL]", "skip": "[skip]"}


@dataclass
class Context:
    """What the checks need when there is no plan (`mpd doctor`); `PlanOptions` has the same attributes."""

    cluster: object | None
    home: Path
    python: str | None = None
    account: str | None = None


# ---------------------------------------------------------------- small helpers
def _run(cmd: list[str], timeout: float = 20.0) -> tuple[int, str, str] | None:
    """(returncode, stdout, stderr), or None when the program cannot be started."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    except OSError:
        return None
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout:.0f}s"
    return proc.returncode, proc.stdout, proc.stderr


def _vtuple(version: str) -> tuple[int, ...]:
    return tuple(int(n) for n in re.findall(r"\d+", version.split("+")[0])[:3])


def _first_line(text: str) -> str:
    return next((ln.strip() for ln in text.splitlines() if ln.strip()), "")


def _nearest_existing(path: Path) -> Path:
    path = Path(path)
    while not path.exists() and path != path.parent:
        path = path.parent
    return path


def _cluster_name(cluster) -> str | None:
    """The profile name when it can be passed back as `--cluster NAME`, else None (a custom YAML)."""
    from .slurm.cluster import available_clusters
    name = getattr(cluster, "name", None)
    return name if name in available_clusters() else None


def _default_home(cluster) -> Path | None:
    """The working root the profile yields when no --home is given, or None (custom profile, $SCRATCH unset)."""
    if _cluster_name(cluster) is None:
        return None
    try:
        return Path(cluster.storage_dir(os.environ.get("MPD_HOME")))
    except Exception:
        return None


def _explicit_home(cluster, home: Path) -> bool:
    """True when fix commands must repeat --home, because the default would point somewhere else."""
    default = _default_home(cluster)
    return default is None or Path(home) != default


def fix_flags(cluster, home: Path | None) -> str:
    """` --cluster NAME` and, when the home is not the profile's default, ` --home DIR`, for fix commands."""
    parts: list[str] = []
    if _cluster_name(cluster):
        parts += ["--cluster", cluster.name]
    if home is not None and _explicit_home(cluster, home):
        parts += ["--home", shlex.quote(str(home))]
    return (" " + " ".join(parts)) if parts else ""


def _flags_or_home(cluster, home: Path) -> str:
    return fix_flags(cluster, home) or f" --home {shlex.quote(str(home))}"


def _repo_file(relative: str) -> Path | None:
    for base in (Path(__file__).resolve().parents[2], Path.cwd()):
        candidate = base / relative
        if candidate.is_file():
            return candidate
    return None


# ---------------------------------------------------------------- home, quota, account
def check_home(ctx) -> list[Check]:
    home, cluster = Path(ctx.home), ctx.cluster
    for prefix in getattr(cluster, "forbidden_storage_prefixes", ()) or ():
        if home == Path(prefix) or Path(prefix) in home.parents:
            gb, files = getattr(cluster, "home_max_gb", 10), getattr(cluster, "home_max_files", 10_000)
            return [Check("home", "error", f"{home} is under {prefix} ({gb:g} GB / {files:,} files on "
                                           f"{str(cluster.name).capitalize()}). Use scratch: export MPD_HOME=$SCRATCH/mpd")]
    where = _nearest_existing(home)
    if not os.access(where, os.W_OK | os.X_OK):
        return [Check("home", "error", f"{home} is not writable (checked {where})",
                      "choose a directory you own: export MPD_HOME=$SCRATCH/mpd")]
    try:
        free_gb = shutil.disk_usage(where).free / 1e9
    except OSError:
        free_gb = None
    detail = ("exists" if home.exists() else "will be created") + "; writable"
    if free_gb is not None:
        detail += f"; {free_gb:,.0f} GB free on this filesystem"
    try:
        user_home = Path.home()
        in_home = home == user_home or user_home in home.parents
    except RuntimeError:                                      # no home directory known (some containers)
        in_home = False
    if in_home:
        return [Check("home", "warn", f"{home} ({detail}) is inside your home directory. Ignore this where that has no small quota; "
                                      "on HPRC clusters it does (10 GB / 10,000 files)", "export MPD_HOME=$SCRATCH/mpd")]
    if free_gb is not None and free_gb < LOW_FREE_GB:
        return [Check("home", "warn", f"{home} ({detail}). Low: the environment, ESM2 weights and data need about "
                                      f"{LOW_FREE_GB:.0f} GB", "choose a larger filesystem with --home or MPD_HOME")]
    return [Check("home", "ok", f"{home} ({detail})")]


_NUM = r"(\d[\d,]*(?:\.\d+)?)\s*([KMGTP]?)B?"
_UNIT = {"": 1, "K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12, "P": 1e15}


def parse_showquota(text: str) -> list[tuple[str, float]]:
    """(line, percent used) for each line that names /home or /scratch and shows a percentage or used/limit pairs.

    The format of Grace's `showquota` was not available when this was written, so anything unrecognised is ignored.
    """
    found: list[tuple[str, float]] = []
    for line in text.splitlines():
        m = re.search(r"/(?:home|scratch)\S*", line)
        if not m:
            continue
        rest = line[m.end():]
        percents = [float(p) for p in re.findall(r"(\d+(?:\.\d+)?)\s*%", rest)]
        if not percents:
            nums = [float(n.replace(",", "")) * _UNIT[u.upper()] for n, u in re.findall(_NUM, rest, re.I)]
            if len(nums) in (2, 4):
                percents = [100.0 * nums[i] / nums[i + 1] for i in range(0, len(nums), 2) if nums[i + 1] > 0]
        if percents:
            found.append((line.strip(), max(percents)))
    return found


def check_quota(ctx, verbose: bool = False) -> list[Check]:
    ran = _run(["showquota"]) if shutil.which("showquota") else None
    if ran is None:
        return [Check("quota", "skip", "showquota not found (it exists on HPRC clusters); check your quota yourself")]
    rc, out, err = ran
    text = out if out.strip() else err
    if rc != 0 and not out.strip():
        return [Check("quota", "skip", f"showquota failed ({_first_line(err) or f'exit {rc}'})")]
    rows = parse_showquota(text)
    raw = ("\n" + "\n".join("  " + ln for ln in text.splitlines()[:12])) if verbose else ""
    full = [(ln, p) for ln, p in rows if p >= QUOTA_WARN_PERCENT]
    if full:
        return [Check("quota", "warn", f"showquota reports {full[0][1]:.0f}% used: {full[0][0]}{raw}",
                      "delete old runs or ask for more quota before the disk fills (a full quota can lock you out)")]
    if rows:
        return [Check("quota", "ok", f"showquota: highest usage {max(p for _, p in rows):.0f}%{raw}")]
    return [Check("quota", "ok", "showquota ran but its output was not recognised; run it yourself and look at the file "
                                 f"counts (Grace: home 10,000 files, scratch 250,000){raw}")]


def check_account(ctx) -> list[Check]:
    account = ctx.account or getattr(ctx.cluster, "account", None)
    if not account:
        return [Check("account", "ok", "not set: jobs are charged to your default Slurm account "
                                       "(`myproject` lists accounts; set --account or MPD_ACCOUNT)")]
    if not ACCOUNT_RE.match(account):
        return [Check("account", "error", f"account {account!r} has characters Slurm accounts do not use "
                                          "(letters, digits, '_', '.', '-')", "use the number `myproject` prints")]
    ran = _run(["myproject"]) if shutil.which("myproject") else None
    if ran is None:
        return [Check("account", "ok", f"{account} (not checked against `myproject`: command not found)")]
    if account in ran[1]:
        return [Check("account", "ok", f"{account} is listed by `myproject`")]
    return [Check("account", "warn", f"{account} was not found in the output of `myproject` (its format is not parsed; "
                                     "look at it yourself)", "myproject")]


# ---------------------------------------------------------------- python environment
_PROBE = (
    "import importlib.metadata as md, importlib.util, json, platform\n"
    "out = {'python_version': platform.python_version(), 'mpd_importable': importlib.util.find_spec('mpd') is not None,\n"
    "       'packages': {}}\n"
    "for name in ('proteinmutpd', 'torch', 'transformers', 'peft', 'numpy', 'scipy', 'pyyaml'):\n"
    "    try:\n"
    "        out['packages'][name] = md.version(name)\n"
    "    except md.PackageNotFoundError:\n"
    "        pass\n"
    "print(json.dumps(out))\n")

_TORCH = (
    "import json, torch\n"
    "ok = torch.cuda.is_available()\n"
    "print(json.dumps({'torch': torch.__version__, 'cuda_build': torch.version.cuda, 'cuda_available': ok,\n"
    "                  'device': torch.cuda.get_device_name(0) if ok else None}))\n")


def probe_python(python: str) -> tuple[dict | None, str]:
    """Versions inside `python` without importing torch: (info, "") or (None, reason)."""
    ran = _run([python, "-I", "-c", _PROBE], timeout=PROBE_TIMEOUT)
    if ran is None:
        return None, "cannot be started"
    rc, out, err = ran
    if rc != 0:
        return None, _first_line(err) or f"exited with status {rc}"
    try:
        return json.loads(out.strip().splitlines()[-1]), ""
    except (ValueError, IndexError):
        return None, "gave unexpected output to the version probe"


def check_environment(ctx, deep: bool = False) -> list[Check]:
    home, cluster = Path(ctx.home), ctx.cluster
    default = str(home / "env" / "bin" / "python")
    python = ctx.python or default
    setup = "bash scripts/setup_env.sh" + _flags_or_home(cluster, home)
    found = python if os.sep in python else shutil.which(python)             # a bare name is looked up on PATH, as in the job
    if not (found and os.path.isfile(found) and os.access(found, os.X_OK)):
        if python == default:
            return [Check("python", "error", f"no Python environment at {home / 'env'}. Run: {setup}")]
        return [Check("python", "error", f"interpreter {python} does not exist or is not executable",
                      f"pass an existing --python (or MPD_PYTHON), or run: {setup}")]
    info, why = probe_python(found)
    if info is None:
        return [Check("python", "error", f"{python} {why}", f"{setup} --rebuild")]
    version = info.get("python_version", "?")
    if _vtuple(version) < MIN_PYTHON:
        out = [Check("python", "error", f"{python} is Python {version}; mpd needs "
                                        f"{'.'.join(map(str, MIN_PYTHON))} or newer", setup)]
    else:
        out = [Check("python", "ok", f"{python} (Python {version})")]
    out += _check_packages(python, info, setup)
    out += _check_record(ctx, python, info.get("packages", {}), setup)
    if deep:
        out += _check_torch_import(python)
    return out


def _check_packages(python: str, info: dict, setup: str) -> list[Check]:
    pkgs = info.get("packages", {})
    if not info.get("mpd_importable"):
        return [Check("packages", "error", f"mpd is not installed in {python}; the jobs run `python -m mpd.stages...`", setup)]
    missing = [n for n in ("torch", "transformers", "peft") if n not in pkgs]
    if missing:
        return [Check("packages", "error", f"{', '.join(missing)} not installed in {python} (the GPU stages import them)",
                      f"{setup}   (without --no-torch)")]
    shown = ", ".join(f"{n} {pkgs[n]}" for n in ("torch", "transformers", "peft"))
    if _vtuple(pkgs["torch"]) < MIN_TORCH:
        out = [Check("packages", "warn", f"{shown}; torch is older than {'.'.join(map(str, MIN_TORCH))}, which the "
                                         "ESM2-3B and SaProt weights (pickle files) need", setup)]
    else:
        out = [Check("packages", "ok", shown)]
    installed = pkgs.get("proteinmutpd")
    if installed and installed != __version__:
        out.append(Check("packages", "warn", f"the jobs use mpd {installed} but this command is mpd {__version__} "
                                             "(the environment is a snapshot of the code at setup time)", setup))
    return out


def _check_record(ctx, python: str, pkgs: dict, setup: str) -> list[Check]:
    home = Path(ctx.home)
    path = envrecord.record_path(home)
    rec = envrecord.read(home)
    if rec is None:
        why = envrecord.problem(home) or f"{path} is missing"
        return [Check("env-record", "warn", f"{why}: job scripts cannot load the modules this Python was built with", setup)]
    problems = []
    name = getattr(ctx.cluster, "name", None)
    if name and rec.get("cluster") and rec["cluster"] != name:
        problems.append(f"it was written for cluster {rec['cluster']!r}, not {name!r}, so the planner ignores it")
    if rec.get("home") and os.path.realpath(rec["home"]) != os.path.realpath(home):
        problems.append(f"it was written for {rec['home']}, not {home}, so the planner ignores it")
    if rec.get("python") and str(rec["python"]) != python:
        problems.append(f"it describes {rec['python']}, not {python}")
    for pkg, was in (rec.get("packages") or {}).items():
        now = pkgs.get(pkg)
        if now and was and now != was:
            problems.append(f"{pkg} is {now} now but was {was} at setup (pip was run after setup?)")
    pinned = rec.get("constraints")
    current = (Path(pinned) if os.path.isabs(pinned) else _repo_file(pinned)) if isinstance(pinned, str) and pinned else None
    if current and current.is_file() and rec.get("constraints_sha256") and \
            hashlib.sha256(current.read_bytes()).hexdigest() != rec["constraints_sha256"]:
        problems.append(f"{pinned} changed since the environment was built")
    if problems:
        return [Check("env-record", "warn", f"{path}: " + "; ".join(problems), setup)]
    modules = " ".join(rec.get("modules") or []) or "none"
    return [Check("env-record", "ok", f"modules loaded by the jobs: {modules}; built {str(rec.get('created', '?'))[:10]}")]


def _check_torch_import(python: str) -> list[Check]:
    ran = _run([python, "-I", "-c", _TORCH], timeout=TORCH_TIMEOUT)
    if ran is None or ran[0] != 0:
        why = "cannot be started" if ran is None else (_first_line(ran[2]) or f"exited with status {ran[0]}")
        return [Check("torch-import", "error", f"`import torch` failed in {python}: {why}")]
    try:
        info = json.loads(ran[1].strip().splitlines()[-1])
    except (ValueError, IndexError):
        return [Check("torch-import", "warn", "`import torch` ran but its output was not understood")]
    gpu = info["device"] if info["cuda_available"] else "no GPU visible here (expected on a login node)"
    return [Check("torch-import", "ok", f"torch {info['torch']} (CUDA build {info['cuda_build']}); {gpu}")]


# ---------------------------------------------------------------- data and assay ids
_DATA = {
    "proteingym": ("ProteinGym", data_sources.PROTEINGYM.dirname, ("DMS_substitutions.csv", "DMS_ProteinGym_substitutions")),
    "venusmuthub": ("VenusMutHub", data_sources.VENUSMUTHUB.dirname, ("single_mutant",)),
}


def _data_dir(dataset: str, home: Path) -> Path:
    return Path(home) / "data" / _DATA[dataset][1]


def _cut_off(root: Path) -> tuple[str, int, int, str] | None:
    """(kind, files lacking, files named, the first lacking) for a ProteinGym folder without FETCHED.json, or None if it is whole.

    `mpd fetch` unpacks the assay zip, then the structure zip, and writes FETCHED.json last, so a fetch that was cut off
    leaves some of the files that the reference table names out."""
    try:
        with open(root / "DMS_substitutions.csv", newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
    except (OSError, UnicodeDecodeError, csv.Error):
        return None
    folders = [("assay", root / "DMS_ProteinGym_substitutions", "DMS_filename")]
    if (root / "ProteinGym_AF2_structures").is_dir():          # no folder: never unpacked, which only SaProt minds
        folders.append(("structure", root / "ProteinGym_AF2_structures", "pdb_file"))
    for kind, folder, column in folders:
        names = sorted({(r.get(column) or "").strip() for r in rows} - {""})
        names = [n for n in names if "|" not in n]             # several files in one entry: the reader gives it no structure
        gone = [n for n in names if not (folder / n).is_file()]
        if gone:
            return kind, len(gone), len(names), gone[0]
    return None


def check_dataset(dataset: str, ctx, required: bool = True) -> Check:
    title, _, needed = _DATA[dataset]
    root = _data_dir(dataset, Path(ctx.home))
    fix = f"mpd fetch {dataset}{_flags_or_home(ctx.cluster, Path(ctx.home))}"
    missing = Check(f"data:{dataset}", "error" if required else "warn",
                    f"{title} data not found under {root}. Run on a login node: {fix}")
    if not all((root / n).exists() for n in needed):
        return missing
    if (root / "FETCHED.json").is_file():
        return Check(f"data:{dataset}", "ok", f"{root} (complete: FETCHED.json present)")
    if dataset == "venusmuthub":
        have = sum(1 for _ in (root / "single_mutant").glob("*/*.csv"))
        if have == 0:
            return missing
        return Check(f"data:{dataset}", "warn", f"{root} has {have} files but no FETCHED.json, so the fetch may have "
                                                "been interrupted (it resumes)", fix)
    cut = _cut_off(root)
    if cut:
        kind, lacking, named, example = cut
        return Check(f"data:{dataset}", "error",
                     f"{title} data is incomplete under {root}. It lacks {lacking} of {named} {kind} files, such as {example}. "
                     f"The fetch was probably cut off. Run it again on a login node: {fix}", fix)
    return Check(f"data:{dataset}", "ok", f"{root} (no FETCHED.json: not fetched by `mpd fetch`, so not hash-checked)")


def assay_ids(dataset: str, home: Path) -> list[str]:
    """Assay ids of a fetched dataset. VenusMutHub ids are the file names (parsing all 796 CSVs would be slow)."""
    root = _data_dir(dataset, home)
    if dataset == "proteingym":
        from .data.benchmarks import ProteinGym
        return sorted(a.assay_id for a in ProteinGym(root).assays())
    from .data.venusmuthub import EXCLUDED_ASSAYS
    return sorted(p.stem for p in (root / "single_mutant").glob("*/*.csv") if p.stem not in EXCLUDED_ASSAYS)


def check_data(spec, ctx) -> list[Check]:
    if spec is None:
        return [check_dataset(d, ctx, required=False) for d in _DATA]
    wanted = list(dict.fromkeys([spec.train_data, *spec.eval_data]))
    return [check_dataset(d, ctx) for d in wanted if d in _DATA]


def check_assays(spec, ctx) -> list[Check]:
    if spec is None or not spec.assays or spec.train_data not in _DATA:
        return []
    if check_dataset(spec.train_data, ctx).level == "error":
        return [Check("assays", "skip", f"cannot check assay ids until the {spec.train_data} data is fetched")]
    try:
        known = assay_ids(spec.train_data, Path(ctx.home))
    except (OSError, KeyError, ValueError) as exc:
        return [Check("assays", "warn", f"could not read the {spec.train_data} assay list: {exc}")]
    bad = [a for a in spec.assays if a not in set(known)]
    if not bad:
        return [Check("assays", "ok", f"{len(spec.assays)} assay id(s) exist in {spec.train_data}")]
    hints = []
    for a in bad[:3]:
        close = difflib.get_close_matches(a, known, n=3, cutoff=0.5)
        hints.append(repr(a) + (f" (closest: {', '.join(close)})" if close else ""))
    more = f" and {len(bad) - 3} more" if len(bad) > 3 else ""
    return [Check("assays", "error", f"not a {spec.train_data} assay id: {'; '.join(hints)}{more}",
                  f"mpd assays --dataset {spec.train_data} --grep <text>")]


# ---------------------------------------------------------------- slurm tools and cost
def check_slurm_tools(level: str, dry_run: bool = False) -> list[Check]:
    absent = [n for n in ("sbatch", "squeue", "sacct", "scancel") if not shutil.which(n)]
    if not absent:
        return [Check("slurm", "ok", "sbatch, squeue, sacct and scancel are on PATH")]
    if "sbatch" in absent:
        if level == "submit" and not dry_run:
            return [Check("slurm", "error", "sbatch not found on PATH: submit from a Slurm login node "
                                            "(`mpd plan` and `mpd submit --dry-run` work anywhere)")]
        return [Check("slurm", "skip", "sbatch not found on PATH: this is not a Slurm login node "
                                       "(plan and --dry-run work anywhere; submit needs it)")]
    return [Check("slurm", "warn", f"{', '.join(absent)} not found on PATH (`mpd status`, `logs` and `cancel` need them)")]


def check_cost(plan, max_su: float | None, level: str) -> list[Check]:
    if plan is None or not getattr(getattr(getattr(plan, "options", None), "cluster", None), "bills_su", True):
        return []                                             # a profile without SU rates (Vision) has no cost to guard
    total = plan.total_su
    if max_su is None or total <= max_su:
        return [Check("cost", "ok", f"estimated upper bound {total:,.0f} SU (an estimate, not measured)")]
    step = 10 ** max(0, len(str(int(total))) - 2)
    suggestion = int(math.ceil(total / step) * step)
    return [Check("cost", "error" if level == "submit" else "warn",
                  f"estimated upper bound {total:,.0f} SU is above --max-su {max_su:,.0f}. sbatch's own "
                  "\"Requested SUs\" counts CPUs only, so compare this figure with your balance (`myproject`)",
                  f"re-run with --max-su {suggestion} if you mean it")]


# ---------------------------------------------------------------- contributed checks
def _contributed(label: str, module: str, func: str, *args, **kwargs) -> list[Check]:
    """Call `module.func` if that area is present in this build; a crash becomes a warning, never a traceback."""
    try:
        fn = getattr(importlib.import_module(module), func, None)
    except ModuleNotFoundError as exc:
        if exc.name == module or (exc.name and module.startswith(exc.name + ".")):
            return [Check(label, "skip", f"{module} is not available in this build")]
        return [Check(label, "warn", f"{module} could not be imported: {exc}")]
    except Exception as exc:
        return [Check(label, "warn", f"{module} could not be imported: {type(exc).__name__}: {exc}")]
    if fn is None:
        return [Check(label, "skip", f"{module}.{func} is not available in this build")]
    try:
        return list(fn(*args, **kwargs))
    except Exception as exc:                                  # a broken helper must not hide the other findings
        return [Check(label, "warn", f"{module}.{func} failed: {type(exc).__name__}: {exc}")]


def _weights_overview(ctx, verify: bool) -> list[Check]:
    """`mpd doctor` has no recipe, so it reports every model the pipeline can use instead of the two a plan needs."""
    try:
        models = importlib.import_module("mpd.data.models")
    except ModuleNotFoundError:
        return [Check("weights", "skip", "mpd.data.models is not available in this build")]
    home, cname = Path(ctx.home), _cluster_name(ctx.cluster)
    explicit = _explicit_home(ctx.cluster, home)
    ready, absent, problems = [], [], []
    for key in models.fetchable_keys():
        try:
            st = models.status(key, home, verify=verify)
        except Exception as exc:
            problems.append(Check("weights", "warn", f"{key}: {type(exc).__name__}: {exc}"))
            continue
        if st.state in ("ready", "override"):
            ready.append(key)
        elif st.state == "missing":
            absent.append(key)
        else:
            problems.append(Check("weights", "warn", f"{key} is {st.state}: {st.detail}".rstrip(": "),
                                  models.fix_command([key], cname, home if explicit else None)))
    said = f"ready: {', '.join(ready)}. " if ready else "none fetched yet. "
    if absent:
        said += f"not fetched: {', '.join(absent)} (a plan needs only its teacher and student; `mpd models` has sizes)"
    example = "esm2-650m" if "esm2-650m" in absent else (absent[0] if absent else "")
    row = Check("weights", "ok" if ready or not absent else "warn", said.strip(),
                f"mpd fetch model {example}{_flags_or_home(ctx.cluster, home)}" if absent else None)
    return [row, *problems]


def _foldseek_overview(ctx) -> list[Check]:
    try:
        foldseek = importlib.import_module("mpd.structure.foldseek")
    except ModuleNotFoundError:
        return []
    try:
        binary = foldseek.locate(Path(ctx.home))
        if binary is None:
            return [Check("foldseek", "ok", "not installed (only SaProt needs it): mpd fetch foldseek"
                          + _flags_or_home(ctx.cluster, Path(ctx.home)))]
        return [Check("foldseek", "ok", f"{binary} ({foldseek.version(binary)})")]
    except Exception as exc:
        return [Check("foldseek", "warn", f"{type(exc).__name__}: {exc}")]


def run_checks(spec, opts, plan=None, scripts=None, level: str = "plan", deep: bool = False, verify: bool = False,
               max_su: float | None = None, dry_run: bool = False) -> list[Check]:
    """All findings for a recipe (`level` plan | submit) or for the machine alone (`level` doctor, `spec` None).

    `dry_run` (level submit): a machine without sbatch is not an error, because a dry run needs no Slurm client."""
    if level not in ("plan", "submit", "doctor"):
        raise ValueError(f"level must be plan, submit or doctor; got {level!r}")
    home = Path(opts.home)
    cname = _cluster_name(getattr(opts, "cluster", None))
    checks: list[Check] = []
    checks += check_home(opts)
    checks += check_quota(opts, verbose=level == "doctor")
    checks += check_account(opts)
    if os.environ.get("SLURM_JOB_ID"):
        checks.append(Check("host", "warn", "running inside a Slurm job: compute nodes have no internet, so "
                                            "`mpd fetch` and `pip` will fail here; use a login node"))
    checks += check_environment(opts, deep=deep)
    checks += check_data(spec, opts)
    checks += check_assays(spec, opts)
    if spec is not None:
        checks += _contributed("weights", "mpd.data.models", "preflight_checks", spec, home, cname,
                               explicit_home=_explicit_home(getattr(opts, "cluster", None), home))
        checks += _contributed("structure", "mpd.structure", "preflight_checks", spec, home, cname,
                               explicit_home=_explicit_home(getattr(opts, "cluster", None), home))
    else:
        checks += _weights_overview(opts, verify)
        checks += _foldseek_overview(opts)
    if level != "plan":
        checks += check_slurm_tools(level, dry_run)
    if plan is not None and level == "submit":
        checks += _contributed("sbatch-test", "mpd.slurm.submit", "preflight_checks", plan, scripts)
    checks += check_cost(plan, max_su, level)
    return checks


# ---------------------------------------------------------------- output
_FOOTER_TAIL = {
    "plan": "Plan and --dry-run only report these (exit 0); `mpd submit` stops at the first error.",
    "dry-run": "A dry run only reports these (exit 0); `mpd submit` stops at the first error.",
    "submit": "Nothing was submitted.",
    "doctor": "Fix the [FAIL] lines (the command to run is under each), then run `mpd doctor` again.",
}


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def render(checks: list[Check], *, mode: str = "plan") -> str:
    """The aligned table with fix commands under the findings, and a READY / NOT READY footer."""
    width = min(18, max((len(c.name) for c in checks), default=4))
    pad = " " * (7 + width + 2)
    lines: list[str] = []
    for c in checks:
        first, *rest = (c.message or "").split("\n")
        if len(c.name) > width:                               # a long name from a contributing area keeps the columns aligned
            lines += [f"{TAGS[c.level]:<7}{c.name}", pad + first]
        else:
            lines.append(f"{TAGS[c.level]:<7}{c.name:<{width}}  {first}")
        lines += [pad + ln for ln in rest]
        if c.fix and c.level in ("warn", "error") and c.fix not in c.message:
            lines.append(f"{pad}fix: {c.fix}")
    n = counts(checks)
    if n["error"]:
        lines += ["", f"NOT READY: {_plural(n['error'], 'error')}, {_plural(n['warn'], 'warning')}. " + _FOOTER_TAIL.get(mode, "")]
    else:
        warned = f" ({_plural(n['warn'], 'warning')})" if n["warn"] else ""
        lines += ["", f"READY{warned}: no blocking problem found. These are static checks; they do not prove that a "
                      "job will run on the cluster (docs/GRACE.md lists what has not been verified)."]
    return "\n".join(lines)


# ---------------------------------------------------------------- paste-safe report
_TOKENS = re.compile(r"\b(?:hf_[A-Za-z0-9]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|git" r"hub_pat_[A-Za-z0-9_]{20,})\b")   # Hugging Face, GitHub
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_LONG_NUMBER = re.compile(r"(?<![\w.])\d{10,14}(?!\d)")
_USER_DIR = re.compile(r"(/(?:scratch/(?:user|group|project)|home|work|users?)/)[^/\s:'\"]+")


def scrub(text: str, *, accounts=(), users=(), paths=()) -> str:
    """Remove what must not be pasted into a public issue: account numbers, user names, home paths, tokens, e-mail.

    `paths` is a list of (path, replacement). The environment's user name, $HOME and $SCRATCH are always removed.
    """
    text = _EMAIL.sub("<email>", _TOKENS.sub("<token>", text))
    replacements = [(str(a), "<account>") for a in accounts if a] + [(str(p), label) for p, label in paths if p]
    for var, label in (("SCRATCH", "$SCRATCH"), ("HOME", "~")):
        value = os.environ.get(var, "")
        if len(value) > 3:                                    # never replace "/" or "/x"
            replacements.append((value, label))
    for old, new in sorted(replacements, key=lambda r: -len(r[0])):
        text = text.replace(old, new)
    text = _USER_DIR.sub(lambda m: m.group(1) + "<name>", text)
    names = {*users, os.environ.get("USER", ""), os.environ.get("LOGNAME", ""), os.environ.get("SUDO_USER", "")}
    try:
        names.add(getpass.getuser())
    except Exception:                                         # no passwd entry in some containers
        pass
    for name in sorted((n for n in names if n and len(n) > 2), key=len, reverse=True):
        text = re.sub(rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])", "<netid>", text)
    return _LONG_NUMBER.sub("<account>", text)


def report(checks: list[Check], opts, *, mode: str = "doctor") -> str:
    """`mpd doctor --report`: versions, the check table and the footer, scrubbed so it can go into an issue."""
    cluster = getattr(opts, "cluster", None)
    head = [f"mpd {__version__} doctor report",
            f"python {platform.python_version()} on {platform.system()} {platform.release()}",
            f"cluster {getattr(cluster, 'name', None) or 'none given'}"]
    safe = [replace(c, message=c.message.split("\n")[0]) if c.name == "quota" else c for c in checks]      # the raw showquota lines stay on your screen
    text = "\n".join(head) + "\n\n" + render(safe, mode=mode)
    account = getattr(opts, "account", None) or os.environ.get("MPD_ACCOUNT")
    return scrub(text, accounts=[account], paths=[(Path(opts.home), "<mpd-home>")])
