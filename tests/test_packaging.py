"""Packaging, pinned dependencies, community files and repository hygiene.

Needs only pytest (which brings packaging) and, for the YAML checks, PyYAML. `test_docs_*` also run in the CI docs job, which
installs nothing else.
The wheel and the sdist are built with the installed setuptools into a temporary copy of the tree: no network, nothing written
into the repository. A wheel is then built from that sdist, and the login-node commands run from the unpacked wheel alone, so
a file missing from the sdist or from the wheel fails here and not on a stranger's machine.
"""

from __future__ import annotations

import ast
import importlib
import importlib.metadata
import os
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.parse
import zipfile
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

REPO = Path(__file__).resolve().parents[1]
GITHUB = REPO / ".github"
WORKFLOW = GITHUB / "workflows" / "ci.yml"
needs_github = pytest.mark.skipif(not GITHUB.is_dir(), reason=".github is not part of the sdist")
# for a tree without git, as in .gitignore: caches anywhere, mpd's working folders and build outputs at the top level only
SKIP_ANYWHERE = {".git", ".ruff_cache", ".pytest_cache", ".mypy_cache", "__pycache__", ".venv", "venv"}
SKIP_TOP = {"build", "dist", "env", "runs", "data", "models", "tools", "cache", "hf_cache", "mpd_home"}


def toml(path: Path) -> dict:
    import tomllib
    return tomllib.loads(path.read_text(encoding="utf-8"))


def walk_tree() -> list[str]:
    names = []
    for folder, dirs, files in os.walk(REPO):
        here = Path(folder).relative_to(REPO)
        dirs[:] = [d for d in dirs if d not in SKIP_ANYWHERE and not d.endswith(".egg-info") and not (str(here) == "." and d in SKIP_TOP)]
        names += [(here / f).as_posix() for f in files]
    return names


@lru_cache(maxsize=None)
def repo_files() -> tuple[Path, ...]:
    """The files that belong to the project: tracked or not ignored (git), else a walk that skips caches and mpd's data."""
    try:
        top = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=REPO, capture_output=True, text=True, check=True)
        if Path(top.stdout.strip()).resolve() != REPO:
            raise OSError("not the top of a git work tree")
        out = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=REPO,
                             capture_output=True, check=True)
        names = [n for n in out.stdout.decode("utf-8").split("\0") if n]
    except (OSError, subprocess.CalledProcessError):
        names = walk_tree()
    return tuple(sorted(p for p in (REPO / n for n in names) if p.is_file()))


def text_of(path: Path) -> str | None:
    if path.stat().st_size > 2_000_000:
        return None
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None


def rel(path: Path) -> str:
    return path.relative_to(REPO).as_posix()


def workflow() -> dict:
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(WORKFLOW.read_text())


# -- version and Python floor ----------------------------------------------------------------------------------------

def test_version_is_the_same_everywhere():
    yaml = pytest.importorskip("yaml")
    declared = toml(REPO / "pyproject.toml")["project"]["version"]
    in_code = re.search(r'^__version__ = "([^"]+)"', (REPO / "src" / "mpd" / "__init__.py").read_text(), re.M).group(1)
    assert declared == in_code
    assert str(yaml.safe_load((REPO / "CITATION.cff").read_text())["version"]) == declared
    assert "## [Unreleased]" in (REPO / "CHANGELOG.md").read_text()


def test_python_floor_is_the_same_everywhere():
    config = toml(REPO / "pyproject.toml")
    spec = config["project"]["requires-python"]
    floor = re.fullmatch(r">=\s*(3)\.(\d+)", spec)
    assert floor, f"requires-python {spec!r}: write it as >=3.MINOR"
    minor = int(floor.group(2))
    assert config["tool"]["ruff"]["target-version"] == f"py3{minor}"
    classified = sorted(int(c.rsplit(".", 1)[1]) for c in config["project"]["classifiers"]
                        if re.fullmatch(r"Programming Language :: Python :: 3\.\d+", c))
    assert classified and classified[0] == minor, "the oldest Python classifier is the floor"
    assert re.search(rf"^MIN_PYTHON = \(3, {minor}\)$", (REPO / "src" / "mpd" / "preflight.py").read_text(), re.M)
    assert f"3.{minor} or newer" in (REPO / "scripts" / "setup_env.sh").read_text()
    if WORKFLOW.is_file():
        ci = {int(m) for m in re.findall(r'python-version:\s*"3\.(\d+)"', WORKFLOW.read_text())}
        assert ci and min(ci) == minor, "CI runs the floor"
    for name in ("CONTRIBUTING.md", "CHANGELOG.md"):
        older = [m for m in re.findall(r"Python 3\.(\d+)", (REPO / name).read_text()) if int(m) < minor]
        assert not older, f"{name} names Python 3.{older[0] if older else ''}, below the floor 3.{minor}"
        assert f"3.{minor}" in (REPO / name).read_text(), f"{name} should name the floor (Python 3.{minor})"


# -- package data, packages, entry point ----------------------------------------------------------------------------

def test_package_data_patterns_match_files_and_cover_every_data_file():
    patterns = toml(REPO / "pyproject.toml")["tool"]["setuptools"]["package-data"]["mpd"]
    pkg = REPO / "src" / "mpd"
    covered = {p for pattern in patterns for p in pkg.glob(pattern)}
    for pattern in patterns:
        assert list(pkg.glob(pattern)), f"package-data pattern {pattern!r} matches no file"
    data = [p for p in repo_files() if p.is_relative_to(pkg) and p.suffix not in {".py", ".pyc"}]
    missing = sorted(rel(p) for p in data if p not in covered)
    assert not missing, f"data files that would be left out of the wheel (add them to [tool.setuptools.package-data]): {missing}"


def test_every_source_directory_with_python_files_is_a_package():
    folders = {p.parent for p in repo_files() if p.suffix == ".py" and p.is_relative_to(REPO / "src")}
    bare = sorted(rel(d) for d in folders if not (d / "__init__.py").is_file())
    assert not bare, f"setuptools would leave these out of the wheel (no __init__.py): {bare}"


def test_console_script_resolves_to_a_callable():
    scripts = toml(REPO / "pyproject.toml")["project"]["scripts"]
    assert set(scripts) == {"mpd"}
    module, _, attr = scripts["mpd"].partition(":")
    assert callable(getattr(importlib.import_module(module), attr))


def test_the_files_the_code_reads_at_run_time_are_package_data():
    """`importlib.resources` reads name a package; the files they can reach must be in the wheel. A new reader needs a new line here."""
    src = REPO / "src"
    readers: dict[str, set[str]] = {}
    for path in repo_files():
        if path.suffix != ".py" or not path.is_relative_to(src):
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "files"
                    and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
                readers.setdefault(node.args[0].value, set()).add(path.relative_to(src / "mpd").as_posix())
    assert readers == {"mpd.clusters": {"slurm/cluster.py"}, "mpd.data": {"data/fetch.py", "data/models.py"}}, readers
    patterns = toml(REPO / "pyproject.toml")["tool"]["setuptools"]["package-data"]["mpd"]
    for package, suffixes in (("clusters", {".yaml"}), ("data", {".tsv"})):
        shipped = {p.name for pattern in patterns for p in (src / "mpd").glob(pattern) if p.parent.name == package}
        assert shipped and {Path(n).suffix for n in shipped} == suffixes, (package, shipped)
    # `__file__` is used to find the clone for constraints/grace.txt; that file is not package data on purpose
    users = sorted(rel(p) for p in repo_files() if p.suffix == ".py" and p.is_relative_to(src) and "__file__" in p.read_text(encoding="utf-8"))
    assert users == ["src/mpd/preflight.py"], users


# -- wheel and sdist ------------------------------------------------------------------------------------------------

def build_hook(hook: str, folder: Path, cwd: Path) -> None:
    folder.mkdir()
    done = subprocess.run([sys.executable, "-c", f"import sys, setuptools.build_meta as b; b.{hook}(sys.argv[1])", str(folder)],
                          cwd=cwd, capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert done.returncode == 0, done.stderr[-3000:]


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    for text in toml(REPO / "pyproject.toml")["build-system"]["requires"]:
        req = Requirement(text)
        try:
            have = importlib.metadata.version(req.name)
        except importlib.metadata.PackageNotFoundError:
            pytest.skip(f"build requirement {text} is not installed (the build runs without build isolation)")
        if not req.specifier.contains(have, prereleases=True):
            pytest.skip(f"build requirement {text} is not met by the installed {req.name} {have}")
    work = tmp_path_factory.mktemp("build")
    tree = work / "tree"
    for path in repo_files():
        target = tree / path.relative_to(REPO)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for hook, folder in (("build_sdist", work / "sdist"), ("build_wheel", work / "wheel")):    # one process per hook, as PEP 517 frontends do
        build_hook(hook, folder, tree)
    sdist, wheel = next((work / "sdist").glob("*.tar.gz")), next((work / "wheel").glob("*.whl"))
    with tarfile.open(sdist) as tf:                     # what a stranger gets from `pip install proteinmutpd-*.tar.gz`
        tf.extractall(work / "unpacked", **({"filter": "data"} if hasattr(tarfile, "data_filter") else {}))
    build_hook("build_wheel", work / "wheel_from_sdist", next((work / "unpacked").iterdir()))
    zipfile.ZipFile(wheel).extractall(work / "installed")
    return SimpleNamespace(wheel=wheel, sdist=sdist, sdist_wheel=next((work / "wheel_from_sdist").glob("*.whl")), installed=work / "installed")


def test_wheel_holds_exactly_the_modules_the_package_data_and_the_metadata(built):
    zf = zipfile.ZipFile(built.wheel)
    names = set(zf.namelist())
    info = next(n.split("/")[0] for n in names if n.endswith(".dist-info/METADATA"))
    src = REPO / "src"
    modules = {p.relative_to(src).as_posix() for p in repo_files() if p.suffix == ".py" and p.is_relative_to(src)}
    patterns = toml(REPO / "pyproject.toml")["tool"]["setuptools"]["package-data"]["mpd"]
    data = {p.relative_to(src).as_posix() for pattern in patterns for p in (src / "mpd").glob(pattern)}
    assert {n for n in names if not n.startswith(f"{info}/")} == modules | data, "the wheel holds more or less than modules + package data"
    for need in ("mpd/clusters/grace.yaml", "mpd/clusters/vision.yaml", "mpd/data/models.tsv", "mpd/data/venusmuthub_single_mutant.tsv"):
        assert need in names
    meta = zf.read(f"{info}/METADATA").decode()
    assert "Name: proteinmutpd" in meta and "Requires-Python: >=3.11" in meta and "License-Expression: MIT" in meta
    assert re.search(r"^Requires-Dist: numpy>=1\.26$", meta, re.M) and "Requires-Dist: torch>=2.7; extra == " in meta
    assert "Provides-Extra: train" in meta and "Provides-Extra: structure" in meta
    assert {"METADATA", "RECORD", "WHEEL", "entry_points.txt", "licenses/LICENSE"} <= {n[len(info) + 1:] for n in names if n.startswith(f"{info}/")}
    assert "mpd = mpd.cli:main" in zf.read(f"{info}/entry_points.txt").decode()


SDIST_TOP = {"LICENSE", "README.md", "CHANGELOG.md", "CITATION.cff", "CONTRIBUTING.md", "SECURITY.md", "MANIFEST.in", "pyproject.toml",
             "PKG-INFO", "setup.cfg", "src", "tests", "docs", "scripts", "constraints"}          # PKG-INFO and setup.cfg are written by setuptools


def test_sdist_carries_constraints_docs_scripts_tests_the_goldens_and_the_community_files(built):
    with tarfile.open(built.sdist) as tf:
        names = {n.split("/", 1)[1] for n in tf.getnames() if "/" in n}
    for need in ("LICENSE", "README.md", "CHANGELOG.md", "CITATION.cff", "CONTRIBUTING.md", "SECURITY.md", "MANIFEST.in", "pyproject.toml"):
        assert need in names
    wanted = {rel(p) for p in repo_files() if rel(p).split("/")[0] in {"constraints", "docs", "scripts", "tests"}}
    assert wanted <= names, f"missing from the sdist: {sorted(wanted - names)}"
    goldens = {rel(p) for p in repo_files() if rel(p).startswith("tests/golden_render/")}
    assert goldens and goldens <= names, "the golden sbatch files are test inputs and ship with the tests"
    assert {n.split("/")[0] for n in names} <= SDIST_TOP, sorted({n.split("/")[0] for n in names} - SDIST_TOP)
    assert not [n for n in names if "__pycache__" in n or n.endswith(".pyc")]


def test_a_wheel_built_from_the_sdist_equals_the_wheel_built_from_the_tree(built):
    """If MANIFEST.in forgets a file the build needs, the second wheel differs or the fixture fails to build it."""
    tree, from_sdist = zipfile.ZipFile(built.wheel), zipfile.ZipFile(built.sdist_wheel)
    assert sorted(tree.namelist()) == sorted(from_sdist.namelist())
    assert [n for n in tree.namelist() if tree.read(n) != from_sdist.read(n)] == []


RUN_FROM = """
import runpy, sys
folder = sys.argv[1]
sys.path.insert(0, folder)
import mpd
assert mpd.__file__.startswith(folder), mpd.__file__    # the wheel's copy, not the source tree or an editable install
sys.argv = ["mpd", *sys.argv[2:]]
runpy.run_module("mpd", run_name="__main__")
"""


def test_the_unpacked_wheel_alone_runs_the_login_node_commands(built, tmp_path):
    """No source tree, no torch needed, nothing on PATH (no sbatch, showquota, myproject): what a login node does with the wheel."""
    for module in ("numpy", "yaml"):
        pytest.importorskip(module)
    (tmp_path / "nobin").mkdir()
    (tmp_path / "empty").mkdir()
    home = tmp_path / "home"
    env = {"PATH": str(tmp_path / "nobin"), "HOME": str(tmp_path), "LANG": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1",
           **{k: os.environ[k] for k in ("LD_LIBRARY_PATH",) if k in os.environ}}

    def mpd(*args):
        done = subprocess.run([sys.executable, "-I", "-c", RUN_FROM, str(built.installed), *args], cwd=tmp_path, env=env,
                              capture_output=True, text=True)
        assert "Traceback" not in done.stdout + done.stderr, done.stderr[-2000:]
        return done

    version = re.search(r'^__version__ = "([^"]+)"', (REPO / "src" / "mpd" / "__init__.py").read_text(), re.M).group(1)
    assert mpd("--version").stdout.strip() == f"mpd {version}"
    profiles = mpd("clusters").stdout
    assert re.search(r"^grace\b", profiles, re.M) and re.search(r"^vision\b", profiles, re.M)     # both YAML files were read
    assert "esm2-650m" in mpd("models").stdout
    flags = ["--cluster", "vision", "--home", str(home), "--teacher", "esm2-650m", "--student", "esm2-150m",
             "--train-data", "proteingym", "--method", "score-all", "--eval-data", "proteingym"]
    plan = mpd("plan", *flags)
    assert plan.returncode == 0 and plan.stdout.startswith("campaign")
    assert mpd("submit", "--dry-run", *flags).returncode == 0
    scripts = sorted(home.glob("runs/*/scripts/*.sbatch"))
    assert scripts and not list(home.glob("runs/*/SUBMITTED.json"))
    for script in scripts:
        assert subprocess.run(["bash", "-n", str(script)]).returncode == 0, script.name
    doctor = mpd("doctor", "--cluster", "vision", "--home", str(home))
    assert doctor.returncode == 1 and "NOT READY" in doctor.stdout          # an empty home has no environment: reported, not a crash
    entries = len((REPO / "src" / "mpd" / "data" / "venusmuthub_single_mutant.tsv").read_text().splitlines()) - 1
    fetch = mpd("fetch", "venusmuthub", "--home", str(home), "--from-dir", str(tmp_path / "empty"))
    assert fetch.returncode == 2 and f"0/{entries} files already present" in fetch.stdout + fetch.stderr    # the wheel's TSV lists every file


# -- pinned dependencies --------------------------------------------------------------------------------------------

PYTHONS = ("3.11", "3.12", "3.13")


def requirement_lines(path: Path):
    for number, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if line:
            yield number, Requirement(line)


def active_pins(python: str) -> dict[str, str]:
    pins: dict[str, str] = {}
    env = {"python_version": python, "python_full_version": f"{python}.0"}
    for path in sorted((REPO / "constraints").glob("*.txt")):
        for number, req in requirement_lines(path):
            if req.marker is not None and not req.marker.evaluate(env):
                continue
            name = canonicalize_name(req.name)
            assert name not in pins, f"{path.name}:{number}: {name} is pinned twice for Python {python}"
            specs = list(req.specifier)
            assert len(specs) == 1 and specs[0].operator == "==", f"{path.name}:{number}: pin {req.name} with a single =="
            pins[name] = specs[0].version
    return pins


def test_constraints_files_exist_and_parse():
    files = sorted(p.name for p in (REPO / "constraints").glob("*.txt"))
    assert files == ["ci.txt", "grace.txt"]
    for python in PYTHONS:
        assert active_pins(python)


def test_constraints_agree_with_the_ranges_pyproject_declares():
    project = toml(REPO / "pyproject.toml")["project"]
    declared = [Requirement(t) for t in project["dependencies"] + [t for ts in project["optional-dependencies"].values() for t in ts]]
    for python in PYTHONS:
        pins = active_pins(python)
        env = {"python_version": python, "python_full_version": f"{python}.0"}
        for req in declared:
            if req.marker is not None and not req.marker.evaluate({**env, "extra": ""}):
                continue
            name = canonicalize_name(req.name)
            if name in pins:
                assert req.specifier.contains(pins[name]), f"Python {python}: pin {name}=={pins[name]} breaks {req}"
        for text in project["dependencies"] + project["optional-dependencies"]["train"]:
            assert canonicalize_name(Requirement(text).name) in pins, f"Python {python}: {text} has no pin in constraints/"


def test_the_test_extra_asks_for_the_build_requirements_the_wheel_tests_need():
    """The `built` fixture skips when the installed setuptools is older than build-system.requires; a stock venv has an older one."""
    config = toml(REPO / "pyproject.toml")
    wanted = {canonicalize_name(Requirement(t).name): Requirement(t).specifier for t in config["build-system"]["requires"]}
    for extra in ("test", "dev"):
        have = {canonicalize_name(Requirement(t).name): Requirement(t).specifier for t in config["project"]["optional-dependencies"][extra]}
        assert {name: have.get(name) for name in wanted} == wanted, f"the {extra} extra must repeat build-system.requires"


@needs_github
def test_ruff_has_one_version_in_the_dev_extra_and_the_workflow():
    dev = [t for t in toml(REPO / "pyproject.toml")["project"]["optional-dependencies"]["dev"] if t.startswith("ruff")]
    flow = re.findall(r"ruff==[0-9.]+", WORKFLOW.read_text())
    assert dev and flow and set(flow) == set(dev)


# -- repository hygiene ---------------------------------------------------------------------------------------------

FORBIDDEN = (    # character classes keep this file from matching itself; hits are reported by place, never by value
    ("maintainer NetID", re.compile(r"y[i]hong", re.I)),
    ("TAMU address", re.compile(r"(?<!hprc\.)t[a]mu\.edu", re.I)),
    ("project or lab name", re.compile(r"prj-0[2]|mol-ce[l]l|Protein[O]PD|[v]sts", re.I)),
    ("project path", re.compile(r"/scratch/pro[j]ect")),
    ("access token", re.compile(r"gh[p]_|github_p[a]t_")),
)
ACCOUNT_LIKE = re.compile(r"(?<![0-9A-Za-z_])[0-9]{12,13}(?![0-9A-Za-z_])")
EXEMPT = {("LICENSE", "maintainer NetID")}            # the copyright holder named there is a public name; the owner may change it


def is_placeholder_number(digits: str) -> bool:
    """The documented fake account, a run of one digit, or counting up (the fakes tests use for invalid accounts)."""
    return digits == "123456789012" or len(set(digits)) == 1 or ("1234567890" * 2).startswith(digits)


def test_repo_contains_no_secrets_or_identity():
    problems = []
    for path in repo_files():
        name = rel(path)
        body = text_of(path)
        for label, pattern in FORBIDDEN:
            if (name, label) in EXEMPT:
                continue
            if pattern.search(name):
                problems.append(f"{name}: {label} in the file name")
            for number, line in enumerate((body or "").splitlines(), 1):
                if pattern.search(line):
                    problems.append(f"{name}:{number}: {label}")
        for number, line in enumerate((body or "").splitlines(), 1):
            if any(not is_placeholder_number(m.group(0)) for m in ACCOUNT_LIKE.finditer(line)):
                problems.append(f"{name}:{number}: a 12 or 13 digit number that looks like an account")
    assert not problems, "\n".join(problems)


def test_no_file_over_one_megabyte():
    allowed = {"src/mpd/data/venusmuthub_single_mutant.tsv"}
    big = [f"{rel(p)} ({p.stat().st_size // 1024} kB)" for p in repo_files() if p.stat().st_size > 1_000_000 and rel(p) not in allowed]
    assert not big, f"data and weights do not belong in the repository: {big}"


def ignored_by_git(path: str) -> bool:
    done = subprocess.run(["git", "check-ignore", "-q", "--no-index", path], cwd=REPO, capture_output=True)
    if done.returncode not in (0, 1):
        pytest.skip("not a git work tree")
    return done.returncode == 0


def test_gitignore_hides_what_mpd_writes_and_keeps_the_source_visible():
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    hidden = ["env/bin/python", "env/mpd_env.json", "hf_cache/hub/x", "models/esm2-3b/pytorch_model.bin",
              "tools/foldseek-10-941cd33/foldseek/bin/foldseek", "cache/pip/x", "cache/3di/x.json", "runs/campaign/SUBMITTED.json",
              "runs/campaign/spec.json", "data/proteingym-v1.3/x.csv", "mpd_home/x", "download.part", "train/unit.incomplete-1700000000",
              "src/mpd/__pycache__/x.pyc", ".ruff_cache/x", "dist/x.whl", "build/lib/x.py", "src/proteinmutpd.egg-info/PKG-INFO",
              ".env", "slurm-123.out"]
    shown = ["src/mpd/data/models.py", "src/mpd/data/models.tsv", "src/mpd/data/venusmuthub_single_mutant.tsv", "src/mpd/data/fetch.py",
             "src/mpd/clusters/grace.yaml", "src/mpd/structure/build.py", "src/mpd/structure/tools.py", "src/mpd/slurm/campaign.py",
             "src/mpd/envrecord.py", "tests/fx_models.py", "tests/golden_render/grace_train_ddp.sbatch", "docs/MODELS.md",
             "constraints/grace.txt", "scripts/setup_env.sh", ".github/workflows/ci.yml", "CITATION.cff"]
    assert [p for p in hidden if not ignored_by_git(p)] == []
    assert [p for p in shown if ignored_by_git(p)] == []


# -- workflow, markers and community files --------------------------------------------------------------------------

@needs_github
def test_workflow_follows_the_ci_policy():
    text = WORKFLOW.read_text()
    flow = workflow()
    assert flow["permissions"] == {"contents": "read"} and flow["concurrency"]
    triggers = flow.get("on", flow.get(True))                        # PyYAML reads the key `on` as True
    assert not {"pull_request_target", "workflow_run", "schedule"} & set(triggers)
    assert "secrets." not in text
    for ref in re.findall(r"^\s*-?\s*uses:\s*(\S+)", text, re.M):
        assert re.fullmatch(r"actions/[a-z-]+@(v[0-9]+(\.[0-9]+){0,2}|[0-9a-f]{40})", ref), f"{ref}: use a major tag of an actions/* action"
    for name, job in flow["jobs"].items():
        assert job["runs-on"] != "ubuntu-latest" and "timeout-minutes" in job, name


@needs_github
def test_the_test_job_runs_the_default_suite_and_the_rehearsal_apart():
    steps = [s["run"] for s in workflow()["jobs"]["test"]["steps"] if "pytest" in s.get("run", "")]
    assert len(steps) == 2, steps
    default, rehearsal = steps
    assert '-m "not slow and not gpu and not rehearsal"' in default and "-eq 5" not in default
    assert re.search(r"-m rehearsal\b", rehearsal) and "-eq 5" in rehearsal      # exit code 5 (nothing selected) is accepted in this step only


@needs_github
def test_the_shell_of_every_workflow_step_parses_and_passes_shellcheck(tmp_path):
    """`bash -n` always; shellcheck (warnings and errors) when it is installed, as it is on the GitHub runners."""
    steps = [(job, step) for job, spec in workflow()["jobs"].items() for step in spec["steps"] if "run" in step]
    assert len(steps) >= 10
    problems = []
    for number, (job, step) in enumerate(steps):
        script = tmp_path / f"step{number}.sh"
        script.write_text("#!/bin/bash\n" + re.sub(r"\$\{\{.*?\}\}", "EXPRESSION", step["run"]) + "\n")      # a GitHub expression is not shell
        done = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
        if done.returncode:
            problems.append(f"{job}: {step.get('name', number)}: {done.stderr.strip()}")
        if shutil.which("shellcheck"):
            done = subprocess.run(["shellcheck", "-s", "bash", "-S", "warning", str(script)], capture_output=True, text=True)
            if done.returncode:
                problems.append(f"{job}: {step.get('name', number)}: {done.stdout.replace(str(script), 'step')}")
    assert not problems, "\n".join(problems)


BUILTIN_MARKS = {"skip", "skipif", "xfail", "parametrize", "usefixtures", "filterwarnings"}


def marks_used(source: str) -> set[str]:
    """The names after `pytest.mark.` in a module."""
    return {node.attr for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute)
            and node.value.attr == "mark" and isinstance(node.value.value, ast.Name) and node.value.value.id == "pytest"}


def test_every_marker_in_use_or_selected_by_a_command_is_registered():
    registered = {m.split(":", 1)[0].strip() for m in toml(REPO / "pyproject.toml")["tool"]["pytest"]["ini_options"]["markers"]}
    used = set()
    for path in repo_files():
        if path.suffix == ".py" and path.parent == REPO / "tests":
            used |= marks_used(path.read_text(encoding="utf-8"))
    assert used - BUILTIN_MARKS <= registered, f"unregistered markers (a typo selects nothing): {sorted(used - BUILTIN_MARKS - registered)}"
    texts = [(REPO / "CONTRIBUTING.md").read_text(), (GITHUB / "pull_request_template.md").read_text() if GITHUB.is_dir() else "",
             WORKFLOW.read_text() if WORKFLOW.is_file() else ""]
    for text in texts:
        for quoted, bare in re.findall(r'pytest\b[^\n]*?\s-m (?:"([^"]+)"|([A-Za-z_]+))', text):
            names = set(re.findall(r"[A-Za-z_]+", quoted or bare)) - {"not", "and", "or"}
            assert names <= registered, f"a command selects the unregistered marker(s) {sorted(names - registered)}"


@needs_github
def test_contributing_and_the_pull_request_template_show_both_pytest_commands():
    for path in (REPO / "CONTRIBUTING.md", GITHUB / "pull_request_template.md"):
        text = path.read_text()
        assert '-m "not slow and not gpu and not rehearsal"' in text and "-m rehearsal" in text, rel(path)


@needs_github
def test_issue_forms_are_well_formed():
    yaml = pytest.importorskip("yaml")
    forms = GITHUB / "ISSUE_TEMPLATE"
    config = yaml.safe_load((forms / "config.yml").read_text())
    assert config["blank_issues_enabled"] is False and config["contact_links"]
    for form in ("bug_report.yml", "feature_request.yml"):
        doc = yaml.safe_load((forms / form).read_text())
        assert doc["name"] and doc["description"] and doc["body"]
        assert all(item["type"] in {"markdown", "input", "textarea", "dropdown", "checkboxes"} for item in doc["body"])
        ids = [item["id"] for item in doc["body"] if "id" in item]
        assert len(ids) == len(set(ids))
    bug = yaml.safe_load((forms / "bug_report.yml").read_text())
    assert {"doctor", "command", "campaign", "redacted"} <= {item.get("id") for item in bug["body"]}
    assert "doctor --report" in (forms / "bug_report.yml").read_text()


def test_citation_metadata_is_well_formed():
    yaml = pytest.importorskip("yaml")
    cff = yaml.safe_load((REPO / "CITATION.cff").read_text())
    assert cff["cff-version"] == "1.2.0" and cff["license"] == "MIT" and cff["authors"] and cff["title"] == "ProteinMutPD"


# -- Markdown links -------------------------------------------------------------------------------------------------

FENCE = re.compile(r"^(```|~~~).*?^\1[^\n]*$", re.M | re.S)
INLINE_CODE = re.compile(r"`[^`\n]*`")
LINK = re.compile(r"!?\[[^\]\n]*\]\(\s*<?([^)\s>]+)>?(?:\s+(?:\"[^\"]*\"|'[^']*'))?\s*\)")
REFERENCE = re.compile(r"^\s{0,3}\[[^\]\n]+\]:\s*<?(\S+?)>?(?:\s+.*)?$", re.M)
HEADING = re.compile(r"^ {0,3}(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$", re.M)
HTML_ANCHOR = re.compile(r"""<a\s[^>]*?(?:name|id)=["']([^"']+)["']""", re.I)


def prose(text: str) -> str:
    return INLINE_CODE.sub("", FENCE.sub("", text))


def github_anchors(text: str) -> set[str]:
    """Heading ids as GitHub renders them: lower case, punctuation dropped, spaces to hyphens, -1/-2 for repeats."""
    seen: dict[str, int] = {}
    anchors = set(HTML_ANCHOR.findall(prose(text)))
    for _, title in HEADING.findall(FENCE.sub("", text)):
        title = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", title)
        title = re.sub(r"[`*~]", "", title).strip().lower()
        slug = re.sub(r"[^\w\- ]", "", title).replace(" ", "-")
        count = seen.get(slug, 0)
        seen[slug] = count + 1
        anchors.add(slug if count == 0 else f"{slug}-{count}")
    return anchors


def broken_links(path: Path) -> list[str]:
    body = path.read_text(encoding="utf-8")
    problems = []
    for target in LINK.findall(prose(body)) + REFERENCE.findall(prose(body)):
        if re.match(r"[a-zA-Z][a-zA-Z0-9+.-]*:", target):             # http:, https:, mailto: ...
            continue
        location, _, fragment = urllib.parse.unquote(target).partition("#")
        location = location.partition("?")[0]
        base = REPO if location.startswith("/") else path.parent
        dest = (base / location.lstrip("/")).resolve() if location else path
        if not dest.exists():
            problems.append(f"{rel(path) if path.is_relative_to(REPO) else path.name}: {target} does not exist")
        elif fragment and dest.suffix == ".md" and fragment not in github_anchors(dest.read_text(encoding="utf-8")):
            problems.append(f"{rel(path) if path.is_relative_to(REPO) else path.name}: {target} has no such heading")
    return list(dict.fromkeys(problems))


def test_docs_relative_links_resolve():
    files = [p for p in repo_files() if p.suffix == ".md" and p.parent in {REPO, REPO / "docs", GITHUB}]
    assert REPO / "README.md" in files and REPO / "CONTRIBUTING.md" in files
    problems = [problem for path in files for problem in broken_links(path)]
    assert not problems, "\n".join(problems)


def test_docs_link_checker_sees_files_anchors_and_ignores_code(tmp_path):
    (tmp_path / "other.md").write_text("# Other\n\n## Two words, and `code`\n\n## Two words, and `code`\n\n<a name=\"hand\"></a>\n")
    (tmp_path / "page.md").write_text(
        "# Page\n[ok](other.md) [anchor](other.md#two-words-and-code) [repeat](other.md#two-words-and-code-1) [html](other.md#hand)\n"
        "[same](#page) [web](https://example.invalid/x) [mail](mailto:a@example.invalid) ![img](missing.png)\n"
        "[no file](nowhere.md) [no heading](other.md#nothing) [no local heading](#nothing)\n"
        "`[in code](gone.md)`\n```\n[in fence](gone.md)\n```\n[ref]: missing-ref.md\n")
    found = broken_links(tmp_path / "page.md")
    assert sorted(p.split(": ", 1)[1] for p in found) == sorted([
        "missing.png does not exist", "nowhere.md does not exist", "other.md#nothing has no such heading",
        "#nothing has no such heading", "missing-ref.md does not exist"])
