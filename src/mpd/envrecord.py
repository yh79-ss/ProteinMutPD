"""The environment record: which Python and modules the setup script built, read back by the planner.

    python -m mpd.envrecord write --home H --cluster C --modules "A B" --constraints FILE

writes ``<home>/env/mpd_env.json`` (schema 1) from inside the new environment. `make_plan` reads it, so jobs load the modules the
environment was built with and call its interpreter without anyone editing a cluster profile. Standard library only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

SCHEMA = 1
RECORD_NAME = "mpd_env.json"
PACKAGES = ("torch", "transformers", "peft", "numpy", "scipy", "pyyaml")
MAX_BYTES = 1 << 20

# same rule as mpd.slurm.cluster.MODULE_RE (a test keeps them equal): the names are written into job scripts
_MODULE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.+/-]*")
_CLUSTER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def record_path(home) -> Path:
    return Path(home) / "env" / RECORD_NAME


def problem(home) -> str | None:
    """Why ``<home>/env/mpd_env.json`` cannot be used, or None when it is usable or absent."""
    path = record_path(home)
    if not path.exists():
        return None
    return _check(path)[1]


def read(home) -> dict | None:
    """The record, or None when it is missing, unreadable or malformed (never raises)."""
    path = record_path(home)
    if not path.is_file():
        return None
    return _check(path)[0]


def _check(path: Path) -> tuple[dict | None, str | None]:
    try:
        if path.stat().st_size > MAX_BYTES:
            return None, f"{path} is larger than {MAX_BYTES} bytes"
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        return None, f"{path} cannot be read ({type(exc).__name__})"
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        return None, f"{path} is not a schema-{SCHEMA} environment record"
    modules = data.get("modules", [])
    if not isinstance(modules, list) or not all(isinstance(m, str) and _MODULE.fullmatch(m) for m in modules):
        return None, f"{path} lists modules that are not module names"
    python = data.get("python")
    if python is not None and (not isinstance(python, str) or not os.path.isabs(python) or "\n" in python):
        return None, f"{path} has a python entry that is not an absolute path"
    return data, None


def write(home, *, cluster: str, modules=(), constraints: str | None = None, python: str | None = None,
          python_version: str | None = None, now: datetime | None = None) -> dict:
    """Write the record atomically and return it. ``constraints`` is the file the environment was installed with."""
    if not _CLUSTER.fullmatch(cluster):
        raise ValueError(f"cluster name {cluster!r} is not a profile name")
    modules = [m for m in modules]
    bad = [m for m in modules if not _MODULE.fullmatch(m)]
    if bad:
        raise ValueError(f"not a module name: {', '.join(map(repr, bad))}")
    home = Path(os.path.abspath(home))
    packages = {}
    for name in PACKAGES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass                                  # e.g. torch under --no-torch
    try:
        from . import __version__ as mpd_version
    except ImportError:
        mpd_version = "unknown"
    record = {
        "schema": SCHEMA,
        "created": (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cluster": cluster,
        "home": str(home),
        "python": python or sys.executable,
        "python_version": python_version or platform.python_version(),
        "modules": modules,
        "mpd_version": mpd_version,
        "constraints": constraints,
        "constraints_sha256": None if constraints is None else hashlib.sha256(Path(constraints).read_bytes()).hexdigest(),
        "packages": packages,
    }
    path = record_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return record


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m mpd.envrecord", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("write", help="write <home>/env/mpd_env.json for the Python that runs this command")
    w.add_argument("--home", required=True)
    w.add_argument("--cluster", required=True)
    w.add_argument("--modules", default="", help='modules the job scripts load, e.g. "Anaconda3/2024.02-1"; "none" for no module')
    w.add_argument("--constraints", help="the constraints file the environment was installed with (its sha256 is recorded)")
    s = sub.add_parser("show", help="print the record, or say why there is none")
    s.add_argument("--home", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "show":
        record = read(args.home)
        if record is None:
            print(problem(args.home) or f"no environment record at {record_path(args.home)}", file=sys.stderr)
            return 1
        print(json.dumps(record, indent=2))
        return 0
    names = [m for m in re.split(r"[\s,]+", args.modules.strip()) if m]
    names = [] if names == ["none"] else names
    try:
        record = write(args.home, cluster=args.cluster, modules=names, constraints=args.constraints)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {record_path(record['home'])} (python {record['python_version']}, modules: {' '.join(names) or 'none'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
