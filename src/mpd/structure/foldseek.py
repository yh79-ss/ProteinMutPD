"""Foldseek as a separate process: find the binary, ask its version, turn a structure file into 3Di strings.

Foldseek is GPL-3.0 and is never bundled; `mpd.structure.tools.fetch_foldseek` downloads the pinned upstream release
and this module only runs it on the command line and reads a text file back. Stdlib only.
"""

from __future__ import annotations

import hashlib
import os
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ..data.sources import FOLDSEEK, FOLDSEEK_REFERENCE_BUILD
from .vocab import STATES

# structure file -> one line per chain: "<file>_<chain>", amino acids, 3Di (upper case), a feature vector
COMMAND = ("structureto3didescriptor", "-v", "0", "--threads", "1", "--chain-name-mode", "1")
TIMEOUT_S = 300


class EngineError(RuntimeError):
    """A 3Di engine could not run or its output could not be read."""


class FoldseekError(EngineError):
    pass


@dataclass(frozen=True)
class Chain:
    name: str          # "<file>_<chain>" as printed (a trailing description is dropped)
    chain: str
    aa: str
    states: str        # 3Di letters as printed, upper case


def where_flags(cluster: str | None = None, home: Path | str | None = None, explicit_home: bool = False) -> str:
    """`--cluster NAME`, plus `--home DIR` when no cluster is known or the home is not the profile's default."""
    parts = [f"--cluster {cluster}"] if cluster else []
    if home and (explicit_home or not cluster):
        parts.append(f"--home {shlex.quote(str(home))}")
    return " ".join(parts) or "--cluster <cluster>"


def fix_command(cluster: str | None = None, home: Path | str | None = None, *, explicit_home: bool = False) -> str:
    return f"mpd fetch foldseek {where_flags(cluster, home, explicit_home)}"


def pinned_path(home: Path | str) -> Path:
    return Path(home) / "tools" / FOLDSEEK.dirname / FOLDSEEK.binary


def _runnable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def override() -> Path | None:
    """`MPD_FOLDSEEK` (a binary, or a folder holding `bin/foldseek`), as given. None when unset."""
    value = os.environ.get("MPD_FOLDSEEK")
    return Path(value).expanduser() if value else None


def locate(home: Path | str | None = None) -> Path | None:
    """The Foldseek binary to use, or None. `MPD_FOLDSEEK` wins and is never second-guessed: if it is set but is not an
    executable file there is no fallback to `<home>/tools`."""
    given = override()
    if given is not None:
        for path in (given, given / "bin" / "foldseek", given / "foldseek" / "bin" / "foldseek"):
            if _runnable(path):
                return path.resolve()
        return None
    home = home or os.environ.get("MPD_HOME")
    if home:
        for path in (pinned_path(home), Path(home) / "tools" / "foldseek" / "bin" / "foldseek"):
            if _runnable(path):
                return path
    return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _tail(text: str, limit: int = 400) -> str:
    text = (text or "").strip()
    return text[-limit:] if len(text) > limit else text


def _exec(args: list[str], *, cwd: str | None = None, timeout: int = TIMEOUT_S) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        raise FoldseekError(f"{Path(args[0]).name} {args[1]} did not finish within {timeout} s") from None
    except OSError as exc:
        raise FoldseekError(f"cannot execute {args[0]}: {exc.strerror or exc}. A noexec mount or a binary built for "
                            f"another CPU can cause this; try `{args[0]} version` by hand.") from None


def version(binary: Path | str) -> str:
    """First line of `foldseek version` (a commit hash or release tag, depending on the build)."""
    proc = _exec([str(binary), "version"], timeout=60)
    text = (proc.stdout or proc.stderr).strip()
    if proc.returncode != 0 or not text:
        raise FoldseekError(f"`{binary} version` exited with status {proc.returncode}: {_tail(proc.stderr or proc.stdout)}")
    return text.splitlines()[0].strip()


def parse(text: str, source: str = "foldseek output") -> list[Chain]:
    chains = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            raise FoldseekError(f"{source}, line {number}: expected tab-separated name, sequence and 3Di; got {line[:60]!r}")
        name, aa, states = parts[0].split()[0] if parts[0].split() else "", parts[1].strip(), parts[2].strip()
        if not name or not aa or len(aa) != len(states):
            raise FoldseekError(f"{source}, line {number}: sequence has {len(aa)} residues but the 3Di string has {len(states)}")
        chains.append(Chain(name, name.rsplit("_", 1)[-1], aa, states))
    if not chains:
        raise FoldseekError(f"{source}: no chains (does the file have protein ATOM records with N, CA and C atoms?)")
    return chains


def run(binary: Path | str, pdb: Path | str, *, timeout: int = TIMEOUT_S) -> list[Chain]:
    """3Di of every chain of `pdb`. Runs in a private temporary directory, so Foldseek's output files are always removed."""
    pdb = Path(pdb)
    with tempfile.TemporaryDirectory(prefix="mpd-3di-") as tmp:
        out = Path(tmp) / "out.tsv"
        proc = _exec([str(binary), *COMMAND, str(pdb.resolve()), str(out)], cwd=tmp, timeout=timeout)
        if proc.returncode != 0:
            raise FoldseekError(f"foldseek exited with status {proc.returncode} on {pdb.name}: {_tail(proc.stderr or proc.stdout)}")
        if not out.is_file():
            raise FoldseekError(f"foldseek wrote no output for {pdb.name}")
        return parse(out.read_text(encoding="utf-8"), pdb.name)


def normalise_states(raw: str) -> tuple[str, int]:
    """Foldseek's upper-case 3Di letters -> SaProt's lower-case states; the unknown state `X` becomes `#`.
    Returns the string and how many unknown states it had."""
    low = raw.lower()
    unknown = low.count("x")
    low = low.replace("x", "#")
    bad = sorted(set(low) - set(STATES))
    if bad:
        raise FoldseekError(f"unexpected 3Di character(s) {''.join(bad)!r} in Foldseek output")
    return low, unknown


class FoldseekEngine:
    """A Foldseek binary as a 3Di source: `chains(pdb)`, an `id` for the cache, a `describe()` for structures.json."""
    name = "foldseek"

    def __init__(self, binary: Path | str):
        self.binary = Path(binary)
        self.version = version(self.binary)                    # fails early if the binary does not run
        self.binary_sha256 = sha256_file(self.binary)

    @property
    def id(self) -> str:
        return f"foldseek-{self.binary_sha256[:12]}"

    def describe(self) -> dict:
        return {"name": self.name, "version": self.version, "binary_sha256": self.binary_sha256,
                "command": "foldseek " + " ".join(COMMAND) + " <structure.pdb> <out.tsv>",
                "pinned_release": self.binary_sha256 == FOLDSEEK.binary_sha256,
                "reference_build": self.binary_sha256 == FOLDSEEK_REFERENCE_BUILD["binary_sha256"]}

    def chains(self, pdb: Path | str) -> list[Chain]:
        return run(self.binary, pdb)
