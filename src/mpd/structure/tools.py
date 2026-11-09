"""Download the pinned Foldseek release into `<home>/tools`. Run on a login node (compute nodes have no internet).

Nothing is executed before its sha256 matches the pin: the archive is hashed before it is opened, the extracted binary
before it is made executable. Foldseek is GPL-3.0 and is fetched from its owners, never bundled. Stdlib only.
"""

from __future__ import annotations

import json
import os
import shutil
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from ..data import venus_structures as vs
from ..data.fetch import FetchError
from ..data.sources import FOLDSEEK, Tool
from .foldseek import FoldseekError, sha256_file, version

RETRIES = 4


def _download(url: str, target: Path, log: Callable[[str], None], *, timeout: int = 60) -> None:
    part = target.with_name(target.name + ".part")
    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "mpd-fetch"}), timeout=timeout) as resp, \
                    open(part, "wb") as out:
                shutil.copyfileobj(resp, out, 1 << 20)
            part.replace(target)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            part.unlink(missing_ok=True)
            if attempt == RETRIES:
                raise FetchError(f"could not download {url}: {exc}. On a cluster without internet access download the file "
                                 f"elsewhere and pass it with --from-dir <folder holding it>.") from exc
            wait = 2 * attempt
            if isinstance(exc, urllib.error.HTTPError) and exc.code in (429, 503):
                wait = min(int(exc.headers.get("Retry-After") or 0) or 20 * attempt, 180)       # honour the server's request
            log(f"  retry {attempt}/{RETRIES - 1} in {wait}s after: {exc}")
            time.sleep(wait)


def _archive_problem(path: Path, pin: Tool) -> str | None:
    if path.stat().st_size != pin.archive_size:
        return f"size {path.stat().st_size} != expected {pin.archive_size}"
    if sha256_file(path) != pin.archive_sha256:
        return "sha256 mismatch"
    return None


def _resolve_local(from_file: Path | str, pin: Tool) -> Path:
    path = Path(from_file).expanduser()
    if path.is_dir():
        path = path / pin.archive
    if not path.is_file():
        raise FetchError(f"{path} not found (--from-dir / from_file); expected the {pin.archive} of release {pin.release}")
    return path


def _install(archive: Path, tools: Path, pin: Tool) -> Path:
    """Extract next to the destination, check the binary against the pin, then move the folder into place."""
    dest = tools / pin.dirname
    stage = tools / f".{pin.dirname}.incomplete-{os.getpid()}"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    try:
        try:
            vs.safe_extract_tar(archive, stage)
        except ValueError as exc:
            raise FetchError(f"{archive.name}: {exc}") from exc
        staged = stage / pin.binary
        if not staged.is_file():
            raise FetchError(f"{archive.name} has no {pin.binary}")
        if sha256_file(staged) != pin.binary_sha256:
            raise FetchError(f"{pin.binary} in {archive.name} does not match the pinned sha256; nothing was run")
        staged.chmod(0o755)
        if dest.exists():
            shutil.rmtree(dest)
        stage.rename(dest)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return dest / pin.binary


def fetch_foldseek(home: Path | str, *, from_file: Path | str | None = None, log: Callable[[str], None] = print,
                   pin: Tool | None = None, url: str | None = None) -> Path:
    """Install the pinned Foldseek under `<home>/tools/<dirname>` and return its binary; safe to repeat.

    `from_file`: the release archive, or a folder holding it, for a machine that cannot download. `pin`, `url`: for tests."""
    pin = pin or FOLDSEEK
    tools = Path(home) / "tools"
    dest = tools / pin.dirname
    binary = dest / pin.binary
    log(f"{pin.title} (licence {pin.license}; fetched from its owners and run as a separate program, never bundled)")
    if binary.is_file() and sha256_file(binary) == pin.binary_sha256:
        binary.chmod(0o755)
        log(f"  already installed: {binary}")
    else:
        tools.mkdir(parents=True, exist_ok=True)
        downloaded = None
        if from_file is not None:
            archive = _resolve_local(from_file, pin)
            log(f"  using {archive}")
        else:
            archive = downloaded = tools / f".{pin.archive}.{os.getpid()}"
            log(f"  downloading {pin.archive} ({pin.archive_size / 1e6:.1f} MB) from {url or pin.url}")
            _download(url or pin.url, downloaded, log)
        try:
            problem = _archive_problem(archive, pin)
            if problem:
                raise FetchError(f"{archive.name}: {problem}; it is not release {pin.release}"
                                 + ("; removed the download" if downloaded else ""))
            binary = _install(archive, tools, pin)
        finally:
            if downloaded is not None:
                downloaded.unlink(missing_ok=True)
    try:
        shown = version(binary)
    except FoldseekError as exc:
        raise FetchError(f"the pinned Foldseek binary (sha256 verified) does not run here: {exc}. The usual causes are a "
                         f"noexec file system or a CPU without AVX2; use another --home, or set MPD_FOLDSEEK to a Foldseek "
                         f"that runs (for example `conda install -c conda-forge -c bioconda foldseek`).") from exc
    expected = pin.commit[:7]
    matches = expected in shown
    if not matches:
        log(f"  warning: `foldseek version` printed {shown!r}, which does not contain the release commit {expected}")
    record = {"tool": pin.key, "release": pin.release, "commit": pin.commit, "url": pin.url, "archive": pin.archive,
              "archive_sha256": pin.archive_sha256, "binary": pin.binary, "binary_sha256": pin.binary_sha256,
              "license": pin.license, "version_output": shown, "version_names_release": matches,
              "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (dest / "FETCHED.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log(f"  ready: foldseek version {shown}")
    return binary
