"""Download pinned data, verify checksums, unpack. Safe on a login node: standard library only, no torch."""

from __future__ import annotations

import email.utils
import errno
import hashlib
import http.client
import json
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from . import venus_structures as vs
from .sources import ZENODO_FILES, RemoteFile, Source

CHUNK = 1 << 20


class FetchError(RuntimeError):
    pass


class DownloadError(FetchError):
    """A transfer that failed. `code` is the HTTP status the server answered with, if it answered."""

    def __init__(self, message: str, code: int | None = None):
        super().__init__(message)
        self.code = code


class NetworkError(DownloadError):
    """Connection trouble, or a server that kept failing: usually no route to the host."""


class TimeBudgetReached(FetchError):
    """The caller's deadline passed. Partial downloads are kept, so running the same command resumes them."""


def _digest(path: Path, algo: str) -> str:
    h = hashlib.new(algo, usedforsecurity=False) if algo == "md5" else hashlib.new(algo)
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _check(path: Path, spec: RemoteFile) -> str | None:
    """None if `path` matches `spec`, else a description of the mismatch."""
    if spec.size is not None and path.stat().st_size != spec.size:
        return f"size {path.stat().st_size} != expected {spec.size}"
    for algo in ("md5", "sha256"):
        want = getattr(spec, algo)
        if want and _digest(path, algo) != want:
            return f"{algo} mismatch"
    return None


def _url(base: str, name: str) -> str:
    url = f"{base.rstrip('/')}/{urllib.parse.quote(name)}"
    return url + "?download=1" if "zenodo.org" in base else url


def _user_agent() -> str:
    from .. import __version__
    return f"proteinmutpd/{__version__}"


def _request(url: str, headers: dict[str, str] | None = None) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": _user_agent(), **(headers or {})})


def _retry_after(headers, default: float) -> float:
    """Seconds to back off: the server's Retry-After (seconds or an HTTP date), else `default`; at most 3 minutes."""
    raw = headers.get("Retry-After") if headers is not None else None
    wait = 0.0
    if raw:
        try:
            wait = float(raw)
        except ValueError:
            try:
                wait = email.utils.parsedate_to_datetime(raw).timestamp() - time.time()
            except (TypeError, ValueError):
                pass
    return min(wait if wait > 0 else default, 180)


def _disk_full(exc: Exception) -> bool:
    return isinstance(exc, OSError) and exc.errno in (errno.ENOSPC, errno.EDQUOT)


def _disk_full_error(path: Path) -> DownloadError:
    return DownloadError(f"disk full or quota exceeded while writing {path} (Grace: run `showquota`, "
                         "and keep --home on $SCRATCH)")


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise TimeBudgetReached("stopped at the time limit; partial downloads are kept")


def _pause(seconds: float, deadline: float | None) -> None:
    if deadline is not None and time.monotonic() + seconds >= deadline:
        raise TimeBudgetReached("stopped at the time limit while waiting to retry; partial downloads are kept")
    time.sleep(seconds)


def _copy(resp, out, done: int, size: int | None, deadline: float | None,
          progress: Callable[[int], None] | None) -> None:
    while True:
        block = resp.read(CHUNK)
        if not block:
            return
        out.write(block)
        done += len(block)
        if progress:
            progress(done)
        if deadline is not None and time.monotonic() >= deadline and done != size:
            raise TimeBudgetReached("stopped at the time limit; partial downloads are kept")


def _download(url: str, target: Path, *, retries: int = 3, timeout: int = 60,
              log: Callable[[str], None] = print, resume: bool = False, size: int | None = None,
              deadline: float | None = None, progress: Callable[[int], None] | None = None) -> None:
    """Fetch `url` to `target` through `<target>.part`.

    By default any failure deletes the `.part` and the next attempt starts from byte 0. With `resume=True` the
    `.part` is kept and continued with an HTTP Range request, so a connection that dies after 9 GB loses
    nothing; `retries` then counts consecutive attempts that brought no new bytes. `size` (the pinned size) makes
    a short body detectable, `deadline` is a `time.monotonic()` value after which TimeBudgetReached is raised,
    and `progress(bytes_on_disk)` is called after every chunk.
    """
    if resume:
        return _download_resumable(url, target, size=size, retries=retries, timeout=timeout, log=log,
                                   deadline=deadline, progress=progress)
    part = target.with_name(target.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(_request(url), timeout=timeout) as resp, open(part, "wb") as out:
                shutil.copyfileobj(resp, out, CHUNK)
            part.replace(target)
            return
        except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
            part.unlink(missing_ok=True)
            if _disk_full(exc):
                raise _disk_full_error(part) from exc
            if attempt == retries:
                raise NetworkError(f"could not download {url}: {exc}", getattr(exc, "code", None)) from exc
            wait = 2 * attempt
            if isinstance(exc, urllib.error.HTTPError) and exc.code in (429, 503):
                # rate limited: Hugging Face and Zenodo both ask clients to back off
                wait = _retry_after(exc.headers, 20 * attempt)
            log(f"  retry {attempt}/{retries - 1} in {wait:g}s after: {exc}")
            time.sleep(wait)


def _download_resumable(url: str, target: Path, *, size: int | None, retries: int, timeout: int,
                        log: Callable[[str], None], deadline: float | None,
                        progress: Callable[[int], None] | None) -> None:
    part = target.with_name(target.name + ".part")
    failures = 0                                  # consecutive attempts that brought no new bytes
    for _ in range(retries * 20):                 # ceiling on requests per file, whatever the progress
        have = part.stat().st_size if part.exists() else 0
        if size is not None and have > size:
            part.unlink()                         # cannot be a prefix of this file
            have = 0
        if size is not None and have == size:
            break
        _check_deadline(deadline)
        floor, error, wait = have, None, 0.0
        try:
            with urllib.request.urlopen(_request(url, {"Range": f"bytes={have}-"} if have else None),
                                        timeout=timeout) as resp:
                if have and resp.status == 206:
                    if not (resp.headers.get("Content-Range") or "").startswith(f"bytes {have}-"):
                        part.unlink(missing_ok=True)
                        raise DownloadError(f"{url}: the server answered a Range request with "
                                            f"{resp.headers.get('Content-Range')!r}; discarded the partial file")
                    mode = "ab"
                else:
                    mode, floor = "wb", 0         # first request, or the server ignored Range: start over
                with open(part, mode) as out:
                    _copy(resp, out, floor, size, deadline, progress)
        except urllib.error.HTTPError as exc:
            exc.close()
            error, wait = exc, 2 * (failures + 1)
            if exc.code == 416:                   # the Range starts at or past the end of the file
                if size is None:
                    break                         # the .part already is the whole file
                part.unlink(missing_ok=True)      # our prefix does not fit what the server has
            elif exc.code in (429, 503):
                wait = _retry_after(exc.headers, 20 * (failures + 1))
            elif exc.code not in (408, 500, 502, 504):
                raise DownloadError(f"could not download {url}: HTTP {exc.code} {exc.reason}", exc.code) from exc
        except (urllib.error.URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
            if _disk_full(exc):
                raise _disk_full_error(part) from exc
            error, wait = exc, 2 * (failures + 1)
        now = part.stat().st_size if part.exists() else 0
        if error is None and (size is None or now == size):
            break
        failures = failures + 1 if now <= floor else 0
        if failures >= retries:
            why = error if error is not None else f"the server ended the body after {now} of {size} bytes"
            raise NetworkError(f"could not download {url}: {why}", getattr(error, "code", None)) from error
        if error is not None:
            log(f"  retry in {wait:g}s after: {error}")
            _pause(wait, deadline)
    if not part.exists() or (size is not None and part.stat().st_size != size):
        raise NetworkError(f"{target.name}: incomplete download; the partial file is kept, run the command again")
    part.replace(target)


def _obtain(spec: RemoteFile, dest: Path, *, base_url: str | None, from_dir: Path | None,
            log: Callable[[str], None]) -> Path:
    target = dest / spec.name
    if target.is_file() and _check(target, spec) is None:
        log(f"  {spec.name}: already present, checksum ok")
        return target
    if from_dir is not None:
        src = from_dir / spec.name
        if not src.is_file():
            raise FetchError(f"{src} not found (--from-dir)")
        log(f"  {spec.name}: copying from {from_dir}")
        shutil.copyfile(src, target)
    else:
        log(f"  {spec.name}: downloading" + (f" ({spec.size / 1e6:.1f} MB)" if spec.size else ""))
        _download(_url(base_url, spec.name), target, log=log)
    problem = _check(target, spec)
    if problem:
        target.unlink()
        raise FetchError(f"{spec.name}: {problem}; removed the file")
    return target


def _unpack_zip(archive: Path, dest: Path) -> None:
    root = dest.resolve()
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            target = (root / name).resolve()
            if root != target and root not in target.parents:
                raise FetchError(f"{archive.name}: unsafe member path {name!r}")
        zf.extractall(root)


def _unpack(archive: Path, dest: Path) -> bool:
    """Extract unless this exact archive was already extracted. True if it extracted."""
    marker = dest / ".unpacked" / (archive.name + ".sha256")
    digest = _digest(archive, "sha256")
    if marker.is_file() and marker.read_text().strip() == digest:
        return False
    if archive.suffix == ".zip":
        _unpack_zip(archive, dest)
    else:
        vs.safe_extract_tar(archive, dest)
    marker.parent.mkdir(exist_ok=True)
    marker.write_text(digest + "\n")
    return True


def _hf_entries(source: Source) -> list[tuple[str, int, str]]:
    from importlib import resources
    text = (resources.files("mpd.data") / source.manifest).read_text(encoding="utf-8")
    return [(p, int(n), h) for p, n, h in (line.split("\t") for line in text.splitlines()[1:] if line)]


def _fetch_hf(source: Source, dest: Path, *, base_url: str | None, from_dir: Path | None, pause: float,
              log: Callable[[str], None]) -> Path:
    """Fetch every file of a pinned Hugging Face dataset revision, one request at a time, checking each sha256."""
    base = base_url or f"https://huggingface.co/datasets/{source.hf_repo}/resolve/{source.hf_revision}"
    entries = _hf_entries(source)
    todo = []
    for path, size, digest in entries:
        target = dest / path
        if not (target.is_file() and _check(target, RemoteFile(path, size, None, digest)) is None):
            todo.append((path, size, digest))
    log(f"  {len(entries) - len(todo)}/{len(entries)} files already present and verified")
    for i, (path, size, digest) in enumerate(todo, start=1):
        target = dest / path
        target.parent.mkdir(parents=True, exist_ok=True)
        if from_dir is not None:
            if not (from_dir / path).is_file():
                raise FetchError(f"{from_dir / path} not found (--from-dir)")
            shutil.copyfile(from_dir / path, target)
        else:
            _download(_url(base, path), target, retries=8, log=log)
            time.sleep(pause)
        problem = _check(target, RemoteFile(path, size, None, digest))
        if problem:
            target.unlink()
            raise FetchError(f"{path}: {problem}; removed the file")
        if i % 100 == 0:
            log(f"  {i}/{len(todo)} fetched")
    record = {"source": source.key, "title": source.title, "license": source.license, "hf_repo": source.hf_repo,
              "hf_revision": source.hf_revision, "files": len(entries),
              "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (dest / "FETCHED.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return dest


def _specs_from_sums(sums: dict[str, str]) -> tuple[RemoteFile, ...]:
    return tuple(RemoteFile(name, None, None, digest) for name, digest in sorted(sums.items()))


def fetch_source(source: Source, dest: Path, *, from_dir: Path | None = None, zenodo_record: str | None = None,
                 base_url: str | None = None, unpack: bool = True, pause: float = 0.12,
                 log: Callable[[str], None] = print) -> Path:
    """Fetch `source` into `dest`, verifying every file. Returns `dest`.

    With pinned files (ProteinGym) each download must match the md5/sha256 in `sources.py`. A source with no
    pinned files (the Venus structure set before its checksums are pinned) is verified against the
    `SHA256SUMS` that ships with the deposit instead; the deposit's own manifest is then checked per structure.
    """
    dest.mkdir(parents=True, exist_ok=True)
    if source.hf_repo:
        log(f"{source.title}  ->  {dest}")
        return _fetch_hf(source, dest, base_url=base_url, from_dir=from_dir, pause=pause, log=log)
    if base_url is None and from_dir is None:
        base_url = ZENODO_FILES.format(record=zenodo_record) if zenodo_record else source.base_url()
    log(f"{source.title}  ->  {dest}")
    specs, paths = source.files, {}
    if not specs:
        (dest / vs.SUMS_NAME).unlink(missing_ok=True)          # never trust a stale copy
        paths[vs.SUMS_NAME] = _obtain(RemoteFile(vs.SUMS_NAME, None, None, None), dest,
                                      base_url=base_url, from_dir=from_dir, log=log)
        specs = _specs_from_sums(vs.parse_sums(paths[vs.SUMS_NAME]))
    paths.update({s.name: _obtain(s, dest, base_url=base_url, from_dir=from_dir, log=log) for s in specs})
    if unpack:
        for name in source.unpack:
            if name in paths:
                log(f"  {'unpacked' if _unpack(paths[name], dest) else 'already unpacked'} {name}")
    if source.key == "venus-structures" and unpack:
        problems = vs.verify_structures(dest, vs.read_manifest(dest / vs.MANIFEST_NAME))
        if problems:
            raise FetchError("structure set failed verification:\n  " + "\n  ".join(problems[:10]))
        log(f"  verified {len(vs.read_manifest(dest / vs.MANIFEST_NAME))} structures against the manifest")
    record = {"source": source.key, "title": source.title, "doi": source.doi, "license": source.license,
              "zenodo_record": zenodo_record or source.zenodo_record,
              "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "files": {n: _digest(p, "sha256") for n, p in sorted(paths.items()) if p.is_file()}}
    (dest / "FETCHED.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return dest
