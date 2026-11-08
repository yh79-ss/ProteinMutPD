"""Download pinned data, verify checksums, unpack. Safe on a login node: standard library only, no torch."""

from __future__ import annotations

import hashlib
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


class FetchError(RuntimeError):
    pass


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


def _download(url: str, target: Path, *, retries: int = 3, timeout: int = 60,
              log: Callable[[str], None] = print) -> None:
    part = target.with_name(target.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp, open(part, "wb") as out:
                shutil.copyfileobj(resp, out, 1 << 20)
            part.replace(target)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            part.unlink(missing_ok=True)
            if attempt == retries:
                raise FetchError(f"could not download {url}: {exc}") from exc
            wait = 2 * attempt
            if isinstance(exc, urllib.error.HTTPError) and exc.code in (429, 503):
                # rate limited: Hugging Face and Zenodo both ask clients to back off
                wait = min(int(exc.headers.get("Retry-After") or 0) or 20 * attempt, 180)
            log(f"  retry {attempt}/{retries - 1} in {wait}s after: {exc}")
            time.sleep(wait)


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
