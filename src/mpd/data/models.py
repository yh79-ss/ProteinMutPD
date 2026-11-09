"""Pinned model weights: where they live, whether they are usable, and how to fetch them.

Weights are plain folders, `<home>/models/<key>/` (or `$MPD_MODELS_DIR/<key>`), holding exactly the files listed in
`models.tsv`, each pinned by size and sha256 to one Hugging Face commit (`ModelInfo.revision`). Jobs run offline, so
everything is fetched on a login node first. Standard library only: the planner and `mpd fetch` run without torch.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import json
import os
import re
import shlex
import shutil
import time
import urllib.parse
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Callable, Iterator

from .. import registry
from ..checks import Check
from . import fetch as F
from .fetch import FetchError, TimeBudgetReached
from .sources import RemoteFile

DEFAULT_ENDPOINT = "https://huggingface.co"
MANIFEST = "models.tsv"
MARKER = "FETCHED.json"                  # same file name and field names as the data fetches
HEADER = ["model_key", "file", "size", "sha256"]
PAUSE_S = 0.3                            # between files; a model is at most 7 requests
GIB = 1 << 30
STATES = ("ready", "override", "missing", "incomplete", "corrupt", "unverified", "stale", "unpinned")
_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")
_SHA = re.compile(r"[0-9a-f]{64}")
_COMMIT = re.compile(r"[0-9a-f]{40}")


class ModelError(FetchError):
    pass


class ModelNotReady(ModelError):
    pass


@dataclass(frozen=True)
class Pin:
    """What `mpd` fetches for one model: the commit and every file with its size and sha256."""
    key: str
    hf_id: str
    revision: str
    license: str
    files: tuple[RemoteFile, ...]

    @property
    def total_bytes(self) -> int:
        return sum(f.size for f in self.files)

    @property
    def weights_format(self) -> str:
        names = [f.name for f in self.files]
        return "safetensors" if any(n.endswith(".safetensors") for n in names) else "pickle"


@dataclass(frozen=True)
class Status:
    """`state`: ready | override | missing | incomplete | corrupt | unverified | stale | unpinned (docs/MODELS.md)."""
    key: str
    state: str
    path: Path | None
    size_bytes: int = 0                  # bytes of the pinned files (0 for an override or an unpinned model)
    missing: tuple[str, ...] = ()        # pinned files that are absent or fail the size/sha256 check
    detail: str = ""
    todo_bytes: int = 0                  # bytes still to fetch

    @property
    def ok(self) -> bool:
        return self.state in ("ready", "override")


# -- manifest -----------------------------------------------------------------

def load_manifest(text: str) -> dict[str, tuple[RemoteFile, ...]]:
    """Parse `models.tsv`: tab-separated, header row, one line per (model, file). Anything irregular is a ModelError."""
    lines = text.splitlines()
    if not lines or lines[0].split("\t") != HEADER:
        raise ModelError(f"{MANIFEST}: the header must be {' <tab> '.join(HEADER)}")
    rows: dict[str, list[RemoteFile]] = {}
    for number, line in enumerate(lines[1:], start=2):
        if not line.strip():
            continue
        parts = line.split("\t")
        if len(parts) != 4:
            raise ModelError(f"{MANIFEST}:{number}: expected 4 tab-separated fields")
        key, name, size, digest = parts
        if not _NAME.fullmatch(name):                 # becomes a URL segment and a path inside the model folder
            raise ModelError(f"{MANIFEST}:{number}: {name!r} is not a plain file name")
        if not re.fullmatch(r"[0-9]+", size) or int(size) <= 0:
            raise ModelError(f"{MANIFEST}:{number}: size {size!r} is not a positive integer")
        if not _SHA.fullmatch(digest):
            raise ModelError(f"{MANIFEST}:{number}: sha256 {digest!r} is not 64 lowercase hex digits")
        if any(f.name == name for f in rows.get(key, ())):
            raise ModelError(f"{MANIFEST}:{number}: {key}/{name} is listed twice")
        rows.setdefault(key, []).append(RemoteFile(name, int(size), None, digest))
    return {k: tuple(v) for k, v in rows.items()}


@lru_cache(maxsize=None)
def pins() -> dict[str, Pin]:
    """Every model with pinned files, in manifest order. Each needs a registry row with a 40-hex commit."""
    result = {}
    for key, files in load_manifest((resources.files("mpd.data") / MANIFEST).read_text(encoding="utf-8")).items():
        info = _info(key)
        if not info.revision or not _COMMIT.fullmatch(info.revision):
            raise ModelError(f"{MANIFEST} lists {key}, but registry.py has no 40-hex revision for it")
        result[key] = Pin(key, info.hf_id, info.revision, info.license, files)
    return result


def pin(key: str) -> Pin | None:
    _info(key)
    return pins().get(key)


def pinned_keys() -> list[str]:
    return list(pins())


def fetchable_keys() -> list[str]:
    """What `mpd fetch model --all` fetches: pinned models the pipeline can use."""
    return [k for k in pins() if registry.get(k).status == "available"]


def weights_format(key: str) -> str | None:
    """"safetensors", "pickle" (PyTorch pickle: needs torch >= 2.6 with current transformers) or None if unpinned."""
    p = pin(key)
    return p.weights_format if p else None


def _info(key: str) -> registry.ModelInfo:
    try:
        return registry.get(key)
    except ValueError as exc:
        raise ModelError(str(exc)) from None


def human_bytes(n: int | float) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.2f} GB"
    if n >= 1e6:
        return f"{n / 1e6:.1f} MB"
    if n >= 1e3:
        return f"{n / 1e3:.1f} kB"
    return f"{int(n)} B"


# -- layout -------------------------------------------------------------------

def override_env(key: str) -> str:
    return "MPD_MODEL_" + key.upper().replace("-", "_")


def override_path(key: str) -> Path | None:
    """The folder named by `MPD_MODEL_<KEY>`, if set. It wins over everything and is not checked against the pin."""
    value = os.environ.get(override_env(key))
    return Path(value).expanduser() if value else None


def models_root(home: Path | str | None = None) -> Path:
    """`$MPD_MODELS_DIR` if set, else `<home or $MPD_HOME>/models`."""
    env = os.environ.get("MPD_MODELS_DIR")
    if env:
        return Path(os.path.abspath(Path(env).expanduser()))
    base = home or os.environ.get("MPD_HOME")
    if not base:
        raise ModelError("cannot locate the models folder: pass --home or --cluster, "
                         "or set MPD_HOME (or MPD_MODELS_DIR)")
    return Path(os.path.abspath(Path(base).expanduser())) / "models"


def model_dir(key: str, home: Path | str | None = None) -> Path:
    return models_root(home) / key


# -- state --------------------------------------------------------------------

def status(key: str, home: Path | str | None = None, *, verify: bool = False) -> Status:
    """Where `key` stands. Stat calls only, so it is cheap on a login node; `verify=True` also re-hashes the files."""
    info = _info(key)
    override = override_path(key)
    if override is not None:
        ok = (override / "config.json").is_file()
        return Status(key, "override" if ok else "missing", override, 0, () if ok else ("config.json",),
                      f"{override_env(key)}={override}" + ("" if ok else " has no config.json"))
    p = pins().get(key)
    if p is None:
        return Status(key, "unpinned", None, 0, (), _unpinned_reason(info))
    dest = model_dir(key, home)
    try:
        return _folder_status(p, dest, verify)
    except OSError as exc:                       # e.g. a folder this user may not read
        raise ModelError(f"cannot read {dest}: {exc.strerror or exc}") from exc


def _unpinned_reason(info: registry.ModelInfo) -> str:
    if info.output_family_lock:
        return f"mpd never downloads {info.key} ({info.license or 'restricted licence'}); see docs/LICENSES.md"
    return f"no pinned files for {info.key} in {MANIFEST} (docs/ADDING_A_MODEL.md)"


def _problems(p: Pin, dest: Path, *, hash_files: bool, log: Callable[[str], None] | None = None) -> dict[str, str]:
    out = {}
    for f in p.files:
        path = dest / f.name
        if not path.is_file():
            out[f.name] = "missing"
        elif path.stat().st_size != f.size:
            out[f.name] = f"size {path.stat().st_size} != expected {f.size}"
        elif hash_files:
            if log and f.size > 100_000_000:
                log(f"  {f.name}: checking sha256 ({human_bytes(f.size)})")
            problem = F._check(path, f)
            if problem:
                out[f.name] = problem
    return out


def _partial(dest: Path, name: str) -> int:
    part = dest / (name + ".part")
    return part.stat().st_size if part.is_file() else 0


def _have(dest: Path, f: RemoteFile) -> int:
    """Bytes of `f` a fetch continues from: its partial download, else the file itself if it was cut short."""
    part = _partial(dest, f.name)
    if part:
        return part if part <= f.size else 0
    target = dest / f.name
    return target.stat().st_size if target.is_file() and target.stat().st_size < f.size else 0


def _folder_status(p: Pin, dest: Path, verify: bool = False) -> Status:
    names = tuple(f.name for f in p.files)
    if not any((dest / n).exists() or (dest / (n + ".part")).exists() for n in names):
        return Status(p.key, "missing", dest, p.total_bytes, names, "nothing fetched yet", p.total_bytes)
    problems = _problems(p, dest, hash_files=verify)
    todo = tuple(n for n in names if n in problems)
    todo_bytes = sum(f.size - _have(dest, f) for f in p.files if f.name in problems)
    bad = [n for n in todo if problems[n] != "missing"]
    if bad:
        why = "; ".join(f"{n}: {problems[n]}" for n in bad)
        return Status(p.key, "corrupt", dest, p.total_bytes, todo, why, todo_bytes)
    if todo:
        notes = [f"{n} (partial download of {human_bytes(_partial(dest, n))} kept)" if _partial(dest, n) else n
                 for n in todo]
        return Status(p.key, "incomplete", dest, p.total_bytes, todo, "missing: " + ", ".join(notes), todo_bytes)
    try:
        record = json.loads((dest / MARKER).read_text(encoding="utf-8"))
        if not isinstance(record, dict):
            raise ValueError(MARKER)
    except FileNotFoundError:
        return Status(p.key, "unverified", dest, p.total_bytes, (),
                      "files are present but were never checksum-verified")
    except (OSError, ValueError):
        return Status(p.key, "unverified", dest, p.total_bytes, (), f"{MARKER} is unreadable")
    if record.get("hf_revision") != p.revision or record.get("files") != {f.name: f.sha256 for f in p.files}:
        return Status(p.key, "stale", dest, p.total_bytes, (),
                      f"fetched at {str(record.get('hf_revision'))[:8]}, this release pins {p.revision[:8]}")
    return Status(p.key, "ready", dest, p.total_bytes)


def needed(spec) -> list[str]:
    """Models a run loads: the teacher when it runs live (a cached-logit teacher needs no weights), then the student."""
    keys = [spec.teacher] if spec.backend == "live" else []
    return list(dict.fromkeys([*keys, spec.student]))


def fix_command(keys, cluster: str | None = None, home: Path | str | None = None) -> str:
    parts = ["mpd", "fetch", "model", *keys]
    if cluster:
        parts += ["--cluster", cluster]
    if home:
        parts += ["--home", str(home)]
    return " ".join(shlex.quote(x) for x in parts)


def _sentence(s: Status) -> str:
    if s.state == "unpinned":
        return f"mpd cannot fetch weights for {s.key}: {s.detail}"
    if s.state == "missing" and override_path(s.key) is not None:
        return f"{s.detail}. Fix the path or unset the variable"
    what = {"missing": "are missing", "incomplete": "are incomplete", "corrupt": "are corrupt",
            "unverified": "have not been verified", "stale": "are not the ones this release pins"}[s.state]
    more = f"; {s.detail}" if s.detail and s.state != "missing" else ""
    return f"weights for {s.key} {what} ({s.path}{more})"


def _not_ready(bad: list[Status], cluster: str | None, home: Path | str | None) -> str:
    fixable = [s.key for s in bad if s.state != "unpinned" and not (s.state == "missing" and override_path(s.key))]
    fix = fix_command(fixable, cluster, home) if fixable else ""
    if len(bad) == 1:
        return _sentence(bad[0]) + (f". Run on a login node: {fix}" if fix else "")
    lines = ["model weights are not ready:"]
    lines += [f"  {s.key:<12}{s.state:<11}{human_bytes(s.size_bytes):>10}  {s.path or ''}" for s in bad]
    total = sum(s.todo_bytes for s in bad)
    if fix:
        lines.append(f"Run on a login node{f' ({human_bytes(total)} to download, resumable)' if total else ''}: {fix}")
    return "\n".join(lines)


def require_ready(keys, home: Path | str | None = None, *, cluster: str | None = None,
                  explicit_home: bool = False) -> None:
    """Raise ModelNotReady, with one fetch command, unless every model in `keys` is ready or overridden.

    `explicit_home`: the home came from --home, so the fetch command must repeat it.
    """
    bad = [s for s in (status(k, home) for k in dict.fromkeys(keys)) if not s.ok]
    if bad:
        raise ModelNotReady(_not_ready(bad, cluster, home if explicit_home else None))


def resolve(key: str, home: Path | str | None = None) -> str:
    """The folder to load `key` from: `MPD_MODEL_<KEY>`, else the verified copy under the models root.

    Never the Hub id: jobs are offline, and a fallback would only delay the failure until a GPU job starts.
    `MPD_ALLOW_HUB=1` (login node or laptop; never forwarded to jobs) allows the Hub id.
    """
    info = _info(key)
    hub = os.environ.get("MPD_ALLOW_HUB") == "1"
    try:
        s = status(key, home)
    except ModelError as exc:                    # no home or models folder configured
        if hub:
            return info.hf_id
        cluster = os.environ.get("MPD_CLUSTER") or None
        raise ModelNotReady(f"weights for {key} are missing ({exc}). Run on a login node: "
                            f"{fix_command([key], cluster)}") from None
    if s.ok:
        return str(s.path)
    if hub:
        return info.hf_id
    where = home or (None if os.environ.get("MPD_MODELS_DIR") else os.environ.get("MPD_HOME"))      # the fix must fetch to where this looked
    raise ModelNotReady(_not_ready([s], os.environ.get("MPD_CLUSTER") or None, where or None))


# -- fetching -----------------------------------------------------------------

def check_space(path: Path, need: int, *, margin: int = GIB, usage=None) -> None:
    """Refuse when the filesystem holding `path` has less than `need` + `margin` bytes free."""
    probe = path
    while not probe.exists():
        probe = probe.parent
    free = (usage or shutil.disk_usage)(probe).free
    if free < need + margin:
        raise ModelError(f"not enough free space under {probe}: need {human_bytes(need)} "
                         f"(+{margin / GIB:g} GiB margin), {human_bytes(free)} free. Choose another --home "
                         "(Grace: $SCRATCH has 1 TB, $HOME 10 GB; `showquota` shows your quota).")


@contextlib.contextmanager
def _locked(root: Path, key: str) -> Iterator[None]:
    """One fetch per model at a time: two processes appending to the same .part would corrupt it."""
    root.mkdir(parents=True, exist_ok=True)
    with open(root / f".{key}.lock", "a") as handle:
        try:
            import fcntl
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except ImportError:
            pass                                  # no flock on this platform
        except BlockingIOError:
            raise ModelError(f"another `mpd fetch model {key}` is already writing to {root / key}; "
                             "wait for it to finish, then run this again") from None
        except OSError:
            pass                                  # a filesystem without flock support
        yield


def _same(a: Path, b: Path) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def _find_source(from_dir: Path, p: Pin, dest: Path, wanted: list[RemoteFile]) -> Path:
    """The folder under `from_dir` with the `wanted` files: `<key>/`, the Hub repo name, a cache snapshot, or itself."""
    if not from_dir.is_dir():
        raise ModelError(f"--from-dir {from_dir} is not a folder")
    hub = "models--" + p.hf_id.replace("/", "--")
    candidates = [from_dir / p.key, from_dir / p.hf_id.split("/")[-1]]
    for root in (from_dir, from_dir / "hub"):
        snapshots = root / hub / "snapshots"
        candidates.append(snapshots / p.revision)
        if snapshots.is_dir():
            candidates += sorted(d for d in snapshots.iterdir() if d.is_dir())
    candidates.append(from_dir)
    partial = None
    for folder in candidates:
        if not folder.is_dir() or _same(folder, dest):
            continue
        missing = [f.name for f in wanted if not (folder / f.name).is_file()]
        if not missing:
            return folder
        if any(os.path.islink(folder / n) for n in missing) or (folder / "config.json").is_file():
            if partial is None or len(missing) < len(partial[1]):
                partial = (folder, missing)
    if partial:
        folder, missing = partial
        broken = [n for n in missing if os.path.islink(folder / n)]
        raise ModelError(f"{p.key}: {folder} lacks {', '.join(missing)}" + (
            " (broken symlinks: a Hugging Face cache loses its blobs when copied without them)" if broken else ""))
    raise ModelError(f"{p.key}: no folder with its files under {from_dir} (looked for {p.key}/, "
                     f"{p.hf_id.split('/')[-1]}/, a Hugging Face cache snapshot, and the folder itself)")


def _progress(name: str, size: int, log: Callable[[str], None]) -> Callable[[int], None] | None:
    if size < GIB:
        return None
    last = [0]

    def report(done: int) -> None:
        if done // GIB > last[0]:
            last[0] = done // GIB
            log(f"    {name}: {done / GIB:.0f} of {size / GIB:.1f} GiB")
    return report


def _download_one(p: Pin, f: RemoteFile, dest: Path, base: str, deadline: float | None,
                  log: Callable[[str], None]) -> None:
    def get(revision: str) -> None:
        url = f"{base}/{p.hf_id}/resolve/{revision}/{urllib.parse.quote(f.name)}"
        F._download(url, dest / f.name, retries=6, resume=True, size=f.size, deadline=deadline, log=log,
                    progress=_progress(f.name, f.size, log))

    try:
        try:
            get(p.revision)
        except F.DownloadError as exc:
            if exc.code != 404:
                raise
            # history rewritten upstream: take `main`; the sha256 check afterwards accepts only the pinned bytes
            log(f"  {f.name}: the pinned commit answers 404; trying 'main' (accepted only if the sha256 matches)")
            get("main")
    except F.NetworkError as exc:
        busy = exc.code in (408, 429) or (exc.code or 0) >= 500
        why = ("Hugging Face is busy or rate-limiting this address: run the same command again later, it resumes"
               if busy else "If this machine cannot reach huggingface.co (many compute nodes cannot), "
                            "fetch on one that can")
        raise ModelError(f"{exc}. {why}, or copy the folder from a machine that can: "
                         f"mpd fetch model {p.key} --from-dir <folder> (docs/MODELS.md)") from exc
    except F.DownloadError as exc:
        raise ModelError(str(exc)) from exc


def _write_marker(p: Pin, dest: Path, endpoint: str | None) -> None:
    record = {"source": p.key, "title": f"{p.hf_id} @ {p.revision[:8]}", "license": p.license, "hf_repo": p.hf_id,
              "hf_revision": p.revision, "endpoint": endpoint, "files": {f.name: f.sha256 for f in p.files},
              "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    tmp = dest / (MARKER + ".tmp")
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(dest / MARKER)                    # last, and atomically: a marker means the folder was complete


def _ready_lines(p: Pin, dest: Path, log: Callable[[str], None]) -> None:
    log(f"{p.key}: ready ({human_bytes(p.total_bytes)}, {dest})")
    if p.weights_format == "pickle":
        log(f"{p.key}: ships PyTorch pickle weights only; loading needs torch >= 2.6")


def _dry_run(p: Pin, dest: Path, st: Status, todo: list[RemoteFile], from_dir: Path | None,
             log: Callable[[str], None]) -> None:
    log(f"{p.key}: dry run, nothing is written ({p.hf_id} @ {p.revision[:8]} -> {dest}, state: {st.state})")
    if not todo:
        log("  every file is present; a real run would " + ("check them and write " + MARKER if st.state != "ready"
                                                             else "have nothing to do"))
        return
    if from_dir is not None:
        log(f"  would copy {len(todo)} file(s) from {_find_source(from_dir, p, dest, todo)}")
    else:
        log(f"  would download {len(todo)} file(s)")
    for f in todo:
        log(f"    {f.name}  {human_bytes(f.size)}")
    probe = dest
    while not probe.exists():
        probe = probe.parent
    need, free = sum(f.size for f in todo), shutil.disk_usage(probe).free
    log(f"  {human_bytes(need)} to fetch; {human_bytes(free)} free under {probe}"
        + ("" if free >= need + GIB else "  <- NOT ENOUGH"))


def fetch_model(key: str, home: Path | str | None, *, from_dir: Path | None = None, endpoint: str | None = None,
                verify: bool = False, force: bool = False, dry_run: bool = False, deadline: float | None = None,
                log: Callable[[str], None] = print) -> Status:
    """Make `<models>/<key>` match the pin: download (or copy from `from_dir`), check every file, mark it last.

    Resumable: a `.part` left by an interrupted run is continued. `verify` re-hashes files that look complete, `force`
    refetches everything, `dry_run` only reports. `deadline` is a `time.monotonic()` value; when it passes the fetch
    stops with TimeBudgetReached and keeps its partial files. Anonymous: no token is ever sent.
    """
    info = _info(key)
    p = pins().get(key)
    if p is None:
        if info.output_family_lock:
            raise ModelError(f"mpd never downloads {key}: {info.license_note} See docs/LICENSES.md.")
        raise ModelError(_unpinned_reason(info))
    dest = model_dir(key, home)
    if override_path(key) is not None:
        log(f"note: {override_env(key)} is set, so runs use that folder and not {dest}")
    if info.status != "available":
        log(f"note: {key} is {info.status} in this release; its weights can be fetched, "
            "but the pipeline cannot use them yet")
    st = _folder_status(p, dest)
    if st.state == "ready" and not (force or verify):
        _ready_lines(p, dest, log)
        return st
    if dry_run:
        names = {f.name for f in p.files} if force else set(st.missing)
        _dry_run(p, dest, st, [f for f in p.files if f.name in names], from_dir, log)
        return st
    base = (endpoint or os.environ.get("HF_ENDPOINT") or DEFAULT_ENDPOINT).rstrip("/")
    try:
        with _locked(models_root(home), key):
            problems = {} if force else _problems(p, dest, hash_files=True, log=log)
            todo = list(p.files) if force else [f for f in p.files if f.name in problems]
            if not todo and st.state == "ready":
                log(f"{key}: all {len(p.files)} files re-hashed; they match the pin")
                _ready_lines(p, dest, log)
                return st
            src = _find_source(from_dir, p, dest, todo) if (from_dir is not None and todo) else None
            need = sum(f.size - (0 if force else _have(dest, f)) for f in todo)
            if todo:
                check_space(dest, need)
            dest.mkdir(parents=True, exist_ok=True)
            (dest / MARKER).unlink(missing_ok=True)      # the folder is about to change
            if todo:
                log(f"{key}: {p.hf_id} @ {p.revision[:8]} -> {dest} ({human_bytes(need)} to get)")
            else:
                log(f"{key}: all files present and checksums ok; recording that in {MARKER}")
            if src is not None:
                log(f"{key}: copying from {src}")
            for f in p.files:
                if f not in todo:
                    log(f"  {f.name}: present, checksum ok")
                    continue
                try:
                    F._check_deadline(deadline)
                    _get_file(p, f, dest, src, base, force, deadline, log)
                except TimeBudgetReached:
                    left = sum(1 for g in todo if not (dest / g.name).is_file())
                    raise TimeBudgetReached(f"{key}: stopped before the login-node time limit with {left} file(s) "
                                            "left; run the same command again to resume (partial downloads are kept)"
                                            ) from None
                if src is None and f is not todo[-1]:
                    time.sleep(PAUSE_S)
            _write_marker(p, dest, None if src is not None else base)
    except OSError as exc:
        if F._disk_full(exc):
            raise ModelError(f"{key}: {F._disk_full_error(dest)}") from exc
        raise ModelError(f"{key}: file system error: {exc}") from exc
    _ready_lines(p, dest, log)
    return _folder_status(p, dest)


def _get_file(p: Pin, f: RemoteFile, dest: Path, src: Path | None, base: str, force: bool, deadline, log) -> None:
    target, part = dest / f.name, dest / (f.name + ".part")
    if force:
        part.unlink(missing_ok=True)
    if target.is_file():
        # a short file may be a good prefix (an interrupted copy): let it resume; anything else is replaced
        if not force and target.stat().st_size < f.size and not part.exists():
            target.replace(part)
        else:
            target.unlink()
    for attempt in (1, 2):
        if src is not None:
            log(f"  {f.name}: copying ({human_bytes(f.size)})")
            try:
                shutil.copyfile(src / f.name, part)
            except OSError as exc:
                if F._disk_full(exc):
                    raise ModelError(str(F._disk_full_error(part))) from exc
                raise
            part.replace(target)
        else:
            have = part.stat().st_size if part.is_file() else 0
            log(f"  {f.name}: " + (f"resuming at {human_bytes(have)} of {human_bytes(f.size)}" if have
                                   else f"downloading ({human_bytes(f.size)})"))
            _download_one(p, f, dest, base, deadline, log)
        problem = F._check(target, f)
        if not problem:
            return
        target.unlink(missing_ok=True)
        part.unlink(missing_ok=True)
        if attempt == 2 or src is not None:
            raise ModelError(f"{p.key}/{f.name}: {problem}; removed the file")
        log(f"  {f.name}: {problem}; retrying once from byte 0")


# -- preflight ----------------------------------------------------------------

def _torch_check(keys: list[str]) -> Check:
    name = "torch for pickle weights"
    try:
        version = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return Check(name, "skip", "torch is not installed in this Python, so its version cannot be checked here "
                     "(`mpd doctor --deep` checks the job environment)")
    major_minor = tuple(int(x) for x in re.findall(r"\d+", version)[:2])
    if major_minor < (2, 6):
        return Check(name, "warn", f"{', '.join(keys)} ship PyTorch pickle weights only, and current transformers "
                     f"refuses to load those with torch {version}; install torch >= 2.7", "bash scripts/setup_env.sh")
    return Check(name, "ok", f"torch {version} can load the pickle weights of {', '.join(keys)}")


def preflight_checks(spec, home, cluster: str | None = None, *, explicit_home: bool = False) -> list[Check]:
    """One check per model the run loads (ready or an override is ok, anything else is an error with its fix)."""
    checks = []
    keys = needed(spec)
    for key in keys:
        name = f"weights {key}"
        try:
            s = status(key, home)
        except ModelError as exc:
            checks.append(Check(name, "error", str(exc), fix_command([key], cluster)))
            continue
        if s.state == "ready":
            checks.append(Check(name, "ok", f"ready: {human_bytes(s.size_bytes)} at {s.path}"))
        elif s.state == "override":
            checks.append(Check(name, "ok", f"using {s.detail} (not checked against the pin)"))
        else:
            fix = None if s.state == "unpinned" else fix_command([key], cluster, home if explicit_home else None)
            checks.append(Check(name, "error", _sentence(s), fix))
    pickled = [k for k in keys if weights_format(k) == "pickle"]
    if pickled:
        checks.append(_torch_check(pickled))
    return checks
