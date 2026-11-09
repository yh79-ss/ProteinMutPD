"""Rebuild `src/mpd/data/models.tsv` and the registry pins from the Hugging Face API. Maintainers only; needs network.

    python scripts/maintainers/pin_models.py               # check: rebuild the rows at the PINNED commits, diff them
    python scripts/maintainers/pin_models.py --crosscheck  # ...and compare each large file with a second HTTP endpoint
    python scripts/maintainers/pin_models.py --at main     # bump: rebuild at each repo's current main, print new pins
    python scripts/maintainers/pin_models.py --at main --write   # ...and overwrite models.tsv (then edit registry.py)

For every model in `registry.py` without an output licence lock (so never AIDO) it reads the repository listing at one
commit, keeps the files `from_pretrained(<folder>)` needs (config, three tokenizer files, and the first weights format
that exists: model.safetensors, sharded safetensors, pytorch_model.bin, sharded .bin), and records each file's size and
sha256. The sha256 of a large file is the Git LFS object id from the listing; `--crosscheck` compares it with the
X-Linked-ETag header of the resolve endpoint. Small files are downloaded and hashed, and each download must match the
listing's git blob id.

The exit status is 0 when the rebuilt manifest and registry pins equal the repository's, 1 when they differ, 2 on
errors.
Anonymous and sequential: no token is ever sent, requests are 1.5 s apart, Retry-After is honoured. A full run is about
30 API calls, 31 downloads of under 100 kB and (with --crosscheck) 8 HEAD requests.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from mpd import registry  # noqa: E402
from mpd.data import models  # noqa: E402

UA = "proteinmutpd-pin-models/0.1"
TOKENIZER_FILES = ("special_tokens_map.json", "tokenizer_config.json", "vocab.txt")
SHARDS = {"model.safetensors.index.json": re.compile(r"model-\d{5}-of-\d{5}\.safetensors"),
          "pytorch_model.bin.index.json": re.compile(r"pytorch_model-\d{5}-of-\d{5}\.bin")}


def select_files(names) -> list[str]:
    """The files needed to load a model from a plain folder; the first weights format that exists wins."""
    names = set(names)
    need = ["config.json", *TOKENIZER_FILES]
    absent = [n for n in need if n not in names]
    if absent:
        raise ValueError(f"the repository has no {', '.join(absent)}")
    if "model.safetensors" in names:
        weights = ["model.safetensors"]
    elif "pytorch_model.bin" in names:
        weights = ["pytorch_model.bin"]
    else:
        for index, pattern in SHARDS.items():
            if index in names:
                weights = sorted(n for n in names if pattern.fullmatch(n)) + [index]
                break
        else:
            raise ValueError("the repository has no weights file "
                             "(model.safetensors, pytorch_model.bin or a sharded index)")
    return sorted({*need, *weights})


def git_blob_oid(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def index_listing(listing: list[dict]) -> dict[str, dict]:
    return {e["path"]: e for e in listing if e.get("type") == "file"}


def render_manifest(rows) -> str:
    return "".join("\t".join(str(x) for x in r) + "\n" for r in [tuple(models.HEADER), *rows])


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def request(url: str, *, method: str = "GET", follow: bool = True, pause: float = 1.5, tries: int = 5):
    """(status, lower-cased headers, body). Sleeps `pause` afterwards; waits out 429/503 and connection errors."""
    opener = urllib.request.build_opener() if follow else urllib.request.build_opener(NoRedirect)
    for attempt in range(1, tries + 1):
        try:
            req = urllib.request.Request(url, method=method, headers={"User-Agent": UA})
            with opener.open(req, timeout=60) as resp:
                headers = {k.lower(): v for k, v in resp.getheaders()}
                body = resp.read() if method == "GET" else b""
                result = (resp.status, headers, body)
        except urllib.error.HTTPError as exc:
            if exc.code in (429, 503) and attempt < tries:
                wait = min(int(exc.headers.get("Retry-After") or 0) or 20 * attempt, 180)
                print(f"  HTTP {exc.code} from {url}; waiting {wait}s", flush=True)
                time.sleep(wait)
                continue
            if exc.code in (301, 302, 303, 307, 308) and not follow:    # a HEAD that must not follow the CDN redirect
                result = (exc.code, {k.lower(): v for k, v in exc.headers.items()}, b"")
            else:
                raise
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == tries:
                raise
            print(f"  {exc}; retrying", flush=True)
            time.sleep(5 * attempt)
            continue
        time.sleep(pause)
        return result
    raise RuntimeError("unreachable")


def api_json(endpoint: str, path: str, pause: float):
    """A JSON answer of the model API; the repository listing is paged through its Link header."""
    url, out = f"{endpoint}/api/models/{path}", []
    while url:
        status, headers, body = request(url, pause=pause)
        data = json.loads(body)
        if not isinstance(data, list):
            return data
        out += data
        match = re.search(r'<([^>]+)>;\s*rel="next"', headers.get("link", ""))
        url = match.group(1) if match else None
    return out


def build_rows(key: str, args, save: Path | None) -> tuple[list[tuple], str, dict]:
    info = registry.get(key)
    api = lambda path: api_json(args.endpoint, f"{info.hf_id}{path}", args.pause)  # noqa: E731
    meta = api("")
    if meta.get("private") or meta.get("gated"):
        raise SystemExit(f"{key}: {info.hf_id} is private or gated; mpd only pins public repositories")
    tags = meta.get("tags") or []
    licence = (meta.get("cardData") or {}).get("license") or next(
        (t[len("license:"):] for t in tags if t.startswith("license:")), None)
    if licence != info.license.lower():
        raise SystemExit(f"{key}: the model card says license {licence!r}, registry.py says {info.license!r}")
    if args.at == "pinned":
        if not info.revision:
            raise SystemExit(f"{key} has no revision in registry.py; use --at main")
        revision = api(f"/revision/{info.revision}")["sha"]
        if revision != info.revision:
            raise SystemExit(f"{key}: asked for {info.revision}, the Hub answered {revision}")
    else:
        revision = api("/revision/main")["sha"]
    listing = api(f"/tree/{revision}?recursive=1")
    entries = index_listing(listing)
    names = select_files(entries)
    print(f"{key}: {info.hf_id} @ {revision}  ({len(names)} of {len(entries)} files)", flush=True)
    if save:
        record = {"info": meta, "revision": revision, "tree": listing}
        (save / f"{key}.json").write_text(json.dumps(record, indent=1) + "\n")
    rows, texts = [], {}
    for name in names:
        entry = entries[name]
        if entry.get("lfs"):
            if entry["lfs"]["size"] != entry["size"]:
                raise SystemExit(f"{key}/{name}: LFS size {entry['lfs']['size']} differs from {entry['size']}")
            digest = entry["lfs"]["oid"]
            if args.crosscheck:
                check_large_file(args, info.hf_id, revision, name, entry["size"], digest)
        else:
            url = f"{args.endpoint}/{info.hf_id}/resolve/{revision}/{urllib.parse.quote(name)}"
            _, headers, body = request(url, pause=args.pause)
            if git_blob_oid(body) != entry["oid"] or len(body) != entry["size"]:
                raise SystemExit(f"{key}/{name}: the download does not match the listing's git blob {entry['oid']}")
            if headers.get("x-repo-commit", revision) != revision:
                raise SystemExit(f"{key}/{name}: served from commit {headers['x-repo-commit']}, not {revision}")
            digest, texts[name] = hashlib.sha256(body).hexdigest(), body
        rows.append((key, name, entry["size"], digest))
    for index, pattern in SHARDS.items():
        if index in texts:
            named = set(json.loads(texts[index])["weight_map"].values())
            listed = {n for n in names if pattern.fullmatch(n)}
            if named != listed:
                raise SystemExit(f"{key}: {index} names {sorted(named)}, but the pinned shards are {sorted(listed)}")
    return rows, revision, meta


def check_large_file(args, hf_id: str, revision: str, name: str, size: int, digest: str) -> None:
    """Second source for a Git LFS file: the resolve endpoint's own headers (HEAD, redirect not followed)."""
    url = f"{args.endpoint}/{hf_id}/resolve/{revision}/{urllib.parse.quote(name)}"
    _, headers, _ = request(url, method="HEAD", follow=False, pause=args.pause)
    got = (headers.get("x-linked-etag", "").strip('"'), headers.get("x-linked-size"), headers.get("x-repo-commit"))
    if got != (digest, str(size), revision):
        raise SystemExit(f"{hf_id}/{name}: the listing says ({digest}, {size}, {revision}) "
                         f"but the resolve endpoint says {got}")
    print(f"  {name}: sha256, size and commit confirmed by the resolve endpoint", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--at", choices=["pinned", "main"], default="pinned",
                    help="rebuild at the registry pins (default) or at main")
    ap.add_argument("--models", nargs="+", metavar="KEY",
                    help="only these models (default: every model without an output lock)")
    ap.add_argument("--write", action="store_true", help="overwrite the manifest with the rebuilt one")
    ap.add_argument("--crosscheck", action="store_true",
                    help="HEAD every large file and compare X-Linked-ETag, size and commit")
    ap.add_argument("--manifest", type=Path, default=ROOT / "src" / "mpd" / "data" / models.MANIFEST)
    ap.add_argument("--endpoint", default=models.DEFAULT_ENDPOINT, help=argparse.SUPPRESS)
    ap.add_argument("--pause", type=float, default=1.5, help="seconds between requests")
    ap.add_argument("--save", type=Path, help="folder to keep the raw API answers in (for the record)")
    args = ap.parse_args(argv)
    keys = args.models or [k for k, m in registry.MODELS.items() if not m.output_family_lock]
    locked = [k for k in keys if registry.get(k).output_family_lock]
    if locked:
        raise SystemExit(f"{', '.join(locked)}: outputs are licence-locked; mpd never pins or fetches these")
    if args.save:
        args.save.mkdir(parents=True, exist_ok=True)

    current = models.load_manifest(args.manifest.read_text(encoding="utf-8")) if args.manifest.is_file() else {}
    rows = {k: [(k, f.name, f.size, f.sha256) for f in fs] for k, fs in current.items()}
    revisions = {}
    for key in keys:
        rows[key], revisions[key], _ = build_rows(key, args, args.save)
    ordered = [r for k in registry.MODELS if k in rows for r in rows[k]]
    text = render_manifest(ordered)
    old = args.manifest.read_text(encoding="utf-8") if args.manifest.is_file() else ""
    changed = text != old
    if changed:
        sys.stdout.writelines(difflib.unified_diff(old.splitlines(True), text.splitlines(True),
                                                    "models.tsv (repository)", "models.tsv (rebuilt)"))
    else:
        print(f"{args.manifest.name}: unchanged ({len(ordered)} rows)")
    moved = {k: v for k, v in revisions.items() if registry.get(k).revision != v}
    for key, revision in moved.items():
        print(f"registry.py: {key} revision={revision!r}   (now {registry.get(key).revision!r})")
    if not moved:
        print("registry.py: every revision matches")
    if args.write and changed:
        args.manifest.write_text(text, encoding="utf-8")
        print(f"wrote {args.manifest}")
    return 1 if (changed or moved) else 0


if __name__ == "__main__":
    sys.exit(main())
