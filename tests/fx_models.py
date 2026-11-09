"""Fixtures for the model-weights tests: a local Hub stand-in with HTTP Range and fault injection, and a miniature pin.

`hub` serves `/<hf_id>/resolve/<revision>/<file>` for the files of a fake `esm2-8m`, and swaps the real manifest for
one that describes those files, so fetches run in milliseconds and never leave the machine.
"""

from __future__ import annotations

import hashlib
import http.server
import os
import random
import socket
import threading
import time
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from mpd import registry
from mpd.data import fetch as fetch_mod, models
from mpd.data.sources import RemoteFile

KEY = "esm2-8m"
_rng = random.Random(7)
FILES = {                                    # name -> bytes; the weights span many 4 KiB chunks
    "config.json": b'{"model_type": "esm", "hidden_size": 8}\n',
    "model.safetensors": _rng.randbytes(120_000),
    "special_tokens_map.json": b'{"cls_token": "<cls>"}\n',
    "tokenizer_config.json": b'{"model_max_length": 1024}\n',
    "vocab.txt": b"<cls>\n<pad>\n<eos>\n<unk>\nL\nA\n",
}
WEIGHTS = "model.safetensors"


def make_pin(files: dict[str, bytes] = FILES, key: str = KEY) -> models.Pin:
    info = registry.get(key)
    rows = tuple(RemoteFile(n, len(b), None, hashlib.sha256(b).hexdigest()) for n, b in sorted(files.items()))
    return models.Pin(key, info.hf_id, info.revision, "MIT", rows)


def write_files(folder: Path, files: dict[str, bytes] = FILES) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (folder / name).write_bytes(data)
    return folder


def hf_cache_snapshot(root: Path, pin: models.Pin, files: dict[str, bytes] = FILES, symlinks: bool = True) -> Path:
    """A Hugging Face cache entry: blobs/ plus snapshots/<commit>/ whose entries are symlinks into blobs/."""
    repo = root / ("models--" + pin.hf_id.replace("/", "--"))
    snap = repo / "snapshots" / pin.revision
    (repo / "blobs").mkdir(parents=True)
    snap.mkdir(parents=True)
    for name, data in files.items():
        (repo / "blobs" / f"blob_{name}").write_bytes(data)
        if symlinks:
            (snap / name).symlink_to(f"../../blobs/blob_{name}")
    return snap


class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        hub: Hub = self.server.hub
        prefix = f"/{hub.pin.hf_id}/resolve/"
        if not self.path.startswith(prefix):
            return self.send_error(404)
        revision, _, quoted = self.path[len(prefix):].partition("/")
        name = urllib.parse.unquote(quoted)
        hub.headers.append(dict(self.headers))
        served = {"main"} if hub.pinned_commit_gone else {hub.pin.revision, "main"}
        if name not in hub.files or revision not in served:
            hub.log.append((name, "404"))
            return self.send_error(404)
        if hub.fail:
            code = hub.fail.pop(0)
            hub.log.append((name, str(code)))
            self.send_response(code)
            self.send_header("Retry-After", hub.retry_after)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        data = hub.files[name]
        if name in hub.corrupt:
            data = bytes([data[0] ^ 1]) + data[1:]
        start, status = 0, 200
        rng = self.headers.get("Range")
        if rng and not hub.ignore_range:
            start = int(rng.split("=")[1].split("-")[0])
            if start >= len(data):
                hub.log.append((name, "416"))
                return self.send_error(416)
            status = 206
        body = data[start:]
        self.send_response(status)
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{len(data) - 1}/{len(data)}")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        hub.log.append((name, f"{status} from {start}"))
        cut = hub.drop_after
        if cut is not None and name == hub.drop_name and len(body) > cut:
            hub.drop_after = None                    # cut the connection once
            self.wfile.write(body[:cut])
            self.wfile.flush()
            self.connection.shutdown(socket.SHUT_RDWR)
            return
        self.wfile.write(body)


@dataclass
class Hub:
    pin: models.Pin
    files: dict[str, bytes]
    url: str = ""
    log: list = field(default_factory=list)      # (file, outcome) for every request, in order
    headers: list = field(default_factory=list)  # the request headers received, in order
    sleeps: list = field(default_factory=list)   # seconds the code under test asked to wait
    drop_after: int | None = None                # cut the connection after this many body bytes, once
    drop_name: str = WEIGHTS
    ignore_range: bool = False                   # answer every Range request with the whole file
    fail: list = field(default_factory=list)     # status codes to answer with before serving normally
    retry_after: str = "3"
    corrupt: set = field(default_factory=set)    # files whose first byte is flipped in transit
    pinned_commit_gone: bool = False             # the pinned commit answers 404; only "main" serves

    def requests(self, name: str | None = None) -> list:
        return [e for e in self.log if name is None or e[0] == name]

    @property
    def waits(self) -> list:
        return [s for s in self.sleeps if s]


@pytest.fixture
def no_model_env(monkeypatch):
    """Nothing from the developer's shell may steer where models are looked for."""
    for name in list(os.environ):
        if name.startswith("MPD_MODEL") or name in ("MPD_HOME", "MPD_CLUSTER", "MPD_ALLOW_HUB", "HF_ENDPOINT"):
            monkeypatch.delenv(name)


@pytest.fixture
def hub(monkeypatch, no_model_env):
    pin = make_pin()
    server_state = Hub(pin, dict(FILES))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    httpd.daemon_threads = True
    httpd.hub = server_state
    server_state.url = f"http://127.0.0.1:{httpd.server_address[1]}"
    threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
    monkeypatch.setattr(models, "pins", lambda: {KEY: pin})
    monkeypatch.setattr(models, "PAUSE_S", 0)
    monkeypatch.setattr(fetch_mod, "CHUNK", 4096)
    monkeypatch.setattr(time, "sleep", server_state.sleeps.append)
    yield server_state
    httpd.shutdown()
    httpd.server_close()
