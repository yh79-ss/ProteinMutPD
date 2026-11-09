import errno
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request
from collections import namedtuple
from pathlib import Path

import pytest

from fx_models import FILES, KEY, WEIGHTS, hf_cache_snapshot, hub, make_pin, no_model_env, write_files  # noqa: F401
from mpd import registry
from mpd.data import fetch as fetch_mod, models
from mpd.data.models import ModelError, ModelNotReady
from mpd.spec import RunSpec

MANIFEST = Path(models.__file__).parent / "models.tsv"
DOC = Path(__file__).resolve().parents[1] / "docs" / "MODELS.md"
SEVEN = ["esm2-8m", "esm2-35m", "esm2-150m", "esm2-650m", "esm2-3b", "saprot-35m", "saprot-650m"]
quiet = lambda _msg: None  # noqa: E731


def run_spec(**kw):
    base = dict(teacher="esm2-3b", student="esm2-650m", train_data="proteingym", method="score-all",
                eval_data=("proteingym",))
    return RunSpec(**{**base, **kw})


def fetch(hub, home, **kw):
    kw.setdefault("log", quiet)
    return models.fetch_model(KEY, home, endpoint=hub.url, **kw)


def dest_of(home):
    return Path(home) / "models" / KEY


# -- the shipped manifest and the registry pins --------------------------------------------------------------------

def test_manifest_has_the_documented_shape():
    text = MANIFEST.read_text()
    assert text.endswith("\n") and "\r" not in text and text.isascii()
    rows = models.load_manifest(text)
    assert list(rows) == SEVEN
    assert sum(len(v) for v in rows.values()) == 37
    for key, files in rows.items():
        names = [f.name for f in files]
        assert names == sorted(names)
        assert {"config.json", "special_tokens_map.json", "tokenizer_config.json", "vocab.txt"} <= set(names), key
        weights = [n for n in names if n.endswith((".safetensors", ".bin"))]
        assert weights, key
        if "pytorch_model.bin.index.json" in names:           # sharded: the shards listed are all the index names
            assert len(weights) > 1 and all(re.fullmatch(r"pytorch_model-\d{5}-of-\d{5}\.bin", n) for n in weights)


def test_documented_sizes():
    """docs/MODELS.md quotes these totals; change both when a pin moves."""
    pins = models.pins()
    assert sum(p.total_bytes for p in pins.values()) == 17_482_684_719
    assert sum(pins[k].total_bytes for k in SEVEN[:5]) == 14_739_300_555
    assert pins["esm2-3b"].total_bytes + pins["esm2-650m"].total_bytes == 13_976_646_488
    assert {k: p.weights_format for k, p in pins.items()} == {
        "esm2-8m": "safetensors", "esm2-35m": "safetensors", "esm2-150m": "safetensors", "esm2-650m": "safetensors",
        "esm2-3b": "pickle", "saprot-35m": "pickle", "saprot-650m": "pickle"}


@pytest.mark.parametrize("edit,message", [
    (lambda t: t.replace("model_key\tfile", "key\tfile", 1), "header"),
    (lambda t: t.replace("config.json", "../config.json", 1), "plain file name"),
    (lambda t: t.replace("vocab.txt", ".vocab.txt", 1), "plain file name"),
    (lambda t: t.replace("vocab.txt", "sub/vocab.txt", 1), "plain file name"),
    (lambda t: t.replace("\t775\t", "\t0\t", 1), "positive integer"),
    (lambda t: t.replace("\t775\t", "\t-5\t", 1), "positive integer"),
    (lambda t: t.replace("facb9355e825", "FACB9355E825", 1), "64 lowercase hex"),
    (lambda t: t.replace("\t775\t", "\t775\textra\t", 1), "4 tab-separated"),
    (lambda t: t + t.splitlines()[1] + "\n", "listed twice"),
])
def test_manifest_loader_rejects_irregular_files(edit, message):
    with pytest.raises(ModelError, match=message):
        models.load_manifest(edit(MANIFEST.read_text()))


def test_every_pinned_model_is_in_the_registry_with_a_40_hex_commit():
    for key, pin in models.pins().items():
        info = registry.get(key)
        assert re.fullmatch(r"[0-9a-f]{40}", info.revision), key
        assert (pin.hf_id, pin.revision, pin.license) == (info.hf_id, info.revision, "MIT"), key


def test_every_available_model_can_be_fetched():
    available = [k for k, m in registry.MODELS.items() if m.status == "available"]
    assert available and set(available) <= set(models.pinned_keys())
    assert models.fetchable_keys() == [k for k in models.pinned_keys() if registry.get(k).status == "available"]
    assert set(SEVEN) <= set(models.fetchable_keys())              # all seven, since SaProt became available


def test_nothing_with_an_output_licence_lock_is_pinned():
    for key in registry.MODELS:
        if registry.get(key).output_family_lock:
            assert key not in models.pins() and registry.get(key).revision is None
    assert "aido-rag-16b" not in models.fetchable_keys()


def test_esm2_activation_estimates_follow_the_meta_device_accounting():
    got = {k: registry.get(k).act_gb for k in SEVEN[:5]}
    assert got == {"esm2-8m": 2, "esm2-35m": 3, "esm2-150m": 9, "esm2-650m": 12, "esm2-3b": 27}
    assert not any(m.act_gb_measured for m in registry.MODELS.values())


def test_the_models_page_agrees_with_the_code():
    text = DOC.read_text(encoding="utf-8")
    for key, pin in models.pins().items():
        assert f"`{key}`" in text and pin.hf_id in text and pin.revision in text and pin.revision[:8] in text, key
        assert models.human_bytes(pin.total_bytes) in text, key
    assert all(f"`{state}`" in text for state in models.STATES)
    assert {"17.48 GB", "14.74 GB", "13.98 GB"} <= set(re.findall(r"\d+\.\d+ GB", text))


def test_every_mpd_command_in_the_models_page_parses():
    from mpd import cli
    if not hasattr(cli, "_parser"):
        pytest.skip("the cli no longer exposes its parser")
    blocks = re.findall(r"```bash\n(.*?)```", DOC.read_text(encoding="utf-8"), re.S)
    commands = [ln.split("#")[0].strip() for block in blocks for ln in block.splitlines() if ln.startswith("mpd ")]
    assert len(commands) >= 8
    for line in commands:
        cli._parser().parse_args(shlex.split(line)[1:])              # exits on a flag or choice that does not exist


def test_models_module_imports_no_torch():
    src = str(Path(__file__).parents[1] / "src")
    code = (f"import sys; sys.path.insert(0, {src!r}); import mpd.data.models, mpd.spec; "
            "assert not {'torch', 'transformers', 'peft'} & set(sys.modules), sorted(sys.modules)")
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


# -- layout and state ----------------------------------------------------------------------------------------------

def test_models_root_prefers_the_models_dir_variable(tmp_path, monkeypatch, no_model_env):
    with pytest.raises(ModelError, match="MPD_HOME"):
        models.models_root()
    monkeypatch.setenv("MPD_HOME", str(tmp_path / "h"))
    assert models.models_root() == tmp_path / "h" / "models"
    assert models.model_dir("esm2-3b", tmp_path / "x") == tmp_path / "x" / "models" / "esm2-3b"
    monkeypatch.setenv("MPD_MODELS_DIR", str(tmp_path / "shared"))
    assert models.model_dir("esm2-3b", tmp_path / "x") == tmp_path / "shared" / "esm2-3b"
    monkeypatch.setenv("MPD_MODELS_DIR", "relative/dir")
    assert models.models_root().is_absolute()


def test_state_missing_then_incomplete_then_unverified(tmp_path, hub):
    home = tmp_path / "home"
    s = models.status(KEY, home)
    total = hub.pin.total_bytes
    assert (s.state, s.path, s.size_bytes, s.todo_bytes) == ("missing", dest_of(home), total, total)
    assert set(s.missing) == set(FILES) and not s.ok
    dest = dest_of(home)
    dest.mkdir(parents=True)
    assert models.status(KEY, home).state == "missing"                       # an empty folder is still nothing
    write_files(dest, {n: b for n, b in FILES.items() if n != WEIGHTS})
    s = models.status(KEY, home)
    assert (s.state, s.missing) == ("incomplete", (WEIGHTS,)) and s.todo_bytes == len(FILES[WEIGHTS])
    (dest / (WEIGHTS + ".part")).write_bytes(FILES[WEIGHTS][:30_000])
    s = models.status(KEY, home)
    assert "partial download" in s.detail and s.todo_bytes == len(FILES[WEIGHTS]) - 30_000
    (dest / (WEIGHTS + ".part")).replace(dest / WEIGHTS)
    (dest / WEIGHTS).write_bytes(FILES[WEIGHTS])
    s = models.status(KEY, home)
    assert s.state == "unverified" and s.missing == () and not s.ok          # right sizes, no FETCHED.json


def test_a_wrong_size_is_corrupt_and_a_flipped_byte_needs_verify(tmp_path, hub):
    home = tmp_path / "home"
    fetch(hub, home)
    dest = dest_of(home)
    assert models.status(KEY, home).state == "ready"
    blob = bytearray(FILES[WEIGHTS])
    blob[10] ^= 1
    (dest / WEIGHTS).write_bytes(bytes(blob))
    assert models.status(KEY, home).state == "ready"                        # stat calls only: same size looks fine
    s = models.status(KEY, home, verify=True)
    assert s.state == "corrupt" and WEIGHTS in s.detail and "sha256 mismatch" in s.detail
    (dest / WEIGHTS).write_bytes(FILES[WEIGHTS][:-1])
    s = models.status(KEY, home)
    assert s.state == "corrupt" and f"size {len(FILES[WEIGHTS]) - 1} != expected" in s.detail


def test_stale_marker_when_the_pin_moved(tmp_path, hub):
    home = tmp_path / "home"
    fetch(hub, home)
    marker = dest_of(home) / "FETCHED.json"
    record = json.loads(marker.read_text())
    marker.write_text(json.dumps({**record, "hf_revision": "0" * 40}))
    s = models.status(KEY, home)
    assert s.state == "stale" and "000000" in s.detail and hub.pin.revision[:8] in s.detail
    marker.write_text(json.dumps({**record, "files": {**record["files"], WEIGHTS: "f" * 64}}))
    assert models.status(KEY, home).state == "stale"
    marker.write_text("{not json")
    assert models.status(KEY, home).state == "unverified"


def test_status_never_hashes_unless_asked(tmp_path, hub, monkeypatch):
    home, hashed = tmp_path / "home", []
    fetch(hub, home)
    monkeypatch.setattr(fetch_mod, "_digest", lambda *a: hashed.append(a) or "0" * 64)
    assert models.status(KEY, home).state == "ready"
    fetch(hub, home)                                                          # ready: nothing to fetch, nothing to hash
    assert hashed == []
    assert models.status(KEY, home, verify=True).state == "corrupt" and len(hashed) == len(FILES)


def test_override_variable_wins_and_is_not_checked(tmp_path, hub, monkeypatch):
    mine = write_files(tmp_path / "mine", {"config.json": b"{}"})
    monkeypatch.setenv("MPD_MODEL_ESM2_8M", str(mine))
    s = models.status(KEY, tmp_path / "home")
    assert (s.state, s.path, s.ok) == ("override", mine, True) and "MPD_MODEL_ESM2_8M" in s.detail
    assert models.resolve(KEY, tmp_path / "home") == str(mine)
    (mine / "config.json").unlink()
    s = models.status(KEY, tmp_path / "home")
    assert s.state == "missing" and "no config.json" in s.detail
    with pytest.raises(ModelNotReady, match=r"MPD_MODEL_ESM2_8M=.* has no config.json. Fix the path or unset"):
        models.require_ready([KEY], tmp_path / "home", cluster="grace")


def test_an_unreadable_folder_is_a_model_error_not_a_traceback(tmp_path, hub):
    if os.geteuid() == 0:
        pytest.skip("root can read anything")
    home = tmp_path / "home"
    write_files(dest_of(home))
    dest_of(home).chmod(0)
    try:
        with pytest.raises(ModelError, match=r"cannot read .*Permission denied"):
            models.status(KEY, home)
    finally:
        dest_of(home).chmod(0o755)


def test_unknown_and_unpinned_models(tmp_path, no_model_env):
    with pytest.raises(ModelError, match="unknown model 'esm9'"):
        models.status("esm9", tmp_path)
    s = models.status("aido-rag-16b", tmp_path)
    assert s.state == "unpinned" and s.path is None and not s.ok and "docs/LICENSES.md" in s.detail
    with pytest.raises(ModelNotReady, match="mpd cannot fetch weights for aido-rag-16b"):
        models.require_ready(["aido-rag-16b"], tmp_path)


# -- fetching ------------------------------------------------------------------------------------------------------

def test_fetch_downloads_checks_and_marks_the_folder(tmp_path, hub):
    home, logs = tmp_path / "home", []
    s = fetch(hub, home, log=logs.append)
    dest = dest_of(home)
    assert s.state == "ready" and s.path == dest
    assert {p.name: p.read_bytes() for p in dest.iterdir() if p.name != "FETCHED.json"} == FILES
    record = json.loads((dest / "FETCHED.json").read_text())
    assert (record["source"], record["hf_repo"], record["hf_revision"], record["license"]) == (
        KEY, hub.pin.hf_id, hub.pin.revision, "MIT")
    assert record["files"] == {f.name: f.sha256 for f in hub.pin.files} and record["endpoint"] == hub.url
    assert str(tmp_path) not in (dest / "FETCHED.json").read_text()          # nothing machine-specific to leak
    assert not list(dest.glob("*.part")) and not list(dest.glob("*.tmp"))
    assert any(line.startswith(f"{KEY}: ready") for line in logs)
    assert sorted(f for f, _ in hub.log) == sorted(FILES)                     # one request per file
    assert all(outcome == "200 from 0" for _, outcome in hub.log)


def test_requests_are_anonymous_and_identify_the_tool(tmp_path, hub, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_secret_value")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_secret_value")
    fetch(hub, tmp_path / "home")
    assert hub.headers and all("authorization" not in {k.lower() for k in h} for h in hub.headers)
    assert not any("hf_secret_value" in v for h in hub.headers for v in h.values())
    assert all(h["User-Agent"].startswith("proteinmutpd/") for h in hub.headers)


def test_a_dropped_connection_is_resumed_with_a_range_request(tmp_path, hub):
    hub.drop_after = 50_000
    fetch(hub, tmp_path / "home")
    assert hub.requests(WEIGHTS) == [(WEIGHTS, "200 from 0"), (WEIGHTS, "206 from 50000")]
    assert models.status(KEY, tmp_path / "home", verify=True).state == "ready"


def test_a_server_that_ignores_range_gets_a_clean_restart(tmp_path, hub):
    dest = dest_of(tmp_path / "home")
    dest.mkdir(parents=True)
    (dest / (WEIGHTS + ".part")).write_bytes(FILES[WEIGHTS][:5_000])
    hub.ignore_range = True
    fetch(hub, tmp_path / "home")
    assert hub.requests(WEIGHTS) == [(WEIGHTS, "200 from 0")]
    assert (dest / WEIGHTS).read_bytes() == FILES[WEIGHTS]


def test_rate_limiting_is_waited_out(tmp_path, hub):
    hub.fail, hub.retry_after = [429], "3"
    fetch(hub, tmp_path / "home")
    assert hub.waits == [3.0] and models.status(KEY, tmp_path / "home").state == "ready"
    hub.fail = [503, 500]
    fetch(hub, tmp_path / "home2")
    assert hub.waits[1:] == [3.0, 4]                       # Retry-After first, then the plain back-off


def test_retry_after_may_be_seconds_a_date_or_missing():
    far = {"Retry-After": "Fri, 31 Dec 2099 23:59:59 GMT"}
    assert fetch_mod._retry_after({"Retry-After": "7"}, 20) == 7
    assert fetch_mod._retry_after(far, 20) == 180                     # capped
    assert fetch_mod._retry_after({"Retry-After": "soon"}, 20) == 20
    assert fetch_mod._retry_after({"Retry-After": "0"}, 20) == 20
    assert fetch_mod._retry_after({}, 20) == 20 and fetch_mod._retry_after(None, 20) == 20


def test_bytes_that_differ_from_the_pin_are_refused_once_retried_and_removed(tmp_path, hub):
    hub.corrupt = {WEIGHTS}
    with pytest.raises(ModelError, match=rf"{KEY}/{re.escape(WEIGHTS)}: sha256 mismatch; removed the file"):
        fetch(hub, tmp_path / "home")
    dest = dest_of(tmp_path / "home")
    assert not (dest / WEIGHTS).exists() and not (dest / (WEIGHTS + ".part")).exists()
    assert hub.requests(WEIGHTS) == [(WEIGHTS, "200 from 0")] * 2          # the retry starts from byte 0
    assert not (dest / "FETCHED.json").exists()
    assert models.status(KEY, tmp_path / "home").state == "incomplete"


def test_a_truncated_file_resumes_from_its_good_prefix(tmp_path, hub):
    home = tmp_path / "home"
    fetch(hub, home)
    hub.log.clear()
    with open(dest_of(home) / WEIGHTS, "r+b") as fh:
        fh.truncate(40_000)
    st = models.status(KEY, home)
    assert st.state == "corrupt" and st.todo_bytes == len(FILES[WEIGHTS]) - 40_000      # only the missing tail counts
    logs = []
    s = fetch(hub, home, log=logs.append)
    assert hub.log == [(WEIGHTS, "206 from 40000")] and s.state == "ready"
    assert any(f"({models.human_bytes(len(FILES[WEIGHTS]) - 40_000)} to get)" in line for line in logs)
    assert any("resuming at 40.0 kB" in line for line in logs)
    assert (dest_of(home) / WEIGHTS).read_bytes() == FILES[WEIGHTS]


def test_verify_repairs_only_the_damaged_file(tmp_path, hub):
    home = tmp_path / "home"
    fetch(hub, home)
    hub.log.clear()
    vocab = dest_of(home) / "vocab.txt"
    vocab.write_bytes(bytes([FILES["vocab.txt"][0] ^ 1]) + FILES["vocab.txt"][1:])
    fetch(hub, home)
    assert hub.log == []                                                # without verify the damage goes unnoticed
    fetch(hub, home, verify=True)
    assert hub.log == [("vocab.txt", "200 from 0")] and vocab.read_bytes() == FILES["vocab.txt"]


def test_verify_says_it_hashed_everything(tmp_path, hub):
    logs = []
    fetch(hub, tmp_path / "home")
    fetch(hub, tmp_path / "home", verify=True, log=logs.append)
    assert hub.requests() and len(hub.requests()) == len(FILES)                # no new request
    assert logs[0] == f"{KEY}: all {len(FILES)} files re-hashed; they match the pin"


def test_a_second_run_touches_neither_network_nor_disk_contents(tmp_path, hub, monkeypatch):
    home = tmp_path / "home"
    fetch(hub, home)
    before = len(hub.log)
    monkeypatch.setattr(fetch_mod, "_digest", lambda *a: pytest.fail("a ready folder must not be re-hashed"))
    s = models.fetch_model(KEY, home, endpoint="http://127.0.0.1:9", log=quiet)    # nothing listens there
    assert s.state == "ready" and len(hub.log) == before


def test_hand_copied_files_are_verified_offline_and_marked(tmp_path, hub):
    home = tmp_path / "home"
    write_files(dest_of(home))
    assert models.status(KEY, home).state == "unverified"
    s = models.fetch_model(KEY, home, endpoint="http://127.0.0.1:9", log=quiet)
    assert s.state == "ready" and hub.log == []
    assert (dest_of(home) / "FETCHED.json").is_file()


def test_a_hand_copied_file_with_the_right_size_but_wrong_bytes_is_replaced(tmp_path, hub):
    home = tmp_path / "home"
    write_files(dest_of(home))
    (dest_of(home) / "vocab.txt").write_bytes(b"x" * len(FILES["vocab.txt"]))
    assert models.status(KEY, home).state == "unverified"
    fetch(hub, home)
    assert hub.log == [("vocab.txt", "200 from 0")] and models.status(KEY, home).state == "ready"


def test_a_moved_pin_refetches_what_changed_and_remarks_the_folder(tmp_path, hub):
    home = tmp_path / "home"
    fetch(hub, home)
    marker = dest_of(home) / "FETCHED.json"
    marker.write_text(json.dumps({**json.loads(marker.read_text()), "hf_revision": "1" * 40}))
    hub.log.clear()
    s = fetch(hub, home)
    assert s.state == "ready" and hub.log == []                         # same bytes: only the marker is rewritten
    assert json.loads(marker.read_text())["hf_revision"] == hub.pin.revision


def test_force_downloads_everything_again(tmp_path, hub):
    home = tmp_path / "home"
    fetch(hub, home)
    hub.log.clear()
    fetch(hub, home, force=True)
    assert sorted(f for f, _ in hub.log) == sorted(FILES) and {o for _, o in hub.log} == {"200 from 0"}


def test_dry_run_reports_and_writes_nothing(tmp_path, hub):
    logs = []
    s = fetch(hub, tmp_path / "home", dry_run=True, log=logs.append)
    assert s.state == "missing" and hub.log == [] and not (tmp_path / "home").exists()
    text = "\n".join(logs)
    assert "dry run" in text and WEIGHTS in text and "would download 5 file(s)" in text


def test_not_enough_disk_is_refused_before_anything_is_written(tmp_path, hub, monkeypatch):
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda p: usage(10**9, 10**9 - 10_000, 10_000))
    with pytest.raises(ModelError, match=r"not enough free space under .*need .*showquota"):
        fetch(hub, tmp_path / "home")
    assert hub.log == [] and not dest_of(tmp_path / "home").exists()


def test_check_space_counts_the_margin(tmp_path):
    usage = namedtuple("usage", "total used free")
    models.check_space(tmp_path / "a" / "b", 1_000, usage=lambda p: usage(0, 0, 3 * models.GIB))
    with pytest.raises(ModelError, match="not enough free space"):
        models.check_space(tmp_path, 2 * models.GIB + 1, usage=lambda p: usage(0, 0, 3 * models.GIB))


def test_the_pinned_commit_disappearing_falls_back_to_main_but_still_checks_the_bytes(tmp_path, hub):
    hub.pinned_commit_gone = True
    fetch(hub, tmp_path / "home")
    assert models.status(KEY, tmp_path / "home", verify=True).state == "ready"
    assert hub.requests(WEIGHTS) == [(WEIGHTS, "404"), (WEIGHTS, "200 from 0")]
    hub.corrupt = {WEIGHTS}
    shutil.rmtree(tmp_path / "home")
    with pytest.raises(ModelError, match="sha256 mismatch"):
        fetch(hub, tmp_path / "home")


def test_no_route_to_the_hub_points_to_the_offline_route(tmp_path, hub):
    with pytest.raises(ModelError) as err:
        models.fetch_model(KEY, tmp_path / "home", endpoint="http://127.0.0.1:9", log=quiet)
    assert "--from-dir" in str(err.value) and "docs/MODELS.md" in str(err.value) and "huggingface.co" in str(err.value)
    hub.fail = [404] * 10                                                 # an HTTP answer is not a connection problem
    with pytest.raises(ModelError) as err:
        fetch(hub, tmp_path / "home2")
    assert "HTTP 404" in str(err.value) and "--from-dir" not in str(err.value)
    hub.fail, hub.sleeps[:] = [429] * 10, []                              # a limit that never lifts: wait, then say so
    with pytest.raises(ModelError) as err:
        fetch(hub, tmp_path / "home3")
    assert "HTTP Error 429" in str(err.value) and "rate-limiting" in str(err.value) and "--from-dir" in str(err.value)
    assert hub.waits == [3.0] * 5 and not (tmp_path / "home3" / "models" / KEY / "FETCHED.json").exists()


def test_endpoint_comes_from_hf_endpoint_when_not_given(tmp_path, hub, monkeypatch):
    monkeypatch.setenv("HF_ENDPOINT", hub.url + "/")
    models.fetch_model(KEY, tmp_path / "home", log=quiet)
    assert models.status(KEY, tmp_path / "home").state == "ready"


def test_a_full_disk_while_writing_says_so(tmp_path, hub, monkeypatch):
    def full(*args, **kwargs):
        raise OSError(errno.ENOSPC, "No space left on device")
    monkeypatch.setattr(fetch_mod, "_copy", full)
    with pytest.raises(ModelError, match=r"disk full or quota exceeded while writing .*showquota"):
        fetch(hub, tmp_path / "home")


def test_deadline_stops_cleanly_and_the_next_run_resumes(tmp_path, hub, monkeypatch):
    home = tmp_path / "home"
    with pytest.raises(models.TimeBudgetReached, match="run the same command again to resume"):
        fetch(hub, home, deadline=time.monotonic() - 1)                   # already over: not one request
    assert hub.log == []
    ticks = iter(range(10_000))
    monkeypatch.setattr(time, "monotonic", lambda: next(ticks))
    with pytest.raises(models.TimeBudgetReached, match=r"\d file\(s\) left"):
        fetch(hub, home, deadline=12.5)                                   # runs out in the middle of the weights
    monkeypatch.undo()
    part = dest_of(home) / (WEIGHTS + ".part")
    assert 0 < part.stat().st_size < len(FILES[WEIGHTS]) and not (dest_of(home) / WEIGHTS).exists()
    kept = part.stat().st_size
    hub.log.clear()
    monkeypatch.setattr(models, "pins", lambda: {KEY: hub.pin})
    monkeypatch.setattr(models, "PAUSE_S", 0)
    monkeypatch.setattr(fetch_mod, "CHUNK", 4096)
    s = fetch(hub, home)
    assert s.state == "ready" and hub.requests(WEIGHTS)[0] == (WEIGHTS, f"206 from {kept}")


def test_two_fetches_of_one_model_do_not_run_at_once(tmp_path, hub):
    fcntl = pytest.importorskip("fcntl")
    home = tmp_path / "home"
    (home / "models").mkdir(parents=True)
    with open(home / "models" / f".{KEY}.lock", "a") as holder:
        fcntl.flock(holder, fcntl.LOCK_EX)
        with pytest.raises(ModelError, match="already writing"):
            fetch(hub, home)
    assert hub.log == []
    fetch(hub, home)                                                      # the lock is released with the holder
    assert models.status(KEY, home).state == "ready"


def test_progress_is_reported_once_per_gib_of_a_big_file():
    lines, gib = [], models.GIB
    report = models._progress("big.bin", 3 * gib, lines.append)
    for done in (gib // 2, gib + 1, gib + 2, 2 * gib + 5, 3 * gib):
        report(done)
    assert [ln.split(":")[1].strip() for ln in lines] == ["1 of 3.0 GiB", "2 of 3.0 GiB", "3 of 3.0 GiB"]
    assert models._progress("small.bin", gib - 1, lines.append) is None


def test_pickle_weights_are_announced(tmp_path, hub, monkeypatch):
    logs = []
    pickled = {**FILES, "pytorch_model.bin": b"x" * 100}
    del pickled[WEIGHTS]
    pin = make_pin(pickled)
    monkeypatch.setattr(models, "pins", lambda: {KEY: pin})
    hub.files, hub.pin = pickled, pin
    fetch(hub, tmp_path / "home", log=logs.append)
    assert any("PyTorch pickle weights only; loading needs torch >= 2.6" in ln for ln in logs)


def test_models_unavailable_to_the_pipeline_are_fetched_with_a_note(tmp_path, hub, monkeypatch):
    planned = registry.ModelInfo(**{**registry.get(KEY).__dict__, "status": "planned"})
    monkeypatch.setitem(registry.MODELS, KEY, planned)
    logs = []
    fetch(hub, tmp_path / "home", log=logs.append)
    assert any("planned in this release" in ln for ln in logs)


# -- --from-dir ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("layout", ["key", "hub-name", "cache-root", "cache-hub-folder", "itself"])
def test_from_dir_accepts_every_documented_layout(tmp_path, hub, layout):
    src = tmp_path / "src"
    if layout == "key":
        write_files(src / KEY)
    elif layout == "hub-name":
        write_files(src / hub.pin.hf_id.split("/")[-1])
    elif layout == "cache-root":
        hf_cache_snapshot(src, hub.pin)
    elif layout == "cache-hub-folder":
        hf_cache_snapshot(src / "hub", hub.pin)
    else:
        write_files(src)
    logs = []
    s = models.fetch_model(KEY, tmp_path / "home", from_dir=src, endpoint="http://127.0.0.1:9", log=logs.append)
    assert s.state == "ready" and hub.log == []
    dest = dest_of(tmp_path / "home")
    assert not any(p.is_symlink() for p in dest.iterdir())                  # real files, even from a symlinked cache
    assert {p.name: p.read_bytes() for p in dest.iterdir() if p.name == WEIGHTS} == {WEIGHTS: FILES[WEIGHTS]}
    assert any("copying from" in ln for ln in logs)
    record = json.loads((dest / "FETCHED.json").read_text())
    assert record["endpoint"] is None and str(tmp_path) not in (dest / "FETCHED.json").read_text()


def test_from_dir_takes_a_snapshot_under_any_commit_name(tmp_path, hub):
    snap = hf_cache_snapshot(tmp_path / "cache", hub.pin)
    snap.rename(snap.with_name("someothercommit"))
    models.fetch_model(KEY, tmp_path / "home", from_dir=tmp_path / "cache", log=quiet)
    assert models.status(KEY, tmp_path / "home").state == "ready"


def test_from_dir_prefers_the_folder_that_has_every_file(tmp_path, hub):
    write_files(tmp_path / "src" / KEY, {n: b for n, b in FILES.items() if n != "vocab.txt"})
    write_files(tmp_path / "src" / hub.pin.hf_id.split("/")[-1])
    models.fetch_model(KEY, tmp_path / "home", from_dir=tmp_path / "src", log=quiet)
    assert models.status(KEY, tmp_path / "home").state == "ready"


def test_from_dir_only_needs_the_files_that_are_missing(tmp_path, hub):
    home = tmp_path / "home"
    write_files(dest_of(home), {n: b for n, b in FILES.items() if n != WEIGHTS})
    write_files(tmp_path / "src" / KEY, {WEIGHTS: FILES[WEIGHTS]})
    logs = []
    s = models.fetch_model(KEY, home, from_dir=tmp_path / "src", log=logs.append)
    assert s.state == "ready" and (dest_of(home) / WEIGHTS).read_bytes() == FILES[WEIGHTS]
    assert sum("copying (" in ln for ln in logs) == 1                         # one file copied, four left alone


def test_from_dir_errors_say_what_is_wrong(tmp_path, hub):
    empty = tmp_path / "empty" / ("models--" + hub.pin.hf_id.replace("/", "--")) / "snapshots" / hub.pin.revision
    empty.mkdir(parents=True)                                              # a cache whose symlinks were lost
    with pytest.raises(ModelError, match=r"no folder with its files under .*looked for"):
        models.fetch_model(KEY, tmp_path / "h1", from_dir=tmp_path / "empty", log=quiet)
    snap = hf_cache_snapshot(tmp_path / "cache", hub.pin)
    shutil.rmtree(snap.parents[1] / "blobs")                               # dangling symlinks
    with pytest.raises(ModelError, match="broken symlinks"):
        models.fetch_model(KEY, tmp_path / "h2", from_dir=tmp_path / "cache", log=quiet)
    write_files(tmp_path / "src" / KEY, {n: b for n, b in FILES.items() if n != "vocab.txt"})
    with pytest.raises(ModelError, match="lacks vocab.txt"):
        models.fetch_model(KEY, tmp_path / "h3", from_dir=tmp_path / "src", log=quiet)
    assert not dest_of(tmp_path / "h3").exists()
    with pytest.raises(ModelError, match="is not a folder"):
        models.fetch_model(KEY, tmp_path / "h4", from_dir=tmp_path / "nowhere", log=quiet)


def test_from_dir_still_checks_every_byte(tmp_path, hub):
    bad = dict(FILES)
    bad[WEIGHTS] = bytes([bad[WEIGHTS][0] ^ 1]) + bad[WEIGHTS][1:]
    write_files(tmp_path / "src" / KEY, bad)
    with pytest.raises(ModelError, match="sha256 mismatch; removed the file"):
        models.fetch_model(KEY, tmp_path / "home", from_dir=tmp_path / "src", log=quiet)
    assert not (dest_of(tmp_path / "home") / WEIGHTS).exists()
    assert not (dest_of(tmp_path / "home") / "FETCHED.json").exists()


def test_from_dir_pointing_at_the_models_folder_itself_is_harmless(tmp_path, hub):
    home = tmp_path / "home"
    write_files(dest_of(home))
    s = models.fetch_model(KEY, home, from_dir=home / "models", log=quiet)    # nothing to copy, only the marker
    assert s.state == "ready"


# -- require_ready, resolve, needed, fix_command, preflight ----------------------------------------------------------

def test_the_canonical_weights_message(tmp_path, no_model_env):
    with pytest.raises(ModelNotReady) as err:
        models.require_ready(["esm2-3b"], tmp_path, cluster="grace")
    assert str(err.value) == (f"weights for esm2-3b are missing ({tmp_path / 'models' / 'esm2-3b'}). "
                              "Run on a login node: mpd fetch model esm2-3b --cluster grace")


def test_the_message_of_a_job_names_the_cluster_and_the_home_it_looked_in(tmp_path, monkeypatch, no_model_env):
    """A job knows its home and cluster from MPD_HOME and MPD_CLUSTER (the script exports both), and calls resolve() without a home."""
    monkeypatch.setenv("MPD_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("MPD_CLUSTER", "grace")
    with pytest.raises(ModelNotReady) as err:
        models.resolve("esm2-8m")
    assert str(err.value) == (f"weights for esm2-8m are missing ({tmp_path / 'home' / 'models' / 'esm2-8m'}). "
                              f"Run on a login node: mpd fetch model esm2-8m --cluster grace --home {tmp_path / 'home'}")
    monkeypatch.setenv("MPD_MODELS_DIR", str(tmp_path / "elsewhere"))              # the folder then does not follow the home
    with pytest.raises(ModelNotReady) as err:
        models.resolve("esm2-8m")
    assert "--home" not in str(err.value) and "--cluster grace" in str(err.value)


def test_several_models_get_one_table_and_one_command(tmp_path, monkeypatch, no_model_env):
    monkeypatch.setenv("MPD_MODEL_ESM2_8M", str(write_files(tmp_path / "mine", {"config.json": b"{}"})))
    with pytest.raises(ModelNotReady) as err:
        models.require_ready(["esm2-3b", "esm2-8m", "esm2-650m", "esm2-3b"], tmp_path / "h", cluster="grace")
    text = str(err.value)
    assert text.startswith("model weights are not ready:") and "esm2-8m" not in text      # the overridden one is fine
    assert text.count("mpd fetch model") == 1
    assert text.splitlines()[-1] == ("Run on a login node (13.98 GB to download, resumable): "
                                     "mpd fetch model esm2-3b esm2-650m --cluster grace")
    assert "esm2-3b" in text.splitlines()[1] and "esm2-650m" in text.splitlines()[2]


def test_the_fix_command_repeats_an_explicit_home_and_quotes_it(tmp_path, no_model_env):
    with pytest.raises(ModelNotReady) as err:
        models.require_ready(["esm2-3b"], tmp_path / "a b", cluster="grace", explicit_home=True)
    assert str(err.value).endswith(f"mpd fetch model esm2-3b --cluster grace --home '{tmp_path / 'a b'}'")
    assert models.fix_command(["esm2-3b", "esm2-650m"]) == "mpd fetch model esm2-3b esm2-650m"
    assert models.fix_command(["esm2-3b"], home="/x") == "mpd fetch model esm2-3b --home /x"


def test_require_ready_passes_for_ready_and_override(tmp_path, hub, monkeypatch):
    fetch(hub, tmp_path / "home")
    models.require_ready([KEY], tmp_path / "home")
    models.require_ready([], tmp_path / "home")
    shutil.rmtree(dest_of(tmp_path / "home"))
    monkeypatch.setenv("MPD_MODEL_ESM2_8M", str(write_files(tmp_path / "mine", {"config.json": b"{}"})))
    models.require_ready([KEY], tmp_path / "home")


def test_resolve_order_is_override_then_ready_folder_then_error(tmp_path, hub, monkeypatch):
    home = tmp_path / "home"
    with pytest.raises(ModelNotReady, match="mpd fetch model esm2-8m"):
        models.resolve(KEY, home)
    write_files(dest_of(home))
    with pytest.raises(ModelNotReady, match="have not been verified"):         # present but never checked: not trusted
        models.resolve(KEY, home)
    fetch(hub, home)
    assert models.resolve(KEY, home) == str(dest_of(home))
    mine = write_files(tmp_path / "mine", {"config.json": b"{}"})
    monkeypatch.setenv("MPD_MODEL_ESM2_8M", str(mine))
    assert models.resolve(KEY, home) == str(mine)


def test_resolve_never_returns_the_hub_id_unless_asked(tmp_path, hub, monkeypatch):
    with pytest.raises(ModelNotReady) as err:
        models.resolve(KEY, tmp_path / "home")
    assert hub.pin.hf_id not in str(err.value)
    monkeypatch.setenv("MPD_ALLOW_HUB", "1")
    assert models.resolve(KEY, tmp_path / "home") == hub.pin.hf_id
    monkeypatch.setenv("MPD_ALLOW_HUB", "true")                               # only the documented value
    with pytest.raises(ModelNotReady):
        models.resolve(KEY, tmp_path / "home")


def test_resolve_without_any_home_is_a_clear_error(no_model_env, monkeypatch):
    with pytest.raises(ModelNotReady, match="cannot locate the models folder"):
        models.resolve("esm2-8m")
    monkeypatch.setenv("MPD_CLUSTER", "grace")
    with pytest.raises(ModelNotReady, match="mpd fetch model esm2-8m --cluster grace"):
        models.resolve("esm2-8m")
    with pytest.raises(ModelError, match="unknown model"):
        models.resolve("esm9")


def test_needed_is_the_teacher_if_it_runs_live_then_the_student(monkeypatch):
    assert models.needed(run_spec()) == ["esm2-3b", "esm2-650m"]
    assert models.needed(run_spec(teacher="esm2-650m")) == ["esm2-650m"]
    cached = registry.ModelInfo(**{**registry.get("esm2-3b").__dict__, "teacher_backends": ("cache", "live")})
    monkeypatch.setitem(registry.MODELS, "esm2-3b", cached)
    assert models.needed(run_spec()) == ["esm2-650m"]                          # a cached-logit teacher needs no weights
    assert models.needed(run_spec(teacher_backend="live")) == ["esm2-3b", "esm2-650m"]


def test_preflight_checks_one_per_model_plus_torch_for_pickle_weights(tmp_path, monkeypatch, no_model_env):
    monkeypatch.setenv("MPD_MODEL_ESM2_650M", str(write_files(tmp_path / "mine", {"config.json": b"{}"})))
    checks = models.preflight_checks(run_spec(), tmp_path / "h", cluster="grace")
    by_name = {c.name: c for c in checks}
    assert by_name["weights esm2-650m"].level == "ok" and "MPD_MODEL_ESM2_650M" in by_name["weights esm2-650m"].message
    c = by_name["weights esm2-3b"]
    assert (c.level, c.fix) == ("error", "mpd fetch model esm2-3b --cluster grace") and "missing" in c.message
    assert by_name["torch for pickle weights"].level in ("ok", "skip")        # esm2-3b ships pickle weights
    for c in checks:
        assert c.level in ("ok", "warn", "error", "skip")


def test_preflight_checks_without_a_home_and_for_ready_models(tmp_path, hub, no_model_env):
    spec = run_spec(teacher=KEY, student=KEY)
    (c,) = models.preflight_checks(spec, None, cluster="grace")
    assert c.level == "error" and "cannot locate" in c.message and c.fix == f"mpd fetch model {KEY} --cluster grace"
    fetch(hub, tmp_path / "home")
    (c,) = models.preflight_checks(spec, tmp_path / "home")
    assert (c.name, c.level) == (f"weights {KEY}", "ok") and c.fix is None        # safetensors: no torch check


def test_old_torch_with_pickle_weights_warns(monkeypatch):
    import importlib.metadata as md
    monkeypatch.setattr(md, "version", lambda name: "2.5.1+cu121")
    c = models._torch_check(["esm2-3b"])
    assert c.level == "warn" and "torch >= 2.7" in c.message
    monkeypatch.setattr(md, "version", lambda name: "2.10.0+cu128")
    assert models._torch_check(["esm2-3b"]).level == "ok"

    def absent(name):
        raise md.PackageNotFoundError(name)
    monkeypatch.setattr(md, "version", absent)
    assert models._torch_check(["esm2-3b"]).level == "skip"


# -- the ESM2 student: where it loads from, what it saves, which dtype it picks ---------------------------------------

@pytest.fixture
def esm2(no_model_env):
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    torch = pytest.importorskip("torch")
    from mpd.students import esm2 as module
    return module, torch


@pytest.fixture
def no_network(monkeypatch):
    """Any attempt to open a connection is recorded and refused, as on a compute node."""
    import socket
    attempts = []

    def refuse(*args, **kwargs):
        attempts.append(args)
        raise OSError("no network in this test")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        monkeypatch.delenv(name, raising=False)             # the point is that no offline switch is needed
    return attempts


def test_model_path_is_the_override_then_the_fetched_folder_and_never_the_hub(tmp_path, hub, esm2, monkeypatch):
    module, _ = esm2
    home = tmp_path / "home"
    monkeypatch.setenv("MPD_HOME", str(home))
    with pytest.raises(ModelNotReady, match="mpd fetch model esm2-8m"):
        module.model_path(KEY)
    fetch(hub, home)
    assert module.model_path(KEY) == str(dest_of(home))
    mine = write_files(tmp_path / "mine", {"config.json": b"{}"})
    monkeypatch.setenv("MPD_MODEL_ESM2_8M", str(mine))
    assert module.model_path(KEY) == str(mine)
    monkeypatch.delenv("MPD_MODEL_ESM2_8M")
    monkeypatch.setenv("MPD_MODELS_DIR", str(tmp_path / "elsewhere"))
    with pytest.raises(ModelNotReady) as err:
        module.model_path(KEY)
    assert hub.pin.hf_id not in str(err.value) and str(tmp_path / "elsewhere" / KEY) in str(err.value)


def tiny_student(tmp_path, esm2, **kwargs):
    from tinydata import make_tiny_esm
    module, torch = esm2
    folder = make_tiny_esm(tmp_path / "tiny", seed=1)
    return folder, module.Esm2(KEY, torch.device("cpu"), torch.float32, **kwargs)


def test_saved_adapters_name_the_hub_model_not_a_local_folder_and_need_no_network(tmp_path, esm2, monkeypatch,
                                                                                  no_network):
    from tinydata import make_tiny_esm
    module, torch = esm2
    monkeypatch.setenv("MPD_MODEL_ESM2_8M", str(make_tiny_esm(tmp_path / "tiny", seed=1)))
    student = module.Esm2(KEY, torch.device("cpu"), torch.float32, lora={"rank": 4, "alpha": 8})
    student.save_adapter(tmp_path / "adapter")
    config = json.loads((tmp_path / "adapter" / "adapter_config.json").read_text())
    assert config["base_model_name_or_path"] == registry.get(KEY).hf_id
    assert config["revision"] is None                          # an override folder is the user's weights, not the pin
    for name in ("adapter_config.json", "README.md"):
        assert str(tmp_path) not in (tmp_path / "adapter" / name).read_text(), name
    reloaded = module.Esm2(KEY, torch.device("cpu"), torch.float32, adapter=tmp_path / "adapter")
    assert reloaded.trainable() == [] and no_network == []


def test_adapters_trained_on_the_pinned_weights_record_the_commit(tmp_path, esm2, monkeypatch, no_network):
    from tinydata import make_tiny_esm
    module, torch = esm2
    tiny = make_tiny_esm(tmp_path / "tiny", seed=1)
    monkeypatch.setattr(models, "resolve", lambda key, home=None: str(tiny))   # as if the fetched folder were this one
    student = module.Esm2(KEY, torch.device("cpu"), torch.float32, lora={"rank": 4, "alpha": 8})
    student.save_adapter(tmp_path / "adapter")
    config = json.loads((tmp_path / "adapter" / "adapter_config.json").read_text())
    info = registry.get(KEY)
    assert (config["base_model_name_or_path"], config["revision"]) == (info.hf_id, info.revision)
    assert str(tmp_path) not in (tmp_path / "adapter" / "README.md").read_text() and no_network == []


def test_the_opt_in_hub_route_asks_for_the_pinned_revision(tmp_path, esm2, monkeypatch):
    import transformers
    module, torch = esm2
    seen = []

    def spy(cls, path, **kwargs):
        seen.append((path, kwargs))
        raise KeyError("stop after the first call")
    monkeypatch.setenv("MPD_ALLOW_HUB", "1")
    monkeypatch.setenv("MPD_HOME", str(tmp_path))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", classmethod(spy))
    with pytest.raises(KeyError):
        module.Esm2(KEY, torch.device("cpu"), torch.float32)
    assert seen == [(registry.get(KEY).hf_id, {"revision": registry.get(KEY).revision})]


def test_default_dtype_is_float32_without_native_bf16_and_says_so_once(esm2, monkeypatch):
    import warnings
    module, torch = esm2
    monkeypatch.delenv("MPD_DTYPE", raising=False)
    monkeypatch.setattr(module, "_warned_bf16", False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert module.default_dtype() is torch.float32                          # CPU: as before
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    asked = []
    monkeypatch.setattr(torch.cuda, "is_bf16_supported",
                        lambda including_emulation=True: asked.append(including_emulation) or False)
    with pytest.warns(RuntimeWarning, match="no native bfloat16"):
        assert module.default_dtype() is torch.float32
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert module.default_dtype() is torch.float32                      # warned once, not on every model load
    assert asked == [False, False]                                          # slow emulation must not count as support
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda including_emulation=True: True)
    assert module.default_dtype() is torch.bfloat16                         # unchanged on a normal GPU
    monkeypatch.setenv("MPD_DTYPE", "float16")
    assert module.default_dtype() is torch.float16                          # the variable still wins
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)     # torch before the argument existed
    monkeypatch.delenv("MPD_DTYPE")
    assert module.default_dtype() is torch.float32


# -- AIDO is never fetched -----------------------------------------------------------------------------------------

def test_fetching_aido_is_refused_with_the_licence_note(tmp_path, no_model_env):
    with pytest.raises(ModelError) as err:
        models.fetch_model("aido-rag-16b", tmp_path, log=quiet)
    text = str(err.value)
    assert "mpd never downloads aido-rag-16b" in text and "clause 1.2.2" in text and "docs/LICENSES.md" in text
    assert not (tmp_path / "models").exists()
    assert models.weights_format("aido-rag-16b") is None and models.pin("aido-rag-16b") is None


# -- the maintainer tool -------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pin_models():
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "scripts" / "maintainers" / "pin_models.py"
    spec = importlib.util.spec_from_file_location("pin_models", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["pin_models"] = module
    spec.loader.exec_module(module)
    return module


def test_the_pin_tool_selects_the_files_loading_needs(pin_models):
    base = {"config.json", "vocab.txt", "special_tokens_map.json", "tokenizer_config.json", ".gitattributes",
            "README.md", "tf_model.h5", "pytorch_model.bin"}
    assert pin_models.select_files(base | {"model.safetensors"}) == [
        "config.json", "model.safetensors", "special_tokens_map.json", "tokenizer_config.json", "vocab.txt"]
    assert "pytorch_model.bin" in pin_models.select_files(base)
    sharded = base - {"pytorch_model.bin"} | {"pytorch_model-00001-of-00002.bin", "pytorch_model-00002-of-00002.bin",
                                               "pytorch_model.bin.index.json", "tf_model-00001-of-00002.h5"}
    assert pin_models.select_files(sharded) == sorted(
        ["config.json", "pytorch_model-00001-of-00002.bin", "pytorch_model-00002-of-00002.bin",
         "pytorch_model.bin.index.json", "special_tokens_map.json", "tokenizer_config.json", "vocab.txt"])
    with pytest.raises(ValueError, match="vocab.txt"):
        pin_models.select_files({"config.json", "model.safetensors", "special_tokens_map.json",
                                 "tokenizer_config.json"})
    with pytest.raises(ValueError, match="weights"):
        pin_models.select_files({"config.json", "vocab.txt", "special_tokens_map.json", "tokenizer_config.json"})


def test_the_pin_tool_renders_the_manifest_byte_for_byte(pin_models):
    rows = [(k, f.name, f.size, f.sha256) for k, p in models.pins().items() for f in p.files]
    assert pin_models.render_manifest(rows) == MANIFEST.read_text()


def test_the_pin_tool_hashes_like_git_and_reads_lfs_listings(pin_models):
    assert pin_models.git_blob_oid(b"") == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"      # git's empty blob
    assert pin_models.git_blob_oid(b"hello\n") == "ce013625030ba8dba906f756967f9e9ca394464a"
    listing = [{"type": "file", "path": "model.safetensors", "size": 5, "oid": "a" * 40,
                "lfs": {"oid": "b" * 64, "size": 5}}, {"type": "directory", "path": "sub"},
               {"type": "file", "path": "config.json", "size": 3, "oid": "c" * 40}]
    entries = pin_models.index_listing(listing)
    assert set(entries) == {"model.safetensors", "config.json"}
    assert entries["model.safetensors"]["lfs"]["oid"] == "b" * 64


# -- the real Hub (opt in: MPD_TEST_NETWORK=1 pytest -m slow) -----------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("MPD_TEST_NETWORK") != "1", reason="set MPD_TEST_NETWORK=1 to fetch esm2-8m (31 MB)")
def test_real_esm2_8m_fetch_and_offline_load(tmp_path, monkeypatch, no_model_env):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    home = tmp_path / "home"
    s = models.fetch_model("esm2-8m", home, log=print)
    assert s.state == "ready" and models.status("esm2-8m", home, verify=True).state == "ready"
    dest = home / "models" / "esm2-8m"
    record = json.loads((dest / "FETCHED.json").read_text())
    assert record["hf_revision"] == "c731040fcd8d73dceaa04b0a8e6329b345b0f5df"
    # the CDN behind huggingface.co must honour Range on the signed redirect: cut the weights at 9 MiB and resume
    with open(dest / "model.safetensors", "r+b") as fh:
        fh.truncate(9 * 2**20)
    (dest / "FETCHED.json").unlink()
    seen, real_urlopen = [], urllib.request.urlopen

    def spy(request, timeout=None):
        resp = real_urlopen(request, timeout=timeout)
        seen.append((request.get_header("Range"), resp.status, resp.headers.get("Content-Range")))
        return resp
    monkeypatch.setattr(urllib.request, "urlopen", spy)
    models.fetch_model("esm2-8m", home, log=print)
    monkeypatch.setattr(urllib.request, "urlopen", real_urlopen)
    assert seen == [("bytes=9437184-", 206, "bytes 9437184-31384291/31384292")], seen
    assert models.status("esm2-8m", home, verify=True).state == "ready"
    for var, value in (("HF_HUB_OFFLINE", "1"), ("TRANSFORMERS_OFFLINE", "1"), ("HF_HOME", str(tmp_path / "empty_hf")),
                       ("MPD_HOME", str(home))):
        monkeypatch.setenv(var, value)
    import socket

    def no_network(*args, **kwargs):
        raise AssertionError("a model load reached for the network")
    monkeypatch.setattr(socket.socket, "connect", no_network)
    from mpd.students.esm2 import Esm2
    student = Esm2("esm2-8m", torch.device("cpu"), torch.float32)
    assert len(student.aa_ids) == 20 and student.tokenizer.mask_token_id == 32
    ids = torch.as_tensor([student.tokenizer.encode("MKTAYIAKQR")])
    assert student.module(input_ids=ids).logits.shape == (1, 12, 33)
