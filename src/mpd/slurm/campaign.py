"""Campaign bookkeeping behind `mpd submit|status|logs|cancel`: the SUBMITTED.json marker, listing and resolving
campaigns, Slurm queries and cancel. Standard library only (no torch, no planner imports), so it runs on a login node.

Every Slurm client is called by an injectable command name and without a terminal, so tests drive fake binaries.
"""

from __future__ import annotations

import getpass
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

MARKER = "SUBMITTED.json"
SCHEMA = 1
# Marker states that stop a second submission of the same campaign. `unreadable` is not written by mpd: a marker we
# cannot parse is treated as "something may be queued".
BLOCKING = ("submitting", "submitted", "partial", "unreadable")
# A job in one of these states can never run again (squeue and sacct spell them alike).
TERMINAL = frozenset({"COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "NODE_FAIL", "BOOT_FAIL", "DEADLINE",
                      "OUT_OF_MEMORY", "REVOKED"})
STAGE_ORDER = ("prepare", "teacher", "train", "score-base", "score", "evaluate")
# scancel on a job that is already gone is not a failure of the cancel.
_BENIGN_CANCEL = re.compile(r"Invalid job id specified|Job/step already completed")
_ACCOUNT_LIKE = re.compile(r"(?<!\d)\d{12,13}(?!\d)")
_EMAIL = re.compile(r"[\w.+-]{1,64}@[\w-]{1,63}(?:\.[\w-]{1,63}){1,5}")        # bounded: a long log line must not make this quadratic
MAX_LINE = 600


class SubmitError(RuntimeError):
    """A submission was refused or failed. The message says what happened and what to do next."""


class CampaignError(SubmitError):
    """The named campaign does not exist, or the name is ambiguous."""


# ---------------------------------------------------------------- running Slurm clients
def run_tool(cmd: list[str], *, timeout: float = 120.0) -> subprocess.CompletedProcess:
    """Run a client with no stdin (a stray sbatch would otherwise wait on the terminal). A missing binary or a hang is
    a failed result, not an exception; 127 and 124 are the shell's codes for those."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, errors="replace", stdin=subprocess.DEVNULL,
                              timeout=timeout)
    except FileNotFoundError:
        return subprocess.CompletedProcess(cmd, 127, "", f"{cmd[0]}: command not found")
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, 124, "", f"{cmd[0]}: no answer after {timeout:g} s")
    except OSError as exc:
        return subprocess.CompletedProcess(cmd, 126, "", f"{cmd[0]}: {exc}")


def message_of(proc: subprocess.CompletedProcess) -> str:
    return (proc.stderr or "").strip() or (proc.stdout or "").strip() or f"{proc.args[0]} exited with code {proc.returncode}"


def current_user() -> str:
    return os.environ.get("USER") or os.environ.get("LOGNAME") or getpass.getuser()


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


# ---------------------------------------------------------------- tags
def job_tag(campaign_id: str, nonce: str, stage: str) -> str:
    """The --comment of a job: which campaign, which submission (nonce) and which stage it belongs to."""
    return f"mpd:{campaign_id}:{nonce}:{stage}"


def parse_tag(comment: str) -> tuple[str, str, str] | None:
    """(campaign_id, nonce, stage) of a job comment written by `job_tag`, else None."""
    if not comment.startswith("mpd:"):
        return None
    parts = comment[4:].rsplit(":", 2)
    return (parts[0], parts[1], parts[2]) if len(parts) == 3 and all(parts) else None


# ---------------------------------------------------------------- the queue
QUEUE_FORMAT = "%F|%j|%k|%T|%M|%r"      # %F, not %A: %A is per task for a running array, %F is the id sbatch printed


@dataclass(frozen=True)
class QueueRow:
    job_id: str
    name: str
    comment: str
    state: str
    elapsed: str
    reason: str

    @property
    def alive(self) -> bool:
        return self.state not in TERMINAL


def read_queue(squeue: str = "squeue", *, user: str | None = None) -> tuple[list[QueueRow] | None, str]:
    """All of this user's jobs the controller still remembers (`-t all` keeps a job that already finished from
    looking like a job that was never queued). (None, message) when squeue cannot answer."""
    proc = run_tool([squeue, "-u", user or current_user(), "-h", "-t", "all", "-o", QUEUE_FORMAT], timeout=60)
    if proc.returncode != 0:
        return None, message_of(proc)
    rows = []
    for line in proc.stdout.splitlines():
        parts = line.rsplit("|", 3)             # peel state, elapsed, reason off the right: a name may hold a "|"
        first = parts[0].split("|", 2) if len(parts) == 4 else []
        if len(first) == 3 and first[0].strip().isdigit():
            comment = "" if first[2] == "(null)" else first[2]          # squeue prints (null) for a job without a comment (seen on Vision)
            rows.append(QueueRow(first[0].strip(), first[1], comment, parts[1].strip().split(" ")[0].upper(),
                                 parts[2].strip(), parts[3].strip().strip("()")))
    return rows, ""


def tagged(rows: list[QueueRow], campaign_id: str, *, nonce: str | None = None, stage: str | None = None,
           alive_only: bool = False) -> list[QueueRow]:
    """Rows whose comment carries this campaign (and optionally this submission and stage)."""
    out = []
    for r in rows:
        tag = parse_tag(r.comment)
        if tag and tag[0] == campaign_id and (nonce is None or tag[1] == nonce) and (stage is None or tag[2] == stage) \
                and (r.alive or not alive_only):
            out.append(r)
    return out


def scancel_ids(ids: list[str], scancel: str = "scancel") -> tuple[bool, str]:
    """Cancel jobs. (True, "") when scancel succeeded or only complained about jobs that are already gone."""
    ids = [str(i) for i in ids if str(i).isdigit()]         # the ids come from a file: never let one become an option
    if not ids:
        return True, ""
    proc = run_tool([scancel, *ids], timeout=120)
    if proc.returncode == 0:
        return True, ""
    text = message_of(proc)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return (bool(lines) and all(_BENIGN_CANCEL.search(ln) for ln in lines)), text


# ---------------------------------------------------------------- the marker
def write_text_atomic(path, text: str) -> None:
    """Write through a temp file and rename, so neither a reader nor a crash ever sees half a file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def write_json_atomic(path, obj, *, indent: int | None = 2, sort_keys: bool = False) -> None:
    write_text_atomic(path, json.dumps(obj, indent=indent, sort_keys=sort_keys) + "\n")


def read_marker(campaign_dir) -> dict | None:
    """The marker as a dict; None when the campaign was never submitted. Markers written before `state` existed were
    only written once the whole chain was queued, so they read as `submitted`."""
    path = Path(campaign_dir) / MARKER
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        return {"state": "unreadable", "message": f"{path}: {exc}", "jobs": {}}
    try:
        data = json.loads(text)
        if not isinstance(data, dict):
            raise ValueError("not a JSON object")
    except ValueError as exc:
        return {"state": "unreadable", "message": f"{path} is not valid JSON ({exc})", "jobs": {}}
    data.setdefault("state", "submitted")
    return data


def write_marker(campaign_dir, data: dict) -> None:
    write_json_atomic(Path(campaign_dir) / MARKER, {"schema": SCHEMA, **data, "updated_at": now_iso()})


def create_marker(campaign_dir, data: dict) -> bool:
    """Write the marker only if none exists: False means another submission got there first. A hard link is atomic and
    arrives complete; a filesystem without hard links gets an exclusive create instead."""
    root = Path(campaign_dir)
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / f"{MARKER}.new{os.getpid()}"
    write_json_atomic(tmp, {"schema": SCHEMA, **data, "updated_at": now_iso()})
    try:
        os.link(tmp, root / MARKER)
        return True
    except FileExistsError:
        return False
    except OSError:
        try:
            fd = os.open(root / MARKER, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(tmp.read_text(encoding="utf-8"))
        return True
    finally:
        tmp.unlink(missing_ok=True)


@dataclass
class CampaignState:
    campaign_id: str
    path: Path
    state: str                      # unsubmitted | submitting | submitted | partial | cancelled | failed | unreadable
    submitted_at: str = ""
    nonce: str = ""
    jobs: dict[str, str] = field(default_factory=dict)      # stage -> job id
    cluster: str = ""
    message: str = ""


def _state_of(root: Path, data: dict) -> CampaignState:
    jobs = data.get("jobs") if isinstance(data.get("jobs"), dict) else {}
    return CampaignState(campaign_id=str(data.get("campaign_id") or root.name), path=root, state=str(data["state"]),
                         submitted_at=str(data.get("submitted_at") or ""), nonce=str(data.get("nonce") or ""),
                         jobs={str(k): str(v) for k, v in jobs.items()}, cluster=str(data.get("cluster") or ""),
                         message=str(data.get("message") or ""))


def campaign_state(campaign_dir) -> CampaignState | None:
    """State from the marker; None when there is no marker (never submitted, or only dry-run)."""
    root = Path(campaign_dir)
    data = read_marker(root)
    return None if data is None else _state_of(root, data)


def _is_campaign_dir(path: Path) -> bool:
    return path.is_dir() and any((path / n).exists() for n in (MARKER, "spec.json", "scripts"))


def list_campaigns(home) -> list[CampaignState]:
    """Every campaign under <home>/runs, newest first. One that was written but never submitted is `unsubmitted`."""
    runs = Path(home) / "runs"
    found = []
    for p in (sorted(runs.iterdir()) if runs.is_dir() else []):
        if p.name.startswith(".") or not _is_campaign_dir(p):
            continue
        st = campaign_state(p)
        stamp = (st.submitted_at if st else "") or time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(p.stat().st_mtime))
        found.append((stamp, st or CampaignState(campaign_id=p.name, path=p, state="unsubmitted")))
    return [st for _, st in sorted(found, key=lambda t: t[0], reverse=True)]


def resolve_campaign(arg, home) -> Path:
    """A campaign directory from a path, a full id, or a unique prefix of an id under <home>/runs."""
    text = str(arg)
    path = Path(text).expanduser()
    if _is_campaign_dir(path):
        return path.absolute()
    runs = Path(home) / "runs" if home else None
    names = sorted(p.name for p in runs.iterdir() if _is_campaign_dir(p)) if runs and runs.is_dir() else []
    if text in names:
        return runs / text
    matches = [n for n in names if n.startswith(text)]
    if len(matches) == 1:
        return runs / matches[0]

    def shown(items: list[str]) -> str:
        return "\n  " + "\n  ".join(items[:12]) + (f"\n  ... and {len(items) - 12} more" if len(items) > 12 else "")

    if matches:
        raise CampaignError(f"{text!r} matches {len(matches)} campaigns under {runs}; give more of the id:{shown(matches)}")
    if runs is None:
        raise CampaignError(f"{text!r} is not a campaign directory. Give the directory, or set --home / MPD_HOME to look "
                            "the campaign up by id")
    if names:
        raise CampaignError(f"no campaign called {text!r} under {runs}. Campaigns there:{shown(names)}")
    raise CampaignError(f"no campaigns under {runs}: `mpd submit` (or `mpd submit --dry-run`) creates one")


# ---------------------------------------------------------------- scrubbing
def scrub(text: str, extra: tuple[str, ...] = ()) -> str:
    """Hide account numbers, e-mail addresses and the user name, so log excerpts can be pasted into an issue.
    `extra` are further strings to hide (an account that is not a long number)."""
    for token in sorted({t for t in extra if t and len(t) >= 3}, key=len, reverse=True):
        text = text.replace(token, "<account>")
    text = _ACCOUNT_LIKE.sub("<account>", _EMAIL.sub("<email>", text))
    names = {current_user(), os.environ.get("LOGNAME", ""), os.environ.get("USER", "")}
    try:
        names.add(getpass.getuser())
    except Exception:       # no passwd entry for this uid (containers)
        pass
    for name in sorted((n for n in names if len(n) >= 3 and n != "root"), key=len, reverse=True):
        text = re.sub(rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])", "<user>", text)
    return text


def _accounts_in_scripts(root: Path) -> tuple[str, ...]:
    found = set()
    for script in sorted((root / "scripts").glob("*.sbatch"))[:20]:
        try:
            found.update(re.findall(r"^#SBATCH\s+(?:--account=|-A\s+)(\S+)", script.read_text(encoding="utf-8"), re.M))
        except OSError:
            pass
    return tuple(found)


def scrub_log(campaign_dir, text: str) -> str:
    """`scrub`, also hiding the account written in the campaign's own job scripts."""
    return scrub(text, _accounts_in_scripts(Path(campaign_dir)))


# ---------------------------------------------------------------- Slurm states
@dataclass
class JobState:
    state: str
    elapsed: str = ""
    exitcode: str = ""
    name: str = ""
    detail: str = ""        # for an array: "7 COMPLETED, 2 RUNNING, 1 FAILED"
    limit: str = ""         # the time limit the job was submitted with, as sacct prints it ("08:00:00", "1-00:00:00"); "" when unknown
    node: str = ""          # where it ran (sacct NodeList); "" when it never ran or sacct did not say


_SEVERITY = ("FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL", "DEADLINE", "CANCELLED", "REVOKED")


def _seconds(elapsed: str) -> int:
    days, _, rest = elapsed.rpartition("-")
    try:
        h, m, s = ([0, 0, 0] + [int(x) for x in rest.split(":")])[-3:]
        return (int(days or 0) * 24 + h) * 3600 + m * 60 + s
    except ValueError:
        return 0


def _headline(counts: dict[str, int]) -> str:
    for state in ("RUNNING", "COMPLETING", "PENDING"):
        if counts.get(state):
            return state
    for state in _SEVERITY:
        if counts.get(state):
            return state
    return max(counts, key=counts.get) if counts else "UNKNOWN"


def _task_count(job_id: str) -> int:
    """Tasks one sacct row stands for: a pending array shows as `123_[3-9%8]` or `123_[1,4-5]`, one row for many tasks."""
    m = re.search(r"_\[([^\]]*)\]", job_id)
    if not m:
        return 1
    total = 0
    for part in m.group(1).split("%")[0].split(","):
        lo, _, hi = part.partition("-")
        total += (int(hi) - int(lo) + 1) if lo.isdigit() and hi.isdigit() else 1
    return total


def _column(cols: list[str], index: int) -> str:
    """An optional sacct column; Slurm's placeholders for "none" are read as empty."""
    value = cols[index].strip() if index < len(cols) else ""
    return "" if value in ("None assigned", "None", "Unknown", "UNLIMITED", "Partition_Limit") else value


def _parse_sacct(text: str, wanted: set[str]) -> dict[str, JobState]:
    groups: dict[str, list[list[str]]] = {}
    for line in text.splitlines():
        cols = line.split("|")
        m = re.match(r"\d+", cols[0]) if len(cols) >= 5 else None
        if m and m.group() in wanted:
            groups.setdefault(m.group(), []).append(cols)
    out = {}
    for job, rows in groups.items():
        counts: dict[str, int] = {}
        for cols in rows:
            state = (cols[2].split() or ["UNKNOWN"])[0].rstrip("+")
            counts[state] = counts.get(state, 0) + _task_count(cols[0])
        exit_codes = [c[4] for c in rows if c[4] and c[4] != "0:0"]
        detail = ", ".join(f"{n} {s}" for s, n in sorted(counts.items(), key=lambda kv: -kv[1])) if sum(counts.values()) > 1 else ""
        headline = _headline(counts)
        node = next((_column(c, 6) for c in rows if _column(c, 6) and (c[2].split() or [""])[0].rstrip("+") == headline), "")
        out[job] = JobState(headline, max((c[3] for c in rows), key=_seconds), (exit_codes or ["0:0"])[0], rows[0][1], detail,
                            _column(rows[0], 5), node)
    return out


SACCT_FIELDS_BASIC = "JobID,JobName,State,Elapsed,ExitCode"
SACCT_FIELDS = SACCT_FIELDS_BASIC + ",Timelimit,NodeList"


def job_states(job_ids, *, sacct: str = "sacct", squeue: str | None = "squeue") -> dict[str, JobState]:
    """State per recorded job id (an array is one entry). sacct is the record; squeue fills in jobs it does not know yet
    (accounting lags the queue). Missing sacct or squeue gives fewer entries, never an error."""
    ids = [str(j) for j in job_ids if str(j).isdigit()]
    if not ids:
        return {}
    out: dict[str, JobState] = {}
    for fields in (SACCT_FIELDS, SACCT_FIELDS_BASIC):        # the extra columns are a bonus: a sacct that refuses them still answers
        proc = run_tool([sacct, "-X", "-n", "-P", "-j", ",".join(ids), f"--format={fields}"], timeout=60)
        if proc.returncode == 0:
            out = _parse_sacct(proc.stdout, set(ids))
            break
    if squeue and any(j not in out for j in ids):
        rows, _ = read_queue(squeue)
        for r in rows or []:
            if r.job_id in ids and r.job_id not in out:
                out[r.job_id] = JobState(r.state, r.elapsed, "", r.name)
    return out


def pending_reasons(job_ids, *, squeue: str = "squeue", any_state: bool = False) -> dict[str, str]:
    """Why each pending job waits (Dependency, Priority, Resources, PartitionTimeLimit, ...). Other jobs are absent, unless
    `any_state`: then every job squeue still remembers with a reason other than None (TimeLimit, DependencyNeverSatisfied, ...)."""
    wanted = {str(j) for j in job_ids}
    rows, _ = read_queue(squeue)
    return {r.job_id: r.reason for r in rows or [] if r.job_id in wanted and r.reason
            and (r.state == "PENDING" or (any_state and r.reason != "None"))}


# ---------------------------------------------------------------- progress on disk
def task_marker(root: Path, stage: str, task: dict) -> Path | None:
    """The file a finished task leaves behind (the same files the stages skip on); None for a stage we do not know."""
    try:
        if stage == "prepare":
            return root / "contexts" / "MANIFEST.json"
        if stage == "teacher":
            return root / "teacher" / "COMPLETE.json"
        if stage == "train":
            return root / "train" / task["unit"] / "COMPLETE.json"
        if stage in ("score", "score-base"):
            return root / "score" / task["dataset"] / task["unit"] / f"shard_{task['shard']:03d}_of_{task['num_shards']:03d}.json"
        if stage == "evaluate":
            return root / "evaluate" / "summary.json"
    except (KeyError, TypeError, ValueError):
        pass
    return None


def tasks_finished(root, stage: str, tasks: list[dict]) -> int:
    root = Path(root)
    return sum(1 for t in tasks if (m := task_marker(root, stage, t)) is not None and m.is_file())


def stage_progress(campaign_dir) -> dict[str, tuple[int, int]]:
    """stage -> (finished tasks, expected tasks), from tasks/*.json and the markers the stages write. A shard counts
    only if its `of` number is the one the plan asked for, so leftovers of another --score-shards do not."""
    root = Path(campaign_dir)
    resolved = _load_json(root / "resolved.json")
    shards = resolved.get("score_shards") if isinstance(resolved, dict) else None
    names = sorted(p.stem for p in (root / "tasks").glob("*.json")) if (root / "tasks").is_dir() else []
    names.sort(key=lambda n: (STAGE_ORDER.index(n) if n in STAGE_ORDER else len(STAGE_ORDER), n))
    out = {}
    for name in names:
        tasks = _load_json(root / "tasks" / f"{name}.json")
        if not isinstance(tasks, list):
            continue
        if isinstance(shards, dict):      # a task without its own shard count falls back to the plan's
            tasks = [t if "num_shards" in t or "dataset" not in t else {**t, "num_shards": shards.get(t["dataset"])} for t in tasks]
        out[name] = (tasks_finished(root, name, tasks), len(tasks))
    return out


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def stage_plan(campaign_dir) -> dict[str, dict]:
    """plan.json's record of each stage by full stage name (walltime_s, gpu, mem_gb, depends_on, partition_limit_s, ...); {} when
    the file is missing or unreadable."""
    data = _load_json(Path(campaign_dir) / "plan.json")
    out = {}
    for rec in (data.get("stages") if isinstance(data, dict) else None) or []:
        if isinstance(rec, dict) and rec.get("name"):
            out[rec["name"] + (f".{rec['part']}" if rec.get("part") else "")] = rec
    return out


# ---------------------------------------------------------------- logs
_LOG_NAME = re.compile(r"^mpd-(?P<stage>.+)-[^-_]+_(?P<job>\d+)(?:_\d+)?\.(?P<ext>err|out)$")     # mpd-<stage>-<student>_<job>[_<task>].err|out


def _stage_matches(wanted: str, full: str) -> bool:
    return full == wanted or full.split(".")[0] == wanted


def stage_log(campaign_dir, stage: str, *, err: bool = True, job: str | None = None) -> Path | None:
    """The log of a stage: the newest non-empty one (an array has one per task), else the newest. `stage` is a full stage
    name (score.1) or a base name (score, which then covers its chunks but not score-base). With `job`, only that job's logs
    count, so a job that never ran does not show the text of an earlier attempt."""
    root = Path(campaign_dir)
    marker = read_marker(root) or {}
    ids = {str(v) for k, v in (marker.get("jobs") or {}).items() if _stage_matches(stage, str(k))}
    files = []
    for p in (root / "logs").glob("*.err" if err else "*.out") if (root / "logs").is_dir() else []:
        m = _LOG_NAME.match(p.name)
        if m and (m.group("job") in ids or _stage_matches(stage, m.group("stage"))) and (job is None or m.group("job") == str(job)):
            files.append(p)
    if not files:
        return None
    files.sort(key=lambda p: p.stat().st_mtime)
    return next((p for p in reversed(files) if p.stat().st_size > 0), files[-1])


def failure_log(campaign_dir, stage: str) -> Path | None:
    return stage_log(campaign_dir, stage, err=True)


def log_tail(campaign_dir, stage: str, n: int = 20, *, err: bool = True, job: str | None = None) -> str:
    """Last n lines of the stage's log, with account numbers and the user name scrubbed. "" when there is none."""
    log = stage_log(campaign_dir, stage, err=err, job=job)
    if log is None:
        return ""
    with open(log, "rb") as fh:
        fh.seek(0, os.SEEK_END)
        fh.seek(max(0, fh.tell() - 256 * 1024))
        text = fh.read().decode("utf-8", errors="replace")
    # progress bars rewrite one line with \r: keep what the terminal would have shown last
    lines = [ln.split("\r")[-1] for ln in text.rstrip("\n").split("\n")][-n:]
    half = MAX_LINE // 2
    lines = [ln if len(ln) <= MAX_LINE else f"{ln[:half]} ... [{len(ln) - MAX_LINE} characters omitted] ... {ln[-half:]}" for ln in lines]
    return scrub("\n".join(lines), _accounts_in_scripts(Path(campaign_dir)))


def failure_tail(campaign_dir, stage: str, n: int = 20, *, job: str | None = None) -> str:
    """Last n lines of the stage's error log (what a failed job printed), scrubbed."""
    return log_tail(campaign_dir, stage, n, err=True, job=job)


# ---------------------------------------------------------------- what `mpd status` tells a student to do next
# The words live here, in dicts and two short functions, so that the docs and the tests quote the same text. None of it was
# checked on Grace. State and reason names are Slurm's (squeue(1), JOB REASON CODES and JOB STATE CODES, Slurm 25.05). The cause of
# a failure is read from the markers mpd's own scripts and stages print (MPD_NO_GPU, MPD_MODULE_LOAD_FAILED, the weights error)
# and from well-known Python messages; a cause or a reason that is not known is shown as it is.
NO_GPU_EXIT = 31                # stages.common.NO_GPU_EXIT (a test keeps the two equal)
TIME_FLAGS = {"train": "--train-time", "teacher": "--teacher-time", "score": "--score-time", "score-base": "--score-time"}

ADVICE = {
    "again": "Run your original `mpd submit ...` command again",
    # a job that hit its time limit
    "timeout": "{head} reached its time limit: it ran {elapsed} of the {limit} it asked for.",
    "timeout_unknown": "{head} reached its time limit after {elapsed}.",
    "timeout_estimate": "That limit was mpd's estimate, not a measurement.",
    "resume_train": "Training saves its progress every MPD_CKPT_EVERY updates (200 unless you changed it) and continues from the latest "
                    "save, so the work done so far is kept.",
    "resume_score": "Scoring keeps the shards it finished; only the unfinished ones run again.",
    "resume_teacher": "The teacher writes its table only when its whole pass is done, so a new attempt starts from the beginning and "
                      "the limit has to cover all of it.",
    "timeout_retry": "{again} with a larger {flag} (for example {flag} {example}; this partition allows up to {biggest}) and --force.",
    "timeout_retry_unknown_max": "{again} with a larger {flag} (for example {flag} {example}) and --force.",
    "timeout_at_maximum_train": "It already asked for the longest time this partition allows ({biggest}). Run the same command with "
                                "--force again: every attempt continues from the latest save.",
    "timeout_at_maximum": "It already asked for the longest time this partition allows ({biggest}). Use more shards (--score-shards) "
                          "or a faster GPU (--gpu) so that one task fits.",
    "timeout_no_flag": "This stage has no time option of its own. Report it with `mpd doctor --report` and the end of its log.",
    # out of memory
    "oom": "{head} ran out of memory{oom_code}.",
    "oom_gpu": "The GPU's memory is too small for this model at this size: use a smaller student or a larger GPU (--gpu; `mpd clusters` "
               "lists them). More GPUs do not help, because every GPU holds a full copy of the model. mpd's memory figures are "
               "estimates, not measurements. Then {again_l} with --force.",
    "oom_unknown": "If its log mentions CUDA, the GPU is too small for this model at this size: use a smaller student or a larger GPU "
                   "(--gpu; `mpd clusters` lists them). More GPUs do not help, because every GPU holds a full copy of the model. "
                   "If it does not, the node ran out of host memory{mem}: add --sbatch-arg=--mem=<more>G. mpd's memory figures are "
                   "estimates, not measurements. Then {again_l} with --force.",
    # the node
    "node_fail": "{head} ended with {state}: the node it ran{where} failed, which is not a problem with your settings.",
    "node_retry": "{again} with --force.{exclude}",
    "node_exclude": " If it lands on {node} again, add --sbatch-arg=--exclude={node}.",
    # a job that exited with an error
    "failed": "{head} failed{code}.",
    "failed_unknown": "The end of its error log is above; `mpd logs {ref} {base} --err` shows all of it. When the cause is fixed, "
                      "{again_l} with --force: finished stages are kept.",
    # a job that was cancelled
    "cancelled_dependency": "{head} was cancelled because a stage it waited for did not finish.",
    "cancelled_dependency_next": "Nothing to do for this stage until that one is fixed; then {again_l} with --force.",
    "cancelled_all": "{stages}: cancelled with `mpd cancel`.",
    "cancelled": "{head} was cancelled: by `mpd cancel`, by an administrator, or by a limit of the account.",
    "cancelled_next": "{again} to queue the campaign again.",
    "cancelled_next_running": "{again} with --force to queue what is missing.",
    "other": "{head} ended with state {state}, which mpd has no advice for.",
    "other_next": "`mpd logs {ref} {base} --err` shows the end of its log; then {again_l} with --force.",
}

# What a failed job's log says about the cause: the first match wins; the names are the keys of CAUSE_ADVICE.
CAUSES = (
    ("no_gpu", re.compile(r"MPD_NO_GPU")),
    ("module", re.compile(r"MPD_MODULE_LOAD_FAILED")),
    ("weights", re.compile(r"weights for \S+ (?:are|have) |mpd cannot fetch weights")),
    ("offline", re.compile(r"huggingface\.co|OfflineModeIsEnabled|offline mode is enabled|LocalEntryNotFoundError", re.I)),
    ("cuda_oom", re.compile(r"CUDA out of memory|OutOfMemoryError")),
)
CAUSE_ADVICE = {
    "no_gpu": "The node has no usable GPU (the job stopped instead of running on the CPU). {again} with "
              "--sbatch-arg=--exclude={node} and --force.",
    "module": "A module the job needs could not be loaded. The modules come from the environment record: `mpd doctor` shows it (row "
              "env-record) and `bash scripts/setup_env.sh --cluster {cluster}` writes it again. Then {again_l} with --force.",
    "weights": "The job needed model weights that are not on disk, and compute nodes are offline. On a login node run `{fetch}`, then "
               "{again_l} with --force.",
    "offline": "The job tried to reach huggingface.co, which compute nodes cannot do. The weights have to be on disk first: on a login "
               "node run `{fetch}`, then {again_l} with --force.",
}
# Text from a log is advice only if it is a plain command or node name: a log can hold anything, and what is shown is meant to be pasted.
_FETCH = re.compile(r"(mpd fetch model [^\n]+)")
_SAFE_FETCH = re.compile(r"mpd fetch model(?: [A-Za-z0-9._-]+)+(?: --cluster [A-Za-z0-9._-]+)?(?: --home [A-Za-z0-9._/+@:=,-]+)?")
_SAFE_NODE = re.compile(r"[A-Za-z0-9_.,\[\]-]+")
_NODE_IN_ERR = re.compile(r"MPD_NO_GPU: no usable GPU on (\S+?);")
_NODE_IN_OUT = re.compile(r"MPD_JOB_START host=(\S+)")

# One line per Slurm reason a job waits for. `{ref}` is the campaign as the user types it. Keys are matched on the first word of
# squeue's reason ("ReqNodeNotAvail, UnavailableNodes:g001" is ReqNodeNotAvail).
PENDING_REASONS = {
    "Priority": "other jobs are ahead of it; this is the usual wait for a GPU. Nothing to do.",
    "Resources": "it is next in line and waits for a node with the GPU it asked for. Nothing to do.",
    "Dependency": "it waits for the stage or stages before it to finish. Nothing to do.",
    "JobArrayTaskLimit": "the array already runs as many tasks at once as its throttle allows (--array-throttle); the rest start as "
                         "tasks finish. Nothing to do.",
    "DependencyNeverSatisfied": "a stage it waited for failed or was cancelled, so it will never start. Fix that stage, then run your "
                                "original `mpd submit ...` command again with --force.",
    "PartitionTimeLimit": "its time limit is longer than the partition allows, so Slurm will never start it. `mpd cancel {ref}`, then "
                          "submit again with a shorter walltime.",
    "ReqNodeNotAvail": "a node it needs is not available (down, drained or reserved). If you excluded nodes with "
                       "--sbatch-arg=--exclude=..., check the names; otherwise wait or ask the cluster's support.",
    "BeginTime": "its start time has not been reached. Nothing to do.",
    "JobHeldUser": "it is held; `scontrol release <job id>` frees it.",
    "JobHeldAdmin": "an administrator holds it; ask the cluster's support.",
    "InvalidAccount": "Slurm does not accept the account. `myproject` lists yours; `mpd cancel {ref}`, then submit again with --account.",
    "InvalidQOS": "Slurm does not accept the QOS. `mpd cancel {ref}`, then submit again without --sbatch-arg=--qos=...",
    "PartitionDown": "its partition is down. Wait, or ask the cluster's support.",
    "PartitionNodeLimit": "its node request is outside what the partition allows, or nodes are down. Ask the cluster's support.",
    "NodeDown": "a node it needs is down. Wait, or ask the cluster's support.",
    "Reservation": "it waits for an advance reservation. Nothing to do.",
    "Licenses": "it waits for a software license. Nothing to do.",
}
# Reasons that come in families (AssocGrpBillingMinutes, QOSMaxJobsPerUserLimit, ...); the first pattern that matches wins.
PENDING_FAMILIES = (
    (re.compile(r"balance|^Assoc\w*Minutes", re.I),
     "the account's SU balance, or a limit on its minutes, cannot cover the job. `myproject` shows the balance; reduce the plan or "
     "ask the cluster's support for more SUs."),
    (re.compile(r"^Assoc"), "a limit of your account (jobs or resources at once) holds it. It starts when your other jobs end."),
    (re.compile(r"^QOS"), "a limit of its QOS (jobs or GPUs per user at once) holds it back. It starts when your other jobs end."),
)


def find_cause(*texts: str) -> str | None:
    """The first known cause that shows in the given log texts (stderr first), else None."""
    for text in texts:
        for name, pattern in CAUSES:
            if pattern.search(text or ""):
                return name
    return None


def pending_help(reason: str, ref: str = "<campaign>") -> str:
    """One line on why a job waits. A reason that is not known is shown as Slurm says it."""
    reason = reason.strip().strip("()")
    word = re.split(r"[,\s]", reason, maxsplit=1)[0]
    text = PENDING_REASONS.get(word) or next((t for p, t in PENDING_FAMILIES if p.search(word) or p.search(reason)), None)
    if text is None:
        return f"Slurm says: {reason}"
    return text.format(ref=ref) + (f" (Slurm: {reason})" if reason != word else "")


def exit_info(exitcode: str) -> tuple[int, int]:
    """(exit status, signal) from sacct's ExitCode "status:signal"; (0, 0) for anything unreadable."""
    status, _, signal = (exitcode or "").partition(":")
    try:
        return int(status or 0), int(signal or 0)
    except ValueError:
        return 0, 0


def node_of(state: JobState | None, err: str = "", out: str = "") -> str:
    """Where a job ran: sacct's NodeList, else the node in the script's own MPD_NO_GPU line, else the host the job announced."""
    m = _NODE_IN_ERR.search(err or "") or _NODE_IN_OUT.search(out or "")
    node = state.node if state is not None and state.node else (m.group(1).split(".")[0] if m else "")
    return node if _SAFE_NODE.fullmatch(node) else ""


def _fmt_walltime(seconds: int) -> str:
    days, rem = divmod(int(seconds), 86400)
    return (f"{days}-" if days else "") + f"{rem // 3600:02d}:{rem % 3600 // 60:02d}:{rem % 60:02d}"


def _timeout_lines(say, base: str, state: JobState, plan: dict) -> list[str]:
    limit_s = _seconds(state.limit) or int(plan.get("walltime_s") or 0)
    biggest = int(plan.get("partition_limit_s") or 0)
    lines = [say("timeout", limit=_fmt_walltime(limit_s)) if limit_s else say("timeout_unknown")]
    if plan.get("walltime_basis") in ("estimate", "measured"):
        lines[0] += " " + ADVICE["timeout_estimate"]
    if base in TIME_FLAGS:
        lines.append(ADVICE["resume_" + ("score" if base == "score-base" else base)])
    flag = TIME_FLAGS.get(base)
    if not flag:
        lines.append(ADVICE["timeout_no_flag"])
    elif biggest and limit_s >= biggest:
        lines.append(say("timeout_at_maximum_train" if base == "train" else "timeout_at_maximum", biggest=_fmt_walltime(biggest)))
    else:
        twice = -(-2 * limit_s // 3600) * 3600 if limit_s else 8 * 3600
        example = _fmt_walltime(min(twice, biggest) if biggest else twice)
        lines.append(say("timeout_retry", flag=flag, example=example, biggest=_fmt_walltime(biggest)) if biggest
                     else say("timeout_retry_unknown_max", flag=flag, example=example))
    return lines


def advise_cancelled(stages: list[str]) -> list[str]:
    """The two lines for a campaign that `mpd cancel` stopped: one sentence for all its cancelled stages instead of one each."""
    return [ADVICE["cancelled_all"].format(stages=", ".join(stages)), ADVICE["cancelled_next"].format(again=ADVICE["again"])]


def advise(stage: str, job: str, state: JobState | None, *, ref: str, reason: str = "", err: str = "", out: str = "",
           plan: dict | None = None, cluster: str = "", cancelled: bool = False) -> list[str]:
    """Plain-words lines for a stage that did not end well: what happened, then the command that continues.

    `plan` is the stage's record in plan.json (walltime_s, mem_gb, partition_limit_s, walltime_basis); `err` and `out` are the ends
    of its logs; `reason` is squeue's reason for the job; `cancelled` says the campaign was cancelled with `mpd cancel`. The first
    line says what happened. An empty list means the state needs no advice."""
    plan = plan or {}
    base = stage.split(".")[0]
    word = state.state.split()[0] if state and state.state else ""
    status, signal = exit_info(state.exitcode if state else "")
    cause = find_cause(err, out)
    node = node_of(state, err, out)
    again = ADVICE["again"]
    ctx = dict(head=f"{stage} (job {job})", elapsed=(state.elapsed if state and state.elapsed else "?"), again=again,
               again_l=again[0].lower() + again[1:], ref=ref, base=base, state=word, node=node or "<node>", cluster=cluster or "<cluster>",
               code=f" with exit code {status}" if status else "", oom_code=" (exit 137)" if status == 137 else "")

    def say(key: str, **more) -> str:
        return ADVICE[key].format(**{**ctx, **more})

    killed = status == 137 or signal == 9
    if word in ("", "COMPLETED", "COMPLETING", "RUNNING", "PENDING", "CONFIGURING"):
        return []
    if word == "TIMEOUT":
        return _timeout_lines(say, base, state, plan)
    if word == "OUT_OF_MEMORY" or (word == "FAILED" and (cause == "cuda_oom" or (killed and cause is None))):
        mem = f" (the job asked for {plan['mem_gb']}G)" if plan.get("mem_gb") else ""
        return [say("oom"), say("oom_gpu") if cause == "cuda_oom" else say("oom_unknown", mem=mem)]
    if word in ("NODE_FAIL", "BOOT_FAIL"):
        return [say("node_fail", where=f" on {node}" if node else ""),
                say("node_retry", exclude=say("node_exclude") if node else "")]
    if word == "FAILED":
        lines = [say("failed")]
        if cause is None and status == NO_GPU_EXIT:
            cause = "no_gpu"
        if cause in CAUSE_ADVICE:
            found = (_FETCH.search(err or "") or _FETCH.search(out or ""))
            fetch = found.group(1).strip() if found and _SAFE_FETCH.fullmatch(found.group(1).strip()) else "mpd fetch model <KEY>"
            lines.append(CAUSE_ADVICE[cause].format(**{**ctx, "fetch": fetch}))
        else:
            lines.append(say("failed_unknown"))
        return lines
    if word == "CANCELLED":
        if reason == "DependencyNeverSatisfied":
            return [say("cancelled_dependency"), say("cancelled_dependency_next")]
        return [say("cancelled"), say("cancelled_next" if cancelled else "cancelled_next_running")]
    return [say("other"), say("other_next")] if word else []


# ---------------------------------------------------------------- cancel
def cancel(campaign_dir, *, scancel: str = "scancel", squeue: str = "squeue", echo=print) -> list[str]:
    """Cancel every queued or running job of the campaign and mark it `cancelled`. The marker's ids are used; when the
    marker is missing, partial or failed, jobs are also found in the queue by their comment, because a job that sbatch
    accepted while reporting an error is not in the marker. Raises SubmitError (marker `partial`) if scancel fails."""
    root = Path(campaign_dir)
    if not root.is_dir():
        raise CampaignError(f"{root} is not a campaign directory")
    data = read_marker(root)
    cid = str((data or {}).get("campaign_id") or root.name)
    recorded = [str(v) for v in ((data or {}).get("jobs") or {}).values() if str(v).isdigit()]
    rows, why = read_queue(squeue)
    if rows is None:
        if not recorded:
            raise SubmitError(f"squeue did not answer ({why}) and no job ids are recorded for {cid}, so its jobs cannot be found. "
                              f"Run `mpd cancel {cid}` again when `squeue -u $USER` answers")
        # cannot tell what is alive: ask scancel about the recorded ids (it ignores finished ones)
        echo(f"squeue did not answer ({why}); cancelling the recorded jobs without checking the queue")
        targets = recorded
    else:
        alive = {r.job_id for r in rows if r.alive}
        found = {r.job_id for r in tagged(rows, cid, alive_only=True)}
        targets = [j for j in dict.fromkeys(recorded) if j in alive] + sorted(found - set(recorded), key=int)
    previous = (data or {}).get("state")
    if not targets:
        if previous in ("submitted", "cancelled"):
            echo(f"no job of {cid} is queued or running" + (": they have finished, nothing to cancel" if previous == "submitted"
                                                              else " (already cancelled)"))
        else:
            _mark(root, data, cid, "cancelled", "cancelled: nothing was queued or running")
            echo(f"no job of {cid} is queued or running; marked cancelled, so it can be submitted again")
        return []
    ok, text = scancel_ids(targets, scancel)
    if not ok:
        line = f"scancel {' '.join(targets)}"
        _mark(root, data, cid, "partial", f"scancel failed ({text.splitlines()[0] if text else 'no message'}); run: {line}")
        raise SubmitError(f"scancel failed for {cid}: {text}\nThe job ids are kept in {root / MARKER}. Run `mpd cancel {cid}` "
                          f"again when Slurm answers, or by hand: {line}")
    _mark(root, data, cid, "cancelled", f"cancelled {len(targets)} job(s) with mpd cancel at {now_iso()}")
    echo(f"cancelled {len(targets)} job(s) of {cid}: {' '.join(targets)}")
    return targets


def _mark(root: Path, data: dict | None, cid: str, state: str, message: str) -> None:
    base = dict(data) if data and data.get("state") != "unreadable" else {"campaign_id": cid, "jobs": {}}
    write_marker(root, {**base, "state": state, "message": message})
