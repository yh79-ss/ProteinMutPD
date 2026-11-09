"""Fake Slurm toolbox for tests (no real Slurm is ever called). Import the fixture where you need it:

    from fx_slurm import fake_slurm  # noqa: F401

`fake_slurm(tmp_path, monkeypatch)` puts a directory of executable stand-ins for sbatch, squeue, scancel, sacct, scontrol,
module, myproject, showquota, nvidia-smi (and tripwires for srun, salloc, sacctmgr) first on PATH, sets BASH_ENV so that
`module` is also a shell function that changes $LOADEDMODULES, and sets USER/LOGNAME to a made-up name. The real Slurm
clients (sinfo, sstat, ... and a removed fake) are hidden from PATH, so a call that is not faked fails with "command not found"
instead of reaching the cluster. The stand-ins are this file run as a script (`python -I -S fx_slurm.py <state dir> <tool>
args...`): stdlib only, about 20-60 ms per call. What they do comes from a JSON scenario, re-read on every call, so a test may
change it between calls. Every call is appended to a log. Time is virtual: nothing here sleeps; `fake_slurm.sleep` advances the
clock the scenario sees (pass `sleep=fake_slurm.sleep, clock=fake_slurm.clock` to submit()).

Test-side API (the object the fixture returns)
    configure(**keys)       merge keys into the scenario (top-level keys replace)      scenario()     the current scenario
    rule(**rule)            append an sbatch rule                                       fail_sbatch(n=None, stage=None, message=...)
    accept_but_fail(stage, times=1, hide_squeue_calls=0)   queued, but sbatch exits 1 (the 2026-07-26 incident)
    reject_stage(stage, message)  refuse a stage for good  outage(until_s, accept=False, tools=...)   everything down until t
    squeue_fails(calls=True)  scancel_fails()  sacct_fails()                            controller errors for one tool
    calls(tool=None) -> [Call(tool, argv, rc, stdout, stderr, t, info)]                 test_only_calls()   submitted()
    jobs() -> {id: job}   job(id_or_stage) -> job | None   alive_ids()   state_of(id)    submitted() -> accepted real jobs in order
    set_state(job, state, reason=None, exitcode=None, elapsed=None, tasks=None)         add_job(**fields) -> id (a foreign or stale job)
    sleep(s), clock(), sleeps, advance(s)     injectable `sleep=` / `clock=`             remove(tool) / restore(tool)
    env(extra=None) -> environ for a subprocess (fake PATH, BASH_ENV, USER)             set_modules(names)  what `module load` accepts
    assert_no_tripwire() / clear_tripwires()   srun/salloc/sacctmgr were called         path   the PATH the fixture installed
A job is a dict: id (int), name, comment, user, stage, script, state, reason, deps (ints), kill_invalid, array (spec or None),
n_array, partition, account, gres, time, cpus, mem, submit_t, elapsed, exitcode, tasks (explicit per-task states or None).
A stand-in for the planner's output is `stub_plan(tmp_path, ...)` with `stub_render` and `write_stub_campaign(plan, state=, jobs=)`.

Scenario format (all keys optional; the defaults are a healthy cluster)
    sbatch:
      rules: [ rule, ... ]      first matching rule wins; none matching = accept
        match keys: stage ("train" or ["train", "score"]; full stage name from the --comment tag or the script file stem),
                    call (Nth real sbatch call for that stage, 1-based; N, [N, M] or {"gte": a, "lte": b}), nth (same, counted
                    over all stages), test_only (true/false), after_s / until_s (virtual clock window)
        action:     do = "accept" | "fail" | "accept_but_fail"; message (stderr text); exit (status, default 1);
                    hide_squeue_calls N (accept_but_fail: the job is invisible to the next N squeue calls)
      duration_s            virtual seconds each sbatch call takes (a hung controller takes 30 to give up)
      job_submit_lines      true: print Grace's charge banner (made-up account 123456789012) on stderr after an accepted job
      drop_comment          true: the cluster keeps no --comment (a job_submit plugin may rewrite it): jobs are queued with ""
      reject_flags          {"--get-user-env": "message"}  a directive or argument starting with the key is refused
      gres_types            ["a100", "t4"]  a typed --gres=gpu:<type>:N with another type is refused (case-sensitive)
      partitions            {"gpu": {"max_time_s": 345600, "exceed": "reject" | "pend"}}  unknown partition refused; a too long
                            request is refused, or accepted and left PENDING with reason PartitionTimeLimit (what Grace does)
      strict_dependencies   default true: --dependency on an unknown job id is refused ("Job dependency problem")
    squeue:  fail (true | N | [N, M] | {"gte", "lte"}: exit 1), message, hide_finished (finished jobs vanish from `-t all`)
    sacct:   fail, message, lag_calls N (the first N sacct calls of the test know no job: accounting lags the queue)
    scancel: fail, message
    outage:  {"from_s": 0, "until_s": 600, "tools": ["sbatch", "squeue", "scancel", "sacct"], "message": ..., "accept": false}
             while the virtual clock is inside the window those tools fail; "accept": true makes sbatch queue the job
             (hidden from squeue) and still report the error
    progress: {"enabled": true, "end_state": {"default": "COMPLETED", "train": "FAILED"}}
             every squeue/sacct call moves jobs one step: PENDING -> RUNNING -> end state, a job waits for its afterok
             dependencies and is CANCELLED (DependencyNeverSatisfied) when one of them ended badly
    module: {"available": [...]}   myproject / showquota: {"stdout": "...", "exit": 0}   scontrol: {"config": {...}}
    nvidia_smi: {"fail": true}   user: made-up user name

Fidelity (what is copied from real Slurm and what is invented). Copied: option parsing and the refusals it implies
(`--get-user-env=L` is rejected, as Slurm 25.05.6 does on Vision), a script must start with #! and have LF line ends,
`sbatch --parsable` prints the bare id, `--test-only` queues nothing, squeue hides finished jobs unless `-t all` and prints
`(null)` for an empty comment (seen on Vision, Slurm 25.05.6), scancel
on a finished or unknown job fails with the usual messages, an array has one id but several squeue/sacct rows. Invented:
the text of myproject/showquota (the real format is not known), job timings, node names, the exact balance-refusal layout.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shlex
import sys
from pathlib import Path

# Part 1, the fake binaries. Stdlib only: `python -I -S fx_slurm.py ...` stops at the `__main__` guard below, before the
# pytest half of the file is read.
CONTROLLER_DOWN = "sbatch: error: Batch job submission failed: Unable to contact slurm controller (connect failure)"
SQUEUE_DOWN = "squeue: error: Unable to contact slurm controller (connect failure)"
SCANCEL_DOWN = "scancel: error: Unable to contact slurm controller (connect failure)"
SACCT_DOWN = "sacct: error: Unable to contact slurm controller (connect failure)"
# reconstructed from the layout in grace_truth.md 5.4 (HPRC primer p.42); the exact text on Grace is not verified
BALANCE = ("sbatch: error: (from job_submit) your account's balance is not sufficient to submit your job\n"
           "              Project Account: 123456789012\n"
           "              Account Balance: 100.0\n"
           "              Requested SUs:   9312.0")
INVALID_ACCOUNT = "sbatch: error: Batch job submission failed: Invalid account or account/partition combination specified"
BAD_GRES = "sbatch: error: Batch job submission failed: Requested node configuration is not available"
BAD_TIME = "sbatch: error: Batch job submission failed: Requested time limit is invalid (missing or exceeds some limit)"
BAD_PARTITION = "sbatch: error: invalid partition specified: {name}"
DEPENDENCY = "sbatch: error: Batch job submission failed: Job dependency problem"
JOB_SUBMIT_BANNER = """(from job_submit) your job is charged as below
              Project Account: 123456789012
              Account Balance: 20000.0
              Requested SUs:   280.0
  Requested SUs is based on CPU only and does not include SU charges for
  requested GPUs. A negative balance is possible if the difference is
  greater than the remaining balance."""
TERMINAL = {"COMPLETED", "CANCELLED", "FAILED", "TIMEOUT", "NODE_FAIL", "BOOT_FAIL", "DEADLINE", "OUT_OF_MEMORY", "REVOKED"}
MESSAGES = {"controller": CONTROLLER_DOWN, "balance": BALANCE, "account": INVALID_ACCOUNT, "gres": BAD_GRES,
            "time": BAD_TIME, "dependency": DEPENDENCY}

_LONG_VALUE = {
    "account", "array", "begin", "chdir", "clusters", "comment", "constraint", "core-spec", "cores-per-socket",
    "cpu-freq", "cpus-per-gpu", "cpus-per-task", "deadline", "delay-boot", "dependency", "distribution", "error",
    "exclude", "export", "export-file", "extra-node-info", "gid", "gpu-bind", "gpu-freq", "gpus", "gpus-per-node",
    "gpus-per-socket", "gpus-per-task", "gres", "gres-flags", "hint", "input", "job-name", "kill-on-invalid-dep",
    "licenses", "mail-type", "mail-user", "mcs-label", "mem", "mem-bind", "mem-per-cpu", "mem-per-gpu", "mincpus",
    "network", "nice", "nodefile", "nodelist", "nodes", "ntasks", "ntasks-per-core", "ntasks-per-gpu",
    "ntasks-per-node", "ntasks-per-socket", "open-mode", "output", "partition", "power", "priority", "profile",
    "propagate", "qos", "reservation", "signal", "sockets-per-node", "switches", "thread-spec", "threads-per-core",
    "time", "time-min", "tmp", "uid", "wait-all-nodes", "wckey", "wrap"}
_LONG_FLAG = {"contiguous", "exclusive", "get-user-env", "help", "hold", "ignore-pbs", "no-kill", "no-requeue",
              "oversubscribe", "overcommit", "parsable", "quiet", "requeue", "spread-job", "test-only", "usage",
              "use-min-nodes", "verbose", "version", "wait"}
_SHORT = {"A": "account", "a": "array", "b": "begin", "C": "constraint", "c": "cpus-per-task", "D": "chdir",
          "d": "dependency", "e": "error", "G": "gpus", "H": "hold", "h": "help", "i": "input", "J": "job-name",
          "k": "no-kill", "L": "licenses", "M": "clusters", "m": "distribution", "N": "nodes", "n": "ntasks",
          "O": "overcommit", "o": "output", "p": "partition", "Q": "quiet", "q": "qos", "s": "oversubscribe",
          "t": "time", "v": "verbose", "W": "wait", "w": "nodelist", "x": "exclude"}
_STATE_CODES = {"PENDING": "PD", "RUNNING": "R", "COMPLETED": "CD", "CANCELLED": "CA", "FAILED": "F", "TIMEOUT": "TO",
                "COMPLETING": "CG", "SUSPENDED": "S", "NODE_FAIL": "NF", "OUT_OF_MEMORY": "OOM"}
_CODE_STATES = {v: k for k, v in _STATE_CODES.items()}
_ELAPSED = {"PENDING": "00:00:00", "RUNNING": "00:10:00"}


def _shim_main(argv: list[str]) -> int:
    state_dir, tool, args = Path(argv[1]), argv[2], argv[3:]
    lock = open(state_dir / ".lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    sc = _load(state_dir / "scenario.json", {})
    st = _load(state_dir / "state.json", {"now": 0.0, "next_id": int(sc.get("first_job_id", 1001)), "jobs": {}, "n": {}})
    ctx = {"dir": state_dir, "sc": sc, "st": st, "tool": tool, "args": args, "info": {}}
    handler = _TOOLS.get(tool)
    started = st["now"]
    rc, out, err = handler(ctx) if handler else (127, "", f"{tool}: not simulated by fx_slurm")
    with open(state_dir / "calls.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"tool": tool, "argv": args, "t": started, "rc": rc, "stdout": out, "stderr": err,
                             "info": ctx["info"]}) + "\n")
    _save(state_dir / "state.json", st)
    sys.stdout.write(out)
    sys.stderr.write(err)
    sys.stdout.flush()
    sys.stderr.flush()
    return rc


def _load(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def _save(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj), encoding="utf-8")
    os.replace(tmp, path)


def _num_match(spec, n: int) -> bool:
    if isinstance(spec, bool):
        return spec
    if isinstance(spec, int):
        return n == spec
    if isinstance(spec, list):
        return n in spec
    if isinstance(spec, dict):
        return spec.get("gte", -10**9) <= n <= spec.get("lte", 10**9)
    return False


def _count(st: dict, key: str) -> int:
    st["n"][key] = st["n"].get(key, 0) + 1
    return st["n"][key]


def _tick(st: dict, seconds: float) -> None:
    st["now"] = st["now"] + seconds


def _outage(sc: dict, st: dict, tool: str) -> dict | None:
    o = sc.get("outage")
    if o and o.get("from_s", 0) <= st["now"] < o.get("until_s", 10**12) and tool in o.get("tools", ["sbatch", "squeue", "scancel", "sacct"]):
        return o
    return None


def _user(sc: dict) -> str:
    return sc.get("user") or os.environ.get("USER") or "tester01"


# sbatch
def _parse_opts(args: list[str]):
    """(options, positional arguments, error text). Long options only need their name; a short one maps to it."""
    opts: dict = {}
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--":
            return opts, args[i + 1:], None
        if a.startswith("--"):
            name, eq, val = a[2:].partition("=")
            if name in _LONG_VALUE:
                if not eq:
                    i += 1
                    if i >= len(args):
                        return None, None, f"sbatch: option '--{name}' requires an argument"
                    val = args[i]
                opts[name] = val
            elif name in _LONG_FLAG:
                if eq and name not in ("exclusive", "parsable"):
                    return None, None, f"sbatch: option '--{name}' doesn't allow an argument"
                opts[name] = val if eq else True
            else:
                return None, None, f"sbatch: unrecognized option '--{name}'"
        elif a.startswith("-") and len(a) > 1:
            name = _SHORT.get(a[1])
            if name is None:
                return None, None, f"sbatch: invalid option -- '{a[1]}'"
            if name in _LONG_VALUE:
                val = a[2:]
                if not val:
                    i += 1
                    if i >= len(args):
                        return None, None, f"sbatch: option requires an argument -- '{a[1]}'"
                    val = args[i]
                opts[name] = val
            else:
                opts[name] = True
        else:
            return opts, args[i:], None
        i += 1
    return opts, [], None


def _directives(text: str):
    """#SBATCH options of a script: they stop at the first line that is neither blank nor a comment, as in Slurm."""
    opts: dict = {}
    for line in text.splitlines()[1:]:
        s = line.strip()
        if not s or (s.startswith("#") and not s.startswith("#SBATCH")):
            continue
        if not s.startswith("#SBATCH"):
            break
        try:
            parsed, rest, err = _parse_opts(shlex.split(s[len("#SBATCH"):], comments=True))
        except ValueError as exc:
            return None, f"sbatch: error: {exc}"
        if err or rest:
            return None, err or f"sbatch: error: unexpected argument in directive: {s}"
        opts.update(parsed)
    return opts, None


def _walltime_s(text: str) -> int | None:
    """Slurm --time to seconds: M, M:S, H:M:S, D-H, D-H:M, D-H:M:S."""
    try:
        days = 0
        if "-" in text:
            d, text = text.split("-", 1)
            days = int(d)
            h, m, s = ([int(p) for p in text.split(":")] + [0, 0])[:3]
        else:
            parts = [int(p) for p in text.split(":")]
            h, m, s = (0, parts[0], 0) if len(parts) == 1 else (0, parts[0], parts[1]) if len(parts) == 2 else tuple(parts[:3])
        return ((days * 24 + h) * 60 + m) * 60 + s
    except ValueError:
        return None


def _array_count(spec: str | None) -> int:
    if not spec:
        return 0
    total = 0
    for part in spec.split("%")[0].split(","):
        lo, _, hi = part.partition("-")
        hi = hi.partition(":")[0]
        total += (int(hi) - int(lo) + 1) if lo.isdigit() and hi.isdigit() else 1
    return total


def _stage_of(comment: str | None, script: str) -> str:
    if comment and comment.startswith("mpd:") and comment.count(":") >= 3:
        return comment.rsplit(":", 1)[1]
    name = Path(script).name
    return name[:-len(".sbatch")] if name.endswith(".sbatch") else Path(script).stem


def _rule_matches(rule: dict, c: dict) -> bool:
    stage = rule.get("stage")
    if stage is not None and c["stage"] not in ([stage] if isinstance(stage, str) else stage):
        return False
    if "test_only" in rule and bool(rule["test_only"]) != c["test_only"]:
        return False
    if "call" in rule and not _num_match(rule["call"], c["call"]):
        return False
    if "nth" in rule and not _num_match(rule["nth"], c["nth"]):
        return False
    return rule.get("after_s", -1) <= c["now"] < rule.get("until_s", 10**12)


def _new_job(ctx: dict, eff: dict, script: str, stage: str, deps: list[int], hide_calls: int = 0) -> dict:
    st = ctx["st"]
    jid = st["next_id"]
    st["next_id"] = jid + 1
    array = eff.get("array")
    job = {"id": jid, "name": eff.get("job-name", Path(script).name),
           "comment": "" if ctx["sc"].get("sbatch", {}).get("drop_comment") else eff.get("comment", ""),
           "user": _user(ctx["sc"]), "stage": stage, "script": script, "state": "PENDING",
           "reason": "Dependency" if deps else "Priority", "deps": deps,
           "kill_invalid": str(eff.get("kill-on-invalid-dep", "")).lower() in ("yes", "true", "1"),
           "array": array, "n_array": _array_count(array), "partition": eff.get("partition", "gpu"),
           "account": eff.get("account", ""), "gres": eff.get("gres", ""), "time": eff.get("time", ""),
           "cpus": eff.get("cpus-per-task", "1"), "mem": eff.get("mem", ""), "submit_t": st["now"],
           "hide_calls": hide_calls, "elapsed": None, "exitcode": "0:0", "tasks": None}
    st["jobs"][str(jid)] = job
    return job


def _refusal(sc: dict, eff: dict, args: list[str], script_text: str, strict_deps: list[int], jobs: dict) -> str | None:
    """The message real sbatch would refuse the request with, or None."""
    sb = sc.get("sbatch", {})
    for flag, message in sb.get("reject_flags", {}).items():
        if any(a.startswith(flag) for a in args) or any(re.match(rf"#SBATCH\s+{re.escape(flag)}", ln.strip()) for ln in script_text.splitlines()):
            return message
    gres = eff.get("gres", "")
    kinds = sb.get("gres_types")
    if kinds is not None and gres.startswith("gpu:"):
        bits = gres.split(":")
        if len(bits) >= 3 and bits[1] not in kinds:
            return BAD_GRES
    parts = sb.get("partitions")
    if parts is not None and eff.get("partition") is not None:
        name = eff["partition"]
        if name not in parts:
            return BAD_PARTITION.format(name=name)
        limit, secs = parts[name].get("max_time_s"), _walltime_s(eff.get("time", "")) if eff.get("time") else None
        if limit and secs and secs > limit and parts[name].get("exceed", "reject") == "reject":
            return BAD_TIME
    if sb.get("strict_dependencies", True) and any(str(d) not in jobs for d in strict_deps):
        return DEPENDENCY
    return None


def _sbatch(ctx: dict):
    sc, st, args = ctx["sc"], ctx["st"], ctx["args"]
    sb = sc.get("sbatch", {})
    _tick(st, float(sb.get("duration_s", 0)))
    cli, rest, err = _parse_opts(args)
    if err:
        return 1, "", err + "\nTry \"sbatch --help\" for more information\n"
    if not rest:
        return 1, "", "sbatch: error: Batch script is empty or missing; this fake does not read stdin\n"
    script = os.path.abspath(rest[0])
    try:
        raw = Path(script).read_bytes()         # bytes: text mode would turn CRLF into LF and hide the problem
    except OSError:
        return 1, "", f"sbatch: error: Unable to open file {rest[0]}\n"
    text = raw.decode("utf-8", errors="replace")
    if not text.startswith("#!"):
        return 1, "", ("sbatch: error: This does not look like a batch script.  The first\n"
                       "sbatch: error: line must start with #! followed by the path to an interpreter.\n"
                       "sbatch: error: For instance: #!/bin/sh\n")
    if b"\r" in raw:
        return 1, "", ("sbatch: error: Batch script contains DOS line breaks (\\r\\n)\n"
                       "sbatch: error: instead of expected UNIX line breaks (\\n).\n")
    directive, err = _directives(text)
    if err:
        return 1, "", err + "\n"
    eff = {**directive, **cli}
    test_only = bool(eff.get("test-only"))
    deps = [int(x) for x in re.findall(r"\d+", eff.get("dependency", "")) if x.isdigit()] if eff.get("dependency") else []
    comment = eff.get("comment", "")
    stage = _stage_of(comment, script)
    ctx["info"].update(stage=stage, test_only=test_only, directives=directive, comment=comment, deps=deps,
                       name=eff.get("job-name", ""), script=script)
    message = _refusal(sc, eff, args, text, deps, st["jobs"])
    if message:
        ctx["info"]["outcome"] = "refused"
        return 1, "", message + "\n"
    c = {"stage": stage, "test_only": test_only, "now": st["now"], "nth": _count(st, "sbatch_test" if test_only else "sbatch"),
         "call": _count(st, f"sbatch_test:{stage}" if test_only else f"sbatch:{stage}")}
    rule = next((r for r in sb.get("rules", []) if _rule_matches(r, c)), None)
    do = (rule or {}).get("do", "accept")
    out_window = _outage(sc, st, "sbatch")
    if rule is None and out_window:
        do = "accept_but_fail" if out_window.get("accept") and not test_only else "fail"
        rule = {"message": out_window.get("message", CONTROLLER_DOWN)}
    message = (rule or {}).get("message", CONTROLLER_DOWN)
    if do == "fail":
        ctx["info"]["outcome"] = "fail"
        return int(rule.get("exit", 1)), "", message + "\n"
    if test_only:
        ctx["info"]["outcome"] = "test_only"
        part = eff.get("partition", "gpu")
        return 0, "", f"sbatch: Job {st['next_id']} to start at 2026-11-09T12:00:00 using {eff.get('cpus-per-task', 1)} processors on nodes fake001 in partition {part}\n"
    job = _new_job(ctx, eff, script, stage, deps, int(rule.get("hide_squeue_calls", 0)) if rule else 0)
    parts = sc.get("sbatch", {}).get("partitions", {}).get(job["partition"])
    if parts and job["time"] and parts.get("max_time_s") and (_walltime_s(job["time"]) or 0) > parts["max_time_s"]:
        job["reason"] = "PartitionTimeLimit"
    if out_window and do == "accept_but_fail":
        job["hidden_until_s"] = out_window.get("until_s", 10**12)       # the controller shows it again once it is back
    ctx["info"].update(job_id=job["id"], outcome=do)
    if do == "accept_but_fail":
        return int(rule.get("exit", 1)), "", message + "\n"
    out = f"{job['id']}\n" if eff.get("parsable") else f"Submitted batch job {job['id']}\n"
    return 0, out, (JOB_SUBMIT_BANNER + "\n") if sb.get("job_submit_lines") else ""


# queue and accounting
def _advance(sc: dict, st: dict) -> None:
    """One step of job progress per squeue/sacct call (scenario `progress.enabled`)."""
    prog = sc.get("progress") or {}
    if not prog.get("enabled"):
        return
    jobs = st["jobs"]
    ends = prog.get("end_state", {})
    for key in sorted(jobs, key=int):
        j = jobs[key]
        if j["state"] in TERMINAL:
            continue
        dep_states = [jobs[str(d)]["state"] for d in j["deps"] if str(d) in jobs]
        if j["state"] == "PENDING":
            if any(s in TERMINAL and s != "COMPLETED" for s in dep_states):
                j["reason"] = "DependencyNeverSatisfied"
                if j["kill_invalid"]:
                    j["state"] = "CANCELLED"
            elif all(s == "COMPLETED" for s in dep_states):
                j["state"], j["reason"] = "RUNNING", "None"
        elif j["state"] == "RUNNING":
            j["state"] = ends.get(j["stage"], ends.get("default", "COMPLETED"))
            j["exitcode"] = "0:0" if j["state"] == "COMPLETED" else "1:0"


def _fails(spec, n: int) -> bool:
    return bool(spec) and (_num_match(spec, n) if not isinstance(spec, bool) else spec)


def _gate(ctx: dict, tool: str, down_message: str):
    """Common failure injection for squeue/sacct/scancel: (rc, out, err) when the call must fail, else None."""
    sc, st = ctx["sc"], ctx["st"]
    n = _count(st, tool)
    spec = sc.get(tool, {})
    if _fails(spec.get("fail"), n):
        return 1, "", spec.get("message", down_message) + "\n"
    o = _outage(sc, st, tool)
    if o:
        return 1, "", o.get("message", down_message).replace("sbatch: error: Batch job submission failed: ", f"{tool}: error: ") + "\n"
    return None


def _task_rows(job: dict) -> list[tuple[str, str]]:
    """(job id as sacct prints it, state) for the rows of one job. An array is several rows; a pending one is a range."""
    n, jid, state = job["n_array"], str(job["id"]), job["state"]
    if n <= 1 and not job.get("tasks"):
        return [(jid, state)]
    tasks = job.get("tasks")
    if tasks:
        return [(f"{jid}_{i}", s) for i, s in enumerate(tasks)]
    if state == "PENDING":
        return [(f"{jid}_[0-{n - 1}]", "PENDING")]
    if state == "RUNNING":
        rows = [(f"{jid}_0", "RUNNING"), (f"{jid}_1", "RUNNING")][:n]
        return rows + ([(f"{jid}_[2-{n - 1}]", "PENDING")] if n > 2 else [])
    return [(f"{jid}_{i}", state) for i in range(n)]


def _squeue_fmt(fmt: str, job: dict, row_state: str, st: dict) -> str:
    pending = row_state == "PENDING"

    def one(m):
        c = m.group("c")
        value = {"A": str(job["id"]), "F": str(job["id"]), "i": str(job["id"]) if job["n_array"] <= 1 else f"{job['id']}_[0-{job['n_array'] - 1}]",
                 "j": job["name"], "k": job["comment"] or "(null)", "T": row_state, "t": _STATE_CODES.get(row_state, row_state[:2]),
                 "r": job["reason"] if pending else "None", "R": f"({job['reason']})" if pending else "fakenode001",
                 "M": job["elapsed"] or _ELAPSED.get(row_state, "00:30:00"), "u": job["user"], "P": job["partition"],
                 "a": job["account"], "D": "1", "C": str(job["cpus"]), "N": "" if pending else "fakenode001",
                 "V": "2026-11-09T11:00:00", "S": "N/A" if pending else "2026-11-09T11:05:00"}.get(c, "")
        width = int(m.group("w") or 0)
        return value.rjust(width) if m.group("dot") else value.ljust(width)

    return re.sub(r"%(?P<dot>\.)?(?P<w>\d+)?(?P<c>[A-Za-z])", one, fmt)


def _squeue(ctx: dict):
    sc, st = ctx["sc"], ctx["st"]
    failed = _gate(ctx, "squeue", SQUEUE_DOWN)
    if failed:
        return failed
    opts = {"user": None, "format": "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R", "states": None, "jobs": None, "noheader": False, "name": None}
    args, i = ctx["args"], 0
    while i < len(args):
        a = args[i]
        key, eq, val = a.partition("=")
        names = {"-u": "user", "--user": "user", "-o": "format", "--format": "format", "-t": "states", "--states": "states",
                 "-j": "jobs", "--jobs": "jobs", "-n": "name", "--name": "name"}
        if key in names:
            if not eq:
                i += 1
                val = args[i] if i < len(args) else ""
            opts[names[key]] = val
        elif a in ("-h", "--noheader"):
            opts["noheader"] = True
        elif a == "--me":
            opts["user"] = _user(sc)
        elif a in ("-a", "--all", "--noconvert", "-r", "--array") or a.startswith("--sort") or a == "-S":
            i += a == "-S"
        else:
            return 1, "", f"squeue: unrecognized option '{a}'\nTry \"squeue --help\" for more information\n"
        i += 1
    _advance(sc, st)
    wanted = {s.strip().upper() for s in (opts["states"] or "").split(",") if s.strip()}
    wanted = {_CODE_STATES.get(s, s) for s in wanted}
    ids = {x for x in (opts["jobs"] or "").split(",") if x}
    sq = sc.get("squeue", {})
    out = [] if opts["noheader"] else [_squeue_fmt(opts["format"], {"id": "JOBID", "name": "NAME", "comment": "COMMENT", "user": "USER", "partition": "PARTITION", "account": "ACCOUNT", "n_array": 0, "reason": "REASON", "elapsed": "TIME", "cpus": "CPUS"}, "STATE", st)]
    seen = set()
    for key in sorted(st["jobs"], key=int):
        job = st["jobs"][key]
        hidden = job.get("hide_calls", 0) > 0 or (job.get("hidden_until_s") is not None and st["now"] < job["hidden_until_s"])
        if job.get("hide_calls", 0) > 0:
            job["hide_calls"] -= 1
        if hidden or (opts["user"] and job["user"] != opts["user"]) or (opts["name"] and job["name"] != opts["name"]):
            continue
        if ids and key not in ids:
            continue
        if job["state"] in TERMINAL and sq.get("hide_finished"):
            continue
        rows = _task_rows(job)
        if job["state"] in TERMINAL:
            rows = rows[:1] if len(rows) > 1 and not job.get("tasks") else rows
        for _, state in rows:
            if (wanted and "ALL" not in wanted and state not in wanted) or (not wanted and state in TERMINAL):
                continue
            seen.add(key)
            out.append(_squeue_fmt(opts["format"], job, state, st))
    if ids and not (ids & seen) and not wanted:
        return 1, "", "slurm_load_jobs error: Invalid job id specified\n"
    return 0, "".join(line + "\n" for line in out), ""


def _elapsed_of(job: dict, state: str) -> str:
    return job["elapsed"] or _ELAPSED.get(state, "00:30:00")


def _sacct(ctx: dict):
    sc, st = ctx["sc"], ctx["st"]
    failed = _gate(ctx, "sacct", SACCT_DOWN)
    if failed:
        return failed
    fields, ids, noheader, parsable = ["JobID", "JobName", "State", "Elapsed", "ExitCode"], set(), False, False
    args, i = ctx["args"], 0
    while i < len(args):
        a = args[i]
        key, eq, val = a.partition("=")
        if key in ("-j", "--jobs", "--format", "-o", "-S", "--starttime"):
            if not eq:
                i += 1
                val = args[i] if i < len(args) else ""
            if key in ("-j", "--jobs"):
                ids = {x for x in val.split(",") if x}
            elif key in ("--format", "-o"):
                fields = [re.sub(r"%-?\d+$", "", f).strip() for f in val.split(",") if f.strip()]
        elif a in ("-n", "--noheader"):
            noheader = True
        elif a in ("-P", "--parsable2"):
            parsable = True
        elif a in ("-X", "--allocations", "-p", "--parsable"):
            parsable = parsable or a in ("-p", "--parsable")
        else:
            return 1, "", f"sacct: unrecognized option '{a}'\n"
        i += 1
    if st["n"].get("sacct", 0) <= int(sc.get("sacct", {}).get("lag_calls", 0)):
        return 0, "", ""
    _advance(sc, st)
    rows = []
    for key in sorted(st["jobs"], key=int):
        job = st["jobs"][key]
        if job.get("hide_calls", 0) > 0 or (job.get("hidden_until_s") is not None and st["now"] < job["hidden_until_s"]):
            continue
        if ids and key not in ids:
            continue
        for jid, state in _task_rows(job):
            values = {"jobid": jid, "jobidraw": jid, "jobname": job["name"], "state": state, "elapsed": _elapsed_of(job, state),
                      "exitcode": job["exitcode"] if state in TERMINAL else "0:0", "account": job["account"] or "123456789012",
                      "partition": job["partition"], "alloccpus": str(job["cpus"]), "reqmem": job["mem"], "user": job["user"],
                      "submit": "2026-11-09T11:00:00", "comment": job["comment"]}
            rows.append("|".join(values.get(f.lower(), "") for f in fields))
    head = [] if noheader else ["|".join(fields)]
    return 0, "".join(line + "\n" for line in head + rows), ""


def _scancel(ctx: dict):
    st = ctx["st"]
    failed = _gate(ctx, "scancel", SCANCEL_DOWN)
    if failed:
        return failed
    errors, cancelled = [], []
    args = ctx["args"]
    targets, i = [], 0
    while i < len(args):
        a = args[i]
        if a in ("-u", "--user") and i + 1 < len(args):
            targets += [k for k, j in st["jobs"].items() if j["user"] == args[i + 1]]
            i += 1
        elif a.startswith("-"):
            pass
        else:
            targets.append(a.split("_")[0])
        i += 1
    for key in targets:
        job = st["jobs"].get(key)
        if job is None:
            errors.append(f"scancel: error: Kill job error on job id {key}: Invalid job id specified")
        elif job["state"] in TERMINAL:
            errors.append(f"scancel: error: Kill job error on job id {key}: Job/step already completed")
        else:
            job["state"], job["reason"], cancelled = "CANCELLED", "None", cancelled + [key]
    ctx["info"]["cancelled"] = cancelled
    return (1 if errors else 0), "", "".join(e + "\n" for e in errors)


# the rest
def _scontrol(ctx: dict):
    sc, st, args = ctx["sc"], ctx["st"], ctx["args"]
    config = {"SLURM_VERSION": "25.05.6 (fake)", "MaxArraySize": "1001", "MaxJobCount": "10000", "ClusterName": "fakecluster",
              **sc.get("scontrol", {}).get("config", {})}
    if args[:2] == ["show", "config"]:
        return 0, "Configuration data as of 2026-11-09T11:00:00\n" + "".join(f"{k:<24}= {v}\n" for k, v in config.items()), ""
    if args[:2] == ["show", "job"] and len(args) > 2 and args[2] in st["jobs"]:
        j = st["jobs"][args[2]]
        return 0, f"JobId={j['id']} JobName={j['name']}\n   JobState={j['state']} Reason={j['reason']} Dependency={'afterok:' + ':'.join(map(str, j['deps'])) if j['deps'] else '(null)'}\n   Comment={j['comment']}\n", ""
    if args[:1] == ["update"]:
        return 0, "", ""
    return 1, "", "scontrol: error: invalid entity or not simulated by fx_slurm\n"


_MYPROJECT = """\
Project accounts of <netid> (invented layout: the real output of myproject on Grace is not known)
 Account        Default  Allocation  Used&Pending  Balance
 123456789012   Y        20000.00    1200.00       18800.00
"""
_SHOWQUOTA = """\
Disk quotas of <netid> (invented layout: the real output of showquota on Grace is not known)
 /home/<netid>       used 1.2 GB of 10 GB    files 3,100 of 10,000
 $SCRATCH            used 40 GB of 1 TB     files 52,000 of 250,000
"""


def _canned(name: str, default: str):
    def run(ctx: dict):
        spec = ctx["sc"].get(name, {})
        return int(spec.get("exit", 0)), spec.get("stdout", default), spec.get("stderr", "")
    return run


def _nvidia_smi(ctx: dict):
    if ctx["sc"].get("nvidia_smi", {}).get("fail"):
        return 9, "", "NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver.\n"
    return 0, "GPU 0: NVIDIA A100-PCIE-40GB (UUID: GPU-00000000-fake)\n", ""


def _module(ctx: dict):
    """`module` as a program. It cannot change its caller's environment; the BASH_ENV function (see bash_env) can."""
    args = ctx["args"]
    avail = ctx["sc"].get("module", {}).get("available", [])
    if args[:1] in (["load"], ["add"]):
        bad = [m for m in args[1:] if m not in avail and not any(a.startswith(m + "/") for a in avail)]
        if bad:
            return 1, "", f"Lmod has detected the following error:  The following module(s) are unknown: {' '.join(repr(b) for b in bad)}\n"
    return 0, "", ""


def _tripwire(ctx: dict):
    ctx["info"]["tripwire"] = True
    return 99, "", f"fx_slurm: {ctx['tool']} is not simulated and must not be called from a test\n"


_TOOLS = {"sbatch": _sbatch, "squeue": _squeue, "scancel": _scancel, "sacct": _sacct, "scontrol": _scontrol,
          "module": _module, "myproject": _canned("myproject", _MYPROJECT), "showquota": _canned("showquota", _SHOWQUOTA),
          "nvidia-smi": _nvidia_smi, "srun": _tripwire, "salloc": _tripwire, "sacctmgr": _tripwire}

if __name__ == "__main__":
    sys.exit(_shim_main(sys.argv))

# Part 2, the pytest side (never run by the fake binaries).
import atexit                                   # noqa: E402
import shutil                                   # noqa: E402
import tempfile                                 # noqa: E402
from dataclasses import dataclass, field        # noqa: E402

import pytest                                   # noqa: E402

_BASH_ENV = r"""# fake Lmod for tests: a shell function, so purge/load change $LOADEDMODULES in the calling script
module() {
  local sub="${1:-}" m
  [ "$#" -gt 0 ] && shift
  { printf '%s' "$sub"; for m in "$@"; do printf '\037%s' "$m"; done; printf '\n'; } >> "@DIR@/module.log"
  case "$sub" in
    purge) export LOADEDMODULES="" ;;
    load|add)
      for m in "$@"; do
        if ! grep -qxF -- "$m" "@DIR@/modules.txt" 2>/dev/null && ! grep -q -- "^$m/" "@DIR@/modules.txt" 2>/dev/null; then
          printf 'Lmod has detected the following error:  The following module(s) are unknown: "%s"\n' "$m" >&2
          return 1
        fi
        case ":${LOADEDMODULES:-}:" in *":$m:"*) ;; *) export LOADEDMODULES="${LOADEDMODULES:+$LOADEDMODULES:}$m" ;; esac
      done ;;
    list) printf 'Currently Loaded Modules: %s\n' "${LOADEDMODULES:-}" >&2 ;;
  esac
  return 0
}
export -f module
"""
_DEFAULT_MODULES = ["Anaconda3/2024.02-1", "GCC/11.3.0", "GCCcore/13.2.0", "Python/3.11.5", "CUDA/12.8.0", "WebProxy"]
_TOOL_NAMES = ("sbatch", "squeue", "scancel", "sacct", "scontrol", "module", "myproject", "showquota", "nvidia-smi",
               "srun", "salloc", "sacctmgr")


@dataclass(frozen=True)
class Call:
    tool: str
    argv: list
    rc: int
    stdout: str
    stderr: str
    t: float
    info: dict

    @property
    def test_only(self) -> bool:
        return "--test-only" in self.argv or bool(self.info.get("test_only"))

    def has(self, text: str) -> bool:
        return any(text in a for a in self.argv)


_SLURM_CLIENTS = {"sbatch", "squeue", "scancel", "sacct", "scontrol", "sinfo", "srun", "salloc", "sacctmgr", "sprio", "sshare",
                  "sstat", "sreport", "sattach", "scrontab", "sdiag", "sview"}
_SHADOWS: dict[tuple, str] = {}


def _shadow_dir(directory: str, hide: frozenset) -> str:
    """`directory` itself, or - when it holds a command in `hide` - a directory of symlinks to everything else in it. A real
    Slurm client further down PATH must not answer when a test removes its fake (or calls a client that is not faked): the
    real cluster is off limits."""
    key = (directory, hide)
    if key not in _SHADOWS:
        names = os.listdir(directory)
        if hide.isdisjoint(names):
            _SHADOWS[key] = directory
        else:
            shadow = tempfile.mkdtemp(prefix="fx_slurm_path_")
            atexit.register(shutil.rmtree, shadow, ignore_errors=True)
            for name in names:
                if name not in hide:
                    try:
                        os.symlink(os.path.join(directory, name), os.path.join(shadow, name))
                    except OSError:
                        pass
            _SHADOWS[key] = shadow
    return _SHADOWS[key]


def clean_path(original: str, hide=()) -> str:
    """`original` PATH without the real Slurm clients and without the commands in `hide`."""
    hide = frozenset(hide) | _SLURM_CLIENTS
    out = []
    for directory in original.split(os.pathsep):
        if directory and os.path.isdir(directory):
            try:
                out.append(_shadow_dir(directory, hide))
            except OSError:
                out.append(directory)
    return os.pathsep.join(out)


class FakeSlurm:
    def __init__(self, root: Path, user: str = "tester01", monkeypatch=None):
        self.dir = Path(root)
        self._mp = monkeypatch
        self._original_path = os.environ.get("PATH", "")
        self.hidden: set[str] = set()
        self.bin = self.dir / "bin"
        self.user = user
        self.sleeps: list[float] = []
        self.bash_env = self.dir / "bash_env.sh"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.bin.mkdir(exist_ok=True)
        (self.dir / "calls.jsonl").write_text("", encoding="utf-8")
        self._write_scenario({"user": user})
        self._write_state({"now": 0.0, "next_id": 1001, "jobs": {}, "n": {}})
        for name in _TOOL_NAMES:
            self._write_shim(name)
        self.bash_env.write_text(_BASH_ENV.replace("@DIR@", str(self.dir)), encoding="utf-8")
        self.set_modules(_DEFAULT_MODULES)

    # -- binaries and environment ---------------------------------------------------------------------------------
    def remove(self, tool: str) -> None:
        """Make `tool` vanish from PATH (a real one further down PATH is hidden too)."""
        (self.bin / tool).unlink(missing_ok=True)
        self.hidden.add(tool)
        self._apply_path()

    def _write_shim(self, tool: str) -> None:
        path = self.bin / tool
        path.write_text(f'#!/bin/sh\nexec "{sys.executable}" -I -S "{Path(__file__).resolve()}" "{self.dir}" {tool} "$@"\n', encoding="utf-8")
        path.chmod(0o755)

    def restore(self, tool: str) -> None:
        self.hidden.discard(tool)
        self._write_shim(tool)
        self._apply_path()

    @property
    def path(self) -> str:
        return f"{self.bin}{os.pathsep}{clean_path(self._original_path, self.hidden)}"

    def _apply_path(self) -> None:
        if self._mp is not None:
            self._mp.setenv("PATH", self.path)

    def env(self, extra: dict | None = None) -> dict:
        env = {**os.environ, "PATH": self.path, "BASH_ENV": str(self.bash_env),
               "USER": self.user, "LOGNAME": self.user, "LOADEDMODULES": ""}
        env.update(extra or {})
        return env

    def set_modules(self, names) -> None:
        (self.dir / "modules.txt").write_text("".join(f"{n}\n" for n in names), encoding="utf-8")
        self.configure(module={"available": list(names)})

    # -- scenario -------------------------------------------------------------------------------------------------
    def scenario(self) -> dict:
        return json.loads((self.dir / "scenario.json").read_text(encoding="utf-8"))

    def _write_scenario(self, sc: dict) -> None:
        (self.dir / "scenario.json").write_text(json.dumps(sc), encoding="utf-8")

    def configure(self, **keys) -> "FakeSlurm":
        self._write_scenario({**self.scenario(), **keys})
        return self

    def rule(self, **rule) -> "FakeSlurm":
        sc = self.scenario()
        sc.setdefault("sbatch", {}).setdefault("rules", []).append(rule)
        self._write_scenario(sc)
        return self

    def fail_sbatch(self, n: int | None = None, *, stage=None, message: str = CONTROLLER_DOWN) -> "FakeSlurm":
        """Fail the first n real sbatch calls (of `stage`, else of every stage counted together); n=None: always."""
        rule = {"do": "fail", "message": message, "test_only": False}
        if stage is not None:
            rule["stage"] = stage
        if n is not None:
            rule["call" if stage is not None else "nth"] = {"lte": n}
        return self.rule(**rule)

    def accept_but_fail(self, stage, *, times: int = 1, message: str = CONTROLLER_DOWN, hide_squeue_calls: int = 0) -> "FakeSlurm":
        return self.rule(stage=stage, call={"lte": times}, do="accept_but_fail", message=message, hide_squeue_calls=hide_squeue_calls,
                         test_only=False)

    def reject_stage(self, stage, message: str) -> "FakeSlurm":
        return self.rule(stage=stage, do="fail", message=message)

    def outage(self, until_s: float | None = None, *, accept: bool = False, from_s: float = 0.0,
               tools=("sbatch", "squeue", "scancel", "sacct"), message: str | None = None) -> "FakeSlurm":
        o = {"from_s": from_s, "tools": list(tools), "accept": accept}
        if until_s is not None:
            o["until_s"] = until_s
        if message:
            o["message"] = message
        return self.configure(outage=o)

    def squeue_fails(self, calls=True) -> "FakeSlurm":
        return self.configure(squeue={**self.scenario().get("squeue", {}), "fail": calls})

    def scancel_fails(self, calls=True) -> "FakeSlurm":
        return self.configure(scancel={"fail": calls})

    def sacct_fails(self, calls=True) -> "FakeSlurm":
        return self.configure(sacct={**self.scenario().get("sacct", {}), "fail": calls})

    # -- clock ----------------------------------------------------------------------------------------------------
    def clock(self) -> float:
        return self._state()["now"]

    def advance(self, seconds: float) -> None:
        st = self._state()
        st["now"] += seconds
        self._write_state(st)

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.advance(seconds)

    # -- state and log --------------------------------------------------------------------------------------------
    def _state(self) -> dict:
        return json.loads((self.dir / "state.json").read_text(encoding="utf-8"))

    def _write_state(self, st: dict) -> None:
        (self.dir / "state.json").write_text(json.dumps(st), encoding="utf-8")

    def calls(self, tool: str | None = None) -> list[Call]:
        out = []
        for line in (self.dir / "calls.jsonl").read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            if tool is None or d["tool"] == tool:
                out.append(Call(**d))
        out.extend(c for c in self._module_calls() if tool in (None, "module"))
        return out

    def _module_calls(self) -> list[Call]:
        path = self.dir / "module.log"
        if not path.exists():
            return []
        return [Call("module", ln.split("\037"), 0, "", "", 0.0, {}) for ln in path.read_text(encoding="utf-8").splitlines()]

    def test_only_calls(self) -> list[Call]:
        return [c for c in self.calls("sbatch") if c.test_only]

    def submitted(self) -> list[dict]:
        """Real (not --test-only) sbatch calls that queued a job, in order: the job record, plus `argv` of the call."""
        out = []
        for c in self.calls("sbatch"):
            if c.info.get("job_id") and not c.test_only:
                out.append({**self.jobs()[str(c.info["job_id"])], "argv": c.argv, "rc": c.rc})
        return out

    def jobs(self) -> dict[str, dict]:
        return self._state()["jobs"]

    def job(self, which) -> dict | None:
        jobs = self.jobs()
        if str(which) in jobs:
            return jobs[str(which)]
        matches = [j for j in jobs.values() if j["stage"] == which]
        return matches[-1] if matches else None

    def alive_ids(self) -> list[str]:
        return [k for k, j in self.jobs().items() if j["state"] not in TERMINAL]

    def state_of(self, which) -> str:
        return self.job(which)["state"]

    def set_state(self, which, state: str, *, reason: str | None = None, exitcode: str | None = None,
                  elapsed: str | None = None, tasks: list | None = None) -> None:
        st = self._state()
        key = str(self.job(which)["id"])
        j = st["jobs"][key]
        j["state"] = state
        if reason is not None:
            j["reason"] = reason
        if exitcode is not None:
            j["exitcode"] = exitcode
        if elapsed is not None:
            j["elapsed"] = elapsed
        if tasks is not None:
            j["tasks"] = tasks
        self._write_state(st)

    def add_job(self, *, name: str = "other", comment: str = "", user: str | None = None, state: str = "RUNNING", stage: str = "",
                reason: str = "None", deps=(), array: str | None = None, hide_calls: int = 0) -> str:
        """A job that no sbatch call of this test queued (another user's, or a leftover of an earlier submission)."""
        st = self._state()
        jid = st["next_id"]
        st["next_id"] = jid + 1
        st["jobs"][str(jid)] = {"id": jid, "name": name, "comment": comment, "user": user or self.user, "stage": stage or name,
                                "script": "", "state": state, "reason": reason, "deps": list(deps), "kill_invalid": False,
                                "array": array, "n_array": _array_count(array), "partition": "gpu", "account": "", "gres": "", "time": "",
                                "cpus": "1", "mem": "", "submit_t": st["now"], "hide_calls": hide_calls, "elapsed": None,
                                "exitcode": "0:0", "tasks": None}
        self._write_state(st)
        return str(jid)

    def assert_no_tripwire(self) -> None:
        hit = [c for c in self.calls() if c.info.get("tripwire")]
        assert not hit, f"a test called a tool that is never simulated: {[c.tool for c in hit]}"

    def clear_tripwires(self) -> None:
        path = self.dir / "calls.jsonl"
        keep = [ln for ln in path.read_text(encoding="utf-8").splitlines() if not json.loads(ln)["info"].get("tripwire")]
        path.write_text("".join(ln + "\n" for ln in keep), encoding="utf-8")


@pytest.fixture
def fake_slurm(tmp_path, monkeypatch):
    fs = FakeSlurm(tmp_path / "fake_slurm", monkeypatch=monkeypatch)
    monkeypatch.setenv("PATH", fs.path)
    monkeypatch.setenv("BASH_ENV", str(fs.bash_env))
    monkeypatch.setenv("USER", fs.user)
    monkeypatch.setenv("LOGNAME", fs.user)
    monkeypatch.setenv("LOADEDMODULES", "")         # the real login shell's modules must not leak into the fake Lmod
    yield fs
    fs.assert_no_tripwire()


# A stand-in plan, so submit/campaign tests do not depend on the planner.
@dataclass
class StubStage:
    name: str
    part: int = 0
    kind: str = "cpu"
    module: str = "mpd.stages.stub"
    gpu: str | None = None
    gpus: int = 0
    cpus: int = 4
    mem_gb: int = 16
    walltime_s: int = 3600
    tasks: list = field(default_factory=lambda: [{}])
    task_offset: int = 0
    launcher: str = "python"
    depends_on: list = field(default_factory=list)
    su_estimate: float = 10.0
    notes: list = field(default_factory=list)

    @property
    def full_name(self) -> str:
        return f"{self.name}.{self.part}" if self.part else self.name

    @property
    def is_array(self) -> bool:
        return len(self.tasks) > 1


@dataclass
class StubSpec:
    campaign: str

    def campaign_id(self) -> str:
        return self.campaign

    def to_json(self) -> str:
        return json.dumps({"campaign": self.campaign}, sort_keys=True)


@dataclass
class StubCluster:
    name: str = "grace"


@dataclass
class StubOptions:
    cluster: StubCluster
    home: Path
    account: str | None = "123456789012"


@dataclass
class StubPlan:
    spec: StubSpec
    options: StubOptions
    campaign_dir: Path
    stages: list
    resolved: dict
    warnings: list
    nonce: str
    campaign_id: str

    @property
    def total_su(self) -> float:
        return sum(s.su_estimate for s in self.stages)


def stub_plan(tmp_path, *, campaign_id: str = "esm2-3b__esm2-650m__proteingym-score-all__seed1__ab12cd34", nonce: str = "0a1b2c3d4e5f",
              units=("all",), datasets=("proteingym",), shards: int = 2, teacher: bool = True, cluster: str = "grace",
              score_chunks: int = 1, minimal: bool = False) -> StubPlan:
    """prepare -> [teacher] -> train -> score (+ score-base from prepare) -> evaluate, shaped like the real planner's output.
    `score_chunks` > 1 splits the score tasks into stages score, score.1, ...; `minimal` is just prepare -> train -> evaluate."""
    home = Path(tmp_path) / "home"
    gpu = dict(kind="gpu", gpu="a100", gpus=1, cpus=24, mem_gb=180)
    train_tasks = [{"unit": u} for u in units]

    def score_tasks(us):
        return [{"unit": u, "dataset": d, "shard": s, "num_shards": shards} for u in us for d in datasets for s in range(shards)]

    stages = [StubStage("prepare")]
    if minimal:
        stages += [StubStage("train", **gpu, walltime_s=7200, tasks=train_tasks, depends_on=["prepare"]),
                   StubStage("evaluate", depends_on=["train"])]
    else:
        if teacher:
            stages.append(StubStage("teacher", **gpu, depends_on=["prepare"]))
        stages.append(StubStage("train", **gpu, walltime_s=7200, tasks=train_tasks, depends_on=["teacher" if teacher else "prepare"]))
        stages.append(StubStage("score-base", **gpu, tasks=score_tasks(["base"]), depends_on=["prepare"]))
        tasks = score_tasks(units)
        size = -(-len(tasks) // score_chunks)
        parts = [StubStage("score", part=k, **gpu, tasks=tasks[k * size:(k + 1) * size], task_offset=k * size, depends_on=["train"])
                 for k in range(score_chunks)]
        stages += parts
        stages.append(StubStage("evaluate", depends_on=["score-base", *[p.full_name for p in parts]]))
    resolved = {"campaign_id": campaign_id, "teacher": "esm2-3b", "student": "esm2-650m", "score_shards": {d: shards for d in datasets}}
    return StubPlan(StubSpec(campaign_id), StubOptions(StubCluster(cluster), home), home / "runs" / campaign_id, stages, resolved,
                    ["stub warning"], nonce, campaign_id)


def stub_render(plan, stage) -> str:
    """A script shaped like the real renderer's: directives (with the --comment tag), then a trivial body."""
    gres = f"#SBATCH --gres=gpu:{stage.gpu}:{stage.gpus}\n#SBATCH --partition=gpu\n" if stage.gpu else ""
    array = f"#SBATCH --array=0-{len(stage.tasks) - 1}%8\n" if stage.is_array else ""
    out = "%x_%A_%a" if stage.is_array else "%x_%j"
    return (f"#!/bin/bash\n#SBATCH --job-name=mpd-{stage.full_name}-esm2650m\n"
            f"#SBATCH --comment=mpd:{plan.campaign_id}:{plan.nonce}:{stage.full_name}\n"
            f"#SBATCH --time={stage.walltime_s // 3600:02d}:00:00\n#SBATCH --nodes=1\n#SBATCH --ntasks=1\n"
            f"#SBATCH --cpus-per-task={stage.cpus}\n#SBATCH --mem={stage.mem_gb}G\n{gres}{array}"
            f"#SBATCH --output={plan.campaign_dir}/logs/{out}.out\n#SBATCH --error={plan.campaign_dir}/logs/{out}.err\n"
            f"\nset -eo pipefail\necho \"MPD_STAGE_DONE {stage.full_name}\"\n")


def write_stub_campaign(plan, *, state: str | None = None, jobs: dict | None = None, message: str = "") -> dict:
    """Write the campaign files with the real write_campaign and, if `state` is given, a SUBMITTED.json."""
    from mpd.slurm.campaign import write_marker
    from mpd.slurm.submit import write_campaign
    scripts = write_campaign(plan, render=stub_render)
    if state:
        write_marker(plan.campaign_dir, {"campaign_id": plan.campaign_id, "state": state, "submitted_at": "2026-11-09T11:00:00-0600",
                                          "nonce": plan.nonce, "cluster": plan.options.cluster.name, "jobs": dict(jobs or {}),
                                          "message": message})
    return scripts
