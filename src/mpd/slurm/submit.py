"""Write a campaign directory and submit its stages with Slurm dependencies, surviving a flaky controller.

Grace's controller goes away for minutes to hours (four outages in 2026) and on 2026-07-26 it accepted a whole job chain
while sbatch printed "Unable to contact slurm controller". So a failed sbatch is not proof that nothing was queued:
every job carries a per-submission tag in its --comment, the marker SUBMITTED.json is written after each accepted job,
and a retry happens only after `squeue` has answered and shows no job with this submission's tag.
"""

from __future__ import annotations

import json
import re
import secrets
import shlex
import shutil
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from .campaign import (BLOCKING, MARKER, CampaignError, CampaignState, JobState, QueueRow, SubmitError, campaign_state, cancel,
                       create_marker, failure_log, failure_tail, job_states, job_tag, list_campaigns, log_tail, message_of,
                       now_iso, parse_tag, pending_reasons, read_marker, read_queue, resolve_campaign, run_tool, scancel_ids,
                       stage_log, stage_progress, tagged, tasks_finished, write_json_atomic, write_marker, write_text_atomic)

if TYPE_CHECKING:
    from ..checks import Check

__all__ = ["BLOCKING", "CampaignError", "CampaignState", "FORCE_APPLIES", "JobState", "QueueRow", "SubmitError", "TRANSIENT_ERRORS", "campaign_state",
           "cancel", "classify", "failure_log", "failure_tail", "job_states", "list_campaigns", "log_tail",
           "pending_reasons", "preflight_checks", "resolve_campaign", "stage_log", "stage_progress", "submit", "write_campaign"]

# Messages that mean "the controller did not answer", lower case. The job may or may not have been queued.
TRANSIENT_ERRORS = ("unable to contact", "socket timed out", "connection refused", "connection timed out",
                    "zero bytes were transmitted", "slurm_persist_conn", "temporarily unable", "temporarily unavailable",
                    "communication connection failure", "no answer after")
SBATCH_TIMEOUT = 180.0

_HINT_ACCOUNT = ("Slurm did not accept the billing account. `myproject` lists the accounts you may charge: pass one with "
                 "--account (or export MPD_ACCOUNT), or leave it out to use your default account.")
_HINT_BALANCE = ("The account cannot cover this request. `myproject` shows the balance. sbatch's own figure counts CPUs only; the "
                 "plan's figure also includes GPUs. Shrink the plan (fewer --updates or assays, a cheaper --gpu) or ask HPRC for SUs.")
_HINT_TIME = ("The requested walltime is longer than the partition allows. Lower --train-time, --score-time or --teacher-time, or "
              "pick a GPU type with a longer limit (`mpd clusters` lists them).")
_HINT_SHAPE = ("Slurm does not offer this partition or GPU shape. Try another --gpu (`mpd clusters` lists them); `mpd submit --dry-run` "
               "test-submits every stage shape, and `sinfo -o '%P %l %D %G'` shows what exists.")
_HINT_LIMITS = ("Your account or QOS limits (running jobs, submitted jobs, requested resources) refuse this request. Submit fewer "
                "assays at once or lower --array-throttle.")
_HINT_DEPENDENCY = ("A job this stage waits for is unknown to Slurm (long finished, or never queued). `mpd status` shows what is "
                    "queued; add --force to queue what is missing.")
_HINT_FLAG = ("A flag in the generated script (or one given with --sbatch-arg) is not known to this Slurm. Remove it, or report it "
              "with `mpd doctor --report`.")
_HINT_SCRIPT = "Slurm refused the generated script itself. Run the command again; if that does not help, report it with `mpd doctor --report`."
_HINTS = (
    (("balance is not sufficient", "balance is insufficient"), _HINT_BALANCE),
    (("invalid account", "account/partition combination"), _HINT_ACCOUNT),
    (("time limit is invalid", "partitiontimelimit"), _HINT_TIME),
    (("invalid partition", "invalid generic resource", "invalid gres", "node configuration is not available", "invalid qos"),
     _HINT_SHAPE),
    (("unrecognized option", "invalid option", "doesn't allow an argument", "requires an argument"), _HINT_FLAG),
    (("violates accounting", "qosmax", "assocmax", "grptres"), _HINT_LIMITS),
    (("dependency problem",), _HINT_DEPENDENCY),
    (("does not look like a batch script", "dos line breaks", "unable to open file", "batch script is empty"), _HINT_SCRIPT),
)


def classify(message: str) -> str:
    """transient (retry after checking the queue) | permanent (a known refusal: stop) | unknown (stop, do not guess)."""
    low = message.lower()
    if any(t in low for t in TRANSIENT_ERRORS):
        return "transient"
    return "permanent" if any(p in low for patterns, _ in _HINTS for p in patterns) else "unknown"


def _hint(message: str) -> str:
    low = message.lower()
    return next((hint for patterns, hint in _HINTS if any(p in low for p in patterns)), "")


def _rejection(stage: str, message: str) -> str:
    text = f"sbatch rejected stage {stage}: {message}"
    hint = _hint(message)
    if hint:
        return f"{text}\n{hint}"
    return f"{text}\nmpd does not recognise this as a temporary Slurm problem, so it did not retry."


def _gist(text: str) -> str:
    """The first non-empty line without Slurm's "sbatch: error:" noise: enough to name a problem inside a sentence."""
    line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    return re.sub(r"^(?:sbatch|squeue|scancel|sacct): (?:error: )?(?:Batch job submission failed: )?", "", line)


# ---------------------------------------------------------------- plan accessors (an older Plan has no nonce or campaign_id)
def _campaign_id(plan) -> str:
    value = getattr(plan, "campaign_id", None)
    value = value() if callable(value) else value
    return str(value or plan.spec.campaign_id())


def _nonce(plan) -> str:
    value = getattr(plan, "nonce", None)
    if not value:
        value = secrets.token_hex(6)
        try:
            plan.nonce = value          # remember it, so the scripts written now and the jobs submitted later agree
        except AttributeError:
            pass
    return str(value)


def _cluster_name(plan) -> str:
    return str(getattr(getattr(getattr(plan, "options", None), "cluster", None), "name", "") or "")


def _default_render():
    from .render import render_script
    return render_script


def _cannot_write(what: Path | str, exc: Exception) -> SubmitError:
    return SubmitError(f"cannot write {what} ({exc}). Is the disk or your quota full? `mpd doctor` checks free space.")


SCHEDULING_OPTIONS = ("--train-time, --score-time, --teacher-time, --array-throttle, --sbatch-arg, --module or --env "
                      "(except MPD_DTYPE, MPD_TF32 and MPD_STRUCTURE_LEGACY_MASK)")
# Said wherever a second submission is refused: the recovery is the same command plus --force, and the student must know that the
# options they changed for the recovery are applied by it. Kept stages are not: Slurm cannot change a queued job's script.
FORCE_APPLIES = (f"A changed {SCHEDULING_OPTIONS} keeps the campaign (its id does not depend on them), and --force applies it to the "
                 "stages it queues again; a stage that is still queued or running keeps the options it was submitted with. "
                 "A changed --gpu, --train-gpus, --eval-data or --score-shards is a different campaign that trains again from the start.")


def _blocked(old: dict, cid: str) -> str:
    state, when = old.get("state"), old.get("submitted_at") or "earlier"
    jobs = ", ".join(f"{s}={j}" for s, j in (old.get("jobs") or {}).items()) or "none recorded"
    follow = f"`mpd status {cid}` shows where it is, `mpd cancel {cid}` removes its jobs"
    if state == "submitted":
        return (f"{cid} was already submitted ({when}; jobs {jobs}). {follow}. Add --force to queue only the stages whose job "
                f"died or is missing; stages that are queued, running or finished are kept. {FORCE_APPLIES} Nothing was written.")
    if state == "partial":
        return (f"{cid} was only partly submitted ({when}; jobs {jobs}): {old.get('message') or 'a stage could not be queued'}. "
                f"{follow}. Run the same command with --force to continue the chain from the first missing stage. {FORCE_APPLIES} "
                "Nothing was written.")
    if state == "submitting":
        return (f"a submission of {cid} started {when} and did not finish (jobs queued so far: {jobs}). It may still be running in "
                f"another terminal, or it was killed. {follow}. Run the same command with --force to continue it. {FORCE_APPLIES} "
                "Nothing was written.")
    return (f"{cid} has a {MARKER} that cannot be read ({old.get('message') or 'unknown error'}), so mpd cannot tell whether it is "
            f"queued. Check `squeue -u $USER`; delete the file to start over, or run `mpd cancel {cid}`. Nothing was written.")


# ---------------------------------------------------------------- the campaign directory
def _same_json(path: Path, text: str) -> bool:
    try:
        return json.loads(path.read_text(encoding="utf-8")) == json.loads(text)
    except (OSError, ValueError):
        return False


def _what_differs(path: Path, text: str) -> str:
    """How a recipe file on disk differs from the one this command would write, in a few words."""
    try:
        old = json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        return "missing on disk"
    except ValueError:
        return "not valid JSON on disk"
    new = json.loads(text)
    if isinstance(old, dict) and isinstance(new, dict):
        keys = sorted(k for k in {*old, *new} if old.get(k, "<absent>") != new.get(k, "<absent>"))

        def short(value) -> str:
            return json.dumps(value)[:40]
        shown = [f"{k}: {short(old.get(k, '<absent>'))} on disk, {short(new.get(k, '<absent>'))} now" for k in keys[:5]]
        return "; ".join(shown) + (f"; and {len(keys) - 5} more" if len(keys) > 5 else "")
    if isinstance(old, list) and isinstance(new, list):
        return f"{len(old)} tasks on disk, {len(new)} now"
    return "different content"


def write_campaign(plan, *, force: bool = False, render=None) -> dict[str, Path]:
    """Create the campaign directory with spec, resolved recipe, task lists and sbatch scripts; returns the scripts.

    Refuses, writing nothing, while the campaign is submitted, being submitted or half submitted. With ``force`` the
    recipe files are left alone (and must equal the plan's: a campaign is never rewritten once it was submitted, whatever
    became of its jobs) and only the scripts and plan.json are written again."""
    root = Path(plan.campaign_dir)
    cid = _campaign_id(plan)
    old = read_marker(root)
    blocked = old is not None and old["state"] in BLOCKING
    if blocked and not force:
        raise SubmitError(_blocked(old, cid))
    recipe = {"spec.json": plan.spec.to_json() + "\n", "resolved.json": json.dumps(plan.resolved, indent=2, sort_keys=True) + "\n"}
    all_tasks: dict[str, list[dict]] = {}
    for stage in plan.stages:
        all_tasks.setdefault(stage.name, []).extend(stage.tasks)
    recipe.update({f"tasks/{name}.json": json.dumps(tasks, indent=1) + "\n" for name, tasks in all_tasks.items()})
    if old is not None:         # also after `mpd cancel` or a failed submission: its finished stages are kept, so the recipe must be the same
        changed = [rel for rel, text in recipe.items() if not _same_json(root / rel, text)]
        if changed:
            lines = [f"  {rel}: {_what_differs(root / rel, recipe[rel])}" for rel in changed]
            head = (f"{root} already holds a submitted campaign with a different recipe ({', '.join(changed)}), so --force cannot "
                    "continue it:" if blocked else
                    f"{root} holds a campaign that ended as {old['state']} and has a different recipe ({', '.join(changed)}), so it "
                    "cannot be submitted again; its finished stages would be mixed with the new recipe:")
            raise SubmitError(
                head + "\n" + "\n".join(lines)
                + "\nA submitted campaign is never rewritten. Its recipe follows from the options in its id (--eval-data, --gpu, "
                  "--train-gpus, --score-shards, the recipe flags and --seed) and from the mpd version, presets and model registry that "
                  "wrote it, so the usual cause is another mpd version or an edited file. Run the command with the mpd version that "
                  "created the campaign, or change --seed to start a new campaign, or remove the directory (`mpd cancel` its jobs first).")
    render = render or _default_render()
    try:
        return _write_files(plan, root, cid, recipe, old is not None, render)
    except OSError as exc:
        raise _cannot_write(f"the campaign directory {root}", exc) from exc


def _write_files(plan, root: Path, cid: str, recipe: dict[str, str], written: bool, render) -> dict[str, Path]:
    for sub in ("scripts", "tasks", "logs"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    if not written:
        for rel, text in recipe.items():
            write_text_atomic(root / rel, text)
    scripts = {}
    for stage in plan.stages:
        path = root / "scripts" / f"{stage.full_name}.sbatch"
        write_text_atomic(path, render(plan, stage))
        scripts[stage.full_name] = path
    summary = {"schema": 1, "campaign_id": cid, "nonce": _nonce(plan), "cluster": _cluster_name(plan),
               "stages": [{**{k: v for k, v in asdict(s).items() if k != "tasks"}, "n_tasks": len(s.tasks),
                           "partition_limit_s": _partition_limit(plan, s)} for s in plan.stages],
               "estimated_su_upper_bound": plan.total_su, "warnings": plan.warnings}
    write_json_atomic(root / "plan.json", summary)
    return scripts


def _partition_limit(plan, stage) -> int | None:
    """Longest walltime the cluster profile allows for this stage, kept in plan.json so `mpd status` can say how much more a
    TIMEOUT may ask for. None when the plan does not know (a stand-in plan, a profile without the figure)."""
    cluster = getattr(getattr(plan, "options", None), "cluster", None)
    try:
        return int(cluster.gpus[stage.gpu].max_walltime_s if stage.gpu else cluster.cpu_max_walltime_s)
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


# ---------------------------------------------------------------- submitting
def _job_id(proc) -> str | None:
    """The job id sbatch --parsable printed (`123` or `123;cluster`), or None."""
    if proc.returncode != 0:
        return None
    for line in proc.stdout.splitlines():
        m = re.fullmatch(r"(\d+)(?:;.*)?", line.strip())
        if m:
            return m.group(1)
    return None


class _Submission:
    """One call of submit(): the marker, the jobs queued so far and the retry state live here."""

    def __init__(self, plan, scripts, *, force, sbatch, squeue, scancel, retries, wait_s, budget_s, echo, sleep, clock):
        self.plan, self.given, self.force = plan, scripts, force
        self.sbatch, self.squeue, self.scancel = sbatch, squeue, scancel
        self.retries, self.wait_s, self.budget_s = retries, wait_s, budget_s
        # a log file (tmux, nohup) must show progress during a 60 s wait, so print flushes
        self.echo = (lambda line: print(line, flush=True)) if echo is print else echo
        self.sleep, self.clock = sleep, clock
        self.root = Path(plan.campaign_dir)
        self.cid, self.nonce = _campaign_id(plan), _nonce(plan)
        self.jobs: dict[str, str] = {}          # stage -> job id, kept and new: the ones later stages may wait for
        self.finished: dict[str, str] = {}      # stage -> job id of a stage whose output files exist (history only)
        self.kept: dict[str, str] = {}          # stage -> job id of a stage --force found queued or running and left alone
        self.new: dict[str, str] = {}           # queued by this call (the only ones a failure cancels when continuing)
        self.replaced: dict[str, list[str]] = {}
        self.retried: set[str] = set()
        self.deadline: float | None = None      # starts at the first controller error, so a healthy submit never spends it
        self.uncertain: str | None = None       # a stage that may have been accepted although sbatch said it was not
        self.tags_lost = False                  # the cluster does not keep the --comment tag, so such a job cannot be recognised
        self.clean_rejection = False
        self.su_noted = False
        self.resume = False
        self.submitted_at = now_iso()

    # -- marker -------------------------------------------------------------------------------------------------
    def _marker(self, state: str, message: str = "") -> dict:
        data = {"campaign_id": self.cid, "state": state, "submitted_at": self.submitted_at, "nonce": self.nonce,
                "cluster": _cluster_name(self.plan), "jobs": self._all_jobs(), "message": message}
        if self.replaced:
            data["replaced"] = self.replaced
        return data

    def _all_jobs(self) -> dict[str, str]:
        ids = {**self.finished, **self.jobs}
        return {s.full_name: ids[s.full_name] for s in self.plan.stages if s.full_name in ids}

    def _save(self, state: str, message: str = "") -> None:
        try:
            write_marker(self.root, self._marker(state, message))
        except OSError as exc:
            raise _cannot_write(self.root / MARKER, exc) from exc

    # -- start --------------------------------------------------------------------------------------------------
    def _check_marker(self, old: dict | None) -> None:
        if old is None:
            return
        if old["state"] in BLOCKING:
            if not self.force:
                raise SubmitError(_blocked(old, self.cid))
            self.resume = True
            self.submitted_at = old.get("submitted_at") or self.submitted_at
        else:
            # cancelled or failed: none of its jobs is meant to be alive, but stages whose output files exist are not queued again
            # (a finished teacher would wait for a GPU only to print "already computed"), and a survivor is kept, never duplicated
            self.resume = True
            self.echo(f"the earlier submission of this campaign {old['state']}"
                      + (f" ({old['message']})" if old.get("message") else "") + "; submitting it again")

    def _scripts(self) -> dict[str, Path]:
        out = {}
        for st in self.plan.stages:
            path = Path((self.given or {}).get(st.full_name) or self.root / "scripts" / f"{st.full_name}.sbatch")
            if not path.is_file():
                raise SubmitError(f"the script of stage {st.full_name} is missing ({path}). Run `mpd submit` again (without --dry-run) "
                                  "so that the scripts are written first.")
            out[st.full_name] = path
        return out

    def _reconcile(self, old: dict) -> tuple[dict[str, str], set[str], list[str]]:
        """Which stages of an earlier submission to keep. Kept: a job of the stage is queued or running (found by the marker's
        id or by its tag). Done: its output files exist. The rest is queued again, and so is any stage waiting on one of them."""
        rows, why = read_queue(self.squeue)
        if rows is None:
            raise SubmitError(f"cannot continue {self.cid}: squeue did not answer ({why}), so mpd cannot tell which stages are still "
                              "queued. Try again when `squeue -u $USER` answers.")
        recorded = {str(k): str(v) for k, v in (old.get("jobs") or {}).items()}
        self.replaced = {str(k): [str(i) for i in v] for k, v in (old.get("replaced") or {}).items() if isinstance(v, list)}   # earlier rounds stay on record
        alive = {r.job_id for r in rows if r.alive}
        by_stage: dict[str, list[str]] = {}
        for r in tagged(rows, self.cid, alive_only=True):
            by_stage.setdefault(parse_tag(r.comment)[2], []).append(r.job_id)
        keep: dict[str, str] = {}
        done: set[str] = set()
        redo: set[str] = set()
        stale: list[str] = []
        for st in self.plan.stages:
            name = st.full_name
            cands = set(by_stage.get(name, [])) | ({recorded[name]} if recorded.get(name) in alive else set())
            finished = bool(st.tasks) and tasks_finished(self.root, st.name, st.tasks) == len(st.tasks)
            if cands and (finished or not any(d in redo for d in st.depends_on)):
                keep[name] = recorded[name] if recorded.get(name) in cands else min(cands, key=int)
                if len(cands) > 1:
                    self.echo(f"note: jobs {', '.join(sorted(cands - {keep[name]}, key=int))} also carry stage {name}; kept {keep[name]}")
            elif finished:
                done.add(name)
                if recorded.get(name):
                    self.finished[name] = recorded[name]
            else:
                redo.add(name)
                stale += sorted(cands)          # queued, but waiting for a job that is gone
                if recorded.get(name):
                    self.replaced.setdefault(name, []).append(recorded[name])
        return keep, done, stale

    # -- one stage ----------------------------------------------------------------------------------------------
    def _delay(self, cycle: int) -> float | None:
        """Seconds to wait before checking the queue again; None when the retry budget is spent."""
        now = self.clock()
        if self.deadline is None:
            self.deadline = now + self.budget_s
        remaining = self.deadline - now
        if remaining <= 0:
            return None
        return min(self.wait_s * min(1.5 ** (cycle - 1), 4.0), remaining)

    def _note_su(self, stderr: str) -> None:
        text = stderr.strip()
        if not text:
            return
        self.echo(text)         # Grace's job_submit plugin prints the pre-charge here
        if "Requested SUs" in text and not self.su_noted:
            self.su_noted = True
            self.echo("note: sbatch's 'Requested SUs' counts CPUs only; the plan's figure also includes GPUs. `myproject` shows your balance.")

    def _submit_stage(self, stage, script: Path) -> None:
        name = stage.full_name
        deps = [self.jobs[d] for d in stage.depends_on if d in self.jobs]
        cmd = [self.sbatch, "--parsable", f"--comment={job_tag(self.cid, self.nonce, name)}"]
        if deps:
            cmd += ["--dependency=afterok:" + ":".join(deps), "--kill-on-invalid-dep=yes"]
        cmd.append(str(script))
        cycles = 0
        started = self.clock()
        while True:
            proc = run_tool(cmd, timeout=SBATCH_TIMEOUT)
            jid = _job_id(proc)
            if jid:
                self._note_su(proc.stderr)
                return self._accepted(name, jid, retried=cycles > 0)
            message = message_of(proc) if proc.returncode else "sbatch exited 0 but printed no job id"
            kind = "transient" if proc.returncode == 0 else classify(message)
            if kind != "transient":
                self.clean_rejection = True
                raise SubmitError(_rejection(name, message))
            # The controller did not answer, so the job may be queued anyway: wait, look, and only then try again.
            self.uncertain = name
            hits: list[QueueRow] | None = None
            while hits is None:
                delay = self._delay(cycles + 1) if cycles < self.retries else None
                if delay is None:
                    hits = self._last_look(name, cycles, message, started)
                    break
                cycles += 1
                self.echo(f"Slurm did not answer for stage {name} ({_gist(message)}); waiting {delay:.0f} s, then checking the "
                          f"queue before trying again (try {cycles} of {self.retries})")
                self.sleep(delay)
                rows, why = read_queue(self.squeue)
                if rows is None:
                    self.echo(f"the queue cannot be read yet ({_gist(why)}); not submitting again until it can")
                    continue
                if not self._tags_survive(rows):
                    self.echo("squeue shows a job of this submission without its --comment tag, so a job that sbatch accepted while "
                              "reporting an error could not be recognised; not submitting again blind")
                    continue
                self.uncertain = None
                seen = tagged(rows, self.cid, nonce=self.nonce, stage=name)
                hits = self._usable(seen)
                if seen and not hits:
                    self.echo(f"a job of stage {name} was queued but has already ended ({seen[0].state}); queueing it again")
            if hits:
                ids = sorted({r.job_id for r in hits}, key=int)
                self.echo(f"stage {name} was queued after all (job {ids[0]}): the controller accepted it while reporting an error")
                return self._accepted(name, ids[0], retried=True)
            self.echo(f"no job of this submission is queued for stage {name}; trying again")

    def _tags_survive(self, rows: list[QueueRow]) -> bool:
        """False when a job this call queued shows up in the queue without its tag. Then the cluster rewrites job comments,
        looking for a tag proves nothing, and the only safe thing left is to wait."""
        by_id = {r.job_id: r for r in rows}
        ok = all(by_id[jid].comment == job_tag(self.cid, self.nonce, stage)
                 for stage, jid in self.new.items() if jid in by_id and by_id[jid].alive)
        self.tags_lost = self.tags_lost or not ok
        return ok

    @staticmethod
    def _usable(rows: list[QueueRow]) -> list[QueueRow]:
        """Jobs a chain can depend on: queued, running, or finished successfully."""
        return [r for r in rows if r.alive or r.state == "COMPLETED"]

    def _last_look(self, name: str, cycles: int, message: str, started: float) -> list[QueueRow]:
        """Retries are used up: one more look without waiting, then give up with an honest account of what is known."""
        rows, _ = read_queue(self.squeue)
        if rows is not None and self._tags_survive(rows):
            hits = self._usable(tagged(rows, self.cid, nonce=self.nonce, stage=name))
            if hits:
                return hits
            self.uncertain = None
        text = (f"sbatch could not queue stage {name}: Slurm did not answer after {cycles + 1} attempt(s) over "
                f"{self.clock() - started:.0f} s (last message: {_gist(message)}).")
        raise SubmitError(text + (" The job may have been accepted anyway; the queue could not be read to check." if self.uncertain else ""))

    def _accepted(self, name: str, jid: str, *, retried: bool) -> None:
        if self.uncertain == name:
            self.uncertain = None
        if retried:
            self.retried.add(name)
            jid = self._dedupe(name, jid)
        self.jobs[name] = self.new[name] = jid
        self._save("submitting")
        self.echo(f"submitted {name:<12} job {jid}")

    def _dedupe(self, name: str, jid: str) -> str:
        """A lost reply can leave an earlier copy of the stage queued. Nothing depends on either yet: keep the first."""
        rows, _ = read_queue(self.squeue)
        if rows is None:
            return jid
        ids = {r.job_id for r in self._usable(tagged(rows, self.cid, nonce=self.nonce, stage=name))} | {jid}
        if len(ids) == 1:
            return jid
        keep = min(ids, key=int)
        extras = sorted(ids - {keep}, key=int)
        ok, text = scancel_ids(extras, self.scancel)
        self.echo(f"stage {name} was queued twice; kept job {keep} and " + (f"cancelled {', '.join(extras)}" if ok else
                  f"could not cancel {', '.join(extras)} ({_gist(text)}): run `scancel {' '.join(extras)}`"))
        return keep

    def _sweep(self) -> None:
        """After the chain is queued: a reply that arrived late may have left a copy of a retried stage. Cancel the extras, never
        a job something already waits for."""
        rows, why = read_queue(self.squeue)
        if rows is None:
            self.echo(f"note: the queue could not be re-checked for duplicate jobs ({_gist(why)}); `mpd status {self.cid}` lists them")
            return
        for name in sorted(self.retried):
            extras = sorted({r.job_id for r in tagged(rows, self.cid, nonce=self.nonce, stage=name, alive_only=True)} - {self.jobs[name]}, key=int)
            if extras:
                ok, text = scancel_ids(extras, self.scancel)
                self.echo(f"stage {name} was queued twice; kept job {self.jobs[name]} and " + (f"cancelled {', '.join(extras)}" if ok else
                          f"could not cancel {', '.join(extras)}: run `scancel {' '.join(extras)}`"))

    # -- failure ------------------------------------------------------------------------------------------------
    def _abort(self, exc: BaseException) -> str:
        """Cancel what this call queued and record the truth in the marker. Returns the text to show the user."""
        # first of all, so that the ids are on disk even if the cleanup is cut short
        unsaved = self._try_save("partial", f"interrupted while submitting ({type(exc).__name__}); cleaning up")
        targets = list(self.new.values())
        queue_known = True
        if not self.clean_rejection or self.retried:      # a retried stage may have left a copy nobody has seen yet
            rows, _ = read_queue(self.squeue)
            queue_known = rows is not None
            if queue_known and self._tags_survive(rows):
                self.uncertain = None       # the sweep below finds and cancels a job that was accepted unnoticed
            known = set(self.jobs.values())
            targets += [r.job_id for r in tagged(rows or [], self.cid, nonce=self.nonce, alive_only=True) if r.job_id not in known]
        targets = list(dict.fromkeys(targets))
        ok, text = scancel_ids(targets, self.scancel)
        kept = {s: j for s, j in self.jobs.items() if s not in self.new}
        lines = []
        if not targets:
            lines.append("Nothing from this submission is queued.")
        elif ok:
            lines.append("Cancelled the jobs already queued by this submission: " + ", ".join(targets) + ".")
        else:
            line = f"scancel {' '.join(targets)}"
            lines.append(f"Could not cancel the jobs already queued ({_gist(text)}). Run `{line}` when Slurm answers, or `mpd cancel "
                         f"{self.cid}`; their ids are saved in {self.root / MARKER}.")
        if self.uncertain and self.tags_lost:
            lines.append(f"Stage {self.uncertain} may have been accepted although sbatch reported an error, and this cluster does not keep the "
                         "--comment tag mpd uses to recognise its jobs. Look for it in `squeue -u $USER` and cancel it with scancel.")
        elif self.uncertain:
            lines.append(f"Stage {self.uncertain} may have been accepted although sbatch reported an error (the queue could not be read "
                         f"to check). `mpd cancel {self.cid}` finds and cancels it by its tag once Slurm answers.")
        if kept:
            lines.append("Jobs from the earlier submission are still queued: " + ", ".join(f"{s}={j}" for s, j in kept.items()) + ".")
        partial = bool(kept) or not ok or not queue_known or bool(self.uncertain)
        lines.append(f"Run the same command with --force to continue from the first missing stage, or `mpd cancel {self.cid}`." if partial
                     else "Run the same command again to start over.")
        reason = "interrupted" if isinstance(exc, KeyboardInterrupt) else (_gist(str(exc)) or type(exc).__name__)
        # the message is what `mpd status` shows later
        unsaved = self._try_save("partial" if partial else "failed", f"{reason} -- " + " ".join(lines[:-1])) or unsaved
        if unsaved:
            lines.append(f"The marker could not be written ({unsaved}). Jobs queued by this run: "
                         + (", ".join(f"{s}={j}" for s, j in self._all_jobs().items()) or "none") + ".")
        return "\n".join(lines)

    def _try_save(self, state: str, message: str) -> str | None:
        """Save during cleanup: a failure is reported, never raised, so the cancel still happens."""
        try:
            self._save(state, message)
        except SubmitError as exc:
            return str(exc)
        return None

    # -- run ----------------------------------------------------------------------------------------------------
    def run(self) -> dict[str, str]:
        old = read_marker(self.root)
        self._check_marker(old)
        if shutil.which(self.sbatch) is None:
            raise SubmitError(f"{self.sbatch} was not found. Submit from a Slurm login node; `mpd submit --dry-run` writes and checks the "
                              "scripts without calling it.")
        scripts = self._scripts()
        keep, done, stale = self._reconcile(old) if self.resume else ({}, set(), [])
        self.jobs, self.kept = dict(keep), dict(keep)
        if old is None:
            try:
                created = create_marker(self.root, self._marker("submitting"))
            except OSError as exc:
                raise _cannot_write(self.root / MARKER, exc) from exc
            if not created:
                raise SubmitError(_blocked(read_marker(self.root) or {"state": "submitting"}, self.cid))
        else:
            self._save("submitting")
        try:
            if stale:
                ok, text = scancel_ids(stale, self.scancel)
                self.echo(("cancelled " if ok else "could not cancel ") + f"queued jobs that waited for a job that is gone: {', '.join(stale)}"
                          + ("" if ok else f" ({_gist(text)})"))
            for stage in self.plan.stages:
                name = stage.full_name
                if name in done:
                    self.echo(f"finished  {name:<12} (its output files exist)")
                elif name in keep:
                    self.echo(f"kept      {name:<12} job {keep[name]} (already queued or running)")
                else:
                    self._submit_stage(stage, scripts[name])
            if self.retried:
                self._sweep()
        except BaseException as exc:
            summary = self._abort(exc)
            if isinstance(exc, SubmitError):
                raise SubmitError(f"{exc}\n{summary}") from None
            self.echo(summary)
            raise
        try:
            self._save("submitted")
        except SubmitError as exc:       # every job is queued and was recorded as it was accepted: only the final label is missing
            raise SubmitError(f"{exc}\nAll jobs are queued, nothing was cancelled: " + ", ".join(f"{s}={j}" for s, j in self._all_jobs().items())) from None
        self._announce()
        return self._all_jobs()

    def _announce(self) -> None:
        if not self.new:
            self.echo(f"nothing to submit: every stage of {self.cid} is queued, running or finished")
        else:
            self.echo(f"\nsubmitted {len(self.new)} job(s) for {self.cid}")
            if self.kept:       # Slurm cannot change the script of a queued job, so a changed walltime or node option is not applied to it
                self.echo(f"note: kept stages ({', '.join(self.kept)}) stay queued or running with the options they were submitted with; "
                          "a changed walltime, --sbatch-arg or --module applies to the stages queued now. To change a job that is still "
                          "waiting, `mpd cancel` the campaign and submit it again")
        for what, cmd in (("follow it", f"mpd status {self.cid}"), ("read logs", f"mpd logs {self.cid} [STAGE] [--err]"),
                          ("cancel", f"mpd cancel {self.cid}"), ("results", f"mpd results {self.cid}   (after the evaluate stage)")):
            self.echo(f"  {what:<10} {cmd}")


def submit(plan, scripts: dict[str, Path] | None = None, *, force: bool = False, sbatch: str = "sbatch", squeue: str = "squeue",
           scancel: str = "scancel", retries: int = 6, wait_s: float = 60.0, budget_s: float = 900.0, echo=print,
           sleep=time.sleep, clock=time.monotonic) -> dict[str, str]:
    """Submit the stages in order with afterok dependencies; returns stage -> job id.

    SUBMITTED.json is written before the first job and after every accepted one. If a stage cannot be queued, the jobs this call
    queued are cancelled (Ctrl-C included) and the marker says `failed`, or `partial` when something may still be queued. A
    controller error is retried after `wait_s` seconds (growing, at most `retries` times, `budget_s` seconds in all) but only
    once `squeue` answers and shows no job carrying this submission's tag. A marker in state submitting, submitted or partial
    refuses a second submission unless ``force``, which then keeps the stages that are queued or finished and queues the rest.
    After `mpd cancel` or a failed submission no ``force`` is needed, and the stages that finished are kept the same way."""
    return _Submission(plan, scripts, force=force, sbatch=sbatch, squeue=squeue, scancel=scancel, retries=retries, wait_s=wait_s,
                       budget_s=budget_s, echo=echo, sleep=sleep, clock=clock).run()


# ---------------------------------------------------------------- sbatch --test-only
_SHORT = {"J": "job-name", "t": "time", "p": "partition", "A": "account", "a": "array", "c": "cpus-per-task", "N": "nodes",
          "n": "ntasks", "q": "qos", "C": "constraint", "x": "exclude", "d": "dependency", "o": "output", "e": "error"}
_NOT_SHAPE = {"job-name", "comment", "output", "error", "open-mode", "dependency", "kill-on-invalid-dep"}
_CULPRIT = (("generic resource", ("gres", "gpus")), ("gres", ("gres", "gpus")), ("time limit", ("time",)), ("account", ("account",)),
            ("partition", ("partition", "gres")), ("memory", ("mem",)), ("qos", ("qos",)), ("array", ("array",)),
            ("node configuration", ("gres", "partition", "cpus-per-task", "mem", "nodes", "ntasks")))


def _directive_lines(text: str) -> list[tuple[str, str, str]]:
    """(option, value, raw line) for each #SBATCH option; the options stop at the first command, as in Slurm."""
    out = []
    for line in text.splitlines()[1:]:
        s = line.strip()
        if s and not s.startswith("#"):
            break
        if not s.startswith("#SBATCH"):
            continue
        try:
            tokens = shlex.split(s[len("#SBATCH"):], comments=True)
        except ValueError:
            continue
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok.startswith("--"):
                name, _, val = tok[2:].partition("=")
            elif tok.startswith("-") and len(tok) > 1:
                name, val = _SHORT.get(tok[1], tok[1]), tok[2:]
                if not val and i + 1 < len(tokens):
                    i += 1
                    val = tokens[i]
            else:
                i += 1
                continue
            out.append((name, val, s))
            i += 1
    return out


def _shape(text: str) -> tuple[tuple, list[str], int]:
    """(hashable shape, the #SBATCH lines that define it, array size). Two scripts of one shape are accepted or refused alike."""
    items, lines, size = [], [], 0
    for name, val, raw in _directive_lines(text):
        if name == "array":
            items.append(("array", "yes"))
            m = re.match(r"(\d+)-(\d+)", val)
            size = (int(m.group(2)) - int(m.group(1)) + 1) if m else 1
        elif name not in _NOT_SHAPE:
            items.append((name, val))
        if name not in _NOT_SHAPE and raw not in lines:
            lines.append(raw)
    return tuple(sorted(items)), lines, size


def _without_dependency(text: str) -> str:
    return "\n".join(ln for ln in text.splitlines() if not re.match(r"#SBATCH\s+(--dependency|-d\b|--kill-on-invalid-dep)", ln.strip())) + "\n"


def _describe(entries: list[tuple[str, str, str]]) -> str:
    """'gpu:a100:1 on partition gpu, 02:00:00, 24 CPUs, 180G, array' from the directives (the account is left out on purpose)."""
    opts = {name: val for name, val, _ in entries}
    where = (opts.get("gres") or "no GPU requested") + (f" on partition {opts['partition']}" if opts.get("partition") else "")
    parts = [where, opts.get("time", ""), f"{opts['cpus-per-task']} CPUs" if opts.get("cpus-per-task") else "", opts.get("mem", ""),
             "array" if "array" in opts else ""]
    return ", ".join(p for p in parts if p)


def _culprits(message: str, entries: list[tuple[str, str, str]]) -> list[str]:
    low = message.lower()
    wanted = {opt for key, opts in _CULPRIT if key in low for opt in opts}
    lines = [raw for name, _, raw in entries if name in wanted]
    return list(dict.fromkeys(lines))


def preflight_checks(plan, scripts: dict[str, Path] | None = None, *, sbatch: str = "sbatch", timeout_s: float = 60.0,
                     budget_s: float = 180.0, clock=time.monotonic, render=None) -> list[Check]:
    """`sbatch --test-only` once per distinct job shape, never queueing anything. Slurm answers whether it would accept the
    gres, partition, time, memory and CPU request, which is what a typo in --gpu or an over-long --train-time breaks."""
    from ..checks import Check
    name = "sbatch --test-only"
    if shutil.which(sbatch) is None:
        return [Check(name, "skip", f"{sbatch} is not on PATH (not a Slurm login node): the generated scripts were not test-submitted")]
    render = render or _default_render()
    shapes: dict[tuple, dict] = {}
    for stage in plan.stages:
        path = (scripts or {}).get(stage.full_name)
        text = Path(path).read_text(encoding="utf-8") if path and Path(path).is_file() else render(plan, stage)
        key, lines, size = _shape(text)
        entry = shapes.setdefault(key, {"stages": [], "text": text, "lines": lines, "size": -1, "entries": _directive_lines(text)})
        entry["stages"].append(stage.full_name)
        if size > entry["size"]:        # the largest array of a shape stands for all of them
            entry["text"], entry["size"] = text, size
    checks: list[Check] = []
    started = clock()
    unreachable = False
    for entry in shapes.values():
        label = f"{name} ({', '.join(entry['stages'])})"
        if unreachable or clock() - started > budget_s:
            checks.append(Check(label, "skip", "not tested: " + ("the Slurm controller did not answer" if unreachable else
                                                                  f"the {budget_s:.0f} s time budget for test submissions was used up")))
            continue
        with tempfile.TemporaryDirectory(prefix="mpd-testonly-") as tmp:
            copy = Path(tmp) / "stage.sbatch"
            copy.write_text(_without_dependency(entry["text"]), encoding="utf-8")
            proc = run_tool([sbatch, "--test-only", str(copy)], timeout=timeout_s)
        message = message_of(proc)
        if proc.returncode == 0:
            checks.append(Check(label, "ok", "Slurm would accept " + _describe(entry["entries"])))
        elif classify(message) == "transient":
            unreachable = True
            checks.append(Check(label, "skip", f"the Slurm controller did not answer ({_gist(message)}); the scripts were not tested",
                                "run `mpd doctor` again when `squeue -u $USER` answers"))
        else:
            shown = _culprits(message, entry["entries"]) or entry["lines"]
            checks.append(Check(label, "error", f"sbatch --test-only refused these lines: {'; '.join(shown)}  Slurm said: {message}",
                                _hint(message) or None))
    return checks
