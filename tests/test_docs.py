"""The documentation against the code: README.md and docs/*.md.

What is checked, and why each check exists:

* every `mpd`, `python -m mpd` and `bash scripts/setup_env.sh` line in a shell block parses with the real argparse (variables set earlier in the same
  block, such as `$TEACHER`, are expanded first);
* every relative link and heading anchor resolves, every table row has the width of its header, and the files follow the house style;
* no file holds the maintainer's identity, a token, or an account-like number (the patterns are assembled from pieces, so this file passes the same scan);
* no page calls a model `planned` or `not connected` while the registry says `available`, and no page says that a walltime, `--sbatch-arg` or `--module`
  change starts a new campaign (it does not: the campaign id leaves scheduling options out);
* what the pages paste as program output is the output of the program: the plan table with the notes under it, the doctor rows, `mpd clusters`, the rendered job script, the
  sentences `mpd status` prints for a failed stage;
* the messages quoted in TROUBLESHOOTING.md exist in the code (up to the `<placeholders>`), so the page cannot keep a message the code no longer prints;
* the Python snippet that RESULTS.md shows for using a trained student runs on a tiny model and reproduces `mpd`'s own logits.

Nothing here needs a GPU, Slurm or the network. The snippet test needs torch, transformers and peft and skips without them.
"""

from __future__ import annotations

import ast
import contextlib
import io
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from fx_cli import cli_env, run_cli, write_record, write_proteingym  # noqa: F401
from fx_render import clean_mpd_env, make  # noqa: F401

from mpd import cli, registry

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"
DOCS = [README, *sorted((REPO / "docs").glob("*.md"))]
IDS = [p.relative_to(REPO).as_posix() for p in DOCS]


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def doc(name: str) -> str:
    return read(REPO / name)


# -- reading Markdown ----------------------------------------------------------------------------------------------------------------

OPEN = re.compile(r"^(?P<indent> *)(?P<fence>```+|~~~+) *(?P<lang>[\w+-]*) *$")


def split_markdown(text: str) -> tuple[list[tuple[int, str]], list[tuple[str, int, list[str]]]]:
    """(prose lines as (number, text), fenced blocks as (language, first line number, lines without the indent of the fence))."""
    prose, blocks, current = [], [], None
    for no, line in enumerate(text.splitlines(), 1):
        if current is None:
            m = OPEN.match(line)
            if m:
                current = (m["fence"], m["lang"].lower(), no, len(m["indent"]), [])
            else:
                prose.append((no, line))
            continue
        fence, lang, start, indent, lines = current
        if re.fullmatch(r" *" + re.escape(fence[0]) + "{" + str(len(fence)) + r",} *", line):
            blocks.append((lang, start, lines))
            current = None
        else:
            lines.append(line[indent:] if line[:indent].strip() == "" else line)
    assert current is None, "an unclosed code fence"
    return prose, blocks


def outside_code(line: str) -> str:
    return re.sub(r"`[^`\n]*`", "", line)


# -- commands ------------------------------------------------------------------------------------------------------------------------

SHELL = {"", "bash", "sh", "shell", "console"}
ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=")


def expand(line: str, env: dict[str, str]) -> str:
    return re.sub(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?", lambda m: env.get(m[1], m[0]), line)


def commands(text: str):
    """(line number, argv) of each mpd / setup_env.sh command line in the shell blocks of a page."""
    _, blocks = split_markdown(text)
    for lang, start, lines in blocks:
        if lang not in SHELL:
            continue
        env: dict[str, str] = {}
        pending, first = "", 0
        for offset, raw in enumerate(lines, 1):
            line = raw.strip()
            if pending:
                line = pending + " " + line
            else:
                first = start + offset
            if line.endswith("\\"):
                pending = line[:-1].rstrip()
                continue
            pending = ""
            if line.startswith("$ "):
                line = line[2:]
            try:
                argv = shlex.split(expand(line, env), comments=True)
            except ValueError as exc:
                pytest.fail(f"line {first}: cannot split {line!r}: {exc}")
            body = argv[1:] if argv[:1] == ["export"] else argv
            if body and all(ASSIGNMENT.match(a) for a in body):
                env.update(a.split("=", 1) for a in body)
                continue
            for i, word in enumerate(argv):                      # stop at a pipe, a list or a redirection
                if word in {"|", "&&", "||", ";", ">", ">>", "2>&1"}:
                    argv = argv[:i]
                    break
            while argv and ASSIGNMENT.match(argv[0]):                  # `PYTHONPATH=src python scripts/...`
                argv = argv[1:]
            if argv[:3] == ["python", "-m", "mpd"]:
                argv = ["mpd", *argv[3:]]
            if (argv[:1] == ["mpd"] or argv[:2] == ["bash", "scripts/setup_env.sh"]
                    or (argv[:1] == ["python"] and len(argv) > 1 and argv[1].startswith("scripts/maintainers/"))):
                yield first, argv


def problem_with(argv: list[str]) -> str | None:
    if argv[0] == "python":                                     # a maintainers' script: its flags are checked by running it with --help
        done = subprocess.run([sys.executable, str(REPO / argv[1]), *argv[2:], "--help"], capture_output=True, text=True, cwd=REPO, timeout=120,
                              env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(REPO / "src"), "PYTHONDONTWRITEBYTECODE": "1"})
        if done.returncode != 0 and "ModuleNotFoundError" in done.stderr:
            return None                                          # the script needs a package this machine lacks (torch): not checkable here
        return None if done.returncode == 0 else f"{argv[1]} rejects it: {done.stderr.strip().splitlines()[-1][:160] if done.stderr.strip() else done.returncode}"
    if argv[0] == "bash":
        done = subprocess.run(["bash", str(REPO / "scripts" / "setup_env.sh"), *argv[2:], "--help"], capture_output=True, text=True,
                              cwd=REPO, timeout=60)
        return None if done.returncode == 0 else f"setup_env.sh rejects it: {(done.stderr or done.stdout).strip()[:160]}"
    if argv[1:3] == ["fetch", "foldseek"] and "--from-dir" in argv:
        return "`mpd fetch foldseek` takes --from-file, not --from-dir"
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err), contextlib.redirect_stdout(io.StringIO()):
            cli._parser().parse_args(argv[1:])
    except SystemExit as exc:
        if exc.code not in (0, None):
            return f"does not parse: {err.getvalue().strip().splitlines()[-1] if err.getvalue().strip() else exc.code}"
    return None


def test_every_command_in_the_docs_parses_with_the_real_argparse():
    found, problems = 0, []
    for path in DOCS:
        for no, argv in commands(read(path)):
            found += 1
            why = problem_with(argv)
            if why:
                problems.append(f"{path.relative_to(REPO)}:{no}: {' '.join(argv)}\n    {why}")
    assert found >= 40, f"only {found} command lines found: the extraction is broken"
    assert not problems, "\n".join(problems)


def test_the_command_checker_catches_what_it_should(tmp_path):
    page = "```bash\nTEACHER=esm2-3b\nmpd plan --teacher $TEACHER --nonsense\nmpd fetch foldseek --from-dir x\nmpd doctor --report  # fine\n```\n"
    seen = {" ".join(argv): problem_with(argv) for _, argv in commands(page)}
    assert seen["mpd plan --teacher esm2-3b --nonsense"] and seen["mpd fetch foldseek --from-dir x"]
    assert seen["mpd doctor --report"] is None
    assert not list(commands("```text\n$ mpd plan --nonsense\n```\n"))                  # transcripts are not commands


def test_the_readme_has_the_one_line_command_and_the_setup_before_it_is_needed():
    text = doc("README.md")
    _, blocks = split_markdown(text)
    bash = [b for b in blocks if b[0] == "bash"]
    lines = [ln for _, _, body in bash for ln in body]
    submit = [ln for ln in lines if ln.startswith("mpd submit")]
    assert len(submit) == 1 and not submit[0].endswith("\\"), "the headline command is one line"
    argv = next(a for _, a in commands("```bash\n" + submit[0] + "\n```\n"))
    for flag in ("--cluster", "--account", "--teacher", "--student", "--method", "--train-data", "--eval-data"):
        assert flag in argv, flag
    assert text.index("## Run a distillation") < text.index("## Set up once") < text.index("## What happens after you submit")
    setup = next(body for _, _, body in bash if any(ln.startswith("bash scripts/setup_env.sh") for ln in body))
    seen = [" ".join(a[:4]) for _, a in commands("```bash\n" + "\n".join(setup) + "\n```\n")]
    for need in ("bash scripts/setup_env.sh", "mpd fetch proteingym", "mpd fetch venusmuthub", "mpd fetch model", "mpd doctor"):
        assert any(s.startswith(need) for s in seen), need
    assert "--yes" in next(a for _, a in commands("```bash\n" + "\n".join(setup) + "\n```\n") if a[1:2] == ["scripts/setup_env.sh"])
    follow = "\n".join(next(body for _, _, body in bash if any(ln.startswith("mpd status") for ln in body)))
    for need in ("mpd status", "mpd logs CAMPAIGN train", "mpd results CAMPAIGN", "mpd cancel CAMPAIGN"):
        assert need in follow, need


def test_the_readme_is_a_guide_for_students_not_a_status_report():
    text = doc("README.md")
    headings = [m[1] for line in text.splitlines() if (m := HEADING.match(line))]
    assert not [h for h in headings if re.search(r"status|development|roadmap", h, re.I)], headings
    assert not re.search(r"\baido\b|alpha|never run|not verified|tested on cpu", text, re.I)


# -- links, tables, style ------------------------------------------------------------------------------------------------------------

LINK = re.compile(r"(?<!!)\[[^\]\n]*\]\(([^)\s]+)\)")
HEADING = re.compile(r"^ {0,3}#{1,6} +(.*?) *#* *$")


def slug(heading: str) -> str:
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading)
    text = re.sub(r"[`*~]", "", text).strip().lower()
    return re.sub(r"[^\w\- ]", "", text).replace(" ", "-")


def anchors(text: str) -> set[str]:
    seen: dict[str, int] = {}
    out = set()
    for _, line in split_markdown(text)[0]:
        m = HEADING.match(line)
        if m:
            s = slug(m[1])
            n = seen.get(s, 0)
            seen[s] = n + 1
            out.add(s if n == 0 else f"{s}-{n}")
    return out


def broken_links(path: Path) -> list[str]:
    bad = []
    for no, line in split_markdown(read(path))[0]:
        for target in LINK.findall(re.sub(r"`[^`\n]*`", "", line)):
            if re.match(r"[A-Za-z][A-Za-z0-9+.-]*:", target):
                continue
            where, _, fragment = target.partition("#")
            dest = (path.parent / where).resolve() if where else path
            if not dest.exists():
                bad.append(f"{path.name}:{no}: {target} does not exist")
            elif fragment and dest.suffix == ".md" and fragment not in anchors(read(dest)):
                bad.append(f"{path.name}:{no}: {target} has no such heading")
    return bad


def test_relative_links_and_anchors_resolve():
    problems = [p for path in DOCS for p in broken_links(path)]
    assert not problems, "\n".join(problems)


def test_the_link_checker_sees_a_missing_file_and_a_missing_heading(tmp_path):
    (tmp_path / "a.md").write_text("# A\n\n## Two words\n")
    (tmp_path / "b.md").write_text("[ok](a.md#two-words) [no file](c.md) [no heading](a.md#nothing) `[code](gone.md)`\n")
    assert [p.split(": ", 1)[1] for p in broken_links(tmp_path / "b.md")] == ["c.md does not exist", "a.md#nothing has no such heading"]


def test_the_readme_links_every_page_of_docs():
    linked = set(LINK.findall(doc("README.md")))
    for path in (REPO / "docs").glob("*.md"):
        assert any(t.split("#")[0] == f"docs/{path.name}" for t in linked), f"README.md does not link docs/{path.name}"


def cells(line: str) -> int:
    body = line.strip()[1:]
    body = body[:-1] if body.endswith("|") and not body.endswith("\\|") else body
    return len(re.split(r"(?<!\\)\|", body))


def test_tables_are_well_formed():
    """GitHub splits a row at every unescaped pipe, even inside a code span: every row needs the header's width and a separator under the header."""
    problems, tables = [], 0
    for path in DOCS:
        width, row = None, 0
        for no, line in split_markdown(read(path))[0] + [(10**6, "")]:
            if line.startswith("|"):
                if width is None:
                    width, row = cells(line), 0
                    tables += 1
                elif row == 1 and not re.fullmatch(r"\|(?: *:?-+:? *\|)+", line.strip()):
                    problems.append(f"{path.name}:{no}: no separator row under the header")
                elif cells(line) != width:
                    problems.append(f"{path.name}:{no}: {cells(line)} cells, the header has {width}")
                row += 1
            else:
                width = None
    assert tables >= 40
    assert not problems, "\n".join(problems)


BANNED_WORDS = re.compile(r"\b(simply|just|easily|seamless|seamlessly|robust)\b", re.I)
EMOJI = re.compile("[\U0001F300-\U0001FAFF☀-➿⭐✅❌]")


def test_the_house_style():
    problems = []
    for path in DOCS:
        raw = path.read_bytes()
        text = raw.decode("utf-8")
        name = path.relative_to(REPO)
        if b"\r" in raw:
            problems.append(f"{name}: CR characters")
        if not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
            problems.append(f"{name}: must end with exactly one newline")
        for no, line in enumerate(text.splitlines(), 1):
            if "\t" in line:
                problems.append(f"{name}:{no}: tab")
            if line != line.rstrip():
                problems.append(f"{name}:{no}: trailing whitespace")
            if EMOJI.search(line):
                problems.append(f"{name}:{no}: emoji")
        for no, line in split_markdown(text)[0]:
            plain = outside_code(line)
            if BANNED_WORDS.search(plain):
                problems.append(f"{name}:{no}: {BANNED_WORDS.search(plain)[0]!r}")
            if "!" in plain:
                problems.append(f"{name}:{no}: exclamation mark")
            if re.search(r"\b(works|will work|worked) on Grace\b", plain, re.I):
                problems.append(f"{name}:{no}: nothing has been verified on Grace; say what was tested")
    assert not problems, "\n".join(problems)


# -- identity ------------------------------------------------------------------------------------------------------------------------

_PIECES = ["yi" + "hong", "tamu" + r"\.edu", "prj" + "-02", "mol" + "-cell", "Protein" + "OPD", "/scratch/" + "project", "vs" + "ts",
           "gh" + "p_", "github" + "_pat_", "BEGIN (OPENSSH|RSA|EC) " + "PRIVATE"]
FORBIDDEN = re.compile("|".join(_PIECES))
LONG_NUMBER = re.compile(r"(?<![0-9A-Za-z_])[0-9]{12,13}(?![0-9A-Za-z_])")
PLACEHOLDER_NUMBER = re.compile(r"0{12,13}|9{12,13}|1234567890123?")


def identity_problems(text: str) -> list[str]:
    found = []
    for no, line in enumerate(text.splitlines(), 1):
        public = re.sub(r"[A-Za-z0-9.-]*hprc\.tamu\.edu", "", line)           # the public HPRC hosts are fine
        if FORBIDDEN.search(public):
            found.append(f"{no}: forbidden text {FORBIDDEN.search(public)[0]!r}")
        for number in LONG_NUMBER.findall(line):
            if not PLACEHOLDER_NUMBER.fullmatch(number):
                found.append(f"{no}: the number {number} looks like an account")
    return found


def test_no_identity_secret_or_account_number_in_the_docs():
    problems = [f"{p.relative_to(REPO)}:{m}" for p in DOCS for m in identity_problems(read(p))]
    assert not problems, "\n".join(problems)


def test_the_identity_scan_catches_what_it_should():
    bad = "\n".join(["my" + " netid is yi" + "hong79", "user@tamu" + ".edu", "/scratch/" + "project/x", "account 98765" + "4321098", "gh" + "p_abc"])
    ok = "ssh <netid>@grace.hprc.tamu.edu and 123456789012 and 999999999999 and 1234567890123"
    assert len(identity_problems(bad)) == 5 and identity_problems(ok) == []


# -- statements that the code can contradict -----------------------------------------------------------------------------------------

def sentences(text: str):
    """(line number, sentence) from the prose of a page. A table row is one unit (its cells joined), so that a model in the first
    cell and a claim about it in the last are read together; elsewhere the sentences of a line."""
    for no, line in split_markdown(text)[0]:
        if line.startswith("|"):
            yield no, " | ".join(c.strip() for c in re.split(r"(?<!\\)\|", line) if c.strip())
            continue
        for sentence in re.split(r"(?<=[.;:?])\s+(?=[A-Z`*(])", line.strip()):
            if sentence:
                yield no, sentence


FAMILIES = {"esm2": "ESM2", "saprot": "SaProt"}


def stale_status_claims(text: str) -> list[str]:
    available = [m for m in registry.MODELS.values() if m.status == "available"]
    names = sorted({m.key for m in available} | {FAMILIES[m.family] for m in available if m.family in FAMILIES}, key=len, reverse=True)
    pattern = re.compile("|".join(map(re.escape, names)))
    return [f"{no}: {s[:110]}" for no, s in sentences(text)
            if pattern.search(s) and re.search(r"\bplanned\b|not (yet )?connected|cannot run them", s)]


def test_no_page_calls_an_available_model_planned_or_not_connected():
    assert {m.key for m in registry.MODELS.values() if m.status == "available"} >= {"esm2-3b", "saprot-35m", "saprot-650m"}
    problems = [f"{p.relative_to(REPO)}:{c}" for p in DOCS for c in stale_status_claims(read(p))]
    assert not problems, "\n".join(problems)
    assert stale_status_claims("SaProt is `planned` for now.\n") and stale_status_claims("| SaProt | not connected yet |\n")
    assert not stale_status_claims("AIDO is `planned`; SaProt is available.\n".replace("; SaProt is available", ""))


SCHEDULING = re.compile(r"walltime|--train-time|--score-time|--teacher-time|--sbatch-arg|--module\b|--array-throttle|--exclude")
NEW_CAMPAIGN = re.compile(r"new campaign|different campaign|another campaign|new directory|new folder|trains? (again )?from (update 0|the start)", re.I)
NEGATION = re.compile(r"\bnot\b|\bnever\b|\bkeeps?\b|\bsame\b|\bno\b|without|instead|n't", re.I)


def wrong_identity_claims(text: str) -> list[str]:
    found = []
    for no, sentence in sentences(text):
        for clause in re.split(r"\bbut\b|\bwhile\b|\bwhereas\b|;|\. ", sentence):
            if SCHEDULING.search(clause) and NEW_CAMPAIGN.search(clause) and not NEGATION.search(clause):
                found.append(f"{no}: {clause.strip()[:120]}")
    return found


def test_no_page_says_that_a_scheduling_option_starts_a_new_campaign():
    problems = [f"{p.relative_to(REPO)}:{c}" for p in DOCS for c in wrong_identity_claims(read(p))]
    assert not problems, "\n".join(problems)
    assert wrong_identity_claims("A new `--train-time` starts a new campaign.\n")
    assert wrong_identity_claims("| Adding `--sbatch-arg=--exclude=g1` makes a different campaign |\n")
    assert not wrong_identity_claims("`--sbatch-arg` and walltimes keep the campaign; `--gpu` starts a new campaign.\n")
    assert not wrong_identity_claims("A longer --train-time does not start a new campaign.\n")


REPO_PATH = re.compile(r"(?<![\w<{$/.-])((?:tests|scripts|src/mpd|docs|constraints)/[A-Za-z0-9_./-]*[A-Za-z0-9_])")


def test_every_repository_path_the_docs_name_exists():
    missing = []
    for path in DOCS:
        for no, line in enumerate(read(path).splitlines(), 1):
            for name in REPO_PATH.findall(line):
                if not (REPO / name).exists():
                    missing.append(f"{path.relative_to(REPO)}:{no}: {name}")
    assert not missing, "\n".join(missing)
    assert REPO_PATH.findall("see `tests/test_x.py`, scripts/a.sh and <home>/tests/y.py") == ["tests/test_x.py", "scripts/a.sh"]


DECISIONS = {
    "README.md": ("--dry-run", "`--force`", "docs/LICENSES.md"),
    "docs/GRACE.md": ("--max-su", "Python 3.11 or newer", "Anaconda3/2024.02-1", "GCCcore/13.2.0 Python/3.11.5", "--from-file", "MPD_ACCOUNT",
                      "one GPU by default", "continue or repair", "never queues a second copy", "--seed", "only the stages whose output files are missing",
                      "delete its marker file", "sbatch --test-only"),
    "docs/TROUBLESHOOTING.md": ("--from-file", "delete its marker file", "--force", "keep the campaign"),
    "docs/RESULTS.md": ("unconfirmed", "assumption of the maintainer"),
    "docs/LICENSES.md": ("clause 1.2.2", "no switch"),
}


def test_the_pages_state_the_decisions_of_the_release():
    for name, needles in DECISIONS.items():
        flat = " ".join(doc(name).split())
        for needle in needles:
            assert needle in flat, f"{name} does not say {needle!r}"
        assert "TODO" not in flat and "ID_NOTE" not in flat, name


# -- pasted output is the output of the program -------------------------------------------------------------------------------------------

def blocks_of(name: str, lang: str = "text") -> list[tuple[int, list[str]]]:
    return [(start, lines) for blang, start, lines in split_markdown(doc(name))[1] if blang == lang]


def block_starting(name: str, prefix: str, lang: str = "text") -> list[str]:
    found = [lines for _, lines in blocks_of(name, lang) if lines and lines[0].startswith(prefix)]
    assert len(found) == 1, f"{name}: expected one {lang} block starting with {prefix!r}, found {len(found)}"
    return found[0]


def normal(text: str, home_root: Path) -> list[str]:
    text = text.replace(str(home_root), "/scratch/user/<netid>")
    text = re.sub(r"\d[\d,]* GB free on this filesystem", "N GB free on this filesystem", text)
    return text.splitlines()


GRACE_SCRATCH = "scratch"


@pytest.fixture
def grace_shell(cli_env, monkeypatch):
    """A login shell on a machine with no Slurm: $SCRATCH set, so the default home is $SCRATCH/mpd as on Grace."""
    root = cli_env.tmp / GRACE_SCRATCH
    (root / "mpd").mkdir(parents=True)
    monkeypatch.setenv("SCRATCH", str(root))
    return root


def test_the_doctor_sample_is_what_doctor_prints_on_an_empty_home(capsys, grace_shell):
    rc, out, err = run_cli(capsys, "doctor", "--cluster", "grace")
    assert rc == 1, err
    live = normal(out, grace_shell)
    sample = [normal(line, grace_shell)[0] if line else "" for line in block_starting("docs/GRACE.md", "[ok]   home")]
    assert sample == live


FIRST_RUN = ["--account", "123456789012", "--teacher", "esm2-8m", "--student", "esm2-8m", "--train-data", "proteingym", "--method", "score-all",
             "--eval-data", "proteingym", "--updates", "50", "--train-gpus", "1", "--score-shards", "proteingym=1"]


def test_the_first_run_plan_in_grace_md_is_what_plan_prints(capsys, grace_shell):
    write_proteingym(grace_shell / "mpd")
    rc, out, err = run_cli(capsys, "plan", "--cluster", "grace", *FIRST_RUN)
    assert rc == 0, err
    live = [line for line in normal(out, grace_shell) if not line.startswith("warning: ")]       # the page leaves the five warnings out
    shown = [normal(line, grace_shell)[0] if line else "" for line in block_starting("docs/GRACE.md", "campaign   esm2-8m__esm2-8m")]
    assert shown == live
    assert len([line for line in normal(out, grace_shell) if line.startswith("warning: ")]) == 5


def test_mpd_clusters_in_clusters_md_is_what_clusters_prints(capsys):
    rc, out, err = run_cli(capsys, "clusters")
    assert rc == 0
    assert block_starting("docs/CLUSTERS.md", "$ mpd clusters")[1:] == out.splitlines()


def test_the_dry_run_rows_in_grace_md_are_what_a_machine_without_sbatch_prints(capsys, grace_shell):
    write_proteingym(grace_shell / "mpd")
    rc, out, err = run_cli(capsys, "submit", "--dry-run", "--cluster", "grace", *FIRST_RUN)
    assert rc == 0, err
    rows = block_starting("docs/GRACE.md", "[skip] slurm")
    assert len(rows) == 2 and all(row in out.splitlines() for row in rows)


def test_the_train_script_in_grace_md_is_the_script_the_renderer_writes(tmp_path):
    """Every line of the pasted script (the lines with `...` are shortened on purpose) is a line of the rendered one."""
    from mpd.slurm.render import render_script
    home = tmp_path / "mpd"
    python = home / "env" / "bin" / "python"
    write_record(home, python)
    plan = make("grace", home=home, spec_kw=dict(teacher="esm2-8m", student="esm2-8m", eval_data=("proteingym",), updates=50, train_gpus=2),
                score_shards={"proteingym": 1}, python=str(python))
    text = render_script(plan, next(s for s in plan.stages if s.name == "train"))
    for old, new in ((str(plan.campaign_dir), "<campaign>"), (plan.campaign_id, "<campaign>"), (plan.nonce, "<nonce>"), (str(python), "<python>")):
        text = text.replace(old, new)
    shown = block_starting("docs/GRACE.md", "#!/bin/bash", lang="bash")
    live = set(text.splitlines())
    missing = [line for line in shown if "..." not in line and line not in live]
    assert shown[0] == "#!/bin/bash" and len(shown) > 25
    assert not missing, "\n".join(missing)


def status_sentences() -> list[str]:
    """The advice `mpd status` prints for one invented job of each kind (the same cases the page tabulates)."""
    from mpd.slurm import campaign as c
    from mpd.slurm.campaign import JobState, advise

    def job(state, exitcode="0:0", limit="08:00:00", elapsed="08:00:12", node="g003"):
        return JobState(state, elapsed, exitcode, "mpd-train", "", limit, node)

    ref, plan = "<campaign>", {"walltime_s": 28800, "partition_limit_s": 345600, "walltime_basis": "user"}
    cases = [
        advise("train", "1003", job("TIMEOUT"), ref=ref, plan=plan),
        advise("score", "1005", job("TIMEOUT", limit="00:30:00", elapsed="00:30:04"), ref=ref, plan={"partition_limit_s": 345600}),
        advise("prepare", "1001", job("TIMEOUT", limit="04:00:00", elapsed="04:00:02"), ref=ref),
        advise("train", "1003", job("OUT_OF_MEMORY", "0:125"), ref=ref, plan={"mem_gb": 180}),
        advise("train", "1003", job("FAILED", "1:0"), ref=ref, err="torch.OutOfMemoryError: CUDA out of memory"),
        advise("train", "1003", job("NODE_FAIL"), ref=ref),
        advise("train", "1003", job("FAILED", "31:0", node="g007"), ref=ref, err="MPD_NO_GPU: no usable GPU on g007"),
        advise("prepare", "1001", job("FAILED", "1:0"), ref=ref, err="MPD_MODULE_LOAD_FAILED Anaconda3/2024.02-1", cluster="grace"),
        advise("train", "1003", job("FAILED", "1:0"), ref=ref,
               err="weights for esm2-3b are missing (...). Run on a login node: mpd fetch model esm2-3b --cluster grace"),
        advise("train", "1003", job("FAILED", "1:0"), ref=ref, err="OfflineModeIsEnabled: Cannot reach https://huggingface.co"),
        advise("train", "1003", job("FAILED", "1:0"), ref=ref, err="ValueError: ..."),
        advise("score", "1005", job("CANCELLED"), ref=ref, reason="DependencyNeverSatisfied"),
        advise("train", "1003", job("CANCELLED"), ref=ref),
        advise("train", "1003", job("PREEMPTED"), ref=ref),
        c.advise_cancelled(["prepare", "train", "evaluate"]),
    ]
    waiting = [c.pending_help(reason, ref) for reason in c.PENDING_REASONS]
    waiting += [c.pending_help(r, ref) for r in ("AssocGrpBillingMinutes", "AssocMaxJobsLimit", "QOSMaxJobsPerUserLimit")]
    return [" ".join(" ".join(lines).split()) for lines in cases] + [" ".join(w.split()) for w in waiting]


def test_what_mpd_status_says_in_troubleshooting_md_is_what_the_code_says():
    page = " ".join(doc("docs/TROUBLESHOOTING.md").split())
    missing = [s for s in status_sentences() if s not in page]
    assert len(status_sentences()) >= 30
    assert not missing, "\n".join(missing)


# -- quoted messages exist in the code -----------------------------------------------------------------------------------------------

def source_pieces() -> list[str]:
    """Every string constant in src/mpd and every text segment of the setup script; f-string holes and {} fields split a message into pieces."""
    pieces: list[str] = []

    def add(text: str) -> None:
        pieces.extend(re.split(r"\{[^{}]*\}", text))

    for path in (REPO / "src" / "mpd").rglob("*.py"):
        for node in ast.walk(ast.parse(read(path))):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                add(node.value)
    script = read(REPO / "scripts" / "setup_env.sh").replace("\\$", "\x01").replace('\\"', '"')      # an escaped dollar is text, not a hole
    pieces += [p.replace("\x01", "$") for p in re.split(r"\$\{[^}]*\}|\$\([^)]*\)|\$[A-Za-z_][A-Za-z0-9_]*", script)]
    return pieces


# Messages that other programs print: the page quotes them so that a student can find the row.
EXTERNAL = ("mpd: command not found", "Disk quota exceeded", "OSError: [Errno 122]", "HTTP Error 429", "/etc/bashrc", "OSError: We couldn't connect",
            "torch.OutOfMemoryError", "ValueError ... torch.load", "your account's balance is not sufficient",
            "Invalid account or account/partition combination", "Requested node configuration is not available", "Requested time limit is invalid",
            "Job violates accounting/QOS policy", "unrecognized option", "Job dependency problem", "Unable to contact slurm controller",
            "Some weights of EsmForMaskedLM", "GPU 0: NVIDIA", "CUDA out of memory", "Priority", "Dependency", "PartitionTimeLimit", "Network is unreachable",
            "Slurm says", "configuration error", "unbound variable", "SaProt results on VenusMutHub differ", "expected one argument", "no job states",
            "not resuming: all.partial", "sacct", "squeue", "assocmax", "QOSMax", "AssocMax", "GLIBCXX", "libstdc++")


def message_chunks(span: str) -> list[str]:
    """The literal pieces of a quoted message: split at <placeholders> and at ...; a piece must be 12 characters or more to be checked."""
    parts = re.split(r"<[^<>\n]*>|\.\.\.", span)
    out = []
    for part in parts:
        part = part.strip(" ,;()[]\"'")
        if len(part) >= 12:
            out.append(part)
    return out


MESSAGE_PAGES = ("docs/TROUBLESHOOTING.md", "docs/CLUSTERS.md")


def first_column_spans(name: str) -> list[tuple[int, str]]:
    """The code spans in the first cell of each row of the tables whose first header cell is `you see` (the tables of messages)."""
    found, position, wanted = [], None, False
    for no, line in split_markdown(doc(name))[0] + [(10**6, "")]:
        if not line.startswith("|"):
            position = None
            continue
        cell = re.split(r"(?<!\\)\|", line)[1]
        position = 0 if position is None else position + 1
        if position == 0:
            wanted = cell.strip() == "you see"
        elif position >= 2 and wanted:
            for double, single in re.findall(r"``(.+?)``|`([^`]+)`", cell):
                found.append((no, (double or single).strip()))
    return found


COMMAND_SPAN = re.compile(r"mpd (fetch|plan|submit|status|logs|cancel|results|models|clusters|assays|doctor)\b|bash scripts/|source |python |--[\w-]+(=\S+| <[^>]+>)?$")


def is_command(span: str) -> bool:
    """A code span that is a command or a bare option rather than a message (`mpd fetch model`, `--force`, `--env MPD_DTYPE=float32`)."""
    return bool(COMMAND_SPAN.match(span))


def test_the_messages_quoted_in_the_docs_exist_in_the_code():
    pieces = source_pieces()
    spans = [(name, no, span) for name in MESSAGE_PAGES for no, span in first_column_spans(name)]
    assert len(spans) > 150
    unknown, checked = [], 0
    for name, no, span in spans:
        if is_command(span) or any(e in span for e in EXTERNAL):
            continue
        for chunk in message_chunks(span):
            words = chunk.split()
            variants = {chunk} | {v for v in (" ".join(words[1:]), " ".join(words[:-1]), " ".join(words[1:-1])) if len(v) >= 12}   # a value next to a hole
            checked += 1
            if not any(any(v in piece for piece in pieces) for v in variants):
                unknown.append(f"{name}:{no}: {chunk!r}")
    assert checked > 150
    assert not unknown, "\n".join(unknown)


def test_external_messages_are_all_quoted_somewhere():
    page = doc("docs/TROUBLESHOOTING.md")
    stale = [e for e in EXTERNAL if e not in page and e.lower() not in page.lower()]
    assert not stale, stale


def test_the_message_checker_notices_a_message_the_code_does_not_print():
    pieces = source_pieces()
    assert any("not enough free space under " in p for p in pieces)
    assert not any("this message was never written" in p for p in pieces)


# -- the snippet for using a trained student -----------------------------------------------------------------------------------------

def test_the_python_snippet_in_results_md_reproduces_the_logits_of_the_student_class(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    pytest.importorskip("peft")
    import numpy as np
    import torch
    from tinydata import make_tiny_esm

    from mpd.contexts import ContextTable
    from mpd.students import load

    base = make_tiny_esm(tmp_path / "base", seed=1)
    monkeypatch.setenv("MPD_MODEL_ESM2_8M", str(base))
    monkeypatch.setenv("MPD_DTYPE", "float32")
    monkeypatch.delenv("MPD_ALLOW_HUB", raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")                       # the snippets must work without a network, like the jobs
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    student = load("esm2-8m", torch.device("cpu"), lora={"rank": 8, "alpha": 16}, dtype=torch.float32)
    torch.manual_seed(0)
    with torch.no_grad():
        for p in student.trainable():
            p.copy_(torch.randn_like(p) * 0.05)               # a trained-looking adapter: B = 0 would hide a loading mistake
    adapter = tmp_path / "checkpoint_00006"
    student.save_adapter(adapter)

    code = next("\n".join(lines) for lang, _, lines in blocks_of_all("docs/RESULTS.md") if lang == "python" and "def masked_logits" in "\n".join(lines))
    code = re.sub(r'^base_dir = .*$', f"base_dir = {str(base)!r}", code, flags=re.M)
    code = re.sub(r'^adapter_dir = .*$', f"adapter_dir = {str(adapter)!r}", code, flags=re.M)
    namespace: dict = {}
    exec(compile(code, "docs/RESULTS.md", "exec"), namespace)

    seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVV"[:40]
    positions = [0, 5, len(seq) - 1]
    table = ContextTable([seq], np.zeros(3, np.int32), np.asarray(positions, np.int32), np.asarray([namespace["AA20"].index(seq[p]) for p in positions], np.uint8))
    reference = load("esm2-8m", torch.device("cpu"), adapter=adapter, dtype=torch.float32)
    reference.encode(table)
    expected = reference.logits(table, np.arange(3)).numpy()
    got = np.stack([namespace["masked_logits"](seq, p).numpy() for p in positions])
    assert np.abs(expected - got).max() < 1e-5
    mutant = "A" if seq[5] != "A" else "C"
    z = namespace["masked_logits"](seq, 5)
    assert namespace["delta"](seq, 5, mutant) == pytest.approx(float(z[namespace["AA20"].index(mutant)] - z[namespace["AA20"].index(seq[5])]), abs=1e-6)
    assert np.isfinite(namespace["dual"](seq, 5, mutant))

    # the shorter snippet of the README loads the same adapter on the same base
    short = next("\n".join(lines) for lang, _, lines in blocks_of_all("README.md") if lang == "python")
    short = re.sub(r'^base = .*$', f"base = {str(base)!r}", short, flags=re.M).replace('"<campaign>/train/all/checkpoint_02103"', repr(str(adapter)))
    readme_ns: dict = {}
    exec(compile(short, "README.md", "exec"), readme_ns)
    ids = readme_ns["tok"](seq, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        readme_logits = readme_ns["model"](input_ids=ids).logits
    assert readme_ns["model"].training is False and readme_logits.shape[-1] == 33


def blocks_of_all(name: str) -> list[tuple[str, int, list[str]]]:
    return split_markdown(doc(name))[1]


# -- the cluster profile template -----------------------------------------------------------------------------------------------------

def test_the_profile_template_in_clusters_md_loads_and_plans(tmp_path):
    """CLUSTERS.md says its template is loaded and planned by a check of the page: this is the check."""
    import yaml

    from mpd.slurm.cluster import Cluster
    from mpd.slurm.plan import PlanOptions, describe, make_plan
    from mpd.spec import RunSpec

    template = next("\n".join(lines) for lang, _, lines in blocks_of_all("docs/CLUSTERS.md") if lang == "yaml")
    assert yaml.safe_load(template)["name"] == "mycluster"
    path = tmp_path / "mycluster.yaml"                                  # the page says: save it as <name>.yaml
    path.write_text(template + "\n", encoding="utf-8")
    cluster = Cluster.load(str(path))
    assert cluster.name == "mycluster" and cluster.default_train_gpus == 1 and not cluster.gpus["t4"].bf16
    plan = make_plan(RunSpec(teacher="esm2-8m", student="esm2-8m", train_data="proteingym", method="score-all", eval_data=("proteingym",)),
                     PlanOptions(cluster=cluster, home=tmp_path / "mpd", python="/env/bin/python"))
    assert [s.name for s in plan.stages][:2] == ["prepare", "teacher"]
    assert "estimated upper bound: not billed" in describe(plan)         # the page: su_per_gpu_hour 0 means "not billed per SU"
    with pytest.raises(Exception, match="bfloat16"):                    # the template's t4 has bf16: false, so training on it is refused
        make_plan(RunSpec(teacher="esm2-8m", student="esm2-8m", train_data="proteingym", method="score-all", eval_data=("proteingym",)),
                  PlanOptions(cluster=cluster, home=tmp_path / "mpd", python="/env/bin/python", gpu="t4"))
