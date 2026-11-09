"""What a student sees from `mpd` when a SaProt run is missing something: the models table and the plan's preflight lines for
Foldseek, the ProteinGym structures and the unpublished VenusMutHub structure set. The real command line, a hermetic home, no
network and no Slurm; the fix commands are the ones the text tells the user to run."""

import pytest
from fx_cli import FAKE_ACCOUNT, cli_env, make_fake_python, override_weights, run_cli, write_proteingym, write_record, write_venus  # noqa: F401
from fx_saprot import fake_foldseek, write_tiny_structure_home, write_tiny_venus  # noqa: F401

pytestmark = pytest.mark.usefixtures("cli_env")


def recipe(home, *, teacher="esm2-3b", student="saprot-650m", train="proteingym", eval_data=("proteingym",), method="score-all"):
    return ["--cluster", "grace", "--home", str(home), "--teacher", teacher, "--student", student, "--train-data", train,
            "--method", method, "--eval-data", *eval_data, "--account", FAKE_ACCOUNT]


def stocked(env, monkeypatch, *models):
    """The interpreter, ProteinGym, VenusMutHub and the weights of `models` are in place; nothing for SaProt's structures."""
    python = make_fake_python(env.home / "env")
    write_record(env.home, python)
    write_proteingym(env.home)
    write_venus(env.home)
    override_weights(env, monkeypatch, *models)


def finding(out: str, name: str) -> str:
    """One preflight line of `mpd plan`, with the lines that continue it."""
    lines = out.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("[") and line[7:].split()[:1] == [name]:
            entry = [line]
            for more in lines[i + 1:]:
                if more.startswith("[") or not more.strip():
                    break
                entry.append(more.strip())
            return " ".join(entry)
    raise AssertionError(f"no {name!r} line in:\n{out}")


def test_both_saprot_models_are_available_and_aido_is_not(capsys):
    rc, out, _ = run_cli(capsys, "models")
    row = {line.split()[0]: line for line in out.splitlines() if line.split() and line.split()[0] in ("saprot-35m", "saprot-650m", "aido-rag-16b")}
    assert rc == 0 and " available " in row["saprot-35m"] and " available " in row["saprot-650m"] and " planned " in row["aido-rag-16b"]
    flat = " ".join(out.split())                                               # notes are wrapped to the terminal width
    assert "Reads protein structure (Foldseek 3Di" in flat and "not published yet" in flat and "Planned: needs the Foldseek" not in flat


def test_without_foldseek_the_plan_names_the_download_and_the_override(capsys, cli_env, monkeypatch):
    stocked(cli_env, monkeypatch, "esm2-3b", "saprot-650m")
    rc, out, err = run_cli(capsys, "plan", *recipe(cli_env.home))
    assert rc == 0, err                                                             # a plan reports; only submit refuses
    text = finding(out, "foldseek")
    assert text.startswith("[FAIL]") and f"Run on a login node: mpd fetch foldseek --cluster grace --home {cli_env.home}" in text
    assert "MPD_FOLDSEEK" in text and "NOT READY" in out.splitlines()[-1]


def test_without_proteingym_structures_the_plan_says_they_come_with_mpd_fetch_proteingym(capsys, cli_env, monkeypatch):
    stocked(cli_env, monkeypatch, "esm2-3b", "saprot-650m")
    rc, out, err = run_cli(capsys, "plan", *recipe(cli_env.home))
    assert rc == 0, err
    text = finding(out, "structures:proteingym")
    assert text.startswith("[FAIL]") and "ProteinGym_AF2_structures.zip, 19.2 MB" in text and "part of the ProteinGym download" in text
    assert f"Run on a login node: mpd fetch proteingym --cluster grace --home {cli_env.home}" in text
    assert "[ok]   data:proteingym" in out                                         # the data itself is there, so only the structures are missing


def test_without_the_venus_structure_set_the_plan_says_it_is_unpublished_and_how_to_use_a_copy(capsys, cli_env, monkeypatch):
    stocked(cli_env, monkeypatch, "esm2-3b", "saprot-650m")
    rc, out, err = run_cli(capsys, "plan", *recipe(cli_env.home, eval_data=("venusmuthub",)))
    assert rc == 0, err
    text = finding(out, "structures:venusmuthub")
    assert text.startswith("[FAIL]") and "not published yet" in text
    assert f"mpd fetch venus-structures --cluster grace --home {cli_env.home} --from-dir <folder with the deposit files>" in text
    assert "structures:proteingym" in out                                         # the SaProt student is trained on ProteinGym too


def test_a_saprot_teacher_needs_structures_for_the_training_data_only(capsys, cli_env, monkeypatch):
    stocked(cli_env, monkeypatch, "saprot-650m", "esm2-650m")
    rc, out, err = run_cli(capsys, "plan", *recipe(cli_env.home, teacher="saprot-650m", student="esm2-650m", eval_data=("venusmuthub",)))
    assert rc == 0, err
    assert finding(out, "structures:proteingym").startswith("[FAIL]")
    assert "structures:venusmuthub" not in out                                    # the ESM2 student scores sequences alone


def test_a_sequence_only_pair_is_not_asked_for_foldseek_or_structures(capsys, cli_env, monkeypatch):
    stocked(cli_env, monkeypatch, "esm2-3b", "esm2-650m")
    rc, out, err = run_cli(capsys, "plan", *recipe(cli_env.home, student="esm2-650m", eval_data=("proteingym", "venusmuthub")))
    assert rc == 0, err
    assert "foldseek" not in out.lower() and "structures:" not in out and out.splitlines()[-1].startswith("READY")


def test_with_foldseek_and_structures_in_place_the_plan_has_no_failure(capsys, cli_env, monkeypatch, fake_foldseek):
    stocked(cli_env, monkeypatch, "esm2-3b", "saprot-650m")
    write_tiny_structure_home(cli_env.home)
    write_tiny_venus(cli_env.home, {"V": "ACDEFGHIKLMNPQRSTVWYACDEFGHIKL"})
    rc, out, err = run_cli(capsys, "plan", *recipe(cli_env.home, eval_data=("proteingym", "venusmuthub")))
    assert rc == 0, err
    assert "[FAIL]" not in out and out.splitlines()[-1].startswith("READY")
    assert finding(out, "structures:proteingym").startswith("[ok]") and finding(out, "structures:venusmuthub").startswith("[ok]")
    assert finding(out, "foldseek").startswith("[warn]") and "not the pinned release" in finding(out, "foldseek")   # a fake, not release 10


def test_a_real_submit_stops_at_the_missing_structures_and_a_dry_run_only_reports_them(capsys, cli_env, monkeypatch):
    stocked(cli_env, monkeypatch, "esm2-3b", "saprot-650m")
    rc, out, err = run_cli(capsys, "submit", *recipe(cli_env.home), "--dry-run")
    assert rc == 0, err
    assert "NOT READY" in out and "mpd fetch foldseek" in out and "mpd fetch proteingym" in out
    rc, out, err = run_cli(capsys, "submit", *recipe(cli_env.home), "--yes")
    assert rc == 2 and "mpd fetch foldseek --cluster grace" in err and "mpd fetch proteingym --cluster grace" in err
