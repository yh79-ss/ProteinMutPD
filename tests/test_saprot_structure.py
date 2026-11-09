"""SaProt structure inputs on CPU, offline, with a fake Foldseek: vocabulary, PDB parsing, alignment, masking, the
Foldseek wrapper, structure sets, benchmark ranges, preflight messages and the pinned Foldseek fetch."""

import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from fx_saprot import (atom_line, fake_calls, fake_foldseek, fake_release, fake_states, install_fake_foldseek,  # noqa: F401
                       is_executable, pdb_text_chains, serve, write_pdb, write_tiny_structure_home)
from mpd.contexts import ContextTable, build_table, sequence_hash
from mpd.data import sources, venus_structures as vs
from mpd.data.benchmarks import Assay, ProteinGym, open_benchmark, pdb_span
from mpd.data.fetch import FetchError
from mpd.spec import RunSpec
from mpd.structure import build, foldseek, pdb as pdbmod, preflight_checks, tools, vocab
from mpd.structure.align import AlignmentError, align_to_sequence
from mpd.structure.set import StructureSet

SEQ = "MKVLAGDE"
MASKED_OR_KEPT = [False, True, True, False, True, False, True, False]     # which residues of SEQ survive pLDDT >= 70


def expected(seq, keep):
    return "".join(s if k else "#" for s, k in zip(fake_states(seq), keep))


def engine_of(binary):
    return foldseek.FoldseekEngine(binary)


# -- vocabulary ------------------------------------------------------------------------------------------------

def test_the_vocabulary_is_the_released_one():
    tokens = vocab.build_vocab()
    assert len(tokens) == vocab.VOCAB_SIZE == 446
    assert hashlib.sha256(vocab.vocab_text().encode()).hexdigest() == vocab.VOCAB_SHA256 \
        == "d4289b3ff55d50b1aa33a3141ec344fc2a69cb3dca43aeefe9fccfee371ffd28"
    ids = vocab.token_ids()
    assert [ids[t] for t in ("<cls>", "<pad>", "<eos>", "<unk>", "<mask>")] == [0, 1, 2, 3, 4]
    assert (ids["Ap"], ids["A#"], ids["Md"], ids["#p"], ids["##"]) == (5, 25, 223, 425, 445)
    assert (vocab.aa_state_ids() == np.arange(5, 425).reshape(20, 21)).all()      # the 20 x 21 block is contiguous
    assert vocab.mask_ids()["d"] == ids["#d"] and vocab.mask_ids()["#"] == 445 and len(vocab.mask_ids()) == 21


def test_the_readme_example_is_one_token_per_residue():
    ids = vocab.encode("MEVQLVQYK", "#vpp#ydav")                  # SaProt's README: M#EvVpQpL#VyQdYaKv
    tokens = vocab.build_vocab()
    assert [tokens[i] for i in ids] == ["<cls>", "M#", "Ev", "Vp", "Qp", "L#", "Vy", "Qd", "Ya", "Kv", "<eos>"]


def test_encode_rules():
    tokens = vocab.build_vocab()
    assert [tokens[i] for i in vocab.encode("aX", "p#")[1:-1]] == ["Ap", "##"]      # case folded, non-standard letter -> '#'
    assert vocab.encode("A", "P") == vocab.encode("A", "p")
    with pytest.raises(ValueError, match="not a 3Di state"):
        vocab.encode("A", "z")
    with pytest.raises(ValueError, match="1 residues but the structure string has 2"):
        vocab.encode("A", "pp")


def test_a_model_folder_must_carry_the_same_vocabulary(tmp_path):
    good = tmp_path / "vocab.txt"
    good.write_text(vocab.vocab_text())
    assert vocab.load_vocab(good) == vocab.build_vocab()
    bad = tmp_path / "other.txt"
    bad.write_text(vocab.vocab_text().replace("Ap\n", "Ax\n"))
    with pytest.raises(vocab.VocabError, match="vocab.txt has sha256"):
        vocab.load_vocab(bad)


# -- PDB records -------------------------------------------------------------------------------------------------

def test_ca_records_follow_the_earlier_studys_rules(tmp_path):
    lines = ["MODEL        1",
             atom_line(1, "CA", "ALA", "A", 1, 80.0, altloc="A"),
             atom_line(2, "CA", "ALA", "A", 1, 10.0, altloc="B"),                 # second alternate location: ignored
             atom_line(3, "CA", "GLY", "A", 2, 70.0),
             atom_line(4, "CA", "GLY", "A", 2, 1.0),                              # same residue again: ignored
             atom_line(5, "CA", "GLY", "A", 2, 60.0, icode="A"),                  # insertion code: a new residue
             atom_line(6, "CA", "CA", "A", 9, 5.0, record="HETATM"),              # a calcium ion
             atom_line(7, "N", "LEU", "A", 3, 5.0),                               # not a CA atom
             atom_line(8, "CA", "VAL", "B", 1, 50.0),
             "ENDMDL", "MODEL        2", atom_line(9, "CA", "TRP", "A", 3, 5.0), "ENDMDL"]
    path = tmp_path / "x.pdb"
    path.write_text("\n".join(lines) + "\n")
    chains = pdbmod.ca_chains(path)
    assert list(chains) == ["A", "B"]
    assert chains["A"].aa == "AGG" and chains["A"].bfactor == (80.0, 70.0, 60.0) and chains["B"].aa == "V"
    assert pdbmod.ca_records(path, "B").aa == "V"
    with pytest.raises(pdbmod.PdbError, match="2 chains"):
        pdbmod.ca_records(path)
    with pytest.raises(pdbmod.PdbError, match="no chain 'C'"):
        pdbmod.ca_records(path, "C")
    (tmp_path / "empty.pdb").write_text("REMARK nothing\n")
    with pytest.raises(pdbmod.PdbError, match="no CA atoms"):
        pdbmod.ca_records(tmp_path / "empty.pdb")


# -- alignment ------------------------------------------------------------------------------------------------------

LONG = "MKVLAGDEWQRSTNYFHPIC"


def swap(seq, indices):
    """`seq` with the residues at `indices` replaced by a different one (G, or A where it was G)."""
    return "".join(("A" if c == "G" else "G") if i in indices else c for i, c in enumerate(seq))


def test_alignment_of_a_structure_that_covers_the_whole_sequence():
    a = align_to_sequence(LONG, LONG)
    assert (a.seq_start, a.chain_start, a.n, a.substitutions) == (0, 0, 20, ())


def test_alignment_with_a_range_like_zikv_or_polg():
    a = align_to_sequence(LONG, LONG[5:15], span=(5, 15))
    assert (a.seq_start, a.chain_start, a.n) == (5, 0, 10)
    assert align_to_sequence(LONG, LONG[17:], span=(17, 25)).n == 3        # a range past the sequence end is clipped to the chain


def test_alignment_ignores_a_tag_at_the_end_of_the_structure_like_cas9():
    tagged = LONG + "SRADPKKKRKV"
    a = align_to_sequence(LONG, tagged, span=(0, 20))
    assert (a.seq_start, a.chain_start, a.n) == (0, 0, 20)
    assert align_to_sequence(LONG, tagged).n == 20                        # also without a range: the sequence occurs once
    assert align_to_sequence(LONG, "GG" + LONG + "SS").chain_start == 2


def test_alignment_records_point_substitutions_like_p53():
    chain = LONG[:3] + "P" + LONG[4:]                                     # the model has P where the assay has L
    a = align_to_sequence(LONG, chain)
    assert a.substitutions == ("L4P",) and a.n == 20
    assert align_to_sequence(LONG, chain, span=(0, 20)).substitutions == ("L4P",)
    assert align_to_sequence(LONG, swap(LONG, (1, 5, 9))).substitutions == ("K2G", "G6A", "Q10G")


def test_alignment_refuses_what_it_cannot_map():
    four = swap(LONG, (1, 5, 9, 12))
    with pytest.raises(AlignmentError, match="4 of 20 structure residues differ.*position 2"):
        align_to_sequence(LONG, four)
    with pytest.raises(AlignmentError, match="does not contain"):                 # an insertion
        align_to_sequence(LONG, LONG[:5] + "G" + LONG[5:])
    with pytest.raises(AlignmentError, match="differ from the sequence"):         # a deletion, even with a range
        align_to_sequence(LONG, LONG[:5] + LONG[6:], span=(0, 20))
    with pytest.raises(AlignmentError, match="whole sequence"):                   # a fragment without a range
        align_to_sequence(LONG, LONG[3:10])
    with pytest.raises(AlignmentError, match="2 times"):
        align_to_sequence("MKV", "MKVGMKV")
    with pytest.raises(AlignmentError, match="outside"):
        align_to_sequence(LONG, LONG, span=(30, 40))
    with pytest.raises(AlignmentError, match="do not overlap"):
        align_to_sequence(LONG, LONG, span=(5, 5))


# -- masking ------------------------------------------------------------------------------------------------------------

def assay_with(tmp_path, bfactors, scale, name="x", seq=SEQ, **kw):
    return Assay(name, "P", seq, write_pdb(tmp_path / f"{name}.pdb", seq, bfactors), scale, **kw)


def string_of(assay, policy=None):
    return build.structure_string(assay, engine_of(os.environ["MPD_FOLDSEEK"]), policy)


def test_masking_rules_per_structure_source(tmp_path, fake_foldseek):
    # AlphaFold DB and ProteinGym: pLDDT 0-100. Strictly below 70 is masked; 70.0 is kept.
    afdb = assay_with(tmp_path, [69.99, 70.0, 95.0, 20.0, 70.01, 69.5, 100.0, 0.0], "percent_0_to_100", "afdb")
    text, info = string_of(afdb)
    assert text == expected(SEQ, MASKED_OR_KEPT)
    assert info["masked_low_confidence"] == info["masked"] == 4 and info["uncovered"] == 0
    # ESMFold: values 0-1, multiplied by 100 first (0.699 -> 69.9 is masked, 0.70 -> 70.0 is kept)
    esm = assay_with(tmp_path, [0.699, 0.70, 0.95, 0.2, 0.701, 0.69, 1.0, 0.0], "fraction_0_to_1", "esm")
    assert string_of(esm)[0] == expected(SEQ, MASKED_OR_KEPT)
    # experimental: the column holds crystallographic B-factors, not confidence, so nothing is masked
    exp = assay_with(tmp_path, [0.0, 549.96, 12.5, 70.0, 69.0, 3.0, 99.9, 1.0], "bfactor", "exp")
    assert string_of(exp)[0] == fake_states(SEQ)


def test_the_legacy_flag_reproduces_the_all_masked_esmfold_inputs(tmp_path, fake_foldseek, monkeypatch):
    esm = assay_with(tmp_path, [0.9] * 8, "fraction_0_to_1", "esm")
    afdb = assay_with(tmp_path, [90.0] * 8, "percent_0_to_100", "afdb")
    exp = assay_with(tmp_path, [5.0] * 8, "bfactor", "exp")
    assert string_of(esm)[0] == fake_states(SEQ)                                        # corrected rule: confident
    assert string_of(esm, build.Policy(legacy_mask=True))[0] == "#" * len(SEQ)          # earlier study: the raw 0.9 < 70
    assert string_of(afdb, build.Policy(legacy_mask=True))[0] == fake_states(SEQ)       # 0-100 columns are unaffected
    assert string_of(exp, build.Policy(legacy_mask=True))[0] == fake_states(SEQ)
    assert not build.Policy.from_env().legacy_mask
    monkeypatch.setenv("MPD_STRUCTURE_LEGACY_MASK", "1")
    assert build.Policy.from_env().legacy_mask and string_of(esm)[0] == "#" * len(SEQ)
    assert build.Policy(legacy_mask=True).describe()["legacy_mask"] is True


def test_a_structure_needs_a_known_confidence_scale(tmp_path, fake_foldseek):
    with pytest.raises(build.StructureError, match="unknown structure confidence scale None"):
        string_of(assay_with(tmp_path, 90.0, None))
    with pytest.raises(build.StructureError, match="structure file .* not found"):
        string_of(Assay("a", "P", SEQ, tmp_path / "missing.pdb", "percent_0_to_100"))
    with pytest.raises(build.StructureError, match="no structure file|\\(none recorded\\)"):
        string_of(Assay("a", "P", SEQ, None, "percent_0_to_100"))


def test_positions_outside_the_structure_and_substitutions_are_recorded(tmp_path, fake_foldseek):
    seq = "MKVLAGDEWQRSTNYFHPIC"
    model = seq[:3] + "P" + seq[4:]                                            # one residue differs from the assay
    path = write_pdb(tmp_path / "part.pdb", model[5:15], 90.0)                  # covers residues 6-15
    a = Assay("a", "P", seq, path, "percent_0_to_100", structure_span=(5, 15))
    text, info = string_of(a)
    assert text == "#" * 5 + fake_states(model[5:15]) + "#" * 5
    assert (info["offset"], info["residues"], info["uncovered"], info["masked"]) == (5, 10, 10, 10)
    full = Assay("b", "P", seq, write_pdb(tmp_path / "full.pdb", model, 90.0), "percent_0_to_100")
    assert string_of(full)[1]["substitutions"] == ["L4P"]


# -- the Foldseek wrapper ----------------------------------------------------------------------------------------------

def test_the_wrapper_runs_the_documented_command_and_cleans_up(tmp_path, fake_foldseek, monkeypatch):
    scratch = tmp_path / "tmp"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))
    pdb = write_pdb(tmp_path / "1abc.pdb", SEQ, 90.0)
    chains = foldseek.run(fake_foldseek, pdb)
    assert len(chains) == 1 and (chains[0].name, chains[0].chain, chains[0].aa) == ("1abc.pdb_A", "A", SEQ)
    assert chains[0].states == fake_states(SEQ).upper() and len(chains[0].states) == len(SEQ)
    call = fake_calls(fake_foldseek)[-1].split()
    assert call[:7] == ["structureto3didescriptor", "-v", "0", "--threads", "1", "--chain-name-mode", "1"]
    assert call[7] == str(pdb.resolve()) and call[8].endswith("out.tsv")
    assert not list(scratch.iterdir()), "Foldseek's output and .dbtype files must be removed"


def test_the_wrapper_reads_descriptions_and_several_chains(tmp_path, fake_foldseek, monkeypatch):
    monkeypatch.setenv("FAKE_FOLDSEEK_MODE", "describe")
    two = tmp_path / "two.pdb"
    two.write_text(pdb_text_chains([("A", "MKV"), ("B", "GGA")]))
    chains = foldseek.run(fake_foldseek, two)
    assert [c.chain for c in chains] == ["A", "B"] and chains[0].name == "two.pdb_A"
    with pytest.raises(build.StructureError, match="2 chains \\(A, B\\).*structure_chain"):
        build.structure_string(Assay("a", "P", "MKV", two, "percent_0_to_100"), engine_of(fake_foldseek))
    text, info = build.structure_string(Assay("a", "P", "GGA", two, "percent_0_to_100", structure_chain="B"), engine_of(fake_foldseek))
    assert text == fake_states("GGA") and info["chain"] == "B"
    with pytest.raises(build.StructureError, match="no chain 'C'"):
        build.structure_string(Assay("a", "P", "GGA", two, "percent_0_to_100", structure_chain="C"), engine_of(fake_foldseek))


def test_unknown_states_become_hash_and_are_counted(tmp_path, fake_foldseek, monkeypatch):
    monkeypatch.setenv("FAKE_FOLDSEEK_MODE", "unknown")
    text, info = string_of(assay_with(tmp_path, 90.0, "percent_0_to_100"))
    assert text[1] == "#" and text[0] != "#" and info["unknown_states"] == 1
    assert foldseek.normalise_states("AcDX") == ("acd#", 1)
    with pytest.raises(foldseek.FoldseekError, match="unexpected 3Di character"):
        foldseek.normalise_states("AB1")


@pytest.mark.parametrize("mode, message", [("fail", "exited with status 7 on .*could not parse"),
                                           ("silent", "wrote no output"), ("badline", "expected tab-separated")])
def test_foldseek_failures_say_what_happened(tmp_path, fake_foldseek, monkeypatch, mode, message):
    monkeypatch.setenv("FAKE_FOLDSEEK_MODE", mode)
    with pytest.raises(foldseek.FoldseekError, match=message):
        foldseek.run(fake_foldseek, write_pdb(tmp_path / "a.pdb", SEQ, 90.0))


def test_a_binary_that_cannot_run_is_reported(tmp_path):
    plain = tmp_path / "foldseek"
    plain.write_text("not a program")
    plain.chmod(0o644)
    with pytest.raises(foldseek.FoldseekError, match="cannot execute"):
        foldseek.version(plain)
    broken = install_fake_foldseek(tmp_path / "b")
    os.environ["FAKE_FOLDSEEK_VERSION"] = ""
    try:
        with pytest.raises(foldseek.FoldseekError, match="version.*exited"):
            foldseek.version(broken)
    finally:
        del os.environ["FAKE_FOLDSEEK_VERSION"]


def test_locating_foldseek(tmp_path, monkeypatch):
    monkeypatch.delenv("MPD_FOLDSEEK", raising=False)
    monkeypatch.delenv("MPD_HOME", raising=False)
    home = tmp_path / "home"
    assert foldseek.locate(home) is None and foldseek.locate(None) is None
    legacy = install_fake_foldseek(home / "tools" / "foldseek" / "bin")
    assert foldseek.locate(home) == legacy
    pinned = foldseek.pinned_path(home)
    install_fake_foldseek(pinned.parent)
    assert foldseek.locate(home) == pinned                                  # the pinned layout wins
    assert str(pinned).endswith("tools/foldseek-10-941cd33/foldseek/bin/foldseek")
    monkeypatch.setenv("MPD_HOME", str(home))
    assert foldseek.locate() == pinned
    other = install_fake_foldseek(tmp_path / "elsewhere" / "bin")
    monkeypatch.setenv("MPD_FOLDSEEK", str(other))
    assert foldseek.locate(home) == other.resolve()                         # the override wins
    monkeypatch.setenv("MPD_FOLDSEEK", str(other.parent.parent))
    assert foldseek.locate(home) == other.resolve()                         # a folder holding bin/foldseek
    monkeypatch.setenv("MPD_FOLDSEEK", str(tmp_path / "nope"))
    assert foldseek.locate(home) is None                                    # set but wrong: no silent fallback
    assert "MPD_FOLDSEEK" in build.missing_foldseek(home)


def test_the_engine_describes_itself_for_structures_json(fake_foldseek):
    e = engine_of(fake_foldseek)
    d = e.describe()
    assert e.id == "foldseek-" + hashlib.sha256(fake_foldseek.read_bytes()).hexdigest()[:12]
    assert d["name"] == "foldseek" and d["version"] == "fake-941cd33" and d["binary_sha256"] == e.binary_sha256
    assert "structureto3didescriptor -v 0 --threads 1 --chain-name-mode 1" in d["command"]
    assert d["pinned_release"] is False and d["reference_build"] is False


# -- building and freezing sets ----------------------------------------------------------------------------------------

@pytest.fixture
def pg(tmp_path):
    home = tmp_path / "home"
    seqs = write_tiny_structure_home(home)
    bm = open_benchmark("proteingym", home)
    table = build_table(bm, ["ASSAY_A", "ASSAY_B", "ASSAY_C"], with_variants=False)
    return home, seqs, bm, table


def test_a_set_for_the_tiny_proteingym(pg, fake_foldseek):
    home, seqs, bm, table = pg
    needed = build.assays_for(bm, table.hashes)
    s = build.build_set({"proteingym": needed}, home, log=lambda m: None)
    p1, p2 = seqs["ASSAY_A"], seqs["ASSAY_B"]
    assert s.states[sequence_hash(p1)] == fake_states(p1)                              # all pLDDT 90
    assert s.states[sequence_hash(p2)] == "#" * 4 + fake_states(p2[4:30]) + "#" * 4    # pdb_range 5-30
    assert s.for_table(table) == [s.states[h] for h in table.hashes]
    lines = "".join(f"{h}\t{t}\n" for h, t in sorted(s.states.items()))
    assert s.digest() == hashlib.sha256(lines.encode()).hexdigest()
    meta = s.meta
    assert meta["sequences"] == 2 and meta["datasets"] == {"proteingym": 2} and meta["dms_labels_opened"] is False
    assert meta["policy"]["plddt_threshold"] == 70.0 and meta["policy"]["legacy_mask"] is False
    assert meta["engine"]["name"] == "foldseek" and len(meta["engine"]["binary_sha256"]) == 64
    assert meta["masked_residues"] == 8 and s.masked_fraction() == pytest.approx(8 / 74)
    detail = meta["per_sequence"][sequence_hash(p2)]
    assert (detail["structure"], detail["offset"], detail["residues"], detail["uncovered"]) == ("P2.pdb", 4, 26, 8)
    assert str(home) not in json.dumps(meta)                                           # provenance names files, not paths


def test_the_per_file_cache_is_shared_and_keyed_by_the_engine(pg, fake_foldseek, tmp_path):
    home, seqs, bm, table = pg
    needed = {"proteingym": build.assays_for(bm, table.hashes)}
    first = build.build_set(needed, home, log=lambda m: None)
    assert len(fake_calls(fake_foldseek)) == 2                                          # one call per structure file
    again = build.build_set(needed, home, log=lambda m: None)
    assert len(fake_calls(fake_foldseek)) == 2 and again.digest() == first.digest()      # served from <home>/cache/3di
    cache = home / "cache" / "3di" / engine_of(fake_foldseek).id
    assert len(list(cache.glob("*.json"))) == 2
    damaged = next(cache.glob("*.json"))
    damaged.write_text("{not json")
    assert build.build_set(needed, home, log=lambda m: None).digest() == first.digest()
    assert len(fake_calls(fake_foldseek)) == 3                                          # only the damaged entry was recomputed
    other = install_fake_foldseek(tmp_path / "other")                                   # another binary: its own cache directory
    build.build_set(needed, home, engine_of(other), log=lambda m: None)
    assert len(list((home / "cache" / "3di").iterdir())) == 2
    no_cache = build.build_set(needed, None, log=lambda m: None)                        # home=None: nothing is cached
    assert no_cache.digest() == first.digest()


def test_threads_do_not_change_the_result(pg, fake_foldseek):
    home, seqs, bm, table = pg
    needed = {"proteingym": build.assays_for(bm, table.hashes)}
    a = build.build_set(needed, None, threads=1, log=lambda m: None)
    b = build.build_set(needed, None, threads=8, log=lambda m: None)
    assert a.digest() == b.digest() and list(a.states) == list(b.states)


def test_every_unusable_structure_is_reported_at_once(pg, fake_foldseek):
    home, seqs, bm, table = pg
    folder = home / "data" / sources.PROTEINGYM.dirname / "ProteinGym_AF2_structures"
    write_pdb(folder / "P1.pdb", seqs["ASSAY_A"][::-1], 90.0)           # a different protein
    (folder / "P2.pdb").unlink()
    needed = {"proteingym": build.assays_for(bm, table.hashes)}
    with pytest.raises(build.StructureError) as err:
        build.build_set(needed, home, log=lambda m: None)
    text = str(err.value)
    assert "2 of 2 structure(s)" in text and "P2.pdb not found" in text and "ASSAY_A:" in text and "differ" in text


def test_a_sequence_in_two_datasets_keeps_each_datasets_structure(tmp_path, fake_foldseek):
    seq = "MKVLAGDEWQ"
    good = Assay("a1", "P", seq, write_pdb(tmp_path / "good.pdb", seq, 90.0), "percent_0_to_100")
    poor = Assay("b1", "P", seq, write_pdb(tmp_path / "poor.pdb", seq, 40.0), "percent_0_to_100")
    same = Assay("c1", "P", seq, write_pdb(tmp_path / "same.pdb", seq, 95.0), "percent_0_to_100")
    h = sequence_hash(seq)
    logs = []
    s = build.build_set({"first": {h: good}, "second": {h: poor}, "third": {h: same}}, tmp_path, log=logs.append)
    assert s.states[h] == fake_states(seq)
    assert s.scopes == {"second": {h: "#" * 10}}                          # 'third' agrees with 'first': nothing to record
    assert s.scoped("second").states[h] == "#" * 10
    assert s.scoped("first").states == s.scoped("third").states == s.scoped(None).states == s.states and not s.scoped("first").scopes
    assert s.meta["conflicts"] == [{"sequence_sha256": h, "kept_dataset": "first", "other_dataset": "second",
                                    "kept_structure": "good.pdb", "other_structure": "poor.pdb"}]
    assert any("different structure" in line for line in logs)
    plain = build.build_set({"first": {h: good}}, tmp_path, log=lambda m: None)
    assert plain.digest() != s.digest()                                  # the digest covers the dataset-specific strings


def test_assays_sharing_a_sequence_must_agree_on_their_structure(tmp_path):
    class Bench:
        name = "toy"

        def assays(self):
            return [Assay("a", "P", SEQ, tmp_path / "1.pdb", "percent_0_to_100"),
                    Assay("b", "P", SEQ, tmp_path / "2.pdb", "percent_0_to_100")]
    with pytest.raises(build.StructureError, match="share a sequence but name different structures"):
        build.assays_for(Bench(), [sequence_hash(SEQ)])
    with pytest.raises(build.StructureError, match="not in the toy benchmark"):
        build.assays_for(Bench(), ["0" * 64])


def test_a_flat_mapping_is_one_unnamed_dataset(pg, fake_foldseek):
    home, seqs, bm, table = pg
    needed = build.assays_for(bm, table.hashes)
    assert build.build_set(needed, home, log=lambda m: None).digest() == \
        build.build_set({"proteingym": needed}, home, log=lambda m: None).digest()
    with pytest.raises(ValueError, match="not the sha256"):
        build.build_set({"0" * 64: next(iter(needed.values()))}, home, log=lambda m: None)


def test_the_engine_is_chosen_by_environment(tmp_path, monkeypatch, fake_foldseek):
    assert build.open_engine(tmp_path).name == "foldseek"
    monkeypatch.setenv("MPD_3DI_ENGINE", "nonsense")
    with pytest.raises(ValueError, match="'foldseek' or 'biotite'"):
        build.open_engine(tmp_path)
    monkeypatch.delenv("MPD_3DI_ENGINE")
    monkeypatch.delenv("MPD_FOLDSEEK")
    with pytest.raises(foldseek.FoldseekError, match="mpd fetch foldseek --cluster grace"):
        build.open_engine(tmp_path, cluster="grace")


# -- the optional biotite engine ---------------------------------------------------------------------------------------------

def test_the_biotite_engine_fills_the_same_slots_as_foldseek(tmp_path, monkeypatch):
    pytest.importorskip("biotite")
    from fx_saprot import helix_pdb_text
    from mpd.structure.biotite_engine import BiotiteEngine
    seq = "MKVLAGDEWQRSTNYFHPIC"
    pdb_file = tmp_path / "h.pdb"
    pdb_file.write_text(helix_pdb_text(seq))
    engine = BiotiteEngine()
    chains = engine.chains(pdb_file)
    assert len(chains) == 1 and (chains[0].name, chains[0].chain, chains[0].aa) == ("h.pdb_A", "A", seq)
    assert len(chains[0].states) == len(seq) and set(chains[0].states) <= set("ACDEFGHIKLMNPQRSTVWY")
    assert engine.id == f"biotite-{engine.version}" and engine.describe()["name"] == "biotite" and engine.describe()["binary_sha256"] is None
    monkeypatch.setenv("MPD_3DI_ENGINE", "biotite")
    assert build.open_engine(tmp_path).name == "biotite"
    a = Assay("h", "P", seq, pdb_file, "percent_0_to_100")
    s = build.build_set({"toy": {sequence_hash(seq): a}}, tmp_path, log=lambda m: None)
    assert s.meta["engine"]["name"] == "biotite" and len(s.states[sequence_hash(seq)]) == len(seq)
    assert (tmp_path / "cache" / "3di" / engine.id).is_dir()


def test_asking_for_biotite_without_it_installed_says_how_to_get_it(monkeypatch):
    monkeypatch.setitem(sys.modules, "biotite", None)
    monkeypatch.setenv("MPD_3DI_ENGINE", "biotite")
    with pytest.raises(foldseek.EngineError, match="pip install biotite"):
        build.open_engine(None)


# -- structure sets --------------------------------------------------------------------------------------------------------

def test_structure_set_lookup_digest_and_round_trip(pg, tmp_path):
    home, seqs, bm, table = pg
    states = {sequence_hash(seq): "p" * len(seq) for seq in table.sequences}
    s = StructureSet(states, {"engine": {"name": "unit"}})
    assert s.for_table(table) == ["p" * len(q) for q in table.sequences]
    assert len(s) == 2 and s.masked_fraction() == 0.0
    path = tmp_path / "contexts" / "structures.json"
    s.save(path)
    again = StructureSet.load(path)
    assert again.states == s.states and again.meta == s.meta and again.digest() == s.digest()
    body = json.loads(path.read_text())
    assert body["schema"] == 1 and body["digest"] == s.digest()
    body["states"][next(iter(body["states"]))] = "q" * 40                 # edit one string
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="do not match the recorded digest"):
        StructureSet.load(path)
    body["schema"] = 2
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="schema"):
        StructureSet.load(path)


def test_structure_set_errors_name_the_sequence(pg):
    home, seqs, bm, table = pg
    first = table.sequences[0]
    s = StructureSet({sequence_hash(first): "p" * len(first)})
    with pytest.raises(KeyError) as err:
        s.for_table(table)
    message = err.value.args[0]
    assert table.sequences[1][:12] in message and str(len(table.sequences[1])) in message and sequence_hash(table.sequences[1])[:12] in message
    short = StructureSet({sequence_hash(q): "p" for q in table.sequences})
    with pytest.raises(ValueError, match="has 1 residues, the sequence"):
        short.for_table(table)
    with pytest.raises(ValueError, match="outside the 3Di alphabet"):
        StructureSet({"a" * 64: "pz"})


def test_the_manifest_block_and_which_datasets_need_structures(pg, fake_foldseek):
    from mpd.structure import structure_needs
    home, seqs, bm, table = pg
    s = build.build_set({"proteingym": build.assays_for(bm, table.hashes)}, home, log=lambda m: None)
    block = s.manifest_block()
    assert block["digest"] == s.digest() and block["sequences"] == 2 and block["dms_labels_opened"] is False
    assert block["masked_fraction"] == pytest.approx(8 / 74, abs=1e-6) and block["engine"]["name"] == "foldseek"
    assert block["datasets"] == {"proteingym": 2} and block["conflicts"] == 0 and block["policy"]["plddt_threshold"] == 70.0
    json.dumps(block)
    both = ("proteingym", "venusmuthub")
    assert structure_needs(spec_for(eval_data=both)) == (True, both)                       # a SaProt student reads structure everywhere
    assert structure_needs(spec_for(student="esm2-8m", teacher="saprot-650m")) == (True, ())    # a live SaProt teacher: training only
    assert structure_needs(spec_for(student="esm2-8m")) == (False, ())


def test_scoped_views_serialize(tmp_path):
    h = "a" * 64
    s = StructureSet({h: "pp"}, {}, {"venusmuthub": {h: "##"}})
    s.save(tmp_path / "s.json")
    again = StructureSet.load(tmp_path / "s.json")
    assert again.scoped("venusmuthub").states[h] == "##" and again.states[h] == "pp" and again.digest() == s.digest()
    table = ContextTable(["M" * 2], np.zeros(1, np.int32), np.zeros(1, np.int32), np.zeros(1, np.uint8))
    h = sequence_hash("MM")
    ambiguous = StructureSet({h: "pp"}, {}, {"venusmuthub": {h: "##"}})
    with pytest.raises(ValueError, match="scoped"):                      # a shared sequence needs the dataset to be named
        ambiguous.for_table(table)
    assert ambiguous.scoped("proteingym").for_table(table) == ["pp"] and ambiguous.scoped("venusmuthub").for_table(table) == ["##"]


# -- benchmark ranges ----------------------------------------------------------------------------------------------------

def reference(tmp_path, rows, extra_columns=("pdb_file", "pdb_range")):
    import csv
    fields = ["DMS_id", "DMS_filename", "UniProt_ID", "target_seq", *extra_columns]
    with open(tmp_path / "DMS_substitutions.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return ProteinGym(tmp_path)


def test_pdb_range_is_one_based_inclusive_and_clipped(tmp_path):
    row = lambda k, rng, f="a.pdb", n=800: {"DMS_id": k, "DMS_filename": k + ".csv", "UniProt_ID": k, "target_seq": "M" * n,
                                            "pdb_file": f, "pdb_range": rng}
    bm = reference(tmp_path, [row("zikv", "291-794"), row("cas9", "1-1390", n=1368), row("whole", ""), row("multi", "1-5|9-12"),
                              row("twofiles", "1-5", f="a.pdb|b.pdb"), row("nofile", "1-5", f="")])
    a = {x.assay_id: x for x in bm.assays()}
    assert a["zikv"].structure_span == (290, 794) and a["zikv"].structure == tmp_path / "ProteinGym_AF2_structures" / "a.pdb"
    assert a["cas9"].structure_span == (0, 1368)                          # the range is longer than the sequence
    assert a["whole"].structure_span is None
    assert a["zikv"].structure_scale == "percent_0_to_100" and a["zikv"].structure_chain is None
    for k in ("multi", "twofiles", "nofile"):                             # several structures or none: no structure at all
        assert (a[k].structure, a[k].structure_span, a[k].structure_scale) == (None, None, None)
    for bad, message in (("abc", "not of the form"), ("0-5", "does not fit"), ("9-3", "does not fit"), ("900-1000", "does not fit")):
        with pytest.raises(ValueError, match=message):
            reference(tmp_path, [row("x", bad)]).assays()
    assert pdb_span("x", " 3 - 9 ", 20) == (2, 9) and pdb_span("x", "", 20) is None


def test_data_without_a_pdb_range_column_means_the_whole_sequence(tmp_path):
    from tinydata import write_tiny_proteingym
    write_tiny_proteingym(tmp_path)                                       # the tiny fixture has no pdb_range column
    a = ProteinGym(tmp_path).assay("ASSAY_A")
    assert a.structure_span is None and a.structure_scale == "percent_0_to_100" and a.structure.name == "P1.pdb"
    bm = reference(tmp_path, [{"DMS_id": "q", "DMS_filename": "q.csv", "UniProt_ID": "Q", "target_seq": "MKV"}], extra_columns=())
    assert bm.assay("q").structure is None and bm.assay("q").structure_span is None


def test_venus_assays_carry_the_scale_of_their_structure_source(tmp_path):
    from mpd.data.venusmuthub import VenusMutHub, protein_id
    wt = "MKLVAGDEWQ"
    root = tmp_path / "venus" / "single_mutant" / "stability"
    root.mkdir(parents=True)
    (root / "S1.csv").write_text(f"mutant,mutated_sequence,fitness_score\nK2A,{wt[:1]}A{wt[2:]},1.0\n")
    pid = protein_id(wt)
    for source, scale in (("alphafold_db", "percent_0_to_100"), ("esmfold", "fraction_0_to_1"), ("experimental", "bfactor")):
        structures = tmp_path / source
        structures.mkdir()
        vs.write_manifest([{"protein_id": pid, "source": source, "confidence_scale": scale,
                            "file": f"structures/{source}/{pid}.pdb", "file_sha256": "0" * 64}], structures / vs.MANIFEST_NAME)
        a = VenusMutHub(tmp_path / "venus", structures).assay("S1")
        assert a.structure_scale == scale and (a.structure_span, a.structure_chain) == (None, None)


# -- preflight ------------------------------------------------------------------------------------------------------------

def spec_for(student="saprot-35m", teacher="esm2-35m", train="proteingym", eval_data=("proteingym",)):
    return RunSpec(teacher=teacher, student=student, train_data=train, method="direct-all", eval_data=eval_data)


def by_name(checks):
    return {c.name: c for c in checks}


def test_preflight_has_nothing_to_say_about_sequence_only_runs(tmp_path):
    assert preflight_checks(spec_for(student="esm2-8m"), tmp_path, "grace") == []


def test_preflight_names_the_fix_for_each_missing_piece(tmp_path, monkeypatch):
    monkeypatch.delenv("MPD_FOLDSEEK", raising=False)
    checks = by_name(preflight_checks(spec_for(eval_data=("proteingym", "venusmuthub")), tmp_path, "grace"))
    assert set(checks) == {"foldseek", "structures:proteingym", "structures:venusmuthub"}
    assert checks["foldseek"].level == "error" and "Run on a login node: mpd fetch foldseek --cluster grace" in checks["foldseek"].message
    assert checks["foldseek"].fix == "mpd fetch foldseek --cluster grace"
    assert checks["structures:proteingym"].fix == "mpd fetch proteingym --cluster grace"
    assert "Run on a login node: mpd fetch proteingym --cluster grace" in checks["structures:proteingym"].message
    venus = checks["structures:venusmuthub"]
    assert venus.level == "error" and "not published yet" in venus.message and "--from-dir" in venus.fix
    # without a cluster name the fix names the home
    plain = by_name(preflight_checks(spec_for(), tmp_path, None))
    assert plain["foldseek"].fix == f"mpd fetch foldseek --home {tmp_path}"


HOME = "/scratch/user/netid/mpd"                       # a placeholder, not anybody's directory


@pytest.mark.parametrize("cluster, home, explicit, expect", [
    ("grace", HOME, False, "mpd fetch foldseek --cluster grace"),                       # the profile's default home: not repeated
    ("grace", HOME, True, f"mpd fetch foldseek --cluster grace --home {HOME}"),         # --home was given: repeated
    (None, HOME, False, f"mpd fetch foldseek --home {HOME}"),                           # no cluster name to put the home behind
    (None, None, False, "mpd fetch foldseek --cluster <cluster>"),                      # neither: a placeholder to fill in
    (None, None, True, "mpd fetch foldseek --cluster <cluster>"),
    ("grace", "/scratch/user/netid/my mpd", True, "mpd fetch foldseek --cluster grace --home '/scratch/user/netid/my mpd'"),
])
def test_the_foldseek_fix_command_has_four_forms(cluster, home, explicit, expect, monkeypatch):
    monkeypatch.delenv("MPD_FOLDSEEK", raising=False)
    assert foldseek.fix_command(cluster, home, explicit_home=explicit) == expect
    said = build.missing_foldseek(home, cluster, explicit_home=explicit)
    assert f"Run on a login node: {expect} " in said and "MPD_FOLDSEEK" in said            # and the way to use a Foldseek you have


@pytest.mark.parametrize("cluster, explicit, flags", [
    ("grace", False, "--cluster grace"),
    ("grace", True, f"--cluster grace --home {HOME}"),
    (None, False, f"--home {HOME}"),
])
def test_every_structure_preflight_fix_repeats_the_home_only_when_it_must(cluster, explicit, flags, monkeypatch):
    monkeypatch.delenv("MPD_FOLDSEEK", raising=False)
    spec = spec_for(eval_data=("proteingym", "venusmuthub"))
    checks = by_name(preflight_checks(spec, Path(HOME), cluster, explicit_home=explicit))
    assert checks["foldseek"].fix == f"mpd fetch foldseek {flags}"
    assert checks["structures:proteingym"].fix == f"mpd fetch proteingym {flags}"
    assert checks["structures:venusmuthub"].fix == f"mpd fetch venus-structures {flags} --from-dir <folder with the deposit files>"
    for check in checks.values():
        assert "mpd fetch" in check.message and check.fix in check.message, check.name


def test_the_proteingym_structures_come_with_the_proteingym_download(tmp_path):
    """`mpd fetch proteingym` downloads and unpacks ProteinGym_AF2_structures.zip (sources.PROTEINGYM), so that is the fix."""
    archive = next(f for f in sources.PROTEINGYM.files if f.name == "ProteinGym_AF2_structures.zip")
    assert archive.name in sources.PROTEINGYM.unpack and archive.size == 19_173_248
    message = by_name(preflight_checks(spec_for(), tmp_path, "grace"))["structures:proteingym"].message
    assert "ProteinGym_AF2_structures.zip, 19.2 MB" in message and "part of the ProteinGym download" in message
    assert message.endswith("Run on a login node: mpd fetch proteingym --cluster grace")


def test_preflight_passes_when_everything_is_in_place(tmp_path, fake_foldseek, monkeypatch):
    write_tiny_structure_home(tmp_path)
    manifest = tmp_path / "data" / sources.VENUS_STRUCTURES.dirname
    manifest.mkdir(parents=True)
    (manifest / vs.MANIFEST_NAME).write_text("protein_id\n")
    checks = by_name(preflight_checks(spec_for(eval_data=("proteingym", "venusmuthub")), tmp_path, "grace"))
    assert checks["structures:proteingym"].level == "ok" and checks["structures:venusmuthub"].level == "ok"
    assert checks["foldseek"].level == "warn" and "not the pinned release 10-941cd33" in checks["foldseek"].message
    # the pinned binary and the binary behind the recorded results are both fine
    monkeypatch.setattr(sources, "FOLDSEEK", dataclasses.replace(sources.FOLDSEEK, binary_sha256=engine_of(fake_foldseek).binary_sha256))
    ok = by_name(preflight_checks(spec_for(), tmp_path, "grace"))["foldseek"]
    assert ok.level == "ok" and "pinned release 10-941cd33" in ok.message and "fake-941cd33" in ok.message


def test_preflight_reports_a_foldseek_that_does_not_run(tmp_path, monkeypatch):
    dead = tmp_path / "dead"
    dead.write_text("#!/bin/sh\necho no >&2\nexit 5\n")
    dead.chmod(0o755)
    monkeypatch.setenv("MPD_FOLDSEEK", str(dead))
    check = by_name(preflight_checks(spec_for(), tmp_path, "grace"))["foldseek"]
    assert check.level == "error" and "does not run" in check.message
    monkeypatch.setenv("MPD_FOLDSEEK", str(tmp_path / "missing"))
    check = by_name(preflight_checks(spec_for(), tmp_path, "grace"))["foldseek"]
    assert check.level == "error" and "MPD_FOLDSEEK=" in check.message


def test_preflight_only_asks_for_the_structures_a_run_reads(tmp_path, fake_foldseek):
    # a SaProt teacher with an ESM2 student: training sequences need structures, evaluation does not
    teacher_only = by_name(preflight_checks(spec_for(student="esm2-8m", teacher="saprot-650m", train="venusmuthub",
                                                      eval_data=("proteingym",)), tmp_path, "grace"))
    assert set(teacher_only) == {"foldseek", "structures:venusmuthub"}
    # a SaProt student trained on ProteinGym and scored on Venus needs both
    student = by_name(preflight_checks(spec_for(eval_data=("venusmuthub",)), tmp_path, "grace"))
    assert set(student) == {"foldseek", "structures:proteingym", "structures:venusmuthub"}


def test_preflight_biotite_engine(tmp_path, fake_foldseek, monkeypatch):
    monkeypatch.setenv("MPD_3DI_ENGINE", "biotite")
    monkeypatch.setitem(sys.modules, "biotite", None)                      # not installed
    check = by_name(preflight_checks(spec_for(), tmp_path, "grace"))["3di-engine"]
    assert check.level == "error" and check.fix == "pip install biotite"
    monkeypatch.setitem(sys.modules, "biotite", type(sys)("biotite"))
    sys.modules["biotite"].__version__ = "9.9"
    check = by_name(preflight_checks(spec_for(), tmp_path, "grace"))["3di-engine"]
    assert check.level == "warn" and "biotite 9.9" in check.message


# -- fetching the pinned Foldseek --------------------------------------------------------------------------------------------

def test_fetch_installs_a_verified_release_and_is_repeatable_offline(tmp_path, monkeypatch):
    monkeypatch.delenv("MPD_FOLDSEEK", raising=False)
    archive, pin = fake_release(tmp_path / "srv")
    home, logs = tmp_path / "home", []
    with serve(tmp_path / "srv") as base:
        binary = tools.fetch_foldseek(home, pin=pin, url=f"{base}/{pin.archive}", log=logs.append)
    assert binary == home / "tools" / "foldseek-10-941cd33" / "foldseek" / "bin" / "foldseek" and is_executable(binary)
    record = json.loads((binary.parents[2] / "FETCHED.json").read_text())
    assert (record["release"], record["version_output"], record["version_names_release"]) == ("10-941cd33", "fake-941cd33", True)
    assert record["binary_sha256"] == pin.binary_sha256 and record["license"] == "GPL-3.0" and str(home) not in json.dumps(record)
    assert [p.name for p in (home / "tools").iterdir()] == ["foldseek-10-941cd33"]        # no staging or download leftovers
    assert any("never bundled" in line for line in logs) and foldseek.locate(home) == binary
    again = tools.fetch_foldseek(home, pin=pin, url="http://127.0.0.1:9/never", log=lambda m: None)    # nothing is downloaded
    assert again == binary


def test_fetch_refuses_an_archive_that_is_not_the_pinned_one(tmp_path):
    archive, pin = fake_release(tmp_path / "srv")
    home = tmp_path / "home"
    with serve(tmp_path / "srv") as base:
        with pytest.raises(FetchError, match="sha256 mismatch.*removed the download"):
            tools.fetch_foldseek(home, pin=dataclasses.replace(pin, archive_sha256="0" * 64), url=f"{base}/{pin.archive}", log=lambda m: None)
        with pytest.raises(FetchError, match="size .* != expected"):
            tools.fetch_foldseek(home, pin=dataclasses.replace(pin, archive_size=1), url=f"{base}/{pin.archive}", log=lambda m: None)
    assert list((home / "tools").iterdir()) == []


def test_fetch_never_runs_a_binary_that_does_not_match_its_pin(tmp_path):
    marker = tmp_path / "ran"
    archive, pin = fake_release(tmp_path / "srv", marker=marker)
    wrong = dataclasses.replace(pin, binary_sha256="0" * 64)               # the archive is right, the binary is not what is pinned
    with pytest.raises(FetchError, match="does not match the pinned sha256; nothing was run"):
        tools.fetch_foldseek(tmp_path / "home", pin=wrong, from_file=archive, log=lambda m: None)
    assert not marker.exists() and not (tmp_path / "home" / "tools" / wrong.dirname).exists()
    assert archive.is_file(), "a file the user supplied is never deleted"
    tools.fetch_foldseek(tmp_path / "home", pin=pin, from_file=archive, log=lambda m: None)
    assert marker.exists()                                                  # after verification it runs (`foldseek version`)


def test_fetch_accepts_a_local_archive_or_the_folder_holding_it(tmp_path):
    archive, pin = fake_release(tmp_path / "srv")
    one = tools.fetch_foldseek(tmp_path / "h1", pin=pin, from_file=archive, log=lambda m: None)
    two = tools.fetch_foldseek(tmp_path / "h2", pin=pin, from_file=archive.parent, log=lambda m: None)
    assert is_executable(one) and is_executable(two)
    with pytest.raises(FetchError, match="not found"):
        tools.fetch_foldseek(tmp_path / "h3", pin=pin, from_file=tmp_path / "empty", log=lambda m: None)


def test_the_executable_bit_comes_back_even_if_the_archive_lost_it(tmp_path):
    archive, pin = fake_release(tmp_path / "srv", binary_mode=0o644)
    binary = tools.fetch_foldseek(tmp_path / "home", pin=pin, from_file=archive, log=lambda m: None)
    assert is_executable(binary) and not (binary.stat().st_mode & 0o022)


def test_tar_extraction_keeps_the_executable_bit_and_nothing_more(tmp_path):
    import io
    import tarfile
    archive = tmp_path / "t.tar"
    with tarfile.open(archive, "w") as tar:
        for name, mode in (("bin/run", 0o755), ("bin/setuid", 0o4777), ("data/plain", 0o644)):
            info = tarfile.TarInfo(name)
            info.size, info.mode = 1, mode
            tar.addfile(info, io.BytesIO(b"x"))
    vs.safe_extract_tar(archive, tmp_path / "out")
    modes = {n: (tmp_path / "out" / n).stat().st_mode & 0o7777 for n in ("bin/run", "bin/setuid", "data/plain")}
    assert modes["bin/run"] == 0o755 and modes["bin/setuid"] == 0o755 and not modes["data/plain"] & 0o111


def test_fetch_refuses_an_archive_with_an_unsafe_member(tmp_path):
    archive, pin = fake_release(tmp_path / "srv", extra=[("../evil", b"x", 0o644)])
    with pytest.raises(FetchError, match="unsafe member path"):
        tools.fetch_foldseek(tmp_path / "home", pin=pin, from_file=archive, log=lambda m: None)
    assert list((tmp_path / "home" / "tools").iterdir()) == [] and not (tmp_path / "evil").exists()


def test_fetch_says_when_the_verified_binary_does_not_run(tmp_path):
    archive, pin = fake_release(tmp_path / "srv", body="echo broken >&2\nexit 3\n")
    with pytest.raises(FetchError, match="does not run here.*noexec"):
        tools.fetch_foldseek(tmp_path / "home", pin=pin, from_file=archive, log=lambda m: None)


def test_fetch_warns_when_the_version_does_not_name_the_release(tmp_path):
    archive, pin = fake_release(tmp_path / "srv", body='echo "something-else"\n')
    logs = []
    tools.fetch_foldseek(tmp_path / "home", pin=pin, from_file=archive, log=logs.append)
    record = json.loads((tmp_path / "home/tools/foldseek-10-941cd33/FETCHED.json").read_text())
    assert record["version_names_release"] is False and any("does not contain the release commit" in l for l in logs)


def test_fetch_waits_out_rate_limits_and_explains_a_dead_server(tmp_path, monkeypatch):
    sleeps = []
    monkeypatch.setattr(tools, "time", type("T", (), {"sleep": staticmethod(sleeps.append)}))
    archive, pin = fake_release(tmp_path / "srv")
    with serve(tmp_path / "srv", throttle=1) as base:
        tools.fetch_foldseek(tmp_path / "home", pin=pin, url=f"{base}/{pin.archive}", log=lambda m: None)
    assert sleeps == [7]                                                    # Retry-After was honoured
    sleeps.clear()
    with pytest.raises(FetchError, match="could not download.*--from-dir"):
        tools.fetch_foldseek(tmp_path / "home2", pin=pin, url="http://127.0.0.1:9/none", log=lambda m: None)
    assert len(sleeps) == tools.RETRIES - 1


def test_the_foldseek_pin_is_well_formed():
    pin = sources.FOLDSEEK
    assert pin.url.startswith("https://github.com/steineggerlab/foldseek/releases/download/10-941cd33/")
    assert pin.commit.startswith("941cd33") and pin.license == "GPL-3.0" and sources.get_tool("foldseek") is pin
    assert len(pin.archive_sha256) == len(pin.binary_sha256) == 64 and pin.binary == "foldseek/bin/foldseek"
    assert pin.dirname == "foldseek-10-941cd33" and "foldseek" not in sources.SOURCES        # not a data source of `mpd fetch`
    with pytest.raises(ValueError, match="unknown tool"):
        sources.get_tool("nope")


# -- the planner stays torch-free ---------------------------------------------------------------------------------------------

def test_the_structure_modules_import_without_torch():
    src = str(Path(__file__).resolve().parents[1] / "src")
    code = ("import importlib, sys\n"
            "for m in ('mpd.structure', 'mpd.structure.vocab', 'mpd.structure.pdb', 'mpd.structure.align', 'mpd.structure.foldseek',\n"
            "          'mpd.structure.build', 'mpd.structure.set', 'mpd.structure.tools', 'mpd.data.sources', 'mpd.data.benchmarks'):\n"
            "    importlib.import_module(m)\n"
            "assert 'torch' not in sys.modules, 'torch was imported'\n")
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": src})
    assert done.returncode == 0, done.stderr[-2000:]


def test_structure_code_never_reads_labels():
    root = Path(__file__).resolve().parents[1] / "src" / "mpd" / "structure"
    for path in sorted(root.glob("*.py")):
        assert ".labels(" not in path.read_text() and "DMS_score" not in path.read_text(), path.name
