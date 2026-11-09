"""The Saprot model class on a tiny random SaProt-shaped model: tokens, masking, windows, the AA20 log-marginal, dropout,
LoRA and adapters, plus the library-level flow prepare -> teacher -> student takes (structure set -> encode -> logits)."""

import json
import random
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

import torch

from fx_saprot import fake_foldseek, make_tiny_saprot, tiny_saprot_models, write_tiny_structure_home  # noqa: F401
from mpd import students
from mpd.contexts import ContextTable, build_table, sequence_hash
from mpd.data.benchmarks import open_benchmark
from mpd.structure import build, vocab
from mpd.structure.set import StructureSet
from mpd.students.esm2 import optimal_window
from mpd.students.saprot import Saprot

CPU = torch.device("cpu")
AA = "ACDEFGHIKLMNPQRSTVWY"
TOKENS = vocab.build_vocab()


@pytest.fixture
def env(monkeypatch, tiny_saprot_models):
    monkeypatch.setenv("MPD_DTYPE", "float32")
    monkeypatch.setenv("MPD_MODEL_SAPROT_35M", str(tiny_saprot_models["student"]))
    monkeypatch.setenv("MPD_MODEL_SAPROT_650M", str(tiny_saprot_models["teacher"]))
    monkeypatch.delenv("MPD_ALLOW_HUB", raising=False)
    return tiny_saprot_models


def table_of(seqs, contexts):
    """ContextTable for [(sequence index, position)]."""
    return ContextTable(list(seqs), np.asarray([s for s, _ in contexts], dtype=np.int32),
                        np.asarray([p for _, p in contexts], dtype=np.int32),
                        np.asarray([AA.index(seqs[s][p]) for s, p in contexts], dtype=np.uint8))


def set_of(seqs, states):
    return StructureSet({sequence_hash(s): st for s, st in zip(seqs, states)})


def random_states(rng, n):
    return "".join(rng.choice(vocab.STATES) for _ in range(n))


def loaded(seqs, states, contexts, key="saprot-35m", **kw):
    table = table_of(seqs, contexts)
    model = Saprot(key, CPU, **kw)
    model.encode(table, set_of(seqs, states))
    return model, table


def test_every_residue_is_one_token_and_the_focal_one_keeps_only_its_state(env):
    seqs = ["MKVLAGDEWQ", "ACDEFGHI"]
    states = ["pynwrqhgd#", "lvtmfsae"]
    model, table = loaded(seqs, states, [(0, 2), (0, 9), (1, 4)])
    assert [len(t) for t in model._tokens] == [12, 10]                    # <cls> + residues + <eos>
    ids, att, focal = model._batch(table, np.arange(3))
    assert focal.tolist() == [3, 10, 5] and ids.shape == (3, 12)
    assert TOKENS[ids[0, focal[0]]] == "#n" and TOKENS[ids[1, focal[1]]] == "##" and TOKENS[ids[2, focal[2]]] == "#f"
    assert [TOKENS[i] for i in ids[0]] == ["<cls>", "M" + "p", "K" + "y", "#n", "L" + "w", "A" + "r", "G" + "q", "D" + "h", "E" + "g",
                                           "W" + "d", "Q" + "#", "<eos>"]
    assert att[2].tolist() == [1] * 10 + [0] * 2 and ids[2, 10:].tolist() == [vocab.PAD] * 2
    assert not (ids == vocab.MASK).any(), "the <mask> token is never used"


def test_a_long_sequence_is_cut_to_the_1024_token_window_around_the_focal_residue(env):
    rng = random.Random(3)
    seq = "".join(rng.choice(AA) for _ in range(1100))
    state = random_states(rng, 1100)
    positions = [0, 300, 549, 560, 1099]
    model, table = loaded([seq], [state], [(0, p) for p in positions])
    ids, att, focal = model._batch(table, np.arange(len(positions)))
    assert ids.shape == (5, 1024) and bool(att.all())
    for row, pos in enumerate(positions):
        window, f = optimal_window(model._tokens[0], pos)
        assert len(window) == 1024 and focal[row].item() == f
        assert ids[row, f].item() == vocab.token_id("#", state[pos])
        expect = list(window)
        expect[f] = vocab.token_id("#", state[pos])
        assert ids[row].tolist() == expect
    assert [focal[i].item() for i in range(5)] == [1, 301, 512, 512, 1022]        # start, middle (centred), end of the sequence
    out = model.logits(table, np.arange(len(positions)))
    assert out.shape == (5, 20) and torch.isfinite(out).all()


def test_logits_are_the_log_marginals_over_the_21_states(env):
    rng = random.Random(1)
    seqs = ["".join(rng.choice(AA) for _ in range(30)), "".join(rng.choice(AA) for _ in range(25))]
    states = [random_states(rng, 30), random_states(rng, 25)]
    model, table = loaded(seqs, states, [(0, 3), (0, 29), (1, 0), (1, 12)])
    model.module.eval()
    with torch.no_grad():
        out = model.logits(table, np.arange(4))
        ids, att, focal = model._batch(table, np.arange(4))
        raw = model.module(input_ids=ids, attention_mask=att).logits.float()
    block = torch.as_tensor(vocab.aa_state_ids())
    manual = torch.stack([torch.logsumexp(raw[i, focal[i]][block], dim=-1) for i in range(4)])
    assert out.shape == (4, 20) and out.dtype == torch.float32 and torch.allclose(out, manual, atol=1e-6)
    # the batch does not change a context's answer (right padding is masked out)
    for row in (1, 3):
        assert torch.allclose(model.logits(table, np.asarray([row]))[0], out[row], atol=1e-5)


def test_the_marginal_difference_is_the_official_mutation_score(env):
    """SaProt's official score is log(sum_s p(mut+s)) - log(sum_s p(wt+s)) with p the softmax over all 446 tokens."""
    rng = random.Random(5)
    seqs = ["".join(rng.choice(AA) for _ in range(20))]
    model, table = loaded(seqs, [random_states(rng, 20)], [(0, 4), (0, 11), (0, 19)])
    ids, att, focal = model._batch(table, np.arange(3))
    torch.manual_seed(0)
    crafted = torch.randn(3, ids.shape[1], 446) * 3
    out = model.logits(table, np.arange(3), forward=lambda input_ids, attention_mask: SimpleNamespace(logits=crafted))
    block = vocab.aa_state_ids()
    for i in range(3):
        p = torch.softmax(crafted[i, focal[i]].double(), dim=-1)
        wt = int(table.wt[i])
        for mut in range(20):
            official = torch.log(p[block[mut]].sum()) - torch.log(p[block[wt]].sum())
            assert abs((out[i, mut] - out[i, wt]).item() - official.item()) < 1e-5


def reference_log_marginals(folder, seq, state, positions):
    """The earlier study's recipe written out on its own: ids straight from vocab.txt, no mpd.structure code."""
    from transformers import AutoModelForMaskedLM
    t2i = {t: i for i, t in enumerate(l.rstrip("\n") for l in open(folder / "vocab.txt"))}
    model = AutoModelForMaskedLM.from_pretrained(folder, torch_dtype=torch.float32, attn_implementation="eager",
                                                 hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0).eval()
    ids = [t2i["<cls>"]] + [t2i[a + s] for a, s in zip(seq, state)] + [t2i["<eos>"]]
    out = []
    for p in positions:
        t = p + 1
        if len(ids) <= 1024:
            window, f = list(ids), t
        else:
            start = 0 if t < 512 else (len(ids) - 1024 if t >= len(ids) - 512 else t - 512)
            window, f = ids[start:start + 1024], t - start
        window[f] = t2i["#" + state[p]]
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([window])).logits[0, f]
        out.append(torch.stack([torch.logsumexp(logits[[t2i[a + s] for s in vocab.STATES]], 0) for a in AA]))
    return torch.stack(out)


def test_the_class_computes_what_the_earlier_recipe_computes(env):
    rng = random.Random(11)
    for length, positions in ((40, [0, 7, 39]), (1100, [3, 560, 1099])):
        seq = "".join(rng.choice(AA) for _ in range(length))
        state = random_states(rng, length)
        model, table = loaded([seq], [state], [(0, p) for p in positions])
        mine = model.logits(table, np.arange(len(positions)))
        assert torch.allclose(mine, reference_log_marginals(env["student"], seq, state, positions), atol=1e-5)


def test_dropout_is_zero_even_though_the_checkpoint_says_otherwise(env):
    from transformers import AutoModelForMaskedLM
    raw = AutoModelForMaskedLM.from_pretrained(env["student"])
    assert raw.config.hidden_dropout_prob == 0.1 and {m.p for m in raw.modules() if isinstance(m, torch.nn.Dropout)} == {0.1}
    model = Saprot("saprot-35m", CPU)
    assert {m.p for m in model.module.modules() if isinstance(m, torch.nn.Dropout)} == {0.0}
    assert model.module.config.hidden_dropout_prob == 0.0 and model.module.config.attention_probs_dropout_prob == 0.0
    seqs = ["MKVLAGDEWQ"]
    m2, table = loaded(seqs, ["pynwrqhgd#"], [(0, 2), (0, 5)])
    m2.module.train()
    assert torch.equal(m2.logits(table, np.arange(2)), m2.logits(table, np.arange(2)))        # training mode is deterministic


def test_lora_parameters_and_the_adapter_round_trip(env, tmp_path):
    seqs = ["MKVLAGDEWQRS"]
    states = ["pynwrqhgd#lv"]
    contexts = [(0, 1), (0, 6), (0, 10)]
    student, table = loaded(seqs, states, contexts, lora={"rank": 8, "alpha": 16})
    trainable = student.trainable()
    assert len(trainable) == 16 and sum(p.numel() for p in trainable) == 4096
    assert all(("lora_" in n) == p.requires_grad for n, p in student.module.named_parameters())      # only the adapter trains
    base, _ = loaded(seqs, states, contexts)
    assert torch.allclose(student.logits(table, np.arange(3)), base.logits(table, np.arange(3)), atol=1e-6)    # B starts at zero
    with torch.no_grad():
        for p in trainable:
            p.add_(0.05 * torch.randn_like(p))
    tuned = student.logits(table, np.arange(3))
    assert not torch.allclose(tuned, base.logits(table, np.arange(3)), atol=1e-4)
    student.save_adapter(tmp_path / "ckpt")
    assert (tmp_path / "ckpt" / "adapter_model.safetensors").is_file()
    config = json.loads((tmp_path / "ckpt" / "adapter_config.json").read_text())
    assert (config["r"], config["lora_alpha"], config["base_model_name_or_path"]) == (8, 16, "westlake-repl/SaProt_35M_AF2")
    assert config["revision"] is None                                     # a MPD_MODEL_* override is not the pinned checkpoint
    for f in (tmp_path / "ckpt").glob("*.json"):
        assert str(env["student"]) not in f.read_text(), f"{f.name} names the local model folder"
    if (tmp_path / "ckpt" / "README.md").is_file():
        assert str(env["student"]) not in (tmp_path / "ckpt" / "README.md").read_text()
    again, _ = loaded(seqs, states, contexts, adapter=tmp_path / "ckpt")
    assert torch.allclose(again.logits(table, np.arange(3)), tuned, atol=1e-6)
    assert again.trainable() == []
    with pytest.raises(ValueError, match="either lora"):
        Saprot("saprot-35m", CPU, lora={"rank": 8, "alpha": 16}, adapter=tmp_path / "ckpt")


def test_a_structure_set_is_required_and_scoped_views_are_honoured(env):
    seqs = ["MKVLAGDEWQRS"]
    model = Saprot("saprot-35m", CPU)
    table = table_of(seqs, [(0, 3)])
    with pytest.raises(ValueError, match="reads protein structure"):
        model.encode(table)
    h = sequence_hash(seqs[0])
    full = StructureSet({h: "pynwrqhgd#lv"}, {}, {"venusmuthub": {h: "#" * 12}})
    with pytest.raises(ValueError, match="scoped"):                        # a sequence two datasets structure differently
        model.encode(table, full)
    model.encode(table, full.scoped("proteingym"))
    own = model.logits(table, np.arange(1))
    model.encode(table, full.scoped("venusmuthub"))
    other = model.logits(table, np.arange(1))
    assert not torch.allclose(own, other, atol=1e-5)


def test_only_a_saprot_vocabulary_is_accepted(env, tmp_path, monkeypatch):
    from tinydata import make_tiny_esm
    esm = make_tiny_esm(tmp_path / "esm", seed=0)
    monkeypatch.setenv("MPD_MODEL_SAPROT_35M", str(esm))
    with pytest.raises(vocab.VocabError, match="not a SaProt AA\\+3Di vocabulary"):
        Saprot("saprot-35m", CPU)
    (esm / "vocab.txt").unlink()                                            # no vocab.txt: the model's own size is checked
    with pytest.raises(RuntimeError, match="the model has 33 tokens"):
        Saprot("saprot-35m", CPU)


def test_students_load_dispatches_on_the_family(env):
    assert isinstance(students.load("saprot-35m", CPU), Saprot) and students.load("saprot-35m", CPU).family == "saprot"
    assert isinstance(students.load("saprot-650m", CPU), Saprot)


# -- the library calls the stages will make ------------------------------------------------------------------------------

@pytest.fixture
def pg(tmp_path, env, fake_foldseek):
    home = tmp_path / "home"
    write_tiny_structure_home(home)
    bm = open_benchmark("proteingym", home)
    table = build_table(bm, ["ASSAY_A", "ASSAY_B", "ASSAY_C"], with_variants=False, dense=False)
    structures = build.build_set({"proteingym": build.assays_for(bm, table.hashes)}, home, log=lambda m: None)
    return table, structures


def logits_of(model, table, structures, batch=8):
    model.encode(table, structures)
    model.module.eval()
    out = np.zeros((len(table), 20), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(table), batch):
            rows = np.arange(len(table))[start:start + batch]
            out[rows] = model.logits(table, rows).numpy()
    return out


def test_a_teacher_identical_to_the_student_agrees_exactly_and_a_different_one_does_not(pg, monkeypatch, env):
    table, structures = pg
    teacher = logits_of(Saprot("saprot-35m", CPU), table, structures)
    student = logits_of(Saprot("saprot-35m", CPU), table, structures)
    assert teacher.shape == (len(table), 20) and np.isfinite(teacher).all() and np.array_equal(teacher, student)
    other = logits_of(Saprot("saprot-650m", CPU), table, structures)
    assert not np.allclose(teacher, other, atol=1e-3)


def test_scoring_a_subset_gives_the_same_logits_as_the_full_table(pg):
    table, structures = pg
    full = logits_of(Saprot("saprot-35m", CPU), table, structures)
    rows = np.arange(len(table))[1::3]
    sub = table.subset(rows)                              # what the score stage does: keeps `sequences`, so lookups by hash work
    part = logits_of(Saprot("saprot-35m", CPU), sub, structures, batch=4)
    assert np.allclose(full[rows], part, atol=1e-5)


def test_structure_strings_change_the_prediction(pg):
    table, structures = pg
    base = logits_of(Saprot("saprot-35m", CPU), table, structures)
    blind = StructureSet({h: "#" * len(s) for h, s in structures.states.items()})
    assert not np.allclose(base, logits_of(Saprot("saprot-35m", CPU), table, blind), atol=1e-4)
