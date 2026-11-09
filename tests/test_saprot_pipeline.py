"""SaProt through the real stages on CPU: tiny random SaProt and ESM2 models, the miniature ProteinGym with PDB files, a
fake Foldseek. The same style as test_stages.py; what is new is where the structure strings come from and who may read them."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

import torch

from fx_saprot import (fake_calls, fake_foldseek, fake_states, no_network, tiny_saprot_models,  # noqa: F401
                       write_tiny_structure_home, write_tiny_venus)
from fx_stages import Killed, curve, finish_ddp_train, kill_after, plan_for, run_stages, start_ddp_train
from mpd.contexts import ContextTable, build_table, sequence_hash, union
from mpd.data.benchmarks import ProteinGym, open_benchmark
from mpd.data.models import ModelNotReady
from mpd.data.venusmuthub import VenusMutHub
from mpd.stages import common, train
from mpd.stages.teacher import aa20_logits
from mpd.structure.set import StructureSet
from mpd.students.saprot import Saprot

SRC = Path(__file__).resolve().parents[1] / "src" / "mpd"
CPU = torch.device("cpu")
OWN = "ACDEFGHIKLMNPQRSTVWYACDEFGHIKL"                        # a VenusMutHub wild type that ProteinGym does not have
# union_digest of the tiny ProteinGym's training contexts as the stages produced it before SaProt existed (dense = the direct-*
# methods, observed = the score-* methods). A campaign that reads no structure must keep these.
UNION_DIGEST = {"direct-all": "e5f8dba6831a2b1d04c5affdcb07d1af14f488fcc19526ba852eba25b3f90c97",
                "score-all": "6e493d02236ee6c37dd11b708fe865c6fd110ac89a1e9e51ff7c6678e77908a2"}


@pytest.fixture
def tiny(monkeypatch, tmp_path, tiny_models, tiny_saprot_models, fake_foldseek):
    """Tiny ESM2 and SaProt weights (MPD_MODEL_*), the tiny ProteinGym with its structures, and a fake Foldseek (MPD_FOLDSEEK)."""
    home = tmp_path / "home"
    seqs = write_tiny_structure_home(home)
    monkeypatch.setenv("MPD_HOME", str(home))
    monkeypatch.setenv("MPD_DTYPE", "float32")
    monkeypatch.setenv("MPD_MODEL_ESM2_8M", str(tiny_models["student"]))
    monkeypatch.setenv("MPD_MODEL_ESM2_35M", str(tiny_models["teacher"]))
    monkeypatch.setenv("MPD_MODEL_SAPROT_35M", str(tiny_saprot_models["student"]))
    monkeypatch.setenv("MPD_MODEL_SAPROT_650M", str(tiny_saprot_models["teacher"]))
    for variable in ("MPD_ALLOW_HUB", "MPD_MODELS_DIR", "MPD_CLUSTER", "MPD_3DI_ENGINE", "MPD_STRUCTURE_LEGACY_MASK"):
        monkeypatch.delenv(variable, raising=False)
    return SimpleNamespace(home=home, seqs=seqs, foldseek=fake_foldseek)


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def saved_structures(plan) -> StructureSet:
    return StructureSet.load(plan.campaign_dir / "contexts" / "structures.json")


def edit_structures(root: Path) -> None:
    """Change one residue's state in structures.json, as another Foldseek build could."""
    path = root / "contexts" / "structures.json"
    current = StructureSet.load(path)
    h = sorted(current.states)[0]
    text = current.states[h]
    StructureSet({**current.states, h: ("a" if text[0] != "a" else "c") + text[1:]}, current.meta, current.scopes).save(path)


def test_prepare_and_the_structure_modules_import_without_torch():
    code = ("import sys; import mpd.stages.prepare, mpd.stages.common, mpd.structure.build, mpd.structure.set, mpd.data.models; "
            "assert 'torch' not in sys.modules, 'torch was imported'")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    done = subprocess.run([sys.executable, "-s", "-c", f"import sys; sys.path.insert(0, {str(SRC.parent)!r}); {code}"], env=env,
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-2000:]


def test_the_whole_saprot_pipeline_makes_no_network_connection(tiny, no_network):
    """Jobs run offline: with SaProt as teacher and student, every stage and every saved adapter must work from local files."""
    plan = plan_for(tiny.home, teacher="saprot-650m", student="saprot-35m")
    run_stages(plan)
    assert (plan.campaign_dir / "evaluate" / "summary.json").is_file() and no_network == []


# -- (a) an ESM2 teacher and a SaProt student -------------------------------------------------------------------------------

def test_prepare_freezes_one_string_per_sequence_and_the_manifest_says_no_label_was_opened(tiny):
    plan = plan_for(tiny.home, student="saprot-35m")
    run_stages(plan, only={"prepare"})
    root = plan.campaign_dir
    structures = saved_structures(plan)
    # P1 is covered whole by its model; P2's model covers residues 5-30 only and the rest is '#'
    assert structures.states[sequence_hash(tiny.seqs["ASSAY_A"])] == fake_states(tiny.seqs["ASSAY_A"])
    assert structures.states[sequence_hash(tiny.seqs["ASSAY_B"])] == "#" * 4 + fake_states(tiny.seqs["ASSAY_B"][4:30]) + "#" * 4
    manifest = load(root / "contexts" / "MANIFEST.json")
    block = manifest["structures"]
    assert manifest["dms_labels_opened"] is False and block["dms_labels_opened"] is False and structures.meta["dms_labels_opened"] is False
    assert block["digest"] == structures.digest() and block["sequences"] == 2 and block["datasets"] == {"proteingym": 2}
    assert block["engine"]["name"] == "foldseek" and block["conflicts"] == 0
    assert block["policy"]["plddt_threshold"] == 70.0 and block["policy"]["legacy_mask"] is False
    # MANIFEST.json is what marks the stage as finished, so the strings must exist before it does
    assert (root / "contexts" / "MANIFEST.json").stat().st_mtime_ns >= (root / "contexts" / "structures.json").stat().st_mtime_ns


@pytest.mark.parametrize("method", ["direct-all", "score-all"])
def test_an_esm2_teacher_and_a_saprot_student_run_every_stage(tiny, method):
    plan = plan_for(tiny.home, student="saprot-35m", method=method)
    run_stages(plan)
    root = plan.campaign_dir
    digest = saved_structures(plan).digest()
    config = load(root / "train" / "all" / "CONFIG.json")
    assert config["structures_digest"] == digest and config["dms_labels_opened"] is False
    assert "structures_digest" not in load(root / "teacher" / "COMPLETE.json")            # an ESM2 teacher reads none
    for unit in ("base", "all"):
        for shard in ("shard_000_of_002", "shard_001_of_002"):
            assert load(root / "score" / "proteingym" / unit / f"{shard}.json")["structures_digest"] == digest
    summary = load(root / "evaluate" / "summary.json")
    assert {(e["unit"], e["dataset"]) for e in summary["results"]} == {("base", "proteingym"), ("all", "proteingym")}
    assert all(np.isfinite(e["macro_spearman"]) and e["assays"] == 3 for e in summary["results"])
    base = np.load(root / "score" / "proteingym" / "base" / "shard_000_of_002.npz")["scores"]
    tuned = np.load(root / "score" / "proteingym" / "all" / "shard_000_of_002.npz")["scores"]
    assert base.shape == tuned.shape and not np.array_equal(base, tuned)                   # the adapter was applied


def test_the_student_is_scored_with_the_strings_prepare_froze(tiny):
    """The score stage's numbers are the model's own on the frozen strings; blind strings would give others."""
    plan = plan_for(tiny.home, student="saprot-35m", score_shards=1)
    run_stages(plan, only={"prepare", "score-base"})
    root = plan.campaign_dir
    structures = saved_structures(plan).scoped("proteingym")
    table = ContextTable.load(root / "contexts" / "eval__proteingym")
    shard = np.load(root / "score" / "proteingym" / "base" / "shard_000_of_001.npz")
    model = Saprot("saprot-35m", CPU)
    np.testing.assert_allclose(shard["scores"], aa20_logits(model, table, structures, batch=8)[shard["rows"]], atol=1e-5)
    blind = StructureSet({h: "#" * len(s) for h, s in structures.states.items()})
    assert not np.allclose(shard["scores"], aa20_logits(model, table, blind, batch=8)[shard["rows"]], atol=1e-4)


# -- (b) the same SaProt weights as teacher and student ---------------------------------------------------------------------

@pytest.mark.parametrize("method", ["direct-all", "score-all"])
def test_a_saprot_teacher_identical_to_the_student_starts_at_zero_loss(tiny, monkeypatch, tiny_saprot_models, method):
    """Fails if a structure string, a masked token, a context index or the union mapping differs between the two stages."""
    monkeypatch.setenv("MPD_MODEL_SAPROT_650M", str(tiny_saprot_models["student"]))        # the weights of saprot-35m
    plan = plan_for(tiny.home, teacher="saprot-650m", student="saprot-35m", method=method)
    run_stages(plan, only={"prepare", "teacher", "train"})
    start = curve(plan.campaign_dir / "train" / "all")[0]
    # the bounds of test_stages.py: float32 noise only. Measured on torch 2.10 (CPU): dev_loss -1.1e-8 (direct-*; the KL term
    # rounds below zero) and 0.0 (score-*), rank correlation 1.0.
    assert start["step"] == 0 and abs(start["dev_loss"]) < 1e-6 and start["dev_rho_ST"] > 0.9999
    digest = saved_structures(plan).digest()
    assert load(plan.campaign_dir / "teacher" / "COMPLETE.json")["structures_digest"] == digest
    assert load(plan.campaign_dir / "train" / "all" / "CONFIG.json")["structures_digest"] == digest


def test_a_different_saprot_teacher_does_not_start_at_zero_loss(tiny):
    plan = plan_for(tiny.home, teacher="saprot-650m", student="saprot-35m")
    run_stages(plan, only={"prepare", "teacher", "train"})
    assert curve(plan.campaign_dir / "train" / "all")[0]["dev_loss"] > 1e-3


# -- (c) a SaProt teacher and an ESM2 student -------------------------------------------------------------------------------

def test_a_saprot_teacher_needs_only_the_training_sequences_and_evaluation_data_needs_no_structure(tiny):
    # ASSAY_B is on P2 alone, while the evaluation tables hold P1 as well; the Venus data has no structure folder at all
    write_tiny_venus(tiny.home, {"V_OWN": OWN}, structures=False)
    plan = plan_for(tiny.home, teacher="saprot-650m", student="esm2-8m", method="direct-one", assays=("ASSAY_B",),
                    eval_data=("proteingym", "venusmuthub"))
    run_stages(plan, only={"prepare", "teacher", "train"})
    structures = saved_structures(plan)
    assert set(structures.states) == {sequence_hash(tiny.seqs["ASSAY_B"])}
    block = load(plan.campaign_dir / "contexts" / "MANIFEST.json")["structures"]
    assert block["sequences"] == 1 and block["datasets"] == {"proteingym": 1}
    assert load(plan.campaign_dir / "teacher" / "COMPLETE.json")["structures_digest"] == structures.digest()
    unit = plan.campaign_dir / "train" / "ASSAY_B"
    assert "structures_digest" not in load(unit / "CONFIG.json") and (unit / "COMPLETE.json").is_file()   # the student reads none


# -- (d) a campaign that reads no structure is unchanged -------------------------------------------------------------------

@pytest.mark.parametrize("method", ["direct-all", "score-all"])
def test_an_esm2_only_campaign_writes_no_structures_and_keeps_its_digests(tiny, method):
    plan = plan_for(tiny.home, method=method)
    run_stages(plan)
    root = plan.campaign_dir
    manifest = load(root / "contexts" / "MANIFEST.json")
    assert "structures" not in manifest and not (root / "contexts" / "structures.json").exists()
    assert manifest["union_digest"] == UNION_DIGEST[method]
    bm = open_benchmark("proteingym", tiny.home)                 # and it is what the table builders give without any stage
    table = build_table(bm, sorted(a.assay_id for a in bm.assays()), with_variants=method == "score-all", dense=method == "direct-all")
    assert union([table]).digest() == manifest["union_digest"] == ContextTable.load(root / "contexts" / "union").digest()
    for record in (root / "teacher" / "COMPLETE.json", root / "train" / "all" / "CONFIG.json",
                   root / "score" / "proteingym" / "all" / "shard_000_of_002.json"):
        assert "structures_digest" not in load(record), record.name
    assert not (tiny.foldseek.parent / "calls.log").exists(), "Foldseek was started for a sequence-only campaign"


# -- (e) one sequence in both benchmarks -------------------------------------------------------------------------------------

@pytest.mark.parametrize("train_data", ["proteingym", "venusmuthub"])
def test_each_dataset_is_read_with_its_own_structure_for_a_sequence_both_benchmarks_hold(tiny, train_data):
    shared = tiny.seqs["ASSAY_A"]                                # a ProteinGym wild type that is a VenusMutHub wild type here too
    write_tiny_venus(tiny.home, {"V_SHARED": shared, "V_OWN": OWN}, low=10)       # its first 10 residues are low confidence there
    plan = plan_for(tiny.home, teacher="saprot-650m", student="saprot-35m", train_data=train_data,
                    eval_data=("proteingym", "venusmuthub"), score_shards=1)
    run_stages(plan)
    root = plan.campaign_dir
    strings = {"proteingym": fake_states(shared), "venusmuthub": "#" * 10 + fake_states(shared)[10:]}
    other = "venusmuthub" if train_data == "proteingym" else "proteingym"
    h = sequence_hash(shared)
    structures = saved_structures(plan)
    assert structures.states[h] == strings[train_data] and structures.scopes == {other: {h: strings[other]}}   # the training data's string is the base
    assert structures.scoped("proteingym").states[h] == strings["proteingym"]
    assert structures.scoped("venusmuthub").states[h] == strings["venusmuthub"]
    block = load(root / "contexts" / "MANIFEST.json")["structures"]
    assert block["conflicts"] == 1 and block["datasets"] == {"proteingym": 2, "venusmuthub": 2}

    # the teacher read the training dataset's view; the student was scored with each dataset's own
    teacher, student = Saprot("saprot-650m", CPU), Saprot("saprot-35m", CPU)
    union_table = ContextTable.load(root / "contexts" / "union")
    np.testing.assert_allclose(np.load(root / "teacher" / "logits.npy"),
                               aa20_logits(teacher, union_table, structures.scoped(train_data)), atol=1e-5)
    assert not np.allclose(np.load(root / "teacher" / "logits.npy"), aa20_logits(teacher, union_table, structures.scoped(other)), atol=1e-4)
    for dataset, wrong in (("proteingym", "venusmuthub"), ("venusmuthub", "proteingym")):
        table = ContextTable.load(root / "contexts" / f"eval__{dataset}")
        shard = np.load(root / "score" / dataset / "base" / "shard_000_of_001.npz")
        np.testing.assert_allclose(shard["scores"], aa20_logits(student, table, structures.scoped(dataset), batch=8)[shard["rows"]], atol=1e-5)
        assert not np.allclose(shard["scores"], aa20_logits(student, table, structures.scoped(wrong), batch=8)[shard["rows"]], atol=1e-4)
    summary = load(root / "evaluate" / "summary.json")
    assert {(e["dataset"], e["unit"]) for e in summary["results"]} == {(d, u) for d in ("proteingym", "venusmuthub") for u in ("base", "all")}


def test_an_unscoped_lookup_of_a_shared_sequence_is_refused_not_guessed(tiny):
    shared = tiny.seqs["ASSAY_A"]
    write_tiny_venus(tiny.home, {"V_SHARED": shared}, low=10)
    plan = plan_for(tiny.home, student="saprot-35m", eval_data=("proteingym", "venusmuthub"))
    run_stages(plan, only={"prepare"})
    campaign = common.Campaign.load(plan.campaign_dir)
    table = ContextTable.load(plan.campaign_dir / "contexts" / "eval__proteingym")
    with pytest.raises(ValueError, match="scoped"):
        StructureSet.load(campaign.stem_structures()).for_table(table)
    assert campaign.load_structures("proteingym").for_table(table)        # what the stages do: name the dataset


# -- (f) the structure strings are part of what a cached result and a checkpoint belong to ------------------------------------

def test_changed_structure_strings_make_a_training_checkpoint_stale(tiny, capsys):
    plan = plan_for(tiny.home, student="saprot-35m")
    run_stages(plan, only={"prepare", "teacher"})
    root = plan.campaign_dir
    with pytest.raises(Killed):
        train.run(common.Campaign.load(root), {"unit": "all"}, ckpt_every=2, on_update=kill_after(3))
    assert train.partial_dir(root / "train" / "all").is_dir()
    edit_structures(root)
    capsys.readouterr()
    train.run(common.Campaign.load(root), {"unit": "all"}, ckpt_every=2)
    assert "not resuming: all.partial: it was written by a run with a different configuration" in capsys.readouterr().out
    done = load(root / "train" / "all" / "COMPLETE.json")
    assert done["resumed_from"] == [] and done["status"] == "PASS"                     # it started again from update 0
    assert len(list((root / "train").glob("all.incomplete-*"))) == 1                   # and kept the old attempt aside
    assert load(root / "train" / "all" / "CONFIG.json")["structures_digest"] == saved_structures(plan).digest()


def test_unchanged_structure_strings_resume_a_killed_run(tiny, capsys):
    plan = plan_for(tiny.home, student="saprot-35m")
    run_stages(plan, only={"prepare", "teacher"})
    root = plan.campaign_dir
    with pytest.raises(Killed):
        train.run(common.Campaign.load(root), {"unit": "all"}, ckpt_every=2, on_update=kill_after(3))
    train.run(common.Campaign.load(root), {"unit": "all"}, ckpt_every=2)
    assert "resuming from update 2 of 6" in capsys.readouterr().out
    assert load(root / "train" / "all" / "COMPLETE.json")["resumed_from"] == [2]


def test_a_teacher_table_computed_with_other_structure_strings_is_recomputed_and_training_refuses_it(tiny, capsys):
    plan = plan_for(tiny.home, teacher="saprot-650m", student="esm2-8m")
    run_stages(plan, only={"prepare", "teacher"})
    capsys.readouterr()
    run_stages(plan, only={"teacher"})
    assert "already computed" in capsys.readouterr().out                               # same strings: skipped
    edit_structures(plan.campaign_dir)
    with pytest.raises(RuntimeError, match=r"computed with structures [0-9a-f]{12}, this campaign's structures\.json is [0-9a-f]{12}"):
        run_stages(plan, only={"train"})
    capsys.readouterr()
    run_stages(plan, only={"teacher"})
    assert "recomputing, it was computed with structures" in capsys.readouterr().out
    assert load(plan.campaign_dir / "teacher" / "COMPLETE.json")["structures_digest"] == saved_structures(plan).digest()
    run_stages(plan, only={"train"})
    assert (plan.campaign_dir / "train" / "all" / "COMPLETE.json").is_file()


def test_an_esm2_teacher_table_ignores_the_structure_strings(tiny, capsys):
    plan = plan_for(tiny.home, teacher="esm2-35m", student="saprot-35m")
    run_stages(plan, only={"prepare", "teacher"})
    edit_structures(plan.campaign_dir)
    capsys.readouterr()
    run_stages(plan, only={"teacher"})
    assert "already computed" in capsys.readouterr().out


def test_a_campaign_without_its_structures_says_which_stage_writes_them(tiny):
    plan = plan_for(tiny.home, teacher="saprot-650m", student="saprot-35m")
    run_stages(plan, only={"prepare"})
    (plan.campaign_dir / "contexts" / "structures.json").unlink()
    for stage in ("teacher", "train", "score-base"):
        with pytest.raises(FileNotFoundError, match=r"(saprot-650m|saprot-35m) reads protein structure, but .*structures\.json does not exist.*prepare"):
            run_stages(plan, only={stage})
    campaign = common.Campaign.load(plan.campaign_dir)
    assert campaign.load_structures("proteingym") is None and campaign.structures_digest() is None


def test_sequence_only_models_never_look_for_structures(tiny):
    campaign = common.Campaign.load(plan_for(tiny.home).campaign_dir)
    assert not campaign.needs_structure("teacher") and not campaign.needs_structure("student")
    assert campaign.structures_for("student", "proteingym") is None and campaign.structures_for("teacher", "proteingym") is None
    assert common.Campaign.load(plan_for(tiny.home, student="saprot-35m").campaign_dir).needs_structure("student")


# -- the same inputs give the same strings, from the cache --------------------------------------------------------------------

def test_prepare_again_writes_identical_strings_and_does_not_start_foldseek_again(tiny):
    plan = plan_for(tiny.home, student="saprot-35m")
    run_stages(plan, only={"prepare"})
    path = plan.campaign_dir / "contexts" / "structures.json"
    first, calls = path.read_bytes(), fake_calls(tiny.foldseek)
    assert len(calls) == 2 and (tiny.home / "cache" / "3di").is_dir()          # one call per structure file (P1, P2), cached under the home
    run_stages(plan, only={"prepare"})
    assert path.read_bytes() == first and fake_calls(tiny.foldseek) == calls     # byte-identical, and nothing was recomputed


def test_the_legacy_masking_switch_reaches_prepare_and_is_recorded(tiny, monkeypatch):
    plan = plan_for(tiny.home, student="saprot-35m")
    run_stages(plan, only={"prepare"})
    assert load(plan.campaign_dir / "contexts" / "MANIFEST.json")["structures"]["policy"]["legacy_mask"] is False
    monkeypatch.setenv("MPD_STRUCTURE_LEGACY_MASK", "1")
    run_stages(plan, only={"prepare"})
    assert load(plan.campaign_dir / "contexts" / "MANIFEST.json")["structures"]["policy"]["legacy_mask"] is True


# -- two processes -------------------------------------------------------------------------------------------------------------

def test_two_process_ddp_training_of_a_saprot_student_matches_single_process(tiny, tmp_path):
    """1 process x 4 micro-batches against 2 processes x 2 (gloo, CPU): every rank reads the frozen strings itself."""
    single = plan_for(tiny.home, student="saprot-35m", train_gpus=1, peak_lr=3e-3, global_batch=8)
    ddp = plan_for(tiny.home, student="saprot-35m", train_gpus=2, peak_lr=3e-3, global_batch=8)
    run_stages(single, only={"prepare", "teacher", "train"})
    run_stages(ddp, only={"prepare", "teacher"})
    code, log = finish_ddp_train(start_ddp_train(ddp, tmp_path, "ddp"))
    assert code == 0, log[-3000:]
    a, b = curve(single.campaign_dir / "train" / "all"), curve(ddp.campaign_dir / "train" / "all")
    assert [r["step"] for r in a] == [r["step"] for r in b]
    assert abs(a[-1]["dev_loss"] - a[0]["dev_loss"]) > 100 * 2e-5, "training barely moved the loss: the comparison would be vacuous"
    for ra, rb in zip(a, b):
        assert ra["dev_loss"] == pytest.approx(rb["dev_loss"], abs=2e-5), (ra, rb)
    assert load(ddp.campaign_dir / "train" / "all" / "CONFIG.json")["structures_digest"] == saved_structures(ddp).digest()


# -- the weights check at the start of prepare -------------------------------------------------------------------------------

def test_prepare_stops_at_once_when_weights_were_never_fetched(tiny, monkeypatch):
    monkeypatch.delenv("MPD_MODEL_SAPROT_35M")
    plan = plan_for(tiny.home, student="saprot-35m")
    with pytest.raises(ModelNotReady) as err:
        run_stages(plan, only={"prepare"})
    text = str(err.value)
    assert text.startswith("weights for saprot-35m are missing (") and str(tiny.home / "models" / "saprot-35m") in text
    assert f"Run on a login node: mpd fetch model saprot-35m --cluster vision --home {tiny.home}" in text
    assert not (plan.campaign_dir / "contexts").exists()                                # nothing was built first


def test_prepare_lists_every_missing_model_with_one_command(tiny, monkeypatch):
    for variable in ("MPD_MODEL_SAPROT_650M", "MPD_MODEL_ESM2_8M"):
        monkeypatch.delenv(variable)
    plan = plan_for(tiny.home, teacher="saprot-650m", student="esm2-8m")
    with pytest.raises(ModelNotReady, match=r"model weights are not ready:(.|\n)*saprot-650m(.|\n)*esm2-8m(.|\n)*mpd fetch model saprot-650m esm2-8m --cluster vision"):
        run_stages(plan, only={"prepare"})


def test_prepare_lets_the_hub_route_through_on_a_login_node_or_laptop(tiny, monkeypatch):
    monkeypatch.delenv("MPD_MODEL_SAPROT_35M")
    monkeypatch.setenv("MPD_ALLOW_HUB", "1")
    plan = plan_for(tiny.home, student="saprot-35m")
    run_stages(plan, only={"prepare"})                          # models.resolve accepts the Hub id then, so prepare does not stop
    assert (plan.campaign_dir / "contexts" / "MANIFEST.json").is_file()


def test_prepare_without_foldseek_names_the_fix_for_the_cluster(tiny, monkeypatch):
    monkeypatch.delenv("MPD_FOLDSEEK")
    plan = plan_for(tiny.home, student="saprot-35m")
    with pytest.raises(Exception, match=r"Run on a login node: mpd fetch foldseek --cluster vision --home ") as err:
        run_stages(plan, only={"prepare"})
    assert type(err.value).__name__ == "FoldseekError"
    assert not (plan.campaign_dir / "contexts" / "MANIFEST.json").exists()            # no marker: the stage is not finished


def test_prepare_names_every_structure_it_cannot_use(tiny):
    (tiny.home / "data" / "proteingym-v1.3" / "ProteinGym_AF2_structures" / "P1.pdb").unlink()
    plan = plan_for(tiny.home, student="saprot-35m")
    with pytest.raises(Exception, match=r"structure file .*P1\.pdb not found") as err:
        run_stages(plan, only={"prepare"})
    assert type(err.value).__name__ == "StructureError"
    assert not (plan.campaign_dir / "contexts" / "MANIFEST.json").exists() and not (plan.campaign_dir / "contexts" / "structures.json").exists()


# -- (g) labels ---------------------------------------------------------------------------------------------------------------

def test_no_stage_but_evaluate_opens_a_label_not_even_with_structures(tiny, monkeypatch):
    """A tripwire on both benchmarks' label readers, then every stage except evaluate with a SaProt recipe on both datasets."""
    write_tiny_venus(tiny.home, {"V_SHARED": tiny.seqs["ASSAY_A"], "V_OWN": OWN}, low=10)

    def tripwire(self, assay_id):
        raise AssertionError(f"{type(self).__name__}.labels({assay_id!r}) was called outside the evaluate stage")
    monkeypatch.setattr(ProteinGym, "labels", tripwire)
    monkeypatch.setattr(VenusMutHub, "labels", tripwire)
    plan = plan_for(tiny.home, teacher="saprot-650m", student="saprot-35m", eval_data=("proteingym", "venusmuthub"), score_shards=1)
    run_stages(plan, only={"prepare", "teacher", "train", "score-base", "score"})
    assert load(plan.campaign_dir / "contexts" / "MANIFEST.json")["dms_labels_opened"] is False
    assert (plan.campaign_dir / "score" / "venusmuthub" / "all" / "shard_000_of_001.json").is_file()
    assert not (plan.campaign_dir / "evaluate").exists()


def test_the_label_firewall_scan_covers_the_stages_and_the_structure_code():
    for folder in (SRC / "stages", SRC / "structure", SRC / "students"):
        for path in sorted(folder.glob("*.py")):
            if path.name == "evaluate.py":
                assert ".labels(" in path.read_text()
            else:
                text = path.read_text()
                assert ".labels(" not in text and "DMS_score" not in text, f"{path.relative_to(SRC)} touches labels"
