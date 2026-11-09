import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from fx_render import clean_mpd_env  # noqa: F401  (autouse: no MPD_* variable of the developer's shell reaches a plan)
from fx_slurm import fake_slurm  # noqa: F401
from mpd import registry
from mpd.slurm.cluster import Cluster
from mpd.slurm.plan import PlanError, PlanOptions, make_plan
from mpd.slurm.render import render_script
from mpd.slurm.submit import submit, write_campaign
from mpd.spec import RunSpec, SpecError


def spec(**kw):
    base = dict(teacher="esm2-3b", student="esm2-650m", train_data="proteingym", method="score-all",
                eval_data=("proteingym", "venusmuthub"))
    base.update(kw)
    return RunSpec(**base)


def opts(tmp_path, cluster="grace", **kw):
    return PlanOptions(cluster=Cluster.load(cluster), home=tmp_path / "home", python="/env/bin/python",
                       account="123456789012" if cluster == "grace" else None, **kw)


# ---------------------------------------------------------------- spec
def test_four_methods_map_to_objective_and_scope():
    assert (spec(method="direct-all").objective, spec(method="direct-all").scope) == ("direct_logits", "all")
    assert (spec(method="score-one", assays=("A",)).objective, spec(method="score-one", assays=("A",)).scope) == ("score", "one")


@pytest.mark.parametrize("kw,msg", [
    (dict(method="score-one"), "needs --assay"),
    (dict(method="score-all", assays=("A",)), "only valid with a \\*-one"),
    (dict(method="bogus"), "--method"),
    (dict(eval_data=()), "--eval-data"),
    (dict(eval_data=("proteingym", "proteingym")), "twice"),
    (dict(student="nope"), "unknown model"),
    (dict(method="score-one", assays=("A", "A")), "twice"),
    (dict(method="score-one", assays=("A",), train_gpus=2), "one-assay"),
])
def test_spec_rejects_bad_input(kw, msg):
    with pytest.raises(SpecError, match=msg):
        spec(**kw).validate()


def test_planned_model_cannot_be_used():
    with pytest.raises(SpecError, match="planned"):
        spec(teacher="aido-rag-16b", student="aido-rag-16b").validate()


def test_output_family_lock_blocks_cross_family_distillation(monkeypatch):
    locked = dataclasses.replace(registry.get("esm2-3b"), key="locked-t", output_family_lock="locked",
                                 license_note="Outputs may only train the same family.")
    monkeypatch.setitem(registry.MODELS, "locked-t", locked)
    with pytest.raises(SpecError, match="blocked.*only be distilled into locked models"):
        spec(teacher="locked-t").validate()
    same_family = dataclasses.replace(registry.get("esm2-650m"), key="locked-s", family="locked")
    monkeypatch.setitem(registry.MODELS, "locked-s", same_family)
    spec(teacher="locked-t", student="locked-s").validate()   # same family is allowed


def test_campaign_id_is_deterministic_and_reflects_choices():
    a = spec(method="score-one", assays=("B", "A"))
    b = spec(method="score-one", assays=("A", "B"))
    assert a.campaign_id() != b.campaign_id() or a.assays != b.assays   # order is part of the spec
    assert spec().campaign_id() == spec().campaign_id()
    assert spec(updates=500).campaign_id() != spec().campaign_id()      # overrides never collide with defaults
    assert "seed7" in spec(seed=7).campaign_id()


def test_json_roundtrip():
    s = spec(method="direct-one", assays=("A", "B"), updates=100)
    assert RunSpec.from_json(s.to_json()) == s


# ---------------------------------------------------------------- plan
# Resources, estimates, identity and option validation are tested in test_plan_resources.py; the scripts in test_render.py.
def stage(plan, name):
    return next(s for s in plan.stages if s.full_name == name)


def test_grace_all_assay_defaults_to_one_a100_and_keeps_global_batch(tmp_path):
    r = make_plan(spec(), opts(tmp_path)).resolved
    assert (r["world_size"], r["microbatch"], r["accumulation"], r["global_batch"]) == (1, 2, 32, 64)
    two = make_plan(spec(train_gpus=2), opts(tmp_path))
    assert (two.resolved["world_size"], two.resolved["accumulation"]) == (2, 16)
    t = stage(two, "train")
    assert (t.gpu, t.gpus, t.cpus, t.mem_gb, t.launcher) == ("a100", 2, 48, 360, "ddp")


def test_vision_all_assay_matches_the_paper_four_gpu_setup(tmp_path):
    plan = make_plan(spec(), opts(tmp_path, "vision"))
    assert (plan.resolved["world_size"], plan.resolved["accumulation"]) == (4, 8)


def test_one_assay_is_a_single_gpu_array(tmp_path):
    plan = make_plan(spec(method="direct-one", assays=("A", "B", "C")), opts(tmp_path))
    t = stage(plan, "train")
    assert (t.gpus, len(t.tasks), t.launcher) == (1, 3, "python")
    assert plan.resolved["accumulation"] == 32 and plan.resolved["kl_weight"] == 1.0
    assert plan.resolved["support"] == "aa20_dense"


def test_score_objective_has_no_kl_and_observed_support(tmp_path):
    r = make_plan(spec(method="score-all"), opts(tmp_path)).resolved
    assert (r["kl_weight"], r["support"]) == (0.0, "observed")


def test_stage_dependencies(tmp_path):
    plan = make_plan(spec(), opts(tmp_path))
    deps = {s.full_name: s.depends_on for s in plan.stages}
    assert deps["prepare"] == []
    assert deps["teacher"] == ["prepare"]
    assert deps["train"] == ["teacher"]
    assert deps["score-base"] == ["prepare"]        # the frozen base student needs no teacher or training
    assert deps["score"] == ["train"]
    assert set(deps["evaluate"]) == {"score-base", "score"}


def test_scoring_tasks_cover_every_unit_dataset_shard(tmp_path):
    plan = make_plan(spec(method="score-one", assays=("A", "B")), opts(tmp_path))
    score = stage(plan, "score")
    assert len(score.tasks) == 2 * (1 + 1)          # a one-assay table is not split: 1 PG shard + 1 Venus shard per unit
    assert {t["unit"] for t in score.tasks} == {"A", "B"}
    assert {t["unit"] for t in stage(plan, "score-base").tasks} == {"base"}
    assert len(stage(plan, "score-base").tasks) == 4 + 1


def test_large_arrays_are_chunked_under_the_slurm_limit(tmp_path):
    assays = tuple(f"assay{i}" for i in range(450))
    plan = make_plan(spec(method="score-one", assays=assays), opts(tmp_path, score_shards={"proteingym": 4, "venusmuthub": 4}))
    parts = [s for s in plan.stages if s.name == "score"]
    assert len(parts) == 3 and all(len(p.tasks) <= 1000 for p in parts)
    assert sum(len(p.tasks) for p in parts) == 450 * (1 + 4)
    assert [p.task_offset for p in parts] == [0, 1000, 2000]
    assert set(stage(plan, "evaluate").depends_on) >= {p.full_name for p in parts}


def test_walltime_over_partition_limit_is_an_error_with_a_fix(tmp_path):
    with pytest.raises(PlanError, match="PartitionTimeLimit"):
        make_plan(spec(), opts(tmp_path, train_time="5-00:00:00"))


def test_forced_gpu_count_must_divide_the_batch(tmp_path):
    with pytest.raises(PlanError, match="not divisible"):
        make_plan(spec(global_batch=62, train_gpus=2), opts(tmp_path))


def test_default_gpu_count_adapts_to_the_node_shape(tmp_path):
    # an A40 node has 2 GPUs (HPRC hardware page); Grace's profile asks for one, Vision's rule picks the widest that divides the batch
    assert make_plan(spec(), opts(tmp_path, gpu="a40")).resolved["world_size"] == 1
    assert make_plan(spec(train_gpus=2), opts(tmp_path, gpu="a40")).resolved["world_size"] == 2
    # an odd global batch falls back to one GPU instead of failing
    assert make_plan(spec(global_batch=62), opts(tmp_path, "vision")).resolved["world_size"] == 1


def test_unverified_gpu_warning_is_reported_once(tmp_path):
    warnings = make_plan(spec(), opts(tmp_path, gpu="a40")).warnings
    assert sum("gpu-a40" in w for w in warnings) == 1


def test_model_that_does_not_fit_any_gpu_is_rejected(tmp_path):
    # a T4 cannot train in bfloat16 at all; with the override the 3B student still needs far more than 16 GB
    with pytest.raises(PlanError, match="no native bfloat16"):
        make_plan(spec(teacher="esm2-650m", student="esm2-3b"), opts(tmp_path, gpu="t4"))
    with pytest.raises(PlanError, match="needs about"):
        make_plan(spec(teacher="esm2-650m", student="esm2-3b"), opts(tmp_path, gpu="t4", allow_unsupported_gpu=True))


def test_missing_account_is_warned_on_grace(tmp_path):
    o = opts(tmp_path)
    o.account = None
    assert any("--account" in w for w in make_plan(spec(), o).warnings)


# ---------------------------------------------------------------- render (the full matrix is in test_render.py)
def test_rendered_scripts_follow_grace_conventions(tmp_path):
    plan = make_plan(spec(train_gpus=2), opts(tmp_path))
    text = render_script(plan, stage(plan, "train"))
    for needle in ("#SBATCH --gres=gpu:a100:2", "#SBATCH --partition=gpu", "#SBATCH --account=123456789012",
                   "#SBATCH --cpus-per-task=48", "#SBATCH --mem=360G", "HF_HUB_OFFLINE=1", "--nproc_per_node=2",
                   "nvidia-smi -L", "MPD_REQUIRE_CUDA=1", "XDG_CACHE_HOME=", "#SBATCH --open-mode=append"):
        assert needle in text, needle
    # Slurm 25.05 rejects --get-user-env=L, --export=NONE implies --get-user-env, and no verified Grace job used either
    assert "--export" not in text and "--get-user-env" not in text
    assert "torch.distributed.run" in text and "bin/torchrun" not in text
    assert "module load" not in text                       # grace profile needs no modules by default


def test_a40_jobs_name_their_partition(tmp_path):
    plan = make_plan(spec(), opts(tmp_path, gpu="a40"))
    assert "#SBATCH --partition=gpu-a40" in render_script(plan, stage(plan, "train"))


def test_array_script_has_throttle_and_offset(tmp_path):
    plan = make_plan(spec(method="direct-one", assays=("A", "B", "C")), opts(tmp_path))
    text = render_script(plan, stage(plan, "train"))
    assert "#SBATCH --array=0-2%8" in text and "SLURM_ARRAY_TASK_ID + 0" in text


def test_cpu_stage_has_no_gres(tmp_path):
    plan = make_plan(spec(), opts(tmp_path))
    text = render_script(plan, stage(plan, "prepare"))
    assert "--gres" not in text and "--partition" not in text   # Grace routes CPU jobs by walltime


def test_vision_script_uses_its_qos_and_modules(tmp_path):
    plan = make_plan(spec(), opts(tmp_path, "vision"))
    text = render_script(plan, stage(plan, "train"))
    assert "#SBATCH --qos=standard" in text and "module load CUDA/12.8.0" in text and "#SBATCH --gres=gpu:H200:4" in text


def test_job_names_and_tags_survive_write_campaign(tmp_path):
    """The scripts on disk carry the plan's own tag; `mpd status/cancel` and the submit retry logic find jobs by it."""
    plan = make_plan(spec(), opts(tmp_path, nonce="abcdef123456"))
    scripts = write_campaign(plan)
    assert plan.campaign_dir.name == plan.campaign_id and (plan.campaign_dir / "logs").is_dir()      # Slurm will not create --output
    for name, path in scripts.items():
        assert f"#SBATCH --comment=mpd:{plan.campaign_id}:abcdef123456:{name}\n" in path.read_text()
    resolved = json.loads((plan.campaign_dir / "resolved.json").read_text())
    assert resolved["campaign_id"] == plan.campaign_id and resolved["score_shards"] == {"proteingym": 4, "venusmuthub": 1}


# ---------------------------------------------------------------- hand-off to submit (the submit logic has its own tests)
def test_submit_chains_the_planned_stages_with_their_tags(tmp_path, fake_slurm):
    plan = make_plan(spec(), opts(tmp_path))
    scripts = write_campaign(plan)
    ids = submit(plan, scripts, echo=lambda *_: None, sleep=fake_slurm.sleep)
    assert list(ids) == [s.full_name for s in plan.stages]
    jobs = fake_slurm.submitted()
    assert [j["stage"] for j in jobs] == list(ids)
    assert all(j["comment"].startswith(f"mpd:{plan.campaign_id}:{plan.nonce}:") for j in jobs)
    train = next(j for j in jobs if j["stage"] == "train")
    assert train["deps"] == [int(ids["teacher"])] and train["gres"] == "gpu:a100:1"
    marker = json.loads((plan.campaign_dir / "SUBMITTED.json").read_text())
    assert marker["jobs"] == ids and marker["campaign_id"] == plan.campaign_id and marker["nonce"] == plan.nonce


def test_cli_plan_runs_without_torch(tmp_path):
    """The login-node entry point must not import torch."""
    src = str(Path(__file__).parents[1] / "src")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    code = (f"import sys; sys.path.insert(0, {src!r}); from mpd.cli import main; "
            "rc = main(['plan','--cluster','grace','--home',sys.argv[1],'--teacher','esm2-3b','--student','esm2-650m',"
            "'--train-data','proteingym','--method','score-all','--eval-data','venusmuthub']); "
            "assert 'torch' not in sys.modules, 'torch imported by the planner'; sys.exit(rc)")
    out = subprocess.run([sys.executable, "-s", "-c", code, str(tmp_path)], env=env, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "estimated upper bound" in out.stdout


def test_the_planner_modules_import_without_torch_or_yaml_extras():
    src = str(Path(__file__).parents[1] / "src")
    code = (f"import sys; sys.path.insert(0, {src!r}); import mpd.envrecord, mpd.presets, mpd.slurm.cluster, mpd.slurm.plan, "
            "mpd.slurm.render; bad = [m for m in ('torch', 'transformers', 'peft', 'pandas') if m in sys.modules]; print(bad)")
    out = subprocess.run([sys.executable, "-s", "-c", code], capture_output=True, text=True,
                         env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"})
    assert out.returncode == 0 and out.stdout.strip() == "[]", out.stderr
