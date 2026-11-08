import dataclasses
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from mpd import registry
from mpd.slurm.cluster import Cluster
from mpd.slurm.plan import PlanError, PlanOptions, make_plan
from mpd.slurm.render import render_script
from mpd.slurm.submit import SubmitError, submit, write_campaign
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
def stage(plan, name):
    return next(s for s in plan.stages if s.full_name == name)


def test_grace_all_assay_uses_two_a100_and_keeps_global_batch(tmp_path):
    plan = make_plan(spec(), opts(tmp_path))
    r = plan.resolved
    assert (r["world_size"], r["microbatch"], r["accumulation"], r["global_batch"]) == (2, 2, 16, 64)
    t = stage(plan, "train")
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
    assert len(score.tasks) == 2 * (4 + 1)          # 2 units x (4 PG shards + 1 Venus shard)
    assert {t["unit"] for t in score.tasks} == {"A", "B"}
    assert {t["unit"] for t in stage(plan, "score-base").tasks} == {"base"}


def test_large_arrays_are_chunked_under_the_slurm_limit(tmp_path):
    assays = tuple(f"assay{i}" for i in range(450))
    plan = make_plan(spec(method="score-one", assays=assays), opts(tmp_path))
    parts = [s for s in plan.stages if s.name == "score"]
    assert len(parts) == 3 and all(len(p.tasks) <= 1000 for p in parts)
    assert sum(len(p.tasks) for p in parts) == 450 * 5
    assert [p.task_offset for p in parts] == [0, 1000, 2000]
    assert set(stage(plan, "evaluate").depends_on) >= {p.full_name for p in parts}


def test_walltime_over_partition_limit_is_an_error_with_a_fix(tmp_path):
    with pytest.raises(PlanError, match="PartitionTimeLimit"):
        make_plan(spec(), opts(tmp_path, train_time="5-00:00:00"))


def test_forced_gpu_count_must_divide_the_batch(tmp_path):
    with pytest.raises(PlanError, match="not divisible"):
        make_plan(spec(global_batch=62, train_gpus=2), opts(tmp_path))


def test_default_gpu_count_adapts_to_the_node_shape(tmp_path):
    # A40 nodes have 3 GPUs; 3 x microbatch 2 does not divide 64, so the default uses 2 of them.
    r = make_plan(spec(), opts(tmp_path, gpu="a40")).resolved
    assert (r["world_size"], r["accumulation"]) == (2, 16)
    # an odd global batch falls back to one GPU instead of failing
    assert make_plan(spec(global_batch=62), opts(tmp_path)).resolved["world_size"] == 1


def test_unverified_gpu_warning_is_reported_once(tmp_path):
    warnings = make_plan(spec(), opts(tmp_path, gpu="a40")).warnings
    assert sum("gpu-a40" in w for w in warnings) == 1


def test_model_that_does_not_fit_any_gpu_is_rejected(tmp_path):
    with pytest.raises(PlanError, match="needs about"):
        make_plan(spec(teacher="esm2-650m", student="esm2-3b"), opts(tmp_path, gpu="t4"))


def test_missing_account_is_warned_on_grace(tmp_path):
    o = opts(tmp_path)
    o.account = None
    assert any("--account" in w for w in make_plan(spec(), o).warnings)


# ---------------------------------------------------------------- render
def test_rendered_scripts_follow_grace_conventions(tmp_path):
    plan = make_plan(spec(), opts(tmp_path))
    text = render_script(plan, stage(plan, "train"))
    for needle in ("#SBATCH --gres=gpu:a100:2", "#SBATCH --partition=gpu", "#SBATCH --account=123456789012",
                   "#SBATCH --export=NONE", "#SBATCH --get-user-env=L", "#SBATCH --cpus-per-task=48",
                   "#SBATCH --mem=360G", "HF_HUB_OFFLINE=1", "--nproc_per_node=2"):
        assert needle in text, needle
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
    assert "#SBATCH --qos=standard" in text and "module load CUDA/12.8.0" in text


# ---------------------------------------------------------------- submit (fake sbatch)
def fake_slurm(tmp_path, fail_on=None):
    d = tmp_path / "bin"
    d.mkdir()
    log = tmp_path / "calls.log"
    (d / "sbatch").write_text(f"""#!/bin/bash
echo "$@" >> {log}
n=$(grep -c . {log})
if [ -n "{fail_on or ''}" ] && [[ "$*" == *"{fail_on or 'NEVER'}"* ]]; then echo "sbatch: error: invalid account" >&2; exit 1; fi
echo "(from job_submit) your job is charged as below" >&2
echo $((1000 + n))
""")
    (d / "scancel").write_text(f"#!/bin/bash\necho \"scancel $@\" >> {log}\n")
    for f in d.iterdir():
        f.chmod(f.stat().st_mode | stat.S_IEXEC)
    return d, log


def test_submit_chains_dependencies_and_records_ids(tmp_path):
    plan = make_plan(spec(), opts(tmp_path))
    scripts = write_campaign(plan)
    d, log = fake_slurm(tmp_path)
    ids = submit(plan, scripts, sbatch=str(d / "sbatch"), echo=lambda *_: None)
    calls = log.read_text().splitlines()
    assert len(calls) == len(plan.stages)
    assert "--dependency" not in calls[0]                                   # prepare
    train_call = next(c for c in calls if "train.sbatch" in c)
    assert f"--dependency=afterok:{ids['teacher']}" in train_call and "--kill-on-invalid-dep=yes" in train_call
    eval_call = next(c for c in calls if "evaluate.sbatch" in c)
    assert ids["score"] in eval_call and ids["score-base"] in eval_call
    assert json.loads((plan.campaign_dir / "SUBMITTED.json").read_text())["jobs"] == ids


def test_resubmitting_is_refused_without_force(tmp_path):
    plan = make_plan(spec(), opts(tmp_path))
    scripts = write_campaign(plan)
    d, _ = fake_slurm(tmp_path)
    submit(plan, scripts, sbatch=str(d / "sbatch"), echo=lambda *_: None)
    with pytest.raises(SubmitError, match="already submitted"):
        submit(plan, scripts, sbatch=str(d / "sbatch"), echo=lambda *_: None)


def test_failure_cancels_already_queued_jobs(tmp_path, monkeypatch):
    plan = make_plan(spec(), opts(tmp_path))
    scripts = write_campaign(plan)
    d, log = fake_slurm(tmp_path, fail_on="score.sbatch")
    monkeypatch.setenv("PATH", f"{d}:{os.environ['PATH']}")
    with pytest.raises(SubmitError, match="rejected stage score"):
        submit(plan, scripts, sbatch=str(d / "sbatch"), echo=lambda *_: None)
    assert "scancel" in log.read_text()
    assert not (plan.campaign_dir / "SUBMITTED.json").exists()


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
