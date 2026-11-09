"""The planner: resources, memory, walltime estimates, campaign identity and the validation of every option."""

import dataclasses
import itertools
import json
import re
from pathlib import Path

import pytest

from fx_render import ACCOUNT, HOME, NONCE, clean_mpd_env, make, opts, render_all, spec, stage  # noqa: F401  (clean_mpd_env is autouse)
from mpd import registry
from mpd.slurm import plan as planner
from mpd.slurm.cluster import Cluster
from mpd.slurm.plan import (PlanError, describe, forward_walltime_s, make_plan, parse_env_pair, parse_score_shards, scored_on_own_assay,
                            train_walltime_s)
from mpd.slurm.render import render_script


def fixed_registry(monkeypatch, **act_gb):
    """Pin activation sizes to the audited values so a test does not move when someone re-measures."""
    table = {"esm2-8m": 2, "esm2-35m": 3, "esm2-150m": 9, "esm2-650m": 12, "esm2-3b": 27, **act_gb}
    for key, value in table.items():
        monkeypatch.setitem(registry.MODELS, key, dataclasses.replace(registry.get(key), act_gb=value))


def warnings_about(plan, word):
    return [w for w in plan.warnings if word in w]


# ---------------------------------------------------------------- training width (R-13, C10)
def test_grace_trains_on_one_gpu_by_default_and_on_two_when_asked():
    one = make("grace")
    assert (one.resolved["world_size"], one.resolved["microbatch"], one.resolved["accumulation"]) == (1, 2, 32)
    t = stage(one, "train")
    assert (t.gpu, t.gpus, t.cpus, t.mem_gb, t.launcher) == ("a100", 1, 24, 180, "python")
    two = make("grace", spec_kw=dict(train_gpus=2))
    assert (two.resolved["world_size"], two.resolved["accumulation"], two.resolved["global_batch"]) == (2, 16, 64)
    t = stage(two, "train")
    assert (t.gpu, t.gpus, t.cpus, t.mem_gb, t.launcher) == ("a100", 2, 48, 360, "ddp")


def test_vision_keeps_the_paper_width():
    plan = make("vision")
    assert (plan.resolved["world_size"], plan.resolved["accumulation"]) == (4, 8)


def test_default_width_follows_the_profile_only_when_it_sets_one():
    base = Cluster.load("grace")
    wide = dataclasses.replace(base, default_train_gpus=2)
    plan = make_plan(spec(), dataclasses.replace(opts(), cluster=wide))
    assert plan.resolved["world_size"] == 2
    huge = dataclasses.replace(base, default_train_gpus=8)
    assert make_plan(spec(), dataclasses.replace(opts(), cluster=huge)).resolved["world_size"] == 2           # capped at the node
    none = dataclasses.replace(base, default_train_gpus=None)
    assert make_plan(spec(), dataclasses.replace(opts(), cluster=none)).resolved["world_size"] == 2           # the old rule: up to 4


def test_an_odd_global_batch_falls_back_to_one_gpu_instead_of_failing():
    assert make("vision", spec_kw=dict(global_batch=62)).resolved["world_size"] == 1
    assert make("grace", spec_kw=dict(global_batch=62)).resolved["world_size"] == 1
    with pytest.raises(PlanError, match="not divisible"):
        make("grace", spec_kw=dict(global_batch=63))


@pytest.mark.parametrize("n,message", [(0, "at least 1"), (-1, "at least 1"), (3, "does not fit one node"), (4, "does not fit one node"),
                                       (8, "does not fit one node")])
def test_train_gpus_is_validated_not_clamped(n, message):
    with pytest.raises(PlanError, match=message):
        make("grace", spec_kw=dict(train_gpus=n))
    assert make("grace", spec_kw=dict(train_gpus=2)).resolved["world_size"] == 2


def test_train_gpus_error_names_the_node_limit_before_any_divisibility_check():
    with pytest.raises(PlanError, match=r"\[train\] 3 x a100 does not fit one node \(2 per node on grace\)"):
        make("grace", spec_kw=dict(train_gpus=3))
    with pytest.raises(PlanError, match="not divisible"):
        make("grace", spec_kw=dict(global_batch=62, train_gpus=2))


def test_a40_nodes_hold_two_gpus():
    plan = make("grace", gpu="a40", spec_kw=dict(train_gpus=2))
    assert (plan.resolved["world_size"], plan.resolved["accumulation"]) == (2, 16)
    with pytest.raises(PlanError, match="does not fit one node"):
        make("grace", gpu="a40", spec_kw=dict(train_gpus=3))
    assert make("grace", gpu="a40").resolved["world_size"] == 1


def test_unverified_gpu_warning_is_reported_once():
    plan = make("grace", gpu="a40")
    assert len(warnings_about(plan, "gpu-a40")) == 1


# ---------------------------------------------------------------- bfloat16 (C10)
@pytest.mark.parametrize("gpu", ["rtx", "t4"])
def test_turing_gpus_are_refused_for_training_and_name_the_alternatives(gpu):
    with pytest.raises(PlanError) as err:
        make("grace", gpu=gpu)
    text = str(err.value)
    assert "no native bfloat16" in text and "--gpu a100 or --gpu a40" in text and "--allow-unsupported-gpu" in text


def test_the_override_runs_float32_and_says_so(monkeypatch):
    fixed_registry(monkeypatch)
    plan = make("grace", gpu="rtx", allow_unsupported_gpu=True, spec_kw=dict(teacher="esm2-35m", student="esm2-150m"))
    assert plan.options.extra_env["MPD_DTYPE"] == "float32"
    assert any("float32" in w and "never tested" in w for w in plan.warnings)
    for st in plan.stages:
        text = render_script(plan, st)
        assert "export MPD_DTYPE=float32" in text


def test_the_override_does_not_beat_an_explicit_dtype():
    plan = make("grace", gpu="rtx", allow_unsupported_gpu=True, extra_env={"MPD_DTYPE": "bfloat16"},
                spec_kw=dict(teacher="esm2-8m", student="esm2-8m"))
    assert plan.options.extra_env["MPD_DTYPE"] == "bfloat16"


def test_the_override_is_inert_on_a_bf16_gpu():
    plan = make("grace", allow_unsupported_gpu=True)
    assert "MPD_DTYPE" not in plan.options.extra_env and not warnings_about(plan, "bfloat16")


def test_float32_doubles_the_memory_model(monkeypatch):
    fixed_registry(monkeypatch)
    with pytest.raises(PlanError, match=r"esm2-650m needs about 1\d GB .*float32.* but t4 offers 14 GB"):
        make("grace", gpu="t4", allow_unsupported_gpu=True)
    assert make("grace", gpu="t4", allow_unsupported_gpu=True, spec_kw=dict(teacher="esm2-8m", student="esm2-35m")).resolved["gpu"] == "t4"


# ---------------------------------------------------------------- memory: micro-batch from the model, global batch kept
def test_the_3b_student_trains_at_micro_batch_one_on_a_40_gb_gpu(monkeypatch):
    fixed_registry(monkeypatch)
    plan = make("grace", spec_kw=dict(teacher="esm2-650m", student="esm2-3b"))
    r = plan.resolved
    assert (r["microbatch"], r["accumulation"], r["world_size"], r["global_batch"]) == (1, 64, 1, 64)
    assert any("micro-batch 1" in w and "global batch of 64 is kept through 64 accumulation steps" in w for w in plan.warnings)
    two = make("grace", spec_kw=dict(teacher="esm2-650m", student="esm2-3b", train_gpus=2))
    assert (two.resolved["microbatch"], two.resolved["accumulation"]) == (1, 32)


def test_bigger_gpus_and_smaller_students_keep_micro_batch_two(monkeypatch):
    fixed_registry(monkeypatch)
    assert make("grace", gpu="a40", spec_kw=dict(teacher="esm2-650m", student="esm2-3b")).resolved["microbatch"] == 2     # 48 GB
    assert make("vision", spec_kw=dict(teacher="esm2-650m", student="esm2-3b")).resolved["microbatch"] == 2
    for student in ("esm2-8m", "esm2-35m", "esm2-150m", "esm2-650m"):
        assert make("grace", spec_kw=dict(teacher="esm2-3b", student=student)).resolved["microbatch"] == 2, student


def test_activation_memory_scales_linearly_with_the_micro_batch(monkeypatch):
    fixed_registry(monkeypatch)
    a100 = Cluster.load("grace").gpu("a100")
    model = registry.get("esm2-650m")
    w, two = planner._need_parts(model, a100, 2)
    _, one = planner._need_parts(model, a100, 1)
    assert two == pytest.approx(12.0) and one == pytest.approx(6.0) and w == pytest.approx(1.3)
    assert planner._need_gb(model, a100, 2) == pytest.approx(1.3 + 12 + planner.FIXED_GB)
    _, inference = planner._need_parts(model, a100, None)
    assert inference == pytest.approx(12 * planner.INFERENCE_ACT_FRACTION)


def test_a_model_that_fits_nowhere_is_rejected_with_the_numbers(monkeypatch):
    fixed_registry(monkeypatch)
    with pytest.raises(PlanError, match=r"esm2-3b needs about 4\d GB .*float32.* but rtx offers 20 GB usable per GPU"):
        make("grace", gpu="rtx", allow_unsupported_gpu=True, spec_kw=dict(teacher="esm2-650m", student="esm2-3b"))


def test_the_memory_warning_gives_the_right_advice():
    plan = make("grace")
    advice = " ".join(warnings_about(plan, "GPU memory"))
    assert "estimate, not a measurement" in advice and "larger GPU" in advice and "data-parallel" in advice
    assert not any("may need more GPUs" in w for w in plan.warnings)           # adding GPUs does not reduce per-GPU memory


def test_large_teachers_still_fit_one_a100_for_inference(monkeypatch):
    fixed_registry(monkeypatch)
    plan = make("grace", spec_kw=dict(teacher="esm2-3b", student="esm2-650m"))
    assert stage(plan, "teacher").gpus == 1


# ---------------------------------------------------------------- walltime estimates (R-13, C11)
def test_training_walltime_for_the_measured_configuration_is_generous():
    plan = make("vision")                                    # esm2-650m on 4 H200s, the configuration that was timed
    measured_job_minutes = 34.3
    assert stage(plan, "train").walltime_s >= 1.5 * measured_job_minutes * 60
    assert stage(plan, "train").walltime_basis == "measured"


def test_documented_example_walltimes():
    """The numbers docs/GRACE.md quotes; a change here is a change to the documentation."""
    plan = make("grace")
    assert stage(plan, "train").walltime_s == 10 * 3600 + 15 * 60
    assert stage(plan, "score").walltime_s == 30 * 60 and stage(plan, "evaluate").walltime_s == 3600
    assert make("vision").stages[2].walltime_s == 3600


def test_walltime_grows_with_the_student(monkeypatch):
    fixed_registry(monkeypatch)
    gpu = Cluster.load("grace").gpu("a100")
    times = [train_walltime_s(registry.get(k), "all", 2103, 1, gpu) for k in ("esm2-8m", "esm2-35m", "esm2-150m", "esm2-650m", "esm2-3b")]
    assert times == sorted(times) and times[-1] > 3 * times[-2] > 3 * 3 * times[0]
    plans = [make("grace", spec_kw=dict(teacher="esm2-650m", student=k)) for k in ("esm2-8m", "esm2-150m", "esm2-650m", "esm2-3b")]
    trains = [stage(p, "train").walltime_s for p in plans]
    assert trains == sorted(trains) and len(set(trains)) == len(trains)
    scores = [stage(p, "score").walltime_s for p in plans]
    assert scores == sorted(scores)


def test_walltime_scales_with_gpu_slowdown_width_and_updates():
    st = registry.get("esm2-650m")
    grace, vision = Cluster.load("grace"), Cluster.load("vision")
    a100, a40, h200 = grace.gpu("a100"), grace.gpu("a40"), vision.gpu("h200")
    assert train_walltime_s(st, "all", 2103, 1, h200) < train_walltime_s(st, "all", 2103, 1, a100) < train_walltime_s(st, "all", 2103, 1, a40)
    assert train_walltime_s(st, "all", 2103, 4, a100) < train_walltime_s(st, "all", 2103, 2, a100) < train_walltime_s(st, "all", 2103, 1, a100)
    assert train_walltime_s(st, "all", 100, 1, a100) < train_walltime_s(st, "all", 2103, 1, a100)
    assert train_walltime_s(st, "one", 2103, 1, a100) < train_walltime_s(st, "all", 2103, 1, a100)      # 53.2 vs 63.8 min per 1,000
    assert train_walltime_s(st, "all", 1, 1, h200) == 15 * 60                                              # start-up allowance, 15 min floor


def test_walltimes_round_up_to_a_quarter_hour():
    plan = make("grace")
    for st in plan.stages:
        assert st.walltime_s % (15 * 60) == 0


def test_teacher_and_score_time_come_from_token_counts(monkeypatch):
    fixed_registry(monkeypatch)
    big = make("grace", spec_kw=dict(teacher="esm2-3b", student="esm2-650m", method="direct-all"))       # dense union
    small = make("grace", spec_kw=dict(teacher="esm2-3b", student="esm2-650m", method="score-all"))      # observed positions only
    assert stage(big, "teacher").walltime_s > stage(small, "teacher").walltime_s
    t650 = make("grace", spec_kw=dict(teacher="esm2-650m", student="esm2-650m", method="direct-all"))
    assert stage(t650, "teacher").walltime_s < stage(big, "teacher").walltime_s
    venus = make("grace", spec_kw=dict(train_data="venusmuthub", method="direct-all", eval_data=("venusmuthub",)))
    assert stage(venus, "teacher").walltime_s > stage(big, "teacher").walltime_s                         # 56.7M tokens against 39.0M
    few = make("grace", spec_kw=dict(method="direct-one", assays=("A",)))
    assert stage(few, "teacher").walltime_s == 15 * 60
    assert stage(big, "teacher").walltime_s < 6 * 3600 and stage(big, "score").walltime_s < 2 * 3600       # the old flat defaults


def test_a_unit_scored_on_its_own_assay_gets_the_smallest_walltime():
    plan = make("grace", spec_kw=dict(method="score-one", eval_data=("proteingym",)))
    assert stage(plan, "score").walltime_s == 15 * 60 < stage(plan, "score-base").walltime_s


def test_forward_walltime_depends_on_tokens_model_and_gpu():
    grace, vision = Cluster.load("grace"), Cluster.load("vision")
    m = registry.get("esm2-650m")
    assert forward_walltime_s(m, 1e6, grace.gpu("a100")) <= forward_walltime_s(m, 40e6, grace.gpu("a100"))
    assert forward_walltime_s(m, 40e6, vision.gpu("h200")) < forward_walltime_s(m, 40e6, grace.gpu("a100"))
    assert forward_walltime_s(registry.get("esm2-3b"), 40e6, grace.gpu("a100")) > forward_walltime_s(m, 40e6, grace.gpu("a100"))


def test_user_times_win_and_are_marked(monkeypatch):
    plan = make("grace", train_time="06:00:00", score_time="01:00:00", teacher_time="02:00:00")
    assert (stage(plan, "train").walltime_s, stage(plan, "score").walltime_s, stage(plan, "teacher").walltime_s) == (21600, 3600, 7200)
    assert {stage(plan, n).walltime_basis for n in ("train", "score", "teacher", "score-base")} == {"user"}
    assert any("--train-time 06:00:00 is shorter than the estimate" in w and "TIMEOUT" in w for w in plan.warnings)
    long = make("grace", train_time="1-00:00:00")
    assert not any("shorter than the estimate" in w and "--train-time" in w for w in long.warnings)


def test_a_user_time_over_the_partition_limit_is_an_error_with_a_fix():
    with pytest.raises(PlanError, match="PartitionTimeLimit") as err:
        make("grace", train_time="5-00:00:00")
    assert "at most 4-00:00:00" in str(err.value)


def test_an_estimate_over_the_partition_limit_suggests_changing_the_plan(monkeypatch):
    fixed_registry(monkeypatch)
    with pytest.raises(PlanError) as err:
        make("grace", gpu="a40", spec_kw=dict(teacher="esm2-650m", student="esm2-3b", updates=20000))
    assert "PartitionTimeLimit" in str(err.value) and "--train-gpus" in str(err.value)


def test_every_number_that_is_not_a_measurement_says_so():
    grace = make("grace")
    for stage_name in ("train", "teacher", "score-base", "score"):
        line = next(w for w in grace.warnings if w.startswith(stage_name) and "walltime" in w)
        assert "an estimate, not measured" in line and "override with --" in line, stage_name
    assert "assumed" in next(w for w in grace.warnings if w.startswith("train"))            # the A100 slowdown is an assumption
    vision = make("vision")
    timed = next(w for w in vision.warnings if w.startswith("train"))
    assert "scaled from timed H200 runs of esm2-650m" in timed and "not measured" not in timed
    teacher = next(w for w in vision.warnings if w.startswith("teacher"))
    assert "an estimate, not measured" in teacher and "extrapolated to esm2-3b" in teacher         # a model nobody timed says so
    other = make("vision", spec_kw=dict(teacher="esm2-650m", student="esm2-150m"))
    assert "extrapolated to esm2-150m" in next(w for w in other.warnings if w.startswith("train"))


def test_describe_labels_estimates_and_explains_the_su_figures():
    text = describe(make("grace"))
    assert "estimate, not measured" in text.lower() or "Estimate, not measured." in text
    assert " est " in text and "meas = scaled from timed runs" in text
    assert re.search(r"^directory  .*/runs/esm2-3b__esm2-650m__proteingym-score-all__seed20260922__[0-9a-f]{8}$", text, re.M)
    assert "\"Requested SUs\" line that counts CPUs only" in text and "myproject" in text
    assert "(effective cores + GPU rate x GPUs) x hours" in text
    assert re.search(r"estimated upper bound: [\d,]+ SU", text)
    vision = describe(make("vision"))
    assert "estimated upper bound: not billed" in vision and "Requested SUs" not in vision and " meas " in vision


def test_su_total_matches_the_ams_rule_stage_by_stage():
    plan = make("grace")
    t = stage(plan, "train")
    assert t.su_estimate == pytest.approx((25 + 72) * t.walltime_s / 3600)           # 24 cores + 180 GB bill 25 cores
    assert stage(plan, "prepare").su_estimate == pytest.approx(8 * 4)                 # 8 cores, 32 GB (5 cores of memory), 4 h
    assert plan.total_su == pytest.approx(sum(s.su_estimate for s in plan.stages))


# ---------------------------------------------------------------- score shards (m2)
def test_a_unit_scored_on_its_own_assay_has_one_shard_and_the_base_keeps_the_default():
    plan = make("grace", spec_kw=dict(method="score-one", assays=("A", "B")))
    base = [t for t in stage(plan, "score-base").tasks]
    assert [t["num_shards"] for t in base if t["dataset"] == "proteingym"] == [4] * 4
    own = [t for t in stage(plan, "score").tasks if t["dataset"] == "proteingym"]
    assert own == [{"unit": "A", "dataset": "proteingym", "shard": 0, "num_shards": 1},
                   {"unit": "B", "dataset": "proteingym", "shard": 0, "num_shards": 1}]
    venus = [t for t in stage(plan, "score").tasks if t["dataset"] == "venusmuthub"]
    assert [(t["unit"], t["num_shards"]) for t in venus] == [("A", 1), ("B", 1)]            # the other dataset: its default
    assert len(stage(plan, "score").tasks) == 4 and len(stage(plan, "score-base").tasks) == 5


def test_all_assay_units_use_the_dataset_default_shards():
    plan = make("grace", spec_kw=dict(method="direct-all"))
    assert [t["num_shards"] for t in stage(plan, "score").tasks] == [4, 4, 4, 4, 1]


def test_the_override_shards_apply_to_full_tables_only():
    plan = make("grace", spec_kw=dict(method="score-one", assays=("A",)), score_shards={"proteingym": 7, "venusmuthub": 2})
    assert len([t for t in stage(plan, "score-base").tasks]) == 9
    assert [t["num_shards"] for t in stage(plan, "score").tasks] == [1, 2, 2]
    assert plan.resolved["score_shards"] == {"proteingym": 7, "venusmuthub": 2}


def test_venus_trained_units_are_scored_on_their_own_assay_too():
    plan = make("grace", spec_kw=dict(method="score-one", train_data="venusmuthub", assays=("V1",)))
    tasks = stage(plan, "score").tasks
    assert [(t["dataset"], t["num_shards"]) for t in tasks] == [("proteingym", 4)] * 4 + [("venusmuthub", 1)]


def test_resolved_records_the_shard_counts_the_stages_filter_by():
    plan = make("grace", score_shards={"proteingym": 3})
    assert plan.resolved["score_shards"] == {"proteingym": 3, "venusmuthub": 1}
    only = make("grace", spec_kw=dict(eval_data=("venusmuthub",)))
    assert only.resolved["score_shards"] == {"venusmuthub": 1}
    assert {t["dataset"] for t in stage(only, "score").tasks} == {"venusmuthub"}


@pytest.mark.parametrize("scope,unit,dataset", list(itertools.product(("all", "one"), ("base", "all", "A"), ("proteingym", "venusmuthub"))))
def test_planner_and_stage_agree_on_what_is_scored_on_an_own_assay(scope, unit, dataset):
    try:
        from mpd.stages.common import Campaign
    except ImportError:
        pytest.skip("stages package unavailable")
    method = "score-one" if scope == "one" else "score-all"
    s = spec(method=method)
    campaign = Campaign(Path("/nowhere"), s, {})
    assert scored_on_own_assay(s, unit, dataset) == campaign.evaluated_on_own_assay(unit, dataset)


def test_score_shard_option_parsing():
    assert parse_score_shards("proteingym=4,venusmuthub=1") == {"proteingym": 4, "venusmuthub": 1}
    assert parse_score_shards(" proteingym = 2 ") == {"proteingym": 2}
    for bad, message in [("bad", "DATASET=N"), ("x", "DATASET=N"), ("", "DATASET=N"), ("proteingym=0", "from 1 to"), ("proteingym=-1", "from 1 to"),
                         ("proteingym=1001", "from 1 to"), ("foo=2", "not a dataset"), ("proteingym=two", "not a whole number"),
                         ("proteingym=1,proteingym=2", "twice"), ("proteingym=1.5", "not a whole number"), ("=3", "DATASET=N")]:
        with pytest.raises(PlanError, match=message):
            parse_score_shards(bad)
    with pytest.raises(PlanError, match="from 1 to"):
        make("grace", score_shards={"proteingym": 0})
    with pytest.raises(PlanError, match="not a dataset"):
        make("grace", score_shards={"nope": 2})


def test_shards_for_a_dataset_that_is_not_evaluated_are_ignored_with_a_note():
    plan = make("grace", spec_kw=dict(eval_data=("proteingym",)), score_shards={"venusmuthub": 3})
    assert any("--score-shards for venusmuthub is ignored" in w for w in plan.warnings)


# ---------------------------------------------------------------- chunking
def test_stage_chunks_cover_every_task_exactly_once():
    assays = tuple(f"assay{i}" for i in range(1001))
    plan = make("grace", spec_kw=dict(method="score-one", assays=assays))
    for name, total in (("score", 2 * 1001), ("score-base", 5), ("train", 1001)):
        parts = [s for s in plan.stages if s.name == name]
        assert all(len(p.tasks) <= 1000 for p in parts), name
        assert [p.task_offset for p in parts] == [1000 * k for k in range(len(parts))], name
        assert sum(len(p.tasks) for p in parts) == total, name
    assert [len(s.tasks) for s in plan.stages if s.name == "train"] == [1000, 1]
    assert set(stage(plan, "evaluate").depends_on) >= {p.full_name for p in plan.stages if p.name == "score"}


def test_large_arrays_are_chunked_under_the_slurm_limit():
    assays = tuple(f"assay{i}" for i in range(450))
    plan = make("grace", spec_kw=dict(method="score-one", assays=assays))
    parts = [s for s in plan.stages if s.name == "score"]
    assert len(parts) == 1 and len(parts[0].tasks) == 900                              # own-assay shards: 2 tasks per unit
    plan = make("grace", spec_kw=dict(method="score-one", assays=assays), score_shards={"venusmuthub": 2})
    assert sum(len(s.tasks) for s in plan.stages if s.name == "score") == 450 * 3
    assert [s.task_offset for s in plan.stages if s.name == "score"] == [0, 1000]


# ---------------------------------------------------------------- campaign identity and nonce (C9 as amended by CONTRACTS 9.2, D2)
ID = re.compile(r"^(?P<spec>.+)__(?P<hash>[0-9a-f]{8})$")


def test_the_id_is_the_spec_id_plus_a_hash_and_names_the_directory():
    plan = make("grace")
    m = ID.match(plan.campaign_id)
    assert m and m["spec"] == plan.spec.campaign_id() and plan.campaign_dir == HOME / "runs" / plan.campaign_id
    assert plan.resolved["campaign_id"] == plan.campaign_id and make("grace").campaign_id == plan.campaign_id


def changed(**kw):
    return make(**kw).campaign_id


BASE = make("grace")
BASE_ID = BASE.campaign_id

# What can change a result, or where results are written: a different value is a different campaign (it trains again).
RESULT_OPTIONS = [
    ("eval data", dict(spec_kw=dict(eval_data=("proteingym",)))),
    ("another eval dataset", dict(spec_kw=dict(eval_data=("venusmuthub",)))),
    ("cluster", dict(cluster="vision")),
    ("gpu", dict(gpu="a40")),
    ("score shards", dict(score_shards={"proteingym": 2})),
    ("training width", dict(spec_kw=dict(train_gpus=2))),
    ("seed", dict(spec_kw=dict(seed=7))),
    ("MPD_DTYPE", dict(extra_env={"MPD_DTYPE": "float32"})),
    ("MPD_TF32", dict(extra_env={"MPD_TF32": "1"})),
    ("MPD_STRUCTURE_LEGACY_MASK", dict(extra_env={"MPD_STRUCTURE_LEGACY_MASK": "1"})),
]
# Where and when a job runs, and who pays: the same campaign (CONTRACTS 9.2), so that a recovery reuses finished work.
SCHEDULING_OPTIONS = [
    ("train time", dict(train_time="08:00:00")),
    ("score time", dict(score_time="01:00:00")),
    ("teacher time", dict(teacher_time="01:00:00")),
    ("array throttle", dict(array_throttle=3)),
    ("extra sbatch", dict(extra_sbatch=("--exclude=g001",))),
    ("extra modules", dict(extra_modules=("WebProxy",))),
    ("extra env", dict(extra_env={"FOO": "1"})),
    ("extra env that is no numerics variable of ours", dict(extra_env={"NCCL_DEBUG": "INFO", "MPD_KEEP_ALL_CHECKPOINTS": "1"})),
    ("all of them at once", dict(train_time="09:00:00", score_time="01:30:00", teacher_time="02:00:00", array_throttle=2,
                                 extra_sbatch=("--exclude=g002", "--mail-type=END"), extra_modules=("WebProxy",), extra_env={"FOO": "bar"})),
    ("python", dict(python="/elsewhere/bin/python")),
    ("home", dict(home="/other/home")),
    ("account", dict(account="another_acct")),
    ("no account", dict(account=None)),
    ("nonce", dict(nonce="ffffffffffff")),
]


@pytest.mark.parametrize("what,kw", RESULT_OPTIONS, ids=[w for w, _ in RESULT_OPTIONS])
def test_what_can_change_a_result_changes_the_id(what, kw):
    assert changed(**kw) != BASE_ID, what


@pytest.mark.parametrize("what,kw", SCHEDULING_OPTIONS, ids=[w for w, _ in SCHEDULING_OPTIONS])
def test_scheduling_options_and_locations_keep_the_id(what, kw):
    assert changed(**kw) == BASE_ID, what


@pytest.mark.parametrize("what,kw", SCHEDULING_OPTIONS, ids=[w for w, _ in SCHEDULING_OPTIONS])
def test_nothing_a_campaign_directory_keeps_depends_on_a_scheduling_option(what, kw):
    """`--force` rewrites only the scripts and plan.json, so spec.json, resolved.json and tasks/ must come out equal."""
    plan = make(**kw)
    assert plan.spec.to_json() == BASE.spec.to_json() and plan.resolved == BASE.resolved
    assert [(s.full_name, s.tasks, s.task_offset) for s in plan.stages] == [(s.full_name, s.tasks, s.task_offset) for s in BASE.stages]
    assert [s.depends_on for s in plan.stages] == [s.depends_on for s in BASE.stages]


def test_scheduling_options_do_reach_the_scripts_of_the_same_campaign():
    plan = make("grace", train_time="09:00:00", array_throttle=2, extra_sbatch=("--exclude=g002",), spec_kw=dict(method="direct-one"))
    assert plan.campaign_id == make("grace", spec_kw=dict(method="direct-one")).campaign_id
    text = render_script(plan, stage(plan, "train"))
    assert "#SBATCH --time=09:00:00" in text and "#SBATCH --array=0-2%2" in text and "#SBATCH --exclude=g002" in text


def test_the_default_shard_count_is_not_a_change():
    assert changed(score_shards={"proteingym": 4, "venusmuthub": 1}) == BASE_ID
    assert changed(spec_kw=dict(eval_data=("venusmuthub", "proteingym"))) == changed(spec_kw=dict(eval_data=("proteingym", "venusmuthub")))
    assert changed(spec_kw=dict(train_gpus=1)) != changed(spec_kw=dict(train_gpus=2))


def test_the_numerics_variables_are_part_of_the_identity(monkeypatch):
    monkeypatch.setenv("MPD_DTYPE", "float32")
    from_shell = changed()
    assert from_shell != BASE_ID                                                          # a float32 run is not a bf16 run
    monkeypatch.delenv("MPD_DTYPE")
    assert changed(extra_env={"MPD_DTYPE": "float32"}) == from_shell                      # the same job environment is the same campaign
    monkeypatch.setenv("MPD_TF32", "1")
    assert changed() != BASE_ID
    monkeypatch.delenv("MPD_TF32")
    monkeypatch.setenv("MPD_MODEL_ESM2_650M", "/weights/here")                            # where weights are is not what is computed
    monkeypatch.setenv("MPD_ACCOUNT", "env_acct_3")
    monkeypatch.setenv("MPD_CKPT_EVERY", "50")
    monkeypatch.setenv("MPD_KEEP_ALL_CHECKPOINTS", "1")
    assert changed() == BASE_ID


def test_a_flag_that_is_off_is_not_a_change():
    """The code reads MPD_TF32 and MPD_STRUCTURE_LEGACY_MASK as "== 1" and MPD_DTYPE as "if set": nothing else differs."""
    for env in ({"MPD_TF32": "0"}, {"MPD_TF32": ""}, {"MPD_STRUCTURE_LEGACY_MASK": "0"}, {"MPD_DTYPE": ""}):
        assert changed(extra_env=env) == BASE_ID, env
    assert changed(extra_env={"MPD_DTYPE": "bfloat16"}) != BASE_ID and changed(extra_env={"MPD_DTYPE": "float16"}) != changed(
        extra_env={"MPD_DTYPE": "bfloat16"})


def test_the_legacy_mask_reaches_a_job_only_through_env(monkeypatch):
    """It reproduces an earlier, wrong input and changes results: a stale export in a shell must not reach a campaign."""
    monkeypatch.setenv("MPD_STRUCTURE_LEGACY_MASK", "1")
    plan = make("grace")
    assert plan.campaign_id == BASE_ID and "MPD_STRUCTURE_LEGACY_MASK" not in "".join(render_all(plan).values())
    assert any("MPD_STRUCTURE_LEGACY_MASK is set in this shell but jobs do not inherit it" in w and "--env" in w for w in plan.warnings)
    via_env = make("grace", extra_env={"MPD_STRUCTURE_LEGACY_MASK": "1"})
    assert via_env.campaign_id != BASE_ID and not any("MPD_STRUCTURE_LEGACY_MASK" in w for w in via_env.warnings)
    assert all("export MPD_STRUCTURE_LEGACY_MASK=1\n" in text for text in render_all(via_env).values())
    monkeypatch.setenv("MPD_STRUCTURE_LEGACY_MASK", "0")                                  # off is off: nothing to warn about
    assert not any("MPD_STRUCTURE_LEGACY_MASK" in w for w in make("grace").warnings)
    monkeypatch.delenv("MPD_STRUCTURE_LEGACY_MASK")
    assert not any("MPD_STRUCTURE_LEGACY_MASK" in w for w in make("grace").warnings)
    assert "MPD_STRUCTURE_LEGACY_MASK" not in planner.FORWARDED_ENV and "MPD_STRUCTURE_LEGACY_MASK" in planner._NUMERICS_ENV


def test_the_nonce_is_per_plan():
    given = make("grace", nonce="abcdef123456")
    assert given.nonce == "abcdef123456"
    a, b = make("grace", nonce=None), make("grace", nonce=None)
    assert re.fullmatch(r"[0-9a-f]{12}", a.nonce) and a.nonce != b.nonce and a.campaign_id == b.campaign_id
    for bad in ("short", "has space!", "x" * 40, "semi;colon", ""):
        if bad == "":
            assert make("grace", nonce=bad).nonce                                         # empty = not given
            continue
        with pytest.raises(PlanError, match="nonce"):
            make("grace", nonce=bad)


def test_resolved_records_what_the_stages_need():
    r = make("grace", spec_kw=dict(train_gpus=2)).resolved
    assert (r["cluster"], r["gpu"], r["world_size"], r["train_gpus"], r["microbatch"], r["accumulation"]) == ("grace", "a100", 2, 2, 2, 16)
    assert r["score_shards"] == {"proteingym": 4, "venusmuthub": 1} and r["support"] == "observed"
    json.dumps(r)                                                                          # resolved.json must serialise


def test_a_campaign_id_with_odd_characters_is_refused():
    with pytest.raises(PlanError, match="campaign id"):
        make("grace", spec_kw=dict(method="score-one", assays=("bad id!",)))


# ---------------------------------------------------------------- paths (R-12)
def test_relative_paths_are_refused_by_the_planner():
    with pytest.raises(PlanError, match="relative path"):
        make("grace", home=Path("relhome"))
    for bad in ("env/bin/python", "./python", "../x/python", "relhome/env/bin/python"):
        with pytest.raises(PlanError, match="relative path"):
            make("grace", python=bad)


def test_a_bare_interpreter_name_is_allowed_but_warned_about():
    plan = make("grace", python="python3")
    assert plan.options.python == "python3" and any("looked up on PATH" in w for w in plan.warnings)
    assert not any("looked up on PATH" in w for w in make("grace").warnings)


def test_planned_paths_in_scripts_are_absolute(tmp_path):
    plan = make("grace", home=tmp_path / "h")
    for text in render_all(plan).values():
        for line in text.splitlines():
            for word in re.findall(r"(?:--campaign-dir|--output=|--error=|^cd|^export MPD_HOME=)\s*(\S+)", line):
                assert word.startswith("/"), line


# ---------------------------------------------------------------- option validation (R-27): messages, never tracebacks
@pytest.mark.parametrize("flag,value,message", [
    ("train_time", "8h", "--train-time: '8h' is not a walltime"),
    ("train_time", "08:00", "--train-time: '08:00' is ambiguous: Slurm reads it as 8 minutes. Write 08:00:00 for hours and minutes"),
    ("train_time", "0", "--train-time: '0' is zero"),
    ("train_time", "-1", "--train-time: '-1' is not a walltime"),
    ("train_time", "garbage", "--train-time: 'garbage' is not a walltime"),
    ("score_time", "00:00:00", "--score-time"),
    ("teacher_time", "8", "--teacher-time: '8' is ambiguous"),
])
def test_walltime_options_explain_themselves(flag, value, message):
    with pytest.raises(PlanError, match=re.escape(message)):
        make("grace", **{flag: value})


@pytest.mark.parametrize("kw,message", [
    (dict(peak_lr=0.0), "--lr must be positive"), (dict(peak_lr=-1e-5), "--lr must be positive"),
    (dict(peak_lr=float("nan")), "--lr must be a finite"), (dict(peak_lr=float("inf")), "--lr must be a finite"),
    (dict(kl_weight=-1.0), "--kl-weight must be zero or positive"), (dict(kl_weight=float("nan")), "--kl-weight must be a finite"),
])
def test_recipe_numbers_are_validated(kw, message):
    with pytest.raises(PlanError, match=message):
        make("grace", spec_kw=kw)
    assert make("grace", spec_kw=dict(peak_lr=1e-4, kl_weight=0.0)).resolved["peak_lr"] == 1e-4
    assert make("grace", spec_kw=dict(kl_weight=0.0, method="direct-all")).resolved["kl_weight"] == 0.0     # zero is a value, not "unset"


@pytest.mark.parametrize("n", [0, -3])
def test_array_throttle_must_be_positive(n):
    with pytest.raises(PlanError, match="--array-throttle must be at least 1"):
        make("grace", array_throttle=n)


def test_env_pair_parser():
    assert parse_env_pair("A=b") == ("A", "b") and parse_env_pair("A=") == ("A", "") and parse_env_pair("A=b=c") == ("A", "b=c")
    for bad, message in [("noequals", "NAME=VALUE"), ("=x", "not an environment"), ("A B=c", "not an environment"), ("MPD_ALLOW_HUB=1", "login nodes"),
                         ("A=x\ny", "one line")]:
        with pytest.raises(PlanError, match=message):
            parse_env_pair(bad)


def test_path_variables_of_the_shell_become_absolute_and_those_of_env_must_already_be(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for name, value in dict(MPD_FOLDSEEK="bin/foldseek", MPD_MODELS_DIR="models", MPD_MODEL_ESM2_8M="~/w", MPD_3DI_ENGINE="foldseek").items():
        monkeypatch.setenv(name, value)
    got = planner.forwarded_env()
    assert got["MPD_FOLDSEEK"] == str(tmp_path / "bin" / "foldseek") and got["MPD_MODELS_DIR"] == str(tmp_path / "models")
    assert got["MPD_MODEL_ESM2_8M"] == str(Path("~/w").expanduser()) and got["MPD_3DI_ENGINE"] == "foldseek"      # an engine name is no path
    monkeypatch.setenv("MPD_FOLDSEEK", "")
    assert planner.forwarded_env()["MPD_FOLDSEEK"] == ""                                                         # empty stays empty, it is not the current folder
    for name in ("MPD_FOLDSEEK", "MPD_MODELS_DIR", "MPD_MODEL_ESM2_650M"):
        with pytest.raises(PlanError, match=f"--env {name}=rel/x is a relative path"):
            make("grace", extra_env={name: "rel/x"})
    assert make("grace", extra_env={"MPD_FOLDSEEK": "/opt/foldseek/bin/foldseek", "MPD_3DI_ENGINE": "biotite"}).options.extra_env


def test_account_warning_only_where_accounts_are_billed():
    assert any("--account" in w and "myproject" in w for w in make("grace", account=None).warnings)
    assert not any("account" in w.lower() for w in make("vision", account=None).warnings)


def test_plan_options_keep_old_callers_working():
    o = planner.PlanOptions(cluster=Cluster.load("vision"), home=Path("/nonexistent/mpd"), python="/env/bin/python")
    assert (o.extra_env, o.extra_modules, o.nonce, o.allow_unsupported_gpu, o.extra_sbatch) == ({}, (), None, False, ())
    plan = make_plan(spec(), o)
    assert plan.nonce and plan.campaign_id and plan.options.python == "/env/bin/python"
    positional = planner.PlanOptions(Cluster.load("grace"), Path("/nonexistent/mpd"), "/env/bin/python")
    assert make_plan(spec(), positional).options.python == "/env/bin/python"


def test_make_plan_does_not_change_the_options_it_is_given():
    o = opts(python=None, account=None)
    before = dataclasses.replace(o)
    plan = make_plan(spec(), o)
    assert o == before and plan.options is not o and plan.options.python
    assert ACCOUNT not in render_all(plan)["train"]
    assert NONCE == plan.nonce
