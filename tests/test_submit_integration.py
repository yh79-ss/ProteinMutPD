"""submit and campaign with the real planner and renderer (slurm-core's code) against the fake Slurm toolbox."""

import json

import pytest

from fx_slurm import BAD_GRES, fake_slurm  # noqa: F401
from mpd.slurm.campaign import campaign_state, stage_progress
from mpd.slurm.cluster import Cluster
from mpd.slurm.plan import PlanOptions, make_plan
from mpd.slurm.submit import SubmitError, preflight_checks, submit, write_campaign
from mpd.spec import RunSpec


def spec(**kw):
    base = dict(teacher="esm2-3b", student="esm2-650m", train_data="proteingym", method="score-all", eval_data=("proteingym", "venusmuthub"))
    base.update(kw)
    return RunSpec(**base)


def plan_for(tmp_path, cluster="grace", **kw):
    opts = dict(cluster=Cluster.load(cluster), home=tmp_path / "home", python="/env/bin/python",
                account="123456789012" if cluster == "grace" else None)
    opts.update({k: kw.pop(k) for k in list(kw) if k in ("nonce", "gpu", "python")})
    return make_plan(spec(**kw), PlanOptions(**opts))


def run(fs, plan, **kw):
    scripts = kw.pop("scripts", None) or write_campaign(plan)
    return submit(plan, scripts, echo=lambda *_: None, sleep=fs.sleep, clock=fs.clock, **kw)


def deps(call):
    return next((a.split(":", 1)[1] for a in call.argv if a.startswith("--dependency=afterok:")), "")


def test_the_real_chain_is_accepted_by_the_fake_sbatch_and_hangs_together(fake_slurm, tmp_path):
    plan = plan_for(tmp_path)
    jobs = run(fake_slurm, plan)
    assert list(jobs) == [s.full_name for s in plan.stages]
    recorded = {j["stage"]: j for j in fake_slurm.submitted()}
    for stage in plan.stages:
        job = recorded[stage.full_name]
        assert job["comment"] == f"mpd:{plan.campaign_id}:{plan.nonce}:{stage.full_name}"             # script directive and command line agree
        assert job["deps"] == [int(jobs[d]) for d in stage.depends_on]
    st = campaign_state(plan.campaign_dir)
    assert (st.state, st.nonce, st.cluster, st.jobs) == ("submitted", plan.nonce, "grace", jobs)
    assert fake_slurm.calls("squeue") == []


def test_vision_chain_with_its_typed_gres_is_accepted_only_when_the_type_matches(fake_slurm, tmp_path):
    fake_slurm.configure(sbatch={"gres_types": ["H200"]})              # Slurm 25.05.6 on Vision: gpu:h200:1 is refused, gpu:H200:1 accepted
    plan = plan_for(tmp_path, cluster="vision")
    jobs = run(fake_slurm, plan)
    assert len(jobs) == len(plan.stages)
    fake_slurm.configure(sbatch={"gres_types": ["h200"]})
    other = plan_for(tmp_path, cluster="vision", seed=7)
    with pytest.raises(SubmitError, match="rejected stage .*Requested node configuration is not available"):
        run(fake_slurm, other)


def test_preflight_with_real_scripts_catches_a_wrong_gres_type_before_anything_is_queued(fake_slurm, tmp_path):
    plan = plan_for(tmp_path)
    scripts = write_campaign(plan)
    fake_slurm.configure(sbatch={"gres_types": ["a100", "a40", "rtx", "t4"]})
    assert {c.level for c in preflight_checks(plan, scripts)} == {"ok"}
    fake_slurm.configure(sbatch={"gres_types": ["T4"]})
    levels = {c.name: c.level for c in preflight_checks(plan, scripts)}
    assert "error" in levels.values() and all(c.test_only for c in fake_slurm.calls("sbatch"))
    assert fake_slurm.jobs() == {}
    bad = next(c for c in preflight_checks(plan) if c.level == "error")           # no scripts given: they are rendered
    assert BAD_GRES.split(": ")[-1] in bad.message and "gpu:a100" in bad.message


def test_a_big_campaign_is_submitted_in_chunks_that_evaluate_waits_for(fake_slurm, tmp_path):
    assays = tuple(f"assay{i}" for i in range(1100))
    plan = plan_for(tmp_path, method="score-one", assays=assays)
    parts = [s.full_name for s in plan.stages if s.name == "score"]
    assert len(parts) >= 2 and parts[1] == "score.1"            # more tasks than one Slurm array may hold
    jobs = run(fake_slurm, plan)
    (call,) = [c for c in fake_slurm.calls("sbatch") if c.info["stage"] == "evaluate"]
    assert set(deps(call).split(":")) >= {jobs[p] for p in parts}
    assert {j["stage"]: j["n_array"] for j in fake_slurm.submitted()}["score.1"] > 1
    progress = stage_progress(plan.campaign_dir)
    total = sum(len(s.tasks) for s in plan.stages if s.name == "score")
    assert progress["score"] == (0, total) and progress["train"] == (0, 1100)               # chunks share one task list
def test_force_with_a_fresh_nonce_continues_the_same_campaign(fake_slurm, tmp_path):
    first = plan_for(tmp_path, nonce="firstnonce01")
    jobs = run(fake_slurm, first)
    again = plan_for(tmp_path, nonce="secondnonce2")
    assert again.campaign_id == first.campaign_id and again.nonce != first.nonce
    with pytest.raises(SubmitError, match="already submitted"):
        write_campaign(again)
    before = len(fake_slurm.calls("sbatch"))
    scripts = write_campaign(again, force=True)                # same recipe: scripts and plan.json only
    assert json.loads((again.campaign_dir / "plan.json").read_text())["nonce"] == "secondnonce2"
    assert "secondnonce2" in scripts["train"].read_text()
    assert run(fake_slurm, again, scripts=scripts, force=True) == jobs and len(fake_slurm.calls("sbatch")) == before


def test_a_dry_run_leaves_files_that_a_later_real_submit_may_rewrite(fake_slurm, tmp_path):
    dry = plan_for(tmp_path, nonce="drynonce0001")
    write_campaign(dry)                                         # `mpd submit --dry-run`: no marker
    assert campaign_state(dry.campaign_dir) is None
    real = plan_for(tmp_path, nonce="realnonce001")
    jobs = run(fake_slurm, real)
    assert "realnonce001" in (real.campaign_dir / "scripts" / "prepare.sbatch").read_text() and len(jobs) == len(real.stages)


@pytest.mark.parametrize("cluster", ["grace", "vision"])
def test_the_rendered_scripts_run_under_the_fake_toolbox_and_stop_when_the_gpu_is_unusable(fake_slurm, tmp_path, cluster):
    import subprocess
    plan = plan_for(tmp_path, cluster=cluster, python="/bin/echo")           # the "interpreter" just prints the stage command line
    scripts = write_campaign(plan)
    fake_slurm.set_modules(list(plan.modules) + ["Anaconda3/2024.02-1"])
    for stage in ("prepare", "train"):
        out = subprocess.run(["bash", str(scripts[stage])], capture_output=True, text=True, env=fake_slurm.env())
        assert out.returncode == 0, out.stderr
        assert "MPD_JOB_START host=" in out.stdout and f"MPD_STAGE_DONE {stage}" in out.stdout and f"--stage {stage}" in out.stdout
    assert [c.argv[0] for c in fake_slurm.calls("module")][:1] == ["purge"]
    if plan.modules:
        assert ["load", *plan.modules] in [c.argv for c in fake_slurm.calls("module")]
    fake_slurm.configure(nvidia_smi={"fail": True})
    out = subprocess.run(["bash", str(scripts["train"])], capture_output=True, text=True, env=fake_slurm.env())
    assert out.returncode == 31 and "MPD_NO_GPU" in out.stderr and "MPD_STAGE_DONE" not in out.stdout
    fake_slurm.set_modules(["Something/Else"])
    out = subprocess.run(["bash", str(scripts["prepare"])], capture_output=True, text=True, env=fake_slurm.env())
    assert (out.returncode == 1 and "MPD_MODULE_LOAD_FAILED" in out.stderr) == bool(plan.modules)
