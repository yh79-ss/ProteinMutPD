"""The GPU guard, the TF32 switch and the process-group arguments, on CPU with `torch.cuda` monkeypatched."""

import datetime
import re

import pytest

torch = pytest.importorskip("torch")

from fx_stages import env, plan_for, run_stages  # noqa: F401
from mpd.spec import RunSpec
from mpd.stages import common, score, teacher, train


def no_cuda(monkeypatch):
    monkeypatch.setenv("MPD_REQUIRE_CUDA", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def test_no_gpu_exits_31_with_a_diagnostic_line(monkeypatch, capsys):
    no_cuda(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    with pytest.raises(SystemExit) as exc:
        common.require_cuda()
    assert exc.value.code == 31
    err = capsys.readouterr().err
    assert re.fullmatch(r"MPD_NO_GPU: torch\.cuda\.is_available\(\) is False CUDA_VISIBLE_DEVICES=3 torch=\S+ cuda=\S+\n", err), err


def test_the_diagnostic_line_says_when_the_device_list_is_unset(monkeypatch, capsys):
    no_cuda(monkeypatch)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(SystemExit):
        common.require_cuda()
    assert "CUDA_VISIBLE_DEVICES=<unset> torch=" in capsys.readouterr().err


def test_a_cuda_runtime_error_is_reported_on_one_line(monkeypatch, capsys):
    """The case behind the bad-node incident: nvidia-smi lists the GPU, the first allocation fails."""
    monkeypatch.setenv("MPD_REQUIRE_CUDA", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    def broken(index):
        raise RuntimeError("CUDA error: unknown error\nCUDA kernel errors might be asynchronously reported")

    monkeypatch.setattr(common, "_probe_cuda", broken)
    with pytest.raises(SystemExit) as exc:
        common.require_cuda()
    err = capsys.readouterr().err
    assert exc.value.code == 31 and err.count("\n") == 1
    assert err.startswith("MPD_NO_GPU: RuntimeError: CUDA error: unknown error CUDA kernel errors might be asynchronously reported")


def test_every_ddp_rank_probes_its_own_device(monkeypatch, capsys):
    monkeypatch.setenv("MPD_REQUIRE_CUDA", "1")
    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    probed = []
    monkeypatch.setattr(common, "_probe_cuda", probed.append)
    common.require_cuda()
    assert probed == [1]
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)       # a node that shows one GPU cannot host rank 1
    with pytest.raises(SystemExit) as exc:
        common.require_cuda()
    assert exc.value.code == 31 and "cuda:1 is needed but 1 device(s) are visible" in capsys.readouterr().err


def test_the_guard_is_silent_without_the_variable(monkeypatch):
    monkeypatch.delenv("MPD_REQUIRE_CUDA", raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("the guard asked torch about CUDA"))
    common.require_cuda()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_the_real_probe_passes_on_a_machine_with_a_gpu(monkeypatch):
    monkeypatch.setenv("MPD_REQUIRE_CUDA", "1")
    common.require_cuda()


def test_a_gpu_stage_without_a_gpu_exits_31_and_writes_nothing(env, monkeypatch):
    """Without the guard these stages would run on the CPU in float32 (tests do exactly that)."""
    plan = plan_for(env)
    run_stages(plan, only={"prepare"})
    no_cuda(monkeypatch)
    for module, stage in ((teacher, "teacher"), (train, "train"), (score, "score-base")):
        with pytest.raises(SystemExit) as exc:
            module.main(["--campaign-dir", str(plan.campaign_dir), "--stage", stage, "--task-index", "0"])
        assert exc.value.code == 31, stage
    root = plan.campaign_dir
    assert not (root / "teacher").exists() and not (root / "train").exists() and not (root / "score").exists()


@pytest.mark.parametrize("stage", [teacher, train, score])
def test_gpu_stages_check_for_cuda_before_anything_else(stage, monkeypatch, tmp_path):
    class Reached(Exception):
        pass

    def stop():
        raise Reached

    monkeypatch.setattr(stage, "require_cuda", stop)
    spec = RunSpec(teacher="esm2-35m", student="esm2-8m", train_data="proteingym", method="direct-all",
                   eval_data=("proteingym",))
    campaign = common.Campaign(tmp_path / "nowhere", spec, {"world_size": 1})     # touching any file would fail differently
    with pytest.raises(Reached):
        stage.run(campaign, {"unit": "all", "dataset": "proteingym", "shard": 0, "num_shards": 1})


def test_tf32_is_opt_in(monkeypatch):
    before = torch.get_float32_matmul_precision()
    try:
        monkeypatch.delenv("MPD_TF32", raising=False)
        common.configure_torch()
        assert torch.get_float32_matmul_precision() == before
        monkeypatch.setenv("MPD_TF32", "1")
        common.configure_torch()
        assert torch.get_float32_matmul_precision() == "high" and common.provenance()["float32_matmul_precision"] == "high"
    finally:
        torch.set_float32_matmul_precision(before)


# -- process group (M4) ----------------------------------------------------------------------

def test_the_process_group_gets_a_60_minute_timeout_and_its_device(monkeypatch):
    calls = []
    monkeypatch.setattr(train.dist, "init_process_group",
                        lambda backend, timeout=None, device_id=None: calls.append((backend, timeout, device_id)))
    gpu = torch.device("cuda", 1)
    train.init_group(True, gpu)
    train.init_group(False, torch.device("cpu"))
    hour = datetime.timedelta(minutes=60)
    assert calls == [("nccl", hour, gpu), ("gloo", hour, None)]


def test_a_torch_without_device_id_still_starts_the_group(monkeypatch):
    calls = []
    monkeypatch.setattr(train.dist, "init_process_group", lambda backend, timeout=None: calls.append((backend, timeout)))
    train.init_group(True, torch.device("cuda", 0))
    assert calls == [("nccl", datetime.timedelta(minutes=60))]


def test_the_installed_torch_takes_a_timeout():
    import inspect
    params = inspect.signature(train.dist.init_process_group).parameters
    assert "timeout" in params                                        # device_id is optional: the stage inspects the signature
