import pytest

from mpd.slurm.cluster import Cluster, ClusterError, format_walltime, parse_walltime


@pytest.fixture(scope="module")
def grace():
    return Cluster.load("grace")


def test_walltime_roundtrip():
    assert parse_walltime("4-00:00:00") == 4 * 86400
    assert parse_walltime("01:30:00") == 5400
    assert parse_walltime("90") == 90 * 60          # bare number = minutes, as in Slurm
    assert parse_walltime("1-12") == 36 * 3600      # D-H
    assert format_walltime(5400) == "01:30:00"
    assert format_walltime(4 * 86400) == "4-00:00:00"
    assert parse_walltime(format_walltime(123456)) == 123456


def test_grace_gpu_inventory_matches_hprc_primer(grace):
    assert {k: g.per_node for k, g in grace.gpus.items()} == {"a100": 2, "a40": 3, "rtx": 2, "t4": 4}
    assert grace.gpus["a100"].su_per_gpu_hour == 72 and grace.gpus["t4"].su_per_gpu_hour == 24
    assert grace.gpus["a100"].max_walltime_s == 4 * 86400
    assert grace.gpus["a40"].partition == "gpu-a40"


def test_proportional_resources_match_maxconfig(grace):
    # `maxconfig -g a100 -G 1` recommends half a node: 24 cores, 180 GB.
    a100 = grace.gpu("a100")
    assert grace.cpus_and_mem(a100, 1) == (24, 180)
    assert grace.cpus_and_mem(a100, 2) == (48, 360)


def test_su_estimate_reproduces_the_three_documented_examples(grace):
    a100 = grace.gpu("a100")
    # 24 cores + 1 A100 + 180 GB for 96 h = 9,312 SU (AlphaFold course, maxconfig output)
    assert grace.estimate_su(gpu=a100, n_gpus=1, cpus=24, mem_gb=180, walltime_s=96 * 3600) == pytest.approx(9312)
    # 24 cores, 48 GB, 6.5 h = 156 SU (primer job file); 1 core, 4 GB, 1.5 h = 1.5 SU
    assert grace.estimate_su(gpu=None, n_gpus=0, cpus=24, mem_gb=48, walltime_s=int(6.5 * 3600)) == pytest.approx(156)
    assert grace.estimate_su(gpu=None, n_gpus=0, cpus=1, mem_gb=4, walltime_s=int(1.5 * 3600)) == pytest.approx(1.5)


def test_policy_checks(grace):
    a100 = grace.gpu("a100")
    assert not grace.check_gpu_job(a100, 2, 3600)
    too_many = grace.check_gpu_job(a100, 3, 3600)
    assert any(level == "error" and "does not fit one node" in msg for level, msg in too_many)
    too_long = grace.check_gpu_job(a100, 1, 5 * 86400)
    assert any(level == "error" and "PartitionTimeLimit" in msg for level, msg in too_long)
    a40 = grace.check_gpu_job(grace.gpu("a40"), 1, 3600)
    assert any(level == "warning" for level, _ in a40)   # unverified access is surfaced, not hidden


def test_unknown_names_fail_clearly(grace):
    with pytest.raises(ClusterError, match="no GPU type"):
        grace.gpu("h100")
    with pytest.raises(ClusterError, match="unknown cluster"):
        Cluster.load("nope")


def test_storage_root_needs_env(monkeypatch, grace):
    monkeypatch.delenv("SCRATCH", raising=False)
    with pytest.raises(ClusterError, match="SCRATCH"):
        grace.storage_dir()
    monkeypatch.setenv("SCRATCH", "/scratch/user/x")
    assert str(grace.storage_dir()) == "/scratch/user/x/mpd"
