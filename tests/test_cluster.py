import copy
import math
import os
import re
from fractions import Fraction
from pathlib import Path

import pytest
import yaml

from fx_render import clean_mpd_env  # noqa: F401  (autouse)
from mpd.slurm.cluster import Cluster, ClusterError, available_clusters, format_walltime, parse_walltime

CLUSTERS_DIR = Path(__file__).resolve().parents[1] / "src" / "mpd" / "clusters"


@pytest.fixture(scope="module")
def grace():
    return Cluster.load("grace")


@pytest.fixture(scope="module")
def vision():
    return Cluster.load("vision")


def profile(**changes) -> dict:
    """The grace profile as a dict, with dotted-path changes applied; a value of KeyError removes the key."""
    data = copy.deepcopy(yaml.safe_load((CLUSTERS_DIR / "grace.yaml").read_text()))
    for path, value in changes.items():
        *parents, leaf = path.split(".")
        node = data
        for key in parents:
            node = node[key]
        if value is KeyError:
            del node[leaf]
        else:
            node[leaf] = value
    return data


# ---------------------------------------------------------------- walltimes
def test_walltime_roundtrip():
    assert parse_walltime("4-00:00:00") == 4 * 86400
    assert parse_walltime("01:30:00") == 5400
    assert parse_walltime("90") == 90 * 60          # bare number = minutes, as in Slurm
    assert parse_walltime("1-12") == 36 * 3600      # D-H
    assert parse_walltime("1-12:30") == 36 * 3600 + 30 * 60
    assert format_walltime(5400) == "01:30:00"
    assert format_walltime(4 * 86400) == "4-00:00:00"
    assert parse_walltime(format_walltime(123456)) == 123456


@pytest.mark.parametrize("text", ["8h", "", "  ", "abc", "-5", "1:2:3:4", "1--2", "0", "00:00:00", "0-0", "00:90:00", "1-30:00",
                                  "00:00:61", "999-00:00:00"])
def test_garbage_walltimes_are_cluster_errors_not_value_errors(text):
    with pytest.raises(ClusterError):
        parse_walltime(text)
    assert not isinstance(ClusterError("x"), (KeyError, IndexError))


@pytest.mark.parametrize("text,fragment", [("08:00", "ambiguous"), ("8", "ambiguous"), ("90", "01:30:00"), ("0", "zero"),
                                           ("8h", "not a walltime"), ("-1", "not a walltime")])
def test_strict_walltime_explains_what_to_write(text, fragment):
    with pytest.raises(ClusterError, match=fragment):
        parse_walltime(text, strict=True)
    assert parse_walltime("08:00:00", strict=True) == 8 * 3600
    assert parse_walltime("1-00:00", strict=True) == 86400            # a day part makes it unambiguous


# ---------------------------------------------------------------- the packaged profiles
def test_grace_gpu_inventory_matches_hprc_documents(grace):
    assert {k: g.per_node for k, g in grace.gpus.items()} == {"a100": 2, "a40": 2, "rtx": 2, "t4": 4}
    assert {k: g.su_per_gpu_hour for k, g in grace.gpus.items()} == {"a100": 72, "a40": 48, "rtx": 48, "t4": 24}
    assert {k: g.mem_gb for k, g in grace.gpus.items()} == {"a100": 40, "a40": 48, "rtx": 24, "t4": 16}
    assert grace.gpus["a100"].max_walltime_s == 4 * 86400
    assert grace.gpus["a40"].partition == "gpu-a40" and grace.gpus["a40"].max_walltime_s == 4 * 86400
    assert {k: g.bf16 for k, g in grace.gpus.items()} == {"a100": True, "a40": True, "rtx": False, "t4": False}
    assert {k: g.gres for k, g in grace.gpus.items()} == {"a100": "a100", "a40": "a40", "rtx": "rtx", "t4": "t4"}


def test_grace_header_and_environment_facts(grace):
    assert grace.sbatch_extra == () and grace.modules == ()          # no --export, no --get-user-env: see the comment in grace.yaml
    assert grace.su_mem_total_gb == 351.5625 and grace.node_cores == 48 and grace.node_mem_gb == 360
    assert grace.forbidden_storage_prefixes == ("/home",)
    assert (grace.home_max_files, grace.home_max_gb, grace.file_quota_warning) == (10000, 10, 250000)
    assert grace.default_train_gpus == 1 and grace.array_throttle == 8 and grace.resource_rule == "proportional_to_node"
    assert grace.cpu_partition is None and grace.cpu_gres is None


def test_the_grace_profile_keeps_its_evidence_labels_and_unverified_list():
    text = (CLUSTERS_DIR / "grace.yaml").read_text()
    for needle in ("UNVERIFIED", "ran ", "doc ", "unverified", "sbatch --test-only", "maxconfig", "MaxArraySize", "gpu-a40",
                   "Nothing rendered from this profile has been run on Grace"):
        assert needle in text, needle
    assert "--export" not in "".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#")) and "sbatch_extra: []" in text


def test_vision_spells_its_gres_type_as_the_site_does(vision):
    assert vision.gpus["h200"].gres == "H200" and vision.gpus["h200"].name == "h200"
    assert vision.default_train_gpus is None and vision.cpu_gres is None
    assert vision.sbatch_extra == ("--qos=standard",) and vision.modules == ("CUDA/12.8.0",)
    assert not vision.bills_su and Cluster.load("grace").bills_su


def test_profiles_loaded_by_name_and_by_path_agree(tmp_path, grace):
    copy_ = tmp_path / "mine.yaml"
    copy_.write_text((CLUSTERS_DIR / "grace.yaml").read_text())
    assert Cluster.load(str(copy_)) == grace
    assert sorted(available_clusters()) == ["grace", "vision"]


def test_gpu_gres_defaults_to_the_key():
    data = profile()
    data["gpus"]["a100"].pop("gres", None)
    assert Cluster.from_dict(data).gpus["a100"].gres == "a100"
    data["gpus"]["a100"]["gres"] = "A100"
    assert Cluster.from_dict(data).gpus["a100"].gres == "A100"


def test_packaged_files_carry_no_identity():
    """The repository is public: no account number, user name, project or home path in what this area ships."""
    root = Path(__file__).resolve().parents[1] / "src" / "mpd"
    files = [*CLUSTERS_DIR.glob("*.yaml"), *(root / "slurm" / f"{n}.py" for n in ("cluster", "plan", "render")),
             root / "presets.py", root / "envrecord.py"]
    banned = ["yi" + "hong", "tamu" + ".edu", "prj" + "-02", "mol" + "-cell", "Protein" + "OPD", "/scratch/" + "project", "vs" + "ts",
              "gh" + "p_", "github" + "_pat_", "MEP" + "-SiPLM"]
    for path in files:
        text = path.read_text().replace("hprc.tamu.edu", "")        # the public HPRC hosts are allowed
        for needle in banned:
            assert needle not in text, f"{path.name} contains {needle!r}"
        digits = [m for m in re.findall(r"(?<![0-9.])[0-9]{12,13}(?![0-9])", text) if m != "123456789012"]
        assert not digits, f"{path.name}: {digits}"


# ---------------------------------------------------------------- resources, policy, SU
def test_proportional_resources_match_maxconfig(grace):
    # `maxconfig -g a100 -G 1` recommends half a node: 24 cores, 180 GB.
    a100 = grace.gpu("a100")
    assert grace.cpus_and_mem(a100, 1) == (24, 180)
    assert grace.cpus_and_mem(a100, 2) == (48, 360)


def test_su_rule_is_the_ams_memory_equivalent_core_rule(grace):
    cb = grace.cpu_billing
    assert (cb(2, 13), cb(2, 20), cb(10, 150)) == (2, 3, 21)                        # the worked examples of the AMS page
    assert (cb(24, 180), cb(1, 360), cb(48, 360), cb(2, 400)) == (25, 48, 48, 48)   # maxconfig; a whole node's memory costs the node
    a100 = grace.gpu("a100")
    assert grace.estimate_su(gpu=a100, n_gpus=2, cpus=2, mem_gb=250, walltime_s=3600) == pytest.approx(179)    # HPRC multi-GPU example
    assert grace.estimate_su(gpu=None, n_gpus=0, cpus=4, mem_gb=100, walltime_s=4 * 3600, tasks=5) == pytest.approx(280)  # a job_submit line


def test_the_billing_rule_has_no_floating_point_error_at_the_boundaries(grace):
    exact = lambda cores, mem: max(cores, min(48, math.ceil(Fraction(str(mem)) / Fraction("351.5625") * 48)))   # noqa: E731
    for n in range(1, 49):
        edge = float(Fraction("351.5625") * n / 48)                 # memory that is exactly n cores' worth
        assert grace.cpu_billing(1, edge) == exact(1, edge) == n
        assert grace.cpu_billing(1, edge + 0.001) == exact(1, edge + 0.001) == min(48, n + 1)
    for cores in (1, 2, 8, 24, 48):
        for mem in [x / 4 for x in range(4, 1600, 7)]:
            assert grace.cpu_billing(cores, mem) == exact(cores, mem), (cores, mem)


def test_su_estimate_reproduces_the_documented_examples(grace):
    a100 = grace.gpu("a100")
    # 24 cores + 1 A100 + 180 GB for 96 h = 9,312 SU (AlphaFold course, maxconfig output)
    assert grace.estimate_su(gpu=a100, n_gpus=1, cpus=24, mem_gb=180, walltime_s=96 * 3600) == pytest.approx(9312)
    # 24 cores, 48 GB, 6.5 h = 156 SU (primer job file); 1 core, 4 GB, 1.5 h = 1.5 SU
    assert grace.estimate_su(gpu=None, n_gpus=0, cpus=24, mem_gb=48, walltime_s=int(6.5 * 3600)) == pytest.approx(156)
    assert grace.estimate_su(gpu=None, n_gpus=0, cpus=1, mem_gb=4, walltime_s=int(1.5 * 3600)) == pytest.approx(1.5)


def test_a40_costs_less_per_hour_than_a100(grace):
    a40 = grace.gpu("a40")
    assert grace.estimate_su(gpu=a40, n_gpus=1, cpus=24, mem_gb=180, walltime_s=3600) == pytest.approx(25 + 48)
    assert grace.cpus_and_mem(a40, 1) == (24, 180) and grace.cpus_and_mem(a40, 2) == (48, 360)


def test_policy_checks(grace):
    a100 = grace.gpu("a100")
    assert not grace.check_gpu_job(a100, 2, 3600)
    too_many = grace.check_gpu_job(a100, 3, 3600)
    assert any(level == "error" and "does not fit one node" in msg for level, msg in too_many)
    assert grace.gpu_count_error(a100, 0) and grace.gpu_count_error(a100, 1) is None
    too_long = grace.check_gpu_job(a100, 1, 5 * 86400)
    assert any(level == "error" and "PartitionTimeLimit" in msg for level, msg in too_long)
    assert any("Reduce --updates" in msg for _, msg in too_long)
    user_set = grace.check_gpu_job(a100, 1, 5 * 86400, user_set=True)
    assert any("at most 4-00:00:00" in msg for _, msg in user_set) and not any("Reduce --updates" in msg for _, msg in user_set)
    a40 = grace.check_gpu_job(grace.gpu("a40"), 1, 3600)
    assert any(level == "warning" and "gpu-a40" in msg for level, msg in a40)       # unverified access is surfaced, not hidden
    assert any(level == "error" for level, _ in grace.check_gpu_job(grace.gpu("a40"), 1, 5 * 86400))   # the 4-day limit


def test_unknown_names_fail_clearly(grace):
    with pytest.raises(ClusterError, match="no GPU type"):
        grace.gpu("h100")
    with pytest.raises(ClusterError, match="unknown cluster"):
        Cluster.load("nope")
    for path in ("../grace", "missing/profile.yaml", "nowhere.yaml"):       # anything path-like is a file, never a packaged name
        with pytest.raises(ClusterError, match="not found"):
            Cluster.load(path)
    with pytest.raises(ClusterError, match="unknown cluster"):
        Cluster.load("gr ace")


# ---------------------------------------------------------------- storage
def test_storage_root_needs_env(monkeypatch, grace):
    monkeypatch.delenv("SCRATCH", raising=False)
    with pytest.raises(ClusterError, match="SCRATCH"):
        grace.storage_dir()
    monkeypatch.setenv("SCRATCH", "")
    with pytest.raises(ClusterError, match="SCRATCH"):                  # an empty $SCRATCH would put the root at /mpd
        grace.storage_dir()
    monkeypatch.setenv("SCRATCH", "/scratch/user/x")
    assert str(grace.storage_dir()) == "/scratch/user/x/mpd"


def test_storage_dir_is_absolute(tmp_path, monkeypatch, grace, vision):
    monkeypatch.chdir(tmp_path)
    assert grace.storage_dir("relhome") == tmp_path / "relhome"
    assert grace.storage_dir("a/../b") == tmp_path / "b"
    monkeypatch.setenv("MPD_HOME", "rel_mpd_home")
    assert vision.storage_dir() == tmp_path / "rel_mpd_home"
    assert vision.storage_dir("~/mpd").is_absolute() and "~" not in str(vision.storage_dir("~/mpd"))


def test_a_home_under_the_forbidden_prefix_is_refused_with_the_canonical_text(grace, monkeypatch):
    monkeypatch.setenv("SCRATCH", "/scratch/user/x")
    with pytest.raises(ClusterError) as err:
        grace.storage_dir("/home/someone/mpd")
    assert str(err.value) == ("/home/someone/mpd is under /home (10 GB / 10,000 files on Grace). "
                              "Use scratch: export MPD_HOME=$SCRATCH/mpd")
    with pytest.raises(ClusterError, match="under /home"):
        grace.storage_dir("/home")
    assert grace.storage_dir("/homestead/mpd") == Path("/homestead/mpd")          # a prefix match is by path component
    assert grace.storage_dir("/scratch/user/x/mpd") == Path("/scratch/user/x/mpd")


def test_a_symlink_into_the_forbidden_prefix_is_refused(grace, tmp_path):
    link = tmp_path / "scratch_link"
    link.symlink_to("/home")
    with pytest.raises(ClusterError, match="under /home"):
        grace.storage_dir(str(link / "mpd"))


def test_vision_has_no_forbidden_prefix(vision):
    assert vision.storage_dir("/home/anyone/mpd") == Path("/home/anyone/mpd")


def test_generic_hint_when_the_root_is_not_an_env_template():
    data = profile(**{"storage.root": "/data/mpd"})
    cluster = Cluster.from_dict(data)
    with pytest.raises(ClusterError, match="pass --home with a path outside /home"):
        cluster.storage_dir("/home/x")


# ---------------------------------------------------------------- malformed custom profiles
@pytest.mark.parametrize("change,field", [
    ({"name": KeyError}, "name"),
    ({"node": KeyError}, "node"),
    ({"gpus": KeyError}, "gpus"),
    ({"default_gpu": KeyError}, "default_gpu"),
    ({"node.cores": KeyError}, "cores"),
    ({"node.usable_mem_gb": KeyError}, "usable_mem_gb"),
    ({"gpus.a100.per_node": KeyError}, "per_node"),
    ({"gpus.a100.mem_gb": KeyError}, "mem_gb"),
    ({"gpus.a100.max_walltime": KeyError}, "max_walltime"),
    ({"gpus.a100.per_node": "two"}, "per_node"),
    ({"gpus.a100.per_node": 0}, "per_node"),
    ({"gpus.a100.mem_gb": -1}, "mem_gb"),
    ({"gpus.a100.time_factor": 0}, "time_factor"),
    ({"gpus.a100.bf16": "yes"}, "bf16"),
    ({"gpus.a100.max_walltime": 345600}, "quoted string"),      # YAML reads an unquoted 96:00:00 as a base-60 integer
    ({"gpus.a100.max_walltime": "4 days"}, "max_walltime"),
    ({"gpus.a100.gres": "bad type"}, "gres"),
    ({"default_gpu": "h100"}, "default_gpu"),
    ({"array_throttle": 0}, "array_throttle"),
    ({"resource_rule": "magic"}, "resource_rule"),
    ({"sbatch_extra": ["--export=NONE --foo"]}, "sbatch_extra"),
    ({"sbatch_extra": ["-J name"]}, "sbatch_extra"),
    ({"modules": ["bad name"]}, "modules"),
    ({"modules": "GCC/12"}, "lists"),
    ({"account": "a b"}, "account"),
    ({"name": "has space"}, "name"),
    ({"storage.forbidden_prefixes": ["home"]}, "forbidden_prefixes"),
    ({"storage.home_max_files": "many"}, "home_max_files"),
    ({"default_train_gpus": 0}, "default_train_gpus"),
    ({"sbatch_extras": []}, "sbatch_extras"),                   # a typo is an error, not a silently ignored field
    ({"gpus.a100.speed": 3}, "speed"),
    ({"cpu.gres": "all of them"}, "gres"),
])
def test_malformed_profile_names_the_field(change, field):
    with pytest.raises(ClusterError, match=field):
        Cluster.from_dict(profile(**change))


def test_an_account_written_as_a_yaml_number_is_a_string():
    assert Cluster.from_dict(profile(account=123456789012)).account == "123456789012"
    assert Cluster.from_dict(profile(account="proj-1")).account == "proj-1"
    with pytest.raises(ClusterError, match="account"):
        Cluster.from_dict(profile(account=True))


def test_fixed_per_gpu_profile_needs_its_numbers():
    data = profile(resource_rule="fixed_per_gpu")
    with pytest.raises(ClusterError, match="fixed_per_gpu"):
        Cluster.from_dict(data)
    data["fixed_per_gpu"] = {"cpus": 8, "mem_gb": 64}
    assert Cluster.from_dict(data).cpus_and_mem(Cluster.from_dict(data).gpu("a100"), 2) == (16, 128)


def test_bad_files_are_cluster_errors(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: [unclosed\n")
    with pytest.raises(ClusterError, match="not valid YAML"):
        Cluster.load(str(bad))
    bad.write_text("- just\n- a list\n")
    with pytest.raises(ClusterError, match="expected a mapping"):
        Cluster.load(str(bad))
    bad.write_text("")
    with pytest.raises(ClusterError, match="bad.yaml"):
        Cluster.load(str(bad))
    bad.write_bytes(b"\xff\xfe\x00bad")
    with pytest.raises(ClusterError, match="cannot read"):
        Cluster.load(str(bad))


def test_optional_fields_default_sensibly():
    data = profile(**{"node.su_mem_total_gb": KeyError, "storage": {"root": "{env:MPD_HOME}"}, "default_train_gpus": KeyError})
    cluster = Cluster.from_dict(data)
    assert cluster.su_mem_total_gb == cluster.node_mem_gb
    assert (cluster.forbidden_storage_prefixes, cluster.home_max_files, cluster.default_train_gpus) == ((), None, None)
    data["cpu"]["gres"] = "gpu:1"
    assert Cluster.from_dict(data).cpu_gres == "gpu:1"


def test_module_name_rule_is_the_same_in_envrecord():
    from mpd import envrecord
    from mpd.slurm.cluster import MODULE_RE
    for name in ["Anaconda3/2024.02-1", "GCCcore/13.2.0", "CUDA/12.8.0", "a", "x+y", "bad name", "$(x)", "-x", "a;b", "a\nb", ""]:
        assert bool(MODULE_RE.fullmatch(name)) == bool(envrecord._MODULE.fullmatch(name)), name


def test_home_environment_is_not_touched(grace):
    before = dict(os.environ)
    grace.storage_dir("/scratch/user/y/mpd")
    assert dict(os.environ) == before
