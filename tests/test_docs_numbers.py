"""The figures that README.md and docs/GRACE.md quote, against the planner, the Grace profile and the pinned sources.

Each figure is recomputed from the code and must appear in the page that quotes it. A change to the resource model, the presets, the registry or a pin
therefore fails here until the pages are updated. The planner never needs a GPU, Slurm or data: ProteinGym's 217 assay ids are only counted, so the
`score-one` figure is computed with 217 invented ids. Whether ProteinGym really has 217 assays is a fact about the data, checked by the loaders' tests.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fx_render import clean_mpd_env, make, stage  # noqa: F401

from mpd import cli
from mpd.data import models, sources
from mpd.slurm.cluster import Cluster, format_walltime

REPO = Path(__file__).resolve().parents[1]
README = (REPO / "README.md").read_text(encoding="utf-8")
GRACE = (REPO / "docs" / "GRACE.md").read_text(encoding="utf-8")
DATA = (REPO / "docs" / "DATA.md").read_text(encoding="utf-8")
BASIC_ALLOCATION_SU = 20_000                      # HPRC's default allocation (deck 2026-01), quoted by the pages, not known to the code


def flat(text: str) -> str:
    return " ".join(text.split())


def hms(seconds: int) -> str:
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def percent(total: float) -> str:
    share = 100 * total / BASIC_ALLOCATION_SU
    return f"{share:.1f}%" if share < 1 else f"{share:.0f}%"


def readme_example():
    return make("grace", spec_kw=dict(teacher="esm2-3b", student="esm2-650m", method="score-all", eval_data=("proteingym", "venusmuthub")))


def first_run():
    return make("grace", spec_kw=dict(teacher="esm2-8m", student="esm2-8m", method="score-all", eval_data=("proteingym",), updates=50, train_gpus=1),
                score_shards={"proteingym": 1})


def big_student():
    return make("grace", spec_kw=dict(teacher="esm2-650m", student="esm2-3b", method="score-all", eval_data=("proteingym", "venusmuthub")))


def every_assay():
    return make("grace", spec_kw=dict(teacher="esm2-3b", student="esm2-650m", method="score-one", eval_data=("proteingym",),
                                      assays=tuple(f"ASSAY_{i:03d}" for i in range(217))))


# -- the planner's figures ------------------------------------------------------------------------------------------------------------

def test_the_readme_example_costs_what_the_pages_say():
    plan = readme_example()
    total = round(plan.total_su)
    assert total == 1806, "the resource model changed: update README.md and docs/GRACE.md"
    assert f"{total:,} SU, {percent(total)} of a 20,000 SU Basic allocation" in flat(README)
    assert f"{total:,} SU, {percent(total)} of 20,000" in flat(GRACE)
    assert hms(stage(plan, "train").walltime_s) == "10:15:00" and "10:15:00" in README and "10:15:00" in GRACE
    assert hms(stage(plan, "teacher").walltime_s) == "03:00:00" and hms(stage(plan, "score").walltime_s) == "00:30:00"


def test_the_first_run_costs_what_the_pages_say():
    plan = first_run()
    total = round(plan.total_su)
    assert total == 133
    assert [round(s.su_estimate) for s in plan.stages] == [32, 24, 24, 24, 24, 4]
    assert [hms(s.walltime_s) for s in plan.stages] == ["04:00:00", "00:15:00", "00:15:00", "00:15:00", "00:15:00", "01:00:00"]
    assert f"{total} SU" in README and f"{total} SU" in GRACE
    assert f"{total} SU is {percent(total)} of a 20,000 SU allocation" in flat(GRACE)
    assert 0.1 < 100 * total / BASIC_ALLOCATION_SU < 1


def test_the_prepare_and_evaluate_stages_ask_for_what_the_pages_say():
    plan = readme_example()
    prepare, evaluate = stage(plan, "prepare"), stage(plan, "evaluate")
    assert (prepare.cpus, prepare.mem_gb, hms(prepare.walltime_s), round(prepare.su_estimate)) == (8, 32, "04:00:00", 32)
    assert (evaluate.cpus, evaluate.mem_gb, hms(evaluate.walltime_s)) == (4, 16, "01:00:00")
    for page in (README, GRACE):
        assert "04:00:00" in page and "8 CPUs" in page and "32 GB" in page
    assert "its 32 SU is an upper bound" in flat(README)


def test_the_big_student_and_the_every_assay_figures():
    big = big_student()
    assert round(big.total_su) == 4983 and hms(stage(big, "train").walltime_s) == "40:00:00"
    assert f"{round(big.total_su):,} SU, {percent(big.total_su)}" in flat(GRACE) and "1-16:00:00" in GRACE
    assert stage(big, "train").walltime_s == (1 * 24 + 16) * 3600
    many = every_assay()
    assert round(many.total_su, -2) == 184_700
    assert "184,700 SU" in README and "184,700 SU" in GRACE
    assert 8.5 < many.total_su / BASIC_ALLOCATION_SU < 9.5 and "about nine times the Basic allocation" in GRACE


def test_the_su_per_hour_table_in_grace_md_is_the_su_rule():
    grace = Cluster.load("grace")
    rows = re.findall(r"^\| (\d) (A100|A40), (\d+) cores?, (\d+) GB[^|]*\| (\d+) \| (\d+) \|$", GRACE, re.M)
    assert len(rows) == 7
    for n_gpus, kind, cores, mem, effective, per_hour in rows:
        gpu = grace.gpus["a100" if kind == "A100" else "a40"]
        assert grace.cpu_billing(int(cores), float(mem)) == int(effective)
        got = grace.estimate_su(gpu=gpu, n_gpus=int(n_gpus), cpus=int(cores), mem_gb=float(mem), walltime_s=3600)
        assert got == pytest.approx(int(per_hour)), (kind, cores, mem)
    a100 = grace.gpus["a100"]
    assert grace.estimate_su(gpu=a100, n_gpus=1, cpus=24, mem_gb=180, walltime_s=96 * 3600) == pytest.approx(9312)
    assert grace.estimate_su(gpu=a100, n_gpus=2, cpus=2, mem_gb=250, walltime_s=3600) == pytest.approx(179)
    assert "9,312 SU" in GRACE and "179 SU" in GRACE


def test_the_gpu_table_in_grace_md_is_the_profile():
    grace = Cluster.load("grace")
    table = {}
    for line in GRACE.splitlines():
        m = re.match(r"^\| `(a100|a40|rtx|t4)`(?: \(default\))? \| [^|]+\| (\d+) [^|]*\| (\d+) GB \| (\d+) \| `([\w-]+)` \| ([^|]+) \|$", line)
        if m:
            table[m[1]] = (int(m[2]), int(m[3]), int(m[4]), m[5], m[6].strip())
    assert set(table) == {"a100", "a40", "rtx", "t4"} == set(grace.gpus)
    for key, (per_node, mem, rate, partition, trained) in table.items():
        gpu = grace.gpus[key]
        assert (per_node, mem, rate, partition) == (gpu.per_node, int(gpu.mem_gb), int(gpu.su_per_gpu_hour), gpu.partition), key
        assert trained.startswith("refused") == (not gpu.bf16), key
    assert grace.default_gpu == "a100" and grace.default_train_gpus == 1


def test_the_partition_limits_in_grace_md_are_the_profile_limits():
    grace = Cluster.load("grace")
    limits = {}
    for line in GRACE.splitlines():
        m = re.match(r"^\| `(gpu|gpu-a40|short|medium|long|xlong|bigmem)` \| ([\d-]+:\d\d:\d\d) \|", line)
        if m:
            limits[m[1]] = m[2]
    assert limits["gpu"] == format_walltime(grace.gpus["a100"].max_walltime_s) == "4-00:00:00"
    assert limits["gpu-a40"] == format_walltime(grace.gpus["a40"].max_walltime_s) == "4-00:00:00"
    assert grace.cpu_partition is None                     # the pages say the CPU stages name no partition


def test_the_login_node_and_cost_guard_figures():
    grace = Cluster.load("grace")
    assert (grace.login_max_cores, grace.login_max_minutes) == (8, 60)
    assert "stops by itself after 55 minutes" in flat(GRACE) and grace.login_max_minutes - 5 == 55
    assert cli.DEFAULT_MAX_SU == 10_000 and "default 10,000" in README and "default 10,000" in GRACE
    assert cli.EXIT_BUDGET == 3 and "exits with status 3" in flat(GRACE)


def test_the_retry_and_checkpoint_figures():
    from inspect import signature

    from mpd.slurm.submit import submit
    sig = signature(submit).parameters
    assert sig["retries"].default == 6 and sig["budget_s"].default == 900.0 and sig["wait_s"].default == 60.0
    assert "up to 6 tries and about 15 minutes" in flat(GRACE) and "(try <k> of 6)" in (REPO / "docs" / "TROUBLESHOOTING.md").read_text()
    source = (REPO / "src" / "mpd" / "stages" / "train.py").read_text(encoding="utf-8")
    assert re.search(r"^CKPT_EVERY = 200\b", source, re.M)                                    # torch is not imported to read it
    assert "every 200 updates" in README and "200 unless you changed it" in GRACE


def test_the_walltime_model_constants_in_grace_md():
    from mpd import presets
    from mpd.slurm import plan as planner
    assert presets.H200_TRAIN_MIN_PER_1000 == {"all": 63.8, "one": 53.2}
    assert (presets.REFERENCE_PARAMS_M, presets.SIZE_EXPONENT, presets.H200_FORWARD_TOKENS_PER_S) == (655.0, 0.9, 36_000.0)
    assert (planner.TIME_SLACK, planner.STARTUP_S) == (1.5, 600)
    grace = Cluster.load("grace")
    assert (grace.gpus["a100"].time_factor, grace.gpus["a40"].time_factor) == (3.0, 6.0)
    block = next(b for b in re.findall(r"```\n(training minutes.*?)```", GRACE, re.S))
    for needle in ("63.8 all-assay | 53.2 one-assay", "(student parameters / 655)^0.9", "A100 3.0, A40 6.0", "x 1.5 slack + 10 minutes start-up", "36,000 per second"):
        assert needle in block, needle


# -- sizes ----------------------------------------------------------------------------------------------------------------------------

def megabytes(n: int) -> str:
    return f"{n / 1e6:.1f} MB"


def test_the_download_sizes_the_pages_quote():
    pins = models.pins()
    pair = pins["esm2-3b"].total_bytes + pins["esm2-650m"].total_bytes
    assert models.human_bytes(pair) == "13.98 GB" and "13.98 GB" in README and "13.98 GB" in GRACE
    assert "11.37 GB + 2.61 GB" in GRACE
    assert models.human_bytes(sum(pins[k].total_bytes for k in ("esm2-8m", "esm2-35m", "esm2-150m", "esm2-650m", "esm2-3b"))) == "14.74 GB" in README
    assert models.human_bytes(sum(p.total_bytes for p in pins.values())) == "17.48 GB" and "17.48 GB" in README
    assert models.human_bytes(pins["esm2-8m"].total_bytes) == "31.4 MB" and "esm2-8m` (31 MB)" in GRACE
    proteingym = sum(f.size for f in sources.PROTEINGYM.files)
    assert megabytes(proteingym) == "62.4 MB" and "62.4 MB" in README and "62.4 MB" in GRACE and "62.4 MB" in DATA
    structures = next(f for f in sources.PROTEINGYM.files if f.name == "ProteinGym_AF2_structures.zip")
    assert f"{structures.size / 1e6:.1f} MB" == "19.2 MB" and f"{structures.size:,} bytes" in DATA
    foldseek = sources.FOLDSEEK
    assert megabytes(foldseek.archive_size) == "11.9 MB" and megabytes(foldseek.binary_size) == "29.4 MB"
    assert "11.9 MB" in GRACE and "29.4 MB" in GRACE and "11.9 MB" in DATA and "29.4 MB" in DATA


def test_the_venusmuthub_file_count():
    manifest = (REPO / "src" / "mpd" / "data" / sources.VENUSMUTHUB.manifest).read_text(encoding="utf-8").splitlines()
    files = len([line for line in manifest if line and not line.startswith("#")]) - 1                  # the header line
    assert files == 796, files
    assert "796 small files" in README and "796 small files" in GRACE and "797" in README
