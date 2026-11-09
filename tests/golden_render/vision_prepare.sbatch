#!/bin/bash
#SBATCH --job-name=mpd-prepare-esm2650m
#SBATCH --comment=mpd:esm2-3b__esm2-650m__proteingym-direct-one__3assays-d00e9e6d__seed20260922__cfe77bc5:0123456789ab:prepare
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --partition=def
#SBATCH --output=/nonexistent/mpd/runs/esm2-3b__esm2-650m__proteingym-direct-one__3assays-d00e9e6d__seed20260922__cfe77bc5/logs/%x_%j.out
#SBATCH --error=/nonexistent/mpd/runs/esm2-3b__esm2-650m__proteingym-direct-one__3assays-d00e9e6d__seed20260922__cfe77bc5/logs/%x_%j.err
#SBATCH --open-mode=append
#SBATCH --qos=standard

set -eo pipefail
type module >/dev/null 2>&1 || source /etc/profile 2>/dev/null || true
type module >/dev/null 2>&1 || { echo "MPD_MODULE_LOAD_FAILED the module command is not available (CUDA/12.8.0)" >&2; exit 1; }
module purge
module load CUDA/12.8.0 || { echo "MPD_MODULE_LOAD_FAILED CUDA/12.8.0" >&2; exit 1; }
for _v in $(compgen -e); do case $_v in MPD_*) unset "$_v";; esac; done; unset _v PYTHONPATH PYTHONHOME
set -u
export MPD_HOME=/nonexistent/mpd MPD_CLUSTER=vision HF_HOME=/nonexistent/mpd/hf_cache XDG_CACHE_HOME=/nonexistent/mpd/cache TORCH_HOME=/nonexistent/mpd/cache/torch TRITON_CACHE_DIR=/nonexistent/mpd/cache/triton MPLCONFIGDIR=/nonexistent/mpd/cache/mpl
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
echo "MPD_JOB_START host=$(hostname) job=${SLURM_JOB_ID:-} stage=prepare"
cd /nonexistent/mpd/runs/esm2-3b__esm2-650m__proteingym-direct-one__3assays-d00e9e6d__seed20260922__cfe77bc5

/env/bin/python -m mpd.stages.prepare --campaign-dir /nonexistent/mpd/runs/esm2-3b__esm2-650m__proteingym-direct-one__3assays-d00e9e6d__seed20260922__cfe77bc5 --stage prepare --task-index 0
echo "MPD_STAGE_DONE prepare"
