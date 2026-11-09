#!/bin/bash
#SBATCH --job-name=mpd-score-esm2650m
#SBATCH --comment=mpd:esm2-3b__esm2-650m__proteingym-direct-one__3assays-d00e9e6d__seed20260922__cfe77bc5:0123456789ab:score
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:H200:1
#SBATCH --partition=def
#SBATCH --array=0-5%10
#SBATCH --output=/nonexistent/mpd/runs/esm2-3b__esm2-650m__proteingym-direct-one__3assays-d00e9e6d__seed20260922__cfe77bc5/logs/%x_%A_%a.out
#SBATCH --error=/nonexistent/mpd/runs/esm2-3b__esm2-650m__proteingym-direct-one__3assays-d00e9e6d__seed20260922__cfe77bc5/logs/%x_%A_%a.err
#SBATCH --open-mode=append
#SBATCH --qos=standard

set -eo pipefail
type module >/dev/null 2>&1 || source /etc/profile 2>/dev/null || true
type module >/dev/null 2>&1 || { echo "MPD_MODULE_LOAD_FAILED the module command is not available (CUDA/12.8.0)" >&2; exit 1; }
module purge
module load CUDA/12.8.0 || { echo "MPD_MODULE_LOAD_FAILED CUDA/12.8.0" >&2; exit 1; }
unset PYTHONPATH PYTHONHOME
set -u
export MPD_HOME=/nonexistent/mpd HF_HOME=/nonexistent/mpd/hf_cache XDG_CACHE_HOME=/nonexistent/mpd/cache TORCH_HOME=/nonexistent/mpd/cache/torch TRITON_CACHE_DIR=/nonexistent/mpd/cache/triton MPLCONFIGDIR=/nonexistent/mpd/cache/mpl
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MPD_REQUIRE_CUDA=1
nvidia-smi -L || { echo "MPD_NO_GPU: no usable GPU on $(hostname -s); resubmit with --sbatch-arg=--exclude=$(hostname -s)" >&2; exit 31; }
echo "MPD_JOB_START host=$(hostname) job=${SLURM_JOB_ID:-} stage=score"
cd /nonexistent/mpd/runs/esm2-3b__esm2-650m__proteingym-direct-one__3assays-d00e9e6d__seed20260922__cfe77bc5

/env/bin/python -m mpd.stages.score --campaign-dir /nonexistent/mpd/runs/esm2-3b__esm2-650m__proteingym-direct-one__3assays-d00e9e6d__seed20260922__cfe77bc5 --stage score --task-index $((SLURM_ARRAY_TASK_ID + 0))
echo "MPD_STAGE_DONE score"
