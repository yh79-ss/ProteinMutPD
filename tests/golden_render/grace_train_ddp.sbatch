#!/bin/bash
#SBATCH --job-name=mpd-train-esm2650m
#SBATCH --comment=mpd:esm2-3b__esm2-650m__proteingym-score-all__seed20260922__26b9eaa0__64b12982:0123456789ab:train
#SBATCH --time=05:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=360G
#SBATCH --gres=gpu:a100:2
#SBATCH --partition=gpu
#SBATCH --account=123456789012
#SBATCH --output=/nonexistent/mpd/runs/esm2-3b__esm2-650m__proteingym-score-all__seed20260922__26b9eaa0__64b12982/logs/%x_%j.out
#SBATCH --error=/nonexistent/mpd/runs/esm2-3b__esm2-650m__proteingym-score-all__seed20260922__26b9eaa0__64b12982/logs/%x_%j.err
#SBATCH --open-mode=append
#SBATCH --exclude=g001

set -eo pipefail
type module >/dev/null 2>&1 || source /etc/profile 2>/dev/null || true
type module >/dev/null 2>&1 || { echo "MPD_MODULE_LOAD_FAILED the module command is not available (Anaconda3/2024.02-1)" >&2; exit 1; }
module purge
module load Anaconda3/2024.02-1 || { echo "MPD_MODULE_LOAD_FAILED Anaconda3/2024.02-1" >&2; exit 1; }
for _v in $(compgen -e); do case $_v in MPD_*) unset "$_v";; esac; done; unset _v PYTHONPATH PYTHONHOME
set -u
export MPD_HOME=/nonexistent/mpd MPD_CLUSTER=grace HF_HOME=/nonexistent/mpd/hf_cache XDG_CACHE_HOME=/nonexistent/mpd/cache TORCH_HOME=/nonexistent/mpd/cache/torch TRITON_CACHE_DIR=/nonexistent/mpd/cache/triton MPLCONFIGDIR=/nonexistent/mpd/cache/mpl
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=24
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MPD_REQUIRE_CUDA=1
export NCCL_DEBUG=WARN
nvidia-smi -L || { echo "MPD_NO_GPU: no usable GPU on $(hostname -s); resubmit with --sbatch-arg=--exclude=$(hostname -s)" >&2; exit 31; }
echo "MPD_JOB_START host=$(hostname) job=${SLURM_JOB_ID:-} stage=train"
cd /nonexistent/mpd/runs/esm2-3b__esm2-650m__proteingym-score-all__seed20260922__26b9eaa0__64b12982

/env/bin/python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=2 -m mpd.stages.train --campaign-dir /nonexistent/mpd/runs/esm2-3b__esm2-650m__proteingym-score-all__seed20260922__26b9eaa0__64b12982 --stage train --task-index 0
echo "MPD_STAGE_DONE train"
