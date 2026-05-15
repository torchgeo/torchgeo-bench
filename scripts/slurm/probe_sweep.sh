#!/bin/bash
#SBATCH --job-name=tgb-probe
#SBATCH --partition=gpu_a100
#SBATCH --account=bgtj-tgirails
#SBATCH --array=0-0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#
# torchgeo-bench probe sweep on TGI RAILS.
#
# Submit as:
#
#   sbatch --account=$PROJECT \
#          --array=0-$(( $(wc -l < scripts/slurm/probe_sweep.jobs) - 1 )) \
#          scripts/slurm/probe_sweep.sh
#
# The job-list file `scripts/slurm/probe_sweep.jobs` contains one whitespace-
# separated record per line: `<model_config> <dataset_name> <bands>`.
# Generate it with `python scripts/slurm/build_probe_jobs.py`.
#
# Each array task probes one (model x dataset x bands) combination on one A100,
# extracts features once, runs KNN + linear-probe + intrinsic-dim, and appends
# rows to results/all_results.csv (file-locked, multi-job-safe).

set -euo pipefail

mkdir -p logs results

JOBS_FILE=${JOBS_FILE:-scripts/slurm/probe_sweep.jobs}
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$JOBS_FILE")
read -r MODEL DATASET BANDS NORM <<< "$LINE"
NORM=${NORM:-bandspec_zscore}

echo "[$(date)] task=$SLURM_ARRAY_TASK_ID model=$MODEL dataset=$DATASET bands=$BANDS norm=$NORM"

cd "$SLURM_SUBMIT_DIR"

VENV=${TGB_VENV:-$SLURM_SUBMIT_DIR/.venv-3.12}
source "$VENV/bin/activate"

# Auto-trust torch.hub repos so AnySat etc. don't block on an interactive
# y/N prompt.  PyTorch reads `<TORCH_HOME>/hub/trusted_list`; honour the
# TORCH_HOME env var if set (cluster-wide caches typically live in /projects).
TORCH_HUB_DIR="${TORCH_HOME:-$HOME/.cache/torch}/hub"
mkdir -p "$TORCH_HUB_DIR"
TRUSTED_LIST="$TORCH_HUB_DIR/trusted_list"
for repo in gastruc_anysat facebookresearch_dinov2; do
  grep -qxF "$repo" "$TRUSTED_LIST" 2>/dev/null || echo "$repo" >> "$TRUSTED_LIST"
done
# Geobreeze CROMA + similar load weights from this dir.
export MODEL_WEIGHTS_DIR=${MODEL_WEIGHTS_DIR:-/projects/bgtj/isaaccorley/cache/geobreeze_weights}
mkdir -p "$MODEL_WEIGHTS_DIR"

# Shared HuggingFace cache so big terratorch/torchgeo backbones don't
# re-download per task.  HF_HOME is the umbrella var (covers hub, datasets,
# transformers) and overrides the per-job $HOME/.cache/huggingface default.
export HF_HOME=${HF_HOME:-/projects/bgtj/isaaccorley/cache/hf}
mkdir -p "$HF_HOME"

torchgeo-bench run \
  model="${MODEL}" \
  dataset.names="[${DATASET}]" \
  dataset.bands="${BANDS}" \
  dataset.batch_size="${TGB_BATCH_SIZE:-256}" \
  dataset.num_workers="${TGB_NUM_WORKERS:-4}" \
  dataset.normalization="${NORM}" \
  resume=true \
  output=results/all_results.csv \
  eval.intrinsic_dim.enabled=true \
  eval.profile.enabled=true
