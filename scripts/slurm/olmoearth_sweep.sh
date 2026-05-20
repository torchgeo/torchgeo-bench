#!/bin/bash
#SBATCH --job-name=oe-sweep
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
# OlmoEarth-specific GeoBench sweep.  Each line of the jobs file:
#
#   <model> <dataset> <bands> <image_size>
#
# where <bands> is "rgb" or a comma-separated list of band names, and
# <image_size> is "null" (native) or an integer.

set -euo pipefail
mkdir -p logs results

JOBS_FILE=${JOBS_FILE:-scripts/slurm/olmoearth_sweep.jobs}
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$JOBS_FILE")
read -r MODEL DATASET BANDS IMAGE_SIZE <<< "$LINE"

echo "[$(date)] task=$SLURM_ARRAY_TASK_ID model=$MODEL dataset=$DATASET bands=$BANDS image_size=$IMAGE_SIZE"

cd "$SLURM_SUBMIT_DIR"
VENV=${TGB_VENV:-$SLURM_SUBMIT_DIR/.venv}
source "$VENV/bin/activate"

export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
mkdir -p "$HF_HOME"

# Hydra needs square-bracket form for list overrides; "rgb" passes as a
# scalar.  Quote the whole list value to keep the shell from word-splitting.
if [[ "$BANDS" == "rgb" ]]; then
  BANDS_OVERRIDE="dataset.bands=rgb"
else
  BANDS_OVERRIDE="dataset.bands=[${BANDS}]"
fi

# Auto-pick batch size by model size so we don't OOM on the 668M-param
# large variant.  Override either side with TGB_BATCH_SIZE if needed.
if [[ -n "${TGB_BATCH_SIZE:-}" ]]; then
  BATCH_SIZE="${TGB_BATCH_SIZE}"
elif [[ "$MODEL" == *"_large" ]]; then
  BATCH_SIZE=32
elif [[ "$MODEL" == *"_base" ]]; then
  BATCH_SIZE=64
else
  BATCH_SIZE=128
fi
echo "[$(date)] batch_size=$BATCH_SIZE (auto-picked from model size)"

torchgeo-bench run \
  model="${MODEL}" \
  "dataset.names=[${DATASET}]" \
  "${BANDS_OVERRIDE}" \
  dataset.image_size="${IMAGE_SIZE}" \
  dataset.batch_size="${BATCH_SIZE}" \
  dataset.num_workers="${TGB_NUM_WORKERS:-4}" \
  resume=true \
  output=results/all_results.csv \
  eval.knn_device=cpu \
  eval.intrinsic_dim.enabled=true \
  eval.profile.enabled=true \
  eval.profile.cpu_throughput.enabled=true
