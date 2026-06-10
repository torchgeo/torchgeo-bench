#!/bin/bash
#SBATCH --job-name=missing_sweeps
#SBATCH --partition=gpu_a100
#SBATCH --account=bgtj-tgirails
#SBATCH --array=0-28%6
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/missing_sweeps-%A_%a.out
#SBATCH --error=logs/missing_sweeps-%A_%a.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

MANIFEST="slurm/missing_sweeps.manifest"
OUTPUT="results/all_results.csv"

LINE=$(grep -vE '^\s*(#|$)' "$MANIFEST" | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
if [ -z "$LINE" ]; then
    echo "No manifest line for array index $SLURM_ARRAY_TASK_ID" >&2
    exit 1
fi

MODEL=$(echo "$LINE" | cut -f1)
BANDS=$(echo "$LINE" | cut -f2)
DATASET=$(echo "$LINE" | cut -f3)

echo "[$(date)] task=$SLURM_ARRAY_TASK_ID model=$MODEL bands='$BANDS' dataset=$DATASET"

EXTRA_OVERRIDES=()
if [ -n "$BANDS" ]; then
    EXTRA_OVERRIDES+=("dataset.bands=$BANDS")
fi

.venv/bin/torchgeo-bench run \
    "model=$MODEL" \
    "dataset.names=[$DATASET]" \
    "${EXTRA_OVERRIDES[@]}" \
    "device=cuda:0" \
    "output=$OUTPUT" \
    "resume=true"
