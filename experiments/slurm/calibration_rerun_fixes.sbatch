#!/bin/bash
#SBATCH --job-name=calsweep_fix
#SBATCH --partition=gpu_a100
#SBATCH --account=bgtj-tgirails
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/calsweep_fix4-%A_%a.out
#SBATCH --error=logs/calsweep_fix4-%A_%a.err

# Fix: OlmoEarth needs bands=all + image_size=null (S2 12-band, no resize).
# swir_cirrus (B10) is now gracefully skipped in the OlmoEarth wrapper.
# Panopticon jobs already completed successfully; not included here.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

OUTPUT="results/calibration_eurosat.csv"

case $SLURM_ARRAY_TASK_ID in
  0) MODEL=olmoearth_large;     DATASET=m-eurosat ;;
  1) MODEL=olmoearth_large;     DATASET=eurosat-spatial ;;
  2) MODEL=olmoearth_base;      DATASET=m-eurosat ;;
  3) MODEL=olmoearth_base;      DATASET=eurosat-spatial ;;
  4) MODEL=olmoearth_v1_1_base; DATASET=m-eurosat ;;
  5) MODEL=olmoearth_v1_1_base; DATASET=eurosat-spatial ;;
  6) MODEL=olmoearth_tiny;      DATASET=m-eurosat ;;
  7) MODEL=olmoearth_tiny;      DATASET=eurosat-spatial ;;
  *) echo "Unknown task index $SLURM_ARRAY_TASK_ID" >&2; exit 1 ;;
esac

echo "[$(date)] fix4 task=$SLURM_ARRAY_TASK_ID model=$MODEL dataset=$DATASET bands=all image_size=null"

.venv/bin/torchgeo-bench run \
    "model=$MODEL" \
    "dataset.names=[$DATASET]" \
    "dataset.bands=all" \
    "dataset.image_size=null" \
    "device=cuda:0" \
    "output=$OUTPUT" \
    "resume=false"
