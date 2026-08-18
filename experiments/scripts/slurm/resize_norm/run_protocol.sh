#!/bin/bash
#SBATCH --job-name=tgb-rn
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/resize_norm/%A_%a.out
#SBATCH --error=logs/resize_norm/%A_%a.err
#
# One array task = one (model, protocol[, seed]) line of a resize_norm .jobs
# file; the task loops over that model's datasets. Each (model, protocol)
# pair gets its own output CSV so parallel tasks never contend and reruns
# resume cleanly. device/num_workers are pinned because both feed the resume
# config_hash.
#
#   JOBS=experiments/scripts/slurm/resize_norm/primary.jobs
#   sbatch --partition=gpu_a100 --array=0-187%18 \
#          --export=ALL,JOBS_FILE=$JOBS experiments/scripts/slurm/resize_norm/run_protocol.sh

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs/resize_norm results/sweeps/resize_norm
source "${TGB_VENV:-$SLURM_SUBMIT_DIR/.venv}/bin/activate"

JOBS_FILE=${JOBS_FILE:?set JOBS_FILE to a resize_norm .jobs file}
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$JOBS_FILE")
IFS=$'\t' read -r NAME CFG NORM IMGSIZE SEED DATASETS EXTRA <<< "$LINE"

BANDS_MAP=experiments/scripts/slurm/resize_norm/bands_map.tsv
SIZE_TAG=$([ "$IMGSIZE" = "null" ] && echo native || echo "$IMGSIZE")
OUT_DIR="results/sweeps/${SWEEP_NAME:-resize_norm}"
mkdir -p "$OUT_DIR"
OUTPUT="${OUT_DIR}/${NAME}__${NORM}__${SIZE_TAG}.csv"

EXTRA_ARGS=()
[ "$EXTRA" != "-" ] && read -r -a EXTRA_ARGS <<< "$EXTRA"

FAILED=0
IFS=',' read -r -a DS_LIST <<< "$DATASETS"
for DATASET in "${DS_LIST[@]}"; do
  BANDS=$(awk -F'\t' -v m="$NAME" -v d="$DATASET" '$1==m && $2==d {print $3}' "$BANDS_MAP")
  if [ -z "$BANDS" ]; then
    echo "[$(date)] SKIP $NAME/$DATASET: no bands_map entry"
    continue
  fi
  case "$BANDS" in
    rgb|all) BANDS_ARG="dataset.bands=${BANDS}" ;;
    *)       BANDS_ARG="dataset.bands=[${BANDS}]" ;;
  esac
  echo "[$(date)] task=$SLURM_ARRAY_TASK_ID model=$NAME ds=$DATASET norm=$NORM size=$IMGSIZE seed=$SEED bands=$BANDS extra='${EXTRA_ARGS[*]:-}'"
  if ! torchgeo-bench run \
      model="${CFG}" \
      dataset.names="[${DATASET}]" \
      "${BANDS_ARG}" \
      dataset.normalization="${NORM}" \
      dataset.image_size="${IMGSIZE}" \
      dataset.interpolation=bilinear \
      dataset.num_workers=8 \
      device=cuda:0 \
      seed="${SEED}" \
      output="${OUTPUT}" \
      resume=true \
      "${EXTRA_ARGS[@]}"; then
    echo "[$(date)] FAILED $NAME/$DATASET norm=$NORM size=$IMGSIZE seed=$SEED"
    FAILED=1
  fi
done

exit $FAILED
