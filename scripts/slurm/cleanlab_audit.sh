#!/bin/bash
#SBATCH --job-name=tgb-cleanlab
#SBATCH --partition=gpu
#SBATCH --account=bgtj-tgirails
#SBATCH --array=0-0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=120G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#
# Cleanlab audit: extract train+test linear-probe probabilities for the top-1
# model on each GeoBench classification dataset, one array task per dataset.
#
# Submit as:
#
#   N=$(wc -l < scripts/slurm/cleanlab_audit.jobs)
#   # H100 partition:
#   sbatch --array=0-$((N-1))%9 scripts/slurm/cleanlab_audit.sh
#   # or A100 partition:
#   sbatch --partition=gpu_a100 --array=0-$((N-1))%9 scripts/slurm/cleanlab_audit.sh
#
# Each line in scripts/slurm/cleanlab_audit.jobs is one dataset name. The
# script picks the top-1 (method=linear) row per dataset from
# results/all_results.csv.

set -euo pipefail

mkdir -p logs results/cleanlab/probs

JOBS_FILE=${JOBS_FILE:-scripts/slurm/cleanlab_audit.jobs}
DATASET=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$JOBS_FILE")

echo "[$(date)] task=$SLURM_ARRAY_TASK_ID dataset=$DATASET"

cd "$SLURM_SUBMIT_DIR"

VENV=${TGB_VENV:-$SLURM_SUBMIT_DIR/.venv}
source "$VENV/bin/activate"

# Same torch.hub / weights-dir setup as probe_sweep_h100.sh.
TORCH_HUB_DIR="${TORCH_HOME:-$HOME/.cache/torch}/hub"
mkdir -p "$TORCH_HUB_DIR"
TRUSTED_LIST="$TORCH_HUB_DIR/trusted_list"
for repo in gastruc_anysat facebookresearch_dinov2; do
  grep -qxF "$repo" "$TRUSTED_LIST" 2>/dev/null || echo "$repo" >> "$TRUSTED_LIST"
done
export MODEL_WEIGHTS_DIR=${MODEL_WEIGHTS_DIR:-$HOME/.cache/geobreeze_weights}
mkdir -p "$MODEL_WEIGHTS_DIR"

python scripts/cleanlab_extract_probs.py \
  --dataset "$DATASET" \
  --results results/all_results.csv \
  --out results/cleanlab/probs \
  --device cuda:0 \
  --batch-size "${TGB_BATCH_SIZE:-64}" \
  --num-workers "${TGB_NUM_WORKERS:-8}" \
  --verbose
