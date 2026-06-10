#!/bin/bash
#SBATCH --job-name=tgb-smoke
#SBATCH --partition=gpu_a100
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=logs/smoke-%j.out
#SBATCH --error=logs/smoke-%j.err
#
# Smoke-test every torchgeo-bench model wrapper: import, instantiate, run
# one tiny forward pass.  Catches typo'd kwargs / missing weights / wrong
# x_dict keys before we burn SUs on a real sweep.
#
# Submit as:
#   sbatch --account=$PROJECT experiments/scripts/slurm/smoke_imports.sh
#
# Output: results/smoke_imports.json + a summary table in the job log.

set -euo pipefail

mkdir -p logs results

cd "$SLURM_SUBMIT_DIR"

VENV=${TGB_VENV:-$SLURM_SUBMIT_DIR/.venv}
source "$VENV/bin/activate"

# Geobreeze CROMA reads weights from this dir and downloads on first use.
export MODEL_WEIGHTS_DIR=${MODEL_WEIGHTS_DIR:-$HOME/.cache/geobreeze_weights}
mkdir -p "$MODEL_WEIGHTS_DIR"

# Auto-trust torch.hub repos so AnySat etc. don't block on an interactive
# y/N prompt.  PyTorch reads `<TORCH_HOME>/hub/trusted_list` (default
# `~/.cache/torch/hub/trusted_list`), one repo per line as
# `<owner>_<name>`.  Honour TORCH_HOME if it's set in the environment.
TORCH_HUB_DIR="${TORCH_HOME:-$HOME/.cache/torch}/hub"
mkdir -p "$TORCH_HUB_DIR"
TRUSTED_LIST="$TORCH_HUB_DIR/trusted_list"
for repo in gastruc_anysat facebookresearch_dinov2; do
  grep -qxF "$repo" "$TRUSTED_LIST" 2>/dev/null || echo "$repo" >> "$TRUSTED_LIST"
done

# Default: only the new wrappers (terratorch, rslearn, geobreeze).
# Override with `sbatch --export=MODEL_GROUPS=all,...` to also smoke older configs.
# rslearn temporarily excluded — its FeatureExtractors expect a
# ModelContext-boxed batch (RasterImage + timestamps + multimodal dict);
# wrappers need rework before they can probe via the standard path.
MODEL_GROUPS=${MODEL_GROUPS:-terratorch,geobreeze}

python experiments/scripts/smoke_imports.py \
  --groups "${MODEL_GROUPS}" \
  --batch 2 \
  --device cuda:0 \
  --out "results/smoke_imports_${SLURM_JOB_ID}.json"
