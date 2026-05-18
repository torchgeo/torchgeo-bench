#!/bin/bash
#SBATCH --job-name=tgb-slow
#SBATCH --partition=gpu_a100
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slow_%j.out
#SBATCH --error=logs/slow_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

VENV=${TGB_VENV:-$SLURM_SUBMIT_DIR/.venv}
# shellcheck disable=SC1091
source "$VENV/bin/activate"

# Install GPU extras if CUDA is available (needed for faissknn KNN GPU tests).
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  pip install -q -e ".[cuda]" --no-deps 2>/dev/null || true
fi

# Same torch.hub / weights-dir setup as the probe sweep.
TORCH_HUB_DIR="${TORCH_HOME:-$HOME/.cache/torch}/hub"
mkdir -p "$TORCH_HUB_DIR"
TRUSTED_LIST="$TORCH_HUB_DIR/trusted_list"
for repo in gastruc_anysat facebookresearch_dinov2; do
  grep -qxF "$repo" "$TRUSTED_LIST" 2>/dev/null || echo "$repo" >> "$TRUSTED_LIST"
done
export MODEL_WEIGHTS_DIR=${MODEL_WEIGHTS_DIR:-$HOME/.cache/geobreeze_weights}

echo "[$(date)] python: $(python --version)"
nvidia-smi || true

# Run the full slow marker suite, junit + verbose stdout.
pytest -m slow tests/ -v --tb=short \
  --junitxml=logs/slow_${SLURM_JOB_ID}.xml \
  -o addopts=
