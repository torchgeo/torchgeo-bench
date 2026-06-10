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

# faiss-cpu is installed by default, and the [cuda] extra adds faiss-cuda.
# Both wheels share the faiss/ Python namespace.  When both are present,
# _loader.py prefers faiss-cpu's CPU-only AVX2 .abi3.so and hides
# StandardGpuResources.  Fix: remove faiss-cpu, then reinstall faiss-cuda so
# its Python files (shared with faiss-cpu) are restored.
FAISS_CUDA_WHEEL="${SLURM_SUBMIT_DIR}/../faiss-cuda/wheelhouse/faiss_cuda-1.14.1.post3-cp313-cp313-manylinux_2_34_x86_64.whl"
FAISS_CUDA_WHEEL_28="/tmp/faiss_cuda-1.14.1.post3-cp313-cp313-manylinux_2_28_x86_64.whl"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null \
    && [[ -f "$FAISS_CUDA_WHEEL" ]]; then
  UV=${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}
  cp "$FAISS_CUDA_WHEEL" "$FAISS_CUDA_WHEEL_28"
  "$UV" pip uninstall faiss-cpu --python "$VENV/bin/python3" 2>/dev/null || true
  "$UV" pip install "$FAISS_CUDA_WHEEL_28" --no-deps --python "$VENV/bin/python3" 2>/dev/null
  rm -f "$FAISS_CUDA_WHEEL_28"
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
