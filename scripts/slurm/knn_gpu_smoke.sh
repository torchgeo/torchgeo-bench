#!/bin/bash
#SBATCH --job-name=tgb-knn-gpu
#SBATCH --partition=gpu_a100
#SBATCH --account=bgtj-tgirails
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#
# Smoke test for the GPU path of torchgeo_bench.knn.KNNClassifier.
# Installs the project's `cuda` extra (faissknn → faiss-cuda-cu128) into a
# scratch venv on the compute node (where glibc may be newer than the login
# node), then runs scripts/test_knn_gpu_smoke.py and compares CPU vs GPU
# predictions on synthetic data.

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
echo "[$(date)] glibc on compute node:"
ldd --version || true

# Use the prebuilt 3.12 venv with `faissknn` (and `faiss-gpu-cu12`)
# preinstalled. faiss-cpu and faiss-gpu-cu12 both install into the same
# `faiss/` namespace, so for the GPU smoke we keep faiss-gpu-cu12 only.
VENV=${TGB_VENV:-$SLURM_SUBMIT_DIR/.venv}
# shellcheck disable=SC1091
source "$VENV/bin/activate"
nvidia-smi | head -3 || true
python -c "import faiss; print('faiss has GPU:', hasattr(faiss, 'GpuIndexFlatL2'))"
python scripts/test_knn_gpu_smoke.py
