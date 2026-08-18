#!/bin/bash
#SBATCH --job-name=tgb-rn-smoke
#SBATCH --account=bgtj-tgirails
#SBATCH --partition=gpu_a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/resize_norm/smoke_%j.out
#SBATCH --error=logs/resize_norm/smoke_%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs/resize_norm results/sweeps/resize_norm
source .venv/bin/activate
python experiments/scripts/slurm/resize_norm/smoke_check.py \
  --jobs experiments/scripts/slurm/resize_norm/smoke.jobs \
  --out results/sweeps/resize_norm/smoke_report.csv
