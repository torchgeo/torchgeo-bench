#!/bin/bash
# Submit the GeoFM sweep on the eurosat-spatial dataset.
#
# Generates the job manifest (idempotent — re-runs the builder each time),
# computes the array size, and sbatches scripts/slurm/probe_sweep.sh with
# the right JOBS_FILE.  probe_sweep.sh already enables intrinsic_dim and
# profile metrics, so results land in results/all_results.csv with rows
# for method in {knn5, linear, intrinsic_dim, profile}.
#
# Usage:
#   bash scripts/slurm/eurosat_spatial_geofm.sh            # submit array
#   DRY_RUN=1 bash scripts/slurm/eurosat_spatial_geofm.sh  # print sbatch cmd, don't submit

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

JOBS_FILE="scripts/slurm/eurosat_spatial_geofm.jobs"

echo "[1/3] Regenerating job manifest -> $JOBS_FILE"
uv run --no-sync python scripts/slurm/build_probe_jobs.py \
  --datasets eurosat-spatial \
  --bands rgb,all \
  --out "$JOBS_FILE"

NUM_JOBS=$(wc -l < "$JOBS_FILE")
ARRAY_MAX=$((NUM_JOBS - 1))
echo "[2/3] $NUM_JOBS jobs -> --array=0-$ARRAY_MAX"

SBATCH_CMD=(
  sbatch
  --array="0-${ARRAY_MAX}"
  --export=ALL,JOBS_FILE="$JOBS_FILE"
  scripts/slurm/probe_sweep.sh
)

echo "[3/3] ${SBATCH_CMD[*]}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1 set; not submitting."
  exit 0
fi
"${SBATCH_CMD[@]}"
