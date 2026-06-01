#!/bin/bash
# Submit the GeoFM sweep on the eurosat-spatial dataset.
#
# Generates the job manifest (idempotent — re-runs the builder each time),
# computes the array size, and sbatches experiments/scripts/slurm/probe_sweep.sh with
# the right JOBS_FILE.  probe_sweep.sh already enables intrinsic_dim and
# profile metrics, so results land in results/all_results.csv with rows
# for method in {knn5, linear, intrinsic_dim, profile}.
#
# Usage:
#   bash experiments/scripts/slurm/eurosat_spatial_geofm.sh                # gpu_a100
#   bash experiments/scripts/slurm/eurosat_spatial_geofm.sh --preempt      # preempt partition
#   DRY_RUN=1 bash experiments/scripts/slurm/eurosat_spatial_geofm.sh ...  # print sbatch cmd, don't submit
#
# --preempt: submit to the preempt partition with --requeue, so jobs
# killed by higher-priority work are re-queued automatically.  Combined
# with resume=true in probe_sweep.sh, killed tasks pick up cleanly on the
# next slot; many more concurrent slots cuts wall time substantially.

set -euo pipefail

PREEMPT=0
for arg in "$@"; do
  case "$arg" in
    --preempt) PREEMPT=1 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

JOBS_FILE="experiments/scripts/slurm/eurosat_spatial_geofm.jobs"

echo "[1/3] Regenerating job manifest -> $JOBS_FILE"
uv run --no-sync python experiments/scripts/slurm/build_probe_jobs.py \
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
)
if [[ "$PREEMPT" -eq 1 ]]; then
  SBATCH_CMD+=(--partition=preempt --requeue)
fi
SBATCH_CMD+=(experiments/scripts/slurm/probe_sweep.sh)

echo "[3/3] ${SBATCH_CMD[*]}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1 set; not submitting."
  exit 0
fi
"${SBATCH_CMD[@]}"
