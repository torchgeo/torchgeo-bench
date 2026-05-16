#!/bin/bash
# CLS-vs-mean pool ablation on eurosat-spatial.
#
# For each (backbone, bands) combo we already have a mean-pool row from
# the main sweep; this submits the matching cls-pool runs.  resume=true
# in probe_sweep.sh keeps each task to feature extraction + KNN/linear/
# intrinsic_dim/profile for the new (model_config, name) — no duplicate
# work against the mean rows.
#
# Usage:
#   bash scripts/slurm/eurosat_spatial_pool_ablation.sh             # gpu_a100
#   DRY_RUN=1 bash scripts/slurm/eurosat_spatial_pool_ablation.sh   # print and exit

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

JOBS_FILE="scripts/slurm/eurosat_spatial_pool_ablation.jobs"

cat > "$JOBS_FILE" <<'EOF'
terratorch/prithvi_eo_v1_100_cls eurosat-spatial rgb bandspec_zscore
terratorch/prithvi_eo_v1_100_cls eurosat-spatial all bandspec_zscore
terratorch/prithvi_eo_v2_100_tl_cls eurosat-spatial rgb bandspec_zscore
terratorch/prithvi_eo_v2_100_tl_cls eurosat-spatial all bandspec_zscore
terratorch/prithvi_eo_v2_300_cls eurosat-spatial rgb bandspec_zscore
terratorch/prithvi_eo_v2_300_cls eurosat-spatial all bandspec_zscore
terratorch/prithvi_eo_v2_300_tl_cls eurosat-spatial rgb bandspec_zscore
terratorch/prithvi_eo_v2_300_tl_cls eurosat-spatial all bandspec_zscore
terratorch/prithvi_eo_v2_600_cls eurosat-spatial rgb bandspec_zscore
terratorch/prithvi_eo_v2_600_cls eurosat-spatial all bandspec_zscore
terratorch/clay_v1_5_cls eurosat-spatial rgb bandspec_zscore
terratorch/clay_v1_5_cls eurosat-spatial all bandspec_zscore
terratorch/terramind_v1_base_cls eurosat-spatial rgb bandspec_zscore
terratorch/terramind_v1_base_cls eurosat-spatial all bandspec_zscore
terratorch/terramind_v1_large_cls eurosat-spatial rgb bandspec_zscore
terratorch/terramind_v1_large_cls eurosat-spatial all bandspec_zscore
torchgeo/scalemae_large_fmow_cls eurosat-spatial rgb bandspec_zscore
EOF

NUM_JOBS=$(wc -l < "$JOBS_FILE")
ARRAY_MAX=$((NUM_JOBS - 1))
echo "[1/2] $NUM_JOBS cls-pool jobs -> --array=0-$ARRAY_MAX"

SBATCH_CMD=(
  sbatch
  --array="0-${ARRAY_MAX}"
  --export=ALL,JOBS_FILE="$JOBS_FILE"
  scripts/slurm/probe_sweep.sh
)

echo "[2/2] ${SBATCH_CMD[*]}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1 set; not submitting."
  exit 0
fi
"${SBATCH_CMD[@]}"
