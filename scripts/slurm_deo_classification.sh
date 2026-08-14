#!/bin/bash
#SBATCH --account=hai_1282
#SBATCH --nodes=1
#SBATCH --output=slurm_out/deo_classification.%j.out
#SBATCH --error=slurm_err/deo_classification.%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --partition=develbooster
#SBATCH --gres=gpu:4

# Submit two independent, non-array jobs:
#   MODE=rgb sbatch scripts/slurm_deo_classification.sh
#   MODE=s2  sbatch scripts/slurm_deo_classification.sh
#
# Each job distributes its datasets over the four allocated GPUs and writes a
# separate local CSV under results/.

set -euo pipefail

module load Stages/2025 GCCcore/.13.3.0 Python/3.12.3

PROJECT_DIR=/p/project1/hai_uqmethodbox/nils/torchgeo-bench
VENV_ACTIVATE=${VENV_ACTIVATE:-/p/project1/hai_uqmethodbox/nils/torchgeo-bench/sc_venv_template/activate.sh}
if [[ ! -f "$VENV_ACTIVATE" ]]; then
  echo "Venv activate script not found: $VENV_ACTIVATE" >&2
  exit 1
fi
source "$VENV_ACTIVATE"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "VIRTUAL_ENV is not set after activation. Check the activate script." >&2
  exit 1
fi
PYTHON_BIN=${PYTHON_BIN:-python}
VENV_PYTHON="$VIRTUAL_ENV/bin/python"
if [[ -x "$VENV_PYTHON" ]]; then
  PYTHON_BIN="$VENV_PYTHON"
else
  PYTHON_BIN_PATH=$(command -v "$PYTHON_BIN" || true)
  if [[ -z "$PYTHON_BIN_PATH" ]]; then
    echo "Python not found after activation (PYTHON_BIN=${PYTHON_BIN})." >&2
    exit 1
  fi
  PYTHON_BIN="$PYTHON_BIN_PATH"
  echo "Template venv Python is not executable on this node; using activated module Python." >&2
fi

echo "Using Python: $PYTHON_BIN"
"$PYTHON_BIN" -c 'import os, torchgeo; expected = os.environ["VIRTUAL_ENV"]; assert torchgeo.__file__.startswith(expected), torchgeo.__file__'

mkdir -p "$PROJECT_DIR/slurm_out" "$PROJECT_DIR/slurm_err"

cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR/src:${PYTHONPATH:-}"
export HYDRA_FULL_ERROR=1

CACHE_ROOT=${CACHE_ROOT:-"$PROJECT_DIR/.cache"}
mkdir -p "$CACHE_ROOT/hf" "$CACHE_ROOT/torch" "$CACHE_ROOT/timm" "$CACHE_ROOT/xdg" results
export HF_HOME="$CACHE_ROOT/hf"
export HUGGINGFACE_HUB_CACHE="$CACHE_ROOT/hf"
export TORCH_HOME="$CACHE_ROOT/torch"
export TIMM_CACHE_DIR="$CACHE_ROOT/timm"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"

case "${MODE:?Set MODE=rgb or MODE=s2 when submitting this job}" in
  rgb)
    MODEL=torchgeo/deo_rgb
    BANDS=rgb
    OUTPUT=results/deo_rgb_classification.csv
    DATASETS=(
      m-eurosat m-forestnet m-so2sat m-pv4ger m-brick-kiln m-bigearthnet
      benv2 treesatai so2sat forestnet eurosat
    )
    ;;
  s2)
    MODEL=torchgeo/deo_s2
    BANDS=all
    OUTPUT=results/deo_s2_classification.csv
    DATASETS=(m-eurosat m-so2sat m-brick-kiln m-bigearthnet benv2 treesatai so2sat eurosat)
    ;;
  *)
    echo "Unsupported MODE: $MODE (expected rgb or s2)" >&2
    exit 2
    ;;
esac

echo "Running DEO $MODE classification sweep"
echo "Results: $OUTPUT"
echo "Datasets: ${DATASETS[*]}"

GPU_COUNT=${GPU_COUNT:-${SLURM_GPUS_ON_NODE:-4}}
if (( GPU_COUNT < 1 )); then
  echo "No GPU was allocated." >&2
  exit 1
fi

BATCH_SIZE=${BATCH_SIZE:-64}
NUM_WORKERS=${NUM_WORKERS:-4}
PIDS=()

for ((gpu = 0; gpu < GPU_COUNT; gpu++)); do
  (
    export CUDA_VISIBLE_DEVICES=$gpu
    status=0
    for ((index = gpu; index < ${#DATASETS[@]}; index += GPU_COUNT)); do
      dataset=${DATASETS[$index]}
      echo "[$(date --iso-8601=seconds)] mode=$MODE gpu=$gpu dataset=$dataset"
      if ! "$PYTHON_BIN" -m torchgeo_bench.cli run \
        "model=$MODEL" \
        "dataset.names=[$dataset]" \
        "dataset.bands=$BANDS" \
        "dataset.normalization=model_native" \
        "dataset.batch_size=$BATCH_SIZE" \
        "dataset.num_workers=$NUM_WORKERS" \
        "output=$OUTPUT" \
        "resume=true" \
        "device=cuda:0"; then
        echo "WARNING: DEO $MODE failed for $dataset; continuing." >&2
        status=1
      fi
    done
    exit $status
  ) &
  PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

if (( status != 0 )); then
  echo "At least one DEO $MODE dataset failed; inspect $OUTPUT and the SLURM logs." >&2
  exit $status
fi

echo "Completed DEO $MODE classification sweep: $OUTPUT"
