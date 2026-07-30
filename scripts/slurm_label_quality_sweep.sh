#!/bin/bash
#SBATCH --account=hai_1282
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --partition=booster
#SBATCH --output=slurm_out/lq.%j.out
#SBATCH --error=slurm_err/lq.%j.err
#
# Segmentation label-quality AUDIT sweep. The job gets a full 4-GPU node and
# runs 4 (model x dataset) pairs at a time — one per GPU — working through the
# pair list until it is done.
#
# Pick the pairs one of three ways (first one set wins):
#
#   1. PAIRS_LIST — inline list, the usual case. Semicolons or newlines
#      separate pairs; each pair is "model dataset":
#        PAIRS_LIST="timm/resnet50 caffe; timm/resnet50 fotw" \
#          sbatch scripts/slurm_label_quality_sweep.sh
#
#   2. MODELS x DATASETS — full cross product of two lists:
#        MODELS="timm/resnet50 terratorch/terramind_v1_base" \
#        DATASETS="caffe fotw spacenet2" \
#          sbatch scripts/slurm_label_quality_sweep.sh
#
#   3. JOBS_FILE — a file of "model dataset" lines (default when neither of the
#      above is set), generated with:
#        python scripts/slurm/build_label_quality_jobs.py > scripts/slurm/label_quality.jobs
#
# Each pair runs the full production budget from conf/label_quality.yaml
# (n_members=5, k=5, max_steps=2000): the augmented ensemble OOF substrate plus
# both cleanlab + AER scorers. RGB only.
#
# FIX-VERIFICATION GATE (run BEFORE the full sweep — see the plan, FM-1/2/5):
# Submit 2 models x 2 datasets incl. flair2, let it write checkpoints, then
# re-submit the SAME command with RESUME=true. Pass = the resume loads each
# model's own checkpoints (never cross-loads), both models appear in the CSV
# with correct `model` values, and flair2 completes (or its traceback is
# captured with HYDRA_FULL_ERROR=1, which this script exports).
#   MODELS="timm/resnet50 timm/vit_base_patch16_224" DATASETS="spacenet2 flair2" \
#     sbatch scripts/slurm_label_quality_sweep.sh
#
# Validate the plumbing first with:
#   sbatch scripts/slurm_label_quality_smoke.sh
#
# Submit — one node, 4 GPUs, working through every pair given:
#   sbatch scripts/slurm_label_quality_sweep.sh
#
# SHARDING (FM-4): the runner statically pins one pair per GPU and only advances
# a GPU when its pair finishes, so a fast GPU idles while a 12 h pair runs. To
# keep all GPUs busy, submit disjoint jobs sharded by dataset speed — each is an
# independent 4-GPU node. Per model, one slow-shard job fills a node:
#   # Slow shard (~10-12 h/pair): 4 pairs fill a node; size --time to the
#   # slowest (spacenet2 ~12 h) + margin.
#   MODELS="timm/resnet50" DATASETS="caffe fotw spacenet2 spacenet7" \
#     sbatch --time=18:00:00 scripts/slurm_label_quality_sweep.sh
#   # Fast shard (runtime TBD until FM-5/flair2 is characterized): keep --time
#   # generous until real timings exist.
#   MODELS="timm/resnet50" DATASETS="cloudsen12 flair2" \
#     sbatch scripts/slurm_label_quality_sweep.sh
# More backbones = more shards (one slow-shard job per model), submitted in
# parallel.
#
# Resume is on by default: reruns reuse fold checkpoints and skip completed
# (dataset, method) rows, so re-submitting a timed-out shard picks up where it
# left off — safe now that checkpoints are model-keyed (FM-1). A worker keeps
# going if one pair fails; the job exits non-zero at the end if any did.

set -euo pipefail

module load Stages/2025 GCCcore/.13.3.0 Python/3.12.3

VENV_ACTIVATE=${VENV_ACTIVATE:-/p/project1/hai_uqmethodbox/nils/torchgeo-bench/sc_venv_template/activate.sh}
if [[ ! -f "$VENV_ACTIVATE" ]]; then
  echo "Venv activate script not found: $VENV_ACTIVATE" >&2
  exit 1
fi
source "$VENV_ACTIVATE"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  echo "VIRTUAL_ENV is not set after activation." >&2
  exit 1
fi
PYTHON_BIN=${PYTHON_BIN:-python}
VENV_PYTHON="$VIRTUAL_ENV/bin/python"
if [[ -x "$VENV_PYTHON" ]]; then
  PYTHON_BIN="$VENV_PYTHON"
else
  PYTHON_BIN_PATH=$(command -v "$PYTHON_BIN" || true)
  [[ -z "$PYTHON_BIN_PATH" ]] && { echo "Python not found." >&2; exit 1; }
  PYTHON_BIN="$PYTHON_BIN_PATH"
fi
echo "Using Python: $PYTHON_BIN"

cd /p/project1/hai_uqmethodbox/nils/torchgeo-bench
export PYTHONPATH="/p/project1/hai_uqmethodbox/nils/torchgeo-bench/src:${PYTHONPATH:-}"

CACHE_ROOT=${CACHE_ROOT:-/p/project1/hai_uqmethodbox/nils/torchgeo-bench/.cache}
mkdir -p "$CACHE_ROOT" "$CACHE_ROOT"/hf "$CACHE_ROOT"/torch "$CACHE_ROOT"/timm "$CACHE_ROOT"/xdg
export HF_HOME="$CACHE_ROOT/hf"
export HUGGINGFACE_HUB_CACHE="$CACHE_ROOT/hf"
export TORCH_HOME="$CACHE_ROOT/torch"
export TIMM_CACHE_DIR="$CACHE_ROOT/timm"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export HYDRA_FULL_ERROR=1

mkdir -p results slurm_out slurm_err
mkdir -p "$(dirname "${OUTPUT:-results/label_quality_v2/label_quality_results.csv}")"

# Where the pairs come from: inline PAIRS, a MODELS x DATASETS cross product,
# or the jobs file. Whichever is set first wins; PAIRS_SOURCE is only for logs.
PAIRS=()
if [[ -n "${PAIRS_LIST:-}" ]]; then
  PAIRS_SOURCE="PAIRS_LIST"
  # Semicolons and newlines both separate pairs, so a one-line PAIRS_LIST and a
  # heredoc-style multi-line one behave the same. Collapse surrounding blanks so
  # "a b; c d" does not yield a pair with a leading space.
  mapfile -t PAIRS < <(printf '%s\n' "${PAIRS_LIST//;/$'\n'}" \
    | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//; s/[[:space:]]+/ /g' \
    | grep -vE '^(#|$)')
elif [[ -n "${MODELS:-}" && -n "${DATASETS:-}" ]]; then
  PAIRS_SOURCE="MODELS x DATASETS"
  for model in $MODELS; do
    for dataset in $DATASETS; do
      PAIRS+=("$model $dataset")
    done
  done
else
  if [[ -n "${MODELS:-}" || -n "${DATASETS:-}" ]]; then
    echo "ERROR: MODELS and DATASETS must be set together (got only one)." >&2
    exit 1
  fi
  JOBS_FILE=${JOBS_FILE:-scripts/slurm/label_quality.jobs}
  PAIRS_SOURCE="$JOBS_FILE"
  if [[ ! -f "$JOBS_FILE" ]]; then
    echo "No pairs given and jobs file not found: $JOBS_FILE" >&2
    echo "Set PAIRS_LIST=\"model dataset; model dataset\", or MODELS=... DATASETS=...," >&2
    echo "or generate the file: python scripts/slurm/build_label_quality_jobs.py > $JOBS_FILE" >&2
    exit 1
  fi
  # Skip blank lines and '#' comments so a hand-edited jobs file still works.
  mapfile -t PAIRS < <(grep -vE '^\s*(#|$)' "$JOBS_FILE")
fi

TOTAL=${#PAIRS[@]}
if [[ "$TOTAL" -eq 0 ]]; then
  echo "No pairs found in $PAIRS_SOURCE" >&2
  exit 1
fi

RESUME=${RESUME:-true}
# Fresh v2 output path (FM-6): the old results/label_quality_results.csv has no
# model column and its checkpoints are not model-keyed, so it is untrusted and
# ambiguous — do NOT resume from it. This path regenerates cleanly with the
# model-keyed checkpoints + `model` column + npz artifacts.
OUTPUT=${OUTPUT:-results/label_quality_v2/label_quality_results.csv}
# Segmentation label-quality starts RGB-only, for every backbone — this is the
# path the smoke test validates. Multispectral is a separate, later sweep.
bands_arg="dataset.bands=${BANDS:-rgb}"

# Extra Hydra overrides appended verbatim to every pair, e.g. a longer training
# budget: EXTRA_ARGS="label_quality.max_steps=4000". MAX_STEPS is a convenience
# shortcut for the common case; anything in EXTRA_ARGS is passed as-is.
EXTRA_ARGS=${EXTRA_ARGS:-}
if [[ -n "${MAX_STEPS:-}" ]]; then
  EXTRA_ARGS="label_quality.max_steps=${MAX_STEPS} ${EXTRA_ARGS}"
fi

GPUS_PER_JOB=${GPUS_PER_JOB:-4}

echo "[$(date)] $TOTAL pairs from $PAIRS_SOURCE on $GPUS_PER_JOB GPUs (bands=${BANDS:-rgb}, resume=$RESUME):"
[[ -n "$EXTRA_ARGS" ]] && echo "    extra overrides: $EXTRA_ARGS"
echo "    output: $OUTPUT"
printf '    %s\n' "${PAIRS[@]}"

# Per-worker logs: 4 pairs run at once, so their output would otherwise be
# interleaved into an unreadable mess in the job log.
WORKER_LOG_DIR=${WORKER_LOG_DIR:-slurm_out/lq_${SLURM_JOB_ID:-$$}}
mkdir -p "$WORKER_LOG_DIR"

# One worker per GPU. Worker g takes its own slice of the pair list and runs
# them one after another, pinned to GPU g via CUDA_VISIBLE_DEVICES.
run_worker() {
  local gpu="$1"
  local -a my=()
  local i
  for (( i = gpu; i < ${#PAIRS[@]}; i += GPUS_PER_JOB )); do
    my+=("${PAIRS[$i]}")
  done
  [[ ${#my[@]} -eq 0 ]] && return 0

  local rc=0 pair MODEL DATASET log
  for pair in "${my[@]}"; do
    read -r MODEL DATASET <<< "$pair"
    [[ -z "$MODEL" || -z "$DATASET" ]] && continue
    log="$WORKER_LOG_DIR/gpu${gpu}_${MODEL//\//-}_${DATASET}.log"

    echo "[$(date)] gpu=$gpu START $MODEL $DATASET -> $log"
    # Keep going if one pair fails so a bad combo can't waste the rest of the
    # worker's walltime; the non-zero rc is reported at the end.
    # EXTRA_ARGS is intentionally unquoted so its space-separated Hydra overrides
    # word-split into individual args (e.g. label_quality.max_steps=4000).
    if CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" -m torchgeo_bench.cli run \
        mode=label_quality \
        "model=${MODEL}" \
        "dataset.names=[${DATASET}]" \
        "${bands_arg}" \
        "label_quality.output=${OUTPUT}" \
        ${EXTRA_ARGS} \
        "resume=${RESUME}" > "$log" 2>&1; then
      echo "[$(date)] gpu=$gpu OK    $MODEL $DATASET"
    else
      echo "[$(date)] gpu=$gpu FAIL  $MODEL $DATASET (see $log)"
      rc=1
    fi
  done
  return $rc
}

pids=()
for (( g = 0; g < GPUS_PER_JOB; g++ )); do
  run_worker "$g" &
  pids+=("$!")
done

# Collect every worker before deciding the job's exit status.
STATUS=0
for pid in "${pids[@]}"; do
  wait "$pid" || STATUS=1
done

echo ""
if [[ $STATUS -ne 0 ]]; then
  echo "[$(date)] finished with failures — grep FAIL above; logs in $WORKER_LOG_DIR" >&2
  exit 1
fi
echo "[$(date)] done — all $TOTAL pairs succeeded."
