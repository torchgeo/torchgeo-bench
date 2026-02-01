#!/bin/bash
# Run experiments on m-eurosat with resnet18 varying normalization, image_size, and interpolation
# Results saved to results/eurosat_example/

set -e

cd "$(dirname "$0")/.."

# Use torchgeo environment python and pip directly
PYTHON=/opt/conda/envs/torchgeo/bin/python
TORCHGEO_BENCH="$PYTHON -m src.cli"

OUTPUT_DIR="results/eurosat_example"
mkdir -p "$OUTPUT_DIR"

# Configuration arrays
NORMALIZATIONS=("mean_stdev" "min_max" "percentile_2_98" "none")
IMAGE_SIZES=("null" "224" "256" "448" "512")
INTERPOLATIONS=("bilinear" "bicubic" "nearest")

# Standard benchmark with LBFGS logistic regression
echo "=== Running standard LBFGS experiments ==="
OUTPUT_FILE="${OUTPUT_DIR}/eurosat_lbfgs.csv"

for norm in "${NORMALIZATIONS[@]}"; do
    for size in "${IMAGE_SIZES[@]}"; do
        for interp in "${INTERPOLATIONS[@]}"; do
            # Skip interpolation variations when image_size is null (no resizing)
            if [[ "$size" == "null" && "$interp" != "bilinear" ]]; then
                continue
            fi

            echo "Running: norm=$norm, size=$size, interp=$interp"
            
            if [[ "$size" == "null" ]]; then
                $TORCHGEO_BENCH run \
                    model=resnet18 \
                    dataset.names=[m-eurosat] \
                    dataset.normalization="$norm" \
                    dataset.image_size=null \
                    eval.merge_val=false \
                    device=cuda:2 \
                    output="$OUTPUT_FILE" \
                    resume=true \
                    verbose=false
            else
                $TORCHGEO_BENCH run \
                    model=resnet18 \
                    dataset.names=[m-eurosat] \
                    dataset.normalization="$norm" \
                    dataset.image_size="$size" \
                    dataset.interpolation="$interp" \
                    eval.merge_val=false \
                    device=cuda:2 \
                    output="$OUTPUT_FILE" \
                    resume=true \
                    verbose=false
            fi
        done
    done
done

echo "=== Standard experiments complete ==="
echo "Results saved to: $OUTPUT_FILE"
