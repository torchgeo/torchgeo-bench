#!/usr/bin/env bash
set -euo pipefail

MODELS=(
  timm/resnet50
  timm/resnet18
  timm/mobilenetv3_large_100
  timm/convnext_large_dinov3
  torchgeo/dofa_base
  torchgeo/panopticon
  olmoearth_base
  terratorch/clay_v1_5
  terratorch/prithvi_eo_v2_300_tl
  terratorch/terramind_v1_base
  timm/vit/swin_tiny_patch4_window7_224
  timm/vit/vit_large_patch16_dinov3sat
  timm/vit/vit_base_patch16_224
)

DATASETS="[m-eurosat,m-forestnet,m-so2sat,m-pv4ger,m-brick-kiln,forestnet,so2sat]"

for MODEL in "${MODELS[@]}"; do
  echo "=== CKA: ${MODEL} ==="
  python -m torchgeo_bench.cka.pipeline \
    model="${MODEL}" \
    dataset.names="${DATASETS}" \
    resume=true
done
