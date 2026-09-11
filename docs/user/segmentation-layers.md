# Segmentation backbone layer reference

This page records how the values for `eval.segmentation.layers` (see
[Configuration](configuration.rst)) were determined for each model
family, and serves as a reference when adding new models. It complements
the [`SegmentationProbe`](../api/models.rst) and the segmentation-head
classes (`LinearHead`, `FPNHead`, `DPTHead`, …).

## Background

`SegmentationProbe` hooks into named PyTorch modules via `backbone.named_modules()`. The layer names in each model config must exactly match these module names. Layers should be listed **deepest first** (coarsest feature map first) for the FPN head.

## How Layer Names Were Discovered

Layer names were found by:
1. Instantiating each timm model with `pretrained=False, num_classes=0`
2. Running a dummy 224×224 input through the model with forward hooks on candidate layers
3. Recording the output spatial size of each layer

Discovery script:

```python
import timm, torch

model = timm.create_model("resnet50", pretrained=False, num_classes=0)
model.eval()
x = torch.zeros(1, 3, 224, 224)
shapes = {}
handles = []

for name in ["layer1", "layer2", "layer3", "layer4"]:
    module = dict(model.named_modules())[name]
    def hook(n):
        def fn(m, inp, out): shapes[n] = out.shape
        return fn
    handles.append(module.register_forward_hook(hook(name)))

with torch.no_grad():
    model(x)
for h in handles:
    h.remove()

print(shapes)
```

## Layer Names by Family

All spatial sizes are for a 224×224 input image.

### ResNet (resnet18, resnet34, resnet50, resnet101)

| Layer | Spatial size | Channels (resnet50) |
|---|---|---|
| `layer4` | 7×7 | 2048 |
| `layer3` | 14×14 | 1024 |
| `layer2` | 28×28 | 512 |
| `layer1` | 56×56 | 256 |

Config: `layers: ["layer4", "layer3", "layer2", "layer1"]`

### DenseNet (densenet121, densenet161)

| Layer | Spatial size | Channels (densenet121) |
|---|---|---|
| `features.denseblock4` | 7×7 | 1024 |
| `features.denseblock3` | 14×14 | 1024 |
| `features.denseblock2` | 28×28 | 512 |
| `features.denseblock1` | 56×56 | 256 |

Config: `layers: ["features.denseblock4", "features.denseblock3", "features.denseblock2", "features.denseblock1"]`

Note: `features.transition1/2/3` downsample between denseblocks. Using the denseblocks (before downsampling) gives richer features.

### VGG16

| Layer | Spatial size | Description |
|---|---|---|
| `features.30` | 7×7 | After pool5 |
| `features.23` | 14×14 | After pool4 |
| `features.16` | 28×28 | After pool3 |
| `features.9` | 56×56 | After pool2 |

Config: `layers: ["features.30", "features.23", "features.16", "features.9"]`

### VGG19

| Layer | Spatial size | Description |
|---|---|---|
| `features.36` | 7×7 | After pool5 |
| `features.27` | 14×14 | After pool4 |
| `features.18` | 28×28 | After pool3 |
| `features.9` | 56×56 | After pool2 |

Config: `layers: ["features.36", "features.27", "features.18", "features.9"]`

Note: VGG19 has more conv layers per stage than VGG16, so the MaxPool indices differ.

### EfficientNet (efficientnet_b0, b1, b2, b3)

| Layer | Spatial size | Channels (b0) |
|---|---|---|
| `blocks.6` | 7×7 | 320 |
| `blocks.5` | 7×7 | 192 |
| `blocks.3` | 14×14 | 80 |
| `blocks.1` | 56×56 | 24 |

Config: `layers: ["blocks.6", "blocks.5", "blocks.3", "blocks.1"]`

**Note:** `blocks.4`, `blocks.5`, and `blocks.6` all operate at 7×7 (EfficientNet uses multiple MBConv stages at the same stride). We include `blocks.5` to get 4 layers, though it shares spatial size with `blocks.6`. If 3 distinct pyramid levels are preferred, use `["blocks.6", "blocks.3", "blocks.1"]`.

### ConvNeXt (convnext_tiny, small, base, large, large_dinov3)

| Layer | Spatial size | Channels (tiny) |
|---|---|---|
| `stages.3` | 7×7 | 768 |
| `stages.2` | 14×14 | 384 |
| `stages.1` | 28×28 | 192 |
| `stages.0` | 56×56 | 96 |

Config: `layers: ["stages.3", "stages.2", "stages.1", "stages.0"]`

### MobileNetV3-Large (mobilenetv3_large_100)

| Layer | Spatial size | Channels |
|---|---|---|
| `blocks.6` | 7×7 | 960 |
| `blocks.5` | 7×7 | 160 |
| `blocks.3` | 14×14 | 80 |
| `blocks.1` | 56×56 | 24 |

Config: `layers: ["blocks.6", "blocks.5", "blocks.3", "blocks.1"]`

**Note:** `blocks.5` and `blocks.6` share 7×7 (same as EfficientNet note above).

### MobileNetV3-Small (mobilenetv3_small_100)

| Layer | Spatial size | Channels |
|---|---|---|
| `blocks.5` | 7×7 | 576 |
| `blocks.4` | 7×7 | 96 |
| `blocks.2` | 14×14 | 40 |
| `blocks.1` | 28×28 | 24 |

Config: `layers: ["blocks.5", "blocks.4", "blocks.2", "blocks.1"]`

Note: MobileNetV3-Small only has 6 blocks (0–5), unlike the Large variant which has 7 (0–6). The shallowest useful stage is 28×28 (no 56×56 stage).

### RegNet (regnetx_002, regnetx_008, regnety_002, regnety_008)

| Layer | Spatial size | Channels (regnetx_002) |
|---|---|---|
| `s4` | 7×7 | 368 |
| `s3` | 14×14 | 152 |
| `s2` | 28×28 | 56 |
| `s1` | 56×56 | 24 |

Config: `layers: ["s4", "s3", "s2", "s1"]`

## Adding a New Model

1. Instantiate the model with `timm.create_model(name, pretrained=False, num_classes=0)`
2. Print top-level modules: `[name for name, _ in model.named_children()]`
3. Run the discovery script above to confirm spatial sizes
4. Add to the model config in coarse-to-fine order (deepest first)
5. Note any stages that share spatial size (common in EfficientNet/MobileNet)

## Comparing optimization and layer connections

The standalone study scripts run from a development checkout, discover all visible GPUs by default, and schedule one process per GPU. They use frozen FP32 features, full dataset splits, and training-loss stopping criteria rather than a wall-clock budget. Run `--dry-run` to inspect the grid without downloading data or starting jobs.

```bash
# Sweep Adam learning rates to a training-loss plateau, then compare full-batch L-BFGS.
python scripts/run_segmentation_optimizer_study.py --gpus all --resume

# Compare equally sized layer sets across three decoder families.
python scripts/run_segmentation_layer_study.py --gpus all --resume \
    --heads linear conv_block fpn \
    --layer-groups late_four spread_four early_four
```

Both scripts default to Burn Scars with a pretrained ViT-Small/16 backbone and RGB inputs at 224 pixels. Adam and L-BFGS use the same fixed minibatch membership so training-mode BatchNorm sees a shared objective; Adam shuffles batch order, not the images within batches. The layer study extracts the union of the requested intermediate features once, then selects the appropriate tensors for each decoder. Four-layer comparisons keep the number of connections fixed; comparing one layer with four also changes decoder capacity. Parameter counts and exact layer names are included in the results.

Each output directory contains `combined.csv` with raw outcomes and timings, `validation_selected.csv` with learning rates selected by mean validation mIoU across seeds, and per-trial learning curves, checkpoints, and logs. Optimization-only time is separate from convergence checks, validation, and checkpoint I/O. Test data never selects learning rates or checkpoints. Convergence requires a flat recent training-loss window, not merely failure to beat a historical minimum. A plateau is an operational stopping rule, not proof of stationarity; unsuccessful learning rates, numerical stalls, and optional `--max-iterations` safety limits are labeled as nonconvergence.

Use `--gpus 0,2` to restrict resources, `--seeds` and `--adam-lrs` to change the sweep, and `--help` for convergence controls or custom layer groups. DPT requires four connections and the patch-linear decoder uses one. DPT also requires the optional `transformers` dependency, available through `pip install "torchgeo-bench[sam3]"`. Resume only a compatible configuration; scientific changes require a new output directory.
