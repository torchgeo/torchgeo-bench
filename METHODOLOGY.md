# Methodology

This document describes the evaluation methodology used by `torchgeo_bench.main` for each of the supported task types. In all cases the backbone model is kept **frozen** — the benchmark measures the quality of learned representations, not end-to-end fine-tuning performance.

---

## Overview

The benchmark script loads a pre-trained backbone model and one or more geospatial datasets. Depending on whether a dataset provides per-pixel masks (segmentation) or per-image labels (classification), a different evaluation path is taken:

| Dataset type   | Evaluation methods      | Metric   |
|----------------|-------------------------|----------|
| Classification | KNN, Linear Probe       | Accuracy |
| Segmentation   | Seg-Linear, Seg-Conv    | mIoU     |

---

## Feature Extraction (Classification Tasks)

For classification tasks (KNN and Linear), features are extracted once and reused by both evaluation methods.

1. The backbone model is placed in eval mode with gradients disabled (`torch.no_grad` + `torch.inference_mode`).
2. Each batch of images is passed through the backbone's `forward()` (equivalently `forward_patch_features()`), which returns embeddings of shape `(B, K)`.
3. If the backbone returns a dictionary (e.g., DINO-style models), the code looks for keys `"norm"`, `"global_pool"`, or `"head.global_pool"` in that order.
4. If the output is 3-D `(B, tokens, K)` — as with ViT models that return per-patch tokens — a **global average pool** across the token dimension is applied to produce a single `(B, K)` vector.
5. All embeddings and labels are concatenated across batches into NumPy arrays for downstream evaluation.

---

## KNN (k-Nearest Neighbors Classification)

**Method name:** `knn5`

A non-parametric baseline that measures how well the feature space clusters by class.

### Procedure

1. Extract feature embeddings for the train and test splits (see above).
2. Fit a **k=5 nearest-neighbor classifier** using FAISS (`FaissKNNClassifier`) on the training embeddings.
3. Predict labels for every test sample.
4. Compute **accuracy** on the test set.
5. Compute **95% bootstrap confidence intervals** (default 500 resamples) by resampling test predictions with replacement.

### Key Details

- **No hyperparameter tuning** — k is fixed at 5.
- **No validation set usage** — the validation split is extracted but not consumed by KNN.
- FAISS is used for efficient nearest-neighbor search and can run on CPU or GPU.
- Feature vectors are cast to `float32` and labels to `int64` before indexing.

---

## Linear Probe (Logistic Regression)

**Method name:** `linear`

A standard linear evaluation protocol that trains a multinomial logistic regression on top of frozen features.

### Procedure

1. Extract feature embeddings for the train, validation, and test splits.
2. **Hyperparameter sweep:** Train a separate logistic regression model for each regularization strength `C` in a log-spaced grid (default: 20 values from 10⁻⁷ to 10²). Each model is evaluated on the validation set to select the best `C`.
3. **Final model:** Retrain a logistic regression with the best `C` on the training data (optionally merged with validation data if `merge_val=true`, which is the default).
4. Evaluate on the test set and compute **accuracy** with **95% bootstrap confidence intervals**.

### Logistic Regression Implementation

The `LogisticRegression` class is a custom PyTorch implementation with the same objective scaling as scikit-learn:

$$\text{loss} = \frac{1}{n} \text{CrossEntropy} + \frac{1}{n} \cdot \frac{1}{2C} \| W \|^2$$

- **Architecture:** A single `nn.Linear(K, num_classes)` layer (weight matrix + bias).
- **Solver (sweep):** L-BFGS with strong Wolfe line search, `max_iter=2000`, `tol=1e-6`.
- **Solver (final):** Same as sweep but with `max_iter=4000` for tighter convergence.
- **Alternative solver:** Adam with mini-batches is available but L-BFGS is the default.
- **No feature standardization** — embeddings are used as-is.
- **TF32** is enabled on CUDA for faster matmul when available.

### Hyperparameters (Configurable)

| Parameter        | Default                  | Description                                       |
|------------------|--------------------------|---------------------------------------------------|
| `c_range`        | `[-7, 2, 20]`           | Log₁₀ start, stop, and number of C values         |
| `merge_val`      | `true`                   | Merge train+val for final model training           |
| `bootstrap`      | `500`                    | Number of bootstrap resamples for confidence intervals |

---

## Segmentation Linear Probe

**Method name:** `seg-linear` (config: `eval.segmentation.head_type: "linear"`)

A lightweight per-pixel linear classifier attached to intermediate backbone feature maps, used to evaluate spatial representation quality.

### Procedure

1. A `SegmentationProbe` wraps the frozen backbone with **forward hooks** on specified intermediate layers to capture feature maps.
2. For each hooked layer, features are processed into spatial maps `(B, C, H, W)`:
   - 2-D features `(B, C)` are reshaped to `(B, C, 1, 1)`.
   - 3-D ViT-style features `(B, L, C)` are reshaped to `(B, C, √L, √L)` (assuming square spatial grids).
   - 4-D features `(B, C, H, W)` are used directly.
3. Each layer's feature map passes through its own **head**: `BatchNorm2d → Conv2d(C, num_classes, kernel_size=1)` (i.e., a 1×1 convolution, which is equivalent to a per-pixel linear classifier).
4. Each head's logits are bilinearly upsampled to the original input resolution.
5. If multiple layers are used, their logits are combined via a **learned weighted sum** (`scale_weights` parameter).
6. The probe head is trained end-to-end (backbone frozen) using the `SegmentationSolver`.

### Training

- **Optimizer:** AdamW (applied only to the unfrozen probe parameters).
- **Loss:** CrossEntropyLoss with `ignore_index=255` for unlabeled pixels.
- **Epochs:** Configurable (default: 1).
- **Learning rate:** Configurable (default: 1e-3).

### Evaluation

- **Metric:** Mean Intersection-over-Union (mIoU) computed via `torchmetrics.MulticlassJaccardIndex`.
- Pixels with the ignore index (255) are excluded from both loss and metric computation.

---

## Segmentation Convolutional Probe

**Method name:** `seg-conv_block` (config: `eval.segmentation.head_type: "conv_block"`)

A slightly more expressive segmentation head that projects and fuses multi-scale features before classification. This tests whether the backbone captures complementary information at different depths.

### Procedure

1. Same hook-based feature extraction as the linear segmentation probe.
2. Each layer's feature map is projected through a **convolutional block**: `Conv2d(C, embed_dim, 1×1, no bias) → BatchNorm2d → SiLU`.
   - `embed_dim` defaults to 256 (configurable via `hidden_dim`).
3. All projected feature maps are **bilinearly upsampled** to match the largest spatial resolution among them (minimum 16×16).
4. The aligned feature maps are **concatenated** along the channel dimension, yielding a tensor of shape `(B, embed_dim × num_layers, H, W)`.
5. A final `Conv2d(embed_dim × num_layers, num_classes, kernel_size=1)` produces per-pixel logits.
6. Logits are bilinearly upsampled to the original input resolution.

### Training & Evaluation

Identical to the linear segmentation probe:

- **Optimizer:** AdamW on unfrozen parameters only.
- **Loss:** CrossEntropyLoss with `ignore_index=255`.
- **Metric:** mIoU via `torchmetrics.MulticlassJaccardIndex`.
- **Epochs/LR:** Same configurable defaults.

---

## Segmentation FPN Probe

**Method name:** `seg-fpn` (config: `eval.segmentation.head_type: "fpn"`)

A Feature Pyramid Network-style decoder that fuses multi-scale feature maps top-down, matching the structure used in dense prediction literature.

### Procedure

1. Same hook-based feature extraction as the other segmentation probes. **Layers must be supplied in coarse-to-fine order** (deepest / lowest-resolution first, e.g. `["layer4", "layer3", "layer2", "layer1"]` for a ResNet).
2. Each layer's feature map is projected to `hidden_dim` channels via a lateral 1×1 conv.
3. A **top-down pathway** accumulates context: starting from the coarsest scale, each level is upsampled 2× and added to the next finer lateral output.
4. Each merged level is refined with a 3×3 conv.
5. All refined levels are upsampled to the finest spatial resolution, **concatenated**, and passed through a 1×1 conv to produce per-pixel class logits.
6. Logits are bilinearly upsampled to the original input resolution.

### Training & Evaluation

Identical to the other segmentation probes (AdamW, CrossEntropyLoss, mIoU).

---

## Segmentation Probe Options

All options are set under `eval.segmentation` in `conf/config.yaml` (global defaults) or in a model's config yaml (per-model override via `eval.segmentation`).

### Head type

| `head_type` | Description |
|---|---|
| `linear` | Per-layer BN + 1×1 conv → upsample. Multiple layers are fused with learned scalar weights. |
| `conv_block` | Per-layer 1×1 proj to `hidden_dim` → upsample + concat → 1×1 head. |
| `fpn` | FPN top-down pathway (see above). Layers must be coarse-to-fine. |
| `dpt` | DPT-style reassemble + fusion transformer decoder. |

### Training knobs

| Option | Default | Description |
|---|---|---|
| `layers` | *(per model)* | List of backbone layer names to hook. For FPN, deepest layer first. |
| `epochs` | `10` | Training epochs for the probe head. |
| `lr` | `1e-3` | Initial learning rate (AdamW). |
| `lr_scheduler` | `cosine` | `cosine` (CosineAnnealingLR to 1e-6) or `none` (constant). |
| `loss` | `ce` | `ce` (CrossEntropyLoss) or `bce` (binary CE over one-hot targets). |
| `hidden_dim` | `256` | Projection dimension for `conv_block` and `fpn` heads. |
| `batch_size` | `64` | Batch size when training the probe head. |

### Feature caching

| Option | Default | Description |
|---|---|---|
| `cache_features` | `true` | Pre-extract backbone features once per split into RAM. Features are stored layer-first as contiguous `(N, C, H, W)` float16 tensors (`CachedFeaturesDataset`). GPU transfer is a single memcpy per layer. Eliminates backbone re-runs across epochs — the dominant speedup. |
| `cache_dtype` | `float16` | Storage dtype for cached features. `float16` halves RAM; autocast upcasts during the head forward pass. |

---
| `loss_delta` | `initial_loss − final_loss`; near 0 means the optimizer made no progress (bad LR or detached graph) |

---

## Common Configuration

All tasks share these settings (configurable via Hydra):

| Setting           | Default          | Description                                    |
|-------------------|------------------|------------------------------------------------|
| `seed`            | `0`              | Random seed for reproducibility                |
| `device`          | `cuda:5`         | PyTorch device                                 |
| `dataset.batch_size` | `64`          | Batch size for data loading                    |
| `dataset.normalization` | `mean_stdev` | Input normalization strategy                 |
| `dataset.image_size` | `224`         | Resize input images (null = no resize)         |
| `dataset.interpolation` | `bilinear` | Resize interpolation method                   |
| `resume`          | `false`          | Skip already-computed (dataset, method, model) combinations |

### Resume Logic

When `resume=true`, the script loads existing results from the output CSV and skips any `(dataset, method, model, name, normalization, image_size, interpolation, partition)` combination that has already been computed.

### Output

All results are appended atomically (with advisory file locking) to a single CSV file. Each row records the evaluation result along with full metadata including model identity, dataset configuration, hyperparameters, and confidence intervals.
