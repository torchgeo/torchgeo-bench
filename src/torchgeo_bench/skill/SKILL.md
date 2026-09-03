---
name: torchgeo-bench
description: >-
  Benchmark frozen geospatial foundation models on GeoBench V1/V2 (KNN-5,
  linear probe, segmentation mIoU) and location encoders on CoordBench with
  the torchgeo-bench CLI. Use when asked to evaluate a backbone, add a model
  or dataset to torchgeo-bench, resume or interpret its result CSVs, or debug
  a benchmark run.
---

# torchgeo-bench

`torchgeo-bench` evaluates **frozen** backbones: it extracts embeddings once
and fits cheap probes on top (KNN-5 and L-BFGS logistic regression for
classification, a hook-based dense probe for segmentation). Backbone weights
are never fine-tuned. Every run appends rows to a CSV with bootstrapped 95%
confidence intervals.

## Setup

```bash
pip install torchgeo-bench            # or: uv sync --extra dev (from a clone)
torchgeo-bench --help
```

Python 3.12+. The default device is `cuda:0`; pass `device=cpu` on a machine
without a working GPU. Optional model families live behind extras
(`olmoearth`, `terratorch`, `sam3`, `coordbench`, `id`, `viz`); install with
`pip install "torchgeo-bench[olmoearth]"`.

## Get data first

Datasets are always read from `./data/<subdir>` relative to the current
working directory — there are no env vars or path overrides.

```bash
torchgeo-bench download geobench_v1                              # -> data/classification_v1.0/
torchgeo-bench download geobench_v2 --datasets burn_scars,benv2  # -> data/geobenchv2/<name>/
torchgeo-bench download eurosat                                  # -> data/eurosat/
torchgeo-bench download resisc45                                 # -> data/resisc45/
```

`--datasets` is only accepted for the two GeoBench targets. Downloads are
large; prefer a named subset over pulling a whole benchmark.

## Run a benchmark

```bash
torchgeo-bench run                                          # default model rcf, all datasets
torchgeo-bench run -m timm/resnet50 -d m-eurosat
torchgeo-bench run -m torchgeo/scalemae_large_fmow -d m-eurosat,m-so2sat --device cuda:1
torchgeo-bench run -m timm/resnet50 -d burn_scars,pastis     # segmentation datasets
torchgeo-bench run mode=coord -m sincos                      # CoordBench location encoders
```

Flags are shorthand for config overrides, and any `key=value` pair overrides
the config directly. The two forms mix freely on one command line, and flags
win over `key=value`:

```bash
torchgeo-bench run model=timm/resnet50 dataset.names=[m-eurosat] eval.bootstrap=100
```

Values parse as YAML (`[a,b]` is a list, `true` is a bool). Unknown keys are
rejected; prefix with `+` to force-add one (`+model.gsd=10`).

Useful `run` flags: `-m/--model`, `-d/--datasets` (comma-separated or `all`),
`--device`, `-o/--output`, `--resume`, `--seed`, `--partition`, `--bands`,
`--batch-size`, `--image-size`, `--normalization`, `--skip-linear`,
`--bootstrap`, `-v/--verbose`.

Other commands:

```bash
torchgeo-bench run --list-models        # every value accepted by model=/-m
torchgeo-bench run --print-config       # merged config, then exit (no compute)
torchgeo-bench flops -m timm/resnet50   # per-sample GFLOPs -> results/compute_cost.csv
torchgeo-bench --skill                  # print this document
```

**Always dry-run with `--print-config` before launching a long sweep** — it
catches typo'd keys and wrong dataset/band combinations in under a second.

## Configuration surface

Frequently used keys (see `--print-config` for the full tree):

| Key | Default | Notes |
| --- | --- | --- |
| `dataset.names` | `all` | list of dataset ids, or `all` |
| `dataset.bands` | `rgb` | `rgb`, `all`, or explicit band names |
| `dataset.normalization` | `bandspec_zscore` | also `model_native`, `minmax`, `minmax_zscore`, `identity` |
| `dataset.image_size` | `224` | `null` disables resizing |
| `dataset.batch_size` | `64` | lower it first when a run OOMs |
| `eval.bootstrap` | `200` | resamples for the CIs |
| `eval.skip_linear` | `false` | KNN-only, much faster smoke test |
| `eval.segmentation.head_type` | `fpn` | `linear`, `conv_block`, `fpn`, `dpt`, `patch_linear` |
| `eval.segmentation.layers` | `[]` | **must** be set per model for segmentation |
| `mode` | `image` | `coord` switches to the CoordBench track |

Segmentation on a new backbone fails until `eval.segmentation.layers` names
real backbone module paths (e.g. `[layer4,layer3,layer2,layer1]` for a
ResNet). Model YAMLs that support segmentation set this themselves.

## Datasets

- GeoBench V1 classification (`m-` prefix): `m-eurosat`, `m-forestnet`,
  `m-so2sat`, `m-pv4ger`, `m-brick-kiln`, `m-bigearthnet` (multilabel →
  reports `micro_mAP`, not accuracy).
- GeoBench V2 classification: `benv2`, `treesatai`, `so2sat`, `forestnet`.
- GeoBench V2 segmentation: `burn_scars`, `caffe`, `cloudsen12`,
  `dynamic_earthnet`, `flair2`, `fotw`, `kuro_siwo`, `pastis`, `spacenet2`,
  `spacenet7`.
- torchgeo-backed: `eurosat`, `eurosat-spatial`, `resisc45`.

`m-forestnet` (V1) and `forestnet` (V2) are different datasets with different
sensors. For the authoritative list:

```bash
python -c "from torchgeo_bench.datasets import list_datasets; print(list_datasets())"
```

Channel count is `len(bands)`, and it changes per dataset — models are
re-instantiated for every dataset with the right `num_channels`, so never
hard-code 3 or 13 channels. With `dataset.bands=rgb` the runner picks each
dataset's declared RGB band names, which differ across families
(`red,green,blue` vs `b04,b03,b02`, `gray` for `caffe`, `vv,vh` for
`kuro_siwo`).

## Results and resume

- Metrics land in `results/models/<model name>.csv` (one file per model).
  `output=path.csv` redirects everything to one file instead.
- One-time hardware measurements are kept apart: `results/profiles/` and
  `results/intrinsic_dim/`, plus `results/compute_cost.csv` from `flops`.
- The CoordBench track (`mode=coord`) streams its benchmarks from
  HuggingFace — no download — and writes `results/coordbench_results.csv`.
- `method` is `knn<eval.knn_k>` (`knn5` by default), `linear`, or
  `seg-<head_type>` (e.g. `seg-fpn`); `metric_name` is `accuracy`,
  `micro_mAP`, or `mIoU`, reported with `ci_lower` / `ci_upper`.
- Rows are appended atomically under a file lock, so parallel jobs are safe.
- `resume=true` skips a `(dataset, method, model, bands, normalization, ...)`
  combo only when the stored `config_hash` matches the current config — any
  hashed config change re-runs it.
- The checked-in CSVs ship with reference results. To start clean, write to
  your own file (`output=results/my_run.csv`) rather than deleting rows.

Read a whole directory back with
`from torchgeo_bench.results import load_results`.

## Add a model

Do **not** edit `main.py`. Subclass `BenchModel` and implement
`_forward_patch_features(images) -> (B, K)`; the base class handles
normalization and sets `self.num_channels = len(bands)` before your code
runs. Then add `src/torchgeo_bench/conf/model/<name>.yaml` with a
`_target_:` dotted path plus your kwargs — the runner injects `bands` at
construction time, so never declare it in the YAML.

Start from `src/torchgeo_bench/models/contrib_template.py`; it documents the
three normalization strategies (`bandspec_zscore`, `identity`,
`model_native`) and when each applies. Wrappers for optional extras import
their dependency lazily, never at module scope.

## Add a dataset

Add `src/torchgeo_bench/datasets/<safe_name>.py` (hyphens → underscores)
subclassing `_V1Dataset`, `_V2Dataset`, or `BenchDataset`, declaring `name`,
`task`, `num_classes`, `multilabel`, `bands`, `rgb_bands`, and
`split_sizes`. Multi-modality V2 datasets also set
`band_order_strategy = "by_sensor"`. Register it in
`datasets/loading.py::_REGISTRY_SPEC`, export it from `datasets/__init__.py`,
and for new V2 downloads extend `download.py` and `geobench_v2.py`.

## Repository conventions

- Python 3.12+ typing (`list[str]`, `X | None`). No `typing.List/Optional`,
  no `from __future__ import annotations`.
- Google-style docstrings (ruff pydocstyle), `logging` instead of `print()`.
- No defensive `try/except ImportError` around hard dependencies and no bare
  `except Exception:` — catch the specific error and let the rest crash.
- Comment only what the code cannot say itself.
- One logical change per PR, with tests. Commit subjects are Conventional
  Commits, one line, no body.

```bash
ruff check . --fix && ruff format .
pytest                     # fast suite; -m slow adds tests that load real data
pytest tests/test_cli.py -v
```

Tests skip themselves when the corresponding data directory is missing, so a
"skipped" result usually means you need `torchgeo-bench download …` first.

## Troubleshooting

- CUDA errors or no GPU → `device=cpu`.
- OOM → lower `dataset.batch_size`, then `eval.segmentation.batch_size`.
- `Unknown model config` → run `torchgeo-bench run --list-models`.
- Dataset not found → check the exact path under `./data/` from the **current
  working directory**; the runner does not search elsewhere.
- Slow iteration → `eval.skip_linear=true eval.bootstrap=100` on one dataset.
