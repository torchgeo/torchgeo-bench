# torchgeo-bench docs

Detailed reference for everything not in the [README](README.md). Sections:

- [Downloading datasets](#downloading-datasets)
- [Running experiments](#running-experiments)
- [Datasets](#datasets)
- [Available models](#available-models)
- [Adding a new model](#adding-a-new-model)
- [Hydra configuration](#hydra-configuration)
- [Multi-band and multi-modal experiments](#multi-band-and-multi-modal-experiments)
- [Output schema and resume mode](#output-schema-and-resume-mode)
- [Evaluation protocol summary](#evaluation-protocol-summary)
- [Development](#development)
- [Releasing to PyPI](#releasing-to-pypi)
- [Troubleshooting](#troubleshooting)

---

## Downloading datasets

All data lives under `./data/` relative to the current working directory. The
runner does not honour environment variables — paths are fixed.

| Target         | Default destination                | Source                              |
|----------------|------------------------------------|-------------------------------------|
| `geobench_v1`  | `data/classification_v1.0/`        | Hugging Face `recursix/geo-bench-1.0` |
| `geobench_v2`  | `data/geobenchv2/<name>/`          | Hugging Face `aialliance/<name>`    |
| `eurosat`      | `data/eurosat/`                    | torchgeo's `EuroSAT` downloader     |

```bash
# Full GeoBench V1 (~ all V1 classification datasets)
torchgeo-bench download geobench_v1

# All benchmark-supported V2 datasets (classification + segmentation)
torchgeo-bench download geobench_v2

# A subset of V2
torchgeo-bench download geobench_v2 --datasets benv2,burn_scars

# torchgeo EuroSAT
torchgeo-bench download eurosat

# Custom root (will write data into <output-dir>/...)
torchgeo-bench download geobench_v2 --output-dir /scratch/data
```

The default V2 set is: `benv2`, `burn_scars`, `caffe`, `cloudsen12`,
`dynamic_earthnet`, `flair2`, `forestnet`, `fotw`, `kuro_siwo`, `pastis`,
`so2sat`, `spacenet2`, `spacenet7`, `treesatai`. (See
`DEFAULT_V2_DATASETS` in `src/torchgeo_bench/download.py`.)

## Running experiments

```bash
# Default: RCF on every available dataset
torchgeo-bench run

# Pick a model
torchgeo-bench run model=timm/resnet50

# Pick datasets
torchgeo-bench run dataset.names=[m-eurosat,m-forestnet]

# Quick smoke test (skip linear probe, minimal bootstrap)
torchgeo-bench run eval.skip_linear=true eval.bootstrap=10

# Smaller training partition (V1 only — see "Data partitions" below)
torchgeo-bench run dataset.partition=0.01x_train output=results/1pct.csv

# Resume a previous run
torchgeo-bench run resume=true

# Segmentation datasets
torchgeo-bench run dataset.names=[burn_scars,pastis,flair2]

# Specific GPU
torchgeo-bench run device=cuda:1
```

`torchgeo-bench run` forwards every remaining argument to Hydra, so any key
in [`conf/config.yaml`](src/torchgeo_bench/conf/config.yaml) can be
overridden inline.

## Datasets

`dataset.names=all` expands to every dataset registered in
`src/torchgeo_bench/datasets/loading.py`. Each dataset class lives in its
own file and declares `name`, `task`, `num_classes`, `multilabel`, `bands`,
`rgb_bands`, and `split_sizes`.

### GeoBench V1 (classification, `m-` prefix)

| Dataset           | Classes | Bands | Multilabel | Notes                                      |
|-------------------|--------:|------:|------------|--------------------------------------------|
| `m-eurosat`       |      10 |    13 | No         | Sentinel-2                                 |
| `m-forestnet`     |      12 |     6 | No         | Landsat                                    |
| `m-so2sat`        |      17 |    18 | No         | Sentinel-1 + Sentinel-2                    |
| `m-pv4ger`        |       2 |     3 | No         | Aerial RGB                                 |
| `m-brick-kiln`    |       2 |    13 | No         | Sentinel-2                                 |
| `m-bigearthnet`   |      43 |    12 | **Yes**    | Sentinel-2; multi-label → micro-mAP        |

### GeoBench V2 — classification

| Dataset      | Classes | Bands | Notes                                            |
|--------------|--------:|------:|--------------------------------------------------|
| `benv2`      |      19 |    14 | Sentinel-1 + Sentinel-2 (multi-modal)            |
| `treesatai`  |      13 |    19 | Aerial + Sentinel-2 + Sentinel-1 (multi-modal)   |
| `so2sat`     |      17 |    12 |                                                  |
| `forestnet`  |      12 |     6 | Sentinel-2 (different from V1 `m-forestnet`)     |

### GeoBench V2 — segmentation

| Dataset            | Classes | Bands | Notes                                          |
|--------------------|--------:|------:|------------------------------------------------|
| `burn_scars`       |       3 |     6 |                                                |
| `caffe`            |       4 |     1 | Aerial grayscale                               |
| `cloudsen12`       |       4 |    12 |                                                |
| `dynamic_earthnet` |       7 |    16 |                                                |
| `flair2`           |      13 |     5 | Aerial + Sentinel-2                            |
| `fotw`             |       4 |     4 | Fields of the World                             |
| `kuro_siwo`        |       4 |     3 | SAR `vv`/`vh` + DEM (no RGB triplet)           |
| `pastis`           |      20 |    16 | Sentinel-2 + Sentinel-1 (multi-modal)          |
| `spacenet2`        |       3 |     9 | WorldView 8-band + pan                         |
| `spacenet7`        |       3 |     3 |                                                |

### torchgeo template

| Dataset    | Classes | Notes                                         |
|------------|--------:|-----------------------------------------------|
| `eurosat`  |      10 | Loaded via `torchgeo.datasets.EuroSAT`         |

> `m-forestnet` and `forestnet` are *different* datasets (V1 vs V2). The V1
> version uses Landsat; the V2 version uses Sentinel-2 with 6 bands.

### Data partitions

Only V1 datasets honour the `partition` argument (controlled by per-dataset
JSON files distributed with the data). V2 datasets ignore it.

```bash
# 1% of the training split on V1 datasets
torchgeo-bench run dataset.partition=0.01x_train
```

Common values: `default`, `0.01x_train`, `0.02x_train`, `0.05x_train`,
`0.10x_train`, `0.20x_train`, `0.50x_train`, `1.00x_train`. The exact set
available depends on which partition JSON files ship with the dataset.

## Available models

All model configs live under
[`src/torchgeo_bench/conf/model/`](src/torchgeo_bench/conf/model/). A
config's `_target_` resolves to a class re-exported from
`torchgeo_bench.models`.

### RCF — Random Convolutional Features

`torchgeo_bench.models.RCFBench`. Gaussian or empirical random features in the
spirit of MOSAIKS.

```bash
torchgeo-bench run model=rcf
torchgeo-bench run model=rcf model.mode=empirical model.features=1024
```

### Image statistics

`torchgeo_bench.models.ImageStatsBench`. A trivial baseline that returns
per-channel mean / std as a feature vector.

```bash
torchgeo-bench run model=imagestats
```

### timm — ImageNet-pretrained CNNs and ViTs

`torchgeo_bench.models.TimmPatchBenchModel`. Configs under
`src/torchgeo_bench/conf/model/timm/` cover ResNet, ConvNeXt, EfficientNet,
DenseNet, RegNet, MobileNetV3, VGG, MaxViT, and more. ViT/DeiT/Swin variants
live under `timm/vit/`.

```bash
torchgeo-bench run model=timm/resnet50
torchgeo-bench run model=timm/convnext_base dataset.names=[m-eurosat]
torchgeo-bench run model=timm/vit/vit_base_patch16_224 dataset.image_size=224
torchgeo-bench run model=timm/vit/swin_base_patch4_window7_224 eval.skip_linear=true
```

ViT-style backbones expect a fixed spatial resolution. Set
`dataset.image_size=224` (`bilinear` by default; switch to `bicubic` /
`nearest` via `dataset.interpolation`) to resize the dataset tiles for any
model.

timm models rebuild their input convolution for any number of channels —
they work with `dataset.bands=all` out of the box (pretrained 3-channel
weights are averaged / replicated as needed).

### torchgeo foundation models

Configs under `src/torchgeo_bench/conf/model/torchgeo/`. Most are RGB-only
self-supervised checkpoints from torchgeo's model hub.

```bash
# Sentinel-2 RGB SSL
torchgeo-bench run model=torchgeo/resnet50_s2rgb_moco
torchgeo-bench run model=torchgeo/resnet18_s2rgb_seco
torchgeo-bench run model=torchgeo/resnet50_fmow_gassl

# ScaleMAE on fMoW RGB
torchgeo-bench run model=torchgeo/scalemae_large_fmow

# DOFA — band-agnostic (currently configured for Sentinel-2 RGB wavelengths)
torchgeo-bench run model=torchgeo/dofa_base

# Satlas Swin-V2 (NAIP / Sentinel-2 RGB)
torchgeo-bench run model=torchgeo/swinv2b_naip_satlas_mi
torchgeo-bench run model=torchgeo/swinv2b_s2rgb_satlas_mi

# EarthLoc place-recognition descriptor
torchgeo-bench run model=torchgeo/earthloc_s2_resnet50
```

### OlmoEarth (AI2)

`torchgeo_bench.models.OlmoEarthBenchModel`. Requires the optional
`olmoearth` extra:

```bash
pip install 'torchgeo-bench[olmoearth]'
torchgeo-bench run model=olmoearth_base
torchgeo-bench run model=olmoearth_large dataset.bands=all
```

> Each model wrapper owns its own normalization policy by overriding
> `BenchModel.normalize_inputs`. There is no `dataset.normalization` key
> anymore — backbones that do their own normalization (OlmoEarth, some
> torchgeo wrappers) override `normalize_inputs` to identity.

### SAM3 vision encoder

`torchgeo_bench.models.SAM3Encoder`. Requires the optional `sam3` extra and
a local checkpoint at `checkpoints/sam3/`:

```bash
pip install 'torchgeo-bench[sam3]'
torchgeo-bench run model=sam3_encoder dataset.bands=[red,green,blue]
```

## Adding a new model

1. Implement [`BenchModel`](src/torchgeo_bench/models/interface.py) in any
   importable module:

   ```python
   import torch

   from torchgeo_bench.datasets.base import BandSpec
   from torchgeo_bench.models.interface import BenchModel


   class MyModel(BenchModel):
       def __init__(self, bands: list[BandSpec], pretrained: bool = True):
           super().__init__(bands=bands)
           # self.num_channels == len(bands) is set for you
           self.backbone = create_my_backbone(
               in_channels=self.num_channels, pretrained=pretrained
           )

       def _forward_patch_features(
           self,
           images: torch.Tensor,
           bboxes: torch.Tensor | None = None,
       ) -> torch.Tensor:
           # `images` has already been normalized via self.normalize_inputs
           return self.backbone(images)  # must return (B, K)
   ```

   Notes:
   - `BenchModel.__init__` takes a `bands: list[BandSpec]` argument; the
     runner builds it from the dataset wrapper and injects it for you. You
     do **not** put `bands` in your YAML.
   - The public `forward_patch_features` is sealed and applies
     `normalize_inputs` (per-channel z-score from `BandSpec.{mean, std}`)
     before dispatching to your `_forward_patch_features`. Override
     `normalize_inputs` if your backbone expects a different policy
     (e.g. backbone-internal normalization → return `images` unchanged).

2. Drop a config at
   `src/torchgeo_bench/conf/model/<name>.yaml`:

   ```yaml
   _target_: my_pkg.MyModel
   pretrained: true
   name: my_model
   ```

3. Run it:

   ```bash
   torchgeo-bench run model=<name>
   ```

## Hydra configuration

The packaged config tree:

```
src/torchgeo_bench/conf/
├── config.yaml          # main config (defaults below)
└── model/
    ├── rcf.yaml
    ├── imagestats.yaml
    ├── sam3_encoder.yaml
    ├── olmoearth_{base,large}.yaml
    ├── timm/
    │   └── ...
    └── torchgeo/
        └── ...
```

### Defaults (excerpt from `conf/config.yaml`)

```yaml
seed: 0
device: cuda:0
output: results/all_results.csv
verbose: false
resume: false

dataset:
  names: all
  partition: default          # V1 only
  batch_size: 64
  bands: rgb                  # rgb | all | [red, green, blue, nir, ...]
  image_size: 224             # null = preserve native size
  interpolation: bilinear     # bilinear | bicubic | nearest

eval:
  bootstrap: 200
  c_range: [-7, 2, 20]        # log10 LBFGS C sweep: start, stop, count
  merge_val: true
  skip_linear: false

  intrinsic_dim:              # optional, requires the [id] extra
    enabled: false
    estimators: [TwoNN, MLE, lPCA]
    splits: [train]
    max_samples: 10000

  segmentation:
    head_type: fpn            # linear | conv_block | fpn | dpt
    layers: []                # set per-model (deepest-first for fpn)
    lr: 1e-3
    epochs: 10
    lr_scheduler: cosine      # cosine | none
    cache_features: true
    cache_dtype: float16
    save_viz: false
```

### Common overrides

```bash
torchgeo-bench run device=cuda:1
torchgeo-bench run eval.bootstrap=1000
torchgeo-bench run output=results/my_sweep.csv
torchgeo-bench run resume=true
torchgeo-bench run eval.segmentation.head_type=linear eval.segmentation.epochs=5
```

## Multi-band and multi-modal experiments

`dataset.bands` controls which spectral bands are loaded:

| Value                          | Behaviour                                             |
|--------------------------------|-------------------------------------------------------|
| `rgb` (default)                | Each dataset's `rgb_bands` triplet                    |
| `all`                          | Every band declared on the wrapper                    |
| `[red, green, blue, nir]` etc. | Explicit short names (must exist in the wrapper)      |

The runner derives `num_channels` from the loaded tensor and constructs the
matching `list[BandSpec]` so the model wrapper can size its input layer and
per-channel normalization correctly. The selected `bands` value is recorded
in the CSV so multiple runs writing to the same file (and `resume=true`)
distinguish RGB from multispectral results.

```bash
# Use all 13 Sentinel-2 bands on EuroSAT with a pretrained timm ResNet-18
torchgeo-bench run model=timm/resnet18 dataset.names=[m-eurosat] dataset.bands=all
```

### Multi-modality (V2)

Some V2 datasets are multi-sensor (e.g. `treesatai` = aerial + S2 + S1,
`pastis` = S2 + S1, `kuro_siwo` = SAR + DEM). The corresponding wrappers
set `band_order_strategy = "by_sensor"` and the V2 base class groups
`BandSpec` entries by sensor before passing them to the upstream
`geobench_v2` loader. End users don't need to do anything special — set
`dataset.bands=all` (or an explicit subset) and the right per-modality
tensors are concatenated into a single `image` key.

### Model compatibility

- **timm** wrappers rebuild the input conv for any `num_channels`.
- **RCF** and **imagestats** are band-agnostic.
- **torchgeo RGB-only wrappers** (most of them) hold fixed-channel
  pretrained weights and don't currently adapt to non-RGB inputs — see
  [#16](https://github.com/torchgeo/torchgeo-bench/issues/16).
- **DOFA** accepts variable channels via wavelength tokens but the current
  wrapper hard-codes Sentinel-2 RGB wavelengths — see
  [#15](https://github.com/torchgeo/torchgeo-bench/issues/15).

## Output schema and resume mode

Results are appended to `output` (default `results/all_results.csv`) using
advisory file locking, so multiple parallel jobs can safely write to the
same file.

```csv
dataset,method,metric_name,metric_value,ci_lower,ci_upper,feature_dim,best_c,n_train,n_val,n_test,seed,model,name,normalization,image_size,interpolation,partition,bands
m-eurosat,knn5,accuracy,0.8234,0.8123,0.8345,512,,21600,5400,5400,0,torchgeo_bench.models.RCFBench,rcf,raw,224,bilinear,default,rgb
m-eurosat,linear,accuracy,0.8567,0.8461,0.8673,512,0.1,21600,5400,5400,0,torchgeo_bench.models.RCFBench,rcf,raw,224,bilinear,default,rgb
burn_scars,seg-fpn,mIoU,0.6234,0.0,0.0,768,,1000,200,300,0,torchgeo_bench.models.TimmPatchBenchModel,resnet50,raw,224,bilinear,default,rgb
```

The `normalization` column is currently always `raw` — datasets emit
unnormalized tensors and each model wrapper applies its own normalization
inside `BenchModel.normalize_inputs`. The column is preserved in the schema
so older results from before that refactor remain distinguishable on
resume.

`method` values:

- `knn5` — KNN-5 classification (or multilabel KNN for `m-bigearthnet`).
- `linear` — L-BFGS logistic regression with C-sweep on the validation set.
- `seg-<head_type>` — segmentation probe with the configured head
  (`linear` / `conv_block` / `fpn` / `dpt`).
- `intrinsic_dim` — optional intrinsic-dimension metrics on extracted
  embeddings (requires the `[id]` extra; rows added when
  `eval.intrinsic_dim.enabled=true`).

`resume=true` reads the existing CSV and skips any
`(dataset, method, model._target_, model.name, normalization, image_size,
interpolation, partition, bands)` tuple already present.

## Evaluation protocol summary

See [METHODOLOGY.md](METHODOLOGY.md) for the full description. In brief:

1. **Per-dataset reinitialisation.** The model is instantiated fresh for
   each dataset because `num_channels` (and therefore the input conv) varies.
2. **Classification** (`m-bigearthnet` is multilabel; everything else is
   single-label):
   - Extract train / val / test embeddings once.
   - KNN-5 with FAISS-CPU + bootstrapped 95% CI on test predictions.
   - L-BFGS logistic regression sweeping `C ∈ logspace(c_range)` on the
     validation set, optionally re-fit on `train ∪ val`
     (`eval.merge_val=true`), then evaluated on test with bootstrapped CIs.
3. **Segmentation:** forward hooks on configured backbone layers capture
   intermediate feature maps; the chosen head (`linear` / `conv_block` /
   `fpn` / `dpt`) is trained with AdamW + CrossEntropy
   (`ignore_index=255`), evaluated with `MulticlassJaccardIndex` (mIoU).
   Features are cached in RAM by default to avoid re-running the frozen
   backbone every epoch.

## Development

The repo uses a conda env named `torchgeo-bench`. After cloning:

```bash
conda env update -n torchgeo-bench -f environment.yml   # or `make install`
conda activate torchgeo-bench
uv sync --extra dev
```

`Makefile` shortcuts:

| Target          | What it does                                      |
|-----------------|---------------------------------------------------|
| `make install`  | Create / update the conda env and install `[dev]` |
| `make tests`    | `pytest` (skips `slow` integration tests)         |
| `make lint`     | `pre-commit run --all-files`                      |
| `make format`   | `ruff format` + `ruff check --fix --select I`     |

Direct commands inside the env:

```bash
pytest                                       # full suite (skips slow)
pytest tests/test_geobench_dataset.py -v     # one file
pytest -k "m-eurosat" -v                     # by keyword
pytest -m slow                               # include integration tests
pytest --no-cov                              # faster iteration

ruff check . --fix                           # lint + autofix
ruff format .                                # format
```

Tests skip gracefully if data is missing — they look under `data/` from the
working directory. Integration tests marked `slow` actually load datasets
and run models; they're excluded by default.

### Code standards

- Python 3.12+; modern type hints (`list[str]`, `X | None`).
- Use `logging.getLogger(__name__)`, not `print()`.
- Google-style docstrings (pydocstyle is enforced via ruff).
- No `from __future__ import annotations`.
- No defensive `try/except ImportError` for hard dependencies — every
  package in `[project.dependencies]` is guaranteed to be installed.

## Releasing to PyPI

1. Configure a [PyPI Trusted Publisher](https://docs.pypi.org/trusted-publishers/)
   for this repository with environment name `pypi`.
2. Tag and push:

   ```bash
   git tag v0.2.0
   git push origin v0.2.0
   ```

The `Publish to PyPI` workflow (`.github/workflows/release.yml`) builds and
uploads the release automatically.

## Troubleshooting

### `Dataset directory not found` / files missing

Make sure data lives under `./data/<canonical-subdir>/` from the directory
where you run `torchgeo-bench`:

- V1: `data/classification_v1.0/<name>/`
- V2: `data/geobenchv2/<name>/`
- EuroSAT: `data/eurosat/`

There are no `GEOBENCH_ROOT` / `GEOBENCH_V2_ROOT` env vars — paths are
fixed. Re-run `torchgeo-bench download …` to fetch missing data.

### `ModuleNotFoundError: geobench`

The legacy `geobench` package is no longer a dependency. V1 datasets are
read directly from HDF5 (`GeoBenchv1` in
`src/torchgeo_bench/datasets/geobench_v1.py`); V2 dispatches to upstream
`geobench_v2.datasets.GeoBench<X>`.

### CUDA out of memory

```bash
torchgeo-bench run dataset.batch_size=32
# or run on CPU
torchgeo-bench run device=cpu
```

For segmentation, also try `eval.segmentation.cache_dtype=float32 eval.segmentation.cache_features=false` if RAM is the bottleneck rather than GPU memory.

### `KeyError: 's2'` on a V2 dataset

A known V2 issue (see `ROADMAP.md`): `geobench_v2.rearrange_bands` expects
modality keys (`'s2'`, `'s1'`, …) that aren't present when a flat band list
is requested. Workaround: use `dataset.bands=all` for affected V2 datasets.
