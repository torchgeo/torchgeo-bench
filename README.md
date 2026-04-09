# torchgeo-bench

A lightweight benchmarking framework for evaluating geospatial foundation models and feature extractors on standardized datasets.

## Setup: Download GeoBench Dataset

Before running any benchmarks, download the GeoBench dataset. The framework supports both GeoBench v1 and v2:

### GeoBench v1 (default)

```bash
# Download all GeoBench v1 datasets
torchgeo-bench download

# Specify custom location
torchgeo-bench download --output-dir /your/path/to/data
```

### GeoBench v2

```bash
# Download all GeoBench v2 datasets
torchgeo-bench download --version v2

# Download specific v2 datasets
torchgeo-bench download --version v2 --datasets forestnet,so2sat,benv2

# Download single dataset
torchgeo-bench download --version v2 --datasets caffe
```

Available GeoBench v2 datasets: `benv2`, `biomassters`, `burn_scars`, `caffe`, `cloudsen12`, `dynamic_earthnet`, `everwatch`, `flair2`, `fotw`, `kuro_siwo`, `pastis`, `spacenet2`, `spacenet7`, `substation`, `treesatai`, `wind_turbine`, `so2sat`, `forestnet`

Alternatively, you can still use the standalone script:

```bash
python torchgeo_bench_download.py --version v1
python torchgeo_bench_download.py --version v2 --datasets forestnet
```

## Overview

`torchgeo-bench` provides:

1. **Simple Model Interface**: Define your model by implementing `forward_features(images, bboxes)` → embeddings
2. **Automated Evaluation**: KNN-5, Linear Probing, and Segmentation (mIoU) with bootstrapped confidence intervals
3. **GeoBench V1 & V2 Integration**: Direct access to classification and segmentation benchmark datasets
4. **Resume Mode**: Skip already-computed experiments when interrupted/restarted
5. **Hydra Configuration**: Flexible experiment configuration without code changes
6. **Efficient Workflows**: Per-dataset model reinitialization handles varying input channels
7. **Atomic CSV Writes**: Results appended with file locking for parallel job safety

## Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd torchgeo-bench

# Install dependencies (Python 3.12+)
pip install -e .

# Or using conda
conda env create -f environment.yml
conda activate torchgeo-bench
```

### Basic Usage

```bash
# Run benchmark with default RCF model on all datasets (expects GeoBench data in 'data/')
torchgeo-bench run

# Use pretrained ResNet50 (timm/ImageNet)
torchgeo-bench run model=timm/resnet50

# Benchmark on specific datasets with verbose output
torchgeo-bench run dataset.names=[m-eurosat,m-forestnet] verbose=true

# Quick evaluation (skip linear probing, minimal bootstrap)
torchgeo-bench run eval.skip_linear=true eval.bootstrap=100

# Use smaller training partition
torchgeo-bench run dataset.partition=0.01x_train output=results_1pct.csv

# Resume a previously interrupted run (skips completed experiments)
torchgeo-bench run resume=true

# Evaluate segmentation datasets
torchgeo-bench run dataset.names=[burn_scars,pastis,flair2]

# Select specific GPU device
torchgeo-bench run device=cuda:1
```

Alternatively, you can still use the standalone script with Hydra directly:

```bash
python torchgeo_bench.py model=timm/resnet50
```

## Model Interface

To benchmark your own model, implement the `BenchModel` abstract base class:

```python
from src.interface import BenchModel
import torch

class MyModel(BenchModel):
    def __init__(self, num_channels: int, **kwargs):
        super().__init__(num_channels=num_channels)
        # num_channels varies per dataset (e.g., 3 for RGB)
        self.backbone = create_my_backbone(in_channels=num_channels)
    
    def forward_features(self, images: torch.Tensor, bboxes=None) -> torch.Tensor:
        """
        Args:
            images: (B, C, H, W) tensor in [0, 1] range (after normalization)
            bboxes: Optional (B, 4) geographic bounds (minx, miny, maxx, maxy)
        
        Returns:
            embeddings: (B, K) tensor
        """
        return self.backbone(images)  # Must return (B, K)
```

### Register Your Model

Create a config file `conf/model/mymodel.yaml`:

```yaml
_target_: mymodule.MyModel
num_channels: 3  # placeholder, auto-set per dataset
pretrained: true
# ... other model kwargs
```

Then run:

```bash
torchgeo-bench run model=mymodel
```

## Available Models

### RCF (Random Convolutional Features)
Gaussian or empirical random features à la MOSAIKS.

```bash
# Gaussian RCF
torchgeo-bench run model=rcf

# Empirical RCF (samples patches from training data)
torchgeo-bench run model=rcf model.mode=empirical model.features=1024
```

### Timm ResNet50
Pretrained ImageNet ResNet50 from `timm`.

```bash
torchgeo-bench run model=timm/resnet50
```

### Vision Transformers (ViT / DeiT / Swin)
Configs generated under `conf/model/timm/vit/` (see `create_vit_configs.py`). Vision backbones often expect a fixed spatial resolution (e.g., 224×224). You can now control resizing globally via dataset config:

```bash
# Resize all dataset tiles to 224 (bicubic by default)
torchgeo-bench run model=timm/vit/vit_base_patch16_224 dataset.image_size=224

# Use bilinear interpolation
torchgeo-bench run model=timm/vit/vit_base_patch16_224 dataset.image_size=224 dataset.interpolation=bilinear
```

If you omit `dataset.image_size`, native tile sizes are preserved. Model-level `auto_resize` remains available as a fallback but dataset-level resizing is preferred for consistency across models.

Examples:

```bash
torchgeo-bench run model=timm/vit/vit_base_patch16_224
torchgeo-bench run model=timm/vit/deit_small_patch16_224 dataset.names=[m-eurosat]
torchgeo-bench run model=timm/vit/swin_base_patch4_window7_224 eval.skip_linear=true
```

To study scale effects without resizing, simply avoid setting `dataset.image_size` and (optionally) disable the model fallback:

```bash
torchgeo-bench run model=timm/vit/vit_base_patch16_224 model.auto_resize=false
```

### torchgeo Foundation Models (RGB)
Pretrained geospatial foundation models loaded via `torchgeo.models`. Configs under `conf/model/torchgeo/`.

```bash
# Sentinel-2 RGB self-supervised (MoCo, SeCo, GASSL, Satlas)
torchgeo-bench run model=torchgeo/resnet50_s2rgb_moco
torchgeo-bench run model=torchgeo/resnet18_s2rgb_seco

# ScaleMAE (fMoW RGB)
torchgeo-bench run model=torchgeo/scalemae_large_fmow

# DOFA — band-agnostic (RGB wavelengths)
torchgeo-bench run model=torchgeo/dofa_base

# Swin-V2-B with Satlas (NAIP / Sentinel-2 RGB)
torchgeo-bench run model=torchgeo/swinv2b_naip_satlas_mi

# EarthLoc (place-recognition descriptor)
torchgeo-bench run model=torchgeo/earthloc_s2_resnet50
```

## Datasets

### Supported GeoBench V1 Datasets (Classification)

| Dataset         | Classes | Task                          | Prefix |
|-----------------|---------|-------------------------------|--------|
| `m-eurosat`     | 10      | Land cover classification     | `m-`   |
| `m-forestnet`   | 12      | Forest type classification    | `m-`   |
| `m-so2sat`      | 17      | Local climate zones           | `m-`   |
| `m-pv4ger`      | 2       | Photovoltaic detection        | `m-`   |
| `m-brick-kiln`  | 2       | Brick kiln detection          | `m-`   |

### Supported GeoBench V2 Datasets

| Dataset           | Classes | Task            |
|-------------------|---------|-----------------|
| `benv2`           | 19      | Classification  |
| `treesatai`       | 13      | Classification  |
| `so2sat`          | 17      | Classification  |
| `forestnet`       | 12      | Classification  |
| `caffe`           | 4       | Segmentation    |
| `cloudsen12`      | 4       | Segmentation    |
| `burn_scars`      | 3       | Segmentation    |
| `dynamic_earthnet`| 7       | Segmentation    |
| `flair2`          | 13      | Segmentation    |
| `fotw`            | 4       | Segmentation    |
| `kuro_siwo`       | 4       | Segmentation    |
| `pastis`          | 20      | Segmentation    |
| `spacenet2`       | 3       | Segmentation    |
| `spacenet7`       | 3       | Segmentation    |

**Note:** V1 datasets use the `m-` prefix (e.g., `m-eurosat`), while V2 datasets use no prefix (e.g., `forestnet`). The `eurosat` dataset in V2 maps to `benv2`.

### Data Partitions

Control training set size:

```bash
# 1% of training data
torchgeo-bench run dataset.partition=0.01x_train

# Available: 0.01x, 0.02x, 0.05x, 0.10x, 0.20x, 0.50x, 1.00x, default
```

## Configuration

### Hydra Configuration Structure

```
conf/
├── config.yaml          # Main configuration
└── model/
    ├── rcf.yaml         # RCF model config
    └── resnet50.yaml    # ResNet50 config
```

### Key Configuration Options

```yaml
# conf/config.yaml
seed: 0
device: cuda:0
output: torchgeo_bench_results.csv
verbose: false
resume: false             # Skip completed experiments on restart

dataset:
  names: all              # or [m-eurosat, m-forestnet, burn_scars]
  partition: default
  batch_size: 64
  normalization: mean_stdev  # or min_max, percentile_2_98, none
  image_size: 224            # null to preserve native size
  interpolation: bilinear    # bilinear, bicubic, or nearest
  geobench_root: data/classification_v1.0    # V1 dataset location
  geobench_v2_root: data/geobenchv2          # V2 dataset location

eval:
  bootstrap: 500          # CI bootstrap samples
  c_range: [-7, 2, 20]    # LogisticRegression C sweep (log10 scale)
  merge_val: true         # Merge train+val for final linear model
  skip_linear: false      # Skip linear probing (KNN only)

  segmentation:
    head_type: linear     # Segmentation probe head type
    layers: []            # Additional MLP layers (empty = linear)
    lr: 0.001             # Learning rate for segmentation training
    epochs: 1             # Training epochs for segmentation probe
```

### Override Any Config

```bash
# Change device
torchgeo-bench run device=cuda:1

# Adjust bootstrap samples
torchgeo-bench run eval.bootstrap=1000

# Custom output file
torchgeo-bench run output=my_results.csv

# Resume after interruption (skips already-computed results)
torchgeo-bench run resume=true

# Custom data paths
torchgeo-bench run dataset.geobench_root=/path/to/v1 dataset.geobench_v2_root=/path/to/v2

# Configure segmentation evaluation
torchgeo-bench run eval.segmentation.epochs=5 eval.segmentation.lr=0.0001
```

## Evaluation Protocol

### Classification Datasets

For each classification dataset:

1. **Model Initialization**: Instantiate model with dataset's `num_channels`
2. **Feature Extraction**: 
   - Embed train, validation, and test sets
   - Returns (B, K) numpy arrays
3. **KNN-5 Evaluation**:
   - Train on train embeddings
   - Predict on test embeddings
   - Bootstrap predictions 500× for 95% CI
4. **Logistic Regression**:
   - Sweep C ∈ [10^-7, 10^2] (20 values)
   - Validate on validation set, pick best C
   - Retrain on train+val (if `merge_val=true`)
   - Evaluate on test set
   - Bootstrap predictions 500× for 95% CI
5. **Results**: Append two rows (knn5, linear) to CSV

### Segmentation Datasets

For each segmentation dataset:

1. **Model Initialization**: Instantiate model with dataset's `num_channels`
2. **Feature Extraction**: Extract dense feature maps from model
3. **Segmentation Probe Training**:
   - Train a linear or MLP head on training features
   - Uses CrossEntropyLoss with `ignore_index=255` for unlabeled pixels
4. **mIoU Evaluation**:
   - Evaluate on test set using MulticlassJaccardIndex (mIoU)
   - Excludes ignore class from metric computation
5. **Results**: Append one row (seg-linear) to CSV

### Output Format

```csv
dataset,method,metric_name,metric_value,ci_lower,ci_upper,feature_dim,best_c,n_train,n_val,n_test,seed,model,name,normalization,image_size,interpolation,partition
m-eurosat,knn5,accuracy,0.8234,0.8123,0.8345,512,,21600,5400,5400,0,src.bench_models.RCFBench,rcf,mean_stdev,224,bilinear,default
m-eurosat,linear,accuracy,0.8567,0.8461,0.8673,512,0.1,21600,5400,5400,0,src.bench_models.RCFBench,rcf,mean_stdev,224,bilinear,default
burn_scars,seg-linear,mIoU,0.6234,0.0,0.0,768,,1000,200,300,0,src.bench_models.ResNet50Bench,resnet50,mean_stdev,224,bilinear,default
```

### Resume Mode

The `resume=true` option enables safe resumption of interrupted runs:

- Reads existing output CSV to find completed experiments
- Skips (dataset, method, model, config) combinations already present
- Useful for long benchmark sweeps or recovering from crashes
- Works with atomic file appending for multi-process safety

## Development

### Testing

```bash
# Run all tests
pytest

# Test GeoBenchDataset implementation
pytest tests/test_geobench_dataset.py -v

# Compare with reference geobench library (requires geobench package)
pytest tests/test_compare_implementations.py -v

# Test specific dataset
pytest tests/test_geobench_dataset.py -k "m-eurosat" -v

# Quick smoke test of benchmark script
torchgeo-bench run dataset.names=[m-eurosat] eval.bootstrap=10 output=test.csv
```

**Note:** The test suite expects the GeoBench dataset to be available in the default directory `data/`. If your data is located elsewhere, configure the paths in `conf/config.yaml` or use command-line overrides:

```bash
# Command-line override
torchgeo-bench run dataset.geobench_root=/your/path/to/classification_v1.0

# Or set environment variable (for tests)
export GEOBENCH_ROOT=/your/path/to/classification_v1.0
pytest
```

If the dataset is not found, relevant tests will be skipped.

The test suite includes:
- **58 tests** covering all datasets, splits, and normalizations
- **27 comparison tests** validating against reference implementation
- All tests use small `0.01x_train` partitions for fast execution

### Adding a New Model

1. Implement `BenchModel` in `src/bench_models.py` or your own module
2. Create config in `conf/model/yourmodel.yaml`
3. Run: `torchgeo-bench run model=yourmodel`

### Code Standards

- Python 3.12+
- Type hints (modern syntax: `list[str]` not `List[str]`)
- Logging via `logging` module (not `print()`)
- Pydantic v2 if using structured configs

## Citation

If you use this framework, please cite the GeoBench paper:

```bibtex
@article{lacoste2023geobench,
  title={GEO-Bench: Toward Foundation Models for Earth Monitoring},
  author={Lacoste, Alexandre and Lehmann, Nils and ...},
  journal={NeurIPS Datasets and Benchmarks Track},
  year={2023}
}
```

And if using MOSAIKS-style features:

```bibtex
@article{rolf2021mosaiks,
  title={A generalizable and accessible approach to machine learning with global satellite imagery},
  author={Rolf, Esther and Proctor, Jonathan and ...},
  journal={Nature Communications},
  year={2021}
}
```

## License

[Specify license - e.g., MIT, Apache 2.0]

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Troubleshooting

### "Dataset directory not found"

Ensure `GEOBENCH_ROOT` points to `classification_v1.0` directory containing dataset folders (e.g., `m-eurosat/`, `m-forestnet/`).

### "Module 'geobench' not found"

Old dependency. The new implementation uses `GeoBenchDataset` which directly reads HDF5 files. No `geobench` package required.

### CUDA out of memory

Reduce batch size:

```bash
torchgeo-bench run dataset.batch_size=32
```

### Slow dataloader

Adjust number of workers:

```bash
# In src/datasets.py, modify num_workers parameter
# Or add config option (future enhancement)
```

## Roadmap

- [ ] Embedding disk caching
- [x] Segmentation benchmark support
- [ ] Multi-label classification metrics
- [ ] Distributed evaluation (multi-GPU)
- [ ] Automated hyperparameter sweeps
- [ ] Integration with Weights & Biases / MLflow
- [ ] Pre-computed baseline results table
- [ ] Docker container for reproducibility

## Contact

[Add contact information or link to issues page]
