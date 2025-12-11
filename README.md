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
2. **Automated Evaluation**: KNN-5 and Linear Probing with bootstrapped confidence intervals
3. **GeoBench Integration**: Direct access to classification benchmark datasets
4. **Hydra Configuration**: Flexible experiment configuration without code changes
5. **Efficient Workflows**: Per-dataset model reinitialization handles varying input channels

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

# Use pretrained ResNet50
torchgeo-bench run model=resnet50

# Benchmark on specific datasets with verbose output
torchgeo-bench run dataset.names=[m-eurosat,m-forestnet] verbose=true

# Quick evaluation (skip linear probing, minimal bootstrap)
torchgeo-bench run eval.skip_linear=true eval.bootstrap=100

# Use smaller training partition
torchgeo-bench run dataset.partition=0.01x_train output=results_1pct.csv
```

Alternatively, you can still use the standalone script with Hydra directly:

```bash
python torchgeo_bench.py model=resnet50
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
torchgeo-bench run model=resnet50
```

### Vision Transformers (ViT / DeiT / Swin)
Configs generated under `conf/model/vit/` (see `create_vit_configs.py`). Vision backbones often expect a fixed spatial resolution (e.g., 224×224). You can now control resizing globally via dataset config:

```bash
# Resize all dataset tiles to 224 (bicubic by default)
torchgeo-bench run model=vit/vit_base_patch16_224 dataset.image_size=224

# Use bilinear interpolation
torchgeo-bench run model=vit/vit_base_patch16_224 dataset.image_size=224 dataset.interpolation=bilinear
```

If you omit `dataset.image_size`, native tile sizes are preserved. Model-level `auto_resize` remains available as a fallback but dataset-level resizing is preferred for consistency across models.

Examples:

```bash
torchgeo-bench run model=vit/vit_base_patch16_224
torchgeo-bench run model=vit/deit_small_patch16_224 dataset.names=[m-eurosat]
torchgeo-bench run model=vit/swin_base_patch4_window7_224 eval.skip_linear=true
```

To study scale effects without resizing, simply avoid setting `dataset.image_size` and (optionally) disable the model fallback:

```bash
torchgeo-bench run model=vit/vit_base_patch16_224 model.auto_resize=false
```

### Custom Wrappers
See `src/bench_models.py` for examples. You can wrap any existing model (timm, torchgeo, transformers) by implementing the interface.

## Datasets

### Supported GeoBench Datasets

| Dataset         | Classes | Task                          | Samples (default) |
|-----------------|---------|-------------------------------|-------------------|
| `m-eurosat`     | 10      | Land cover classification     | ~27,000           |
| `m-forestnet`   | 12      | Forest type classification    | ~500,000          |
| `m-so2sat`      | 17      | Local climate zones           | ~400,000          |
| `m-pv4ger`      | 2       | Photovoltaic detection        | ~100,000          |
| `m-brick-kiln`  | 2       | Brick kiln detection          | ~100,000          |

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

dataset:
  names: all  # or [m-eurosat, m-forestnet]
  partition: default
  batch_size: 64
  normalization: mean_stdev  # or min_max, percentile_2_98, none

eval:
  bootstrap: 500        # CI bootstrap samples
  c_range: [-7, 2, 20]  # LogisticRegression C sweep (log10 scale)
  merge_val: true       # Merge train+val for final linear model
  skip_linear: false    # Skip linear probing (KNN only)
```

### Override Any Config

```bash
# Change device
torchgeo-bench run device=cuda:1

# Adjust bootstrap samples
torchgeo-bench run eval.bootstrap=1000

# Custom output file
torchgeo-bench run output=my_results.csv
```

## Evaluation Protocol

For each dataset:

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

### Output Format

```csv
dataset,method,accuracy,ci_lower,ci_upper,feature_dim,best_c,n_train,n_val,n_test,seed,model
m-eurosat,knn5,0.8234,0.8123,0.8345,512,,21600,5400,5400,0,src.bench_models.RCFBench
m-eurosat,linear,0.8567,0.8461,0.8673,512,0.1,21600,5400,5400,0,src.bench_models.RCFBench
```

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

**Note:** The test suite expects the GeoBench dataset to be available in the default directory `data/`. If your data is located elsewhere, set the environment variable `GEOBENCH_ROOT` to the full path of your dataset root before running tests:

```bash
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
- [ ] Segmentation benchmark support
- [ ] Multi-label classification metrics
- [ ] Distributed evaluation (multi-GPU)
- [ ] Automated hyperparameter sweeps
- [ ] Integration with Weights & Biases / MLflow
- [ ] Pre-computed baseline results table
- [ ] Docker container for reproducibility

## Contact

[Add contact information or link to issues page]
