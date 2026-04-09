# AGENTS.md

Guidelines for AI coding agents working in the torchgeo-bench repository.

## Project Overview

**torchgeo-bench** is a Python benchmarking framework for evaluating geospatial foundation models on GeoBench datasets (V1 and V2). Uses PyTorch, Hydra configuration, and provides KNN-5, Linear Probing, and Segmentation (mIoU) evaluation with bootstrapped confidence intervals.

### Key Features
- **Resume Mode**: Skip already-computed experiments when interrupted/restarted
- **Atomic CSV Writes**: Results appended with file locking for parallel job safety
- **GeoBench V1 & V2**: Classification and segmentation benchmark datasets

### Key Directories

```
src/                  # Main source package
  ├── cli.py          # CLI entry point (torchgeo-bench command)
  ├── datasets.py     # Dataset loading utilities (V1 and V2)
  ├── geobench_dataset.py  # PyTorch Dataset for GeoBench V1
  ├── linear.py       # Custom LogisticRegression (PyTorch-based)
  ├── segmentation_task.py  # Segmentation task solver
  └── models/         # Model implementations (interface.py, bench_models.py, timm.py)
conf/                 # Hydra configuration files (config.yaml, model/*.yaml)
scripts/              # Analysis and benchmark scripts
tests/                # Test suite (pytest)
pyproject.toml        # Project config, dependencies, tool settings
```

## Environment Setup

```bash
conda activate torchgeo-bench   # Always activate before running commands
pip install -e ".[dev]"         # Or install with dev dependencies
```

## Build/Lint/Test Commands

### Running Tests

```bash
pytest                                    # Run ALL tests
pytest tests/test_geobench_dataset.py -v  # Run a SINGLE test file
pytest tests/test_geobench_dataset.py::TestClass::test_method -v  # Single function
pytest -k "m-eurosat" -v                  # Run tests matching a pattern
pytest --no-cov                           # Skip coverage for faster iteration
```

Tests require GeoBench data. Set `GEOBENCH_ROOT` (V1) or `GEOBENCH_V2_ROOT` (V2) if data is not in default locations.

### Linting and Formatting

```bash
ruff check .           # Check for lint errors
ruff check . --fix     # Auto-fix lint errors
ruff format .          # Format code
```

### Running the Benchmark

```bash
# Basic usage
torchgeo-bench run model=timm/resnet50 dataset.names=[m-eurosat]

# Quick eval (skip linear probing, minimal bootstrap)
torchgeo-bench run eval.skip_linear=true eval.bootstrap=100

# Resume a previously interrupted run (skips completed experiments)
torchgeo-bench run resume=true

# Evaluate segmentation datasets (V2)
torchgeo-bench run dataset.names=[burn_scars,pastis,flair2]

# Select specific GPU device
torchgeo-bench run device=cuda:1

# Direct Hydra invocation
python torchgeo_bench.py model=timm/resnet50
```

## Datasets

### GeoBench V1 (Classification) - use `m-` prefix
`m-eurosat`, `m-forestnet`, `m-so2sat`, `m-pv4ger`, `m-brick-kiln`

### GeoBench V2 (Classification)
`benv2`, `treesatai`, `so2sat`, `forestnet`

### GeoBench V2 (Segmentation)
`caffe`, `cloudsen12`, `burn_scars`, `dynamic_earthnet`, `flair2`, `fotw`, `kuro_siwo`, `pastis`, `spacenet2`, `spacenet7`

**Note:** V1 datasets use the `m-` prefix (e.g., `m-eurosat`), V2 datasets use no prefix.

## Code Style Guidelines

### Python Version and Type Hints

- **Python 3.11+** (targeting 3.12)
- Use modern type hints: `list[str]`, `dict[str, Any]`, `X | None`
- Do NOT use deprecated typing imports: `List`, `Dict`, `Optional`, `Union`
- Use `from __future__ import annotations` for forward references

### Import Ordering

```python
from __future__ import annotations

import logging                          # 1. Standard library
from dataclasses import dataclass

import numpy as np                      # 2. Third-party
import torch

from src.datasets import get_datasets   # 3. Local imports

logger = logging.getLogger(__name__)
```

### Naming Conventions

| Type | Convention | Example |
|------|------------|---------|
| Variables/functions | `snake_case` | `get_datasets`, `embed_split` |
| Classes | `PascalCase` | `GeoBenchDataset`, `BenchModel` |
| Constants | `SCREAMING_SNAKE_CASE` | `NUM_CLASSES_PER_DATASET` |
| Private methods | `_leading_underscore` | `_load_sample_metadata` |

### Type Annotations

Always annotate function signatures:

```python
def get_datasets(
    dataset_name: str = "m-forestnet",
    partition_name: str = "default",
    batch_size: int = 32,
    geobench_root: str | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    ...
```

### Documentation Style (Google-style)

```python
def forward_patch_features(
    self,
    images: torch.Tensor,
    bboxes: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return a batch of vector embeddings (B, K).

    Args:
        images: Input images, shape (B, C, H, W).
        bboxes: Optional bounding boxes, shape (B, 4).

    Returns:
        Embeddings tensor of shape (B, K).
    """
```

### Error Handling and Logging

Use explicit exceptions with descriptive messages. Use `logging`, NOT `print()`:

```python
logger = logging.getLogger(__name__)
logger.info("Processing dataset: %s", dataset_name)

if not self.dataset_dir.exists():
    raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")
```

### Class Patterns

```python
@dataclass
class BandStats:
    mean: list[float]
    std: list[float]

class BenchModel(nn.Module, ABC):
    @abstractmethod
    def forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
```

## Ruff Configuration

From `pyproject.toml`:
- **Line length:** 100 characters
- **Target:** Python 3.12
- **Enabled rules:** E, W, F, I (isort), B (bugbear), C4, UP, ARG, SIM
- **Ignored:** E501 (line too long), B008 (function calls in defaults), B905 (zip strict)

## Testing Patterns

```python
class TestGeoBenchDatasetBasics:
    def test_dataset_initialization(self, geobench_root):
        dataset = GeoBenchDataset(...)
        assert len(dataset) > 0

    @pytest.mark.parametrize("dataset_name", ["m-eurosat", "m-forestnet"])
    def test_dataset_loads(self, geobench_root, dataset_name):
        dataset = GeoBenchDataset(dataset_name=dataset_name, ...)
        assert len(dataset) > 0, f"{dataset_name} has no samples"
```

## Key Dependencies

`torch>=2.0`, `torchvision>=0.15`, `numpy>=1.24`, `scikit-learn>=1.3`, `hydra-core>=1.3`, `timm>=0.9`, `torchgeo>=0.8`, `h5py>=3.8`, `faissknn>=0.0.2`

## Common Gotchas

1. **Pydantic v2**: Use v2 syntax if using Pydantic for configs
2. **No documentation for refactoring**: Don't create docs for internal refactors
3. **Tests need data**: Tests skip if `GEOBENCH_ROOT` / `GEOBENCH_V2_ROOT` not set
4. **Hydra outputs**: Benchmark runs create `outputs/` directory with logs
5. **Model reinitialization**: Models are reinitialized per-dataset to handle varying input channels
6. **V1 vs V2 datasets**: V1 uses `m-` prefix, V2 uses no prefix

## Copilot/Cursor Instructions

From `.github/copilot-instructions.md` (applies to `**/*.py, **/*.ipynb`):
- Always `conda activate torchgeo-bench` before running commands
- Assume Python 3.12+ and Pydantic v2.0
- Prefer modern type hints (`list[str]` not `List[str]`)
- Use `logging` for logging, not `print()`
- Don't create documentation for refactoring
