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
src/torchgeo_bench/        # Main source package (importable as torchgeo_bench)
  ├── cli.py               # CLI entry point (torchgeo-bench command)
  ├── main.py              # Hydra-decorated benchmark runner
  ├── download.py          # Dataset downloads (geobench_v1/v2 + torchgeo eurosat)
  ├── datasets/            # Per-dataset BenchDataset wrappers + V1/V2 base classes
  ├── linear.py            # Custom LogisticRegression (PyTorch-based)
  ├── knn.py               # FAISS-CPU KNN classifier
  ├── segmentation_task.py # Segmentation task solver
  ├── segmentation_probe.py# Hook-based segmentation probe
  ├── conf/                # Hydra config files (packaged inside the source tree)
  └── models/              # Model implementations (interface.py, bench_models.py, timm.py, etc.)
data/                      # Datasets always live here (relative to CWD)
  ├── classification_v1.0/ # GeoBench V1
  ├── geobenchv2/          # GeoBench V2
  └── eurosat/             # torchgeo EuroSAT
experiments/               # Experiment runners, analysis scripts, SLURM jobs
  ├── scripts/             # Analysis + benchmark scripts (with a slurm/ subdir)
  └── slurm/               # Standalone SLURM batch files
tests/                     # Test suite (pytest)
pyproject.toml             # Project config, dependencies, tool settings
```

## Environment Setup

Two supported workflows — pick **one** (they manage *separate* environments):

```bash
# Option A — uv (the README's canonical path). Creates and manages its own
# .venv and ignores any active conda env. Run tools via `uv run …`:
uv sync --extra dev

# Option B — conda (matches the Makefile). Create the env with `make install`,
# then install editable:
conda activate torchgeo-bench
pip install -e ".[dev]"
```

> Note: `uv sync` always uses its own `.venv`, so a preceding
> `conda activate` does **not** change what `uv sync` installs into.

## Build/Lint/Test Commands

### Running Tests

```bash
pytest                                    # Run the fast suite (excludes `slow`)
pytest tests/test_geobench_dataset.py -v  # Run a SINGLE test file
pytest tests/test_geobench_dataset.py::TestClass::test_method -v  # Single function
pytest -k "m-eurosat" -v                  # Run tests matching a pattern
pytest --no-cov                           # Skip coverage for faster iteration
pytest -m slow                            # Include the slow integration suite
```

The default `addopts` include `-m "not slow"`, so a bare `pytest` runs only the
fast subset; use `-m slow` (or `-m ""` for everything) to run the integration
tests, which load real data and run models.

Tests skip gracefully if GeoBench data is missing — they look for it under
`./data/classification_v1.0`, `./data/geobenchv2`, and `./data/eurosat`.
Note the V1 *slow* tests need the legacy HDF5 bundle from
`torchgeo-bench download geobench_v1`; the single-dataset auto-download writes a
webdataset layout under `./data/classification_v1.0_wds/` that those tests do
**not** read (they will skip).

### Linting and Formatting

```bash
ruff check .           # Check for lint errors
ruff check . --fix     # Auto-fix lint errors
ruff format .          # Format code
```

### Downloading Datasets

```bash
torchgeo-bench download geobench_v1                       # GeoBench V1 -> data/classification_v1.0/
torchgeo-bench download geobench_v2                       # all benchmark V2 datasets -> data/geobenchv2/<name>
torchgeo-bench download geobench_v2 --datasets benv2,burn_scars  # subset
torchgeo-bench download eurosat                           # torchgeo EuroSAT -> data/eurosat
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
```

## Datasets

All datasets are loaded from `./data/<canonical-subdir>` relative to the
current working directory (no env vars, no overrides — keep it simple).

### GeoBench V1 (Classification) - use `m-` prefix
`m-eurosat`, `m-forestnet`, `m-so2sat`, `m-pv4ger`, `m-brick-kiln`, `m-bigearthnet`

### GeoBench V2 (Classification)
`benv2`, `treesatai`, `so2sat`, `forestnet`

### GeoBench V2 (Segmentation)
`burn_scars`, `caffe`, `cloudsen12`, `dynamic_earthnet`, `flair2`, `fotw`, `kuro_siwo`, `pastis`, `spacenet2`, `spacenet7`

### torchgeo template
`eurosat` (loads via `torchgeo.datasets.EuroSAT`)

**Note:** V1 datasets use the `m-` prefix (e.g., `m-eurosat`), V2 datasets use no prefix.

## Code Style Guidelines

### Python Version and Type Hints

- **Python 3.12+** (targeting 3.12)
- Use modern type hints: `list[str]`, `dict[str, Any]`, `X | None`
- Do NOT use deprecated typing imports: `List`, `Dict`, `Optional`, `Union`
- Do NOT use `from __future__ import annotations`; use `Self`, quoted annotations, or explicit imports for forward references

### Import Ordering

```python
import logging                          # 1. Standard library
from dataclasses import dataclass

import numpy as np                      # 2. Third-party
import torch

from torchgeo_bench.datasets import get_datasets   # 3. Local imports

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

### No defensive imports or bare-`Exception` catches

**Do NOT write fallback `try`/`except ImportError` blocks for hard dependencies.**
Every package listed under `[project.dependencies]` in `pyproject.toml` is
guaranteed to be installed; pretending otherwise just papers over real
breakage and forces every reader to mentally evaluate the fallback path.

```python
# ❌ BAD: hides the real failure mode behind a fake fallback
try:
    from torchgeo.datasets import DatasetNotFoundError
except ImportError:  # pragma: no cover - older torchgeo versions
    DatasetNotFoundError = FileNotFoundError

# ✅ GOOD: torchgeo is a hard dep, just import it
from torchgeo.datasets import DatasetNotFoundError
```

The same rule applies to bare `except Exception:` blocks that swallow errors
to "keep going". If you want to skip a single iteration in a sweep, catch
the *specific* exception you expect (e.g. `FileNotFoundError`,
`DatasetNotFoundError`, `pandas.errors.ParserError`). Letting unexpected
failures propagate is a feature, not a bug.

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
        bench = get_bench_dataset_class("m-eurosat")()
        dataset = bench.get_dataset("train", bands=tuple(bench.rgb_bands))
        assert len(dataset) > 0

    @pytest.mark.parametrize("dataset_name", ["m-eurosat", "m-forestnet"])
    def test_dataset_loads(self, geobench_root, dataset_name):
        bench = get_bench_dataset_class(dataset_name)()
        dataset = bench.get_dataset("train", bands=tuple(bench.rgb_bands))
        assert len(dataset) > 0, f"{dataset_name} has no samples"
```

## Key Dependencies

`torch>=2.0`, `torchvision>=0.15`, `numpy>=1.24`, `scikit-learn>=1.3`, `hydra-core>=1.3`, `timm>=0.9`, `torchgeo>=0.8`, `h5py>=3.8`, `faiss-cpu>=1.7`, `huggingface-hub>=0.20`, `geobenchv2>=0.9`

## Common Gotchas

1. **Data lives at `data/`**: Always `data/<canonical-subdir>/` from CWD. No env vars, no overrides.
2. **No documentation for refactoring**: Don't create docs for internal refactors.
3. **Tests need data**: Tests skip if `data/classification_v1.0` / `data/geobenchv2` / `data/eurosat` aren't on disk.
4. **Hydra outputs**: Benchmark runs create `outputs/` directory with logs.
5. **Model reinitialization**: Models are reinitialized per-dataset to handle varying input channels.
6. **V1 vs V2 datasets**: V1 uses `m-` prefix, V2 uses no prefix.

## Copilot/Cursor Instructions

From `.github/copilot-instructions.md` (applies to `**/*.py, **/*.ipynb`):
- Always `conda activate torchgeo-bench` before running commands
- Assume Python 3.12+ and Pydantic v2.0
- Prefer modern type hints (`list[str]` not `List[str]`)
- Use `logging` for logging, not `print()`
- Don't create documentation for refactoring
