# AGENTS.md

Guidelines for AI coding agents working in the torchgeo-bench repository.

## Project Overview

**torchgeo-bench** is a Python benchmarking framework for evaluating geospatial foundation models on GeoBench datasets (V1 and V2) and location encoders on CoordBench. It uses PyTorch and strict Pydantic configuration with explicit CLI flags and YAML, and provides KNN-5, linear probing, and segmentation (mIoU) evaluation with bootstrapped confidence intervals. Legacy `key=value` and `+key=value` overrides are rejected.

### Key Features
- **Resume Mode**: Skip already-computed experiments when interrupted/restarted
- **Atomic CSV Writes**: Results appended with file locking for parallel job safety
- **GeoBench V1 & V2**: Classification and segmentation benchmark datasets

### Key Directories

```
src/torchgeo_bench/        # Main source package (importable as torchgeo_bench)
  ├── cli.py               # Unified CLI: run/models/datasets/download/profile/flops/coord
  ├── main.py              # Benchmark runner (classification + segmentation)
  ├── config_schema.py     # Strict image RunConfig and safe YAML loading
  ├── presets.py           # ModelPreset resolution + explicit build_model construction
  ├── profile_config.py    # Real-batch standalone profile settings
  ├── flops_config.py      # Synthetic compute settings
  ├── coordbench/          # Location encoders, typed config, probes, and runner
  ├── resume.py            # config_hash-based resume/skip logic
  ├── results.py            # EvaluationResult schema + atomic per-model CSV writes
  ├── download.py          # Dataset downloads (geobench_v1/v2 + torchgeo eurosat)
  ├── datasets/            # Per-dataset BenchDataset wrappers + V1/V2 base classes
  ├── linear.py            # Custom LogisticRegression (PyTorch-based)
  ├── knn.py               # FAISS-backed KNN via faissknn
  ├── segmentation_task.py # Segmentation task solver
  ├── segmentation_probe.py# Hook-based segmentation probe
  ├── conf/                # Config YAMLs (packaged inside the source tree)
  └── models/              # Model implementations (interface.py, timm.py, torchgeo_models.py, etc.)
data/                      # Datasets always live here (relative to CWD)
  ├── classification_v1.0_wds/ # GeoBench V1 JSON-metadata shards
  ├── classification_v1.0/ # Custom V1 HDF5 with JSON metadata
  ├── geobenchv2/          # GeoBench V2
  └── eurosat/             # torchgeo EuroSAT
experiments/               # Experiment runners, analysis scripts, SLURM jobs
  ├── scripts/             # Analysis + benchmark scripts (with a slurm/ subdir)
  └── slurm/               # Standalone SLURM batch files
tests/                     # Test suite (pytest)
pyproject.toml             # Project config, dependencies, tool settings
```

## Contributing: Commits, PRs, and Splitting Changes

Most contributors here are driving an AI coding agent and may not write Python
day to day themselves — that's fine, but it means the PR is often the only
thing a human reviewer has to go on. Optimize for a reviewer who wants to
understand *and approve* a change in one pass, not one who has to reconstruct
what happened from a wall of text or untangle five unrelated edits from one
diff.

**One PR per logical change, each with its own tests.**
If an agent's exploration turns up more than one fix, bug, or improvement,
that's multiple PRs, not one. Bundling unrelated changes together forces a
reviewer to approve or block all of them as a set, and it flattens the
changelog — a small, real bug fix deserves its own commit/PR so it shows up
as its own trackable entry in history, not a buried line in something else's
diff. If change B only makes sense on top of change A (e.g. B extends a
function A just added), stack B's PR on A's branch instead of merging them
into one.

**Commit messages: one Conventional Commits subject line, nothing else.**
`fix(models): handle missing coastal band` — no body, no bullet list, no
restating the diff. Nobody reads commit bodies; the PR description is where
context goes.

**PR descriptions: say what changed and why, in a few plain sentences.**
- No "Test plan" / "How I tested this" section unless the reviewer asks.
- No AI-assistance disclosure boilerplate.
- Don't restate the diff line by line — the diff is right there.
- Skip it if the code and a one-line summary already make the change obvious.

**Every PR should be independently reviewable.**
A reviewer should be able to read the diff top to bottom, understand the
change, check the tests cover it, and approve — without needing chat history,
a linked doc, or a walkthrough. If a change needs more than a few sentences
to explain *why* it's safe, that's a sign it should be split further or the
code needs a comment (see below), not that the PR description needs to be
longer.

**Minimal code comments.**
Don't comment what the code already says. A comment earns its place only
when it captures something the code can't: a non-obvious constraint, a
citation for a magic number, a workaround for a specific upstream bug, or a
"why not the obvious alternative" note. When in doubt, leave it out.

```python
# BAD: restates the line below
# increment the counter
count += 1

# GOOD: explains a non-obvious constraint
# Rounding avoids the downward bias from truncating positive wavelengths.
wavelength = int(round(wavelength_um))
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

Tests skip gracefully if GeoBench data is missing. V1 slow tests use the JSON shards under `./data/classification_v1.0_wds`; V2 and EuroSAT use `./data/geobenchv2` and `./data/eurosat`. Present legacy pickle caches must be replaced, not skipped or unpickled.

### Linting and Formatting

```bash
ruff check .           # Check for lint errors
ruff check . --fix     # Auto-fix lint errors
ruff format .          # Format code
```

### Downloading Datasets

```bash
torchgeo-bench download geobench_v1                       # GeoBench V1 -> data/classification_v1.0_wds/
torchgeo-bench download geobench_v2                       # all benchmark V2 datasets -> data/geobenchv2/<name>
torchgeo-bench download geobench_v2 --datasets benv2,burn_scars  # subset
torchgeo-bench download eurosat                           # torchgeo EuroSAT -> data/eurosat
torchgeo-bench download resisc45                          # torchgeo RESISC45 -> data/resisc45
```

### Running the Benchmark

```bash
# Basic usage
torchgeo-bench run --model timm/resnet50 --dataset m-eurosat

# Quick eval (skip linear probing, minimal bootstrap)
torchgeo-bench run --model rcf --dataset m-eurosat --methods knn --bootstrap-samples 100

# Linear-only evaluation
torchgeo-bench run --model rcf --dataset m-eurosat --methods linear

# Resume a previously interrupted run (skips completed experiments)
torchgeo-bench run --config examples/image-run.yaml --resume

# Evaluate segmentation datasets (V2)
torchgeo-bench run --model timm/resnet50 --dataset burn_scars --dataset pastis --dataset flair2

# Select specific GPU device
torchgeo-bench run --model rcf --dataset m-eurosat --device cuda:1

# Measure per-sample compute cost (GFLOPs, params, throughput) -> results/compute_cost.csv
torchgeo-bench flops --model timm/resnet50

# Measure a fixed real batch -> JSON stdout
torchgeo-bench profile --model rcf --dataset m-eurosat --device cpu

# Coordinate-only linear probe
torchgeo-bench coord --model sincos --dataset california_housing --methods linear
```

## Configuration Architecture

- The installed CLI, `python -m torchgeo_bench`, and
  `python -m torchgeo_bench.cli` dispatch the same commands. Do not reintroduce
  a legacy override parser or a second configuration engine.
- `config_schema.py` defines strict Pydantic image settings and
  `load_run_config`. Unknown fields, duplicate YAML keys, and wrong types
  are errors. Profile, FLOPs, and CoordBench have separate typed schemas.
- A run selects `model: {name: rcf}` or a custom
  `model: {name: my-model, target: my_package.MyModel, kwargs: {...}}`.
  Presets in `conf/model/` use `name`, `target`, `track`, `seed_from_run`,
  `kwargs`, `input`, `classification`, `segmentation`, and
  `dataset_overrides`. Metadata is never a constructor kwarg.
- Preserve precedence: built-in defaults < model preset < preset's dataset
  defaults < explicit YAML < explicit flags. Use unset-aware serialization;
  explicit `false`, `null`, and `[]` must not be replaced by defaults.
- `resolve_run_config` resolves effective settings per dataset. `build_model`
  receives runtime `BandSpec` objects and empirical-RCF datasets explicitly;
  never serialize them into YAML. Existing timm/RCF model dataclasses remain
  construction boundaries, not a competing configuration system.
- Optional model dependencies stay lazy. Catalogs, schemas, and dry runs
  must not load weights or dataset samples.

## Results Layout & Resume

- Each model writes to its own `results/models/<model name>.csv` (not one
  shared file), so re-running one model only touches that file. Rows are
  appended, never rewritten in place.
- `--resume` skips a (dataset, method, bands, normalization, ...) combo
  only if an existing row's `config_hash` matches the current run's config.
  Changing a hashed config field invalidates the match. Additive `profile`
  and `intrinsic_dim` settings do not invalidate equivalent classification
  results; their own metric completeness is checked separately.
- One-time, hardware-dependent measurements (`torchgeo-bench flops`,
  intrinsic-dimension probes) live in their own side files —
  `results/compute_cost.csv`, `results/profiles/<model name>.csv`, and
  `results/intrinsic_dim/<model name>.csv` —
  so a routine metrics rerun doesn't touch them.
- `--output` / `output.file` combines selected image-run measurements in one
  CSV; otherwise `output.directory`, `output.profile_directory`, and
  `output.intrinsic_dim_directory` keep their respective defaults. Standalone
  `profile` writes JSON stdout, and `coord` has its own CSV schema/resume key.
- Don't hand-edit `config_hash`/`KEY_COLS` logic without checking
  `resume.py`'s docstring first; it's easy to silently invalidate every
  existing row across `results/models/`.

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
def _forward_patch_features(
    self,
    images: torch.Tensor,
) -> torch.Tensor:
    """Return a batch of vector embeddings (B, K).

    Args:
        images: Normalized input images, shape (B, C, H, W).

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
    def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
```

Implement `_forward_patch_features`, not the public `forward_patch_features`:
the public method applies the configured normalization before calling the hook.

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

Core (see `pyproject.toml` for the authoritative list): `torch>=2`, `torchvision>=0.15`,
`numpy>=1.24`, `scikit-learn>=1.3`, `timm>=0.9`, `torchgeo>=0.9`, `torchmetrics>=1.4`,
`pydantic>=2`, `pyyaml>=6`, `h5py>=3.8`, `faissknn` (CPU or CUDA variant, picked by platform),
`huggingface-hub>=0.20`, `geobenchv2>=0.9`, `pandas>=2`, `pyarrow>=14`, `safetensors>=0.4`,
`filelock>=3.12`, `rich>=13`.

Optional extras (`pip install 'torchgeo-bench[extra]'`, or `[all]` for everything):
`coordbench`, `dev`, `docs`, `id` (intrinsic-dimension estimators), `olmoearth`, `sam3`, `terratorch`. Model wrappers behind an extra (OlmoEarth, SAM3, terratorch-backed models) import that dependency lazily — don't add a top-level import for one at module scope.

## Common Gotchas

1. **Data lives at `data/`**: Always `data/<canonical-subdir>/` from CWD. No env vars, no overrides. V1 uses the pinned `calebrob6/geobenchv1-webdataset` JSON mirror with archive SHA-256 checks. Pickle metadata is not supported.
2. **No documentation for refactoring**: Don't create docs for internal refactors.
3. **Tests need data**: Tests skip if `data/classification_v1.0_wds` / `data/geobenchv2` / `data/eurosat` aren't on disk.
4. **Model reinitialization**: Models are reinitialized per-dataset to handle varying input channels.
5. **V1 vs V2 datasets**: V1 uses `m-` prefix, V2 uses no prefix.

## Copilot/Cursor Instructions

`.github/copilot-instructions.md` covers the same ground in more depth
(source layout, build/test/lint, architecture notes, conventions) — read it
directly for anything not covered here rather than relying on a summary.
