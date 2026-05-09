"""Pytest configuration and fixtures for torchgeo-bench tests."""

import os
from pathlib import Path

import pytest

# Datasets always live under ./data/<canonical>/ from the test invocation CWD.
GEOBENCH_ROOT = Path("data/classification_v1.0")
GEOBENCH_V2_ROOT = Path("data/geobenchv2")
EUROSAT_ROOT = Path("data/eurosat")

# Tests rely on the dataset-not-on-disk path raising FileNotFoundError so the
# test-skip branch fires.  The V1 loader otherwise auto-downloads the public
# WebDataset mirror — which would force CI to pull tens of GBs and time out.
os.environ.setdefault("GEOBENCH_V1_NO_HF_DOWNLOAD", "1")


@pytest.fixture
def geobench_root():
    """Fixture providing GeoBench V1 data root path."""
    if not GEOBENCH_ROOT.exists():
        pytest.skip(f"GeoBench V1 data not found at {GEOBENCH_ROOT}")
    return str(GEOBENCH_ROOT)


@pytest.fixture
def geobench_v2_root():
    """Fixture providing GeoBench V2 data root path."""
    if not GEOBENCH_V2_ROOT.exists():
        pytest.skip(f"GeoBench V2 data not found at {GEOBENCH_V2_ROOT}")
    return str(GEOBENCH_V2_ROOT)


@pytest.fixture
def eurosat_root():
    """Fixture providing the torchgeo EuroSAT data root path."""
    if not EUROSAT_ROOT.exists():
        pytest.skip(f"EuroSAT (torchgeo) data not found at {EUROSAT_ROOT}")
    return str(EUROSAT_ROOT)


@pytest.fixture
def all_datasets():
    """Fixture providing list of all available dataset names."""
    return [
        "m-eurosat",
        "m-forestnet",
        "m-so2sat",
        "m-pv4ger",
        "m-brick-kiln",
    ]


@pytest.fixture
def small_partition():
    """Fixture providing a small partition name for fast tests."""
    return "0.01x_train"


@pytest.fixture
def all_splits():
    """Fixture providing all split names."""
    return ["train", "valid", "test"]
