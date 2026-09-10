"""Pytest configuration and fixtures for torchgeo-bench tests."""

from pathlib import Path

import pytest

# Dataset paths are relative to the directory where pytest is run.
GEOBENCH_ROOT = Path("data/classification_v1.0_wds")
GEOBENCH_V2_ROOT = Path("data/geobenchv2")
EUROSAT_ROOT = Path("data/eurosat")


@pytest.fixture
def geobench_root():
    if not (GEOBENCH_ROOT / "m-eurosat").exists():
        pytest.skip(f"GeoBench V1 data not found at {GEOBENCH_ROOT}")
    return str(GEOBENCH_ROOT)


@pytest.fixture
def geobench_v2_root():
    if not GEOBENCH_V2_ROOT.exists():
        pytest.skip(f"GeoBench V2 data not found at {GEOBENCH_V2_ROOT}")
    return str(GEOBENCH_V2_ROOT)


@pytest.fixture
def small_partition():
    return "0.01x_train"
