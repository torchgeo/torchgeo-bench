"""Pytest configuration and fixtures for torchgeo-bench tests."""

from pathlib import Path

import pytest

# Datasets always live under ./data/<canonical>/ from the test invocation CWD.
GEOBENCH_ROOT = Path("data/classification_v1.0_wds")
GEOBENCH_V2_ROOT = Path("data/geobenchv2")
EUROSAT_ROOT = Path("data/eurosat")

@pytest.fixture
def geobench_root():
    """Fixture providing the published GeoBench V1 sharded data root."""
    if not (GEOBENCH_ROOT / "m-eurosat").exists():
        pytest.skip(f"GeoBench V1 data not found at {GEOBENCH_ROOT}")
    return str(GEOBENCH_ROOT)


@pytest.fixture
def geobench_v2_root():
    """Fixture providing GeoBench V2 data root path."""
    if not GEOBENCH_V2_ROOT.exists():
        pytest.skip(f"GeoBench V2 data not found at {GEOBENCH_V2_ROOT}")
    return str(GEOBENCH_V2_ROOT)


@pytest.fixture
def small_partition():
    """Fixture providing a small partition name for fast tests."""
    return "0.01x_train"
