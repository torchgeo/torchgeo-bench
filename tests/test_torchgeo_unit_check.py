"""Tests for torchgeo pretrained input-unit plausibility checks."""

import warnings

import pytest

from torchgeo_bench.datasets.m_eurosat import MEurosat
from torchgeo_bench.datasets.m_so2sat import MSo2Sat
from torchgeo_bench.models.torchgeo_models import _warn_unit_mismatch


def _bands(cls, names: tuple[str, ...]):
    by_name = {b.name: b for b in cls.bands}
    return [by_name[name] for name in names]


def test_s2_dn_weights_warn_on_reflectance_dataset() -> None:
    """Reflectance-scale So2Sat must not pass as raw Sentinel-2 DN."""
    bands = _bands(MSo2Sat, ("red", "green", "blue"))
    with pytest.warns(UserWarning, match="look like reflectance_0_1"):
        _warn_unit_mismatch("Demo", "s2_dn_div10000", bands, "warn")


def test_s2_dn_weights_accept_raw_s2_dataset() -> None:
    """Raw Sentinel-2 DN datasets remain accepted for /10000 torchgeo weights."""
    bands = _bands(MEurosat, ("red", "green", "blue"))
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        _warn_unit_mismatch("Demo", "s2_dn_div10000", bands, "warn")
    assert records == []


def test_unit_mismatch_error_mode_raises() -> None:
    """input_unit_check=error turns the plausibility warning into a hard failure."""
    bands = _bands(MSo2Sat, ("red", "green", "blue"))
    with pytest.raises(RuntimeError, match="look like reflectance_0_1"):
        _warn_unit_mismatch("Demo", "s2_dn_div10000", bands, "error")
