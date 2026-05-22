"""Tests for _input_units: detect_input_unit, convert_unit, and helpers."""

import pytest
import torch

from torchgeo_bench.datasets.base import BandSpec
from torchgeo_bench.models._input_units import (
    InputUnit,
    convert_unit,
    detect_input_unit,
    to_reflectance,
    to_s2_dn,
    to_uint8,
)


def _band(max_val: float, sensor: str = "s2") -> BandSpec:
    return BandSpec(sensor=sensor, name="b", source_name="B", mean=max_val / 2, std=max_val / 4, min=0.0, max=max_val)


# ---------------------------------------------------------------------------
# detect_input_unit
# ---------------------------------------------------------------------------


def test_detect_s2_dn():
    assert detect_input_unit([_band(10000.0)]) == InputUnit.S2_DN


def test_detect_uint8():
    assert detect_input_unit([_band(255.0)]) == InputUnit.UINT8


def test_detect_reflectance():
    assert detect_input_unit([_band(1.0)]) == InputUnit.REFLECTANCE_0_1


def test_detect_mixed_sensors_raises():
    bands = [_band(10000.0, sensor="s2"), _band(1.0, sensor="aerial")]
    with pytest.raises(ValueError, match="Cannot infer one input unit"):
        detect_input_unit(bands)


# ---------------------------------------------------------------------------
# to_s2_dn
# ---------------------------------------------------------------------------


def test_to_s2_dn_from_s2_dn_noop():
    x = torch.tensor([5000.0])
    assert torch.allclose(to_s2_dn(x, InputUnit.S2_DN), x)


def test_to_s2_dn_from_reflectance():
    x = torch.tensor([0.5])
    result = to_s2_dn(x, InputUnit.REFLECTANCE_0_1)
    assert torch.allclose(result, torch.tensor([5000.0]))


def test_to_s2_dn_from_uint8():
    x = torch.tensor([255.0])
    result = to_s2_dn(x, InputUnit.UINT8)
    assert torch.allclose(result, torch.tensor([10000.0]))


# ---------------------------------------------------------------------------
# to_reflectance
# ---------------------------------------------------------------------------


def test_to_reflectance_from_reflectance_noop():
    x = torch.tensor([0.3])
    assert torch.allclose(to_reflectance(x, InputUnit.REFLECTANCE_0_1), x)


def test_to_reflectance_from_s2_dn():
    x = torch.tensor([10000.0])
    result = to_reflectance(x, InputUnit.S2_DN)
    assert torch.allclose(result, torch.tensor([1.0]))


def test_to_reflectance_from_uint8():
    x = torch.tensor([255.0])
    result = to_reflectance(x, InputUnit.UINT8)
    assert torch.allclose(result, torch.tensor([1.0]))


# ---------------------------------------------------------------------------
# to_uint8
# ---------------------------------------------------------------------------


def test_to_uint8_from_uint8_noop():
    x = torch.tensor([128.0])
    assert torch.allclose(to_uint8(x, InputUnit.UINT8), x)


def test_to_uint8_from_reflectance():
    x = torch.tensor([1.0])
    result = to_uint8(x, InputUnit.REFLECTANCE_0_1)
    assert torch.allclose(result, torch.tensor([255.0]))


def test_to_uint8_from_s2_dn():
    x = torch.tensor([10000.0])
    result = to_uint8(x, InputUnit.S2_DN)
    assert torch.allclose(result, torch.tensor([255.0]))


# ---------------------------------------------------------------------------
# convert_unit
# ---------------------------------------------------------------------------


def test_convert_unit_noop_same_src_dst():
    x = torch.tensor([500.0])
    assert torch.allclose(convert_unit(x, InputUnit.S2_DN, InputUnit.S2_DN), x)


def test_convert_unit_s2dn_to_reflectance():
    x = torch.tensor([10000.0])
    assert torch.allclose(convert_unit(x, InputUnit.S2_DN, InputUnit.REFLECTANCE_0_1), torch.tensor([1.0]))


def test_convert_unit_reflectance_to_s2dn():
    x = torch.tensor([0.5])
    assert torch.allclose(convert_unit(x, InputUnit.REFLECTANCE_0_1, InputUnit.S2_DN), torch.tensor([5000.0]))


def test_convert_unit_reflectance_to_uint8():
    x = torch.tensor([0.5])
    assert torch.allclose(convert_unit(x, InputUnit.REFLECTANCE_0_1, InputUnit.UINT8), torch.tensor([127.5]))


def test_convert_unit_s2dn_to_uint8():
    x = torch.tensor([10000.0])
    result = convert_unit(x, InputUnit.S2_DN, InputUnit.UINT8)
    assert torch.allclose(result, torch.tensor([255.0]))


def test_convert_unit_unknown_dst_raises():
    x = torch.tensor([1.0])
    with pytest.raises(ValueError, match="unknown target unit"):
        convert_unit(x, InputUnit.S2_DN, "bogus")  # type: ignore[arg-type]
