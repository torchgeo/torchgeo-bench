"""Tests for _input_units: detect_input_unit and convert_unit."""

import pytest
import torch

from torchgeo_bench.datasets.base import BandSpec
from torchgeo_bench.models._input_units import (
    InputUnit,
    convert_unit,
    detect_input_unit,
)


def _band(max_val: float, sensor: str = "s2") -> BandSpec:
    return BandSpec(
        sensor=sensor,
        name="b",
        source_name="B",
        mean=max_val / 2,
        std=max_val / 4,
        min=0.0,
        max=max_val,
    )


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
# convert_unit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("src", "dst", "value", "expected"),
    [
        (InputUnit.S2_DN, InputUnit.REFLECTANCE_0_1, 10000.0, 1.0),
        (InputUnit.S2_DN, InputUnit.UINT8, 10000.0, 255.0),
        (InputUnit.REFLECTANCE_0_1, InputUnit.S2_DN, 0.5, 5000.0),
        (InputUnit.REFLECTANCE_0_1, InputUnit.UINT8, 0.5, 127.5),
        (InputUnit.UINT8, InputUnit.S2_DN, 255.0, 10000.0),
        (InputUnit.UINT8, InputUnit.REFLECTANCE_0_1, 255.0, 1.0),
    ],
)
def test_convert_unit_pairs(src: InputUnit, dst: InputUnit, value: float, expected: float):
    result = convert_unit(torch.tensor([value]), src, dst)
    assert torch.allclose(result, torch.tensor([expected]))


def test_convert_unit_noop_same_src_dst():
    x = torch.tensor([500.0])
    assert torch.allclose(convert_unit(x, InputUnit.S2_DN, InputUnit.S2_DN), x)


def test_convert_unit_unknown_dst_raises():
    x = torch.tensor([1.0])
    with pytest.raises(ValueError, match="unknown target unit"):
        convert_unit(x, InputUnit.S2_DN, "bogus")  # type: ignore[arg-type]
