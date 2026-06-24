"""Tests for the UniverSat wrapper.

Fast tests exercise the sensor→modality grouping without loading weights. The
forward test loads the released UniverSat weights via torch.hub and is marked
``slow`` (skipped unless ``-m slow``) since it pulls ~201M params.
"""

import pytest
import torch

from torchgeo_bench.datasets.base import BandSpec
from torchgeo_bench.models import UniverSatBenchModel
from torchgeo_bench.models.universat import _build_sensor_groups, _sar_code


def _s2_band(name: str, wavelength_um: float) -> BandSpec:
    return BandSpec(
        "s2", name, name.upper(), mean=0.1, std=0.05, min=0.0, max=1.0, wavelength_um=wavelength_um
    )


def _sar_band(name: str) -> BandSpec:
    return BandSpec("s1", name, name.upper(), mean=0.0, std=1.0, min=-1.0, max=1.0)


_RGB = [_s2_band("b04", 0.665), _s2_band("b03", 0.56), _s2_band("b02", 0.49)]


def test_single_sensor_group():
    groups = _build_sensor_groups(_RGB)
    assert len(groups) == 1
    assert groups[0]["modality"] == "s2"
    assert groups[0]["indices"] == [0, 1, 2]
    assert groups[0]["wavelengths"] == [0.665, 0.56, 0.49]


def test_multi_sensor_grouping():
    # s2 + s1 interleaved -> two groups, indices preserved, s1 -> sensor codes
    bands = [_s2_band("b04", 0.665), _sar_band("vv"), _s2_band("b03", 0.56), _sar_band("vh")]
    groups = {g["modality"]: g for g in _build_sensor_groups(bands)}
    assert set(groups) == {"s2", "s1"}
    assert groups["s2"]["indices"] == [0, 2]
    assert groups["s1"]["indices"] == [1, 3]
    assert groups["s1"]["wavelengths"] == ["VV", "VH"]


def test_sar_code_mapping():
    assert _sar_code("vh_lee_real") == "VH"
    assert _sar_code("vv_imag") == "VV"
    assert _sar_code("vv_vh") == "Ratio_VV_VH"


def test_rejects_unmapped_sensor():
    bands = [BandSpec("lidar", "z", "Z", mean=0, std=1, min=0, max=1, wavelength_um=None)]
    with pytest.raises(ValueError, match="modality"):
        _build_sensor_groups(bands)


@pytest.mark.slow
@pytest.mark.parametrize("n_bands", [3, 6])
def test_forward_shapes_s2(n_bands):
    extra = [_s2_band("b08", 0.842), _s2_band("b11", 1.61), _s2_band("b12", 2.19)]
    bands = _RGB + extra[: n_bands - 3]
    model = UniverSatBenchModel(bands=bands).eval()
    x = torch.randn(2, n_bands, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, UniverSatBenchModel.embed_dim)


@pytest.mark.slow
def test_forward_multi_sensor():
    bands = _RGB + [_sar_band("vv"), _sar_band("vh")]
    model = UniverSatBenchModel(bands=bands).eval()
    x = torch.randn(2, 5, 32, 32)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, UniverSatBenchModel.embed_dim)
