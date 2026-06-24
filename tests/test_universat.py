"""Tests for the UniverSat wrapper.

The integration test loads the released UniverSat weights via torch.hub and is
marked ``slow`` (skipped unless ``-m slow``) since it pulls ~201M params.
"""

import pytest
import torch

from torchgeo_bench.datasets.base import BandSpec
from torchgeo_bench.models import UniverSatBenchModel


def _s2_band(name: str, wavelength_um: float) -> BandSpec:
    return BandSpec(
        "s2", name, name.upper(), mean=0.1, std=0.05, min=0.0, max=1.0, wavelength_um=wavelength_um
    )


_RGB = [_s2_band("b04", 0.665), _s2_band("b03", 0.56), _s2_band("b02", 0.49)]


def test_rejects_mixed_sensors():
    bands = [_s2_band("b04", 0.665), BandSpec("s1", "vv", "VV", mean=0, std=1, min=0, max=1)]
    with pytest.raises(ValueError, match="single sensor"):
        UniverSatBenchModel(bands=bands)


def test_rejects_unmapped_sensor():
    bands = [BandSpec("lidar", "z", "Z", mean=0, std=1, min=0, max=1, wavelength_um=None)]
    with pytest.raises(ValueError, match="modality"):
        UniverSatBenchModel(bands=bands, repo_ref=None, modality=None)


@pytest.mark.slow
@pytest.mark.parametrize("n_bands", [3, 6])
def test_forward_shapes(n_bands):
    extra = [_s2_band("b08", 0.842), _s2_band("b11", 1.61), _s2_band("b12", 2.19)]
    bands = _RGB + extra[: n_bands - 3]
    model = UniverSatBenchModel(bands=bands).eval()
    x = torch.randn(2, n_bands, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (2, UniverSatBenchModel.embed_dim)
