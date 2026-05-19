"""Regression tests for silent-failure audit fixes."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from torchgeo_bench.datasets.base import BandSpec
from torchgeo_bench.datasets.caffe import CaFFe
from torchgeo_bench.datasets.loading import _make_resize_transform
from torchgeo_bench.models import InputUnit
from torchgeo_bench.models._input_units import detect_input_unit
from torchgeo_bench.utils import extract_features


class _HeadPoolModel(torch.nn.Module):
    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = images.shape[0]
        return {
            "head.global_pool": torch.arange(batch * 4, dtype=torch.float32).reshape(batch, 1, 4)
        }


def test_extract_features_preserves_batch_dimension_for_head_global_pool() -> None:
    dataset = [
        {"image": torch.zeros(3, 2, 2), "label": torch.tensor(0)},
        {"image": torch.zeros(3, 2, 2), "label": torch.tensor(1)},
    ]
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    features, labels = extract_features(_HeadPoolModel(), loader, "cpu", verbose=False)
    assert features.shape == (2, 4)
    np.testing.assert_array_equal(labels, np.array([0, 1]))


def test_invalid_interpolation_raises_instead_of_falling_back() -> None:
    with pytest.raises(ValueError, match="interpolation must be one of"):
        _make_resize_transform(224, "bilnear")


def test_mixed_scale_unit_detection_raises() -> None:
    bands = [
        BandSpec("s2", "red", "red", mean=0.1, std=0.1, min=0.0, max=1.0, wavelength_um=0.665),
        BandSpec("sar", "vv", "VV", mean=20.0, std=4.0, min=0.0, max=255.0),
    ]
    with pytest.raises(ValueError, match="mixed-scale bands"):
        detect_input_unit(bands)


def test_unit_detection_keeps_low_magnitude_bands_in_raw_sensor_stack() -> None:
    bands = [
        BandSpec("s2", "red", "red", mean=950.0, std=500.0, min=0.0, max=28000.0),
        BandSpec("s2", "cirrus", "B10", mean=12.0, std=5.0, min=0.0, max=90.0),
    ]
    assert detect_input_unit(bands) == InputUnit.S2_DN


def test_caffe_rgb_mode_uses_single_declared_gray_channel() -> None:
    assert CaFFe.rgb_bands == ["gray"]
