"""Unit tests for ``torchgeo_bench.models.rcf`` wrappers."""

import pytest
import torch
from torch.utils.data import Dataset

from torchgeo_bench.models.rcf import RCF, RCFBench

from .test_bench_model import _bands


def test_gaussian_mean_output_shape():
    model = RCF(in_channels=3, features=16, mode="gaussian", stats_mode="mean")
    out = model(torch.rand(2, 3, 64, 64))
    assert out.shape == (2, 16)


def test_gaussian_stdev_output_shape():
    model = RCF(in_channels=3, features=16, mode="gaussian", stats_mode="stdev")
    out = model(torch.rand(2, 3, 64, 64))
    assert out.shape == (2, 32)


def test_gaussian_all_output_shape():
    model = RCF(in_channels=3, features=16, mode="gaussian", stats_mode="all")
    out = model(torch.rand(2, 3, 64, 64))
    assert out.shape == (2, 64)


def test_empirical_requires_dataset():
    with pytest.raises(ValueError, match="dataset must be provided"):
        RCF(in_channels=3, features=16, mode="empirical", dataset=None)


def test_rcf_bench_forward_shape():
    model = RCFBench(bands=_bands(3), features=16, normalization="identity")
    out = model.forward_patch_features(torch.rand(2, 3, 64, 64) * 3000.0)
    assert out.ndim == 2
    assert out.shape[0] == 2
    assert torch.isfinite(out).all()


class _TinyImageDataset(Dataset):
    def __init__(self, channels: int, n: int = 8) -> None:
        self._images = torch.rand(n, channels, 16, 16)

    def __len__(self) -> int:
        return int(self._images.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"image": self._images[idx]}


@pytest.mark.parametrize("stats_mode", ["mean", "stdev", "all"])
def test_rcf_stats_mode_variants(stats_mode: str):
    model = RCF(in_channels=3, features=16, mode="gaussian", stats_mode=stats_mode)
    out = model(torch.rand(2, 3, 16, 16))
    assert out.ndim == 2
    assert out.shape[0] == 2


def test_rcf_empirical_init_runs() -> None:
    dataset = _TinyImageDataset(channels=3)
    model = RCF(in_channels=3, features=16, mode="empirical", dataset=dataset)
    assert model.weights.shape[1] == 3


def test_rcf_empirical_forward_runs() -> None:
    dataset = _TinyImageDataset(channels=3)
    model = RCF(in_channels=3, features=16, mode="empirical", dataset=dataset)
    out = model(torch.rand(2, 3, 16, 16))
    assert out.shape == (2, 16)
    assert torch.isfinite(out).all()


def test_rcf_bench_non_rgb_bands_forward_shape() -> None:
    model = RCFBench(bands=_bands(6), features=16, normalization="identity")
    out = model.forward_patch_features(torch.rand(2, 6, 64, 64) * 3000.0)
    assert out.ndim == 2
    assert out.shape[0] == 2
    assert torch.isfinite(out).all()


def test_rcf_bench_bandspec_zscore_normalization_branch() -> None:
    model = RCFBench(bands=_bands(6), features=16, normalization="bandspec_zscore")
    out = model.forward_patch_features(torch.rand(2, 6, 64, 64) * 3000.0)
    assert out.ndim == 2
    assert out.shape[0] == 2
    assert torch.isfinite(out).all()


def test_rcf_bench_empirical_mode_uses_dataset() -> None:
    dataset = _TinyImageDataset(channels=6)
    model = RCFBench(
        bands=_bands(6),
        features=16,
        mode="empirical",
        dataset=dataset,
        normalization="bandspec_zscore",
    )
    out = model.forward_patch_features(torch.rand(2, 6, 16, 16))
    assert out.shape == (2, 16)
