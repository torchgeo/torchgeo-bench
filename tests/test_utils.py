"""Tests for extract_features in torchgeo_bench.utils."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from torchgeo_bench.utils import extract_features


class _IdentityModel(torch.nn.Module):
    """Returns input image as features (flattened)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten(1)


class _DictModel(torch.nn.Module):
    """Returns features as a dict under 'norm' key."""

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"norm": x.flatten(1)}


class _GlobalPoolModel(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"global_pool": x.flatten(1)}


class _1DModel(torch.nn.Module):
    """Returns 1-D output (single sample)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten(1).squeeze(0)  # (C,) for batch=1


def _make_loader(n: int = 8, c: int = 3, h: int = 4, multi_label: bool = False) -> DataLoader:
    images = torch.rand(n, c, h, h)
    labels = torch.randint(0, 4, (n,)) if not multi_label else torch.randint(0, 2, (n, 5))
    dataset = [{"image": images[i], "label": labels[i]} for i in range(n)]
    return DataLoader(dataset, batch_size=4)


@pytest.mark.parametrize("verbose", [False, True])
def test_basic_extraction(verbose: bool, capsys: pytest.CaptureFixture[str]) -> None:
    loader = _make_loader()
    model = _IdentityModel()
    X, y = extract_features(
        model, loader, device="cpu", verbose=verbose, description="Extracting (train)"
    )
    expected_images = torch.stack([sample["image"] for sample in loader.dataset])
    expected_labels = torch.stack([sample["label"] for sample in loader.dataset])
    np.testing.assert_array_equal(X, expected_images.flatten(1).numpy())
    np.testing.assert_array_equal(y, expected_labels.numpy())
    assert ("Extracting (train)" in capsys.readouterr().err) == verbose


def test_extraction_uses_inference_mode_and_restores_grad_state() -> None:
    class _InferenceModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            assert torch.is_inference_mode_enabled()
            assert not torch.is_grad_enabled()
            return x.flatten(1)

    with torch.enable_grad():
        X, y = extract_features(_InferenceModel(), _make_loader(), device="cpu", verbose=False)
        assert torch.is_grad_enabled()
        assert not torch.is_inference_mode_enabled()

    assert X.shape == (8, 48)
    assert y.shape == (8,)


def test_dict_norm_output():
    loader = _make_loader()
    model = _DictModel()
    X, y = extract_features(model, loader, device="cpu", verbose=False)
    assert X.shape[0] == 8
    assert np.isfinite(X).all()


def test_dict_global_pool_output():
    loader = _make_loader()
    model = _GlobalPoolModel()
    X, y = extract_features(model, loader, device="cpu", verbose=False)
    assert X.shape[0] == 8


def test_missing_label_key_raises():
    images = torch.rand(4, 3, 4, 4)
    dataset = [{"image": images[i]} for i in range(4)]
    loader = DataLoader(dataset, batch_size=4)
    model = _IdentityModel()
    with pytest.raises(KeyError, match="label"):
        extract_features(model, loader, device="cpu", verbose=False)


def test_unknown_dict_key_raises():
    class _BadModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
            return {"unknown_key": x.flatten(1)}

    loader = _make_loader()
    with pytest.raises(ValueError, match="Unexpected features"):
        extract_features(_BadModel(), loader, device="cpu", verbose=False)


def test_with_transforms():
    loader = _make_loader(c=3)
    model = _IdentityModel()
    transform = lambda x: x * 2.0  # noqa: E731
    X, y = extract_features(model, loader, device="cpu", transforms=transform, verbose=False)
    assert X.shape[0] == 8


def test_3d_output_mean_pooled():
    """3-D model output (B, T, C) should be mean-pooled to (B, C)."""

    class _SeqModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.flatten(2).permute(0, 2, 1)  # (B, HW, C)

    loader = _make_loader(n=4, c=2, h=3)
    X, y = extract_features(_SeqModel(), loader, device="cpu", verbose=False)
    assert X.ndim == 2
    assert X.shape[0] == 4
