"""Unit tests for the SAM3 wrapper."""

import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from torchgeo_bench.datasets.base import BandSpec
from torchgeo_bench.models.sam3 import SAM3Encoder


def _bands(n: int) -> list[BandSpec]:
    return [
        BandSpec(
            sensor="s2",
            name=f"b{i}",
            source_name=f"B{i}",
            mean=1000.0,
            std=250.0,
            min=0.0,
            max=10000.0,
        )
        for i in range(n)
    ]


def _rgb_bands() -> list[BandSpec]:
    return [
        BandSpec(
            sensor="s2",
            name=name,
            source_name=name.upper(),
            mean=1000.0,
            std=250.0,
            min=0.0,
            max=10000.0,
        )
        for name in ("red", "green", "blue")
    ]


@pytest.fixture(autouse=True)
def _mock_sam3_pretrained(monkeypatch):
    class _FakeVisionEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(1))

        def forward(self, pixel_values, **_kwargs):
            b, _, h, w = pixel_values.shape
            h_tokens = max(1, h // 14)
            w_tokens = max(1, w // 14)
            return types.SimpleNamespace(
                fpn_hidden_states=[torch.zeros(b, 256, h_tokens, w_tokens, device=pixel_values.device)]
            )

    class _FakeSam3(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vision_encoder = _FakeVisionEncoder()

    def _from_pretrained(source, local_files_only=False, **_kwargs):
        if local_files_only and not Path(source).exists():
            raise FileNotFoundError(source)
        return _FakeSam3()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.Sam3Model = type("Sam3Model", (), {"from_pretrained": staticmethod(_from_pretrained)})
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr("torchgeo_bench.models.sam3._reset_sam3_rope", lambda *_args, **_kwargs: None)


def test_rgb_only_enforcement():
    with pytest.raises(ValueError, match="RGB"):
        SAM3Encoder(bands=_bands(4))


def test_local_checkpoint_path(tmp_path: Path):
    ckpt = tmp_path / "model.pt"
    ckpt.touch()
    model = SAM3Encoder(bands=_rgb_bands(), checkpoint_path=str(ckpt))
    assert isinstance(model, SAM3Encoder)


def test_missing_local_checkpoint_raises(tmp_path: Path):
    missing = tmp_path / "missing.pt"
    with pytest.raises(FileNotFoundError):
        SAM3Encoder(bands=_rgb_bands(), checkpoint_path=str(missing))


def test_forward_output_shape():
    model = SAM3Encoder(bands=_rgb_bands())
    out = model.forward_patch_features(torch.rand(2, 3, 224, 224))
    assert out.ndim == 2
    assert out.shape[0] == 2
    assert out.shape[1] > 0
    assert torch.isfinite(out).all()


def test_small_image_raises():
    model = SAM3Encoder(bands=_rgb_bands())
    with pytest.raises(ValueError, match="smaller than patch_size"):
        model.forward_patch_features(torch.rand(2, 3, 4, 4))
