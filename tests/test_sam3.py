"""Unit tests for the SAM3 wrapper."""

import types
from pathlib import Path
from typing import Any

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
def mock_sam3_pretrained(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {"loads": [], "configs": [], "inputs": []}

    class _FakeVisionEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(1))

        def forward(self, pixel_values: torch.Tensor, **_kwargs: object) -> types.SimpleNamespace:
            state["inputs"].append(pixel_values)
            coarse = pixel_values.mean(dim=1, keepdim=True).expand(-1, 256, -1, -1)
            return types.SimpleNamespace(fpn_hidden_states=[torch.zeros_like(coarse), coarse])

    class _FakeSam3(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vision_encoder = _FakeVisionEncoder()

    def _config_from_pretrained(
        source: str, *, local_files_only: bool = False, **_kwargs: object
    ) -> types.SimpleNamespace:
        if local_files_only and not Path(source).exists():
            raise FileNotFoundError(source)
        return types.SimpleNamespace(vision_config=types.SimpleNamespace(image_size=1008))

    def _model_from_pretrained(
        source: str,
        *,
        config: Any = None,
        local_files_only: bool = False,
        **_kwargs: object,
    ) -> _FakeSam3:
        state["loads"].append((source, local_files_only))
        state["configs"].append(config)
        if local_files_only and not Path(source).exists():
            raise FileNotFoundError(source)
        return _FakeSam3()

    monkeypatch.setattr(
        "torchgeo_bench.models.sam3.Sam3Config",
        type("Sam3Config", (), {"from_pretrained": staticmethod(_config_from_pretrained)}),
    )
    monkeypatch.setattr(
        "torchgeo_bench.models.sam3.Sam3Model",
        type("Sam3Model", (), {"from_pretrained": staticmethod(_model_from_pretrained)}),
    )
    return state


def test_rgb_only_enforcement() -> None:
    with pytest.raises(ValueError, match="RGB"):
        SAM3Encoder(bands=_bands(4), image_size=224)


def test_missing_image_size_raises() -> None:
    with pytest.raises(ValueError, match="fixed input size"):
        SAM3Encoder(bands=_rgb_bands())


def test_local_checkpoint_path(tmp_path: Path, mock_sam3_pretrained: dict[str, Any]) -> None:
    ckpt = tmp_path / "checkpoint"
    ckpt.mkdir()
    model = SAM3Encoder(bands=_rgb_bands(), image_size=224, checkpoint_path=str(ckpt))
    assert mock_sam3_pretrained["loads"] == [(str(ckpt), True)]
    assert not model.backbone.training
    assert all(not p.requires_grad for p in model.backbone.parameters())


def test_missing_local_checkpoint_raises(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pt"
    with pytest.raises(FileNotFoundError):
        SAM3Encoder(bands=_rgb_bands(), image_size=224, checkpoint_path=str(missing))


def test_image_size_reaches_config(mock_sam3_pretrained: dict[str, Any]) -> None:
    """The config-time override replaces the old runtime RoPE reset."""
    SAM3Encoder(bands=_rgb_bands(), image_size=252)
    assert mock_sam3_pretrained["configs"][0].vision_config.image_size == 252


def test_forward_pools_coarsest_level(mock_sam3_pretrained: dict[str, Any]) -> None:
    model = SAM3Encoder(bands=_rgb_bands(), image_size=28, normalization="identity")
    images = torch.arange(2 * 3 * 28 * 28, dtype=torch.float32).reshape(2, 3, 28, 28)
    out = model(images)
    torch.testing.assert_close(mock_sam3_pretrained["inputs"][0], images)
    expected = images.mean(dim=(1, 2, 3)).view(2, 1).expand(2, 256)
    torch.testing.assert_close(out, expected)
    assert mock_sam3_pretrained["loads"] == [("facebook/sam3", False)]


def test_forward_size_mismatch_raises() -> None:
    model = SAM3Encoder(bands=_rgb_bands(), image_size=224, normalization="identity")
    with pytest.raises(ValueError, match="built for 224x224"):
        model(torch.zeros(2, 3, 256, 256))
