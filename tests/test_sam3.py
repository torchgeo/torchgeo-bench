"""Unit tests for the SAM3 wrapper."""

import sys
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
    state: dict[str, Any] = {"loads": [], "rope": [], "inputs": []}

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

    def _from_pretrained(
        source: str, *, local_files_only: bool = False, **_kwargs: object
    ) -> _FakeSam3:
        state["loads"].append((source, local_files_only))
        if local_files_only and not Path(source).exists():
            raise FileNotFoundError(source)
        return _FakeSam3()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.Sam3Model = type(
        "Sam3Model", (), {"from_pretrained": staticmethod(_from_pretrained)}
    )
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(
        "torchgeo_bench.models.sam3._reset_sam3_rope",
        lambda _encoder, h, w: state["rope"].append((h, w)),
    )
    return state


def test_rgb_only_enforcement() -> None:
    with pytest.raises(ValueError, match="RGB"):
        SAM3Encoder(bands=_bands(4))


def test_local_checkpoint_path(tmp_path: Path, mock_sam3_pretrained: dict[str, Any]) -> None:
    ckpt = tmp_path / "checkpoint"
    ckpt.mkdir()
    model = SAM3Encoder(bands=_rgb_bands(), checkpoint_path=str(ckpt))
    assert mock_sam3_pretrained["loads"] == [(str(ckpt), True)]
    assert not model.backbone.training
    assert all(not p.requires_grad for p in model.backbone.parameters())


def test_missing_local_checkpoint_raises(tmp_path: Path) -> None:
    missing = tmp_path / "missing.pt"
    with pytest.raises(FileNotFoundError):
        SAM3Encoder(bands=_rgb_bands(), checkpoint_path=str(missing))


def test_forward_crops_and_pools_coarsest_level(mock_sam3_pretrained: dict[str, Any]) -> None:
    model = SAM3Encoder(bands=_rgb_bands(), normalization="identity")
    images = torch.arange(2 * 3 * 31 * 45, dtype=torch.float32).reshape(2, 3, 31, 45)
    out = model(images)
    expected_input = images[..., :28, :42]
    torch.testing.assert_close(mock_sam3_pretrained["inputs"][0], expected_input)
    expected = expected_input.mean(dim=(1, 2, 3)).view(2, 1).expand(2, 256)
    torch.testing.assert_close(out, expected)
    model(images)
    model(images[..., :20, :20])
    assert mock_sam3_pretrained["rope"] == [(28, 42), (14, 14)]
    assert mock_sam3_pretrained["loads"] == [("facebook/sam3", False)]


def test_small_image_raises() -> None:
    model = SAM3Encoder(bands=_rgb_bands())
    with pytest.raises(ValueError, match="smaller than patch_size"):
        model.forward_patch_features(torch.zeros(2, 3, 4, 4))
