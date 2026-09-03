"""ResNet and Swin-V2 torchgeo wrappers (timm/torchvision backbones)."""

from typing import Any

import torch
import torch.nn as nn

from torchgeo_bench.datasets.base import BandSpec

from ._input_units import InputUnit
from ._torchgeo_base import _adapt_first_conv, _auto_resize, _TorchGeoBackboneBench


class TorchGeoResNetBench(_TorchGeoBackboneBench):
    """Wrapper for torchgeo ResNet models (resnet18 / resnet50 / resnet152).

    These return ``timm.models.resnet.ResNet`` instances.  We replace ``.fc``
    with ``Identity()`` to get headless ``(B, K)`` feature vectors.

    Defaults match the SeCo / MoCo Sentinel-2 RGB pretrained weights, whose
    ``Normalize`` transform expects raw Sentinel-2 DN values divided into
    a single global scale.
    """

    weights_input_unit = "s2_dn_div10000"
    expected_input_unit = InputUnit.S2_DN

    def __init__(
        self,
        bands: list[BandSpec],
        *,
        factory: str = "resnet50",
        weights_class: str = "ResNet50_Weights",
        weights_member: str = "SENTINEL2_RGB_MOCO",
        auto_resize: bool = False,
        target_size: int | None = 224,
        normalization_input_unit: str | None = None,
        skip_weight_normalize: int = 0,
        input_unit_check: str = "warn",
        **_kwargs: Any,
    ) -> None:
        super().__init__(
            bands=bands,
            factory=factory,
            weights_class=weights_class,
            weights_member=weights_member,
            auto_resize=auto_resize,
            target_size=target_size,
            normalization_input_unit=normalization_input_unit,
            skip_weight_normalize=skip_weight_normalize,
            input_unit_check=input_unit_check,
            **_kwargs,
        )
        self.backbone.fc = nn.Identity()
        # Adapt input conv to dataset channel count via timm's averaging /
        # replication of pretrained weights.  Lets a 13-band MoCo-MSI run
        # on 3-band RGB or 18-band S1+S2 stacks without crashing.
        _adapt_first_conv(self.backbone, "conv1", len(bands))

    @torch.no_grad()
    def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        if self.auto_resize and self.target_size:
            images = _auto_resize(images, self.target_size)
        return self.backbone(images)


class TorchGeoSwinBench(_TorchGeoBackboneBench):
    """Wrapper for torchgeo Swin-V2 models (NAIP / Sentinel-2 SatLAS variants)."""

    weights_input_unit = "uint8_div255"

    def __init__(
        self,
        bands: list[BandSpec],
        *,
        factory: str = "swin_v2_b",
        weights_class: str = "Swin_V2_B_Weights",
        weights_member: str = "NAIP_RGB_MI_SATLAS",
        auto_resize: bool = True,
        target_size: int | None = 256,
        input_unit_check: str = "warn",
        **_kwargs: Any,
    ) -> None:
        super().__init__(
            bands=bands,
            factory=factory,
            weights_class=weights_class,
            weights_member=weights_member,
            auto_resize=auto_resize,
            target_size=target_size,
            input_unit_check=input_unit_check,
            **_kwargs,
        )
        self.backbone.head = nn.Identity()
        # Adapt the patch-embed projection conv so RGB-pretrained Swin
        # weights can run on N-channel input.  Result rows should be
        # marked as "adapted" in any leaderboard since the input conv
        # weights are no longer the pretrained RGB ones.
        _adapt_first_conv(self.backbone, "features.0.0", len(bands))

    @torch.no_grad()
    def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        if self.auto_resize and self.target_size:
            images = _auto_resize(images, self.target_size)
        return self.backbone(images)
