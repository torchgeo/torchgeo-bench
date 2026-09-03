"""ScaleMAE torchgeo wrapper (resolution-conditioned ViT)."""

from typing import Any

import torch

from torchgeo_bench.datasets.base import BandSpec

from ._pooling import pool_tokens
from ._torchgeo_base import _auto_resize, _TorchGeoBackboneBench


class TorchGeoScaleMAEBench(_TorchGeoBackboneBench):
    """Wrapper for torchgeo ScaleMAE-Large.

    ``forward_features()`` returns ``(B, N+1, D)`` tokens; ``pool`` selects
    between CLS and mean-pooled post-normalization patch tokens. ``res`` is the
    scale supplied to Scale-MAE's resolution-conditioned positional embedding.
    """

    weights_input_unit = "uint8_div255"

    def __init__(
        self,
        bands: list[BandSpec],
        *,
        factory: str = "scalemae_large_patch16",
        weights_class: str = "ScaleMAELarge16_Weights",
        weights_member: str = "FMOW_RGB",
        auto_resize: bool = False,
        target_size: int | None = None,
        image_size: int | None = None,
        input_unit_check: str = "warn",
        pool: str = "cls",
        res: float = 1.0,
        **_kwargs: Any,
    ) -> None:
        band_names = tuple(band.name.lower() for band in bands)
        valid_rgb_orders = {("red", "green", "blue"), ("b04", "b03", "b02")}
        if band_names not in valid_rgb_orders:
            raise ValueError(
                "Scale-MAE FMOW_RGB requires exactly three ordered RGB bands: "
                "[red, green, blue] or [b04, b03, b02]."
            )
        if pool not in ("cls", "mean"):
            raise ValueError("Scale-MAE pool must be 'cls' or 'mean'.")

        image_size = image_size or target_size or 224
        super().__init__(
            bands=bands,
            factory=factory,
            weights_class=weights_class,
            weights_member=weights_member,
            auto_resize=auto_resize,
            target_size=image_size,
            input_unit_check=input_unit_check,
            factory_kwargs={"img_size": image_size, "res": float(res)},
            **_kwargs,
        )
        self.image_size = image_size
        self.res = float(res)
        self.pool = pool

    @torch.no_grad()
    def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        if self.auto_resize and self.target_size:
            images = _auto_resize(images, self.target_size)
        tokens = self.backbone.forward_features(images)  # (B, N+1, D)
        return pool_tokens(tokens, mode=self.pool)
