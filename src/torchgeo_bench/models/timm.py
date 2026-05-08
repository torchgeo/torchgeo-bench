"""Timm backbone wrapper for patch-level feature extraction."""

import logging

import timm
import torch
import torch.nn.functional as F

from torchgeo_bench.datasets.base import BandSpec

from .interface import BenchModel

logger = logging.getLogger(__name__)

_VALID_INPUT_NORMALIZATIONS = ("bands_zscore", "imagenet", "timm_default", "none")


class TimmPatchBenchModel(BenchModel):
    """BenchModel wrapper for any timm backbone.

    Args:
        bands: Ordered :class:`BandSpec` list (channel count = ``len(bands)``).
        model_name: Any timm model name (e.g. ``"resnet50"``,
            ``"convnext_small"``, ``"vit_base_patch16_224"``).
        pretrained: Load pretrained weights when available.
        normalize: If ``True``, L2-normalize the output embedding.
        global_pool: Global pooling strategy for timm headless models.
        use_cls_token: For ViT-family models, use the CLS token instead of
            averaging spatial tokens.
        auto_resize: If ``True``, bilinearly resize each batch to ``target_size``.
        target_size: Square target size; auto-inferred from
            ``backbone.default_cfg["input_size"]`` when not given.
        input_normalization: One of ``"bands_zscore"`` (default; per-channel
            z-score using :class:`BandSpec` statistics — correct for raw
            remote-sensing data of any channel count), ``"imagenet"``
            (``(images / scale - mean) / std`` with ImageNet RGB stats; only
            safe for true 0-255 RGB inputs — pair with ``scale=255``),
            ``"timm_default"`` (same shape as ``"imagenet"`` but reads
            ``backbone.default_cfg["mean"]/["std"]``; refuses to instantiate
            when ``len(bands) != 3``), or ``"none"`` (identity).
        scale: Divisor applied before subtracting ``mean`` for
            ``"imagenet"``/``"timm_default"`` modes.  Defaults to ``1.0``
            but most pretrained models want ``255.0``.
    """

    def __init__(
        self,
        bands: list[BandSpec],
        *,
        model_name: str,
        pretrained: bool = True,
        normalize: bool = False,
        global_pool: str | None = "avg",
        auto_resize: bool = False,
        target_size: int | None = None,
        use_cls_token: bool = False,
        input_normalization: str = "bands_zscore",
        scale: float = 1.0,
        **_kwargs,
    ) -> None:
        super().__init__(bands=bands, **_kwargs)

        if input_normalization not in _VALID_INPUT_NORMALIZATIONS:
            raise ValueError(
                f"input_normalization must be one of {_VALID_INPUT_NORMALIZATIONS}, "
                f"got {input_normalization!r}."
            )

        self.model_name = model_name
        self.pretrained = pretrained
        self.normalize = normalize
        self.auto_resize = auto_resize
        self.use_cls_token = use_cls_token
        self.input_normalization = input_normalization
        self.scale = float(scale)

        # When using CLS token, disable timm's internal pooling.
        if self.use_cls_token and global_pool != "":
            global_pool = ""

        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=self.num_channels,
            num_classes=0,
            global_pool=global_pool,
        )

        # Determine default input size from timm config; store square size.
        default_cfg = getattr(self.backbone, "default_cfg", {}) or {}
        cfg_input_size = default_cfg.get("input_size", None)
        inferred_size: int | None = None
        if isinstance(cfg_input_size, (list, tuple)) and len(cfg_input_size) == 3:
            inferred_size = int(cfg_input_size[1])
        self.target_size = int(target_size) if target_size is not None else inferred_size

        if self.auto_resize and self.target_size is None:
            logger.warning(
                "auto_resize=True but target_size could not be inferred for %s; "
                "disabling auto-resize.",
                model_name,
            )
            self.auto_resize = False

        # Pre-compute fixed-RGB normalization tensors when applicable.  Stored
        # as buffers so they move with .to(device).
        if self.input_normalization == "imagenet":
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
            self.register_buffer("_rgb_mean", mean)
            self.register_buffer("_rgb_std", std)
        elif self.input_normalization == "timm_default":
            if self.num_channels != 3:
                raise ValueError(
                    f"input_normalization='timm_default' requires 3 input channels; "
                    f"got {self.num_channels}.  Use 'bands_zscore' for multispectral data."
                )
            cfg_mean = default_cfg.get("mean")
            cfg_std = default_cfg.get("std")
            if cfg_mean is None or cfg_std is None:
                raise ValueError(
                    f"input_normalization='timm_default' but timm '{model_name}' has no "
                    "default_cfg mean/std — pick a different normalization mode."
                )
            mean = torch.tensor(cfg_mean, dtype=torch.float32).view(1, 3, 1, 1)
            std = torch.tensor(cfg_std, dtype=torch.float32).view(1, 3, 1, 1)
            self.register_buffer("_rgb_mean", mean)
            self.register_buffer("_rgb_std", std)

    def normalize_inputs(self, images: torch.Tensor) -> torch.Tensor:
        """Apply the configured normalization policy."""
        mode = self.input_normalization
        if mode == "bands_zscore":
            return super().normalize_inputs(images)
        if mode == "none":
            return images
        # imagenet / timm_default share the same shape:  (x / scale - mean) / std
        scaled = images / self.scale if self.scale != 1.0 else images
        mean = self._rgb_mean.to(dtype=images.dtype)  # type: ignore[attr-defined]
        std = self._rgb_std.to(dtype=images.dtype)  # type: ignore[attr-defined]
        return (scaled - mean) / std

    @torch.no_grad()
    def _forward_patch_features(
        self,
        images: torch.Tensor,
        bboxes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return pooled patch embeddings of shape ``(B, K)`` from normalized inputs."""
        del bboxes
        if self.auto_resize and self.target_size is not None:
            h, w = images.shape[-2], images.shape[-1]
            if h != self.target_size or w != self.target_size:
                images = F.interpolate(
                    images,
                    size=(self.target_size, self.target_size),
                    mode="bicubic",
                    align_corners=False,
                )
        x = self.backbone(images)

        # timm usually returns (B, C); be defensive about (B, C, h, w) and (B, N, C).
        if x.ndim == 4:
            x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        elif x.ndim == 3:
            if self.use_cls_token and self._has_cls_token_like():
                x = x[:, 0, :]
            elif x.shape[1] > 1:
                x = x[:, 1:, :] if self._has_cls_token_like() else x
                x = x.mean(dim=1)
            else:
                x = x.squeeze(1)

        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)

        return x

    def _has_cls_token_like(self) -> bool:
        """Cheap heuristic for ViT/DeiT/MAE-style backbones with a CLS token."""
        return any(hasattr(self.backbone, attr) for attr in ("cls_token", "dist_token"))
