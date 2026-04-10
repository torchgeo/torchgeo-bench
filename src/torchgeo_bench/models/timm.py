"""Timm backbone wrapper for patch-level feature extraction."""

import timm
import torch
import torch.nn.functional as F

from .interface import BenchModel


class TimmPatchBenchModel(BenchModel):
    """BenchModel wrapper for any timm backbone that returns a single vector embedding ``(B, K)``.

    Key ideas:

    - Use timm's ``num_classes=0`` (headless) to get pooled features out of ``forward()``.
    - A small projection head is added optionally to force a target embedding size (K).
    - Robust to CNNs and ViTs: if a model still returns tokens or spatial maps,
      we reduce to ``(B, C)`` safely.

    Args:
        model_name: Any timm model name (e.g., ``"resnet50"``, ``"convnext_small"``,
            ``"vit_base_patch16_224"``, ``"swin_base_patch4_window7_224"``).
        num_channels: Number of input channels (passed as ``in_chans``).
        pretrained: Load pretrained weights when available.
        normalize: If True, L2-normalize the output embedding.
        global_pool: Global pooling strategy for timm headless models.
            Common values: ``"avg"`` (default for CNNs), ``"token"``/``"avg"`` for ViTs.
            If None, uses the model's default. Overridden to ``""`` when
            ``use_cls_token=True``.
        use_cls_token: If True, use the CLS token representation instead of averaging
            spatial tokens. Only applies to ViT/DeiT models that have a CLS token.
            Automatically disables timm's internal pooling so raw tokens are returned.
    """

    def __init__(
        self,
        model_name: str,
        num_channels: int,
        *,
        pretrained: bool = True,
        normalize: bool = False,
        global_pool: str | None = "avg",
        auto_resize: bool = False,
        target_size: int | None = None,
        use_cls_token: bool = False,
        **_kwargs,
    ) -> None:
        super().__init__(num_channels=num_channels)

        self.model_name = model_name
        self.pretrained = pretrained
        self.normalize = normalize
        self.auto_resize = auto_resize
        self.use_cls_token = use_cls_token

        # When using CLS token, disable timm's internal pooling so we get raw tokens
        if self.use_cls_token and global_pool != "":
            global_pool = ""

        # Create a headless backbone that returns pooled features from forward()
        # (timm convention with num_classes=0)
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=num_channels,
            num_classes=0,
            global_pool=global_pool,
        )

        # Determine default input size from timm config; store square size.
        # timm default_cfg has key 'input_size' like (3, 224, 224)
        default_cfg = getattr(self.backbone, "default_cfg", {}) or {}
        cfg_input_size = default_cfg.get("input_size", None)
        inferred_size: int | None = None
        if isinstance(cfg_input_size, (list, tuple)) and len(cfg_input_size) == 3:
            # (C, H, W)
            inferred_size = int(cfg_input_size[1])
        # Allow override via target_size argument; fall back to inferred; else None
        self.target_size = int(target_size) if target_size is not None else inferred_size

        if self.auto_resize and self.target_size is None:
            # Warn user once that resize cannot happen automatically.
            import logging

            logging.getLogger(__name__).warning(
                "auto_resize=True but target_size could not be inferred for %s; disabling auto-resize.",
                model_name,
            )
            self.auto_resize = False

    @torch.no_grad()
    def forward_patch_features(
        self,
        images: torch.Tensor,
        bboxes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return pooled patch embeddings of shape ``(B, K)``.

        Optionally resizes inputs to the backbone's expected resolution when
        ``auto_resize`` is enabled.
        """
        del bboxes
        # Optionally resize to backbone's expected resolution. We assume square size.
        if self.auto_resize and self.target_size is not None:
            h, w = images.shape[-2], images.shape[-1]
            if h != self.target_size or w != self.target_size:
                # Use bicubic for transformers; preserve range.
                images = F.interpolate(
                    images,
                    size=(self.target_size, self.target_size),
                    mode="bicubic",
                    align_corners=False,
                )
        x = self.backbone(images)

        # timm usually returns (B, C) here, but be defensive:
        if x.ndim == 4:
            # (B, C, h, w) -> (B, C)
            x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        elif x.ndim == 3:
            # (B, N, C) tokens
            if self.use_cls_token and self._has_cls_token_like():
                # Use the CLS token (first token)
                x = x[:, 0, :]
            elif x.shape[1] > 1:
                # Average spatial tokens, optionally dropping CLS
                x = x[:, 1:, :] if self._has_cls_token_like() else x
                x = x.mean(dim=1)
            else:
                x = x.squeeze(1)

        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)

        return x  # (B, K)

    # ---- helpers ----
    def _has_cls_token_like(self) -> bool:
        # A cheap heuristic to decide whether to drop the first token before averaging.
        # Works for ViT/DeiT/MAE families in timm.
        # (We probe the module presence rather than the current batch tensor.)
        return any(hasattr(self.backbone, attr) for attr in ("cls_token", "dist_token"))
