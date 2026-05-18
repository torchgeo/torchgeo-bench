import logging
import math
from collections.abc import Iterator
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from torchgeo_bench.models.segmentation_heads import (
    ConvBlockHead,
    DPTHead,
    FPNHead,
    LinearHead,
)

logger = logging.getLogger(__name__)


class CachedFeaturesDataset(Dataset):
    """In-RAM cache of pre-extracted backbone features and masks.

    Stores data layer-first: ``layer_tensors[li]`` is a ``(N, C, H, W)``
    float16 tensor for layer *li*, and ``masks`` is an ``(N, H, W)`` long
    tensor.  This contiguous layout eliminates per-sample Python iteration
    during :meth:`GPUTensorCache.from_cached` — the GPU transfer becomes a
    single ``Tensor.to(device)`` call per layer.

    Each ``__getitem__`` returns a ``(features, mask)`` tuple.
    """

    def __init__(
        self,
        layer_tensors: list[torch.Tensor],
        masks: torch.Tensor,
    ) -> None:
        self.layer_tensors = layer_tensors  # list of (N, C, H, W)
        self.masks = masks  # (N, H, W)

    def __len__(self) -> int:
        return self.masks.shape[0]

    def __getitem__(self, i: int) -> tuple[list[torch.Tensor], torch.Tensor]:
        return [t[i] for t in self.layer_tensors], self.masks[i]


def _estimate_cache_bytes(cache: "CachedFeaturesDataset") -> int:
    """Estimate total bytes occupied by a CachedFeaturesDataset."""
    if not cache.layer_tensors:
        return 0
    return (
        sum(t.numel() * t.element_size() for t in cache.layer_tensors)
        + cache.masks.numel() * cache.masks.element_size()
    )


class GPUTensorCache:
    """All cached features pre-stacked and moved to GPU as contiguous tensors.

    Eliminates per-batch CPU→GPU transfers and per-batch ``torch.stack`` calls
    in the training loop.  Use :meth:`from_cached` to build from a
    :class:`CachedFeaturesDataset`, then iterate with :meth:`shuffled_batches`
    (training) or :meth:`ordered_batches` (evaluation).

    Args:
        layer_tensors: One ``(N, C, H, W)`` float16 tensor per hooked layer,
            already on the target device.
        masks: ``(N, H, W)`` long tensor on the target device.
        device: The device these tensors live on.
    """

    def __init__(
        self,
        layer_tensors: list[torch.Tensor],
        masks: torch.Tensor,
        device: torch.device | str,
    ) -> None:
        self.layer_tensors = layer_tensors
        self.masks = masks
        self.device = device

    def __len__(self) -> int:
        return self.masks.shape[0]

    @classmethod
    def from_cached(
        cls,
        cache: "CachedFeaturesDataset",
        device: torch.device | str,
    ) -> "GPUTensorCache":
        """Stack and move all features + masks to *device* in one shot.

        Args:
            cache: CPU-resident cached features.
            device: Target device (must be CUDA for the speedup to be useful).

        Returns:
            A :class:`GPUTensorCache` with all data on *device*.
        """
        target_device = torch.device(device)
        # Keep float32 on CPU (no autocast); use float16 on CUDA for AMP efficiency.
        dtype = torch.float16 if target_device.type == "cuda" else torch.float32
        layer_tensors = [t.to(target_device, dtype=dtype) for t in cache.layer_tensors]
        masks = cache.masks.to(target_device, dtype=torch.long)
        return cls(layer_tensors, masks, target_device)

    def shuffled_batches(
        self, batch_size: int
    ) -> Iterator[tuple[list[torch.Tensor], torch.Tensor]]:
        """Yield *(features, masks)* mini-batches in random order.

        All tensors are already on the GPU — zero host→device transfer per batch.
        """
        idx = torch.randperm(len(self), device=self.device)
        for start in range(0, len(self), batch_size):
            b = idx[start : start + batch_size]
            yield [t[b] for t in self.layer_tensors], self.masks[b]

    def ordered_batches(self, batch_size: int) -> Iterator[tuple[list[torch.Tensor], torch.Tensor]]:
        """Yield *(features, masks)* mini-batches in sequential order."""
        for start in range(0, len(self), batch_size):
            s = slice(start, start + batch_size)
            yield [t[s] for t in self.layer_tensors], self.masks[s]


class SegmentationProbe(nn.Module):
    """Multi-scale segmentation probe that hooks into backbone feature layers.

    Backbone layers are tapped via forward hooks. Features are passed to a
    decoder head (``LinearHead``, ``ConvBlockHead``, ``FPNHead``, or
    ``DPTHead``) that produces per-pixel class logits.

    Layer ordering convention (applies to all head types):
      - **Coarse-to-fine** — deepest / lowest-resolution layer first.
      - Example for ResNet: ``["layer4", "layer3", "layer2", "layer1"]``.
      - For ``DPTHead`` this means index 0 = coarsest, which is also what the
        DPT cascade expects.

    Args:
        backbone: Feature extractor. May be a raw backbone or a ``BenchModel``
            wrapper (``backbone.*`` prefixes are stripped automatically).
        layer_names: Ordered list of layer names to hook (coarse-to-fine).
        num_classes: Number of segmentation output classes.
        freeze_backbone: If ``True`` (default), backbone parameters are frozen
            and the backbone runs in eval mode during inference.
        head_type: Decoder architecture — one of ``"linear"``, ``"conv_block"``,
            ``"fpn"``, ``"dpt"``.
        hidden_dim: Hidden channel dimension for ``conv_block``, ``fpn``, and
            ``dpt`` heads (default 256).
    """

    def __init__(
        self,
        backbone: nn.Module,
        layer_names: list[str],
        num_classes: int,
        freeze_backbone: bool = True,
        head_type: str = "linear",
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.layer_names = layer_names
        self.freeze_backbone = freeze_backbone
        self.head_type = head_type
        self.effective_classes = num_classes

        self._features: dict[str, torch.Tensor] = {}
        self.hooks: list[Any] = []

        found_layers = set()
        for name, module in self.backbone.named_modules():
            if name.startswith("backbone."):
                name = name.replace("backbone.", "", 1)
            if name in self.layer_names:
                self.hooks.append(module.register_forward_hook(self._hook_fn(name)))
                found_layers.add(name)

        missing_layers = set(self.layer_names) - found_layers
        if missing_layers:
            logger.warning(f"The following layers were not found in the backbone: {missing_layers}")

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        self.channels_list = self._dry_run_channels()
        hdim = hidden_dim or 256

        if head_type == "linear":
            self.head = LinearHead(self.channels_list, num_classes)
        elif head_type == "conv_block":
            self.head = ConvBlockHead(self.channels_list, num_classes, hidden_dim=hdim)
        elif head_type == "fpn":
            self.head = FPNHead(self.channels_list, num_classes, hidden_dim=hdim)
        elif head_type == "dpt":
            self.head = DPTHead(self.channels_list, num_classes, hidden_dim=hdim)
        else:
            raise ValueError(
                f"Unknown head_type: {head_type!r}. Choose from: linear, conv_block, fpn, dpt"
            )

    # ------------------------------------------------------------------
    # Hook / dry-run helpers
    # ------------------------------------------------------------------

    def _hook_fn(self, name: str):
        """Return a forward hook that captures the output of the named layer."""

        def hook(module, input, output):  # noqa: ARG001
            self._features[name] = output

        return hook

    def _backbone_device(self) -> torch.device:
        """Return the device of the backbone, falling back to CPU for parameterless backbones."""
        p = next(self.backbone.parameters(), None)
        if p is not None:
            return p.device
        b = next(self.backbone.buffers(), None)
        if b is not None:
            return b.device
        return torch.device("cpu")

    def _dry_run_channels(self) -> list[int]:
        device = self._backbone_device()
        in_channels = int(getattr(self.backbone, "num_channels", 3))
        dummy = torch.randn(1, in_channels, 224, 224, device=device)
        if not self.layer_names:
            self.layer_names = ["backbone_output"]
            self.hooks.append(self.backbone.register_forward_hook(self._hook_fn("backbone_output")))

        was_training = self.backbone.training
        self.backbone.eval()
        self._features.clear()
        with torch.no_grad():
            self.backbone(dummy)

        channels = []
        for name in self.layer_names:
            feat = self._process_feature(self._features[name])
            channels.append(feat.shape[1])
        self.backbone.train(was_training)
        return channels

    def _process_feature(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.ndim == 2:
            return feat.view(feat.shape[0], feat.shape[1], 1, 1)
        if feat.ndim == 3:
            # Handle transformer token features in either (B, L, C) or (B, C, L) layout.
            # Prefer exact square token grids; if L-1 is square, drop CLS token.
            bsz, d1, d2 = feat.shape

            # Try (B, L, C)
            side = math.isqrt(d1)
            if side * side == d1:
                return feat.permute(0, 2, 1).reshape(bsz, d2, side, side)
            side_no_cls = math.isqrt(d1 - 1) if d1 > 1 else 0
            if side_no_cls * side_no_cls == d1 - 1:
                return feat[:, 1:, :].permute(0, 2, 1).reshape(bsz, d2, side_no_cls, side_no_cls)

            # Try (B, C, L)
            side = math.isqrt(d2)
            if side * side == d2:
                return feat.reshape(bsz, d1, side, side)
            side_no_cls = math.isqrt(d2 - 1) if d2 > 1 else 0
            if side_no_cls * side_no_cls == d2 - 1:
                return feat[:, :, 1:].reshape(bsz, d1, side_no_cls, side_no_cls)

            raise ValueError(
                "Could not reshape 3D feature map to 2D grid. "
                f"Got shape={tuple(feat.shape)}. Expected tokens with L=s^2 or L=s^2+1 (CLS)."
            )
        # 4D tensor: NCHW (standard) or NHWC (Swin-family).
        # Detect NHWC: spatial dims are square (H==W) and channel dim (last) is
        # larger than the spatial dims — the opposite of typical NCHW feature maps.
        if feat.ndim == 4:
            _, d1, d2, d3 = feat.shape
            if d1 == d2 and d3 > d1:
                # NHWC → NCHW
                return feat.permute(0, 3, 1, 2).contiguous()
        return feat

    # ------------------------------------------------------------------
    # Feature caching
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_segmentation_features(
        self,
        dataloader: "torch.utils.data.DataLoader",
        cache_dtype: torch.dtype = torch.float16,
    ) -> "CachedFeaturesDataset":
        """Run the frozen backbone once over *dataloader* and cache features.

        Args:
            dataloader: DataLoader that yields ``dict`` or ``(image, mask)`` batches.
            cache_dtype: Storage dtype for cached feature tensors. Use
                ``torch.float16`` (default) to halve RAM, or ``torch.float32``
                for full precision.

        Returns:
            A :class:`CachedFeaturesDataset` with one entry per sample.
        """
        self.backbone.eval()
        # Accumulate per-batch tensors layer-wise, then cat once at the end.
        # This avoids N individual per-sample allocations during GPU transfer.
        batches_per_layer: list[list[torch.Tensor]] = [[] for _ in self.layer_names]
        all_masks: list[torch.Tensor] = []
        device = self._backbone_device()

        for batch in dataloader:
            if isinstance(batch, dict):
                images = batch["image"].to(device)
                masks = batch["mask"]
            else:
                images, masks = batch[0].to(device), batch[1]

            if masks.ndim == 4:
                masks = masks.squeeze(1)
            masks = masks.long()
            self._features.clear()
            _ = self.backbone(images)

            for li, n in enumerate(self.layer_names):
                feat = self._process_feature(self._features[n]).to(dtype=cache_dtype, device="cpu")
                batches_per_layer[li].append(feat)
            all_masks.append(masks.cpu())

        layer_tensors = [torch.cat(batches) for batches in batches_per_layer]
        masks_tensor = torch.cat(all_masks)
        logger.info(f"Cached features for {masks_tensor.shape[0]} samples.")
        return CachedFeaturesDataset(layer_tensors, masks_tensor)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute segmentation logits from input images.

        Args:
            x: Input tensor of shape ``(B, C, H, W)``.

        Returns:
            Logits tensor of shape ``(B, num_classes, H, W)``.
        """
        input_h, input_w = x.shape[-2:]

        if self.freeze_backbone:
            self.backbone.eval()
            use_amp = x.device.type == "cuda"
            with torch.no_grad(), torch.autocast(device_type=x.device.type, enabled=use_amp):
                _ = self.backbone(x)
        else:
            _ = self.backbone(x)

        features = [self._process_feature(self._features[n]) for n in self.layer_names]
        return self.head(features, input_h, input_w)
