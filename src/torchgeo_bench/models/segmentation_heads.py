"""Segmentation decoder heads for use with SegmentationProbe."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearHead(nn.Module):
    """Per-layer BN + 1×1 conv heads with learned scale-weighted fusion.

    For a single layer the output is returned directly. For multiple layers,
    each head's logits are upsampled to the input resolution and combined via
    learned scalar weights.

    Args:
        channels_list: Channel count for each hooked feature layer.
        num_classes: Number of segmentation output classes.
    """

    def __init__(self, channels_list: list[int], num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.heads = nn.ModuleList(
            [
                nn.Sequential(nn.BatchNorm2d(c), nn.Conv2d(c, num_classes, kernel_size=1))
                for c in channels_list
            ]
        )
        if len(channels_list) > 1:
            self.scale_weights = nn.Parameter(torch.ones(len(channels_list)))

    def forward(self, features: list[torch.Tensor], input_h: int, input_w: int) -> torch.Tensor:
        """Upsample and sum per-layer logits."""
        total_logits: torch.Tensor | int = 0
        for idx, (feat, head) in enumerate(zip(features, self.heads)):
            logits = head(feat)
            if logits.shape[-2:] != (input_h, input_w):
                logits = F.interpolate(
                    logits, size=(input_h, input_w), mode="bilinear", align_corners=False
                )
            if len(self.heads) == 1:
                return logits
            total_logits = total_logits + logits * self.scale_weights[idx]
        return total_logits  # type: ignore[return-value]


class ConvBlockHead(nn.Module):
    """Per-layer 1×1 projection to hidden_dim, aligned concat, 1×1 classification head.

    All feature maps are projected to the same channel count, upsampled to the
    finest spatial resolution in the batch, concatenated, and classified with a
    single 1×1 conv.

    Args:
        channels_list: Channel count for each hooked feature layer.
        num_classes: Number of segmentation output classes.
        hidden_dim: Projection dimension (default 256).
    """

    def __init__(self, channels_list: list[int], num_classes: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.projectors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(c, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.SiLU(inplace=True),
                )
                for c in channels_list
            ]
        )
        self.head = nn.Conv2d(hidden_dim * len(channels_list), num_classes, kernel_size=1)

    def forward(self, features: list[torch.Tensor], input_h: int, input_w: int) -> torch.Tensor:
        """Project, upsample, concat, and classify features."""
        proj_feats = [proj(f) for f, proj in zip(features, self.projectors)]

        target_h, target_w = 0, 0
        for f in proj_feats:
            if f.shape[-2] > target_h:
                target_h, target_w = f.shape[-2:]
        if target_h <= 1:
            target_h, target_w = 16, 16

        aligned = []
        for f in proj_feats:
            if f.shape[-2:] != (target_h, target_w):
                f = F.interpolate(
                    f, size=(target_h, target_w), mode="bilinear", align_corners=False
                )
            aligned.append(f)

        logits = self.head(torch.cat(aligned, dim=1))
        if logits.shape[-2:] != (input_h, input_w):
            logits = F.interpolate(
                logits, size=(input_h, input_w), mode="bilinear", align_corners=False
            )
        return logits


class FPNHead(nn.Module):
    """Feature Pyramid Network decoder head.

    Applies lateral 1×1 convs, a top-down merging pathway, 3×3 refinement
    convs, then upsamples all levels to the finest resolution, concatenates,
    and classifies with a 1×1 conv.

    Layers must be provided in **coarse-to-fine order** (deepest / lowest-
    resolution first). Example for ResNet:
    ``["layer4", "layer3", "layer2", "layer1"]``.

    Args:
        channels_list: Channel count for each hooked feature layer (coarse-to-fine).
        num_classes: Number of segmentation output classes.
        hidden_dim: Feature dimension used throughout the FPN (default 256).
    """

    def __init__(self, channels_list: list[int], num_classes: int, hidden_dim: int = 256) -> None:
        super().__init__()
        # Normalise raw CNN features before projection. BN is appropriate here:
        # CNN channels have per-filter semantics and batch stats are stable.
        self.input_norms = nn.ModuleList([nn.BatchNorm2d(c) for c in channels_list])
        self.laterals = nn.ModuleList(
            [nn.Conv2d(c, hidden_dim, kernel_size=1, bias=False) for c in channels_list]
        )
        self.fpn_convs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(inplace=True),
                )
                for _ in channels_list
            ]
        )
        self.fpn_head = nn.Conv2d(hidden_dim * len(channels_list), num_classes, kernel_size=1)

    def forward(self, features: list[torch.Tensor], input_h: int, input_w: int) -> torch.Tensor:
        """Top-down FPN forward pass.

        Args:
            features: Feature maps in coarse-to-fine order (index 0 = coarsest).
            input_h: Target output height (input image height).
            input_w: Target output width (input image width).
        """
        laterals = [lat(norm(f)) for f, norm, lat in zip(features, self.input_norms, self.laterals)]

        # Top-down merging: from coarsest (0) to finest (-1)
        for i in range(len(laterals) - 1):
            target_size = laterals[i + 1].shape[-2:]
            laterals[i + 1] = laterals[i + 1] + F.interpolate(
                laterals[i], size=target_size, mode="bilinear", align_corners=False
            )

        fpn_outs = [conv(p) for p, conv in zip(laterals, self.fpn_convs)]

        finest_size = fpn_outs[-1].shape[-2:]
        aligned = []
        for f in fpn_outs:
            if f.shape[-2:] != finest_size:
                f = F.interpolate(f, size=finest_size, mode="bilinear", align_corners=False)
            aligned.append(f)

        logits = self.fpn_head(torch.cat(aligned, dim=1))
        if logits.shape[-2:] != (input_h, input_w):
            logits = F.interpolate(
                logits, size=(input_h, input_w), mode="bilinear", align_corners=False
            )
        return logits


# ---------------------------------------------------------------------------
# DPT helper modules (adapted from probe3d — mbanani/probe3d)
# ---------------------------------------------------------------------------


class ChannelLayerNorm(nn.Module):
    """LayerNorm over the channel dimension of a (B, C, H, W) feature map.

    Normalises each spatial position independently across channels — equivalent
    to the LayerNorm inside a ViT block.  This is the natural choice before
    projecting ViT intermediate features, where residual-stream outliers can
    cause large inter-layer scale differences that BatchNorm handles poorly
    (sample-wise norm is immune to per-batch outlier corruption).
    """

    def __init__(self, num_channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer norm over channels."""
        # x: (B, C, H, W) → permute to (B, H, W, C) → LN → back
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


class ResidualConvUnit(nn.Module):
    """Two conv+ReLU layers with a residual skip connection."""

    def __init__(self, features: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(features, features, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size, padding=padding),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual conv block."""
        return self.conv(x) + x


class FeatureFusionBlock(nn.Module):
    """Fuses a feature map with an optional skip connection via residual conv units."""

    def __init__(self, features: int, kernel_size: int = 3, with_skip: bool = True) -> None:
        super().__init__()
        self.with_skip = with_skip
        if with_skip:
            self.resConfUnit1 = ResidualConvUnit(features, kernel_size)
        self.resConfUnit2 = ResidualConvUnit(features, kernel_size)

    def forward(self, x: torch.Tensor, skip_x: torch.Tensor | None = None) -> torch.Tensor:
        """Fuse skip connection and refine features."""
        if skip_x is not None:
            if skip_x.shape[-2:] != x.shape[-2:]:
                skip_x = F.interpolate(
                    skip_x, size=x.shape[-2:], mode="bilinear", align_corners=False
                )
            x = self.resConfUnit1(x) + skip_x
        return self.resConfUnit2(x)


class DPTHead(nn.Module):
    """DPT-style decoder head (adapted from probe3d at mbanani/probe3d, single-view).

    Requires exactly **4** feature layers in **coarse-to-fine order** (same
    convention as FPN, e.g. ``["layer4", "layer3", "layer2", "layer1"]`` for
    ResNet). The forward pass processes features from coarsest to finest
    through a cascade of :class:`FeatureFusionBlock` modules.

    Upsampling chain (mirroring probe3d):
      1. 1×1 project each map to ``hidden_dim``
      2. 2× upsample all projected maps
      3. Top-down fusion cascade (coarse → fine)
      4. 4× upsample the fused result
      5. 3×3 → ReLU → 3×3 output conv to ``num_classes``
      6. Final resize to input resolution

    Args:
        channels_list: Channel count for each hooked feature layer (coarse-to-fine).
            Must have exactly 4 entries.
        num_classes: Number of segmentation output classes.
        hidden_dim: Hidden channel dimension (default 256).
        kernel_size: Conv kernel size for residual units (default 3).
    """

    def __init__(
        self,
        channels_list: list[int],
        num_classes: int,
        hidden_dim: int = 256,
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if len(channels_list) != 4:
            raise ValueError(
                f"DPTHead requires exactly 4 feature layers, got {len(channels_list)}. "
                "Specify exactly 4 layer names in coarse-to-fine order in the model config."
            )
        # Normalise ViT residual-stream features before projection. LayerNorm
        # over channels (per spatial position) matches the ViT's own internal
        # normalisation and is sample-wise — robust to the per-layer outlier
        # activations common in specialist ViTs (e.g. DOFA).
        self.input_norms = nn.ModuleList([ChannelLayerNorm(c) for c in channels_list])
        # 1×1 projection — index 0 = coarsest
        self.convs = nn.ModuleList(
            [nn.Conv2d(c, hidden_dim, kernel_size=1, padding=0) for c in channels_list]
        )
        # Fusion blocks: index 0 = coarsest (no skip), 1-3 receive skip from previous level
        self.ref = nn.ModuleList(
            [
                FeatureFusionBlock(hidden_dim, kernel_size, with_skip=False),  # coarsest
                FeatureFusionBlock(hidden_dim, kernel_size),
                FeatureFusionBlock(hidden_dim, kernel_size),
                FeatureFusionBlock(hidden_dim, kernel_size),  # finest
            ]
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_classes, kernel_size=3, padding=1),
        )

    def forward(self, features: list[torch.Tensor], input_h: int, input_w: int) -> torch.Tensor:
        """DPT forward pass.

        Args:
            features: Feature maps in coarse-to-fine order (index 0 = coarsest).
            input_h: Target output height.
            input_w: Target output width.
        """
        # Normalise → project → 2× upsample
        projected = [
            F.interpolate(conv(norm(f)), scale_factor=2, mode="bilinear", align_corners=True)
            for norm, conv, f in zip(self.input_norms, self.convs, features)
        ]

        # Top-down cascade: coarsest (0) → finest (3)
        out = self.ref[0](projected[0], None)
        out = self.ref[1](projected[1], out)
        out = self.ref[2](projected[2], out)
        out = self.ref[3](projected[3], out)

        # 4× upsample → output conv → resize to input resolution
        out = F.interpolate(out, scale_factor=4, mode="bilinear", align_corners=True)
        out = self.out_conv(out)
        if out.shape[-2:] != (input_h, input_w):
            out = F.interpolate(out, size=(input_h, input_w), mode="bilinear", align_corners=False)
        return out
