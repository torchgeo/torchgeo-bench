"""Segmentation decoder heads for use with SegmentationProbe."""

from types import SimpleNamespace

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


class PatchLinearHead(nn.Module):
    """ViT-specific patch decoder using a per-token linear projection.

    The first hooked feature map is interpreted as a token grid. Each token is
    independently projected from ``D`` channels to ``C * P^2`` logits via a
    1x1 conv, then rearranged to pixel space with ``pixel_shuffle(P)``.

    Args:
        channels_list: Channel count for each hooked feature layer. Only the
            first entry is used.
        num_classes: Number of segmentation output classes.
    """

    def __init__(self, channels_list: list[int], num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = channels_list[0]
        self.norm = ChannelLayerNorm(self.in_channels)
        self.conv: nn.Conv2d | None = None
        self.patch_size: int | None = None

    def _ensure_conv(
        self,
        patch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> nn.Conv2d:
        if patch_size <= 0:
            raise ValueError(f"Patch size must be positive, got {patch_size}.")

        if self.conv is None:
            conv = nn.Conv2d(self.in_channels, self.num_classes * patch_size**2, kernel_size=1)
            self.conv = conv.to(device=device, dtype=dtype)
            self.patch_size = patch_size
            return self.conv

        if self.patch_size != patch_size:
            raise ValueError(
                f"PatchLinearHead was initialized for patch_size={self.patch_size}, "
                f"but got patch_size={patch_size}."
            )

        if self.conv.weight.device != device or self.conv.weight.dtype != dtype:
            self.conv = self.conv.to(device=device, dtype=dtype)
        return self.conv

    def forward(self, features: list[torch.Tensor], input_h: int, input_w: int) -> torch.Tensor:
        """Project the token grid into pixel logits."""
        feat = features[0]
        grid_h, grid_w = feat.shape[-2:]
        patch_h = round(input_h / grid_h)
        patch_w = round(input_w / grid_w)
        if patch_h != patch_w:
            raise ValueError(
                "PatchLinearHead requires square patch geometry. "
                f"Got input=({input_h}, {input_w}) and grid=({grid_h}, {grid_w})."
            )

        conv = self._ensure_conv(patch_h, feat.device, feat.dtype)
        logits = conv(self.norm(feat))
        logits = F.pixel_shuffle(logits, patch_h)
        if logits.shape[-2:] != (input_h, input_w):
            logits = F.interpolate(
                logits, size=(input_h, input_w), mode="bilinear", align_corners=False
            )
        return logits


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
# DPT decoder
#
# The fusion cascade is the reference implementation from
# ``transformers.models.dpt.modeling_dpt`` (Apache-2.0), itself a faithful port
# of Intel-ISL DPT. It is imported rather than reimplemented so the residual
# and upsampling arithmetic cannot silently drift from the paper.
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


def _dpt_fusion_layer(hidden_dim: int) -> nn.Module:
    """Construct a reference ``DPTFeatureFusionLayer`` at ``hidden_dim`` channels.

    ``DPTFeatureFusionLayer`` and the ``DPTPreActResidualLayer`` s it owns are
    plain ``nn.Module`` s that read exactly three scalars off ``config``; they
    never touch the rest of ``DPTConfig``. A ``SimpleNamespace`` therefore lets
    us reuse the reference implementation verbatim without pulling in
    ``DPTConfig``, ``DPTNeck``, or the bundled ViT.

    The two boolean settings reproduce the stock ViT-DPT defaults:
    ``use_batch_norm_in_fusion_residual=False`` (the reference disables BN for
    the ViT variants), which in turn makes the residual convs biased.

    ``transformers`` is an optional dependency, so the import is deferred to
    construction time — only the ``dpt`` head needs it.
    """
    try:
        from transformers.models.dpt.modeling_dpt import DPTFeatureFusionLayer
    except ImportError as e:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "DPTHead requires the 'transformers' package for its fusion cascade. "
            "Install it with: pip install 'torchgeo-bench[sam3]' or pip install transformers"
        ) from e

    config = SimpleNamespace(
        fusion_hidden_size=hidden_dim,
        use_batch_norm_in_fusion_residual=False,
        use_bias_in_fusion_residual=None,  # → ``not use_batch_norm`` → True
    )
    return DPTFeatureFusionLayer(config)


class DPTHead(nn.Module):
    """DPT decoder head using the reference ``transformers`` fusion cascade.

    Requires exactly **4** feature layers in **coarse-to-fine order** (same
    convention as FPN, e.g. ``["layer4", "layer3", "layer2", "layer1"]`` for
    ResNet). Features are normalised and projected to ``hidden_dim``, then run
    through a top-down cascade of :class:`DPTFeatureFusionLayer`.

    The fusion layers are imported from ``transformers`` rather than
    reimplemented, so the residual/fusion arithmetic matches Intel-ISL DPT
    exactly: pre-activation residual units, refinement applied to the *skip*
    (not the primary input), a post-fusion 1×1 projection, and a 2× upsample
    inside every fusion layer. Four stacked layers therefore take a 14×14 ViT
    token grid to 224×224 with no separate pre- or post-upsampling step.

    Only the reassemble stage is ours: ``ChannelLayerNorm`` + a 1×1 projection
    stands in for ``DPTReassembleStage``, because ``SegmentationProbe`` hands
    the head backbone-agnostic ``(B, D, H, W)`` grids rather than the token
    sequences plus patch geometry that the reference reassemble stage consumes.

    Args:
        channels_list: Channel count for each hooked feature layer (coarse-to-fine).
            Must have exactly 4 entries.
        num_classes: Number of segmentation output classes.
        hidden_dim: Hidden channel dimension (default 256).
    """

    def __init__(
        self,
        channels_list: list[int],
        num_classes: int,
        hidden_dim: int = 256,
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
        # Fusion cascade: index 0 consumes the coarsest map alone, 1-3 each take
        # the running fused state plus the next finer map as the skip.
        self.ref = nn.ModuleList([_dpt_fusion_layer(hidden_dim) for _ in channels_list])
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
        # Reassemble stand-in: normalise → 1×1 project to hidden_dim.
        projected = [conv(norm(f)) for norm, conv, f in zip(self.input_norms, self.convs, features)]

        # Top-down cascade, coarsest (0) → finest (3). The running fused state is
        # the primary input and the next feature map enters as the skip, matching
        # ``DPTFeatureFusionStage``. Each layer upsamples 2×, so the cascade
        # supplies all the upsampling (14×14 → 224×224 for a patch16 ViT).
        out = self.ref[0](projected[0])
        for layer, feat in zip(self.ref[1:], projected[1:]):
            out = layer(out, feat)

        out = self.out_conv(out)
        if out.shape[-2:] != (input_h, input_w):
            out = F.interpolate(out, size=(input_h, input_w), mode="bilinear", align_corners=False)
        return out
