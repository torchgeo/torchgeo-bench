"""SAM3 image encoder wrapper for torchgeo-bench.

Uses SAM3's ViT-H vision encoder (with built-in FPN neck) as a frozen backbone
for dense segmentation probing. Only RGB (3-channel) input is supported.

The vision encoder supports arbitrary input sizes. On the first forward pass the
RoPE position embeddings are reset to match the actual image dimensions (rounded
down to the nearest multiple of patch_size=14 if needed). Absolute position
embeddings are already tiled dynamically by the transformers implementation.

Checkpoint:
    Load from a local HuggingFace-format directory (``model.safetensors`` +
    ``config.json``) via the ``checkpoint_path`` config key, or pass
    ``model_name_or_path="facebook/sam3"`` to download from the Hub (requires
    authentication and access approval for the gated repo).

Layer naming for SegmentationProbe:
    The FPN neck produces 4 multi-scale feature maps, each with 256 channels.
    Their spatial dimensions scale with the input resolution:
    at 252×252 input (18×18 patch grid) the levels are:
        neck.fpn_layers.3 — coarsest (scale 0.5×)
        neck.fpn_layers.2 — medium   (scale 1×)
        neck.fpn_layers.1 — fine     (scale 2×)
        neck.fpn_layers.0 — finest   (scale 4×)
    Use coarse-to-fine order for FPN/DPT heads.
"""

import logging

import torch

from .interface import BenchModel

logger = logging.getLogger(__name__)

_PATCH_SIZE = 14  # SAM3 ViT-H patch size (fixed by architecture)


def _reset_sam3_rope(vision_encoder: torch.nn.Module, input_h: int, input_w: int) -> None:
    """Recompute RoPE buffers in every ViT layer for a new input resolution.

    Each ``Sam3ViTLayer`` owns a ``Sam3ViTRotaryEmbedding`` (``self.rotary_emb``)
    with pre-computed ``rope_embeddings_cos / rope_embeddings_sin`` buffers sized
    for the pretrain token grid (72×72 at 1008×1008 input).  We rebuild them for
    the actual token grid derived from ``input_h × input_w``.

    For windowed-attention layers the RoPE grid is always ``(window_size,
    window_size)`` — identical to pretrain, so nothing changes there.  For global
    attention layers the RoPE grid is ``(h_tokens, w_tokens)`` and the scale
    factor is adjusted accordingly.
    """
    cfg = vision_encoder.config.backbone_config  # Sam3ViTConfig
    patch_size: int = cfg.patch_size  # 14
    window_size: int = cfg.window_size  # 24
    global_attn_indexes: set[int] = set(cfg.global_attn_indexes)

    h_tokens = input_h // patch_size
    w_tokens = input_w // patch_size

    logger.info(
        f"SAM3: resetting RoPE embeddings for {input_h}×{input_w} "
        f"({h_tokens}×{w_tokens} token grid)"
    )

    for i, layer in enumerate(vision_encoder.backbone.layers):
        rotary_emb = layer.rotary_emb

        if i in global_attn_indexes:
            end_x, end_y = h_tokens, w_tokens
            scale = window_size / h_tokens
        else:
            # Windowed layers: RoPE always covers one window — no change needed,
            # but we recompute anyway to keep everything consistent.
            end_x, end_y = window_size, window_size
            scale = 1.0

        dim: int = rotary_emb.dim
        freqs = 1.0 / (
            rotary_emb.rope_theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim)
        )
        flat = torch.arange(end_x * end_y, dtype=torch.long)
        x_pos = (flat % end_x).float() * scale
        y_pos = torch.div(flat, end_x, rounding_mode="floor").float() * scale
        inv_freq = torch.cat(
            [torch.outer(x_pos, freqs), torch.outer(y_pos, freqs)], dim=-1
        ).repeat_interleave(2, dim=-1)

        device = rotary_emb.rope_embeddings_cos.device
        dtype = rotary_emb.rope_embeddings_cos.dtype
        rotary_emb.rope_embeddings_cos = inv_freq.cos().to(device=device, dtype=dtype)
        rotary_emb.rope_embeddings_sin = inv_freq.sin().to(device=device, dtype=dtype)
        rotary_emb.end_x = end_x
        rotary_emb.end_y = end_y


class SAM3Encoder(BenchModel):
    """Frozen SAM3 vision encoder (ViT-H + FPN neck) as a benchmark backbone.

    The full SAM3 model is loaded but only the vision encoder is retained.
    The text encoder, geometry encoder, DETR encoder/decoder, and mask decoder
    are discarded to save memory.

    On the first forward pass the RoPE buffers are reset to match the actual
    input resolution.  If the image dimensions are not multiples of
    ``patch_size=14``, images are cropped to the nearest valid size and a warning
    is logged.  Only 3-channel RGB input is supported.

    Args:
        num_channels: Number of input channels. Must be 3 (RGB only).
        checkpoint_path: Path to a local HuggingFace-format checkpoint
            directory containing ``model.safetensors`` and ``config.json``.
        model_name_or_path: HuggingFace Hub model ID. Used only if
            ``checkpoint_path`` is not set.
    """

    def __init__(
        self,
        num_channels: int,
        checkpoint_path: str | None = None,
        model_name_or_path: str = "facebook/sam3",
        **_kwargs,
    ) -> None:
        super().__init__(num_channels=num_channels)

        if num_channels != 3:
            raise ValueError(
                f"SAM3Encoder only supports 3-channel RGB input, got num_channels={num_channels}. "
                "Run with dataset.bands=[red,green,blue] or skip this dataset."
            )

        try:
            from transformers import Sam3Model
        except ImportError as e:
            raise ImportError(
                "SAM3Encoder requires the 'transformers' package. "
                "Install it with: pip install torchgeo-bench[sam3]"
            ) from e

        source = checkpoint_path or model_name_or_path
        logger.info(f"Loading SAM3 from {source!r} …")
        full_model = Sam3Model.from_pretrained(
            source,
            local_files_only=(checkpoint_path is not None),
        )

        self.backbone = full_model.vision_encoder
        del full_model

        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        # RoPE is reset lazily on the first forward call for the actual input size.
        self._rope_size: tuple[int, int] | None = None

    def _maybe_reset_rope(self, H: int, W: int) -> None:
        """Reset RoPE buffers the first time a new spatial size is seen."""
        h = (H // _PATCH_SIZE) * _PATCH_SIZE
        w = (W // _PATCH_SIZE) * _PATCH_SIZE
        if (h, w) == self._rope_size:
            return
        if h != H or w != W:
            logger.warning(
                f"SAM3: input {H}×{W} is not a multiple of patch_size={_PATCH_SIZE}; "
                f"images will be cropped to {h}×{w} before encoding."
            )
        _reset_sam3_rope(self.backbone, h, w)
        self._rope_size = (h, w)

    def _crop_to_patch_multiple(self, images: torch.Tensor) -> torch.Tensor:
        """Crop spatial dims to the nearest multiple of ``_PATCH_SIZE``."""
        H, W = images.shape[-2:]
        h = (H // _PATCH_SIZE) * _PATCH_SIZE
        w = (W // _PATCH_SIZE) * _PATCH_SIZE
        if h == H and w == W:
            return images
        return images[..., :h, :w]

    def forward(
        self,
        images: torch.Tensor,
        bboxes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the vision encoder.

        Called by :class:`~torchgeo_bench.segmentation_probe.SegmentationProbe`
        during feature extraction. Forward hooks on ``neck.fpn_layers.*`` capture
        the multi-scale FPN outputs; the return value itself is not used by the
        probe.

        Args:
            images: ``(B, 3, H, W)`` float tensor.
            bboxes: Unused.

        Returns:
            Pooled image embedding ``(B, 256)`` (average of the coarsest FPN level).
        """
        del bboxes
        images = self._crop_to_patch_multiple(images)
        self._maybe_reset_rope(*images.shape[-2:])
        with torch.no_grad():
            out = self.backbone(pixel_values=images)
        coarsest = out.fpn_hidden_states[-1]  # (B, 256, H', W')
        return coarsest.mean(dim=[-2, -1])  # (B, 256)

    def forward_patch_features(
        self,
        images: torch.Tensor,
        bboxes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return a pooled image embedding for classification evaluation.

        Args:
            images: ``(B, 3, H, W)`` float tensor.
            bboxes: Unused.

        Returns:
            Embeddings ``(B, 256)``.
        """
        return self.forward(images, bboxes)
