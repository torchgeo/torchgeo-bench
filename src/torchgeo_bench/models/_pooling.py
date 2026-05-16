"""Token pooling strategies for ViT-style frozen backbones.

The choice between CLS-token and mean-pooled patch tokens is not innocent.
Different pretraining objectives leave the [CLS] token with different
quality of information:

- MAE / SSL reconstruction (Prithvi, Clay): [CLS] is not directly
  supervised; the encoder ends up routing whatever it likes through it.
  Mean-pooled patch tokens tend to be richer for downstream probing.
- DINO / DINOv3: explicit [CLS] supervision via self-distillation; the
  CLS token is the canonical feature.
- ImageNet-supervised ViT (most timm baselines): [CLS] is read by the
  classifier; both CLS and pooled patches are reasonable.

This module hosts a single helper so every wrapper picks pooling the
same way and the choice is easy to ablate from configs.
"""

import torch

PoolMode = str  # "cls" | "mean" | "both"
VALID_MODES = ("cls", "mean", "both")


def pool_tokens(tokens: torch.Tensor, mode: PoolMode = "mean") -> torch.Tensor:
    """Pool ``(B, N, D)`` ViT tokens to a single ``(B, D)`` (or ``(B, 2D)``) vector.

    Args:
        tokens: ``(B, N, D)`` token sequence. If ``N`` is a perfect square plus
            one, the first token is treated as ``[CLS]``; otherwise the whole
            sequence is treated as patch tokens.
        mode: ``"cls"`` returns the CLS token, ``"mean"`` mean-pools the patch
            tokens (dropping CLS if present), ``"both"`` concatenates them.

    Returns:
        ``(B, D)`` (cls or mean) or ``(B, 2D)`` (both).
    """
    if tokens.ndim != 3:
        raise ValueError(f"expected (B, N, D), got shape {tuple(tokens.shape)}")
    if mode not in VALID_MODES:
        raise ValueError(f"pool mode {mode!r} not in {VALID_MODES}")

    n = tokens.shape[1]
    side = int(round(n**0.5))
    has_cls = side * side == n - 1

    if mode == "cls":
        if not has_cls:
            raise ValueError(
                f"pool='cls' requested but tokens shape {tuple(tokens.shape)} has "
                f"no detectable CLS slot (N={n} is not square+1)."
            )
        return tokens[:, 0, :]

    patches = tokens[:, 1:, :] if has_cls else tokens
    mean = patches.mean(dim=1)
    if mode == "mean":
        return mean

    # both — concat cls and mean_patches; falls back to mean+mean when no CLS
    cls = tokens[:, 0, :] if has_cls else mean
    return torch.cat([cls, mean], dim=-1)
