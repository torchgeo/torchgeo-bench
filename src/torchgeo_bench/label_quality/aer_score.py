"""Augmented Ensemble Ranking (AER) disagreement scoring.

Ranks images by how far the OOF ensemble members disagree with the label. Each
member's hard prediction is compared to the label by macro-mean IoU over the
classes *present in that label*; the image score is ``1 - mean-over-members`` of
that macro-IoU. Macro (equal-per-class) aggregation is deliberate: a rare-class
error is not masked by a dominant background class, and spurious foreground on an
otherwise-empty label surfaces as a low IoU on the background class.

Absent classes (not in the label) and ``ignore_index`` pixels never enter the
IoU — the former because the macro mean is taken only over present classes, the
latter because they are dropped from every intersection and union.
"""

import logging
from collections.abc import Iterable

import numpy as np
import torch

logger = logging.getLogger(__name__)


def score(
    labels: torch.Tensor | np.ndarray,
    member_preds: torch.Tensor | np.ndarray,
    present_classes: Iterable[int] | None = None,
    *,
    ignore_index: int = 255,
) -> tuple[np.ndarray, np.ndarray]:
    """Score labels by ensemble-member disagreement (``1 - macro-IoU``).

    Args:
        labels: Integer masks ``(N, H, W)`` with values in ``0..C-1`` or
            ``ignore_index``.
        member_preds: Per-member **hard predictions** ``(M, N, H, W)`` (typically
            ``OOFResult.member_preds``, uint8 argmax). A full ``(M, N, C, H, W)``
            softmax stack is also accepted for backward compatibility, in which
            case the argmax over the class axis is taken here.
        present_classes: Universe of valid class ids to consider. Per image the
            macro mean is taken over the intersection of this set with the classes
            actually present in the label. ``None`` uses all classes in the label.
        ignore_index: Label value excluded from every IoU (default 255).

    Returns:
        ``(image_scores, pixel_disagreement_maps)``:

        - ``image_scores`` ``(N,)`` in ``[0, 1]``, higher = more disagreement.
        - ``pixel_disagreement_maps`` ``(N, H, W)`` in ``[0, 1]``: the fraction of
          members whose prediction differs from the label at each pixel (``0`` at
          ignored pixels).
    """
    labels_t = _as_long_tensor(labels)
    preds = _as_long_tensor(member_preds)
    if preds.ndim == 5:  # (M, N, C, H, W) softmax stack -> hard predictions
        preds = preds.argmax(dim=2)

    n = labels_t.shape[0]
    allowed = None if present_classes is None else {int(c) for c in present_classes}

    image_scores = np.zeros(n, dtype=np.float64)
    pixel_maps = np.zeros((n,) + tuple(labels_t.shape[1:]), dtype=np.float64)
    for i in range(n):
        image_scores[i], pixel_maps[i] = _score_image(
            labels_t[i], preds[:, i], allowed, ignore_index
        )
    return image_scores, pixel_maps


def _score_image(
    label: torch.Tensor,
    preds: torch.Tensor,
    allowed: set[int] | None,
    ignore_index: int,
) -> tuple[float, np.ndarray]:
    """AER score and per-pixel disagreement for a single image."""
    valid = label != ignore_index  # (H, W)

    disagree = ((preds != label) & valid).float().mean(dim=0)  # (H, W)

    present = [int(c) for c in torch.unique(label[valid]).tolist()]
    if allowed is not None:
        present = [c for c in present if c in allowed]
    if not present:
        return 0.0, disagree.numpy()

    ious = torch.empty(preds.shape[0], len(present))  # (M, |present|)
    for j, c in enumerate(present):
        label_c = (label == c) & valid  # (H, W)
        pred_c = (preds == c) & valid  # (M, H, W)
        inter = (pred_c & label_c).sum(dim=(-2, -1)).float()
        union = (pred_c | label_c).sum(dim=(-2, -1)).float()
        ious[:, j] = inter / union  # union > 0 since c is present in the label

    macro_iou = ious.mean(dim=1)  # (M,) per-member macro-IoU
    return float(1.0 - macro_iou.mean()), disagree.numpy()


def _as_tensor(x: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Return ``x`` as a CPU tensor without copying when already one."""
    return x.detach().cpu() if isinstance(x, torch.Tensor) else torch.as_tensor(x)


def _as_long_tensor(x: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Return ``x`` as a CPU ``int64`` tensor."""
    return _as_tensor(x).long()
