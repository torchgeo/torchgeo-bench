"""Cleanlab per-pixel label-quality scoring with soft-min aggregation.

Wraps :mod:`cleanlab.segmentation` over the OOF ``mean_softmax`` substrate. Each
pixel gets a self-confidence score (``1`` = trusted, ``0`` = suspect); these are
aggregated to a per-image score by a soft-minimum, so a small confidently-wrong
region drags the image score down rather than being averaged away. A boolean
per-pixel issue mask comes from ``find_label_issues``.

``ignore_index`` pixels carry no valid class, so they are neutralised before
scoring: the label is remapped to class ``0`` and the predicted probability is
overwritten with confidence ``1`` on that class. Those pixels then read as
trusted, contribute nothing to the soft-min, and are cleared from the issue
mask — the image score is invariant to whatever the model predicted there.
"""

import logging

import numpy as np
import torch
from cleanlab.segmentation.filter import find_label_issues
from cleanlab.segmentation.rank import get_label_quality_scores

logger = logging.getLogger(__name__)


def score(
    labels: torch.Tensor | np.ndarray,
    oof_probs: torch.Tensor | np.ndarray,
    *,
    ignore_index: int = 255,
    soft_min_temp: float = 0.1,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Score segmentation labels against OOF predictions with Cleanlab.

    Args:
        labels: Integer masks ``(N, H, W)`` with values in ``0..C-1`` or
            ``ignore_index``.
        oof_probs: OOF softmax probabilities ``(N, C, H, W)`` at the native label
            frame (typically ``OOFResult.mean_softmax``).
        ignore_index: Label value excluded from scoring (default 255).
        soft_min_temp: Temperature of the soft-min image aggregation. Lower
            values weight the worst pixels more heavily.
        batch_size: Number of images processed per Cleanlab call.

    Returns:
        ``(image_scores, pixel_score_maps, issue_masks)``:

        - ``image_scores`` ``(N,)`` in ``[0, 1]``, lower = more suspect.
        - ``pixel_score_maps`` ``(N, H, W)`` per-pixel self-confidence in ``[0, 1]``.
        - ``issue_masks`` boolean ``(N, H, W)``, ``True`` where a pixel is flagged.
    """
    labels_np = _to_numpy(labels).astype(np.int64)
    probs_np = _to_numpy(oof_probs).astype(np.float64)

    ignore = labels_np == ignore_index
    labels_np, probs_np = _neutralise_ignore(labels_np, probs_np, ignore)

    image_chunks: list[np.ndarray] = []
    pixel_chunks: list[np.ndarray] = []
    issue_chunks: list[np.ndarray] = []
    for start in range(0, len(labels_np), batch_size):
        sl = slice(start, start + batch_size)
        image_scores, pixel_scores = get_label_quality_scores(
            labels_np[sl],
            probs_np[sl],
            method="softmin",
            temperature=soft_min_temp,
            verbose=False,
        )
        issues = find_label_issues(
            labels_np[sl], probs_np[sl], downsample=1, n_jobs=1, verbose=False
        )
        image_chunks.append(image_scores)
        pixel_chunks.append(pixel_scores)
        issue_chunks.append(issues)

    image_scores = np.concatenate(image_chunks, axis=0)
    pixel_score_maps = np.concatenate(pixel_chunks, axis=0)
    issue_masks = np.concatenate(issue_chunks, axis=0).astype(bool)
    issue_masks[ignore] = False  # never flag ignored pixels

    return image_scores, pixel_score_maps, issue_masks


def _neutralise_ignore(
    labels: np.ndarray, probs: np.ndarray, ignore: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Remap ignored pixels to a trusted class-0 prediction (confidence 1)."""
    if not ignore.any():
        return labels, probs
    labels = labels.copy()
    probs = probs.copy()
    labels[ignore] = 0
    ignore_c = ignore[:, None, :, :]  # (N, 1, H, W)
    probs = np.where(ignore_c, 0.0, probs)
    probs[:, 0, :, :] = np.where(ignore, 1.0, probs[:, 0, :, :])
    return labels, probs


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    """Detached CPU numpy view of a tensor or array."""
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)
