"""Degenerate-cell detection for the label-quality audit.

A (model, dataset) cell whose OOF members collapsed to the majority class still
produces a full ranking — one that is pure noise. The existing scalar guard
(``low_capacity``, OOF macro-mIoU below ``LOW_CAPACITY_THRESHOLD``) is
structurally blind to that failure: AER's macro-IoU is taken per image over the
classes present in *that image's* label, so a background-only predictor scores
≈1 on background and ≈0 on the minority class, averaging to ≈0.5 — comfortably
above a 0.3 threshold. Measured on the collapsed ``dinov3sat/spacenet7`` cell
shape: OOF mIoU 0.475 (passes) while the minority class gets zero predicted mass.

Two cell-level metrics plus one per-method metric close that gap:

- ``min_class_coverage`` (primary) — predicted class mass divided by
  ground-truth class mass, minimised over members and over the classes actually
  present in the labels. Being GT-relative it is immune to intrinsic class
  imbalance, which a raw majority-fraction test is not: spacenet7's labels are
  already ~95% background, so "95% of predictions are background" is healthy
  there and collapse only shows up against the GT mass.
- ``oof_per_class_iou_min`` (corroborating) — the minimum over present classes of
  the *global* confusion-matrix IoU under a majority vote over members. Distinct
  from AER's per-image macro, it makes legible in the CSV *why* a collapsed cell
  cleared the scalar gate: high background IoU masking a near-zero minority IoU.
- ``score_iqr`` (per method) — the robust spread of a method's own image-score
  distribution. Diagnoses "this ranking carries no information" independently of
  cause, so it also catches saturation the coverage arm would miss.

All metrics are computed from tensors already in memory (labels and the
``(M, N, H, W)`` uint8 ``member_preds``) — never by re-running inference, and
never by materializing the ``(M, N, C, H, W)`` float stack. Accumulation is
chunked along the sample axis so only a chunk is ever upcast to int64;
``aer_score`` already holds an int64 copy of the whole prediction tensor in the
same scope, and a second full-size peak alongside it would OOM the large-N
datasets.
"""

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Minimum GT-relative predicted mass for the rarest present class. Below this a
# member has effectively stopped predicting that class.
DEGENERATE_COVERAGE_THRESHOLD = 0.25
# Minimum IQR of a method's image-score distribution for the ranking to carry
# information. Deliberately a *backstop*, not a competing primary signal: the
# healthy cells of the v3 sweep bottom out at IQR 0.0335 (dinov3sat/spacenet7
# aer), so a threshold anywhere near that flags healthy cells while splitting a
# collapsed cell across its two methods (its cleanlab half sits at 0.0537).
# 0.01 clears every real group by >3x and still catches a flat ranking.
MIN_SCORE_IQR = 0.01
# Samples per accumulation chunk (bounds the int64 upcast).
_CHUNK_SIZE = 64


def class_mass_coverage(
    labels: torch.Tensor | np.ndarray,
    member_preds: torch.Tensor | np.ndarray,
    *,
    num_classes: int,
    ignore_index: int = 255,
    chunk_size: int = _CHUNK_SIZE,
) -> tuple[float, np.ndarray]:
    """GT-relative predicted class mass, minimised over members and present classes.

    ``coverage(m, c) = pred_mass(m, c) / gt_mass(c)`` over valid pixels only,
    restricted to the classes present in the labels. A member that never predicts
    a present class scores 0 for it; a member matching the GT distribution scores
    ≈1. Values above 1 mean over-prediction, which is not collapse.

    Args:
        labels: Integer masks ``(N, H, W)`` with values in ``0..C-1`` or
            ``ignore_index``.
        member_preds: Per-member hard predictions ``(M, N, H, W)``.
        num_classes: Size of the class universe ``C``.
        ignore_index: Label value excluded from both masses (default 255).
        chunk_size: Samples per accumulation chunk.

    Returns:
        ``(min_class_coverage, coverage)``: the scalar minimum over members and
        present classes, and the full ``(M, |C_present|)`` coverage matrix.
        Returns ``(nan, empty)`` when no class is present (all-ignore labels).
    """
    labels_t = _as_tensor(labels)
    preds_t = _as_tensor(member_preds)
    n_members = int(preds_t.shape[0])

    gt_mass = torch.zeros(num_classes, dtype=torch.int64)
    pred_mass = torch.zeros(n_members, num_classes, dtype=torch.int64)
    for lo, hi in _chunks(int(labels_t.shape[0]), chunk_size):
        label_chunk = labels_t[lo:hi].long()
        valid = label_chunk != ignore_index
        gt_mass += torch.bincount(label_chunk[valid].reshape(-1), minlength=num_classes)
        for m in range(n_members):
            pred_chunk = preds_t[m, lo:hi].long()
            pred_mass[m] += torch.bincount(pred_chunk[valid].reshape(-1), minlength=num_classes)

    present = gt_mass > 0
    if not bool(present.any()):
        logger.warning("Degeneracy: no class present in labels; coverage undefined.")
        return float("nan"), np.zeros((n_members, 0), dtype=np.float64)

    coverage = pred_mass[:, present].double() / gt_mass[present].double()
    return float(coverage.min()), coverage.numpy()


def global_per_class_iou(
    labels: torch.Tensor | np.ndarray,
    member_preds: torch.Tensor | np.ndarray,
    *,
    num_classes: int,
    ignore_index: int = 255,
    chunk_size: int = _CHUNK_SIZE,
) -> tuple[float, np.ndarray]:
    """Global (dataset-wide) per-class IoU of the members' majority vote.

    Deliberately *not* AER's per-image macro-IoU: pooling the confusion matrix
    over the whole dataset means a class the ensemble never predicts scores 0
    outright, instead of being averaged against images where it is absent.

    Args:
        labels: Integer masks ``(N, H, W)``.
        member_preds: Per-member hard predictions ``(M, N, H, W)``.
        num_classes: Size of the class universe ``C``.
        ignore_index: Label value excluded from the confusion matrix.
        chunk_size: Samples per accumulation chunk.

    Returns:
        ``(min_iou, per_class_iou)`` over the present classes. ``(nan, empty)``
        when no class is present.
    """
    labels_t = _as_tensor(labels)
    preds_t = _as_tensor(member_preds)

    confusion = torch.zeros(num_classes * num_classes, dtype=torch.int64)
    for lo, hi in _chunks(int(labels_t.shape[0]), chunk_size):
        label_chunk = labels_t[lo:hi].long()
        valid = label_chunk != ignore_index
        vote = _majority_vote(preds_t[:, lo:hi], num_classes)
        idx = label_chunk[valid].reshape(-1) * num_classes + vote[valid].reshape(-1)
        confusion += torch.bincount(idx, minlength=num_classes * num_classes)

    cm = confusion.reshape(num_classes, num_classes).double()
    inter = cm.diagonal()
    union = cm.sum(dim=0) + cm.sum(dim=1) - inter
    present = cm.sum(dim=1) > 0  # GT mass per class
    if not bool(present.any()):
        return float("nan"), np.zeros(0, dtype=np.float64)

    iou = torch.where(union > 0, inter / union, torch.zeros_like(inter))[present]
    return float(iou.min()), iou.numpy()


def score_iqr(image_scores) -> float:
    """Interquartile range of an image-score distribution (NaN-tolerant).

    IQR rather than the standard deviation because both methods produce heavy
    tails by construction — a handful of genuinely broken labels should not make
    an otherwise-flat ranking look informative.
    """
    scores = np.asarray(image_scores, dtype=np.float64).reshape(-1)
    if scores.size == 0 or not np.isfinite(scores).any():
        return float("nan")
    q75, q25 = np.nanpercentile(scores, [75, 25])
    return float(q75 - q25)


def cell_metrics(
    labels: torch.Tensor | np.ndarray,
    member_preds: torch.Tensor | np.ndarray,
    *,
    num_classes: int,
    ignore_index: int = 255,
    chunk_size: int = _CHUNK_SIZE,
) -> dict[str, float]:
    """Both cell-level degeneracy metrics for one (model, dataset) cell.

    Cell-level means method-independent: these depend only on the OOF substrate,
    so they are identical on the Cleanlab and AER rows of the same cell.
    ``score_iqr`` is *not* here — it is per method.

    Returns:
        ``{"min_class_coverage": float, "oof_per_class_iou_min": float}``.
    """
    min_coverage, _ = class_mass_coverage(
        labels,
        member_preds,
        num_classes=num_classes,
        ignore_index=ignore_index,
        chunk_size=chunk_size,
    )
    min_iou, _ = global_per_class_iou(
        labels,
        member_preds,
        num_classes=num_classes,
        ignore_index=ignore_index,
        chunk_size=chunk_size,
    )
    return {"min_class_coverage": min_coverage, "oof_per_class_iou_min": min_iou}


def is_degenerate(
    min_class_coverage: float,
    score_iqr: float,
    *,
    coverage_threshold: float = DEGENERATE_COVERAGE_THRESHOLD,
    iqr_threshold: float = MIN_SCORE_IQR,
) -> bool:
    """Whether a cell's ranking should be treated as noise.

    Either arm is sufficient: a collapsed predictor (low coverage) or a ranking
    with no spread (low IQR). A NaN on either input counts as degenerate — an
    undefined metric is never evidence of health.
    """
    for value, threshold in ((min_class_coverage, coverage_threshold), (score_iqr, iqr_threshold)):
        value = float(value)
        if not np.isfinite(value) or value < threshold:
            return True
    return False


def _majority_vote(preds: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Per-pixel modal class over the member axis of a ``(M, n, H, W)`` chunk."""
    if preds.shape[0] == 1:
        return preds[0].long()
    counts = torch.zeros((num_classes,) + tuple(preds.shape[1:]), dtype=torch.int16)
    for m in range(preds.shape[0]):
        counts.scatter_add_(0, preds[m].long().unsqueeze(0), torch.ones_like(counts[:1]))
    return counts.argmax(dim=0)


def _chunks(n: int, chunk_size: int):
    """``(lo, hi)`` slice bounds covering ``range(n)`` in ``chunk_size`` steps."""
    step = max(1, int(chunk_size))
    for lo in range(0, n, step):
        yield lo, min(lo + step, n)


def _as_tensor(x: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Return ``x`` as a CPU tensor without copying when already one."""
    return x.detach().cpu() if isinstance(x, torch.Tensor) else torch.as_tensor(x)
