"""Calibration metrics for classification probes.

Wraps ``torchmetrics`` calibration error so KNN and Linear Probing
evaluations can report ECE (L1), RMS calibration error (L2), and the
maximum calibration error (MCE) alongside their primary metric.
"""

import numpy as np
import torch
from torchmetrics.classification import (
    BinaryCalibrationError,
    MulticlassCalibrationError,
)

_NORMS: tuple[str, ...] = ("l1", "l2", "max")
_KEYS: dict[str, str] = {"l1": "ece", "l2": "rms_ce", "max": "mce"}


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    multi_label: bool,
    n_bins: int = 15,
) -> dict[str, float]:
    """Compute calibration metrics from class probabilities.

    Args:
        y_true: For single-label, integer class labels of shape ``(N,)``.
            For multi-label, binary indicators of shape ``(N, C)``.
        y_proba: Predicted probabilities of shape ``(N, C)``. For
            multi-label these are per-label sigmoid probabilities (not
            required to sum to 1).
        multi_label: Whether the task is multi-label.
        n_bins: Number of bins used by torchmetrics calibration error.

    Returns:
        Dict with keys ``ece`` (L1), ``rms_ce`` (L2), ``mce`` (max).
        Multi-label values are macro-averaged over labels.
    """
    probs = torch.as_tensor(y_proba, dtype=torch.float32)

    if multi_label:
        targets = torch.as_tensor(y_true, dtype=torch.long)
        n_classes = probs.shape[1]
        per_class: dict[str, list[float]] = {key: [] for key in _KEYS.values()}
        for c in range(n_classes):
            t_c = targets[:, c]
            # Skip degenerate columns: a single observed class makes
            # binning meaningless and torchmetrics emits a warning.
            if t_c.min() == t_c.max():
                continue
            p_c = probs[:, c].clamp(0.0, 1.0)
            for norm in _NORMS:
                metric = BinaryCalibrationError(n_bins=n_bins, norm=norm)
                per_class[_KEYS[norm]].append(float(metric(p_c, t_c).item()))
        return {
            key: (float(np.mean(vals)) if vals else float("nan")) for key, vals in per_class.items()
        }

    targets = torch.as_tensor(y_true, dtype=torch.long)
    n_classes = probs.shape[1]
    # Renormalize defensively; some sklearn estimators return rows that
    # don't sum to exactly 1 due to fp32 accumulation.
    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
    out: dict[str, float] = {}
    for norm in _NORMS:
        metric = MulticlassCalibrationError(num_classes=n_classes, n_bins=n_bins, norm=norm)
        out[_KEYS[norm]] = float(metric(probs, targets).item())
    return out
