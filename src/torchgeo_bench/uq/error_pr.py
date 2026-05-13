"""Error-detection PR metrics from uncertainty scores."""

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score


def compute_error_pr(
    *,
    is_error: np.ndarray,
    uncertainty: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Compute PR curve and scalar metrics for error detection.

    Args:
        is_error: Binary error labels (1=error) with shape ``(N,)``.
        uncertainty: Uncertainty scores where higher indicates more likely error.

    Returns:
        Dictionary with keys ``precision``, ``recall``, ``aupr``/``ap``, and ``auroc``.

    Raises:
        ValueError: If input arrays are invalid or degenerate.
    """
    y = np.asarray(is_error)
    u = np.asarray(uncertainty, dtype=np.float64)

    if y.ndim != 1 or u.ndim != 1:
        raise ValueError("is_error and uncertainty must be 1D arrays")
    if y.shape[0] != u.shape[0]:
        raise ValueError("is_error and uncertainty must have equal length")
    if y.shape[0] < 2:
        raise ValueError("At least two samples are required")
    if not np.isfinite(u).all():
        raise ValueError("uncertainty must be finite")

    y_int = y.astype(np.int64)
    classes = np.unique(y_int)
    if classes.size < 2:
        raise ValueError("is_error must contain both 0 and 1 labels")

    precision, recall, _ = precision_recall_curve(y_int, u)
    ap = float(average_precision_score(y_int, u))
    auroc = float(roc_auc_score(y_int, u))
    return {
        "precision": precision,
        "recall": recall,
        "ap": ap,
        "aupr": ap,
        "auroc": auroc,
    }
