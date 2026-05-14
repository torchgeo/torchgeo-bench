"""Uncertainty and selective-prediction metrics for UQ evaluation."""

import numpy as np


def ece(
    probs: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 15,
    binning: str = "equal_width",
) -> float:
    """Compute expected calibration error with configurable binning.

    Args:
        probs: Class probabilities with shape ``(N, C)``.
        y_true: True labels with shape ``(N,)``.
        n_bins: Number of confidence bins.
        binning: ``"equal_width"`` or ``"equal_mass"``.

    Returns:
        Expected calibration error.
    """
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2D, got shape {probs.shape}")
    if y_true.ndim != 1:
        raise ValueError(f"y_true must be 1D, got shape {y_true.shape}")
    if probs.shape[0] != y_true.shape[0]:
        raise ValueError("probs and y_true must have matching first dimension")
    if n_bins <= 0:
        raise ValueError(f"n_bins must be positive, got {n_bins}")

    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(np.float64)
    n = conf.shape[0]
    if n == 0:
        return 0.0

    if binning == "equal_width":
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif binning == "equal_mass":
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)
        bin_edges = np.quantile(conf, quantiles)
        bin_edges[0] = 0.0
        bin_edges[-1] = 1.0
    else:
        raise ValueError(
            f"Unknown ECE binning mode '{binning}'. Expected 'equal_width' or 'equal_mass'."
        )

    total = 0.0
    for i in range(n_bins):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]
        mask = (conf >= lo) & (conf <= hi) if i == n_bins - 1 else (conf >= lo) & (conf < hi)
        if not np.any(mask):
            continue
        acc = correct[mask].mean()
        avg_conf = conf[mask].mean()
        total += (mask.sum() / n) * abs(acc - avg_conf)
    return float(total)


def nll(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Return mean negative log-likelihood.

    Args:
        probs: Class probabilities with shape ``(N, C)``.
        y_true: True labels with shape ``(N,)``.

    Returns:
        Mean negative log-likelihood.
    """
    clipped = np.clip(probs, 1e-12, 1.0)
    return float(-np.log(clipped[np.arange(y_true.shape[0]), y_true]).mean())


def brier_score(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Return multiclass Brier score in ``[0, 2]``.

    Args:
        probs: Class probabilities with shape ``(N, C)``.
        y_true: True labels with shape ``(N,)``.

    Returns:
        Multiclass Brier score.
    """
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2D, got shape {probs.shape}")
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(y_true.shape[0]), y_true] = 1.0
    return float(np.sum((probs - one_hot) ** 2, axis=1).mean())


def predictive_entropy(probs: np.ndarray) -> float:
    """Return mean predictive entropy.

    Args:
        probs: Class probabilities with shape ``(N, C)``.

    Returns:
        Mean predictive entropy.
    """
    clipped = np.clip(probs, 1e-12, 1.0)
    return float((-clipped * np.log(clipped)).sum(axis=1).mean())


def normalized_predictive_entropy(probs: np.ndarray) -> float:
    """Return mean predictive entropy normalized to ``[0, 1]``.

    Args:
        probs: Class probabilities with shape ``(N, C)``.

    Returns:
        Mean predictive entropy divided by ``log(C)``.
    """
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2D, got shape {probs.shape}")
    n_classes = probs.shape[1]
    if n_classes <= 1:
        return 0.0
    return float(predictive_entropy(probs) / np.log(float(n_classes)))


def max_probability(probs: np.ndarray) -> float:
    """Return mean maximum class probability.

    Args:
        probs: Class probabilities with shape ``(N, C)``.

    Returns:
        Mean confidence of top predicted class.
    """
    return float(probs.max(axis=1).mean())


def _risk_coverage_curve(
    confidence: np.ndarray,
    y_pred: np.ndarray,
    y_true: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the risk-coverage curve for confidence-ranked predictions.

    Args:
        confidence: Confidence scores with shape ``(N,)``.
        y_pred: Predicted labels with shape ``(N,)``.
        y_true: True labels with shape ``(N,)``.

    Returns:
        Tuple of ``(risk_curve, correct)`` where:
        - ``risk_curve`` has shape ``(N,)`` and stores selective risk at each
          prefix coverage level ``k / N``.
        - ``correct`` has shape ``(N,)`` with 1 for correct predictions.
    """
    if confidence.ndim != 1:
        raise ValueError(f"confidence must be 1D, got shape {confidence.shape}")
    if y_pred.ndim != 1 or y_true.ndim != 1:
        raise ValueError("y_pred and y_true must be 1D arrays")
    if not (len(confidence) == len(y_pred) == len(y_true)):
        raise ValueError("confidence, y_pred, and y_true must have equal length")

    n = len(y_true)
    if n == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.int32)

    correct = (y_pred == y_true).astype(np.int32)
    order = np.argsort(-confidence, kind="stable")
    sorted_correct = correct[order]
    cum_errors = np.cumsum(1 - sorted_correct)
    risk_curve = cum_errors / (np.arange(n) + 1)
    return risk_curve.astype(np.float64), correct


def raw_aurc(confidence: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Return raw AURC (area under risk-coverage curve).

    This is the standard selective-classification AURC:
    ``mean_k risk@coverage(k / N)`` over confidence-ranked prefixes.

    Args:
        confidence: Confidence scores with shape ``(N,)``.
        y_pred: Predicted labels with shape ``(N,)``.
        y_true: True labels with shape ``(N,)``.

    Returns:
        Raw AURC. Lower is better.
    """
    risk_curve, _ = _risk_coverage_curve(confidence, y_pred, y_true)
    if risk_curve.size == 0:
        return 0.0
    return float(risk_curve.mean())


def excess_aurc(confidence: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Return excess AURC (E-AURC) over the model-specific optimum.

    E-AURC is defined as:
    ``raw_aurc - optimal_raw_aurc``,
    where ``optimal_raw_aurc`` is obtained by the ideal ranking that puts all
    correct predictions before all incorrect predictions for the same base error
    rate. This isolates confidence-ranking quality from base accuracy.

    Args:
        confidence: Confidence scores with shape ``(N,)``.
        y_pred: Predicted labels with shape ``(N,)``.
        y_true: True labels with shape ``(N,)``.

    Returns:
        Excess AURC. ``0`` means confidence ranking is optimal.
    """
    risk_curve, correct = _risk_coverage_curve(confidence, y_pred, y_true)
    if risk_curve.size == 0:
        return 0.0

    n = correct.shape[0]
    aurc_raw = float(risk_curve.mean())
    optimal_order = np.argsort(-(correct.astype(np.float64)), kind="stable")
    optimal_correct = correct[optimal_order]
    optimal_cum_errors = np.cumsum(1 - optimal_correct)
    optimal_risk_curve = optimal_cum_errors / (np.arange(n) + 1)
    aurc_opt = float(optimal_risk_curve.mean())
    return float(max(0.0, aurc_raw - aurc_opt))


def aurc(confidence: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Backward-compatible alias for :func:`excess_aurc`.

    Use :func:`raw_aurc` or :func:`excess_aurc` explicitly in new code.
    """
    return excess_aurc(confidence, y_pred, y_true)


def selective_accuracy(
    confidence: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray, coverage: float
) -> float:
    """Return accuracy on the top-coverage fraction by confidence.

    Args:
        confidence: Confidence scores with shape ``(N,)``.
        y_pred: Predicted labels with shape ``(N,)``.
        y_true: True labels with shape ``(N,)``.
        coverage: Fraction of highest-confidence predictions to keep.

    Returns:
        Accuracy on selected predictions.
    """
    if coverage <= 0.0 or coverage > 1.0:
        raise ValueError(f"coverage must be in (0, 1], got {coverage}")
    n = len(y_true)
    if n == 0:
        return 0.0
    k = max(1, int(np.ceil(coverage * n)))
    order = np.argsort(-confidence, kind="stable")
    kept = order[:k]
    return float((y_pred[kept] == y_true[kept]).mean())


def empirical_coverage(pred_sets: np.ndarray, y_true: np.ndarray) -> float:
    """Return fraction of rows where true class is in the predicted set.

    Args:
        pred_sets: Boolean prediction-set matrix with shape ``(N, C)``.
        y_true: True labels with shape ``(N,)``.

    Returns:
        Empirical set coverage.
    """
    if pred_sets.ndim != 2:
        raise ValueError(f"pred_sets must be 2D, got shape {pred_sets.shape}")
    if y_true.ndim != 1:
        raise ValueError(f"y_true must be 1D, got shape {y_true.shape}")
    return float(pred_sets[np.arange(y_true.shape[0]), y_true].mean())


def mean_set_size(pred_sets: np.ndarray) -> float:
    """Return mean size of prediction sets.

    Args:
        pred_sets: Boolean prediction-set matrix with shape ``(N, C)``.

    Returns:
        Average number of classes per prediction set.
    """
    if pred_sets.ndim != 2:
        raise ValueError(f"pred_sets must be 2D, got shape {pred_sets.shape}")
    return float(pred_sets.sum(axis=1).mean())
