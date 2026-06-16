"""Pure NumPy metrics for CKA drift analysis."""

import numpy as np


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman correlation via rank-transform + Pearson.

    Args:
        x: One-dimensional values.
        y: One-dimensional values.

    Returns:
        Spearman rho, or ``nan`` when either side is constant.
    """
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("x and y must have the same length.")
    if x_arr.shape[0] == 0:
        return float("nan")
    if np.allclose(x_arr, x_arr[0]) or np.allclose(y_arr, y_arr[0]):
        return float("nan")

    rx = np.argsort(np.argsort(x_arr)).astype(np.float64)
    ry = np.argsort(np.argsort(y_arr)).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt(float(np.sum(rx * rx) * np.sum(ry * ry)))
    if denom <= 0:
        return float("nan")
    return float(np.sum(rx * ry) / denom)


def _biased_linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Biased linear CKA on pre-cast float64 2D arrays (no input validation)."""
    x_centered = x - x.mean(axis=0, keepdims=True)
    y_centered = y - y.mean(axis=0, keepdims=True)
    xtx = x_centered.T @ x_centered
    yty = y_centered.T @ y_centered
    ytx = y_centered.T @ x_centered
    num = float(np.sum(ytx * ytx))
    den_x = float(np.sum(xtx * xtx))
    den_y = float(np.sum(yty * yty))
    denom = np.sqrt(den_x * den_y)
    if denom <= 0:
        return float("nan")
    cka = num / denom
    if cka < 0:
        return 0.0
    return min(float(cka), 1.0)


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute biased linear CKA between two activation matrices.

    Args:
        X: Activation matrix ``(N, D_x)``.
        Y: Activation matrix ``(N, D_y)``.

    Returns:
        Linear CKA in ``[0, 1]`` for non-degenerate inputs, or ``nan`` when
        either matrix has zero centered variance.
    """
    x = np.asarray(X, dtype=np.float64)
    y = np.asarray(Y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("X and Y must be 2D arrays.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("X and Y must have the same number of samples.")
    n = int(x.shape[0])

    if n >= 200:
        unbiased = _unbiased_linear_cka(x, y)
        if np.isfinite(unbiased):
            return float(np.clip(unbiased, 0.0, 1.0))

    return _biased_linear_cka(x, y)


def cosine_drift(X_clean: np.ndarray, X_corrupted: np.ndarray) -> float:
    """Compute mean paired cosine similarity between clean and corrupted rows.

    Args:
        X_clean: Clean activations with shape ``(N, D)``.
        X_corrupted: Corrupted activations with shape ``(N, D)``.

    Returns:
        Scalar mean cosine similarity across samples.
    """
    x_clean = np.asarray(X_clean, dtype=np.float64)
    x_corr = np.asarray(X_corrupted, dtype=np.float64)
    if x_clean.ndim != 2 or x_corr.ndim != 2:
        raise ValueError("X_clean and X_corrupted must be 2D arrays.")
    if x_clean.shape != x_corr.shape:
        raise ValueError("X_clean and X_corrupted must have the same shape.")

    dot = np.sum(x_clean * x_corr, axis=1)
    norm_clean = np.linalg.norm(x_clean, axis=1)
    norm_corr = np.linalg.norm(x_corr, axis=1)
    denom = norm_clean * norm_corr
    cosine = np.divide(dot, denom, out=np.zeros_like(dot), where=denom > 0)
    return float(np.clip(np.mean(cosine), 0.0, 1.0))


def participation_ratio(X: np.ndarray) -> float:
    """Compute effective rank of the centered activation matrix.

    The per-column mean is subtracted before the SVD (mirroring the
    centering in :func:`linear_cka`) so the participation ratio measures the
    effective dimensionality of the *variation* in the activations rather
    than being dominated by a constant mean direction.

    Args:
        X: Activation matrix with shape ``(N, D)``.

    Returns:
        Participation ratio ``(sum(lambda))^2 / sum(lambda^2)`` where
        ``lambda`` are singular values squared of the centered matrix.
    """
    x = np.asarray(X, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if x.shape[0] == 0:
        return float("nan")

    x = x - x.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(x, compute_uv=False)
    eigenvalues = singular_values * singular_values
    numerator = float(np.sum(eigenvalues) ** 2)
    denominator = float(np.sum(eigenvalues * eigenvalues))
    if denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def bootstrap_cka_ci(
    X_clean: np.ndarray,
    X_corr: np.ndarray,
    n_boot: int = 200,
    frac: float = 0.8,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Subsample-without-replacement bootstrap CI for :func:`linear_cka`.

    Each resample draws ``round(frac * N)`` row indices without replacement,
    recomputes linear CKA on the paired subsample, and the routine returns the
    2.5/97.5 percentiles of the resampled estimates plus their width.

    Args:
        X_clean: Activation matrix ``(N, D)``.
        X_corr: Activation matrix ``(N, D)``.
        n_boot: Number of bootstrap resamples.
        frac: Fraction of rows per resample.
        seed: Seed controlling the resampling.

    Returns:
        Tuple ``(ci_low, ci_high, width)``; all ``nan`` when the subsample is
        too small or every resample is degenerate.
    """
    x = np.asarray(X_clean, dtype=np.float64)
    y = np.asarray(X_corr, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("X_clean and X_corr must be 2D arrays.")
    if x.shape[0] != y.shape[0]:
        raise ValueError("X_clean and X_corr must have the same number of samples.")

    n = int(x.shape[0])
    k = int(round(float(frac) * n))
    if k < 2:
        return (float("nan"), float("nan"), float("nan"))

    # Pre-compute centered kernel matrices once (O(N·D) centering + O(N²·D) kernels),
    # then each bootstrap resample is a cheap O(k²) kernel slice — avoids recomputing
    # O(k·D²) or O(k²·D) inside the loop for high-D activations.
    x_c = x - x.mean(axis=0, keepdims=True)
    y_c = y - y.mean(axis=0, keepdims=True)
    K = x_c @ x_c.T  # (N, N)
    L = y_c @ y_c.T  # (N, N)

    rng = np.random.default_rng(seed)
    estimates: list[float] = []
    for _ in range(int(n_boot)):
        idx = rng.choice(n, size=k, replace=False)
        Ks = K[np.ix_(idx, idx)]
        Ls = L[np.ix_(idx, idx)]
        hsic_xy = float(np.sum(Ks * Ls))
        hsic_xx = float(np.sum(Ks * Ks))
        hsic_yy = float(np.sum(Ls * Ls))
        den = np.sqrt(hsic_xx * hsic_yy)
        if den <= 0:
            continue
        value = float(np.clip(hsic_xy / den, 0.0, 1.0))
        estimates.append(value)

    if not estimates:
        return (float("nan"), float("nan"), float("nan"))

    arr = np.asarray(estimates, dtype=np.float64)
    ci_low = float(np.percentile(arr, 2.5))
    ci_high = float(np.percentile(arr, 97.5))
    return (ci_low, ci_high, float(ci_high - ci_low))


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / exps.sum(axis=1, keepdims=True)


def _unbiased_hsic(K: np.ndarray, L: np.ndarray) -> float:
    n = int(K.shape[0])
    if n < 4:
        return float("nan")
    K0 = K.copy()
    L0 = L.copy()
    np.fill_diagonal(K0, 0.0)
    np.fill_diagonal(L0, 0.0)
    term1 = float(np.sum(K0 * L0))
    term2 = float(np.sum(K0) * np.sum(L0) / ((n - 1) * (n - 2)))
    term3 = float(2.0 * np.sum(K0 @ L0) / (n - 2))
    return float((term1 + term2 - term3) / (n * (n - 3)))


def _unbiased_linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    K = X @ X.T
    L = Y @ Y.T
    hsic_xy = _unbiased_hsic(K, L)
    hsic_xx = _unbiased_hsic(K, K)
    hsic_yy = _unbiased_hsic(L, L)
    denom = np.sqrt(max(hsic_xx, 0.0) * max(hsic_yy, 0.0))
    if denom <= 0:
        return float("nan")
    return float(hsic_xy / denom)


def track_b_summary(
    X_clean: np.ndarray,
    X_corrupted: np.ndarray,
    probe: object,
    y_true: np.ndarray,
    confidence_threshold: float = 0.9,
) -> dict[str, float]:
    """Compute Track B drift-confidence summary scalars.

    Args:
        X_clean: Clean final-layer activations ``(N, D)``.
        X_corrupted: Corrupted final-layer activations ``(N, D)``.
        probe: Fitted probe exposing either ``coef_``/``intercept_`` or
            ``predict_proba``.
        y_true: Ground-truth labels ``(N,)``.
        confidence_threshold: Confidence threshold for overconfident errors.

    Returns:
        Dictionary with Track B scalar metrics.
    """
    x_clean = np.asarray(X_clean, dtype=np.float64)
    x_corr = np.asarray(X_corrupted, dtype=np.float64)
    y = np.asarray(y_true).reshape(-1)
    if x_clean.ndim != 2 or x_corr.ndim != 2:
        raise ValueError("X_clean and X_corrupted must be 2D arrays.")
    if x_clean.shape != x_corr.shape:
        raise ValueError("X_clean and X_corrupted must have the same shape.")
    if x_clean.shape[0] != y.shape[0]:
        raise ValueError("y_true must align with activation rows.")

    # Default to embedding-space L2 drift; overridden below to logit space
    # when the probe exposes linear weights (the only drift the classifier
    # actually responds to).
    drift = np.linalg.norm(x_corr - x_clean, axis=1)

    if hasattr(probe, "coef_") and hasattr(probe, "intercept_"):
        coef = np.asarray(probe.coef_, dtype=np.float64)
        intercept = np.asarray(probe.intercept_, dtype=np.float64)
        drift = np.linalg.norm((x_corr - x_clean) @ coef.T, axis=1)
        logits = x_corr @ coef.T + intercept.reshape(1, -1)
        probs = _softmax(logits)
    elif hasattr(probe, "predict_proba"):
        predict_proba = probe.predict_proba
        probs = np.asarray(predict_proba(x_corr), dtype=np.float64)
    else:
        raise TypeError("probe must expose either coef_/intercept_ or predict_proba().")

    pred_index = np.argmax(probs, axis=1)
    if hasattr(probe, "classes_"):
        classes = np.asarray(probe.classes_)
        y_pred = classes[pred_index]
    else:
        y_pred = pred_index

    confidence = probs.max(axis=1)
    correct = (y_pred == y).astype(np.float64)
    high_drift = drift > np.median(drift)
    overconfident = confidence > float(confidence_threshold)
    wrong = correct < 0.5
    frac = float(np.mean(high_drift & overconfident & wrong))

    return {
        "spearman_drift_confidence": _spearman(drift, confidence),
        "spearman_drift_correctness": _spearman(drift, correct),
        "frac_overconfident_high_drift": frac,
    }
