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
    if n >= 200:
        unbiased = _unbiased_linear_cka(x, y)
        if np.isfinite(unbiased):
            cka = unbiased
    if cka < 0:
        return 0.0
    if cka > 1:
        return 1.0
    return float(cka)


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
    """Compute effective rank of an activation matrix.

    Args:
        X: Activation matrix with shape ``(N, D)``.

    Returns:
        Participation ratio ``(sum(lambda))^2 / sum(lambda^2)`` where
        ``lambda`` are singular values squared.
    """
    x = np.asarray(X, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if x.shape[0] == 0:
        return float("nan")

    singular_values = np.linalg.svd(x, compute_uv=False)
    eigenvalues = singular_values * singular_values
    numerator = float(np.sum(eigenvalues) ** 2)
    denominator = float(np.sum(eigenvalues * eigenvalues))
    if denominator <= 0:
        return float("nan")
    return float(numerator / denominator)


def split_half_cka(X: np.ndarray, seed: int) -> tuple[float, float]:
    """Compute split-half clean baseline metrics.

    Args:
        X: Activation matrix ``(N, D)``.
        seed: Random seed controlling the split.

    Returns:
        Tuple ``(split_half_cka, split_half_cosine)``.
    """
    x = np.asarray(X, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("X must be a 2D array.")
    if x.shape[0] < 2:
        raise ValueError("split_half_cka requires at least two samples.")

    n_half = x.shape[0] // 2
    if n_half == 0:
        raise ValueError("split_half_cka requires at least two samples.")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(x.shape[0])
    a = x[perm[:n_half]]
    b = x[perm[n_half : 2 * n_half]]
    return linear_cka(a, b), cosine_drift(a, b)


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

    drift = np.linalg.norm(x_corr - x_clean, axis=1)

    if hasattr(probe, "coef_") and hasattr(probe, "intercept_"):
        coef = np.asarray(getattr(probe, "coef_"), dtype=np.float64)
        intercept = np.asarray(getattr(probe, "intercept_"), dtype=np.float64)
        logits = x_corr @ coef.T + intercept.reshape(1, -1)
        probs = _softmax(logits)
    elif hasattr(probe, "predict_proba"):
        predict_proba = getattr(probe, "predict_proba")
        probs = np.asarray(predict_proba(x_corr), dtype=np.float64)
    else:
        raise TypeError("probe must expose either coef_/intercept_ or predict_proba().")

    pred_index = np.argmax(probs, axis=1)
    if hasattr(probe, "classes_"):
        classes = np.asarray(getattr(probe, "classes_"))
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
