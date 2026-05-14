import numpy as np

from torchgeo_bench.cka.metrics import (
    cosine_drift,
    linear_cka,
    participation_ratio,
    split_half_cka,
    track_b_summary,
)


def test_linear_cka_identical_matrices():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 8)).astype(np.float32)
    assert np.isclose(linear_cka(X, X), 1.0, atol=1e-6)


def test_linear_cka_orthogonal_matrices():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(20, 16))
    Z = Z - Z.mean(axis=0, keepdims=True)
    Q, _ = np.linalg.qr(Z)
    X = Q[:, :8].astype(np.float32)
    Y = Q[:, 8:16].astype(np.float32)
    assert np.isclose(linear_cka(X, Y), 0.0, atol=1e-6)


def test_linear_cka_returns_nan_on_zero_variance():
    rng = np.random.default_rng(0)
    X = np.zeros((10, 4), dtype=np.float32)
    Y = rng.normal(size=(10, 4)).astype(np.float32)
    assert np.isnan(linear_cka(X, Y))


def test_linear_cka_range():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 16)).astype(np.float32)
    Y = rng.normal(size=(50, 16)).astype(np.float32)
    cka = linear_cka(X, Y)
    assert 0.0 <= cka <= 1.0


def test_cosine_drift_identical():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 8)).astype(np.float32)
    assert np.isclose(cosine_drift(X, X), 1.0, atol=1e-6)


def test_cosine_drift_shape():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 8)).astype(np.float32)
    Y = rng.normal(size=(20, 8)).astype(np.float32)
    value = cosine_drift(X, Y)
    assert isinstance(value, float)


def test_participation_ratio_rank1():
    v = np.arange(1, 11, dtype=np.float32)
    X = np.outer(v, v).astype(np.float32)
    assert np.isclose(participation_ratio(X), 1.0, atol=1e-6)


def test_participation_ratio_full_rank():
    X = np.eye(10, dtype=np.float32)
    assert np.isclose(participation_ratio(X), 10.0, atol=1e-6)


def test_split_half_cka_returns_tuple():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 8)).astype(np.float32)
    cka, cos = split_half_cka(X, seed=42)
    assert isinstance(cka, float)
    assert isinstance(cos, float)
    assert 0.0 <= cka <= 1.0
    assert 0.0 <= cos <= 1.0


def test_split_half_cka_reproducible():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 8)).astype(np.float32)
    a = split_half_cka(X, seed=42)
    b = split_half_cka(X, seed=42)
    assert np.isclose(a[0], b[0])
    assert np.isclose(a[1], b[1])


def test_spearman_monotone():
    X_clean = np.zeros((10, 2), dtype=np.float32)
    X_corr = np.stack([np.arange(10, dtype=np.float32), np.zeros(10, dtype=np.float32)], axis=1)
    y_true = np.zeros(10, dtype=np.int64)

    class _Probe:
        def predict_proba(self, X):  # noqa: ANN001
            del X
            conf = np.linspace(0.6, 1.0, 10, dtype=np.float32)
            return np.stack([1.0 - conf, conf], axis=1)

    out = track_b_summary(X_clean, X_corr, _Probe(), y_true, confidence_threshold=0.9)
    assert np.isclose(abs(out["spearman_drift_confidence"]), 1.0, atol=1e-6)


def test_spearman_constant_returns_nan():
    X_clean = np.zeros((10, 2), dtype=np.float32)
    X_corr = np.zeros((10, 2), dtype=np.float32)
    y_true = np.zeros(10, dtype=np.int64)

    class _Probe:
        def predict_proba(self, X):  # noqa: ANN001
            del X
            conf = np.linspace(0.6, 1.0, 10, dtype=np.float32)
            return np.stack([1.0 - conf, conf], axis=1)

    out = track_b_summary(X_clean, X_corr, _Probe(), y_true, confidence_threshold=0.9)
    assert np.isnan(out["spearman_drift_confidence"])


def test_track_b_summary_keys():
    X_clean = np.zeros((10, 2), dtype=np.float32)
    X_corr = np.ones((10, 2), dtype=np.float32)
    y_true = np.zeros(10, dtype=np.int64)

    class _Probe:
        def predict_proba(self, X):  # noqa: ANN001
            del X
            return np.tile(np.array([[0.4, 0.6]], dtype=np.float32), (10, 1))

    out = track_b_summary(X_clean, X_corr, _Probe(), y_true, confidence_threshold=0.9)
    assert set(out) == {
        "spearman_drift_confidence",
        "spearman_drift_correctness",
        "frac_overconfident_high_drift",
    }


def test_track_b_summary_frac_range():
    rng = np.random.default_rng(0)
    X_clean = rng.normal(size=(50, 8)).astype(np.float32)
    X_corr = rng.normal(size=(50, 8)).astype(np.float32)
    y_true = rng.integers(0, 3, size=(50,), dtype=np.int64)

    class _Probe:
        def predict_proba(self, X):  # noqa: ANN001
            del X
            probs = np.full((50, 3), 1 / 3, dtype=np.float32)
            probs[:, 0] = 0.8
            probs[:, 1:] = 0.1
            return probs

    out = track_b_summary(X_clean, X_corr, _Probe(), y_true, confidence_threshold=0.9)
    assert 0.0 <= out["frac_overconfident_high_drift"] <= 1.0


def test_track_b_summary_identical_clean_corrupted():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 8)).astype(np.float32)
    y_true = np.zeros(40, dtype=np.int64)

    class _Probe:
        def predict_proba(self, X):  # noqa: ANN001
            del X
            probs = np.zeros((40, 2), dtype=np.float32)
            probs[:, 0] = 0.95
            probs[:, 1] = 0.05
            return probs

    out = track_b_summary(X, X, _Probe(), y_true, confidence_threshold=0.9)
    assert np.isclose(out["frac_overconfident_high_drift"], 0.0)


def test_linear_cka_matches_inline_reference():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 12)).astype(np.float64)
    Y = rng.normal(size=(30, 12)).astype(np.float64)
    n = X.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ (X @ X.T) @ H
    Lc = H @ (Y @ Y.T) @ H
    ref = float(np.sum(Kc * Lc) / np.sqrt(np.sum(Kc * Kc) * np.sum(Lc * Lc)))
    assert abs(linear_cka(X, Y) - ref) < 1e-6


def test_linear_cka_large_n_agrees_with_unbiased():
    def _unbiased_hsic(K: np.ndarray, L: np.ndarray) -> float:
        n = K.shape[0]
        K0 = K.copy()
        L0 = L.copy()
        np.fill_diagonal(K0, 0.0)
        np.fill_diagonal(L0, 0.0)
        ones = np.ones((n, 1), dtype=np.float64)
        term1 = np.trace(K0 @ L0)
        term2 = float((ones.T @ K0 @ ones) * (ones.T @ L0 @ ones)) / ((n - 1) * (n - 2))
        term3 = float(ones.T @ K0 @ L0 @ ones) * (2.0 / (n - 2))
        return float((term1 + term2 - term3) / (n * (n - 3)))

    def _unbiased_cka(X: np.ndarray, Y: np.ndarray) -> float:
        K = X @ X.T
        L = Y @ Y.T
        hsic_xy = _unbiased_hsic(K, L)
        hsic_xx = _unbiased_hsic(K, K)
        hsic_yy = _unbiased_hsic(L, L)
        denom = np.sqrt(max(hsic_xx, 0.0) * max(hsic_yy, 0.0))
        if denom <= 0:
            return float("nan")
        return float(hsic_xy / denom)

    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 32)).astype(np.float64)
    Y = rng.normal(size=(500, 32)).astype(np.float64)
    biased = linear_cka(X, Y)
    unbiased = _unbiased_cka(X, Y)
    assert abs(biased - unbiased) < 0.02
