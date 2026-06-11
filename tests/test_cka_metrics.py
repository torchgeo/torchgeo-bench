import numpy as np

from torchgeo_bench.cka import metrics
from torchgeo_bench.cka.metrics import (
    bootstrap_cka_ci,
    cosine_drift,
    linear_cka,
    participation_ratio,
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
    # A rank-1 outer product stays rank-1 after per-column centering
    # (each column is v scaled by a constant, centering preserves the
    # single shared direction), so the centered PR remains ~1.
    v = np.arange(1, 11, dtype=np.float32)
    X = np.outer(v, v).astype(np.float32)
    assert np.isclose(participation_ratio(X), 1.0, atol=1e-6)


def test_participation_ratio_full_rank():
    # Centering the identity yields the centering matrix H (rank N-1),
    # which has N-1 unit eigenvalues and one zero eigenvalue, so the
    # centered PR of eye(N) is N-1 = 9, not N. This confirms PR measures
    # effective rank of the *centered* activations.
    X = np.eye(10, dtype=np.float32)
    assert np.isclose(participation_ratio(X), 9.0, atol=1e-6)


def test_participation_ratio_centers_before_svd():
    # A large constant mean direction plus low-variance noise: without
    # centering the mean dominates and PR collapses to ~1; with centering
    # the residual noise (full rank) drives PR well above 1.
    rng = np.random.default_rng(0)
    n, d = 50, 8
    mean_vec = rng.normal(size=d) * 100.0
    noise = rng.normal(size=(n, d)) * 0.1
    X = np.outer(np.ones(n), mean_vec) + noise

    pr = participation_ratio(X)
    assert np.isfinite(pr)
    assert pr >= 1.0
    assert pr > 2.0


def test_bootstrap_cka_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    n, d = 300, 8
    X = rng.normal(size=(n, d))
    Y = X + 0.3 * rng.normal(size=(n, d))  # strongly correlated
    point = linear_cka(X, Y)
    ci_low, ci_high, width = bootstrap_cka_ci(X, Y, n_boot=100, frac=0.8, seed=1)
    assert ci_high >= ci_low
    assert np.isclose(width, ci_high - ci_low)
    assert ci_low <= point <= ci_high


def test_bootstrap_cka_ci_subsamples_without_replacement(monkeypatch):
    rng = np.random.default_rng(0)
    n, d = 200, 6
    X = rng.normal(size=(n, d))
    Y = X + 0.2 * rng.normal(size=(n, d))

    recorded = []
    real_default_rng = np.random.default_rng

    class _SpyGen:
        def __init__(self, gen):
            self._gen = gen

        def choice(self, a, size=None, replace=True, **kw):  # noqa: ANN001
            idx = self._gen.choice(a, size=size, replace=replace, **kw)
            recorded.append((np.asarray(idx), replace))
            return idx

        def __getattr__(self, name):
            return getattr(self._gen, name)

    monkeypatch.setattr(
        metrics.np.random, "default_rng", lambda seed: _SpyGen(real_default_rng(seed))
    )

    frac, n_boot = 0.8, 5
    bootstrap_cka_ci(X, Y, n_boot=n_boot, frac=frac, seed=123)

    k = round(frac * n)
    assert len(recorded) == n_boot
    for idx, replace in recorded:
        assert replace is False
        assert idx.shape[0] == k
        assert np.unique(idx).shape[0] == k


def test_bootstrap_cka_ci_reproducible():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(250, 8))
    Y = X + 0.3 * rng.normal(size=(250, 8))
    a = bootstrap_cka_ci(X, Y, n_boot=50, frac=0.7, seed=7)
    b = bootstrap_cka_ci(X, Y, n_boot=50, frac=0.7, seed=7)
    assert np.allclose(a, b)


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


def test_track_b_drift_is_logit_space():
    # Probe with a 2x3 weight matrix; the z-axis (3rd feature) is orthogonal
    # to every coef_ row, so displacement along z changes embedding-space L2
    # drift but NOT logit-space drift. We craft x_corr so logit drift (== s,
    # increasing) and embedding drift (decreasing) have opposite orderings,
    # while confidence increases with s. The returned spearman(drift,
    # confidence) is +1 only if the internal drift is logit-space.
    coef = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    intercept = np.zeros(2)

    class _LinearProbe:
        def __init__(self):
            self.coef_ = coef
            self.intercept_ = intercept

    n = 5
    x_clean = np.zeros((n, 3))
    s = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    target_emb = np.array([20.0, 18.0, 16.0, 14.0, 12.0])  # decreasing
    t = np.sqrt(target_emb**2 - s**2)
    x_corr = np.column_stack([s, np.zeros(n), t])
    y_true = np.zeros(n, dtype=np.int64)

    out = track_b_summary(x_clean, x_corr, _LinearProbe(), y_true, confidence_threshold=0.9)
    # +1 for logit-space drift; embedding-space drift would give -1.
    assert np.isclose(out["spearman_drift_confidence"], 1.0, atol=1e-6)

    # Displacement purely orthogonal to W must not change any returned scalar.
    orth = np.column_stack(
        [np.zeros(n), np.zeros(n), np.array([7.0, -3.0, 11.0, -5.0, 2.0])]
    )
    out2 = track_b_summary(x_clean, x_corr + orth, _LinearProbe(), y_true, confidence_threshold=0.9)
    assert np.isclose(out2["spearman_drift_confidence"], out["spearman_drift_confidence"])
    assert np.isnan(out2["spearman_drift_correctness"]) and np.isnan(out["spearman_drift_correctness"])
    assert np.isclose(out2["frac_overconfident_high_drift"], out["frac_overconfident_high_drift"])


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
