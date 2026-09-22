"""Sklearn parity for CoordBench ridge fitting and alpha selection."""

import numpy as np
import pytest
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from torchgeo_bench.coordbench.probe import linear_probe_score


@pytest.mark.parametrize("standardize", [False, True])
@pytest.mark.parametrize("task_type", ["regression", "classification"])
@pytest.mark.parametrize("official", [False, True])
def test_ridge_matches_sklearn(*, standardize: bool, task_type: str, official: bool) -> None:
    rng = np.random.default_rng(42)
    features = (rng.normal(size=(90, 8)) + 12).astype(np.float32)
    features[:, -1] = 7  # Constant and duplicate columns exercise rank-deficient features.
    features[:, -2] = features[:, 0]
    labels = (features[:, 0] * 3 + rng.normal(size=90) + 100).astype(np.float32)
    if task_type == "classification":
        labels = np.digitize(labels, np.quantile(labels, [0.15, 0.4])).astype(np.int64)
    targets = labels[:, None]
    if task_type == "classification":
        targets = np.eye(3, dtype=np.float32)[labels]
    alphas = (0.01, 10.0, 10000.0)
    test_mask = np.arange(90) >= 60 if official else None
    pool = np.arange(60 if official else 90)
    folds = [pool[i::3] for i in range(3)]

    def score(train: np.ndarray, test: np.ndarray, alpha: float) -> float:
        x_train, x_test = torch.from_numpy(features[train]), torch.from_numpy(features[test])
        if standardize:
            mean = x_train.mean(0, keepdim=True)
            std = x_train.std(0, keepdim=True).clamp_min(1e-6)
            x_train, x_test = (x_train - mean) / std, (x_test - mean) / std
        model = Ridge(alpha=alpha, fit_intercept=True, solver="cholesky")
        model.fit(x_train.double().numpy(), targets[train].astype(np.float64))
        pred = model.predict(x_test.double().numpy())
        if task_type == "classification":
            return float(np.mean(pred.argmax(1) == labels[test]))
        return float(r2_score(targets[test], pred))

    cv_scores = [
        [score(np.setdiff1d(pool, fold), fold, alpha) for fold in folds] for alpha in alphas
    ]
    best = int(np.argmax(np.mean(cv_scores, axis=1)))
    expected = [score(pool, np.arange(60, 90), alphas[best])] if official else cv_scores[best]
    actual, fold_scores = linear_probe_score(
        features,
        labels,
        task_type,
        folds=3,
        alphas=alphas,
        test_mask=test_mask,
        fold_assign=np.arange(90) % 3,
        standardize=standardize,
    )
    np.testing.assert_allclose(fold_scores, expected, atol=2e-7, rtol=2e-7)
    assert actual == pytest.approx(np.mean(expected), abs=2e-7)


def test_fixed_alpha_skips_holdout_tuning(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_cv(*args: object, **kwargs: object) -> None:
        pytest.fail("A fixed alpha must not trigger inner CV")

    monkeypatch.setattr("torchgeo_bench.coordbench.probe._cv_alpha_scores", unexpected_cv)
    features = np.arange(40, dtype=np.float32).reshape(20, 2)
    labels = features[:, 0] + 50
    score, folds = linear_probe_score(
        features, labels, "regression", alphas=(0.01,), test_mask=np.arange(20) >= 15
    )
    assert score > 0.99
    assert folds == [score]
