"""Tests for LogisticRegression (linear.py) — validation, fitting, inference."""

import numpy as np
import pytest
import torch

from torchgeo_bench.linear import LogisticRegression


def _xy(
    n: int = 50, d: int = 8, n_classes: int = 3, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = torch.Generator()
    rng.manual_seed(seed)
    X = torch.randn(n, d, generator=rng)
    y = torch.randint(0, n_classes, (n,), generator=rng)
    return X, y


def _xy_ml(
    n: int = 50, d: int = 8, n_classes: int = 4, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    rng = torch.Generator()
    rng.manual_seed(seed)
    X = torch.randn(n, d, generator=rng)
    y = torch.randint(0, 2, (n, n_classes), generator=rng).float()
    return X, y


# ---------------------------------------------------------------------------
# __init__ validation
# ---------------------------------------------------------------------------


def test_invalid_c_raises():
    with pytest.raises(ValueError, match="C must be > 0"):
        LogisticRegression(C=0.0)


def test_invalid_c_negative_raises():
    with pytest.raises(ValueError, match="C must be > 0"):
        LogisticRegression(C=-1.0)


def test_invalid_solver_raises():
    with pytest.raises(ValueError, match="solver must be one of"):
        LogisticRegression(solver="sgd")


def test_cuda_fallback_to_cpu(monkeypatch):
    """When CUDA unavailable, device should silently fall back to CPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = LogisticRegression(device="cuda")
    assert model.device.type == "cpu"


# ---------------------------------------------------------------------------
# fit validation
# ---------------------------------------------------------------------------


def test_fit_non_tensor_raises():
    model = LogisticRegression()
    with pytest.raises(TypeError, match="torch.Tensor"):
        model.fit(np.ones((10, 4)), torch.zeros(10, dtype=torch.long))  # type: ignore[arg-type]


def test_fit_y_non_tensor_raises():
    model = LogisticRegression()
    X = torch.randn(10, 4)
    with pytest.raises(TypeError, match="torch.Tensor"):
        model.fit(X, np.zeros(10))  # type: ignore[arg-type]


def test_fit_x_wrong_ndim_raises():
    model = LogisticRegression()
    with pytest.raises(ValueError, match="X must be 2D"):
        model.fit(torch.randn(10, 4, 4), torch.zeros(10, dtype=torch.long))


def test_fit_multilabel_y_wrong_ndim_raises():
    model = LogisticRegression(multi_label=True)
    with pytest.raises(ValueError, match="Multi-label"):
        model.fit(torch.randn(10, 4), torch.zeros(10, dtype=torch.long))


def test_fit_singlelabel_y_wrong_ndim_raises():
    model = LogisticRegression(multi_label=False)
    with pytest.raises(ValueError, match="y must be 1D"):
        model.fit(torch.randn(10, 4), torch.zeros((10, 2), dtype=torch.long))


def test_fit_empty_data_raises():
    model = LogisticRegression()
    with pytest.raises(ValueError, match="Empty"):
        model.fit(torch.zeros(0, 4), torch.zeros(0, dtype=torch.long))


def test_fit_xy_length_mismatch_raises():
    model = LogisticRegression()
    with pytest.raises(ValueError, match="length mismatch"):
        model.fit(torch.randn(10, 4), torch.zeros(5, dtype=torch.long))


# ---------------------------------------------------------------------------
# Fitting and inference — single-label
# ---------------------------------------------------------------------------


def test_fit_and_predict_singlelabel():
    X, y = _xy()
    model = LogisticRegression(C=1.0, max_iter=50, random_state=0)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X),)
    assert set(preds).issubset({0, 1, 2})


def test_predict_proba_shape_and_range():
    X, y = _xy()
    model = LogisticRegression(C=1.0, max_iter=30, random_state=0)
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 3)
    assert np.all(proba >= 0) and np.all(proba <= 1)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_decision_function_shape():
    X, y = _xy()
    model = LogisticRegression(C=1.0, max_iter=30, random_state=0)
    model.fit(X, y)
    logits = model.decision_function(X)
    assert logits.shape == (len(X), 3)


def test_coef_intercept_shapes():
    X, y = _xy(d=8, n_classes=3)
    model = LogisticRegression(C=1.0, max_iter=10, random_state=0)
    model.fit(X, y)
    assert model.coef_.shape == (3, 8)
    assert model.intercept_.shape == (3,)


def test_coef_before_fit_raises():
    model = LogisticRegression()
    with pytest.raises(AttributeError, match="not fitted"):
        _ = model.coef_


# ---------------------------------------------------------------------------
# Fitting and inference — multi-label
# ---------------------------------------------------------------------------


def test_fit_and_predict_multilabel():
    X, y = _xy_ml()
    model = LogisticRegression(C=1.0, max_iter=50, random_state=0, multi_label=True)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (len(X), 4)
    assert set(preds.flatten()).issubset({0, 1})


def test_predict_proba_multilabel_range():
    X, y = _xy_ml()
    model = LogisticRegression(C=1.0, max_iter=30, random_state=0, multi_label=True)
    model.fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X), 4)
    assert np.all(proba >= 0) and np.all(proba <= 1)


# ---------------------------------------------------------------------------
# predict_proba / decision_function validation
# ---------------------------------------------------------------------------


def test_predict_proba_before_fit_raises():
    model = LogisticRegression()
    with pytest.raises(RuntimeError, match="not been fit"):
        model.predict_proba(torch.randn(5, 4))


def test_predict_proba_non_tensor_raises():
    X, y = _xy()
    model = LogisticRegression(C=1.0, max_iter=10, random_state=0)
    model.fit(X, y)
    with pytest.raises(TypeError, match="torch.Tensor"):
        model.predict_proba(np.ones((5, 8)))  # type: ignore[arg-type]


def test_predict_proba_wrong_ndim_raises():
    X, y = _xy()
    model = LogisticRegression(C=1.0, max_iter=10, random_state=0)
    model.fit(X, y)
    with pytest.raises(ValueError, match="X must be 2D"):
        model.predict_proba(torch.randn(5, 4, 4))


def test_decision_function_before_fit_raises():
    model = LogisticRegression()
    with pytest.raises(RuntimeError, match="not been fit"):
        model.decision_function(torch.randn(5, 4))


# ---------------------------------------------------------------------------
# LBFGS solver
# ---------------------------------------------------------------------------


def test_lbfgs_solver_fits():
    X, y = _xy(n=40, d=6, n_classes=2)
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=20, random_state=0)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (40,)
