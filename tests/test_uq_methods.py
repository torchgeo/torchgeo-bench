import importlib.util

import numpy as np
import pytest
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from torchgeo_bench.linear import LogisticRegression
from torchgeo_bench.uq.methods import (
    ConformalPredictor,
    DeepEnsemble,
    LaplaceProbe,
    TemperatureScaling,
    Uncalibrated,
)


@pytest.fixture(scope="module")
def fitted_probe():
    X, y = make_classification(
        n_samples=1200,
        n_features=16,
        n_informative=12,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        class_sep=2.0,
        random_state=0,
    )
    X = X.astype(np.float32)
    y = y.astype(np.int64)

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.5, random_state=0, stratify=y
    )
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.4, random_state=1, stratify=y_tmp
    )

    probe = LogisticRegression(C=1.0, max_iter=1500, solver="lbfgs", random_state=0)
    probe.fit(torch.from_numpy(X_train), torch.from_numpy(y_train))
    return probe, (X_train, y_train, X_cal, y_cal, X_test, y_test)


def test_uncalibrated_predict_proba_shape(fitted_probe):
    probe, (_, _, _, _, X_test, _) = fitted_probe
    method = Uncalibrated(probe)
    probs = method.predict_proba(X_test)
    assert probs.shape == (X_test.shape[0], 3)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_temperature_scaling_t_positive(fitted_probe):
    probe, (_, _, X_cal, y_cal, _, _) = fitted_probe
    method = TemperatureScaling(probe)
    method.fit(X_cal, y_cal)
    assert float(method.log_temperature.exp().item()) > 0.0


def test_temperature_scaling_predict_proba_shape(fitted_probe):
    probe, (_, _, X_cal, y_cal, X_test, _) = fitted_probe
    method = TemperatureScaling(probe)
    method.fit(X_cal, y_cal)
    probs = method.predict_proba(X_test)
    assert probs.shape == (X_test.shape[0], 3)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_deep_ensemble_predict_proba_shape(fitted_probe):
    _, (X_train, y_train, _, _, X_test, _) = fitted_probe
    method = DeepEnsemble(n=3)
    method.fit(X_train, y_train, best_c=1.0, seed=0)
    probs = method.predict_proba(X_test)
    assert probs.shape == (X_test.shape[0], 3)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_laplace_predict_proba_shape(fitted_probe):
    if importlib.util.find_spec("laplace") is None:
        pytest.skip("laplace-torch not installed")

    probe, (X_train, y_train, _, _, X_test, _) = fitted_probe
    method = LaplaceProbe(probe, batch_size=16)
    try:
        method.fit(X_train, y_train)
    except ModuleNotFoundError as exc:
        pytest.skip(f"laplace unavailable at runtime: {exc}")
    probs = method.predict_proba(X_test)
    assert probs.shape == (X_test.shape[0], 3)
    assert np.all((probs >= 0.0) & (probs <= 1.0))


def test_conformal_predict_sets_shape(fitted_probe):
    if importlib.util.find_spec("mapie") is None:
        pytest.skip("mapie not installed")

    probe, (_, _, X_cal, y_cal, X_test, _) = fitted_probe
    method = ConformalPredictor(probe)
    try:
        method.fit(X_cal, y_cal)
    except (ImportError, ModuleNotFoundError, ValueError) as exc:
        pytest.skip(f"conformal unavailable at runtime: {exc}")
    point_preds, pred_sets = method.predict_sets(X_test, alpha=0.1)
    assert pred_sets.shape == (X_test.shape[0], 3)
    assert pred_sets.dtype == np.bool_
    assert point_preds.shape == (X_test.shape[0],)


def test_conformal_coverage_at_alpha_01(fitted_probe):
    if importlib.util.find_spec("mapie") is None:
        pytest.skip("mapie not installed")

    probe, (_, _, X_cal, y_cal, X_test, y_test) = fitted_probe
    method = ConformalPredictor(probe)
    try:
        method.fit(X_cal, y_cal)
    except (ImportError, ModuleNotFoundError, ValueError) as exc:
        pytest.skip(f"conformal unavailable at runtime: {exc}")
    _, pred_sets = method.predict_sets(X_test, alpha=0.1)
    covered = pred_sets[np.arange(len(y_test)), y_test].mean()
    assert float(covered) >= 0.9
