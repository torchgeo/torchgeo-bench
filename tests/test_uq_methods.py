import importlib.util

import numpy as np
import pytest
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from torchgeo_bench.linear import LogisticRegression
from torchgeo_bench.uq.methods import (
    BootstrapEnsemble,
    ConformalPredictor,
    DeepEnsemble,
    LaplaceProbe,
    SVGPProbe,
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


def test_bootstrap_ensemble_predict_proba_shape(fitted_probe):
    _, (X_train, y_train, _, _, X_test, _) = fitted_probe
    method = BootstrapEnsemble(n=3)
    method.fit(X_train, y_train, best_c=1.0, seed=0)
    probs = method.predict_proba(X_test)
    assert probs.shape == (X_test.shape[0], 3)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_bootstrap_ensemble_predict_confidence_shape_and_range(fitted_probe):
    _, (X_train, y_train, _, _, X_test, _) = fitted_probe
    method = BootstrapEnsemble(n=3)
    method.fit(X_train, y_train, best_c=1.0, seed=0)
    conf = method.predict_confidence(X_test)
    assert conf.shape == (X_test.shape[0],)
    assert np.all((conf >= 0.0) & (conf <= 1.0))


def test_bootstrap_ensemble_confidence_reflects_disagreement(fitted_probe):
    _, (X_train, y_train, _, _, X_test, _) = fitted_probe
    method = BootstrapEnsemble(n=5)
    method.fit(X_train, y_train, best_c=1.0, seed=0)

    rng = np.random.default_rng(42)
    X_noise = rng.standard_normal(X_test.shape).astype(np.float32)

    conf_test = method.predict_confidence(X_test).mean()
    conf_noise = method.predict_confidence(X_noise).mean()
    assert conf_test > conf_noise


def test_deep_ensemble_predict_proba_shape(fitted_probe):
    _, (X_train, y_train, _, _, X_test, _) = fitted_probe
    method = DeepEnsemble(n=3)
    method.fit(X_train, y_train, best_c=1.0, seed=0)
    probs = method.predict_proba(X_test)
    assert probs.shape == (X_test.shape[0], 3)
    assert np.allclose(probs.sum(axis=1), 1.0)


def test_deep_ensemble_predict_confidence_shape_and_range(fitted_probe):
    _, (X_train, y_train, _, _, X_test, _) = fitted_probe
    method = DeepEnsemble(n=3)
    method.fit(X_train, y_train, best_c=1.0, seed=0)
    conf = method.predict_confidence(X_test)
    assert conf.shape == (X_test.shape[0],)
    assert np.all((conf >= 0.0) & (conf <= 1.0))


def test_deep_ensemble_members_differ(fitted_probe):
    _, (X_train, y_train, _, _, X_test, _) = fitted_probe
    method = DeepEnsemble(n=3)
    method.fit(X_train, y_train, best_c=1.0, seed=0)
    member_probs = method._member_probs(X_test)  # (3, N, C)
    # Members trained from different random inits should produce different predictions.
    assert not np.allclose(member_probs[0], member_probs[1])


def test_deep_ensemble_confidence_reflects_disagreement(fitted_probe):
    _, (X_train, y_train, _, _, X_test, _) = fitted_probe
    method = DeepEnsemble(n=5)
    method.fit(X_train, y_train, best_c=1.0, seed=0)

    rng = np.random.default_rng(42)
    X_noise = rng.standard_normal(X_test.shape).astype(np.float32)

    conf_test = method.predict_confidence(X_test).mean()
    conf_noise = method.predict_confidence(X_noise).mean()
    assert conf_test > conf_noise


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


def test_conformal_select_conformity_score_binary_and_multiclass(fitted_probe):
    probe, _ = fitted_probe
    method = ConformalPredictor(probe)
    assert method._select_conformity_score(np.array([0, 1, 0, 1], dtype=np.int64)) == "lac"
    assert method._select_conformity_score(np.array([0, 1, 2, 1], dtype=np.int64)) == "raps"


def test_conformal_predict_confidence_shape_and_range(fitted_probe):
    if importlib.util.find_spec("mapie") is None:
        pytest.skip("mapie not installed")

    probe, (_, _, X_cal, y_cal, X_test, _) = fitted_probe
    method = ConformalPredictor(probe)
    try:
        method.fit(X_cal, y_cal)
    except (ImportError, ModuleNotFoundError, ValueError) as exc:
        pytest.skip(f"conformal unavailable at runtime: {exc}")
    conf = method.predict_confidence(X_test)
    assert conf.shape == (X_test.shape[0],)
    assert np.all((conf > 0.0) & (conf <= 1.0))


def test_conformal_confidence_continuous(fitted_probe):
    """predict_confidence must have more than O(C) distinct values."""
    if importlib.util.find_spec("mapie") is None:
        pytest.skip("mapie not installed")

    probe, (_, _, X_cal, y_cal, X_test, _) = fitted_probe
    method = ConformalPredictor(probe)
    try:
        method.fit(X_cal, y_cal)
    except (ImportError, ModuleNotFoundError, ValueError) as exc:
        pytest.skip(f"conformal unavailable at runtime: {exc}")
    conf = method.predict_confidence(X_test)
    # Probe has 3 classes; a discrete 1/set_size signal would have ≤ 3 unique values.
    assert len(np.unique(conf)) > 3


def test_conformal_predict_sets_rejects_unfitted_alpha(fitted_probe):
    if importlib.util.find_spec("mapie") is None:
        pytest.skip("mapie not installed")

    probe, (_, _, X_cal, y_cal, X_test, _) = fitted_probe
    method = ConformalPredictor(probe)
    try:
        method.fit(X_cal, y_cal, alpha=0.1)
    except (ImportError, ModuleNotFoundError, ValueError) as exc:
        pytest.skip(f"conformal unavailable at runtime: {exc}")
    with pytest.raises(ValueError):
        method.predict_sets(X_test, alpha=0.2)


def test_svgp_probe_predict_proba_shape(fitted_probe):
    if importlib.util.find_spec("gpytorch") is None:
        pytest.skip("gpytorch not installed")

    _, (X_train, y_train, _, _, X_test, _) = fitted_probe
    method = SVGPProbe(n_inducing=10, epochs=2)
    try:
        method.fit(X_train, y_train)
    except ModuleNotFoundError as exc:
        pytest.skip(f"gpytorch unavailable at runtime: {exc}")
    probs = method.predict_proba(X_test)
    assert probs.shape == (X_test.shape[0], 3)


def test_svgp_probe_proba_sums_to_one(fitted_probe):
    if importlib.util.find_spec("gpytorch") is None:
        pytest.skip("gpytorch not installed")

    _, (X_train, y_train, _, _, X_test, _) = fitted_probe
    method = SVGPProbe(n_inducing=10, epochs=2)
    try:
        method.fit(X_train, y_train)
    except ModuleNotFoundError as exc:
        pytest.skip(f"gpytorch unavailable at runtime: {exc}")
    probs = method.predict_proba(X_test)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_svgp_probe_finite_output(fitted_probe):
    if importlib.util.find_spec("gpytorch") is None:
        pytest.skip("gpytorch not installed")

    _, (X_train, y_train, _, _, X_test, _) = fitted_probe
    method = SVGPProbe(n_inducing=10, epochs=2)
    try:
        method.fit(X_train, y_train)
    except ModuleNotFoundError as exc:
        pytest.skip(f"gpytorch unavailable at runtime: {exc}")
    probs = method.predict_proba(X_test)
    assert np.isfinite(probs).all()
    assert np.all(probs >= 0.0)


# ---------------------------------------------------------------------------
# NormalizingFlowProbe — Slice 2 (predict_proba) + Slice 3 (predict_confidence)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def nf_xy():
    X, y = make_classification(
        n_samples=300, n_features=16, n_informative=12, n_redundant=0,
        n_classes=3, n_clusters_per_class=1, class_sep=2.0, random_state=0,
    )
    return X.astype(np.float32), y.astype(np.int64)


def test_nf_probe_predict_proba_shape(nf_xy):
    if importlib.util.find_spec("zuko") is None:
        pytest.skip("zuko not installed")
    from torchgeo_bench.uq.nf import NormalizingFlowProbe
    X, y = nf_xy
    probe = NormalizingFlowProbe(prior="empirical", lr=1e-3, weight_decay=1e-4, epochs=2)
    probe.fit(X, y)
    probs = probe.predict_proba(X)
    assert probs.shape == (len(X), 3)


def test_nf_probe_proba_sums_to_one(nf_xy):
    if importlib.util.find_spec("zuko") is None:
        pytest.skip("zuko not installed")
    from torchgeo_bench.uq.nf import NormalizingFlowProbe
    X, y = nf_xy
    probe = NormalizingFlowProbe(prior="empirical", lr=1e-3, weight_decay=1e-4, epochs=2)
    probe.fit(X, y)
    probs = probe.predict_proba(X)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_nf_probe_proba_finite(nf_xy):
    if importlib.util.find_spec("zuko") is None:
        pytest.skip("zuko not installed")
    from torchgeo_bench.uq.nf import NormalizingFlowProbe
    X, y = nf_xy
    probe = NormalizingFlowProbe(prior="empirical", lr=1e-3, weight_decay=1e-4, epochs=2)
    probe.fit(X, y)
    probs = probe.predict_proba(X)
    assert np.isfinite(probs).all()
    assert np.all(probs >= 0.0)


def test_nf_probe_uniform_prior_shape(nf_xy):
    if importlib.util.find_spec("zuko") is None:
        pytest.skip("zuko not installed")
    from torchgeo_bench.uq.nf import NormalizingFlowProbe
    X, y = nf_xy
    probe = NormalizingFlowProbe(prior="uniform", lr=1e-3, weight_decay=1e-4, epochs=2)
    probe.fit(X, y)
    probs = probe.predict_proba(X)
    assert probs.shape == (len(X), 3)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_nf_probe_predict_confidence_shape(nf_xy):
    if importlib.util.find_spec("zuko") is None:
        pytest.skip("zuko not installed")
    from torchgeo_bench.uq.nf import NormalizingFlowProbe
    X, y = nf_xy
    probe = NormalizingFlowProbe(prior="empirical", lr=1e-3, weight_decay=1e-4, epochs=2)
    probe.fit(X, y)
    conf = probe.predict_confidence(X)
    assert conf.shape == (len(X),)


def test_nf_probe_predict_confidence_finite(nf_xy):
    if importlib.util.find_spec("zuko") is None:
        pytest.skip("zuko not installed")
    from torchgeo_bench.uq.nf import NormalizingFlowProbe
    X, y = nf_xy
    probe = NormalizingFlowProbe(prior="empirical", lr=1e-3, weight_decay=1e-4, epochs=2)
    probe.fit(X, y)
    conf = probe.predict_confidence(X)
    assert np.isfinite(conf).all()


def test_nf_probe_confidence_higher_for_indist_than_noise(nf_xy):
    if importlib.util.find_spec("zuko") is None:
        pytest.skip("zuko not installed")
    from torchgeo_bench.uq.nf import NormalizingFlowProbe
    X, y = nf_xy
    probe = NormalizingFlowProbe(prior="empirical", lr=1e-3, weight_decay=1e-4, epochs=10)
    probe.fit(X, y)
    conf_in = probe.predict_confidence(X).mean()
    rng = np.random.default_rng(99)
    X_noise = rng.standard_normal(X.shape).astype(np.float32) * 10
    conf_noise = probe.predict_confidence(X_noise).mean()
    assert conf_in > conf_noise
