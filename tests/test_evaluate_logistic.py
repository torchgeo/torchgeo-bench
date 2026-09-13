"""Regression test: a diverged candidate C must not crash the whole C sweep."""

from unittest import mock

import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import average_precision_score

from torchgeo_bench.main import LinearProbeDivergedError, evaluate_logistic
from torchgeo_bench.utils import FeatureSplit, FeatureSplits


@pytest.fixture
def feature_splits() -> FeatureSplits[np.ndarray]:
    rng = np.random.default_rng(0)
    splits = [
        FeatureSplit(
            rng.normal(size=(n, 4)).astype(np.float32),
            rng.integers(0, 2, size=(n, 3)).astype(np.float32),
        )
        for n in (16, 8, 8)
    ]
    return FeatureSplits(*splits)


@pytest.fixture
def probe_config() -> DictConfig:
    return OmegaConf.create(
        {
            "seed": 0,
            "device": "cpu",
            "verbose": False,
            "eval": {
                "bootstrap": 5,
                "merge_val": False,
                "calibration": {"temp_scale": False},
            },
        }
    )


class _StubModel:
    """Simulate divergence only at C=1."""

    def __init__(self, C, **kwargs):
        del kwargs
        self._c = C
        self._n_classes = 3

    def fit(self, x, y):
        del x, y

    def predict_proba(self, x):
        n = x.shape[0]
        if self._c == 1.0:
            return np.full((n, self._n_classes), np.nan, dtype=np.float32)
        rng = np.random.default_rng(int(self._c * 1000) % 2**31)
        return rng.uniform(size=(n, self._n_classes)).astype(np.float32)

    def predict(self, x):
        return (self.predict_proba(x) > 0.5).astype(np.int64)


def test_nan_candidate_c_does_not_crash_sweep(
    feature_splits: FeatureSplits[np.ndarray], probe_config: DictConfig
) -> None:
    with mock.patch("torchgeo_bench.main.LogisticRegression", _StubModel):
        metric, lo, hi, best_c, calibration, calibration_ts = evaluate_logistic(
            feature_splits,
            c_values=[0.1, 1.0, 10.0],
            cfg=probe_config,
        )

    expected_c = max(
        (0.1, 10.0),
        key=lambda c: average_precision_score(
            feature_splits.val.labels,
            _StubModel(c).predict_proba(feature_splits.val.features),
            average="micro",
        ),
    )
    assert best_c == expected_c
    assert metric == pytest.approx(
        average_precision_score(
            feature_splits.test.labels,
            _StubModel(expected_c).predict_proba(feature_splits.test.features),
            average="micro",
        )
    )
    assert 0 <= lo <= hi <= 1
    assert set(calibration) == {"ece", "rms_ce", "mce"}
    assert all(value is None for value in calibration_ts.values())


class _AlwaysDivergesModel(_StubModel):
    def predict_proba(self, x):
        return np.full((x.shape[0], self._n_classes), np.nan, dtype=np.float32)


def test_total_divergence_raises_named_error_not_bare_assert(
    feature_splits: FeatureSplits[np.ndarray], probe_config: DictConfig
) -> None:
    """A dedicated error lets the runner skip failed probes without hiding unrelated failures."""
    with (
        mock.patch("torchgeo_bench.main.LogisticRegression", _AlwaysDivergesModel),
        pytest.raises(LinearProbeDivergedError),
    ):
        evaluate_logistic(
            feature_splits,
            c_values=[0.1, 1.0, 10.0],
            cfg=probe_config,
        )


def test_unexpected_fit_errors_are_not_treated_as_divergence(
    feature_splits: FeatureSplits[np.ndarray], probe_config: DictConfig
) -> None:
    error = ValueError("invalid feature implementation")
    with (
        mock.patch("torchgeo_bench.main.LogisticRegression", _StubModel),
        mock.patch.object(_StubModel, "fit", side_effect=error),
        pytest.raises(ValueError, match="invalid feature implementation") as caught,
    ):
        evaluate_logistic(feature_splits, c_values=[0.1, 1.0], cfg=probe_config)
    assert caught.value is error
