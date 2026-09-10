"""Regression test: a diverged candidate C must not crash the whole C sweep."""

from unittest import mock

import numpy as np
import pytest
from omegaconf import OmegaConf

from torchgeo_bench.main import LinearProbeDivergedError, evaluate_logistic
from torchgeo_bench.utils import FeatureSplit, FeatureSplits


def _multilabel_data(n_classes: int = 3):
    rng = np.random.default_rng(0)
    x_train = rng.normal(size=(16, 4)).astype(np.float32)
    x_val = rng.normal(size=(8, 4)).astype(np.float32)
    x_test = rng.normal(size=(8, 4)).astype(np.float32)
    y_train = rng.integers(0, 2, size=(16, n_classes)).astype(np.float32)
    y_val = rng.integers(0, 2, size=(8, n_classes)).astype(np.float32)
    y_test = rng.integers(0, 2, size=(8, n_classes)).astype(np.float32)
    return x_train, y_train, x_val, y_val, x_test, y_test


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


def test_nan_candidate_c_does_not_crash_sweep():
    x_train, y_train, x_val, y_val, x_test, y_test = _multilabel_data()

    with mock.patch("torchgeo_bench.main.LogisticRegression", _StubModel):
        metric, lo, hi, best_c, calibration, calibration_ts = evaluate_logistic(
            FeatureSplits(
                FeatureSplit(x_train, y_train),
                FeatureSplit(x_val, y_val),
                FeatureSplit(x_test, y_test),
            ),
            c_values=[0.1, 1.0, 10.0],
            cfg=OmegaConf.create(
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
            ),
        )

    assert best_c != 1.0
    assert not np.isnan(metric)
    del lo, hi, calibration, calibration_ts


class _AlwaysDivergesModel(_StubModel):
    def predict_proba(self, x):
        return np.full((x.shape[0], self._n_classes), np.nan, dtype=np.float32)


def test_total_divergence_raises_named_error_not_bare_assert():
    """A dedicated error lets the runner skip failed probes without hiding unrelated failures."""
    x_train, y_train, x_val, y_val, x_test, y_test = _multilabel_data()

    with (
        mock.patch("torchgeo_bench.main.LogisticRegression", _AlwaysDivergesModel),
        pytest.raises(LinearProbeDivergedError),
    ):
        evaluate_logistic(
            FeatureSplits(
                FeatureSplit(x_train, y_train),
                FeatureSplit(x_val, y_val),
                FeatureSplit(x_test, y_test),
            ),
            c_values=[0.1, 1.0, 10.0],
            cfg=OmegaConf.create(
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
            ),
        )
