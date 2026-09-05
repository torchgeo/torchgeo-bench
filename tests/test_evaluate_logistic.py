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
    """Stands in for LogisticRegression: NaN val_scores for one C, finite otherwise."""

    def __init__(self, C, **kwargs):
        del kwargs
        self._c = C
        self._n_classes = 3

    def fit(self, x, y):
        del x, y

    def predict_proba(self, x):
        n = x.shape[0]
        if self._c == 1.0:
            # The candidate whose weights "diverged" mid-sweep.
            return np.full((n, self._n_classes), np.nan, dtype=np.float32)
        rng = np.random.default_rng(int(self._c * 1000) % 2**31)
        return rng.uniform(size=(n, self._n_classes)).astype(np.float32)

    def predict(self, x):
        return (self.predict_proba(x) > 0.5).astype(np.int64)


def test_nan_candidate_c_does_not_crash_sweep():
    """One divergent C in the sweep must be skipped, not raise past the whole run."""
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
    """Every candidate C produces non-finite scores -- no usable C exists at all."""

    def predict_proba(self, x):
        return np.full((x.shape[0], self._n_classes), np.nan, dtype=np.float32)


def test_total_divergence_raises_named_error_not_bare_assert():
    """When every C in the sweep diverges, the caller must get a catchable error.

    A bare AssertionError can't be narrowly caught by main.py's per-row handler
    without also swallowing unrelated bugs, so this must be the dedicated
    LinearProbeDivergedError -- one (backbone, dataset) pairing failing to
    converge shouldn't be indistinguishable from a real assertion violation
    elsewhere in the sweep.
    """
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
