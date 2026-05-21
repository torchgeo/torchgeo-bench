"""Tests for calibration metrics helper."""

import numpy as np
import pytest

from torchgeo_bench.calibration import compute_calibration_metrics


def test_perfect_calibration_singlelabel():
    """One-hot probabilities matching labels => zero calibration error."""
    rng = np.random.default_rng(0)
    n, c = 200, 4
    y_true = rng.integers(0, c, size=n)
    y_proba = np.eye(c, dtype=np.float32)[y_true]
    out = compute_calibration_metrics(y_true, y_proba, multi_label=False)
    assert set(out) == {"ece", "rms_ce", "mce"}
    for v in out.values():
        assert v == pytest.approx(0.0, abs=1e-5)


def test_worst_calibration_singlelabel():
    """High-confidence wrong predictions => ECE near 1."""
    n, c = 200, 4
    y_true = np.zeros(n, dtype=np.int64)
    y_proba = np.zeros((n, c), dtype=np.float32)
    y_proba[:, 1] = 1.0  # always confidently predict class 1
    out = compute_calibration_metrics(y_true, y_proba, multi_label=False)
    assert out["ece"] == pytest.approx(1.0, abs=1e-4)
    assert out["mce"] == pytest.approx(1.0, abs=1e-4)


def test_multilabel_shapes_and_range():
    """Multi-label path returns the same keys with values in [0, 1]."""
    rng = np.random.default_rng(0)
    n, c = 100, 5
    y_true = rng.integers(0, 2, size=(n, c))
    y_proba = rng.uniform(0.0, 1.0, size=(n, c)).astype(np.float32)
    out = compute_calibration_metrics(y_true, y_proba, multi_label=True)
    assert set(out) == {"ece", "rms_ce", "mce"}
    for v in out.values():
        assert 0.0 <= v <= 1.0


def test_multilabel_perfect_calibration():
    """Hard 0/1 probabilities matching labels => zero per-label error."""
    rng = np.random.default_rng(1)
    n, c = 80, 3
    y_true = rng.integers(0, 2, size=(n, c))
    # Ensure each column has both classes so no column is skipped.
    y_true[0] = 0
    y_true[1] = 1
    y_proba = y_true.astype(np.float32)
    out = compute_calibration_metrics(y_true, y_proba, multi_label=True)
    for v in out.values():
        assert v == pytest.approx(0.0, abs=1e-5)
