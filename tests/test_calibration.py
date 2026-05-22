"""Tests for calibration metrics helper."""

import numpy as np
import pytest

from torchgeo_bench.calibration import (
    apply_temperature,
    compute_calibration_metrics,
    fit_temperature,
)


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


def test_temperature_overconfident_singlelabel():
    """Sharp logits with many wrong predictions => T > 1 (flatten)."""
    rng = np.random.default_rng(0)
    n, c = 1000, 4
    y_true = rng.integers(0, c, size=n)
    # 50% accuracy but logits are very sharp => model is overconfident.
    pred = y_true.copy()
    flip = rng.choice(n, size=n // 2, replace=False)
    pred[flip] = (y_true[flip] + 1) % c
    logits = np.full((n, c), -5.0, dtype=np.float32)
    logits[np.arange(n), pred] = 5.0
    t = fit_temperature(logits, y_true, multi_label=False)
    assert t > 1.5


def test_temperature_underconfident_singlelabel():
    """Sharp logits with mostly correct predictions => T < 1 (sharpen)."""
    rng = np.random.default_rng(0)
    n, c = 500, 4
    y_true = rng.integers(0, c, size=n)
    logits = np.full((n, c), -0.5, dtype=np.float32)
    logits[np.arange(n), y_true] = 0.5
    t = fit_temperature(logits, y_true, multi_label=False)
    assert t < 1.0


def test_temperature_scaling_reduces_ece():
    """TS applied on overconfident logits should reduce ECE on the same split."""
    rng = np.random.default_rng(0)
    n, c = 1000, 4
    y_true = rng.integers(0, c, size=n)
    logits = np.full((n, c), -5.0, dtype=np.float32)
    logits[np.arange(n), y_true] = 5.0
    # Inject some errors so calibration isn't trivially zero.
    flip = rng.choice(n, size=300, replace=False)
    wrong = (y_true[flip] + 1) % c
    logits[flip, y_true[flip]] = -5.0
    logits[flip, wrong] = 5.0
    raw_probs = apply_temperature(logits, 1.0, multi_label=False)
    raw_cal = compute_calibration_metrics(y_true, raw_probs, multi_label=False)
    t = fit_temperature(logits, y_true, multi_label=False)
    ts_probs = apply_temperature(logits, t, multi_label=False)
    ts_cal = compute_calibration_metrics(y_true, ts_probs, multi_label=False)
    assert ts_cal["ece"] < raw_cal["ece"]


def test_temperature_multilabel_runs():
    """Multi-label TS produces a positive T and valid calibration."""
    rng = np.random.default_rng(0)
    n, c = 200, 5
    y_true = rng.integers(0, 2, size=(n, c))
    logits = rng.normal(size=(n, c)).astype(np.float32) * 3.0
    t = fit_temperature(logits, y_true, multi_label=True)
    assert t > 0
    probs = apply_temperature(logits, t, multi_label=True)
    out = compute_calibration_metrics(y_true, probs, multi_label=True)
    for v in out.values():
        assert 0.0 <= v <= 1.0
