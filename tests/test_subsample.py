"""Tests for stratified_subsample_indices (utils.py) — the label-budget selector."""

import numpy as np
import pytest

from torchgeo_bench.utils import stratified_subsample_indices

FRACTIONS = [0.01, 0.05, 0.1, 0.25, 1.0]


def _make_y(counts: dict[int, int]) -> np.ndarray:
    """Build a fixed-order label vector with the given per-class counts."""
    parts = [np.full(n, cls, dtype=np.int64) for cls, n in counts.items()]
    return np.concatenate(parts)


def test_stratification_per_class_counts():
    counts = {0: 100, 1: 40, 2: 7}
    y = _make_y(counts)
    f = 0.1
    idx = stratified_subsample_indices(y, f, seed=0)
    assert idx.dtype == np.int64
    assert np.array_equal(idx, np.sort(idx))
    assert np.all(np.diff(idx) > 0)  # sorted and unique
    sel = y[idx]
    for cls, n in counts.items():
        expected = min(n, max(1, round(f * n)))
        assert (sel == cls).sum() == expected
    assert set(np.unique(sel).tolist()) == set(counts)


def test_tiny_class_floor():
    # A class where f * n_c < 1 must still yield >= 1 example.
    y = _make_y({0: 100, 1: 3})
    idx = stratified_subsample_indices(y, 0.1, seed=0)
    assert (y[idx] == 1).sum() >= 1

    # A singleton class must appear at every fraction.
    y_single = _make_y({0: 200, 1: 1})
    for f in FRACTIONS:
        idx_f = stratified_subsample_indices(y_single, f, seed=1)
        assert (y_single[idx_f] == 1).sum() == 1


def test_nesting():
    y = _make_y({0: 100, 1: 40, 2: 7})
    seed = 3
    a = set(stratified_subsample_indices(y, 0.05, seed).tolist())
    b = set(stratified_subsample_indices(y, 0.1, seed).tolist())
    c = set(stratified_subsample_indices(y, 1.0, seed).tolist())
    assert a < b < c  # proper nested subsets


def test_determinism_and_model_independence():
    y = _make_y({0: 50, 1: 30, 2: 20})
    for f in (0.1, 0.25):
        i1 = stratified_subsample_indices(y, f, seed=7)
        i2 = stratified_subsample_indices(y, f, seed=7)
        assert np.array_equal(i1, i2)


def test_seed_changes_membership_not_sizes():
    y = _make_y({0: 100, 1: 60})
    f = 0.2
    i1 = stratified_subsample_indices(y, f, seed=0)
    i2 = stratified_subsample_indices(y, f, seed=1)
    for cls in (0, 1):
        assert (y[i1] == cls).sum() == (y[i2] == cls).sum()
    assert not np.array_equal(i1, i2)


def test_fraction_one_is_seed_independent():
    y = _make_y({0: 10, 1: 5, 2: 3})
    expected = np.arange(len(y))
    for seed in (0, 1, 99):
        idx = stratified_subsample_indices(y, 1.0, seed)
        assert np.array_equal(idx, expected)
        assert idx.dtype == np.int64


def test_fraction_out_of_range_raises():
    y = _make_y({0: 10, 1: 10})
    for bad in (0.0, -0.1, 1.5, 2.0):
        with pytest.raises(ValueError):
            stratified_subsample_indices(y, bad, seed=0)
