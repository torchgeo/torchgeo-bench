import numpy as np

from torchgeo_bench.uq.metrics import (
    aurc,
    brier_score,
    ece,
    empirical_coverage,
    excess_aurc,
    max_probability,
    mean_set_size,
    nll,
    normalized_predictive_entropy,
    predictive_entropy,
    raw_aurc,
    selective_accuracy,
)


def test_ece_perfect_calibration():
    # Single-confidence bin with exact agreement between confidence (0.7)
    # and empirical accuracy (70/100).
    probs = np.tile(np.array([[0.7, 0.3]], dtype=np.float64), (100, 1))
    y_true = np.array([0] * 70 + [1] * 30, dtype=np.int64)
    assert abs(ece(probs, y_true, n_bins=4)) < 1e-8


def test_ece_worst_case():
    probs = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float64,
    )
    y_true = np.array([1, 1, 1, 1], dtype=np.int64)
    assert abs(ece(probs, y_true, n_bins=2) - 1.0) < 1e-8


def test_ece_equal_width_and_equal_mass_modes():
    probs = np.array(
        [
            [0.6721, 0.2847, 0.0432],
            [0.0095, 0.4667, 0.5238],
            [0.3227, 0.3881, 0.2892],
            [0.5332, 0.4652, 0.0016],
            [0.5291, 0.0207, 0.4502],
            [0.1112, 0.5462, 0.3426],
            [0.3992, 0.5630, 0.0377],
            [0.0862, 0.4650, 0.4488],
        ],
        dtype=np.float64,
    )
    y_true = np.array([0, 1, 2, 1, 1, 2, 2, 2], dtype=np.int64)
    ece_width = ece(probs, y_true, n_bins=5, binning="equal_width")
    ece_mass = ece(probs, y_true, n_bins=5, binning="equal_mass")
    assert np.isclose(ece_width, 0.4845423426343915)
    assert np.isclose(ece_mass, 0.4025625)
    assert not np.isclose(ece_width, ece_mass)


def test_ece_invalid_binning_raises():
    probs = np.array([[0.7, 0.3], [0.2, 0.8]], dtype=np.float64)
    y_true = np.array([0, 1], dtype=np.int64)
    with np.testing.assert_raises(ValueError):
        ece(probs, y_true, n_bins=4, binning="invalid")


def test_nll_uniform():
    C = 5
    probs = np.full((20, C), 1.0 / C, dtype=np.float64)
    y_true = np.arange(20) % C
    assert np.isclose(nll(probs, y_true), np.log(C), atol=1e-5)


def test_brier_score_range():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(200, 4))
    exps = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exps / exps.sum(axis=1, keepdims=True)
    y_true = rng.integers(0, 4, size=200)
    brier = brier_score(probs, y_true)
    assert 0.0 <= brier <= 2.0

    perfect = np.eye(4)[y_true]
    assert brier_score(perfect, y_true) == 0.0


def test_predictive_entropy_uniform():
    C = 7
    uniform = np.full((16, C), 1.0 / C, dtype=np.float64)
    assert np.isclose(predictive_entropy(uniform), np.log(C), atol=1e-5)

    one_hot = np.eye(C, dtype=np.float64)[np.arange(16) % C]
    assert np.isclose(predictive_entropy(one_hot), 0.0, atol=1e-8)


def test_normalized_predictive_entropy_range():
    C = 7
    uniform = np.full((16, C), 1.0 / C, dtype=np.float64)
    assert np.isclose(normalized_predictive_entropy(uniform), 1.0, atol=1e-8)

    one_hot = np.eye(C, dtype=np.float64)[np.arange(16) % C]
    assert np.isclose(normalized_predictive_entropy(one_hot), 0.0, atol=1e-8)


def test_max_probability():
    probs = np.array(
        [
            [0.8, 0.2],
            [0.1, 0.9],
            [0.5, 0.5],
        ],
        dtype=np.float64,
    )
    assert np.isclose(max_probability(probs), (0.8 + 0.9 + 0.5) / 3.0, atol=1e-8)


def test_raw_aurc_matches_manual():
    y_true = np.array([0, 1, 0, 1], dtype=np.int64)
    y_pred = np.array([0, 1, 1, 0], dtype=np.int64)
    confidence = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float64)
    # Risk curve by confidence ranking: [0, 0, 1/3, 1/2].
    manual_raw = (0.0 + 0.0 + (1.0 / 3.0) + 0.5) / 4.0
    assert np.isclose(raw_aurc(confidence, y_pred, y_true), manual_raw, atol=1e-8)


def test_excess_aurc_perfect_confidence():
    y_true = np.array([0, 1, 0, 1], dtype=np.int64)
    y_pred = np.array([0, 1, 1, 0], dtype=np.int64)
    confidence = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float64)
    assert np.isclose(excess_aurc(confidence, y_pred, y_true), 0.0, atol=1e-8)


def test_aurc_alias_matches_excess_aurc():
    y_true = np.array([0, 1, 0, 1], dtype=np.int64)
    y_pred = np.array([0, 1, 1, 0], dtype=np.int64)
    confidence = np.array([0.9, 0.8, 0.2, 0.1], dtype=np.float64)
    assert np.isclose(aurc(confidence, y_pred, y_true), excess_aurc(confidence, y_pred, y_true))


def test_selective_accuracy_full_coverage():
    y_true = np.array([0, 1, 0, 1], dtype=np.int64)
    y_pred = np.array([0, 0, 0, 1], dtype=np.int64)
    confidence = np.array([0.9, 0.2, 0.8, 0.7], dtype=np.float64)
    overall = float((y_pred == y_true).mean())
    assert np.isclose(selective_accuracy(confidence, y_pred, y_true, coverage=1.0), overall)


def test_selective_accuracy_top_fraction():
    y_true = np.array([1, 0, 1, 0], dtype=np.int64)
    y_pred = np.array([1, 0, 0, 0], dtype=np.int64)
    confidence = np.array([0.95, 0.8, 0.2, 0.1], dtype=np.float64)
    # top-50% keeps first two entries, both correct
    assert np.isclose(selective_accuracy(confidence, y_pred, y_true, coverage=0.5), 1.0)


def test_empirical_coverage_all_in():
    pred_sets = np.ones((10, 4), dtype=bool)
    y_true = np.arange(10) % 4
    assert empirical_coverage(pred_sets, y_true) == 1.0


def test_mean_set_size():
    pred_sets = np.array(
        [
            [True, False, False],
            [True, True, False],
            [False, False, True],
            [True, True, True],
        ],
        dtype=bool,
    )
    assert mean_set_size(pred_sets) == 1.75
