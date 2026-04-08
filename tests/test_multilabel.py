"""Tests for multi-label support and KNNClassifier."""

import numpy as np
import pytest
import torch

from src.knn import KNNClassifier
from src.linear import LogisticRegression


@pytest.fixture
def multilabel_data():
    """Synthetic multi-label dataset: 200 train, 50 val, 50 test, 10 classes."""
    rng = np.random.default_rng(42)
    n_train, n_val, n_test = 200, 50, 50
    n_features, n_classes = 32, 10
    n_total = n_train + n_val + n_test

    X = rng.standard_normal((n_total, n_features)).astype(np.float32)
    # Generate multi-hot labels with ~3 active classes per sample
    Y = (rng.random((n_total, n_classes)) > 0.7).astype(np.float32)
    # Ensure at least one positive label per sample
    for i in range(n_total):
        if Y[i].sum() == 0:
            Y[i, rng.integers(0, n_classes)] = 1.0

    return {
        "x_train": X[:n_train],
        "y_train": Y[:n_train],
        "x_val": X[n_train : n_train + n_val],
        "y_val": Y[n_train : n_train + n_val],
        "x_test": X[n_train + n_val :],
        "y_test": Y[n_train + n_val :],
        "n_classes": n_classes,
    }


@pytest.fixture
def singlelabel_data():
    """Synthetic single-label dataset for KNN tests."""
    rng = np.random.default_rng(99)
    n_train, n_test = 100, 30
    n_features, n_classes = 16, 4

    X_train = rng.standard_normal((n_train, n_features)).astype(np.float32)
    y_train = rng.integers(0, n_classes, size=n_train).astype(np.int64)
    X_test = rng.standard_normal((n_test, n_features)).astype(np.float32)
    y_test = rng.integers(0, n_classes, size=n_test).astype(np.int64)

    return {
        "x_train": X_train,
        "y_train": y_train,
        "x_test": X_test,
        "y_test": y_test,
        "n_classes": n_classes,
    }


# ---- KNNClassifier tests ----


class TestKNNClassifierSingleLabel:
    def test_fit_predict_shapes(self, singlelabel_data):
        d = singlelabel_data
        clf = KNNClassifier(n_neighbors=5, device="cpu")
        clf.fit(d["x_train"], d["y_train"])

        preds = clf.predict(d["x_test"])
        assert preds.shape == (len(d["x_test"]),)
        assert all(0 <= p < d["n_classes"] for p in preds)

    def test_predict_proba_shapes(self, singlelabel_data):
        d = singlelabel_data
        clf = KNNClassifier(n_neighbors=5, device="cpu")
        clf.fit(d["x_train"], d["y_train"])

        probs = clf.predict_proba(d["x_test"])
        assert probs.shape == (len(d["x_test"]), d["n_classes"])
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_multi_label_property_false(self, singlelabel_data):
        d = singlelabel_data
        clf = KNNClassifier(n_neighbors=5, device="cpu")
        clf.fit(d["x_train"], d["y_train"])
        assert clf.multi_label is False

    def test_k_clamped_to_train_size(self):
        """k > n_train should not crash."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((3, 8)).astype(np.float32)
        y = np.array([0, 1, 2], dtype=np.int64)
        clf = KNNClassifier(n_neighbors=10, device="cpu")
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (3,)


class TestKNNClassifierMultiLabel:
    def test_fit_predict_shapes(self, multilabel_data):
        d = multilabel_data
        clf = KNNClassifier(n_neighbors=5, device="cpu")
        clf.fit(d["x_train"], d["y_train"])

        preds = clf.predict(d["x_test"])
        assert preds.shape == (len(d["x_test"]), d["n_classes"])
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_proba_shapes(self, multilabel_data):
        d = multilabel_data
        clf = KNNClassifier(n_neighbors=5, device="cpu")
        clf.fit(d["x_train"], d["y_train"])

        probs = clf.predict_proba(d["x_test"])
        assert probs.shape == (len(d["x_test"]), d["n_classes"])
        assert np.all(probs >= 0) and np.all(probs <= 1)

    def test_multi_label_property_true(self, multilabel_data):
        d = multilabel_data
        clf = KNNClassifier(n_neighbors=5, device="cpu")
        clf.fit(d["x_train"], d["y_train"])
        assert clf.multi_label is True


# ---- LogisticRegression multi-label tests ----


class TestMultiLabelLogisticRegression:
    def test_fit_and_predict_shapes(self, multilabel_data):
        d = multilabel_data
        X_t = torch.from_numpy(d["x_train"])
        Y_t = torch.from_numpy(d["y_train"])
        X_test = torch.from_numpy(d["x_test"])

        clf = LogisticRegression(C=1.0, max_iter=100, multi_label=True, device="cpu")
        clf.fit(X_t, Y_t)

        preds = clf.predict(X_test)
        assert preds.shape == (len(d["x_test"]), d["n_classes"])
        assert set(np.unique(preds)).issubset({0, 1})

        probs = clf.predict_proba(X_test)
        assert probs.shape == (len(d["x_test"]), d["n_classes"])
        assert np.all(probs >= 0) and np.all(probs <= 1)

    def test_rejects_1d_labels(self, multilabel_data):
        d = multilabel_data
        X_t = torch.from_numpy(d["x_train"])
        y_1d = torch.from_numpy(d["y_train"][:, 0])

        clf = LogisticRegression(C=1.0, multi_label=True, device="cpu")
        with pytest.raises(ValueError, match="2D"):
            clf.fit(X_t, y_1d)

    def test_lbfgs_solver(self, multilabel_data):
        d = multilabel_data
        X_t = torch.from_numpy(d["x_train"])
        Y_t = torch.from_numpy(d["y_train"])

        clf = LogisticRegression(
            C=1.0, max_iter=200, solver="lbfgs", multi_label=True, device="cpu"
        )
        clf.fit(X_t, Y_t)
        assert clf._fitted

    def test_adam_solver(self, multilabel_data):
        d = multilabel_data
        X_t = torch.from_numpy(d["x_train"])
        Y_t = torch.from_numpy(d["y_train"])

        clf = LogisticRegression(C=1.0, max_iter=50, solver="adam", multi_label=True, device="cpu")
        clf.fit(X_t, Y_t)
        assert clf._fitted

    def test_classes_attribute(self, multilabel_data):
        d = multilabel_data
        X_t = torch.from_numpy(d["x_train"])
        Y_t = torch.from_numpy(d["y_train"])

        clf = LogisticRegression(C=1.0, max_iter=50, multi_label=True, device="cpu")
        clf.fit(X_t, Y_t)
        assert clf.classes_ is not None
        np.testing.assert_array_equal(clf.classes_, np.arange(d["n_classes"]))


# ---- Bootstrap mAP tests ----


class TestBootstrapMAP:
    def test_bootstrap_map_basic(self):
        from torchgeo_bench import bootstrap_map

        rng = np.random.default_rng(0)
        n, c = 100, 5
        y_true = (rng.random((n, c)) > 0.7).astype(np.float32)
        for i in range(n):
            if y_true[i].sum() == 0:
                y_true[i, 0] = 1.0
        y_scores = rng.random((n, c)).astype(np.float32)

        mean, lo, hi = bootstrap_map(y_true, y_scores, n_boot=100, seed=42)
        assert 0 <= lo <= mean <= hi <= 1.0

    def test_perfect_scores(self):
        from torchgeo_bench import bootstrap_map

        y_true = np.eye(5, dtype=np.float32)
        y_scores = np.eye(5, dtype=np.float32)

        mean, lo, hi = bootstrap_map(y_true, y_scores, n_boot=50, seed=0)
        assert mean == pytest.approx(1.0)


# ---- Unified evaluate_knn / evaluate_logistic tests ----


class TestUnifiedEvaluateKNN:
    def test_single_label(self, singlelabel_data):
        from torchgeo_bench import evaluate_knn

        d = singlelabel_data
        score, lo, hi = evaluate_knn(
            d["x_train"],
            d["y_train"],
            d["x_test"],
            d["y_test"],
            seed=42,
            n_bootstrap=50,
            device="cpu",
        )
        assert 0 <= lo <= score <= hi <= 1.0

    def test_multi_label(self, multilabel_data):
        from torchgeo_bench import evaluate_knn

        d = multilabel_data
        score, lo, hi = evaluate_knn(
            d["x_train"],
            d["y_train"],
            d["x_test"],
            d["y_test"],
            seed=42,
            n_bootstrap=50,
            device="cpu",
        )
        assert 0 <= lo <= score <= hi <= 1.0


class TestUnifiedEvaluateLogistic:
    def test_single_label(self, singlelabel_data):
        from torchgeo_bench import evaluate_logistic

        d = singlelabel_data
        score, lo, hi, best_c = evaluate_logistic(
            d["x_train"],
            d["y_train"],
            d["x_test"][:15],
            d["y_test"][:15],  # use as val
            d["x_test"][15:],
            d["y_test"][15:],
            c_values=[0.1, 1.0],
            seed=42,
            n_bootstrap=50,
            merge_val=True,
            device="cpu",
        )
        assert 0 <= lo <= score <= hi <= 1.0
        assert best_c in [0.1, 1.0]

    def test_multi_label(self, multilabel_data):
        from torchgeo_bench import evaluate_logistic

        d = multilabel_data
        score, lo, hi, best_c = evaluate_logistic(
            d["x_train"],
            d["y_train"],
            d["x_val"],
            d["y_val"],
            d["x_test"],
            d["y_test"],
            c_values=[0.01, 0.1, 1.0],
            seed=42,
            n_bootstrap=50,
            merge_val=True,
            device="cpu",
        )
        assert 0 <= lo <= score <= hi <= 1.0
        assert best_c in [0.01, 0.1, 1.0]
