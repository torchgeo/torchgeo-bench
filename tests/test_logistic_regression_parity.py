import numpy as np
import pytest
import torch
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression as SkLogReg

from src.linear import LogisticRegression


@pytest.fixture(scope="module")
def c_values() -> list[float]:
    return [0.1, 0.5, 1.0, 5.0, 10.0]


def test_logistic_regression_accuracy_parity_iris(c_values: list[float]):
    iris_any = load_iris()
    X_np = np.asarray(iris_any.data, dtype=np.float32)
    y_np = np.asarray(iris_any.target, dtype=np.int64)
    # enlarge dataset to reduce variance
    X_np = np.tile(X_np, (100, 1))
    y_np = np.tile(y_np, 100)

    X = torch.from_numpy(X_np)
    y = torch.from_numpy(y_np)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    acc_diffs = []

    for C in c_values:
        torch_clf = LogisticRegression(
            C=C, max_iter=500, solver="lbfgs", device=device, verbose=False
        )
        torch_clf.fit(X, y)
        torch_preds = torch_clf.predict(X)
        torch_acc = (torch_preds == y_np).mean()

        sk_clf = SkLogReg(C=C, max_iter=500, solver="lbfgs")
        sk_clf.fit(X_np, y_np)
        sk_preds = sk_clf.predict(X_np)
        sk_acc = (sk_preds == y_np).mean()

        acc_diff = float(torch_acc - sk_acc)
        acc_diffs.append(abs(acc_diff))

    assert max(acc_diffs) <= 1e-3, f"Accuracy parity failed; diffs: {acc_diffs}"
