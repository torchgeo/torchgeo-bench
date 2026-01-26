"""Shared utilities for benchmark scripts."""

import numpy as np
from sklearn.datasets import make_classification


def get_dataset(
    n_samples: int = 1000,
    n_classes: int = 5,
    n_features: int = 20,
    n_informative: int | None = None,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic multiclass classification dataset.

    Args:
        n_samples: Number of samples to generate.
        n_classes: Number of distinct classes.
        n_features: Total number of features.
        n_informative: Number of informative features; defaults to min(n_features, n_classes*2).
        random_state: Seed for reproducibility.

    Returns:
        Feature matrix X (n_samples, n_features) and label vector y (n_samples,).
    """
    if n_informative is None:
        n_informative = min(n_features, n_classes * 2)
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_repeated=0,
        n_classes=n_classes,
        n_clusters_per_class=1,
        class_sep=1.0,
        flip_y=0.0,
        random_state=random_state,
    )
    return X.astype(np.float32), y.astype(np.int64)
