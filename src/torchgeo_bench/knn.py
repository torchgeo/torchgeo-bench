"""KNN classifier for torchgeo-bench.

Unified k-nearest neighbors classifier supporting both single-label and
multi-label classification using FAISS (CPU) for efficient nearest-neighbor
search.
"""

import logging
from typing import Self

# Suppress noisy INFO messages from faiss loader (AVX512/AVX2 fallback probing)
logging.getLogger("faiss.loader").setLevel(logging.WARNING)

import faiss  # noqa: E402
import numpy as np  # noqa: E402

logger = logging.getLogger(__name__)


class KNNClassifier:
    """FAISS-backed KNN classifier supporting single-label and multi-label tasks.

    Mirrors the LogisticRegression API: ``fit(X, y)`` / ``predict(X)`` /
    ``predict_proba(X)``.

    Multi-label mode is auto-detected from the shape of ``y`` during ``fit``:
    - 1-D ``y`` of shape ``(n_samples,)`` → single-label classification.
    - 2-D ``y`` of shape ``(n_samples, n_classes)`` → multi-label classification.

    Args:
        n_neighbors: Number of nearest neighbors (k). Clamped to ``min(k, n_train)``
            at predict time.
    """

    def __init__(self, n_neighbors: int = 5) -> None:
        self.n_neighbors = n_neighbors

        self._index: faiss.Index | None = None
        self._y: np.ndarray | None = None
        self._multi_label: bool = False
        self._n_classes: int | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Index training features and store labels.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``, float32.
            y: Labels — ``(n_samples,)`` for single-label or
               ``(n_samples, n_classes)`` for multi-label.
        """
        X = np.ascontiguousarray(np.atleast_2d(X).astype(np.float32))

        self._index = faiss.IndexFlatL2(X.shape[1])
        self._index.add(X)

        if y.ndim == 2:
            self._multi_label = True
            self._n_classes = y.shape[1]
            self._y = y.astype(np.float32)
        else:
            self._multi_label = False
            self._y = y.astype(np.int64)
            self._n_classes = int(np.max(self._y)) + 1

        return self

    @property
    def multi_label(self) -> bool:
        """Whether the classifier is in multi-label mode."""
        return self._multi_label

    def _search(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (distances, indices) for k-nearest neighbors."""
        assert self._index is not None and self._y is not None, "Call fit() first."
        X = np.ascontiguousarray(np.atleast_2d(X).astype(np.float32))
        k_eff = min(self.n_neighbors, self._index.ntotal)
        return self._index.search(X, k_eff)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for X.

        Returns:
            Single-label: integer class indices ``(n_samples,)``.
            Multi-label: binary predictions ``(n_samples, n_classes)`` at threshold 0.5.
        """
        assert self._y is not None
        _, indices = self._search(X)

        if self._multi_label:
            neighbor_labels = self._y[indices]  # (n_test, k, n_classes)
            scores = neighbor_labels.mean(axis=1)
            return (scores > 0.5).astype(np.int32)
        neighbor_labels = self._y[indices]  # (n_test, k)
        counts = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=self._n_classes),
            axis=1,
            arr=neighbor_labels.astype(np.int16),
        )
        return np.argmax(counts, axis=1)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict per-class probabilities for X.

        Returns:
            Single-label: ``(n_samples, n_classes)`` with vote fractions.
            Multi-label: ``(n_samples, n_classes)`` with mean neighbor labels.
        """
        assert self._y is not None
        _, indices = self._search(X)
        k_eff = indices.shape[1]

        if self._multi_label:
            neighbor_labels = self._y[indices]
            return neighbor_labels.mean(axis=1)
        neighbor_labels = self._y[indices]
        counts = np.apply_along_axis(
            lambda x: np.bincount(x, minlength=self._n_classes),
            axis=1,
            arr=neighbor_labels.astype(np.int16),
        )
        return counts.astype(np.float32) / k_eff
