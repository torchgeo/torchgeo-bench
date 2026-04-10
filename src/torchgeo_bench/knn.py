"""KNN classifier for torchgeo-bench.

Unified k-nearest neighbors classifier supporting both single-label and
multi-label classification using FAISS for efficient nearest-neighbor search.
"""

import logging
from typing import Any, Self

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
        device: ``"cpu"`` or ``"cuda:<id>"`` for GPU-accelerated search.
    """

    def __init__(self, n_neighbors: int = 5, device: str = "cpu") -> None:
        self.n_neighbors = n_neighbors
        self.device = device

        self._index: faiss.Index | None = None
        self._y: np.ndarray | None = None
        self._multi_label: bool = False
        self._n_classes: int | None = None
        self._gpu_res: Any | None = None

    def _build_index(self, d: int) -> faiss.Index:
        if self.device != "cpu" and self.device.startswith("cuda"):
            if not all(
                hasattr(faiss, attr) for attr in ("StandardGpuResources", "GpuIndexFlatConfig")
            ):
                logger.warning(
                    "CUDA device requested but faiss GPU support is unavailable; using CPU."
                )
                return faiss.IndexFlatL2(d)
            gpu_id = int(self.device.split(":")[-1]) if ":" in self.device else 0
            self._gpu_res = faiss.StandardGpuResources()
            config = faiss.GpuIndexFlatConfig()
            config.device = gpu_id
            return faiss.GpuIndexFlatL2(self._gpu_res, d, config)
        return faiss.IndexFlatL2(d)

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Index training features and store labels.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``, float32.
            y: Labels — ``(n_samples,)`` for single-label or
               ``(n_samples, n_classes)`` for multi-label.
        """
        X = np.atleast_2d(X).astype(np.float32)
        X = np.ascontiguousarray(X)

        self._index = self._build_index(X.shape[1])
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
        X = np.atleast_2d(X).astype(np.float32)
        X = np.ascontiguousarray(X)
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
        else:
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
        else:
            neighbor_labels = self._y[indices]
            counts = np.apply_along_axis(
                lambda x: np.bincount(x, minlength=self._n_classes),
                axis=1,
                arr=neighbor_labels.astype(np.int16),
            )
            return counts.astype(np.float32) / k_eff

    def __del__(self) -> None:
        if hasattr(self, "_index") and self._index is not None:
            self._index.reset()
            del self._index
        if hasattr(self, "_gpu_res") and self._gpu_res is not None:
            self._gpu_res.noTempMemory()
            del self._gpu_res
