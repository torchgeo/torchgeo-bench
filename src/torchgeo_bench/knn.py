"""KNN classifier for torchgeo-bench.

Single-label and multi-label k-nearest neighbours backed by FAISS.

The CPU path uses the selected FAISS backend's ``IndexFlatL2`` implementation.
The GPU path delegates to :mod:`faissknn` when that backend provides CUDA
resources. The two paths produce identical predictions modulo float-precision
noise.
"""

import logging
import os
import sys
from typing import Literal, Self

if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logging.getLogger("faiss.loader").setLevel(logging.WARNING)

import faiss  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

if sys.platform == "darwin":
    faiss.omp_set_num_threads(1)

logger = logging.getLogger(__name__)


def _is_cpu_device(device: str) -> bool:
    return str(device).lower() == "cpu"


def gpu_faiss_available() -> bool:
    """Return whether the installed FAISS package provides CUDA resources."""
    return hasattr(faiss, "StandardGpuResources")


def resolve_knn_device(requested_device: str | None, model_device: str) -> str:
    """Resolve the KNN device, falling back for an implicit CUDA request.

    Args:
        requested_device: Explicit KNN device, or ``None`` to follow the model.
        model_device: Device used by the feature extractor.

    Returns:
        Device to use for KNN evaluation.
    """
    device = requested_device if requested_device is not None else model_device
    if _is_cpu_device(device) or gpu_faiss_available():
        return device
    if requested_device is not None:
        raise RuntimeError(
            f"GPU-enabled FAISS is unavailable for explicit KNN device {requested_device!r}. "
            "Set eval.knn_device=cpu or install a GPU-enabled FAISS backend."
        )
    logger.warning(
        "GPU-enabled FAISS is unavailable; using CPU for KNN while the model remains on %s. "
        "Set eval.knn_device=cpu to make this choice explicit.",
        model_device,
    )
    return "cpu"


class KNNClassifier:
    """FAISS-backed KNN classifier with single- and multi-label support.

    Multi-label mode is auto-detected from the shape of ``y`` during
    :meth:`fit`: 1-D labels → single-label, 2-D labels → multi-label.

    Args:
        n_neighbors: Number of neighbours (k). Clamped to ``min(k, n_train)``
            before either backend is constructed.
        device: ``"cpu"`` (default) → the FAISS CPU index. Anything else
            (``"cuda"``, ``"cuda:0"``) requires ``faissknn`` with a GPU FAISS
            backend (installed automatically on Linux x86_64); raises an
            actionable error if unavailable.
        metric: Distance metric — ``"l2"`` (default), ``"ip"`` (inner
            product), or ``"cosine"`` (cosine similarity; auto-normalizes
            inputs). GPU path only; CPU path always uses L2.
        use_fp16: Use fp16 for GPU index computation (~30 % speedup on
            Ampere+). GPU path only; ignored on CPU.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        device: str = "cpu",
        metric: Literal["l2", "ip", "cosine"] = "l2",
        use_fp16: bool = False,
    ) -> None:
        if isinstance(n_neighbors, bool) or not isinstance(n_neighbors, int) or n_neighbors < 1:
            raise ValueError(f"n_neighbors must be a positive integer, got {n_neighbors!r}.")
        self.n_neighbors = n_neighbors
        self._effective_n_neighbors: int | None = None
        self.device = device
        self.metric = metric
        self.use_fp16 = use_fp16

        # CPU path state
        self._index: faiss.Index | None = None
        self._y: np.ndarray | None = None
        self._n_classes: int | None = None
        self._multi_label: bool = False

        # GPU path state (faissknn delegate)
        self._impl = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Index training features and store labels.

        Args:
            X: ``(n_samples, n_features)`` float32 feature matrix.
            y: ``(n_samples,)`` int single-label or
               ``(n_samples, n_classes)`` multi-hot multi-label.
        """
        X = np.ascontiguousarray(np.atleast_2d(X).astype(np.float32))
        if len(X) == 0:
            raise ValueError("KNNClassifier.fit requires at least one training sample.")
        if len(y) != len(X):
            raise ValueError(f"X has {len(X)} samples but y has {len(y)} labels.")
        self._multi_label = y.ndim == 2
        self._effective_n_neighbors = min(self.n_neighbors, len(X))

        if _is_cpu_device(self.device):
            self._fit_cpu(X, y)
        else:
            self._fit_gpu(X, y)
        return self

    # ---- CPU path (faiss IndexFlatL2) -------------------------------------

    def _fit_cpu(self, X: np.ndarray, y: np.ndarray) -> None:
        self._index = faiss.IndexFlatL2(X.shape[1])
        self._index.add(X)
        if self._multi_label:
            self._n_classes = int(y.shape[1])
            self._y = y.astype(np.float32)
        else:
            self._y = y.astype(np.int64)
            self._n_classes = int(np.max(self._y)) + 1

    def _search_cpu(self, X: np.ndarray) -> np.ndarray:
        assert self._index is not None
        X = np.ascontiguousarray(np.atleast_2d(X).astype(np.float32))
        assert self._effective_n_neighbors is not None
        _, indices = self._index.search(X, self._effective_n_neighbors)
        return indices

    def _neighbour_counts(self, indices: np.ndarray) -> np.ndarray:
        """Vectorized per-row bincount: shape (n_test, n_classes)."""
        assert self._y is not None
        assert self._n_classes is not None
        n_test, k = indices.shape
        labels = self._y[indices].astype(np.int64)  # (n_test, k)
        offsets = (np.arange(n_test) * self._n_classes)[:, None]
        flat = (labels + offsets).ravel()
        return np.bincount(flat, minlength=n_test * self._n_classes).reshape(
            n_test, self._n_classes
        )

    def _predict_cpu(self, X: np.ndarray) -> np.ndarray:
        assert self._y is not None
        indices = self._search_cpu(X)
        if self._multi_label:
            scores = self._y[indices].mean(axis=1)
            return (scores > 0.5).astype(np.int32)
        return np.argmax(self._neighbour_counts(indices), axis=1)

    def _predict_proba_cpu(self, X: np.ndarray) -> np.ndarray:
        assert self._y is not None
        indices = self._search_cpu(X)
        k_eff = indices.shape[1]
        if self._multi_label:
            return self._y[indices].mean(axis=1)
        return self._neighbour_counts(indices).astype(np.float32) / k_eff

    # ---- GPU path (faissknn delegate) -------------------------------------

    def _fit_gpu(self, X: np.ndarray, y: np.ndarray) -> None:
        try:
            from faissknn import FaissKNNClassifier, FaissKNNMultilabelClassifier
        except ImportError as exc:  # pragma: no cover — covered by env, not unit tests
            raise ImportError(
                f"KNNClassifier(device={self.device!r}): faissknn is not installed. "
                "GPU KNN requires Linux x86_64, where it installs automatically; "
                'otherwise request device="cpu".'
            ) from exc

        if not gpu_faiss_available():
            raise RuntimeError(
                f"KNNClassifier(device={self.device!r}): GPU-enabled FAISS is unavailable. "
                "Set eval.knn_device=cpu for CLI runs or request device='cpu'."
            )

        assert self._effective_n_neighbors is not None
        kwargs = {
            "n_neighbors": self._effective_n_neighbors,
            "device": self.device,
            "metric": self.metric,
            "use_fp16": self.use_fp16,
        }
        if self._multi_label:
            self._n_classes = int(y.shape[1])
            self._impl = FaissKNNMultilabelClassifier(**kwargs)
        else:
            # faissknn uses len(unique(y)) as n_classes, which breaks when labels
            # have gaps (e.g. a small partition missing class 4 but containing class 11).
            # Pass n_classes=max(y)+1 to guarantee the counts array is large enough.
            self._n_classes = int(np.max(y)) + 1
            self._impl = FaissKNNClassifier(n_classes=self._n_classes, **kwargs)
        self._impl.fit(X, y.astype(np.int64))

    def _to_gpu_tensor(self, X: np.ndarray) -> torch.Tensor:
        """Convert numpy array to a CUDA tensor for zero-copy faissknn input."""
        return torch.from_numpy(np.ascontiguousarray(X.astype(np.float32))).to(self.device)

    # ---- Public API -------------------------------------------------------

    @property
    def multi_label(self) -> bool:
        """Whether the classifier is in multi-label mode."""
        return self._multi_label

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for ``X``.

        Returns single-label class indices ``(n_samples,)`` or multi-label
        binary predictions ``(n_samples, n_classes)``.
        """
        if _is_cpu_device(self.device):
            return self._predict_cpu(X)
        assert self._impl is not None, "Call fit() first."
        result = self._impl.predict(self._to_gpu_tensor(X))
        return result.cpu().numpy() if isinstance(result, torch.Tensor) else result

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict per-class probabilities ``(n_samples, n_classes)``."""
        if _is_cpu_device(self.device):
            return self._predict_proba_cpu(X)
        assert self._impl is not None, "Call fit() first."
        result = self._impl.predict_proba(self._to_gpu_tensor(X))
        return result.cpu().numpy() if isinstance(result, torch.Tensor) else result
