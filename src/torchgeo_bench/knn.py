"""KNN classifier for torchgeo-bench.

Single-label and multi-label k-nearest neighbours backed by FAISS.

CPU path (always available) uses ``faiss-cpu`` with ``IndexFlatL2`` and
matches the historic implementation. GPU path (opt-in via the ``cuda``
extra: ``pip install -e ".[cuda]"``) delegates to :mod:`faissknn`, which
links against the GPU FAISS wheels (``faiss-cuda-cu128``). The two paths
produce identical predictions modulo float-precision noise.
"""

import logging
from typing import Literal, Self

# Suppress noisy INFO messages from faiss loader (AVX512/AVX2 fallback probing)
logging.getLogger("faiss.loader").setLevel(logging.WARNING)

import faiss  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

logger = logging.getLogger(__name__)


def _is_cpu_device(device: str) -> bool:
    return str(device).lower() == "cpu"


class KNNClassifier:
    """FAISS-backed KNN classifier with single- and multi-label support.

    Multi-label mode is auto-detected from the shape of ``y`` during
    :meth:`fit`: 1-D labels → single-label, 2-D labels → multi-label.

    Args:
        n_neighbors: Number of neighbours (k). Clamped to ``min(k, n_train)``
            on the CPU path; faissknn does not clamp internally.
        device: ``"cpu"`` (default) → ``faiss-cpu``. Anything else
            (``"cuda"``, ``"cuda:0"``) requires the ``cuda`` extra
            (``faissknn``); raises :class:`ImportError` if not installed.
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
        self.n_neighbors = int(n_neighbors)
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
        self._multi_label = y.ndim == 2

        if _is_cpu_device(self.device):
            self._fit_cpu(X, y)
        else:
            self._fit_gpu(X, y)
        return self

    # ---- CPU path (faiss-cpu) ---------------------------------------------

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
        k_eff = min(self.n_neighbors, self._index.ntotal)
        _, indices = self._index.search(X, k_eff)
        return indices

    def _neighbour_counts(self, indices: np.ndarray) -> np.ndarray:
        """Vectorized per-row bincount: shape (n_test, n_classes)."""
        n_test, k = indices.shape
        labels = self._y[indices].astype(np.int64)  # (n_test, k)
        offsets = (np.arange(n_test) * self._n_classes)[:, None]
        flat = (labels + offsets).ravel()
        return np.bincount(flat, minlength=n_test * self._n_classes).reshape(n_test, self._n_classes)

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
        except ImportError:  # pragma: no cover — covered by env, not unit tests
            logger.warning(
                "KNNClassifier(device=%r): the 'cuda' extra is not installed "
                "(faissknn missing); falling back to the CPU faiss-cpu path. "
                'Install with `pip install -e ".[cuda]"` to enable GPU KNN.',
                self.device,
            )
            self.device = "cpu"
            self._fit_cpu(X, y)
            return

        kwargs = dict(
            n_neighbors=self.n_neighbors,
            device=self.device,
            metric=self.metric,
            use_fp16=self.use_fp16,
        )
        if self._multi_label:
            self._n_classes = int(y.shape[1])
            self._impl = FaissKNNMultilabelClassifier(**kwargs)
        else:
            self._impl = FaissKNNClassifier(**kwargs)
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
