"""Intrinsic dimension (ID) estimation over feature embeddings.

Thin wrapper around ``torchid`` (https://github.com/isaaccorley/torchid).
Provides a single entry point to compute one or more global ID estimates on a
feature matrix and return scalar values per estimator.

ID is computed on raw embeddings (no L2-normalization) to match the distance
geometry used by KNN/linear probes elsewhere in this package.
"""

import logging
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)

SUPPORTED_ESTIMATORS: tuple[str, ...] = (
    "lPCA",
    "TwoNN",
    "MLE",
    "CorrInt",
    "MiND_ML",
    "KNN",
    "DANCo",
    "FisherS",
)


def _load_estimator(name: str) -> type:
    """Lazy-import a torchid global estimator class by name."""
    try:
        from torchid import estimators as _est
    except ImportError as e:
        raise ImportError(
            "torchid is required for intrinsic-dimension metrics. "
            "Install with `pip install 'torchgeo-bench[id]'` "
            "(requires Python >=3.13)."
        ) from e
    if not hasattr(_est, name):
        raise ValueError(
            f"Unknown torchid estimator '{name}'. Supported: {', '.join(SUPPORTED_ESTIMATORS)}."
        )
    return getattr(_est, name)


def _resolve_device(device: str | torch.device | None) -> torch.device:
    """Resolve the requested device, falling back to CPU when CUDA unavailable."""
    if device is None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested for intrinsic-dim but unavailable; using CPU.")
        dev = torch.device("cpu")
    return dev


def _subsample(X: np.ndarray, max_samples: int | None, seed: int) -> np.ndarray:
    """Deterministically subsample rows of X if it exceeds max_samples."""
    if max_samples is None or X.shape[0] <= max_samples:
        return X
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=max_samples, replace=False)
    return X[idx]


def compute_intrinsic_dim(
    X: np.ndarray,
    estimators: list[str],
    device: str | torch.device | None = None,
    max_samples: int | None = 10_000,
    seed: int = 0,
) -> dict[str, float]:
    """Compute intrinsic dimension of X for each requested estimator.

    Args:
        X: Feature matrix of shape ``(n_samples, n_features)``.
        estimators: Names of torchid global estimators (see
            ``SUPPORTED_ESTIMATORS``).
        device: ``"cuda"``, ``"cpu"``, a ``torch.device``, or ``None`` to
            auto-select (CUDA when available, otherwise CPU).
        max_samples: Cap row count via random subsampling for speed/memory.
            ``None`` disables subsampling.
        seed: RNG seed for subsampling determinism.

    Returns:
        Mapping ``{estimator_name: dimension}``. Failed estimators yield
        ``float('nan')`` and a logged warning rather than aborting the run.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    if not estimators:
        return {}

    dev = _resolve_device(device)
    Xs = _subsample(X, max_samples, seed)
    X_tensor = torch.from_numpy(np.ascontiguousarray(Xs)).to(dev, dtype=torch.float32)

    out: dict[str, float] = {}
    for name in estimators:
        try:
            cls = _load_estimator(name)
            est: Any = cls().fit(X_tensor)
            value = float(est.dimension_)
        except Exception as e:  # noqa: BLE001 — log and continue per-estimator
            logger.warning(f"[intrinsic-dim] {name} failed on X{tuple(X_tensor.shape)}: {e}")
            value = float("nan")
        out[name] = value
    return out
