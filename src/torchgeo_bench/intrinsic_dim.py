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


def _drop_zero_distance_rows(X_tensor: torch.Tensor) -> torch.Tensor:
    """Drop rows whose computed nearest-neighbour distance underflows to zero.

    TwoNN's slope is ``sum(x * y) / sum(x * x)`` over ``x = log(mu)`` where
    ``mu = d2 / d1``.  When two rows are close enough that their fp32 squared
    distance underflows, ``d1 == 0``; the estimator's inner ``clamp_min``
    leaves ``mu = 0``, and ``log(0) = -inf`` poisons the slope to ``nan`` —
    observed in the wild on Prithvi / Clay CLS-token embeddings (uniform
    land-cover patches in the same crop / time produce indistinguishable
    pooled features in fp32).

    Bit-exact ``np.unique`` doesn't catch this case because the rows differ
    in their last few bits; only the *distance* underflows.  Drop the rows
    where ``d1 == 0`` or ``d2 == 0`` so the remaining set has well-defined
    distance ratios.
    """
    from torchid.primitives import knn

    d, _ = knn(X_tensor, k=2)
    keep = (d[:, 0] > 0) & (d[:, 1] > 0)
    n_drop = int((~keep).sum().item())
    if n_drop > 0:
        logger.info(
            f"[intrinsic-dim] dropped {n_drop} rows with zero-distance neighbours "
            f"({X_tensor.shape[0]} -> {int(keep.sum().item())}) before estimation."
        )
        return X_tensor[keep]
    return X_tensor


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
    X_tensor = _drop_zero_distance_rows(X_tensor)

    out: dict[str, float] = {}
    for name in estimators:
        try:
            cls = _load_estimator(name)
            est: Any = cls().fit(X_tensor)
            value = float(est.dimension_)
        except Exception as e:  # noqa: BLE001 — log and continue per-estimator
            logger.warning(f"[intrinsic-dim] {name} failed on X{tuple(X_tensor.shape)}: {e}")
            value = float("nan")
        if not np.isfinite(value):
            with torch.no_grad():
                try:
                    from torchid.primitives import knn

                    d, _ = knn(X_tensor, k=2)
                    d1 = d[:, 0]
                    d2 = d[:, 1]
                    mu = d2 / d1.clamp_min(torch.finfo(X_tensor.dtype).tiny)
                    stats = (
                        f"d1[min={d1.min():.3e} median={d1.median():.3e} max={d1.max():.3e} "
                        f"zeros={(d1 == 0).sum().item()}] "
                        f"d2[min={d2.min():.3e} median={d2.median():.3e}] "
                        f"mu[min={mu.min():.3e} max={mu.max():.3e} "
                        f"zeros={(mu == 0).sum().item()} "
                        f"finite={torch.isfinite(mu).sum().item()}/{mu.numel()}] "
                        f"X[norm_min={X_tensor.norm(dim=1).min():.3e} "
                        f"norm_max={X_tensor.norm(dim=1).max():.3e} "
                        f"std={X_tensor.std():.3e}]"
                    )
                except Exception as inner:  # noqa: BLE001
                    stats = f"(diag failed: {inner})"
            logger.warning(
                f"[intrinsic-dim] {name} returned non-finite value ({value}) on "
                f"X{tuple(X_tensor.shape)} — {stats}"
            )
        out[name] = value
    return out
