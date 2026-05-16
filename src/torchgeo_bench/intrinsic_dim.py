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


def _two_nearest_distances(X: torch.Tensor) -> torch.Tensor:
    """Pairwise (d1, d2) for each row, matching torchid's knn precision.

    We deliberately replicate torchid's exact squared-distance formula
    (``x_sq + y_sq − 2·x·y.T`` then ``clamp_(min=0)``) instead of using
    ``torch.cdist``.  ``cdist`` is more numerically stable on CUDA, so
    its distances disagree with torchid's at the underflow boundary —
    that mismatch was hiding a TwoNN nan we just debugged (sweep 88205,
    Prithvi v1_100): the dedup said ``d1.min = 9.96e-3, zeros = 0`` but
    torchid's internal knn produced ``d1 == 0`` for the same rows
    because its squared-distance formula cancels to a tiny negative,
    gets clamped to 0, and underflows to 0 in fp32 after ``.sqrt()``.

    Replicating the formula keeps dedup and the estimator agreeing on
    which rows are degenerate.
    """
    x_sq = (X * X).sum(dim=1, keepdim=True)
    y_sq = x_sq.squeeze(1)
    d_sq = (x_sq + y_sq.unsqueeze(0) - 2.0 * (X @ X.T)).clamp_(min=0.0)
    d_sq.fill_diagonal_(float("inf"))
    top2_sq = d_sq.topk(k=2, largest=False).values
    return top2_sq.sqrt()


def _drop_zero_distance_rows(X_tensor: torch.Tensor) -> torch.Tensor:
    """Drop rows whose computed nearest-neighbour distance underflows to zero.

    TwoNN's slope is ``sum(x * y) / sum(x * x)`` over ``x = log(mu)`` where
    ``mu = d2 / d1``.  When two rows are close enough that their fp32 squared
    distance underflows, ``d1 == 0``; the estimator's inner ``clamp_min``
    leaves ``mu = 0``, and ``log(0) = -inf`` poisons the slope to ``nan`` —
    observed in the wild on Prithvi / Clay CLS-token embeddings.

    Bit-exact dedup doesn't catch this case because the rows differ in
    their last few bits; only the *distance* underflows.  Drop the rows
    where ``d1 == 0`` or ``d2 == 0`` so the remaining set has well-defined
    distance ratios.
    """
    d = _two_nearest_distances(X_tensor)
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
        Mapping ``{estimator_name: dimension}``.  Estimator-internal
        exceptions propagate; we no longer swallow them as NaN, because
        doing so previously hid the TwoNN/fp32-zero-distance bug.
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
        # Estimator-internal exceptions propagate so we actually fix the
        # bug instead of silently emitting NaN to the CSV.  The only
        # tolerated "soft" failure path is a numerical NaN/inf in
        # dimension_ after a clean fit — even there we raise with a full
        # diagnostic dump so the next person debugging has a real lead.
        cls = _load_estimator(name)
        est: Any = cls().fit(X_tensor)
        value = float(est.dimension_)
        if not np.isfinite(value):
            d = _two_nearest_distances(X_tensor)
            d1, d2 = d[:, 0], d[:, 1]
            raise ValueError(
                f"[intrinsic-dim] {name} returned non-finite dimension ({value}) on "
                f"X{tuple(X_tensor.shape)} after dedup. "
                f"d1[min={d1.min():.3e} median={d1.median():.3e} zeros={(d1 == 0).sum().item()}] "
                f"d2[min={d2.min():.3e} zeros={(d2 == 0).sum().item()}] "
                f"X[norm_min={X_tensor.norm(dim=1).min():.3e} "
                f"norm_max={X_tensor.norm(dim=1).max():.3e} std={X_tensor.std():.3e}]. "
                f"Investigate before writing this to the CSV."
            )
        out[name] = value
    return out
