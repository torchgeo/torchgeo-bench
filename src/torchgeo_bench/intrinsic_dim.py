"""Intrinsic dimension (ID) estimation over feature embeddings.

Thin wrapper around ``torchid`` (https://github.com/isaaccorley/torchid).
Provides a single entry point to compute one or more global ID estimates on a
feature matrix and return scalar values per estimator.

The module also provides dependency-free effective-rank, participation-ratio,
variance-explained, and anisotropy diagnostics from centered embeddings.

ID is computed on raw embeddings (no L2-normalization) to match the distance
geometry used by KNN/linear probes elsewhere in this package.
"""

import logging
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


class DegenerateManifoldError(ValueError):
    """Feature manifold is degenerate; the estimator returned a non-finite dimension."""


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

FEATURE_SPECTRUM_METRICS: tuple[str, ...] = (
    "effective_rank",
    "participation_ratio",
    "pc1_variance_ratio",
    "pc10_variance_ratio",
    "spectral_anisotropy",
)


class DegenerateSpectrumError(ValueError):
    """Feature spectrum is undefined because the centered matrix has no variance."""


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


def compute_feature_spectrum(
    X: np.ndarray,
    max_samples: int | None = 10_000,
    seed: int = 0,
) -> dict[str, float]:
    """Compute scale-invariant spectral diagnostics on centered embeddings.

    The squared singular values of the centered feature matrix are normalized
    into variance proportions ``p``. Effective rank is ``exp(H(p))``;
    participation ratio is ``1 / sum(p**2)``. Spectral anisotropy normalizes
    leading-component dominance against the isotropic ``1 / d`` baseline:
    ``(d * p[0] - 1) / (d - 1)``.

    Args:
        X: Feature matrix of shape ``(n_samples, n_features)``.
        max_samples: Cap row count via deterministic random subsampling.
            ``None`` disables subsampling.
        seed: RNG seed for subsampling determinism.

    Returns:
        Mapping containing effective rank, participation ratio, variance
        explained by PC1 and the first 10 PCs, and spectral anisotropy.

    Raises:
        ValueError: If the input is not a finite 2-D matrix with at least two
            samples and one feature.
        DegenerateSpectrumError: If centering leaves no feature variance.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    if X.shape[0] < 2:
        raise ValueError(f"X must contain at least two samples, got shape {X.shape}")
    if X.shape[1] < 1:
        raise ValueError(f"X must contain at least one feature, got shape {X.shape}")
    if not np.isfinite(X).all():
        raise ValueError("X must contain only finite values")
    if max_samples is not None and (
        isinstance(max_samples, bool) or not isinstance(max_samples, int) or max_samples < 2
    ):
        raise ValueError(
            f"max_samples must be an integer of at least 2 or None, got {max_samples!r}"
        )

    Xs = np.asarray(_subsample(X, max_samples, seed), dtype=np.float64)
    centered = Xs - Xs.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False, full_matrices=False)
    variances = singular_values**2
    total_variance = float(variances.sum())
    if not np.isfinite(total_variance) or total_variance <= 0:
        raise DegenerateSpectrumError(
            "Feature spectrum is undefined: centering left zero total variance "
            f"for X{tuple(Xs.shape)}."
        )

    proportions = variances / total_variance
    positive = proportions[proportions > 0]
    effective_rank = float(np.exp(-np.sum(positive * np.log(positive))))
    participation_ratio = float(1.0 / np.sum(proportions**2))
    pc1_variance_ratio = float(proportions[0])
    pc10_variance_ratio = float(proportions[:10].sum())

    feature_dim = Xs.shape[1]
    if feature_dim == 1:
        spectral_anisotropy = 0.0
    else:
        spectral_anisotropy = float(
            np.clip(
                (feature_dim * pc1_variance_ratio - 1.0) / (feature_dim - 1.0),
                0.0,
                1.0,
            )
        )

    return {
        "effective_rank": effective_rank,
        "participation_ratio": participation_ratio,
        "pc1_variance_ratio": pc1_variance_ratio,
        "pc10_variance_ratio": pc10_variance_ratio,
        "spectral_anisotropy": spectral_anisotropy,
    }


def _two_nearest_distances(X: torch.Tensor) -> torch.Tensor:
    """Pairwise (d1, d2) for each row, matching torchid's knn precision.

    Replicates torchid's exact squared-distance formula
    (``x_sq + y_sq − 2·x·y.T`` then ``clamp_(min=0)``) rather than using
    ``torch.cdist``.  ``cdist`` is more stable on CUDA, so its distances
    disagree with torchid's at the underflow boundary: torchid's formula
    can cancel to a tiny negative, clamp to 0, and underflow to 0 in fp32
    after ``.sqrt()`` for rows this function would otherwise call
    non-degenerate.  Matching it keeps dedup and the estimator agreeing on
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
        exceptions propagate rather than becoming NaN.
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
        # Only a non-finite dimension_ after a clean fit is a soft failure.
        cls = _load_estimator(name)
        est: Any = cls().fit(X_tensor)
        value = float(est.dimension_)
        if not np.isfinite(value):
            d = _two_nearest_distances(X_tensor)
            d1, d2 = d[:, 0], d[:, 1]
            raise DegenerateManifoldError(
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
