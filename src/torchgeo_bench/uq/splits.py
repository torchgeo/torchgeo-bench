"""Data splitting helpers for uncertainty calibration."""

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def stratified_cal_split(
    X: np.ndarray,
    y: np.ndarray,
    cal_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split arrays into a fixed-size stratified calibration subset and remainder.

    Args:
        X: Feature array with shape ``(N, D)``.
        y: Label array with shape ``(N,)``.
        cal_size: Number of samples for the calibration split.
        seed: Random seed for deterministic partitioning.

    Returns:
        Tuple ``(X_cal, y_cal, X_rem, y_rem)``.
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got shape {y.shape}")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows")
    if cal_size <= 0 or cal_size >= X.shape[0]:
        raise ValueError(f"cal_size must be in [1, {X.shape[0] - 1}], got {cal_size}")

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=cal_size, random_state=seed)
    rem_idx, cal_idx = next(splitter.split(X, y))
    return X[cal_idx], y[cal_idx], X[rem_idx], y[rem_idx]
