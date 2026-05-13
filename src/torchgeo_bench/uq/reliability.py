"""Reliability binning utilities for per-sample UQ traces."""

import numpy as np
import pandas as pd

RELIABILITY_COLUMNS: tuple[str, ...] = (
    "bin_id",
    "n_bin",
    "sum_conf",
    "sum_correct",
    "mean_conf",
    "accuracy",
)


def _validate_inputs(confidence: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    conf = np.asarray(confidence, dtype=np.float64)
    corr = np.asarray(correct, dtype=np.float64)

    if conf.ndim != 1 or corr.ndim != 1:
        raise ValueError("confidence and correct must both be 1D arrays")
    if conf.shape[0] != corr.shape[0]:
        raise ValueError("confidence and correct must have the same length")
    if conf.shape[0] == 0:
        return conf, corr
    if not np.isfinite(conf).all() or not np.isfinite(corr).all():
        raise ValueError("confidence and correct must be finite")
    return conf, corr


def _aggregate_bins(bin_ids: np.ndarray, conf: np.ndarray, corr: np.ndarray, bins: int) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for idx in range(bins):
        mask = bin_ids == idx
        n_bin = int(mask.sum())
        if n_bin == 0:
            rows.append(
                {
                    "bin_id": int(idx),
                    "n_bin": 0,
                    "sum_conf": 0.0,
                    "sum_correct": 0.0,
                    "mean_conf": np.nan,
                    "accuracy": np.nan,
                }
            )
            continue
        sum_conf = float(conf[mask].sum())
        sum_correct = float(corr[mask].sum())
        rows.append(
            {
                "bin_id": int(idx),
                "n_bin": n_bin,
                "sum_conf": sum_conf,
                "sum_correct": sum_correct,
                "mean_conf": sum_conf / n_bin,
                "accuracy": sum_correct / n_bin,
            }
        )
    return pd.DataFrame(rows, columns=RELIABILITY_COLUMNS)


def reliability_bins_equal_width(
    confidence: np.ndarray,
    correct: np.ndarray,
    bins: int,
) -> pd.DataFrame:
    """Return reliability statistics using equal-width confidence bins."""
    if bins <= 0:
        raise ValueError(f"bins must be positive, got {bins}")

    conf, corr = _validate_inputs(confidence, correct)
    if conf.shape[0] == 0:
        return _aggregate_bins(np.array([], dtype=int), conf, corr, bins)

    clipped = np.clip(conf, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    # Put confidence=1.0 in the last bin.
    bin_ids = np.digitize(clipped, edges[1:-1], right=False).astype(int)
    return _aggregate_bins(bin_ids, clipped, corr, bins)


def reliability_bins_equal_mass(
    confidence: np.ndarray,
    correct: np.ndarray,
    bins: int,
) -> pd.DataFrame:
    """Return reliability statistics using equal-mass confidence bins."""
    if bins <= 0:
        raise ValueError(f"bins must be positive, got {bins}")

    conf, corr = _validate_inputs(confidence, correct)
    if conf.shape[0] == 0:
        return _aggregate_bins(np.array([], dtype=int), conf, corr, bins)

    order = np.argsort(conf, kind="stable")

    n = conf.shape[0]
    bin_ids = np.zeros(n, dtype=int)
    for idx in range(n):
        bin_ids[idx] = min((idx * bins) // n, bins - 1)

    inv_order = np.empty_like(order)
    inv_order[order] = np.arange(n)
    unsorted_bins = bin_ids[inv_order]

    return _aggregate_bins(unsorted_bins, conf, corr, bins)


def build_reliability_frame(
    *,
    confidence: np.ndarray,
    correct: np.ndarray,
    bins: int,
    binning: str,
) -> pd.DataFrame:
    """Build reliability bins using either equal-width or equal-mass strategy."""
    mode = binning.strip().lower()
    if mode == "equal_width":
        return reliability_bins_equal_width(confidence, correct, bins)
    if mode == "equal_mass":
        return reliability_bins_equal_mass(confidence, correct, bins)
    raise ValueError(f"binning must be one of ['equal_width', 'equal_mass'], got: {binning}")
