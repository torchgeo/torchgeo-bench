"""Result rows, bootstrap CIs, atomic CSV writing, and per-model storage.

``EvaluationResult`` is the flat CSV schema — every field is a column with
downstream consumers in ``scripts/`` and ``experiments/``, so treat it as
frozen: append-only, never reorder.

Each run writes to ``results/models/<model name>.csv`` rather than one shared
CSV, so adding or re-running a model touches only that model's file.  Profile
and intrinsic-dim rows -- one-time model+hardware measurements -- are split
into their own ``results/profiles/<model name>.csv`` and
``results/intrinsic_dim/<model name>.csv`` files so a routine metrics rerun
doesn't touch them.  Use :func:`load_results` to read a whole directory back
as a single DataFrame.
"""

import io
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torchgeo_bench.resume import KEY_COLS, _canonical_key_cell

logger = logging.getLogger(__name__)

# The true "same measurement" identity for dedup: the resume key minus
# ``config_hash`` (that column is exactly what fragments legacy rows, written
# before it existed, from later hashed reruns of the same measurement), plus
# ``metric_name`` (one run emits several metric rows under the same resume key).
DEDUP_KEY_COLS = tuple(c for c in KEY_COLS if c != "config_hash") + ("metric_name",)

DEFAULT_RESULTS_DIR = "results/models"
# Profile (throughput/latency/params) and intrinsic-dim rows are one-time
# model+hardware measurements, so they live in their own per-model files
# instead of the metrics file that routine classification/segmentation
# sweeps rewrite.
DEFAULT_PROFILE_RESULTS_DIR = "results/profiles"
DEFAULT_INTRINSIC_DIM_RESULTS_DIR = "results/intrinsic_dim"

_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")


def sanitize_name(name: str) -> str:
    """Return ``name`` reduced to characters that are safe in a filename."""
    cleaned = _UNSAFE.sub("_", str(name)).strip("._")
    if not cleaned:
        raise ValueError(f"model name {name!r} has no filename-safe characters")
    return cleaned


def model_results_path(results_dir: str | Path, model_name: str) -> Path:
    """Return the CSV path holding ``model_name``'s results."""
    return Path(results_dir) / f"{sanitize_name(model_name)}.csv"


def load_results(
    results_dir: str | Path = DEFAULT_RESULTS_DIR,
    *,
    names: list[str] | None = None,
) -> pd.DataFrame:
    """Concatenate every per-model CSV under ``results_dir``.

    Args:
        results_dir: Directory holding ``<model name>.csv`` files.
        names: Restrict to these model names; ``None`` loads all.

    Returns:
        One DataFrame of all rows, or an empty one if nothing is stored yet.
    """
    directory = Path(results_dir)
    if names is not None:
        paths = [model_results_path(directory, n) for n in names]
        paths = [p for p in paths if p.exists()]
    else:
        paths = sorted(directory.glob("*.csv"))
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        logger.warning("No results found under %s", directory)
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True, sort=False)
    df = _relabel_olmoearth_normalization(df)
    return _dedup_results(df)


def _relabel_olmoearth_normalization(df: pd.DataFrame) -> pd.DataFrame:
    """Fix stale ``normalization`` labels on OlmoEarth rows.

    OlmoEarth handles its own input normalization (``handles_own_normalization
    = True`` in ``models/olmoearth.py``), but every existing result row
    predates that code change and is still labeled ``bandspec_zscore``.
    Relabel to ``model_native`` so OlmoEarth qualifies for the leaderboard
    generator's native-normalization branch like every other model does.
    """
    if "name" not in df.columns or "normalization" not in df.columns:
        return df
    is_olmoearth = df["name"].astype(str).str.startswith("olmoearth")
    is_zscore = df["normalization"].astype(str) == "bandspec_zscore"
    df.loc[is_olmoearth & is_zscore, "normalization"] = "model_native"
    return df


def _dedup_results(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate measurement rows left by the resume-hash gap.

    ``config_hash`` was added after many rows were already written, so those
    legacy rows (empty hash) don't match the resume key of a later rerun
    (real hash) and ``resume=true`` reran and appended instead of skipping.
    Within each "same measurement" group (:data:`DEDUP_KEY_COLS`, the resume
    key minus ``config_hash`` plus ``metric_name``), a hashed row always wins
    over a legacy one; among multiple hashed rows, the most recently appended
    one wins (``pd.concat`` above preserves file order).
    """
    missing = [c for c in DEDUP_KEY_COLS if c not in df.columns]
    if missing or "config_hash" not in df.columns:
        return df
    has_hash = df["config_hash"].fillna("").astype(str).str.len() > 0
    key = df[list(DEDUP_KEY_COLS)].apply(
        lambda row: tuple(_canonical_key_cell(v) for v in row), axis=1
    )
    order = pd.DataFrame({"key": key, "has_hash": has_hash, "pos": range(len(df))})
    keep_positions = order.sort_values(["has_hash", "pos"]).groupby("key", sort=False).tail(1)["pos"]
    return df.loc[sorted(keep_positions)].reset_index(drop=True)


def bootstrap_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 1000,
    ci: float = 95.0,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Bootstrapped accuracy with confidence interval. Returns (mean, ci_lower, ci_upper)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    idx = rng.integers(0, n, size=(n_boot, n))
    accs = (y_true[idx] == y_pred[idx]).mean(axis=1).astype(np.float32)
    acc_mean = float((y_true == y_pred).mean())
    lo = (100 - ci) / 2
    hi = 100 - lo
    lower = float(np.percentile(accs, lo))
    upper = float(np.percentile(accs, hi))
    return acc_mean, lower, upper


def bootstrap_map(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_boot: int = 1000,
    ci: float = 95.0,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Bootstrap micro-averaged mean Average Precision."""
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(seed)
    n = len(y_true)
    map_mean = float(average_precision_score(y_true, y_scores, average="micro"))
    valid_maps: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        # Skip degenerate resamples with no positive labels
        if yt.sum() == 0:
            continue
        valid_maps.append(average_precision_score(yt, y_scores[idx], average="micro"))
    if not valid_maps:
        return map_mean, map_mean, map_mean
    maps = np.array(valid_maps, dtype=np.float32)
    lo = (100 - ci) / 2
    hi = 100 - lo
    lower = float(np.percentile(maps, lo))
    upper = float(np.percentile(maps, hi))
    return map_mean, lower, upper


def bootstrap_miou(
    confusion_matrices: torch.Tensor,
    n_boot: int = 1000,
    ci: float = 95.0,
    seed: int | None = None,
) -> tuple[float, float]:
    """Bootstrap interval for dataset-level mIoU from per-image confusion matrices.

    ``confusion_matrices`` holds one ``(C, C)`` matrix per test image. Each
    resample draws images with replacement, sums their confusion matrices,
    and recomputes mIoU from the summed matrix -- mirroring how the metric is
    computed over the whole test set rather than averaging a per-image score.
    """
    if n_boot < 1:
        return float("nan"), float("nan")
    if confusion_matrices.ndim != 3 or confusion_matrices.shape[0] == 0:
        raise ValueError("Expected non-empty per-image confusion matrices with shape (N, C, C).")
    if confusion_matrices.shape[1] != confusion_matrices.shape[2]:
        raise ValueError("Confusion matrices must be square.")

    confusions = confusion_matrices.to(dtype=torch.float64, device="cpu")
    generator = torch.Generator().manual_seed(seed if seed is not None else 0)
    n_samples = len(confusions)
    scores = torch.empty(n_boot, dtype=torch.float64)
    for index in range(n_boot):
        sample = torch.randint(n_samples, (n_samples,), generator=generator)
        confusion = confusions[sample].sum(dim=0)
        intersection = confusion.diagonal()
        union = confusion.sum(dim=0) + confusion.sum(dim=1) - intersection
        present = union > 0
        scores[index] = (
            (intersection[present] / union[present]).mean() if present.any() else float("nan")
        )

    lower = (100.0 - ci) / 2.0
    return float(torch.nanquantile(scores, lower / 100.0)), float(
        torch.nanquantile(scores, 1 - lower / 100.0)
    )


@dataclass
class EvaluationResult:
    """Container for a single evaluation result row."""

    dataset: str
    method: str  # 'knn5', 'linear', or seg head type
    metric_name: str  # 'accuracy', 'micro_mAP', or 'mIoU' (primary metric)
    metric_value: float
    ci_lower: float
    ci_upper: float
    feature_dim: int
    best_c: float | None
    best_lr: float | None
    best_batch_size: int | None
    n_train: int
    n_val: int
    n_test: int
    seed: int
    model: str
    name: str
    normalization: str
    image_size: int | None
    interpolation: str
    partition: str
    bands: str
    num_classes: int
    # Fingerprint of the full config (seed, device, dataset, eval, model); see
    # ``resume._resume_config_hash``.  Part of the resume key.
    config_hash: str
    c_range_start: float
    c_range_stop: float
    c_range_num: int
    merge_val: bool
    bootstrap: int
    # Scale-MAE's dataset_overrides vary these per dataset (#215); they are
    # part of the resume key, so they must round-trip through the CSV.
    res: float | None = None
    pool: str | None = None
    # Segmentation-only metrics (None for classification rows)
    fw_iou: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    # Calibration metrics for KNN / Linear Probing (None for segmentation rows)
    ece: float | None = None
    rms_ce: float | None = None
    mce: float | None = None
    # Post temperature-scaling calibration (Linear Probing only; None for KNN/seg)
    ece_ts: float | None = None
    rms_ce_ts: float | None = None
    mce_ts: float | None = None
    temperature: float | None = None
    calibration_n_bins: int | None = None

    def to_row(self) -> dict:
        """Convert to a flat dictionary suitable for CSV/DataFrame export."""
        return self.__dict__.copy()


def metric_row(
    common_meta: dict,
    *,
    method: str,
    metric_name: str,
    metric_value: float,
    feature_dim: int,
    n_counts: dict[str, int],
    **extra: object,
) -> dict:
    """Build one CSV row dict; CIs default to 0 and probe hyperparams to None."""
    return EvaluationResult(
        **common_meta,
        method=method,
        metric_name=metric_name,
        metric_value=metric_value,
        feature_dim=feature_dim,
        ci_lower=extra.pop("ci_lower", 0.0),
        ci_upper=extra.pop("ci_upper", 0.0),
        best_c=extra.pop("best_c", None),
        best_lr=extra.pop("best_lr", None),
        best_batch_size=extra.pop("best_batch_size", None),
        n_train=n_counts.get("train", 0),
        n_val=n_counts.get("val", 0),
        n_test=n_counts.get("test", 0),
        **extra,  # type: ignore[arg-type]
    ).to_row()


def append_rows_atomic(path: str, rows: list[dict]) -> None:
    """Append rows to a CSV atomically, with advisory file lock and schema healing.

    Behavior:

    - Empty/missing file: writes the header derived from ``rows`` and the rows.
    - Existing file whose header matches ``rows[0]`` keys exactly: appends
      rows without rewriting the header (fast path).
    - Existing file with a different schema (e.g. ``EvaluationResult`` gained
      a field since the file was first written): the file is rewritten with
      the unioned schema so every value lives under a named column instead
      of being silently stuffed into an unnamed position.

    Args:
        path: Output CSV path; created if missing.
        rows: List of dicts to append.  All dicts should share the same keys.
    """
    from filelock import FileLock

    if not rows:
        return
    df_local = pd.DataFrame(rows)
    # Cross-platform advisory lock (Linux/macOS/Windows) so concurrent SLURM
    # array tasks don't interleave their read-modify-append and corrupt the CSV.
    with FileLock(f"{path}.lock"):
        fd = os.open(path, os.O_RDWR | os.O_CREAT)
        # newline="" keeps the csv line terminators from being translated to
        # CRLF on Windows, which would otherwise inject blank rows.
        with os.fdopen(fd, "r+", newline="", closefd=True) as f:
            f.seek(0, os.SEEK_END)
            empty = f.tell() == 0
            buf = io.StringIO()
            if empty:
                df_local.to_csv(buf, header=True, index=False)
                f.write(buf.getvalue())
            else:
                f.seek(0)
                existing_df = pd.read_csv(f)
                if list(existing_df.columns) == list(df_local.columns):
                    df_local.to_csv(buf, header=False, index=False)
                    f.seek(0, os.SEEK_END)
                    f.write(buf.getvalue())
                else:
                    extra = [c for c in existing_df.columns if c not in df_local.columns]
                    ordered = list(df_local.columns) + extra
                    combined = pd.concat(
                        [existing_df, df_local], ignore_index=True, sort=False
                    ).reindex(columns=ordered)
                    logger.warning(
                        "CSV schema drift detected at %s: existing columns %s, "
                        "new columns %s. Rewriting with unioned schema %s.",
                        path,
                        list(existing_df.columns),
                        list(df_local.columns),
                        ordered,
                    )
                    f.seek(0)
                    f.truncate()
                    combined.to_csv(buf, header=True, index=False)
                    f.write(buf.getvalue())
            f.flush()
            os.fsync(f.fileno())
