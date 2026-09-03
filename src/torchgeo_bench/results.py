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

import logging
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from filelock import FileLock

from torchgeo_bench.bootstrap import (
    bootstrap_accuracy as bootstrap_accuracy,
)
from torchgeo_bench.bootstrap import (
    bootstrap_map as bootstrap_map,
)
from torchgeo_bench.bootstrap import (
    bootstrap_miou as bootstrap_miou,
)

logger = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = "results/models"
# Profile (throughput/latency/params) and intrinsic-dim rows are one-time
# model+hardware measurements, so they live in their own per-model files
# instead of the metrics file that routine classification/segmentation
# sweeps rewrite.
DEFAULT_PROFILE_RESULTS_DIR = "results/profiles"
DEFAULT_INTRINSIC_DIM_RESULTS_DIR = "results/intrinsic_dim"

_UNSAFE = re.compile(r"[^A-Za-z0-9_.-]+")


class ResultSchemaError(ValueError):
    """Raised when a result CSV or appended row has an incompatible schema."""


def _read_results_csv(path: Path) -> pd.DataFrame:
    """Read a result CSV and add path context to parse failures."""
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise ResultSchemaError(f"Could not read result CSV {path}: {exc}") from exc


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
    columns: list[str] | None = None
    for path in paths:
        frame = _read_results_csv(path)
        if not frame.empty:
            current = list(frame.columns)
            if columns is None:
                columns = current
            elif current != columns:
                raise ResultSchemaError(
                    f"Result schema mismatch under {directory}: expected columns "
                    f"{columns}, but {path} has {current}."
                )
            frames.append(frame)
    if not frames:
        logger.warning("No results found under %s", directory)
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


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
    """Append rows with a lock and atomic whole-file replacement.

    Args:
        path: Output CSV path; created if missing.
        rows: Rows to append. Every row must have the same ordered keys as the
            existing CSV schema.

    Raises:
        ResultSchemaError: If appended rows or the existing file disagree on
            columns.
    """
    if not rows:
        return

    columns = list(rows[0])
    for index, row in enumerate(rows[1:], start=1):
        if list(row) != columns:
            raise ResultSchemaError(
                f"Row {index} columns {list(row)} do not match first row columns {columns}."
            )

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    new_rows = pd.DataFrame(rows, columns=columns)
    with FileLock(f"{output}.lock"):
        existing: pd.DataFrame | None = None
        existing_mode: int | None = None
        if output.exists() and output.stat().st_size:
            existing = _read_results_csv(output)
            existing_mode = stat.S_IMODE(output.stat().st_mode)
            if list(existing.columns) != columns:
                raise ResultSchemaError(
                    f"Result schema mismatch for {output}: existing columns "
                    f"{list(existing.columns)}, appended columns {columns}."
                )

        combined = (
            new_rows
            if existing is None
            else pd.concat([existing, new_rows], ignore_index=True, sort=False)
        )
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            text=True,
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", newline="") as file:
                combined.to_csv(file, index=False)
                file.flush()
                os.fsync(file.fileno())
            if existing_mode is not None:
                temporary.chmod(existing_mode)
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
