"""Per-sample UQ trace persistence utilities."""

import hashlib
import json
import logging
import os
import subprocess
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

TRACE_REQUIRED_COLUMNS: tuple[str, ...] = (
    "trace_block_key",
    "run_id",
    "config_hash",
    "git_sha",
    "created_at_utc",
    "model",
    "backbone",
    "dataset",
    "partition",
    "bands",
    "normalization",
    "image_size",
    "interpolation",
    "uq_method",
    "corruption_type",
    "severity",
    "seed",
    "sample_id",
    "sample_idx",
    "y_true",
    "y_pred",
    "max_probability",
    "confidence",
    "predictive_entropy",
    "normalized_predictive_entropy",
    "is_error",
    "correct",
)

TRACE_KEY_COLUMNS: tuple[str, ...] = (
    "model",
    "backbone",
    "dataset",
    "normalization",
    "image_size",
    "interpolation",
    "partition",
    "bands",
    "uq_method",
    "corruption_type",
    "severity",
    "seed",
)

TRACE_LINK_COLUMNS: tuple[str, ...] = (
    "trace_dataset_root",
    "trace_run_id",
    "trace_block_key",
)

TRACE_PARTITION_COLUMNS: tuple[str, ...] = (
    "dataset",
    "backbone",
    "uq_method",
    "corruption_type",
    "severity",
)


class TraceConfigError(ValueError):
    """Raised when trace configuration is invalid."""


class TraceWriteError(RuntimeError):
    """Raised when trace persistence fails."""


def utc_now_iso() -> str:
    """Return UTC timestamp in ISO-8601 format."""
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def build_config_hash(config_obj: dict[str, object]) -> str:
    """Return a deterministic hash for the provided config object."""
    payload = json.dumps(config_obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve_git_sha() -> str:
    """Return the current git SHA when available."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ""
    return out.strip()


def _safe_part(value: object) -> str:
    text = str(value)
    return (
        text.replace("/", "_")
        .replace("\\", "_")
        .replace("=", "-")
        .replace(":", "-")
        .replace(" ", "_")
    )


def _generate_run_id() -> str:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}-{uuid4().hex[:8]}"


def _trace_partitioning() -> ds.Partitioning:
    schema = pa.schema(
        [
            pa.field("dataset", pa.string()),
            pa.field("backbone", pa.string()),
            pa.field("uq_method", pa.string()),
            pa.field("corruption_type", pa.string()),
            pa.field("severity", pa.int32()),
        ]
    )
    return ds.partitioning(schema=schema, flavor="hive")


def _trace_fragment_paths(root: Path) -> list[str]:
    paths: list[str] = []
    for path in root.rglob("*.parquet"):
        rel_parts = path.relative_to(root).parts
        if len(rel_parts) < 6:
            continue
        if not rel_parts[0].startswith("dataset="):
            continue
        if not rel_parts[1].startswith("backbone="):
            continue
        if not rel_parts[2].startswith("uq_method="):
            continue
        if not rel_parts[3].startswith("corruption_type="):
            continue
        if not rel_parts[4].startswith("severity="):
            continue
        paths.append(str(path))
    return paths


def init_trace_run(
    *,
    trace_dataset_root: str,
    run_id: str | None,
    config_hash: str,
    resume: bool,
) -> dict[str, str]:
    """Initialize trace dataset metadata for the current run."""
    root = Path(trace_dataset_root)
    root.mkdir(parents=True, exist_ok=True)

    resolved_run_id = (run_id or "").strip()
    if not resolved_run_id:
        resolved_run_id = f"cfg-{config_hash[:12]}" if resume else _generate_run_id()

    return {
        "run_id": resolved_run_id,
        "trace_dataset_root": str(root),
        "config_hash": config_hash,
        "git_sha": resolve_git_sha(),
        "created_at_utc": utc_now_iso(),
    }


def build_trace_block_key(
    *,
    run_id: str,
    common_meta: Mapping[str, object],
    uq_method: str,
    corruption_type: str,
    severity: int,
) -> str:
    """Return a deterministic block key for one trace-producing evaluation block."""
    payload = {
        "run_id": run_id,
        "model": str(common_meta.get("model", "")),
        "backbone": str(common_meta.get("backbone", common_meta.get("name", ""))),
        "dataset": str(common_meta.get("dataset", "")),
        "partition": str(common_meta.get("partition", "")),
        "bands": str(common_meta.get("bands", "")),
        "normalization": str(common_meta.get("normalization", "")),
        "image_size": common_meta.get("image_size"),
        "interpolation": str(common_meta.get("interpolation", "")),
        "uq_method": str(uq_method),
        "corruption_type": str(corruption_type),
        "severity": int(severity),
        "seed": int(common_meta.get("seed", 0)),
    }
    data = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()[:20]


def resolve_trace_partition_path(
    *,
    trace_dataset_root: str,
    trace_block_key: str,
    dataset: str,
    backbone: str,
    uq_method: str,
    corruption_type: str,
    severity: int,
) -> Path:
    """Return the parquet fragment path for one completed trace block."""
    return (
        Path(trace_dataset_root)
        / f"dataset={_safe_part(dataset)}"
        / f"backbone={_safe_part(backbone)}"
        / f"uq_method={_safe_part(uq_method)}"
        / f"corruption_type={_safe_part(corruption_type)}"
        / f"severity={int(severity)}"
        / f"trace_block_key={trace_block_key}.parquet"
    )


def build_trace_link_row(
    *,
    trace_dataset_root: str,
    run_id: str,
    trace_block_key: str,
) -> dict[str, str]:
    """Return scalar-result columns that link back to trace rows."""
    return {
        "trace_dataset_root": str(Path(trace_dataset_root)),
        "trace_run_id": run_id,
        "trace_block_key": trace_block_key,
    }


def _per_sample_entropy(probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(probs, 1e-12, 1.0)
    return (-clipped * np.log(clipped)).sum(axis=1)


def _per_sample_normalized_entropy(probs: np.ndarray) -> np.ndarray:
    if probs.shape[1] <= 1:
        return np.zeros(probs.shape[0], dtype=np.float64)
    return _per_sample_entropy(probs) / np.log(float(probs.shape[1]))


def _normalize_sample_ids(
    *,
    sample_ids: Sequence[str] | np.ndarray | None,
    sample_idx: np.ndarray,
    common_meta: Mapping[str, object],
) -> np.ndarray:
    dataset = str(common_meta.get("dataset", "unknown"))
    partition = str(common_meta.get("partition", "default"))
    fallback = np.array(
        [f"{dataset}:{partition}:{int(idx)}" for idx in sample_idx],
        dtype=object,
    )
    if sample_ids is None:
        return fallback

    values = np.asarray(sample_ids, dtype=object)
    if values.shape[0] != sample_idx.shape[0]:
        raise ValueError("sample_ids and sample_idx must have equal first dimension")

    normalized = np.array([str(value).strip() for value in values], dtype=object)
    empty_mask = normalized == ""
    normalized[empty_mask] = fallback[empty_mask]
    return normalized


def _base_trace_columns(
    *,
    trace_block_key: str,
    run_id: str,
    common_meta: Mapping[str, object],
    uq_method: str,
    corruption_type: str,
    severity: int,
    config_hash: str,
    git_sha: str,
    created_at_utc: str,
    sample_idx: np.ndarray,
    sample_ids: Sequence[str] | np.ndarray | None,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, object]:
    correct = (y_true == y_pred).astype(np.int8)
    return {
        "trace_block_key": trace_block_key,
        "run_id": run_id,
        "config_hash": config_hash,
        "git_sha": git_sha,
        "created_at_utc": created_at_utc,
        "model": str(common_meta.get("model", "")),
        "backbone": str(common_meta.get("backbone", common_meta.get("name", ""))),
        "dataset": str(common_meta.get("dataset", "")),
        "partition": str(common_meta.get("partition", "")),
        "bands": str(common_meta.get("bands", "")),
        "normalization": str(common_meta.get("normalization", "")),
        "image_size": common_meta.get("image_size"),
        "interpolation": str(common_meta.get("interpolation", "")),
        "uq_method": str(uq_method),
        "corruption_type": str(corruption_type),
        "severity": int(severity),
        "seed": int(common_meta.get("seed", 0)),
        "sample_id": _normalize_sample_ids(
            sample_ids=sample_ids,
            sample_idx=sample_idx,
            common_meta=common_meta,
        ),
        "sample_idx": sample_idx.astype(np.int64),
        "y_true": y_true.astype(np.int64),
        "y_pred": y_pred.astype(np.int64),
        "is_error": (1 - correct).astype(np.int8),
        "correct": correct,
    }


def build_probabilistic_trace_frame(
    *,
    trace_block_key: str,
    run_id: str,
    common_meta: Mapping[str, object],
    uq_method: str,
    corruption_type: str,
    severity: int,
    config_hash: str,
    git_sha: str,
    created_at_utc: str,
    y_true: np.ndarray,
    probs: np.ndarray,
    sample_ids: Sequence[str] | np.ndarray | None = None,
) -> pd.DataFrame:
    """Build a per-sample trace frame for probabilistic UQ methods."""
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2D, got shape {probs.shape}")
    if y_true.ndim != 1:
        raise ValueError(f"y_true must be 1D, got shape {y_true.shape}")
    if probs.shape[0] != y_true.shape[0]:
        raise ValueError("probs and y_true must have equal first dimension")

    n = y_true.shape[0]
    y_pred = probs.argmax(axis=1).astype(np.int64)
    confidence = probs.max(axis=1).astype(np.float64)
    sample_idx = np.arange(n, dtype=np.int64)

    data = _base_trace_columns(
        trace_block_key=trace_block_key,
        run_id=run_id,
        common_meta=common_meta,
        uq_method=uq_method,
        corruption_type=corruption_type,
        severity=severity,
        config_hash=config_hash,
        git_sha=git_sha,
        created_at_utc=created_at_utc,
        sample_idx=sample_idx,
        sample_ids=sample_ids,
        y_true=y_true,
        y_pred=y_pred,
    )
    data["max_probability"] = confidence
    data["confidence"] = confidence
    data["predictive_entropy"] = _per_sample_entropy(probs).astype(np.float64)
    data["normalized_predictive_entropy"] = _per_sample_normalized_entropy(probs).astype(
        np.float64
    )

    frame = pd.DataFrame(data)
    return frame[list(TRACE_REQUIRED_COLUMNS)]


def build_conformal_trace_frame(
    *,
    trace_block_key: str,
    run_id: str,
    common_meta: Mapping[str, object],
    uq_method: str,
    corruption_type: str,
    severity: int,
    config_hash: str,
    git_sha: str,
    created_at_utc: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pred_sets: np.ndarray,
    sample_ids: Sequence[str] | np.ndarray | None = None,
) -> pd.DataFrame:
    """Build a per-sample trace frame for conformal prediction."""
    if pred_sets.ndim != 2:
        raise ValueError(f"pred_sets must be 2D, got shape {pred_sets.shape}")
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError("y_true and y_pred must be 1D")
    n = y_true.shape[0]
    if pred_sets.shape[0] != n or y_pred.shape[0] != n:
        raise ValueError("pred_sets, y_true, and y_pred must align on first dimension")

    sample_idx = np.arange(n, dtype=np.int64)
    set_size = pred_sets.sum(axis=1).astype(np.int64)
    confidence = (1.0 / np.maximum(set_size.astype(np.float64), 1.0)).astype(np.float64)

    data = _base_trace_columns(
        trace_block_key=trace_block_key,
        run_id=run_id,
        common_meta=common_meta,
        uq_method=uq_method,
        corruption_type=corruption_type,
        severity=severity,
        config_hash=config_hash,
        git_sha=git_sha,
        created_at_utc=created_at_utc,
        sample_idx=sample_idx,
        sample_ids=sample_ids,
        y_true=y_true,
        y_pred=y_pred,
    )
    data["max_probability"] = confidence
    data["confidence"] = confidence
    data["predictive_entropy"] = np.nan
    data["normalized_predictive_entropy"] = np.nan

    frame = pd.DataFrame(data)
    frame["set_size"] = set_size
    frame["is_covered"] = pred_sets[np.arange(n), y_true.astype(np.int64)].astype(np.int8)
    ordered_cols = list(TRACE_REQUIRED_COLUMNS) + ["set_size", "is_covered"]
    return frame[ordered_cols]


def write_trace_block_atomic(
    *,
    trace_path: Path,
    trace_df: pd.DataFrame,
    compression: str,
) -> None:
    """Write a trace parquet fragment atomically."""
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = trace_path.with_name(f"{trace_path.name}.tmp.{os.getpid()}.{uuid4().hex}")

    try:
        trace_df.to_parquet(tmp_path, index=False, compression=compression)
    except (ImportError, ModuleNotFoundError) as exc:
        raise TraceWriteError("Parquet trace writing requires pyarrow.") from exc

    os.replace(tmp_path, trace_path)


def read_trace_row_count(trace_path: Path) -> int | None:
    """Return the number of rows in a parquet trace fragment when it exists."""
    if not trace_path.exists():
        return None
    return int(pq.ParquetFile(trace_path).metadata.num_rows)


def check_trace_block_status(
    *,
    trace_path: Path,
    expected_n_test: int,
) -> dict[str, object]:
    """Return completeness and integrity status for one trace block."""
    n_rows = read_trace_row_count(trace_path)
    exists = n_rows is not None
    is_complete = exists and n_rows == expected_n_test
    return {
        "exists": exists,
        "is_complete": bool(is_complete),
        "row_count_mismatch": exists and n_rows != expected_n_test,
        "n_rows": n_rows,
    }


def maybe_warn_trace_integrity(
    *,
    status: Mapping[str, object],
    trace_path: Path,
    block_key: str,
) -> None:
    """Emit warnings for incomplete or inconsistent trace block state."""
    if bool(status.get("row_count_mismatch", False)):
        logger.warning("Trace row-count mismatch for block=%s (path=%s)", block_key, trace_path)


def _coerce_filter_values(key: str, values: Sequence[object]) -> list[object]:
    if key == "severity":
        return [int(value) for value in values]
    return [str(value) for value in values]


def _coerce_filter_value(key: str, value: object) -> object:
    if key == "severity":
        return int(value)
    return str(value)


def _build_filter_expression(
    *,
    filters: Mapping[str, object] | None,
    block_keys: Sequence[str] | None,
) -> ds.Expression | None:
    expression: ds.Expression | None = None
    if block_keys:
        key_expr = ds.field("trace_block_key").isin([str(key) for key in block_keys])
        expression = key_expr if expression is None else expression & key_expr

    if not filters:
        return expression

    for key, value in filters.items():
        if value is None:
            continue
        if isinstance(value, Sequence) and not isinstance(value, str):
            values = list(value)
            if not values:
                continue
            clause = ds.field(key).isin(_coerce_filter_values(key, values))
        else:
            clause = ds.field(key) == _coerce_filter_value(key, value)
        expression = clause if expression is None else expression & clause
    return expression


def scan_traces(
    trace_dataset_root: str | Path,
    *,
    filters: Mapping[str, object] | None = None,
    block_keys: Sequence[str] | None = None,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Scan the parquet trace dataset with predicate pushdown."""
    root = Path(trace_dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Trace dataset root not found: {root}")

    fragment_paths = _trace_fragment_paths(root)
    if not fragment_paths:
        return pd.DataFrame(columns=list(columns or TRACE_REQUIRED_COLUMNS))

    dataset = ds.dataset(fragment_paths, format="parquet")
    expression = _build_filter_expression(filters=filters, block_keys=block_keys)
    table = dataset.to_table(columns=list(columns) if columns is not None else None, filter=expression)
    return table.to_pandas()
