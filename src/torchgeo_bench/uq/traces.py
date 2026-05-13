"""Per-sample UQ trace persistence utilities."""

import hashlib
import json
import logging
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from torchgeo_bench.main import append_rows_atomic

logger = logging.getLogger(__name__)

TRACE_REQUIRED_COLUMNS: tuple[str, ...] = (
    "run_id",
    "model",
    "backbone",
    "name",
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
    "name",
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

MANIFEST_REQUIRED_COLUMNS: tuple[str, ...] = (
    "run_id",
    "model",
    "backbone",
    "name",
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
    "trace_path",
    "trace_format",
    "n_test",
    "schema_version",
    "created_at_utc",
    "config_hash",
    "git_sha",
)

MANIFEST_KEY_COLUMNS: tuple[str, ...] = (
    "run_id",
    "model",
    "name",
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


def _find_resume_run_id(trace_root: Path, config_hash: str) -> str | None:
    candidates: list[tuple[float, str]] = []
    for run_dir in trace_root.glob("run_id=*"):
        meta_path = run_dir / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if str(meta.get("config_hash", "")) != config_hash:
            continue
        run_id = str(meta.get("run_id", "")).strip()
        if not run_id:
            continue
        candidates.append((run_dir.stat().st_mtime, run_id))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]


def init_trace_run(
    *,
    trace_root: str,
    run_id: str | None,
    config_hash: str,
    trace_format: str,
    schema_version: str,
    resume: bool,
) -> dict[str, str]:
    """Initialize run directory and run metadata for trace persistence."""
    fmt = trace_format.lower().strip()
    if fmt not in {"parquet", "csv"}:
        raise TraceConfigError(f"uq.trace.format must be one of ['parquet', 'csv'], got: {trace_format}")

    root = Path(trace_root)
    root.mkdir(parents=True, exist_ok=True)

    resolved_run_id = (run_id or "").strip() or None
    if resolved_run_id is None and resume:
        resolved_run_id = _find_resume_run_id(root, config_hash)
    if resolved_run_id is None:
        resolved_run_id = _generate_run_id()

    run_dir = root / f"run_id={resolved_run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    meta_path = run_dir / "meta.json"
    manifest_path = run_dir / "manifest.csv"
    git_sha = resolve_git_sha()

    meta = {
        "run_id": resolved_run_id,
        "trace_root": str(root),
        "run_dir": str(run_dir),
        "trace_format": fmt,
        "schema_version": schema_version,
        "config_hash": config_hash,
        "git_sha": git_sha,
        "updated_at_utc": utc_now_iso(),
    }
    if meta_path.exists():
        try:
            prev_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            prev_meta = {}
        created_at = str(prev_meta.get("created_at_utc", "")).strip() or utc_now_iso()
    else:
        created_at = utc_now_iso()
    meta["created_at_utc"] = created_at

    tmp_path = meta_path.with_name(f"{meta_path.name}.tmp.{os.getpid()}.{uuid4().hex}")
    tmp_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp_path, meta_path)

    return {
        "run_id": resolved_run_id,
        "run_dir": str(run_dir),
        "manifest_path": str(manifest_path),
        "trace_format": fmt,
        "schema_version": schema_version,
        "config_hash": config_hash,
        "git_sha": git_sha,
    }


def resolve_trace_partition_path(
    *,
    trace_root: str,
    run_id: str,
    dataset: str,
    backbone: str,
    uq_method: str,
    corruption_type: str,
    severity: int,
    trace_format: str,
) -> Path:
    """Return partitioned trace output path for one evaluation block."""
    ext = "parquet" if trace_format.lower() == "parquet" else "csv"
    return (
        Path(trace_root)
        / f"run_id={_safe_part(run_id)}"
        / f"dataset={_safe_part(dataset)}"
        / f"backbone={_safe_part(backbone)}"
        / f"uq_method={_safe_part(uq_method)}"
        / f"corruption_type={_safe_part(corruption_type)}"
        / f"severity={int(severity)}"
        / f"part-000.{ext}"
    )


def _per_sample_entropy(probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(probs, 1e-12, 1.0)
    return (-clipped * np.log(clipped)).sum(axis=1)


def _per_sample_normalized_entropy(probs: np.ndarray) -> np.ndarray:
    if probs.shape[1] <= 1:
        return np.zeros(probs.shape[0], dtype=np.float64)
    return _per_sample_entropy(probs) / np.log(float(probs.shape[1]))


def _base_trace_columns(
    *,
    run_id: str,
    common_meta: dict[str, object],
    uq_method: str,
    corruption_type: str,
    severity: int,
    sample_idx: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, object]:
    correct = (y_true == y_pred).astype(np.int8)
    return {
        "run_id": run_id,
        "model": str(common_meta.get("model", "")),
        "backbone": str(common_meta.get("name", "")),
        "name": str(common_meta.get("name", "")),
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
        "sample_idx": sample_idx.astype(np.int64),
        "y_true": y_true.astype(np.int64),
        "y_pred": y_pred.astype(np.int64),
        "is_error": (1 - correct).astype(np.int8),
        "correct": correct,
    }


def build_probabilistic_trace_frame(
    *,
    run_id: str,
    common_meta: dict[str, object],
    uq_method: str,
    corruption_type: str,
    severity: int,
    y_true: np.ndarray,
    probs: np.ndarray,
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
        run_id=run_id,
        common_meta=common_meta,
        uq_method=uq_method,
        corruption_type=corruption_type,
        severity=severity,
        sample_idx=sample_idx,
        y_true=y_true,
        y_pred=y_pred,
    )
    data["max_probability"] = confidence
    data["confidence"] = confidence
    data["predictive_entropy"] = _per_sample_entropy(probs).astype(np.float64)
    data["normalized_predictive_entropy"] = _per_sample_normalized_entropy(probs).astype(np.float64)

    frame = pd.DataFrame(data)
    return frame[list(TRACE_REQUIRED_COLUMNS)]


def build_conformal_trace_frame(
    *,
    run_id: str,
    common_meta: dict[str, object],
    uq_method: str,
    corruption_type: str,
    severity: int,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pred_sets: np.ndarray,
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
        run_id=run_id,
        common_meta=common_meta,
        uq_method=uq_method,
        corruption_type=corruption_type,
        severity=severity,
        sample_idx=sample_idx,
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
    trace_format: str,
    compression: str,
) -> None:
    """Write trace block atomically in configured format."""
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = trace_path.with_name(f"{trace_path.name}.tmp.{os.getpid()}.{uuid4().hex}")

    fmt = trace_format.lower().strip()
    if fmt == "parquet":
        try:
            trace_df.to_parquet(tmp_path, index=False, compression=compression)
        except (ImportError, ModuleNotFoundError) as exc:
            raise TraceWriteError(
                "Parquet trace writing requires a parquet engine (e.g., pyarrow)."
            ) from exc
    elif fmt == "csv":
        trace_df.to_csv(tmp_path, index=False)
    else:
        raise TraceConfigError(f"Unsupported trace format: {trace_format}")

    os.replace(tmp_path, trace_path)


def append_manifest_row_atomic(manifest_path: str, row: dict[str, object]) -> None:
    """Append one manifest row with atomic CSV semantics."""
    append_rows_atomic(manifest_path, [row])


def build_manifest_row(
    *,
    trace_path: Path,
    trace_format: str,
    schema_version: str,
    config_hash: str,
    git_sha: str,
    n_test: int,
    run_id: str,
    common_meta: dict[str, object],
    uq_method: str,
    corruption_type: str,
    severity: int,
) -> dict[str, object]:
    """Build one manifest row for a completed trace block."""
    row: dict[str, object] = {
        "run_id": run_id,
        "model": str(common_meta.get("model", "")),
        "backbone": str(common_meta.get("name", "")),
        "name": str(common_meta.get("name", "")),
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
        "trace_path": str(trace_path),
        "trace_format": trace_format,
        "n_test": int(n_test),
        "schema_version": str(schema_version),
        "created_at_utc": utc_now_iso(),
        "config_hash": config_hash,
        "git_sha": git_sha,
    }
    return row


def _manifest_filter(df: pd.DataFrame, row: dict[str, object]) -> pd.Series:
    if df.empty:
        return pd.Series([], dtype=bool)

    mask = pd.Series(True, index=df.index)
    for col in MANIFEST_KEY_COLUMNS:
        lhs = df[col].fillna("").astype(str) if col in df.columns else ""
        rhs = "" if row.get(col) is None else str(row.get(col))
        mask &= lhs == rhs
    return mask


def _read_trace_n_rows(path: Path, trace_format: str) -> int | None:
    if not path.exists():
        return None
    fmt = trace_format.lower().strip()
    if fmt == "parquet":
        df = pd.read_parquet(path, columns=["sample_idx"])
        return int(len(df))
    if fmt == "csv":
        df = pd.read_csv(path, usecols=["sample_idx"])
        return int(len(df))
    raise TraceConfigError(f"Unsupported trace format: {trace_format}")


def check_trace_block_status(
    *,
    manifest_path: str,
    manifest_row: dict[str, object],
) -> dict[str, object]:
    """Return completeness and integrity status for one trace block."""
    status: dict[str, object] = {
        "manifest_rows": 0,
        "is_complete": False,
        "missing": False,
        "duplicate_manifest": False,
        "row_count_mismatch": False,
    }

    trace_path = Path(str(manifest_row["trace_path"]))
    trace_format = str(manifest_row["trace_format"])
    expected_n_test = int(manifest_row["n_test"])

    if not os.path.exists(manifest_path):
        status["missing"] = not trace_path.exists()
        n_rows = _read_trace_n_rows(trace_path, trace_format)
        status["is_complete"] = n_rows == expected_n_test
        status["row_count_mismatch"] = (n_rows is not None) and (n_rows != expected_n_test)
        return status

    manifest_df = pd.read_csv(manifest_path)
    for col in MANIFEST_KEY_COLUMNS:
        if col not in manifest_df.columns:
            manifest_df[col] = ""

    match_mask = _manifest_filter(manifest_df, manifest_row)
    match_count = int(match_mask.sum())
    status["manifest_rows"] = match_count
    status["duplicate_manifest"] = match_count > 1

    n_rows = _read_trace_n_rows(trace_path, trace_format)
    trace_ok = n_rows == expected_n_test
    status["row_count_mismatch"] = (n_rows is not None) and (n_rows != expected_n_test)
    status["missing"] = (match_count == 0) or (n_rows is None)
    status["is_complete"] = (match_count >= 1) and trace_ok
    return status


def maybe_warn_trace_integrity(
    *,
    status: dict[str, object],
    row: dict[str, object],
) -> None:
    """Emit warnings for incomplete or inconsistent trace block state."""
    key = (
        f"dataset={row['dataset']} backbone={row['backbone']} method={row['uq_method']} "
        f"corruption={row['corruption_type']} severity={row['severity']}"
    )
    if bool(status.get("duplicate_manifest", False)):
        logger.warning("Trace manifest has duplicate rows for %s", key)
    if bool(status.get("row_count_mismatch", False)):
        logger.warning("Trace row-count mismatch for %s (path=%s)", key, row["trace_path"])
    if bool(status.get("missing", False)) and int(status.get("manifest_rows", 0)) > 0:
        logger.warning("Trace file missing for manifest entry %s", key)
