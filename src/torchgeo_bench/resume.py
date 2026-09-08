"""Resume keys and plans for unfinished benchmark work.

Use :func:`_canonical_key_cell` to make equivalent config and CSV values compare equal.
"""

import hashlib
import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from torchgeo_bench.intrinsic_dim import FEATURE_SPECTRUM_METRICS

logger = logging.getLogger(__name__)

KEY_COLS = (
    "dataset",
    "method",
    "model",
    "name",
    "normalization",
    "image_size",
    "interpolation",
    "partition",
    "bands",
    "num_classes",
    # Dataset overrides can change resolution or pooling under one model name (Scale-MAE, #215).
    "res",
    "pool",
    # Include result-affecting settings not represented by the other columns.
    "config_hash",
)


def _normalize_bands_value(bands: Iterable[object] | None) -> str:
    """Convert a band selection to a stable string for logs, CSVs, and resume keys.

    Accept ``"rgb"``/``"all"``, explicit lists (``ListConfig`` or ``list[str]``), or ``None``.
    Lists become comma-separated names; ``None`` becomes ``"all"``.
    """
    if bands is None:
        return "all"
    if isinstance(bands, str):
        return bands
    return ",".join(str(b) for b in bands)


def _resume_config_hash(cfg: DictConfig) -> str:
    """Return a stable fingerprint of settings that can change a result row.

    Excluding ``eval.profile`` and ``eval.intrinsic_dim`` preserves existing probe keys.
    These passes have separate completion checks and do not change probe scores.
    """
    dataset_cfg = OmegaConf.to_container(cfg.dataset, resolve=True)
    assert isinstance(dataset_cfg, dict)
    dataset_cfg.pop("names", None)
    eval_cfg = OmegaConf.to_container(cfg.eval, resolve=True)
    assert isinstance(eval_cfg, dict)
    eval_cfg.pop("profile", None)
    eval_cfg.pop("intrinsic_dim", None)
    # Keep removed defaults only in the hash payload so existing CSV keys remain valid.
    if "segmentation" in eval_cfg:
        eval_cfg["segmentation"].setdefault("save_viz", False)
        eval_cfg["segmentation"].setdefault("viz_dir", "viz")
        eval_cfg["segmentation"].setdefault("n_viz_samples", 8)
    payload = {
        "version": 1,
        "seed": cfg.seed,
        "device": cfg.device,
        "dataset": dataset_cfg,
        "eval": eval_cfg,
        "model": OmegaConf.to_container(cfg.model, resolve=True),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()[:16]


def _canonical_key_cell(value: object) -> str:
    """Normalize a resume-key cell so config and CSV values compare equally.

    Pandas may read ``224`` as ``224.0`` when a CSV column has missing values.
    Convert whole-number floats to integers for comparison; leave non-numeric values unchanged.
    """
    if value is None:
        return ""
    s = str(value).strip()
    if not s:
        return ""
    try:
        f = float(s)
    except ValueError:  # allow-except: resume keys contain both text and numeric CSV cells
        return s
    return str(int(f)) if f.is_integer() else s


def _completed_run_keys(
    existing_df: pd.DataFrame,
    key_cols: Sequence[str],
    metric_name: str | None = None,
) -> set[tuple[str, ...]]:
    """Build resume keys from existing rows, optionally requiring a metric."""
    df = existing_df
    if metric_name is not None:
        if "metric_name" not in df.columns:
            return set()
        df = df[df["metric_name"].fillna("").astype(str) == metric_name]
    rows = df[list(key_cols)].fillna("").to_numpy()
    return {tuple(_canonical_key_cell(cell) for cell in row) for row in rows}


def _row_key(row: dict, key_cols: Sequence[str]) -> tuple[str, ...]:
    return tuple(_canonical_key_cell(row.get(col, "")) for col in key_cols)


def _filter_completed_metric_rows(
    rows: list[dict],
    completed_metrics: dict[str, set[tuple[str, ...]]],
    key_cols: Sequence[str],
) -> list[dict]:
    """Drop rows whose (metric_name, resume-key) already exists in the output CSV."""
    filtered: list[dict] = []
    for row in rows:
        metric_name = str(row.get("metric_name", ""))
        key = _row_key(row, key_cols)
        if key in completed_metrics.get(metric_name, set()):
            continue
        filtered.append(row)
    return filtered


def _profile_metric_names(profile_cfg: DictConfig | None) -> list[str]:
    """Return the required profile metrics for resume completeness checks."""
    names = [
        "throughput_samples_per_sec",
        "latency_ms_per_batch_p50",
        "params_m",
    ]
    cpu_cfg = profile_cfg.get("cpu_throughput", {}) if profile_cfg else {}
    if bool(cpu_cfg.get("enabled", False)):
        names.extend(["throughput_samples_per_sec_cpu", "latency_ms_per_batch_p50_cpu"])
    return names


def load_completed(
    output_path: str, key_cols: Sequence[str] = KEY_COLS
) -> tuple[set[tuple[str, ...]], dict[str, set[tuple[str, ...]]]]:
    """Read an existing results CSV into resume-key sets (overall and per-metric)."""
    existing_df = pd.read_csv(output_path)
    for col in key_cols:
        if col not in existing_df.columns:
            existing_df[col] = ""
    completed_runs = _completed_run_keys(existing_df, key_cols)
    completed_metrics: dict[str, set[tuple[str, ...]]] = {}
    if "metric_name" in existing_df.columns:
        completed_metrics = {
            str(metric): _completed_run_keys(existing_df, key_cols, str(metric))
            for metric in existing_df["metric_name"].dropna().unique()
        }
    return completed_runs, completed_metrics


@dataclass
class ResumeState:
    """Completed result keys and per-metric keys from earlier runs."""

    completed_runs: set[tuple[str, ...]]
    completed_metrics: dict[str, set[tuple[str, ...]]]


@dataclass(frozen=True)
class DatasetRunPlan:
    """Resume-aware execution plan for one dataset/config combination."""

    metric_name: str
    skip_dataset: bool
    skip_knn: bool
    skip_linear: bool
    skip_id: bool
    skip_profile: bool
    id_missing_metrics: frozenset[str] = frozenset()
    knn_device: str | None = None


def _plan_dataset_run(
    cfg: DictConfig,
    ds_cls: type,
    common_meta: Mapping[str, object],
    completed: ResumeState,
    eval_cfg: DictConfig,
) -> DatasetRunPlan:
    """Plan which work remains for a dataset before loading data or a model."""
    ds_name = common_meta["dataset"]
    model_key = (
        cfg.model._target_,
        cfg.model.name,
        *(_canonical_key_cell(common_meta[key]) for key in KEY_COLS[4:]),
    )
    completed_runs, completed_metrics = completed.completed_runs, completed.completed_metrics

    if ds_cls.task == "segmentation":
        seg_key = (ds_name, f"seg-{eval_cfg.segmentation.head_type}", *model_key)
        return DatasetRunPlan(
            metric_name="mIoU",
            skip_dataset=bool(cfg.resume and seg_key in completed_runs),
            skip_knn=True,
            skip_linear=True,
            skip_id=True,
            skip_profile=True,
        )

    knn_key = (ds_name, f"knn{int(eval_cfg.get('knn_k', 5))}", *model_key)
    linear_key = (ds_name, "linear", *model_key)
    id_key = (ds_name, "intrinsic_dim", *model_key)
    profile_key = (ds_name, "profile", *model_key)

    skip_knn = bool(cfg.resume and knn_key in completed_runs)
    skip_linear = bool((cfg.resume and linear_key in completed_runs) or cfg.eval.skip_linear)

    id_cfg = getattr(cfg.eval, "intrinsic_dim", None)
    id_enabled = bool(id_cfg and id_cfg.get("enabled", False))
    id_metric_names = []
    if id_cfg is not None and id_enabled:
        for split in id_cfg.splits:
            id_metric_names.extend(f"id_{est}_{split}" for est in id_cfg.estimators)
            id_metric_names.extend(
                f"spectrum_{metric}_{split}" for metric in FEATURE_SPECTRUM_METRICS
            )
    # Track individual metrics so missing spectrum rows do not force expensive estimators to rerun.
    id_missing_metrics = frozenset(
        metric
        for metric in id_metric_names
        if not (cfg.resume and id_key in completed_metrics.get(metric, set()))
    )
    skip_id = (not id_enabled) or bool(id_metric_names and not id_missing_metrics)

    profile_cfg = getattr(cfg.eval, "profile", None)
    profile_enabled = bool(profile_cfg and profile_cfg.get("enabled", False))
    profile_metric_names = _profile_metric_names(profile_cfg) if profile_enabled else []
    skip_profile = (not profile_enabled) or bool(
        cfg.resume
        and profile_metric_names
        and all(
            profile_key in completed_metrics.get(metric, set()) for metric in profile_metric_names
        )
    )

    return DatasetRunPlan(
        metric_name="micro_mAP" if ds_cls.multilabel else "accuracy",
        skip_dataset=skip_knn and skip_linear and skip_id and skip_profile,
        skip_knn=skip_knn,
        skip_linear=skip_linear,
        skip_id=skip_id,
        skip_profile=skip_profile,
        id_missing_metrics=id_missing_metrics,
    )
