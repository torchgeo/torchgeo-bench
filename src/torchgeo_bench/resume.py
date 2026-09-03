"""Resume-mode bookkeeping: canonical run keys and skip planning.

Resume keys are assembled from two sources that format values differently
(the config vs. a re-read CSV), so every cell goes through
:func:`_canonical_key_cell` before comparison.
"""

import hashlib
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from torchgeo_bench.intrinsic_dim import FEATURE_SPECTRUM_METRICS
from torchgeo_bench.settings import ProfileSettings, RunSettings, to_dict

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
    # Scale-MAE's dataset_overrides vary these per dataset (#215); without them
    # two runs at different resolutions share a resume key and the second is
    # silently skipped.
    "res",
    "pool",
    # Fingerprint of the full config (seed, device, dataset, eval, model) so a
    # changed setting that doesn't show up in the other key columns can't be
    # mistaken for an already-completed run.
    "config_hash",
)


def _normalize_bands_value(bands: object) -> str:
    """Canonicalize the ``cfg.dataset.bands`` value for logging/CSV/resume.

    The config hands us either ``"rgb"``/``"all"``, an explicit list
    (``ListConfig`` or ``list[str]``), or ``None``.  Reduce all of those to a
    stable string so that the resume key and the CSV column are comparable
    across runs.
    """
    if bands is None:
        return "all"
    if isinstance(bands, str):
        return bands
    try:
        items = [str(b) for b in bands]
    except TypeError:
        return str(bands)
    return ",".join(items)


def _resume_config_hash(cfg: RunSettings) -> str:
    """Return a stable fingerprint of settings that can change a result row.

    Resume must not treat a row from a different model/evaluation setup as
    complete merely because its display metadata happens to match.

    ``eval.profile`` and ``eval.intrinsic_dim`` are excluded: both are
    additive, independently-gated passes (their own ``skip_profile``/
    ``skip_id`` resume keys already track completion), not settings that
    change what a knn/linear row measures. Hashing them would mean flipping
    ``eval.profile.enabled=true`` for a follow-up profiling pass changes the
    fingerprint and makes resume treat already-computed knn/linear rows as a
    different config -- rerunning and duplicating them instead of skipping.
    """
    dataset_cfg = to_dict(cfg.dataset)
    dataset_cfg.pop("names", None)
    eval_cfg = to_dict(cfg.eval)
    eval_cfg.pop("profile", None)
    eval_cfg.pop("intrinsic_dim", None)
    payload = {
        "version": 1,
        "seed": cfg.seed,
        "device": cfg.device,
        "dataset": dataset_cfg,
        "eval": eval_cfg,
        "model": to_dict(cfg.model),
    }
    # Strict JSON: every value here comes from plain settings/mappings
    # (yaml.safe_load-parsed scalars, dicts, lists), so nothing should ever
    # need `default=str` to serialize -- a value that does is a bug, not a
    # type this hash should silently paper over.
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()[:16]


def _canonical_key_cell(value: object) -> str:
    """Canonicalize a single resume-key cell to a comparable string.

    The config formats ``image_size`` as the int ``224`` (→ ``"224"``) while
    pandas types any CSV column containing a missing value as ``float64``, so
    the same value round-trips as ``"224.0"``.  Without canonicalization those
    never compare equal and ``resume=true`` silently appends duplicate rows.
    Whole-number floats collapse to their integer form; non-numeric values
    pass through unchanged.
    """
    if value is None:
        return ""
    s = str(value).strip()
    if not s:
        return ""
    try:
        f = float(s)
    except (TypeError, ValueError):
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
    """Build a normalized resume key tuple from a result row dict."""
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


def _profile_metric_names(profile_cfg: ProfileSettings | None) -> list[str]:
    """Return the required profile metrics for resume completeness checks."""
    names = [
        "throughput_samples_per_sec",
        "latency_ms_per_batch_p50",
        "params_m",
    ]
    if profile_cfg is not None and profile_cfg.cpu_throughput.enabled:
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


def _plan_dataset_run(
    cfg: RunSettings,
    ds_name: str,
    ds_cls: type,
    knn_k: int,
    seg_method: str,
    config_tuple: tuple[str, ...],
    completed_runs: set[tuple[str, ...]],
    completed_metrics: dict[str, set[tuple[str, ...]]],
) -> DatasetRunPlan:
    """Plan which work remains for a dataset before loading data or a model."""
    model_key = (cfg.model["_target_"], cfg.model["name"], *config_tuple)

    if ds_cls.task == "segmentation":
        seg_key = (ds_name, seg_method, *model_key)
        return DatasetRunPlan(
            metric_name="mIoU",
            skip_dataset=bool(cfg.resume and seg_key in completed_runs),
            skip_knn=True,
            skip_linear=True,
            skip_id=True,
            skip_profile=True,
        )

    knn_key = (ds_name, f"knn{knn_k}", *model_key)
    linear_key = (ds_name, "linear", *model_key)
    id_key = (ds_name, "intrinsic_dim", *model_key)
    profile_key = (ds_name, "profile", *model_key)

    skip_knn = bool(cfg.resume and knn_key in completed_runs)
    skip_linear = bool((cfg.resume and linear_key in completed_runs) or cfg.eval.skip_linear)

    id_cfg = cfg.eval.intrinsic_dim
    id_enabled = id_cfg.enabled
    id_metric_names = []
    if id_enabled:
        for split in id_cfg.splits:
            id_metric_names.extend(f"id_{est}_{split}" for est in id_cfg.estimators)
            id_metric_names.extend(
                f"spectrum_{metric}_{split}" for metric in FEATURE_SPECTRUM_METRICS
            )
    # Per-metric, not just per-dataset: a run that already has its (expensive)
    # torchid estimator rows but is missing only the (cheap) newer spectrum
    # rows shouldn't have to redo the estimators just to backfill the rest.
    id_missing_metrics = frozenset(
        metric
        for metric in id_metric_names
        if not (cfg.resume and id_key in completed_metrics.get(metric, set()))
    )
    skip_id = (not id_enabled) or bool(id_metric_names and not id_missing_metrics)

    profile_cfg = cfg.eval.profile
    profile_enabled = profile_cfg.enabled
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
