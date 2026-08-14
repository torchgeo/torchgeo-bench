"""Hydra entry point and execution pipeline for UQ benchmark runs."""

import logging
import os
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchgeo.datasets.errors import DatasetNotFoundError

from torchgeo_bench.datasets import get_bench_dataset_class, get_datasets, list_datasets
from torchgeo_bench.datasets.base import BandSpec
from torchgeo_bench.linear import LogisticRegression
from torchgeo_bench.main import append_rows_atomic, resolve_init
from torchgeo_bench.models.interface import BenchModel
from torchgeo_bench.cka.hooks import HookCollector, validate_collected_activations
from torchgeo_bench.uq.corruptions import SKIP_POISSON_GAUSSIAN, CorruptionTransform
from torchgeo_bench.uq.distance import (
    SCORE_KINDS,
    ActivationReference,
    build_activation_references,
)
from torchgeo_bench.uq.error_pr import compute_error_pr
from torchgeo_bench.uq.methods import (
    BootstrapEnsemble,
    ConformalPredictor,
    DeepEnsemble,
    LaplaceProbe,
    SVGPProbe,
    TemperatureScaling,
    Uncalibrated,
)
from torchgeo_bench.uq.metrics import (
    brier_score,
    ece,
    empirical_coverage,
    excess_aurc,
    max_probability,
    mean_set_size,
    nll,
    normalized_predictive_entropy,
    predictive_entropy,
    raw_aurc,
    selective_accuracy,
    signed_ece,
)
from torchgeo_bench.uq.splits import stratified_cal_split
from torchgeo_bench.uq.traces import (
    build_config_hash,
    build_conformal_trace_frame,
    build_distance_trace_frame,
    build_probabilistic_trace_frame,
    build_trace_block_key,
    build_trace_link_row,
    check_trace_block_status,
    init_trace_run,
    maybe_warn_trace_integrity,
    resolve_trace_partition_path,
    write_trace_block_atomic,
)
from torchgeo_bench.utils import extract_features

logger = logging.getLogger(__name__)


class _ScaledProbe:
    """Wraps a probe so inputs are z-scored with a pre-fitted StandardScaler."""

    def __init__(self, probe: Any, scaler: Any) -> None:
        self._probe = probe
        self._scaler = scaler
        if hasattr(probe, "predict_confidence"):
            def predict_confidence(X: np.ndarray) -> np.ndarray:
                return probe.predict_confidence(scaler.transform(X))
            self.predict_confidence = predict_confidence

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._probe.predict_proba(self._scaler.transform(X))

class _L2NormProbe:
    """Wraps a probe so inputs are L2-normalized before each call, matching nf_pipeline."""

    def __init__(self, probe: Any) -> None:
        self._probe = probe

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-8)
        return self._probe.predict_proba(X / norms)

    def predict_confidence(self, X: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-8)
        return self._probe.predict_confidence(X / norms)


class _ZCAProbe:
    """Wraps a probe with ZCA + L2-norm preprocessing, matching nf_pipeline zca_l2 mode."""

    def __init__(self, probe: Any, mean: np.ndarray, W: np.ndarray) -> None:
        self._probe = probe
        self._mean = mean
        self._W = W

    def _transform(self, X: np.ndarray) -> np.ndarray:
        Z = (X - self._mean) @ self._W.T
        return Z / np.linalg.norm(Z, axis=1, keepdims=True).clip(min=1e-8)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._probe.predict_proba(self._transform(X))

    def predict_confidence(self, X: np.ndarray) -> np.ndarray:
        return self._probe.predict_confidence(self._transform(X))


warnings.filterwarnings("ignore", message="Dataset has no geotransform", category=UserWarning)

_RESUME_KEY_COLS: tuple[str, ...] = (
    "model",
    "name",
    "init",
    "seed",
    "dataset",
    "normalization",
    "image_size",
    "interpolation",
    "partition",
    "bands",
    "uq_method",
    "corruption_type",
    "severity",
    "svgp_inducing_init",
    "svgp_mixing_weights",
)

# Scalar metrics emitted per activation-distance block. Only ranking-based
# metrics are defined for a score with no probabilities. This tuple is the single
# source of truth shared by _expected_metrics (resume) and _run_distance_block
# (emission) — see _expected_metrics for why they must not drift apart.
DISTANCE_METRIC_NAMES: set[str] = {
    "raw_aurc",
    "eaurc",
    "selective_acc_90",
    "error_auroc",
}

_CLOUD_PATTERN_MODE_MAP: dict[str, str] = {
    "fixed_across_severity": "fixed",
    "independent_per_severity": "independent",
    "fixed": "fixed",
    "independent": "independent",
}


def _lookup_nf_hyperparams(
    nf_df: pd.DataFrame | None,
    model: str,
    name: str,
    dataset: str,
    partition: str,
    bands: str,
    preprocessing: str = "l2",
) -> tuple[float, float] | None:
    """Return ``(best_lr, best_wd)`` from NF prior results, or ``None`` if missing.

    Args:
        nf_df: DataFrame loaded from ``nf_results.csv``, or ``None``.
        model: Model ``_target_`` string.
        name: Model ``name`` field.
        dataset: Dataset name.
        partition: Partition name.
        bands: Bands value.
        preprocessing: Preprocessing variant used during HPO (``"l2"``, ``"standard"``,
            or ``"zca_l2"``). Only used when the ``preprocessing`` column is present.

    Returns:
        ``(best_lr, best_wd)`` tuple, or ``None`` if no matching row found.
    """
    if nf_df is None or nf_df.empty:
        return None
    mask = (
        (nf_df["model"] == model)
        & (nf_df["name"] == name)
        & (nf_df["dataset"] == dataset)
        & (nf_df["partition"] == partition)
        & (nf_df["bands"] == bands)
    )
    if "preprocessing" in nf_df.columns:
        mask = mask & (nf_df["preprocessing"] == preprocessing)
    sub = nf_df.loc[mask]
    if sub.empty:
        return None
    row = sub.iloc[0]
    return float(row["best_lr"]), float(row["best_wd"])


def _is_uq_classification_dataset(ds_cls: type) -> bool:
    """Return whether a dataset class is in scope for UQ runs.

    Args:
        ds_cls: Dataset class returned by ``get_bench_dataset_class``.

    Returns:
        ``True`` for single-label classification datasets, else ``False``.
    """
    return ds_cls.task == "classification" and not bool(getattr(ds_cls, "multilabel", False))


def _expand_dataset_list(names: str | Sequence[str]) -> list[str]:
    """Normalize dataset selectors into an explicit dataset name list.

    Args:
        names: Either ``"all"``, a comma-delimited string, or a sequence of names.

    Returns:
        Explicit dataset name list.
    """
    if isinstance(names, str):
        if names == "all":
            return list_datasets()
        return [name.strip() for name in names.split(",") if name.strip()]
    return list(names)


def _normalize_bands_value(bands: object) -> str:
    """Normalize a band selector to a stable CSV-friendly string value.

    Args:
        bands: Band selector from config.

    Returns:
        Canonical string representation used for result keys.
    """
    if bands is None:
        return "all"
    if isinstance(bands, str):
        return bands
    try:
        values = [str(v) for v in bands]
    except TypeError:
        return str(bands)
    return ",".join(values)


def _normalize_cloud_pattern_mode(cloud_pattern_mode: str) -> str:
    """Map pipeline cloud pattern modes to transform-compatible mode values.

    Args:
        cloud_pattern_mode: Cloud mode from config or internal callers.

    Returns:
        Mode accepted by ``CorruptionTransform``.

    Raises:
        ValueError: If the mode is not recognized.
    """
    try:
        return _CLOUD_PATTERN_MODE_MAP[cloud_pattern_mode]
    except KeyError as exc:
        raise ValueError(
            "uq.cloud_pattern_mode must be one of "
            f"{sorted(_CLOUD_PATTERN_MODE_MAP)}."
        ) from exc


def _is_distance_method(uq_method: str) -> bool:
    """Return whether a ``uq_method`` name denotes an activation-distance scorer.

    Args:
        uq_method: UQ method name, e.g. ``"maha@backbone.layer4"``.

    Returns:
        ``True`` for ``"<score_kind>@<layer>"`` keys built by
        :func:`torchgeo_bench.uq.distance.build_activation_references`.
    """
    return any(str(uq_method).startswith(f"{kind}@") for kind in SCORE_KINDS)


def _expected_metrics(uq_method: str) -> set[str]:
    """Return expected metric names for a UQ method.

    Args:
        uq_method: UQ method name.

    Returns:
        Set of metric names required to mark a resume block as complete.
    """
    if uq_method == "conformal":
        return {"accuracy", "empirical_coverage", "mean_set_size"}
    if _is_distance_method(uq_method):
        # Only ranking metrics are defined for a score with no probabilities.
        # This set and the metrics dict built in _run_distance_block are ONE
        # contract: if they diverge, every distance cell silently re-runs on
        # every resume. Change both or neither.
        return DISTANCE_METRIC_NAMES
    return {
        "accuracy",
        "ece",
        "signed_ece",
        "nll",
        "brier",
        "predictive_entropy",
        "normalized_predictive_entropy",
        "max_probability",
        "raw_aurc",
        "eaurc",
        "selective_acc_90",
    }


def _build_resume_set(csv_path: str) -> set[tuple[str, ...]]:
    """Build the set of completed UQ blocks from an existing CSV.

    Args:
        csv_path: Path to the UQ results CSV file.

    Returns:
        Set of resume keys that already contain a full metric set.
    """
    if not os.path.exists(csv_path):
        return set()

    df = pd.read_csv(csv_path)
    for col in (*_RESUME_KEY_COLS, "metric_name"):
        if col not in df.columns:
            # `init` predates no rows: every CSV written before the random-init
            # arm existed is entirely pretrained. Backfilling it with "" instead
            # would make every stored key miss the "pretrained" key that
            # `build_resume_key` now emits, silently re-running finished cells.
            df[col] = "pretrained" if col == "init" else ""
    if "init" in df.columns:
        df["init"] = df["init"].fillna("").replace("", "pretrained")
    df = df.fillna("")

    completed: set[tuple[str, ...]] = set()
    for key_vals, group in df.groupby(list(_RESUME_KEY_COLS), dropna=False):
        key_tuple = tuple(str(v) for v in key_vals)
        method_name = str(group["uq_method"].iloc[0])
        expected = _expected_metrics(method_name)
        present = {str(x) for x in group["metric_name"].tolist()}
        if expected.issubset(present):
            completed.add(key_tuple)
    return completed


def _do_lookup(prior_results: pd.DataFrame, row_filter: dict[str, Any]) -> float | None:
    """Resolve ``best_c`` from prior linear-probe results.

    Args:
        prior_results: Prior benchmark results table.
        row_filter: Key-value pairs used to filter to one matching row.

    Returns:
        ``best_c`` when a unique row exists, otherwise ``None``.

    Raises:
        ValueError: If the filter matches multiple linear rows.
    """
    subset = prior_results.copy()

    if "method" in subset.columns and "best_c" in subset.columns:
        for col, val in row_filter.items():
            if col not in subset.columns:
                continue
            subset = subset[subset[col].fillna("").astype(str) == str(val)]

        subset = subset[subset["method"].fillna("").astype(str).str.startswith("linear")]
        if subset.empty:
            return None
        if len(subset) > 1:
            raise ValueError(f"Found duplicate prior linear rows for lookup key: {row_filter}")

        best_c = subset["best_c"].iloc[0]
        if pd.isna(best_c):
            return None
        return float(best_c)

    # Fallback for sweep CSVs that only include dataset/model and C values.
    if "C" not in subset.columns or "dataset" not in subset.columns or "model" not in subset.columns:
        return None

    dataset = row_filter.get("dataset")
    if dataset is not None:
        subset = subset[subset["dataset"].fillna("").astype(str) == str(dataset)]
        if subset.empty:
            return None

    name = row_filter.get("name")
    if name is not None:
        subset = subset[subset["model"].fillna("").astype(str) == str(name)]
    if subset.empty:
        return None

    if "val_acc" in subset.columns:
        subset = subset.sort_values(by="val_acc", ascending=False)
    best_c = subset["C"].iloc[0]
    if pd.isna(best_c):
        return None
    logger.info(
        "Using sweep prior_results format without method/best_c; selecting C=%s for %s.",
        best_c,
        row_filter,
    )
    return float(best_c)


def _lookup_best_c(
    prior_results: pd.DataFrame,
    row_filter: dict[str, Any],
    *,
    alias_dataset: str | None = None,
) -> float | None:
    """Resolve ``best_c`` from direct match, with optional dataset alias fallback.

    Args:
        prior_results: Prior benchmark results table.
        row_filter: Key-value pairs used to filter to one matching row.
        alias_dataset: Optional dataset name to try when direct lookup misses.

    Returns:
        ``best_c`` when resolved; otherwise ``None``.
    """
    best_c = _do_lookup(prior_results, row_filter)
    if best_c is not None or alias_dataset is None:
        return best_c

    alias_filter = {**row_filter, "dataset": alias_dataset}
    best_c = _do_lookup(prior_results, alias_filter)
    if best_c is not None:
        logger.info(
            "best_c for dataset=%s not found; using alias=%s.",
            row_filter.get("dataset"),
            alias_dataset,
        )
    return best_c


def build_resume_key(
    *,
    common_meta: Mapping[str, Any],
    cfg: Any,
    dataset_name: str,
    normalization: str,
    bands_value: str,
    uq_method: str,
    corruption_type: str,
    severity: str,
) -> tuple[str, ...]:
    """Build a resume key matching the schema :func:`_build_resume_set` produces.

    Derived from :data:`_RESUME_KEY_COLS` rather than spelled out positionally:
    the two must agree in *width* as well as content, and a hand-written tuple
    silently stops matching whenever a key column is added (as happened when the
    two ``svgp_*`` columns were appended, leaving a 12-wide lookup against
    14-wide stored keys, so no cell ever resumed). ``init`` was appended for the
    same reason: it separates the pretrained and random-init arms structurally
    rather than relying on them being written to different files.

    Args:
        common_meta: Common metadata columns for the current cell.
        cfg: Hydra config, used for the model/seed fields.
        dataset_name: Dataset being evaluated.
        normalization: Normalization identifier.
        bands_value: Canonical band selector string.
        uq_method: UQ method or scorer name.
        corruption_type: Corruption name (or ``"clean"``).
        severity: Severity, already stringified.

    Returns:
        Tuple aligned with :data:`_RESUME_KEY_COLS`.
    """
    values: dict[str, Any] = {
        **common_meta,
        "model": str(cfg.model._target_),
        "name": str(cfg.model.name),
        "init": resolve_init(cfg.model),
        "seed": str(cfg.seed),
        "dataset": dataset_name,
        "normalization": normalization,
        "image_size": str(getattr(cfg.dataset, "image_size", None)),
        "interpolation": str(getattr(cfg.dataset, "interpolation", "bilinear")),
        "partition": str(cfg.dataset.partition),
        "bands": bands_value,
        "uq_method": uq_method,
        "corruption_type": corruption_type,
        "severity": severity,
    }
    return tuple(str(values.get(col, "")) for col in _RESUME_KEY_COLS)


def remap_labels_to_probe_classes(y: np.ndarray, probe_like: Any) -> np.ndarray:
    """Remap labels through ``probe.classes_`` so indices match the probe's columns.

    Probes re-index training labels to ``[0, n_unique)``; test labels are still
    original values. This matters when a class present in the test set was absent
    from training (e.g. sen12ms class 7), which would otherwise cause an
    ``IndexError`` in ``nll`` / ``ece`` / ``brier``. Ensembles expose ``classes_``
    directly; single probes expose it via ``_probe``.

    Shared by :func:`_run_uq_block` and :func:`_run_distance_block` so the two
    cannot drift: they describe the same samples and are joined on ``sample_id``
    downstream, so disagreeing label indexing would corrupt the join silently.

    Args:
        y: Labels with shape ``(N,)`` in original dataset indexing.
        probe_like: Object exposing ``classes_`` directly or via ``_probe``.

    Returns:
        Labels remapped to probe column indices, or ``y`` unchanged when no
        ``classes_`` attribute is reachable.
    """
    classes = getattr(probe_like, "classes_", None)
    if classes is None:
        inner = getattr(probe_like, "_probe", None)
        classes = getattr(inner, "classes_", None)
    if classes is None:
        return y
    classes = np.asarray(classes).astype(np.int64)
    remap = np.full(int(classes.max()) + 1, -1, dtype=np.int64)
    remap[classes] = np.arange(len(classes), dtype=np.int64)
    return remap[y]


def _run_uq_block(
    *,
    method_name: str,
    method: Any,
    output_path: str,
    common_meta: dict[str, Any],
    corruption_type: str,
    severity: int,
    ece_bins: int,
    conformal_alpha: float,
    n_cal: int,
    n_train: int,
    feature_dim: int,
    best_c: float,
    seed: int,
    ece_binning: str = "equal_width",
    X_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    sample_ids: np.ndarray | None = None,
    model: BenchModel | None = None,
    test_loader: DataLoader | None = None,
    device: str | torch.device = "cpu",
    band_specs: list[BandSpec] | None = None,
    cloud_pattern_mode: str = "fixed_across_severity",
    trace_ctx: dict[str, Any] | None = None,
    verbose: bool = False,
) -> list[dict]:
    """Evaluate one UQ method for one corruption condition.

    Args:
        method_name: UQ method identifier.
        method: UQ method instance exposing ``predict_proba`` or ``predict_sets``.
        output_path: Destination CSV path.
        common_meta: Common metadata columns shared across result rows.
        corruption_type: Corruption name (or ``"clean"``).
        severity: Corruption severity level.
        ece_bins: Number of bins for ECE.
        ece_binning: Binning mode for scalar ECE (``equal_width`` or ``equal_mass``).
        conformal_alpha: Miscoverage level for conformal methods.
        n_cal: Calibration sample count.
        n_train: Probe training sample count.
        feature_dim: Embedding feature dimension.
        best_c: Probe regularization hyperparameter.
        seed: Random seed used for corruption determinism.
        X_test: Optional precomputed test embeddings.
        y_test: Optional precomputed test labels.
        sample_ids: Optional stable sample identifiers aligned with ``y_test``.
        model: Optional model used to extract test embeddings when ``X_test`` is not provided.
        test_loader: Optional test loader used with ``model``.
        device: Device used for feature extraction when needed.
        band_specs: Band metadata required for non-clean corruptions.
        cloud_pattern_mode: Cloud RNG mode for cloud corruption.
        trace_ctx: Optional trace persistence context dictionary.
        verbose: Whether to enable verbose extraction logs.

    Returns:
        Appended CSV rows for this method/corruption block.

    Raises:
        ValueError: If required inputs are missing.
    """
    cloud_pattern_mode = _normalize_cloud_pattern_mode(cloud_pattern_mode)

    if X_test is None or y_test is None:
        if model is None or test_loader is None:
            raise ValueError("Either (X_test, y_test) or (model, test_loader) must be provided.")
        transforms = None
        if corruption_type != "clean":
            if band_specs is None:
                raise ValueError("band_specs is required for non-clean corruptions.")
            transforms = CorruptionTransform(
                corruption_type=corruption_type,
                severity=severity,
                seed=seed,
                band_specs=band_specs,
                dataset_name=common_meta["dataset"],
                cloud_pattern_mode=cloud_pattern_mode,
            )
        extracted = extract_features(
            model,
            test_loader,
            device,
            transforms=transforms,
            verbose=verbose,
            return_sample_ids=trace_ctx is not None,
        )
        if trace_ctx is not None:
            X_test, y_test, sample_ids = extracted
        else:
            X_test, y_test = extracted

    assert X_test is not None
    assert y_test is not None

    y_test = remap_labels_to_probe_classes(y_test, method)

    rows: list[dict[str, Any]] = []
    trace_df = None
    trace_link: dict[str, str] = {}
    trace_path: Path | None = None
    trace_block_key: str | None = None
    if method_name == "conformal":
        point_preds, pred_sets = method.predict_sets(X_test, alpha=conformal_alpha)
        metrics = {
            "accuracy": float((point_preds == y_test).mean()),
            "empirical_coverage": empirical_coverage(pred_sets, y_test),
            "mean_set_size": mean_set_size(pred_sets),
        }
        if trace_ctx and bool(trace_ctx.get("include_conformal", False)):
            trace_block_key = build_trace_block_key(
                common_meta=common_meta,
                uq_method=method_name,
                corruption_type=corruption_type,
                severity=int(severity),
            )
            trace_df = build_conformal_trace_frame(
                trace_block_key=trace_block_key,
                run_id=str(trace_ctx["run_id"]),
                common_meta=common_meta,
                uq_method=method_name,
                corruption_type=corruption_type,
                severity=int(severity),
                config_hash=str(trace_ctx["config_hash"]),
                git_sha=str(trace_ctx["git_sha"]),
                created_at_utc=str(trace_ctx["created_at_utc"]),
                y_true=y_test,
                y_pred=point_preds.astype(np.int64, copy=False),
                pred_sets=pred_sets,
                sample_ids=sample_ids,
            )
    else:
        probs = method.predict_proba(X_test)
        y_pred = probs.argmax(axis=1)
        # For methods with a dedicated confidence signal (e.g. ensemble BALD),
        # use it for ranking metrics; fall back to max probability otherwise.
        if hasattr(method, "predict_confidence"):
            conf = method.predict_confidence(X_test)
        else:
            conf = probs.max(axis=1)
        metrics = {
            "accuracy": float((y_pred == y_test).mean()),
            "ece": ece(probs, y_test, n_bins=ece_bins, binning=ece_binning),
            "signed_ece": signed_ece(probs, y_test, n_bins=ece_bins, binning=ece_binning),
            "nll": nll(probs, y_test),
            "brier": brier_score(probs, y_test),
            "predictive_entropy": predictive_entropy(probs),
            "normalized_predictive_entropy": normalized_predictive_entropy(probs),
            "max_probability": max_probability(probs),
            "raw_aurc": raw_aurc(conf, y_pred, y_test),
            "eaurc": excess_aurc(conf, y_pred, y_test),
            "selective_acc_90": selective_accuracy(conf, y_pred, y_test, coverage=0.9),
        }
        if trace_ctx:
            trace_block_key = build_trace_block_key(
                common_meta=common_meta,
                uq_method=method_name,
                corruption_type=corruption_type,
                severity=int(severity),
            )
            trace_df = build_probabilistic_trace_frame(
                trace_block_key=trace_block_key,
                run_id=str(trace_ctx["run_id"]),
                common_meta=common_meta,
                uq_method=method_name,
                corruption_type=corruption_type,
                severity=int(severity),
                config_hash=str(trace_ctx["config_hash"]),
                git_sha=str(trace_ctx["git_sha"]),
                created_at_utc=str(trace_ctx["created_at_utc"]),
                y_true=y_test,
                probs=probs,
                sample_ids=sample_ids,
            )

    if trace_ctx and trace_df is not None and trace_block_key is not None:
        trace_path = resolve_trace_partition_path(
            trace_dataset_root=str(trace_ctx["trace_dataset_root"]),
            trace_block_key=trace_block_key,
            dataset=str(common_meta["dataset"]),
            backbone=str(common_meta["backbone"]),
            uq_method=method_name,
            corruption_type=corruption_type,
            severity=int(severity),
        )
        status = check_trace_block_status(
            trace_path=trace_path,
            expected_n_test=int(len(y_test)),
        )
        maybe_warn_trace_integrity(
            status=status,
            trace_path=trace_path,
            block_key=trace_block_key,
        )
        if not bool(status["is_complete"]) or bool(trace_ctx["overwrite"]):
            write_trace_block_atomic(
                trace_path=trace_path,
                trace_df=trace_df,
                compression=str(trace_ctx["compression"]),
            )
        trace_link = build_trace_link_row(
            trace_dataset_root=str(trace_ctx["trace_dataset_root"]),
            run_id=str(trace_ctx["run_id"]),
            trace_block_key=trace_block_key,
        )

    for metric_name, metric_value in metrics.items():
        row = {
            **common_meta,
            "uq_method": method_name,
            "corruption_type": corruption_type,
            "severity": int(severity),
            "metric_name": metric_name,
            "metric_value": float(metric_value),
            "n_cal": int(n_cal),
            "n_train": int(n_train),
            "n_test": int(len(y_test)),
            "best_c": float(best_c),
            "feature_dim": int(feature_dim),
            **trace_link,
        }
        rows.append(row)

    append_rows_atomic(output_path, rows)
    return rows


def _run_distance_block(
    *,
    reference: ActivationReference,
    acts_test: np.ndarray,
    probe: Any,
    output_path: str,
    common_meta: dict[str, Any],
    corruption_type: str,
    severity: int,
    n_cal: int,
    n_train: int,
    feature_dim: int,
    best_c: float,
    X_test: np.ndarray,
    y_test: np.ndarray,
    sample_ids: np.ndarray | None = None,
    trace_ctx: dict[str, Any] | None = None,
) -> list[dict]:
    """Evaluate one activation-distance scorer for one corruption condition.

    Sibling of :func:`_run_uq_block` rather than a third branch inside it: a
    distance reference has no ``predict_proba``, and only the four ranking
    metrics in :data:`DISTANCE_METRIC_NAMES` are defined for it. Everything else
    — block keys, partition paths, integrity checks, atomic writes, CSV row shape
    — is the existing plumbing, reused unchanged.

    Args:
        reference: Fitted reference, already fitted on clean *train* activations.
        acts_test: Test activations for ``reference.layer_name``, shape ``(N, D)``.
        probe: Probe supplying predictions and probabilities (unchanged by the score).
        output_path: Destination CSV path.
        common_meta: Common metadata columns shared across result rows.
        corruption_type: Corruption name (or ``"clean"``).
        severity: Corruption severity level.
        n_cal: Calibration sample count.
        n_train: Probe training sample count.
        feature_dim: Head embedding feature dimension (matches the probabilistic rows).
        best_c: Probe regularization hyperparameter.
        X_test: Head embeddings used for the probe's predictions.
        y_test: Test labels in original dataset indexing.
        sample_ids: Optional stable sample identifiers aligned with ``y_test``.
        trace_ctx: Optional trace persistence context dictionary.

    Returns:
        Appended CSV rows for this scorer/corruption block.

    Raises:
        ValueError: If activations and labels are not row-aligned.
    """
    if acts_test.shape[0] != y_test.shape[0]:
        raise ValueError(
            f"acts_test has {acts_test.shape[0]} rows but y_test has {y_test.shape[0]}; "
            "hooked activations must be row-aligned with the extracted labels."
        )

    # Same remap as _run_uq_block, via the shared helper: y_pred/correct come from
    # the probe, so disagreeing label indexing would make these rows describe
    # different samples than the uncalibrated rows they are joined against.
    y_test = remap_labels_to_probe_classes(y_test, probe)

    probs = probe.predict_proba(X_test)
    y_pred = probs.argmax(axis=1)
    distance = reference.score(acts_test).astype(np.float64, copy=False)
    is_error = (y_pred != y_test).astype(np.int64)

    # Distance is an uncertainty; _risk_coverage_curve sorts by DESCENDING
    # confidence, so it enters negated. Getting this backwards inverts the curve
    # and produces a spectacular-looking spurious result.
    confidence = -distance

    metrics: dict[str, float] = {
        "raw_aurc": raw_aurc(confidence, y_pred, y_test),
        "eaurc": excess_aurc(confidence, y_pred, y_test),
        "selective_acc_90": selective_accuracy(confidence, y_pred, y_test, coverage=0.9),
    }
    try:
        # compute_error_pr returns 'auroc'; the resume contract demands the key
        # 'error_auroc'. This mapping IS that contract — see _expected_metrics.
        metrics["error_auroc"] = float(
            compute_error_pr(is_error=is_error, uncertainty=distance)["auroc"]
        )
    except ValueError:
        # Degenerate block (all-correct or all-wrong): AUROC is undefined. Emit
        # NaN rather than omitting the row, or the cell never resumes as complete.
        logger.warning(
            "error_auroc undefined for %s (dataset=%s corruption=%s severity=%d): "
            "is_error has a single class.",
            reference.key,
            common_meta.get("dataset"),
            corruption_type,
            severity,
        )
        metrics["error_auroc"] = float("nan")

    assert set(metrics) == DISTANCE_METRIC_NAMES, (
        "distance metric names diverged from the resume contract: "
        f"{sorted(set(metrics) ^ DISTANCE_METRIC_NAMES)}"
    )

    trace_link: dict[str, str] = {}
    if trace_ctx:
        trace_block_key = build_trace_block_key(
            common_meta=common_meta,
            uq_method=reference.key,
            corruption_type=corruption_type,
            severity=int(severity),
        )
        trace_df = build_distance_trace_frame(
            trace_block_key=trace_block_key,
            run_id=str(trace_ctx["run_id"]),
            common_meta=common_meta,
            uq_method=reference.key,
            corruption_type=corruption_type,
            severity=int(severity),
            config_hash=str(trace_ctx["config_hash"]),
            git_sha=str(trace_ctx["git_sha"]),
            created_at_utc=str(trace_ctx["created_at_utc"]),
            y_true=y_test,
            probs=probs,
            distance_score=distance,
            layer_name=reference.layer_name,
            score_kind=reference.score_kind,
            sample_ids=sample_ids,
        )
        trace_path = resolve_trace_partition_path(
            trace_dataset_root=str(trace_ctx["trace_dataset_root"]),
            trace_block_key=trace_block_key,
            dataset=str(common_meta["dataset"]),
            backbone=str(common_meta["backbone"]),
            uq_method=reference.key,
            corruption_type=corruption_type,
            severity=int(severity),
        )
        status = check_trace_block_status(
            trace_path=trace_path,
            expected_n_test=int(len(y_test)),
        )
        maybe_warn_trace_integrity(
            status=status,
            trace_path=trace_path,
            block_key=trace_block_key,
        )
        if not bool(status["is_complete"]) or bool(trace_ctx["overwrite"]):
            write_trace_block_atomic(
                trace_path=trace_path,
                trace_df=trace_df,
                compression=str(trace_ctx["compression"]),
            )
        trace_link = build_trace_link_row(
            trace_dataset_root=str(trace_ctx["trace_dataset_root"]),
            run_id=str(trace_ctx["run_id"]),
            trace_block_key=trace_block_key,
        )

    rows: list[dict[str, Any]] = []
    for metric_name, metric_value in metrics.items():
        rows.append(
            {
                **common_meta,
                "uq_method": reference.key,
                "corruption_type": corruption_type,
                "severity": int(severity),
                "metric_name": metric_name,
                "metric_value": float(metric_value),
                "n_cal": int(n_cal),
                "n_train": int(n_train),
                "n_test": int(len(y_test)),
                "best_c": float(best_c),
                "feature_dim": int(feature_dim),
                **trace_link,
            }
        )

    append_rows_atomic(output_path, rows)
    return rows


def _run_conditions(
    *,
    cfg: DictConfig,
    dataset_name: str,
    model: BenchModel,
    test_loader: DataLoader,
    device: torch.device,
    band_specs: list[BandSpec],
    cloud_pattern_mode: str,
    normalization: str,
    bands_value: str,
    methods: dict[str, Any],
    references: dict[str, ActivationReference],
    probe_predictor: Any,
    test_collector: HookCollector | None,
    hook_paths: list[str],
    common_meta: dict[str, Any],
    completed: set[tuple[str, ...]],
    output_path: str,
    trace_ctx: dict[str, Any] | None,
    X_cal: np.ndarray | None,
    X_final_train: np.ndarray,
    best_c: float | None,
) -> None:
    """Evaluate every (corruption, severity) condition for one (model, dataset) cell.

    Extracted from :func:`main` so the hook lifecycle around it stays a plain
    ``try/finally`` rather than an extra level of nesting inside the dataset loop.

    Args:
        cfg: Hydra configuration for the run.
        dataset_name: Dataset being evaluated.
        model: Instantiated backbone, already on ``device`` and in eval mode.
        test_loader: Test dataloader.
        device: Device used for feature extraction.
        band_specs: Band metadata required for non-clean corruptions.
        cloud_pattern_mode: Normalized cloud RNG mode.
        normalization: Normalization identifier for result keys.
        bands_value: Canonical band selector string for result keys.
        methods: Probabilistic UQ methods keyed by name.
        references: Fitted activation references keyed by ``"<kind>@<layer>"``.
        probe_predictor: Numpy-in/numpy-out probe wrapper supplying predictions
            for distance blocks (an :class:`Uncalibrated` around the probe).
        test_collector: Shared hook collector, or ``None`` when distance is off.
        hook_paths: Hook paths registered on ``test_collector``.
        common_meta: Common metadata columns shared across result rows.
        completed: Resume keys already present in the output CSV.
        output_path: Destination CSV path.
        trace_ctx: Optional trace persistence context dictionary.
        X_cal: Calibration embeddings, or ``None``.
        X_final_train: Probe training embeddings.
        best_c: Probe regularization hyperparameter, or ``None``.
    """
    for corruption_type in cfg.uq.corruptions:
        severities = [0] if corruption_type == "clean" else [int(s) for s in cfg.uq.corruption_severities]
        if corruption_type == "poisson_gaussian" and dataset_name in SKIP_POISSON_GAUSSIAN:
            logger.info("Skipping poisson_gaussian for dataset %s", dataset_name)
            continue

        for severity in severities:
            transform = None
            if corruption_type != "clean":
                transform = CorruptionTransform(
                    corruption_type=corruption_type,
                    severity=severity,
                    seed=cfg.seed,
                    band_specs=band_specs,
                    dataset_name=dataset_name,
                    cloud_pattern_mode=cloud_pattern_mode,
                )
            sample_ids: np.ndarray | None = None
            extracted = extract_features(
                model,
                test_loader,
                device,
                transforms=transform,
                verbose=cfg.verbose,
                return_sample_ids=trace_ctx is not None,
            )
            if trace_ctx:
                X_test, y_test, sample_ids = extracted
            else:
                X_test, y_test = extracted

            # Ordering is load-bearing: collect() must run after this extraction
            # returns and before the next one begins, or two conditions'
            # activations concatenate silently. The row-count check in
            # validate_collected_activations makes that failure loud.
            acts_test: dict[str, np.ndarray] = {}
            if test_collector is not None:
                acts_test = test_collector.collect()
                validate_collected_activations(
                    acts_test,
                    hook_paths,
                    int(len(y_test)),
                    f"{corruption_type}/severity={severity}",
                )

            # corruption/severity are bound as defaults rather than captured, so
            # the closure cannot drift with the loop variables.
            def _resume_key(
                uq_method: str,
                _corruption: str = str(corruption_type),
                _severity: str = str(int(severity)),
            ) -> tuple[str, ...]:
                return build_resume_key(
                    common_meta=common_meta,
                    cfg=cfg,
                    dataset_name=dataset_name,
                    normalization=normalization,
                    bands_value=bands_value,
                    uq_method=uq_method,
                    corruption_type=_corruption,
                    severity=_severity,
                )

            for method_name, method in methods.items():
                if cfg.resume and _resume_key(method_name) in completed:
                    if trace_ctx:
                        trace_block_key = build_trace_block_key(
                            common_meta=common_meta,
                            uq_method=str(method_name),
                            corruption_type=str(corruption_type),
                            severity=int(severity),
                        )
                        trace_path = resolve_trace_partition_path(
                            trace_dataset_root=str(trace_ctx["trace_dataset_root"]),
                            trace_block_key=trace_block_key,
                            dataset=str(dataset_name),
                            backbone=str(cfg.model.name),
                            uq_method=str(method_name),
                            corruption_type=str(corruption_type),
                            severity=int(severity),
                        )
                        status = check_trace_block_status(
                            trace_path=trace_path,
                            expected_n_test=int(len(y_test)),
                        )
                        if not bool(status["is_complete"]):
                            logger.warning(
                                "Resume skip: scalar metrics exist but trace block is incomplete/missing "
                                "(dataset=%s backbone=%s method=%s corruption=%s severity=%d).",
                                dataset_name,
                                cfg.model.name,
                                method_name,
                                corruption_type,
                                severity,
                            )
                    continue

                _run_uq_block(
                    method_name=method_name,
                    method=method,
                    output_path=output_path,
                    common_meta=common_meta,
                    corruption_type=str(corruption_type),
                    severity=int(severity),
                    ece_bins=int(cfg.uq.ece_bins),
                    ece_binning=str(getattr(cfg.uq, "ece_binning", "equal_width")),
                    conformal_alpha=float(cfg.uq.conformal_alpha),
                    n_cal=len(X_cal) if X_cal is not None else 0,
                    n_train=len(X_final_train),
                    feature_dim=int(X_final_train.shape[1]),
                    best_c=float(best_c) if best_c is not None else float("nan"),
                    seed=int(cfg.seed),
                    X_test=X_test,
                    y_test=y_test,
                    sample_ids=sample_ids,
                    cloud_pattern_mode=cloud_pattern_mode,
                    trace_ctx=trace_ctx,
                )

            # Activation scorers loop separately from the probabilistic methods:
            # a reference has no predict_proba, and only the four ranking metrics
            # are defined for it (see _run_distance_block).
            for scorer_key, reference in references.items():
                if cfg.resume and _resume_key(scorer_key) in completed:
                    continue
                _run_distance_block(
                    reference=reference,
                    acts_test=acts_test[reference.layer_name],
                    # Uncalibrated, not the bare probe: it supplies the same
                    # numpy-in/numpy-out predict_proba contract _run_uq_block
                    # uses (the raw LogisticRegression wants a torch.Tensor),
                    # and still exposes classes_ via _probe for the remap.
                    probe=probe_predictor,
                    output_path=output_path,
                    common_meta=common_meta,
                    corruption_type=str(corruption_type),
                    severity=int(severity),
                    n_cal=len(X_cal) if X_cal is not None else 0,
                    n_train=len(X_final_train),
                    feature_dim=int(X_final_train.shape[1]),
                    best_c=float(best_c) if best_c is not None else float("nan"),
                    X_test=X_test,
                    y_test=y_test,
                    sample_ids=sample_ids,
                    trace_ctx=trace_ctx,
                )


@hydra.main(config_path="../conf", config_name="uq_config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the Hydra-configured UQ evaluation pipeline.

    Args:
        cfg: Hydra configuration for model, datasets, and UQ settings.
    """
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    output_path = str(cfg.uq.output)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    completed = _build_resume_set(output_path) if bool(cfg.resume) else set()
    dataset_names = _expand_dataset_list(cfg.dataset.names)
    device = torch.device(str(cfg.device))
    bands_value = _normalize_bands_value(getattr(cfg.dataset, "bands", "rgb"))
    normalization = str(getattr(cfg.dataset, "normalization", "bandspec_zscore"))
    cloud_pattern_mode = _normalize_cloud_pattern_mode(
        str(getattr(cfg.uq, "cloud_pattern_mode", "fixed_across_severity"))
    )
    trace_cfg = getattr(cfg.uq, "trace", None)
    trace_enabled = bool(getattr(trace_cfg, "enabled", False)) if trace_cfg is not None else False
    trace_ctx: dict[str, Any] | None = None
    if trace_enabled:
        cfg_dict = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(cfg_dict, dict):
            raise TypeError("Hydra config must resolve to a dictionary.")
        config_hash = build_config_hash(cfg_dict)
        trace_dataset_root = str(
            getattr(trace_cfg, "dataset_root", getattr(trace_cfg, "root", "results/uq_traces"))
        )
        trace_run = init_trace_run(
            trace_dataset_root=trace_dataset_root,
            run_id=getattr(trace_cfg, "run_id", None),
            config_hash=config_hash,
            resume=bool(cfg.resume),
        )
        trace_ctx = {
            **trace_run,
            "compression": str(getattr(trace_cfg, "compression", "zstd")),
            "overwrite": bool(getattr(trace_cfg, "overwrite", False)),
            "include_conformal": bool(getattr(trace_cfg, "include_conformal", False)),
        }
        logger.info(
            "Trace persistence enabled (run_id=%s, root=%s).",
            trace_ctx["run_id"],
            trace_ctx["trace_dataset_root"],
        )

    distance_cfg = getattr(cfg.uq, "distance", None)
    distance_enabled = (
        bool(getattr(distance_cfg, "enabled", False)) if distance_cfg is not None else False
    )
    distance_score_kinds: list[str] = []
    distance_layers_cfg: dict[str, list[str]] = {}
    distance_knn_k = 10
    distance_shrinkage = "ledoit_wolf"
    if distance_enabled:
        distance_score_kinds = [str(k) for k in getattr(distance_cfg, "scores", SCORE_KINDS)]
        layers_obj = getattr(distance_cfg, "layers", None)
        if layers_obj is not None:
            resolved_layers = OmegaConf.to_container(layers_obj, resolve=True)
            if isinstance(resolved_layers, dict):
                distance_layers_cfg = {
                    str(k): [str(p) for p in v] for k, v in resolved_layers.items()
                }
        distance_knn_k = int(getattr(distance_cfg, "knn_k", 10))
        distance_shrinkage = str(getattr(distance_cfg, "shrinkage", "ledoit_wolf"))

    _PROBE_DEPENDENT_METHODS = frozenset(
        {"uncalibrated", "temp_scaling", "bootstrap_ensemble", "deep_ensemble", "laplace", "conformal"}
    )
    requested_methods = set(cfg.uq.methods)
    # Distance scorers report the probe's predictions (they only reorder deferral),
    # so enabling them also requires the probe and its best_c lookup.
    needs_probe = bool(requested_methods & _PROBE_DEPENDENT_METHODS) or distance_enabled

    prior_results = None
    if needs_probe:
        prior_results = pd.read_csv(str(cfg.prior_results)) if os.path.exists(str(cfg.prior_results)) else None
        if prior_results is None:
            logger.warning("Prior results file missing at %s; skipping all datasets.", cfg.prior_results)
            return

    for dataset_name in dataset_names:
        try:
            ds_cls = get_bench_dataset_class(dataset_name)
        except KeyError:
            logger.warning("Skipping unknown dataset %s", dataset_name)
            continue
        if not _is_uq_classification_dataset(ds_cls):
            if ds_cls.task != "classification":
                logger.info("Skipping segmentation dataset %s in UQ classification pipeline.", dataset_name)
            else:
                logger.info("Skipping multi-label dataset %s in UQ pipeline.", dataset_name)
            continue

        try:
            loaded = get_datasets(
                dataset_name=dataset_name,
                partition_name=cfg.dataset.partition,
                batch_size=int(cfg.dataset.batch_size),
                num_workers=int(cfg.dataset.get("num_workers", 4)),
                return_val=True,
                image_size=getattr(cfg.dataset, "image_size", None),
                interpolation=getattr(cfg.dataset, "interpolation", "bilinear"),
                bands=getattr(cfg.dataset, "bands", "rgb"),
            )
        except (FileNotFoundError, DatasetNotFoundError) as exc:
            logger.warning("Skipping dataset %s (data missing: %s)", dataset_name, exc)
            continue
        if loaded is None:
            logger.warning("Skipping dataset %s (loader returned None)", dataset_name)
            continue

        train_dataset, train_loader, val_loader, test_loader = loaded
        bench = ds_cls()
        bands_resolved = (
            tuple(bench.rgb_bands)
            if cfg.dataset.bands == "rgb"
            else None
            if cfg.dataset.bands in ("all", None)
            else tuple(cfg.dataset.bands)
        )
        band_specs = bench.select_band_specs(bands_resolved)

        is_rcf_empirical = (
            hasattr(cfg.model, "mode")
            and str(cfg.model._target_).endswith("RCFBench")
            and str(cfg.model.mode) == "empirical"
        )
        instantiate_kwargs: dict[str, Any] = {
            "bands": band_specs,
            "normalization": normalization,
            "_convert_": "object",
        }
        if is_rcf_empirical:
            instantiate_kwargs["dataset"] = train_dataset
        # Re-seed immediately before construction. This loop runs once per
        # dataset, so without it the N-th dataset's model would draw from a
        # different RNG state: with `pretrained=false` the backbone weights
        # themselves would then depend on dataset ordering and on which cells
        # resume skipped. A no-op for pretrained backbones.
        torch.manual_seed(int(cfg.seed))
        model: BenchModel = instantiate(cfg.model, **instantiate_kwargs)
        model.to(device).eval()

        # Activation-distance layer paths for this model, if the scorers are on.
        # Iterate the configured list — never assume four layers: rcf_empirical
        # has exactly one hook path.
        hook_paths: list[str] = []
        if distance_enabled:
            hook_paths = list(distance_layers_cfg.get(str(cfg.model.name), []))
            if not hook_paths:
                logger.warning(
                    "uq.distance.enabled=true but no uq.distance.layers entry for model %s; "
                    "distance scorers are disabled for this run.",
                    cfg.model.name,
                )

        if hook_paths:
            # One extra hooked pass over the train split per (model, dataset)
            # cell: the reference must be fit on clean TRAIN activations.
            with HookCollector(model, hook_paths) as collector:
                X_train, y_train = extract_features(
                    model, train_loader, device, transforms=None, verbose=cfg.verbose
                )
                acts_train = collector.collect()
            validate_collected_activations(
                acts_train, hook_paths, int(len(y_train)), "clean-train"
            )
        else:
            acts_train = {}
            X_train, y_train = extract_features(
                model, train_loader, device, transforms=None, verbose=cfg.verbose
            )
        X_val, y_val = extract_features(model, val_loader, device, transforms=None, verbose=cfg.verbose)

        X_cal: np.ndarray | None = None
        y_cal: np.ndarray | None = None
        probe = None
        best_c = None

        if needs_probe:
            cal_size = int(cfg.uq.cal_size)
            if cal_size >= len(X_val):
                logger.warning(
                    "Skipping dataset %s: uq.cal_size=%d >= val size=%d.",
                    dataset_name,
                    cal_size,
                    len(X_val),
                )
                continue

            X_cal, y_cal, X_val_rem, y_val_rem = stratified_cal_split(X_val, y_val, cal_size, cfg.seed)
            X_final_train = np.concatenate([X_train, X_val_rem], axis=0)
            y_final_train = np.concatenate([y_train, y_val_rem], axis=0)

            alias_dataset = getattr(ds_cls, "prior_results_alias", None)
            best_c = _lookup_best_c(
                prior_results,
                {
                    "model": cfg.model._target_,
                    "name": cfg.model.name,
                    "dataset": dataset_name,
                    "partition": cfg.dataset.partition,
                    "bands": bands_value,
                },
                alias_dataset=alias_dataset,
            )
            if best_c is None:
                logger.warning(
                    "Skipping dataset %s: no prior best_c found for model=%s name=%s partition=%s bands=%s.",
                    dataset_name,
                    cfg.model._target_,
                    cfg.model.name,
                    cfg.dataset.partition,
                    bands_value,
                )
                continue

            probe = LogisticRegression(
                C=best_c,
                max_iter=4000,
                tol=1e-6,
                random_state=cfg.seed,
                device=str(device),
            )
            probe.fit(torch.from_numpy(X_final_train), torch.from_numpy(y_final_train.astype(np.int64)))
        else:
            X_final_train = np.concatenate([X_train, X_val], axis=0)
            y_final_train = np.concatenate([y_train, y_val], axis=0)

        # Fit the activation references ONCE per (model, dataset) cell, on clean
        # train activations only. Decision 3: refitting or renormalizing per
        # condition would destroy comparability across severities, which the
        # pooled risk-coverage evaluation depends on. ActivationReference.fit
        # raises on a second call to make that guarantee enforced, not documented.
        references: dict[str, ActivationReference] = {}
        if hook_paths and acts_train:
            y_ref = remap_labels_to_probe_classes(y_train.astype(np.int64), probe)
            references = build_activation_references(
                acts_train=acts_train,
                y_train=y_ref,
                score_kinds=distance_score_kinds,
                knn_k=distance_knn_k,
                shrinkage=distance_shrinkage,
            )
            logger.info(
                "Fitted %d activation references for %s / %s over layers %s.",
                len(references),
                cfg.model.name,
                dataset_name,
                hook_paths,
            )
            # Free the train activations: the largest (resisc45, 24800 x 2048 x 4
            # layers) is ~813 MB and is only needed during reference fitting.
            acts_train = {}

        methods: dict[str, Any] = {}
        if "uncalibrated" in cfg.uq.methods:
            methods["uncalibrated"] = Uncalibrated(probe)
        if "temp_scaling" in cfg.uq.methods:
            ts = TemperatureScaling(probe)
            ts.fit(X_cal, y_cal)
            methods["temp_scaling"] = ts
        if "bootstrap_ensemble" in cfg.uq.methods:
            be = BootstrapEnsemble(n=int(cfg.uq.n_ensemble))
            be.fit(X_final_train, y_final_train, best_c=best_c, seed=cfg.seed)
            methods["bootstrap_ensemble"] = be
        if "deep_ensemble" in cfg.uq.methods:
            de = DeepEnsemble(n=int(cfg.uq.n_ensemble))
            de.fit(X_final_train, y_final_train, best_c=best_c, seed=cfg.seed)
            methods["deep_ensemble"] = de
        if "laplace" in cfg.uq.methods:
            try:
                la = LaplaceProbe(probe, batch_size=int(cfg.uq.laplace_batch_size), pred_batch_size=int(cfg.uq.laplace_pred_batch_size))
                la.fit(X_final_train, y_final_train)
                methods["laplace"] = la
            except ModuleNotFoundError as exc:
                logger.warning("Skipping laplace for dataset %s: %s", dataset_name, exc)
        if "conformal" in cfg.uq.methods:
            try:
                conf = ConformalPredictor(probe)
                conf.fit(X_cal, y_cal, alpha=float(cfg.uq.conformal_alpha))
                methods["conformal"] = conf
            except (ModuleNotFoundError, ValueError) as exc:
                logger.warning("Skipping conformal for dataset %s: %s", dataset_name, exc)
        if "svgp" in cfg.uq.methods:
            try:
                svgp_pca_dim = getattr(cfg.model, "svgp_pca_dim", None) or getattr(cfg.uq, "svgp_pca_dim", None)
                svgp_pca_dim = int(svgp_pca_dim) if svgp_pca_dim is not None else None
                svgp_zca = bool(getattr(cfg.uq, "svgp_zca", False))
                svgp_kernel = str(getattr(cfg.uq, "svgp_kernel", "matern"))
                svgp_inducing_init = str(getattr(cfg.uq, "svgp_inducing_init", "kmeans"))
                svgp_mixing_weights = bool(getattr(cfg.uq, "svgp_mixing_weights", False))
                svgp = SVGPProbe(
                    n_inducing=int(cfg.uq.svgp_n_inducing),
                    epochs=int(cfg.uq.svgp_epochs),
                    lr=float(cfg.uq.svgp_lr),
                    batch_size=int(cfg.uq.svgp_batch_size),
                    n_mc_samples=int(cfg.uq.svgp_n_mc_samples),
                    device=device,
                    pca_dim=svgp_pca_dim,
                    zca=svgp_zca,
                    kernel=svgp_kernel,
                    inducing_init=svgp_inducing_init,
                    mixing_weights=svgp_mixing_weights,
                )
                if svgp_pca_dim is None and not svgp_zca and svgp_kernel != "linear":
                    svgp_scaler = StandardScaler()
                    X_svgp_train = svgp_scaler.fit_transform(X_final_train)
                    svgp.fit(X_svgp_train, y_final_train)
                    methods["svgp"] = _ScaledProbe(svgp, svgp_scaler)
                else:
                    svgp.fit(X_final_train, y_final_train)
                    methods["svgp"] = svgp
            except ModuleNotFoundError as exc:
                logger.warning("Skipping svgp for dataset %s: %s", dataset_name, exc)

        for nf_method in ("nf_empirical", "nf_uniform"):
            if nf_method not in cfg.uq.methods:
                continue
            nf_prior_path = str(getattr(cfg.uq, "nf_prior_results", "results/nf_results.csv"))
            nf_preprocessing = str(getattr(cfg.uq, "nf_preprocessing", "l2"))
            nf_prior_df = pd.read_csv(nf_prior_path) if os.path.exists(nf_prior_path) else None
            hp = _lookup_nf_hyperparams(
                nf_prior_df,
                model=str(cfg.model._target_),
                name=str(cfg.model.name),
                dataset=dataset_name,
                partition=str(cfg.dataset.partition),
                bands=bands_value,
                preprocessing=nf_preprocessing,
            )
            if hp is None:
                logger.warning(
                    "No NF hyperparams for %s / %s — run nf_pipeline first. Skipping %s.",
                    str(cfg.model.name), dataset_name, nf_method,
                )
                continue
            best_lr, best_wd = hp
            nf_prior = "empirical" if nf_method == "nf_empirical" else "uniform"
            try:
                from torchgeo_bench.uq.nf import NormalizingFlowProbe
                from torchgeo_bench.nf_pipeline import _preprocess

                nf_probe = NormalizingFlowProbe(
                    prior=nf_prior,
                    lr=best_lr,
                    weight_decay=best_wd,
                    epochs=int(getattr(cfg.uq, "nf_epochs", 100)),
                    batch_size=int(getattr(cfg.uq, "nf_batch_size", 512)),
                    device=device,
                )
                X_nf_train, _, _ = _preprocess(
                    X_final_train, X_final_train[:1], X_final_train[:1], nf_preprocessing
                )
                nf_probe.fit(X_nf_train, y_final_train)
                if nf_preprocessing == "zca_l2":
                    mean = X_final_train.mean(axis=0)
                    cov  = np.cov(X_final_train, rowvar=False)
                    vals, vecs = np.linalg.eigh(cov)
                    W = vecs @ np.diag(1.0 / np.sqrt(vals + 1e-3)) @ vecs.T
                    methods[nf_method] = _ZCAProbe(nf_probe, mean, W)
                elif nf_preprocessing == "standard":
                    from sklearn.preprocessing import StandardScaler
                    nf_scaler = StandardScaler()
                    nf_scaler.fit(X_final_train)
                    methods[nf_method] = _ScaledProbe(nf_probe, nf_scaler)
                else:
                    methods[nf_method] = _L2NormProbe(nf_probe)
            except ModuleNotFoundError as exc:
                logger.warning("Skipping %s for dataset %s: %s", nf_method, dataset_name, exc)

        _svgp_pca_dim = getattr(cfg.model, "svgp_pca_dim", None) or getattr(cfg.uq, "svgp_pca_dim", None)
        common_meta = {
            "model": str(cfg.model._target_),
            "name": str(cfg.model.name),
            "init": resolve_init(cfg.model),
            "backbone": str(cfg.model.name),
            "dataset": dataset_name,
            "normalization": normalization,
            "image_size": getattr(cfg.dataset, "image_size", None),
            "interpolation": getattr(cfg.dataset, "interpolation", "bilinear"),
            "partition": str(cfg.dataset.partition),
            "bands": bands_value,
            "seed": int(cfg.seed),
            "svgp_n_inducing": int(cfg.uq.svgp_n_inducing),
            "svgp_pca_dim": int(_svgp_pca_dim) if _svgp_pca_dim is not None else None,
            "svgp_zca": bool(getattr(cfg.uq, "svgp_zca", False)),
            "svgp_kernel": str(getattr(cfg.uq, "svgp_kernel", "linear")),
            "svgp_epochs": int(cfg.uq.svgp_epochs),
            "svgp_lr": float(cfg.uq.svgp_lr),
            "svgp_inducing_init": str(getattr(cfg.uq, "svgp_inducing_init", "kmeans")),
            "svgp_mixing_weights": bool(getattr(cfg.uq, "svgp_mixing_weights", False)),
        }

        # One collector instance for the whole (model, dataset) cell, reused
        # across all conditions: HookCollector.collect() clears its buffers, so
        # re-registering hooks per condition would be pure overhead. The
        # try/finally guarantees the hooks come off even if a condition raises.
        test_collector = HookCollector(model, hook_paths) if references else None
        try:
            _run_conditions(
                cfg=cfg,
                dataset_name=dataset_name,
                model=model,
                test_loader=test_loader,
                device=device,
                band_specs=band_specs,
                cloud_pattern_mode=cloud_pattern_mode,
                normalization=normalization,
                bands_value=bands_value,
                methods=methods,
                references=references,
                probe_predictor=Uncalibrated(probe) if references else None,
                test_collector=test_collector,
                hook_paths=hook_paths,
                common_meta=common_meta,
                completed=completed,
                output_path=output_path,
                trace_ctx=trace_ctx,
                X_cal=X_cal,
                X_final_train=X_final_train,
                best_c=best_c,
            )
        finally:
            if test_collector is not None:
                test_collector.remove()

    logger.info("UQ benchmark complete. Results appended to %s", output_path)


if __name__ == "__main__":  # pragma: no cover
    main()  # type: ignore[misc]
