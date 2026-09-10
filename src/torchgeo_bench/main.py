"""Benchmark script for torchgeo-bench."""

import logging
import math
import os
from collections.abc import Iterator, Sequence, Sized
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, NotRequired, TypedDict, cast

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from torchgeo_bench.calibration import (
    apply_temperature,
    compute_calibration_metrics,
    fit_temperature,
)
from torchgeo_bench.config import instantiate
from torchgeo_bench.datasets import (
    BenchDataset,
    get_bench_dataset_class,
    get_datasets,
    list_datasets,
)
from torchgeo_bench.intrinsic_dim import (
    FEATURE_SPECTRUM_METRICS,
    DegenerateManifoldError,
    DegenerateSpectrumError,
    compute_feature_spectrum,
    compute_intrinsic_dim,
)
from torchgeo_bench.knn import KNNClassifier, resolve_knn_device
from torchgeo_bench.linear import LogisticRegression
from torchgeo_bench.model_profile import ProfileTiming, measure_cpu_throughput, measure_profile
from torchgeo_bench.models.interface import BenchModel
from torchgeo_bench.results import (
    DEFAULT_INTRINSIC_DIM_RESULTS_DIR,
    DEFAULT_PROFILE_RESULTS_DIR,
    DEFAULT_RESULTS_DIR,
    EvaluationResult,
    append_rows_atomic,
    bootstrap_accuracy,
    bootstrap_map,
    bootstrap_miou,
    model_results_path,
)
from torchgeo_bench.resume import (  # noqa: F401  (re-exported for back-compat)
    KEY_COLS,
    DatasetRunPlan,
    ResumeState,
    _canonical_key_cell,
    _completed_run_keys,
    _filter_completed_metric_rows,
    _normalize_bands_value,
    _plan_dataset_run,
    _profile_metric_names,
    _resume_config_hash,
    _row_key,
    load_completed,
)
from torchgeo_bench.utils import FeatureSplit, FeatureSplits, extract_features, resolve_device

if TYPE_CHECKING:
    import torchgeo_bench.segmentation_task

logger = logging.getLogger(__name__)


class ResultMetadata(TypedDict):
    """Dataset and model fields shared by every result in a run."""

    dataset: str
    seed: int
    model: str
    name: str
    normalization: str
    image_size: int | None
    interpolation: str
    partition: str
    bands: str
    num_classes: int
    config_hash: str
    c_range_start: float
    c_range_stop: float
    c_range_num: int
    merge_val: bool
    bootstrap: int
    res: float | None
    pool: str | None
    feature_dim: NotRequired[int]
    n_train: NotRequired[int]
    n_val: NotRequired[int]
    n_test: NotRequired[int]


@dataclass
class LoaderSplits:
    """Data loaders for training, validation, and testing."""

    train: DataLoader
    val: DataLoader
    test: DataLoader


def resolve_model_config(model_cfg: DictConfig, dataset_name: str) -> DictConfig:
    """Apply a dataset-specific partial override to a model configuration."""
    resolved = DictConfig(OmegaConf.to_container(model_cfg, resolve=True))
    dataset_overrides = resolved.pop("dataset_overrides", {})
    resolved.merge_with(dataset_overrides.get(dataset_name, {}))
    return resolved


def _expand_dataset_list(names: str | Sequence[str]) -> list[str]:
    """Expand dataset names to a flat list.

    Args:
        names: Dataset name(s) — ``"all"``, comma-separated string, or sequence.

    Returns:
        List of individual dataset name strings.
    """
    if isinstance(names, str):
        if names == "all":
            return list_datasets()
        return [n.strip() for n in names.split(",") if n.strip()]
    return list(names)


def embed_split(
    model: BenchModel,
    dataloader: DataLoader,
    device: torch.device,
    *,
    verbose: bool,
    split: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature embeddings and labels from a data split."""
    description = f"Extracting ({split})" if split else "Extracting"
    return extract_features(
        model, dataloader, device, transforms=None, description=description if verbose else None
    )


def evaluate_knn(
    train: FeatureSplit[np.ndarray],
    test: FeatureSplit[np.ndarray],
    cfg: DictConfig,
    device: str,
    n_neighbors: int = 5,
) -> tuple[float, float, float, dict[str, float], int]:
    """Evaluate KNN, using the label shape to select single-label or multilabel scoring.

    Returns:
        Primary metric and bootstrap bounds, calibration (ECE/RMS-CE/MCE), and bin count.
        The default bin count is ``n_neighbors + 1``.
    """
    x_train, y_train = train.features, train.labels
    x_test, y_test = test.features, test.labels
    seed, n_bootstrap, verbose = cfg.seed, cfg.eval.bootstrap, cfg.verbose
    calibration_n_bins = (cfg.eval.get("calibration") or {}).get("n_bins_knn")
    n_bins = calibration_n_bins if calibration_n_bins is not None else n_neighbors + 1
    multi_label = y_train.ndim == 2
    clf = KNNClassifier(n_neighbors=n_neighbors, device=device, use_fp16=False)
    clf.fit(x_train, y_train)

    if multi_label:
        if verbose:
            logger.info("[KNN] Fit KNN5 multilabel (train=%s, test=%s)", len(x_train), len(x_test))
        y_scores = clf.predict_proba(x_test)
        metric, lo, hi = bootstrap_map(y_test, y_scores, n_boot=n_bootstrap, seed=seed)
        if verbose:
            logger.info("[KNN] Test micro_mAP=%.4f (CI %.4f-%.4f)", metric, lo, hi)
    else:
        if verbose:
            logger.info(
                "[KNN] Fit KNN5 (train=%s, test=%s, boot=%s)",
                len(x_train),
                len(x_test),
                n_bootstrap,
            )
        preds = clf.predict(x_test)
        y_scores = clf.predict_proba(x_test)
        metric, lo, hi = bootstrap_accuracy(y_test, preds, n_boot=n_bootstrap, seed=seed)
        if verbose:
            logger.info("[KNN] Test accuracy=%.4f (CI %.4f-%.4f)", metric, lo, hi)

    calibration = compute_calibration_metrics(
        y_test, y_scores, multi_label=multi_label, n_bins=n_bins
    )

    if verbose:
        logger.info(
            "[KNN] Calibration (n_bins=%s) ECE=%.4f RMS-CE=%.4f MCE=%.4f",
            n_bins,
            calibration["ece"],
            calibration["rms_ce"],
            calibration["mce"],
        )

    return metric, lo, hi, calibration, n_bins


class LinearProbeDivergedError(RuntimeError):
    """Raised when no C in the sweep produces a finite validation score."""


def select_logistic_c(
    train: FeatureSplit[torch.Tensor],
    val: FeatureSplit[np.ndarray],
    c_values: Sequence[float],
    cfg: DictConfig,
) -> float:
    """Choose C by validation accuracy or micro average precision."""
    x_train, y_train = train.features, train.labels
    x_val, y_val = torch.from_numpy(val.features), val.labels
    seed, device, verbose = cfg.seed, cfg.device, cfg.verbose
    from sklearn.metrics import accuracy_score, average_precision_score

    multi_label = y_train.ndim == 2
    label_tag = "LogReg-ML" if multi_label else "LogReg"
    best_c: float | None = None
    best_val_score = -1.0
    if verbose:
        logger.info(
            "[%s] C sweep start over %s values (train=%s, val=%s)",
            label_tag,
            len(c_values),
            len(x_train),
            len(x_val),
        )

    for idx, c in enumerate(tqdm(c_values, desc="C values", disable=not verbose)):
        model = LogisticRegression(
            C=c,
            max_iter=2000,
            tol=1e-6,
            random_state=seed,
            device=device,
            multi_label=multi_label,
        )
        model.fit(x_train, y_train)

        if multi_label:
            val_scores = model.predict_proba(x_val)
            if not np.all(np.isfinite(val_scores)):
                # Reject non-finite scores before average_precision_score raises.
                val_metric = float("-inf")
            else:
                val_metric = float(average_precision_score(y_val, val_scores, average="micro"))
        else:
            val_pred = model.predict(x_val)
            val_metric = accuracy_score(y_val, val_pred)

        if verbose and (idx < 10 or idx % 50 == 0):
            logger.info("[%s] C=%.4g val_score=%.4f", label_tag, c, val_metric)
        if val_metric > best_val_score:
            best_val_score = val_metric
            best_c = c

    if best_c is None:
        raise LinearProbeDivergedError(
            f"Every candidate C in {list(c_values)} produced a non-finite val score; "
            "features are unusable for a linear probe at any regularization strength tried."
        )
    if verbose:
        logger.info("[%s] Best C=%.4g val_score=%.4f", label_tag, best_c, best_val_score)

    return best_c


def calibrate_logistic(
    model: LogisticRegression,
    val: FeatureSplit[np.ndarray],
    test: FeatureSplit[np.ndarray],
    n_bins: int,
) -> dict[str, float | None]:
    """Fit temperature on validation logits and calibrate test probabilities."""
    x_val, y_val = torch.from_numpy(val.features), val.labels
    x_test, y_test = torch.from_numpy(test.features), test.labels
    multi_label = y_val.ndim == 2
    class_labels = None if multi_label else model.classes_
    val_logits = model.decision_function(x_val)
    test_logits = model.decision_function(x_test)
    temperature = fit_temperature(
        val_logits, y_val, multi_label=multi_label, class_labels=class_labels
    )
    test_scores_ts = apply_temperature(test_logits, temperature, multi_label=multi_label)
    cal_ts = compute_calibration_metrics(
        y_test,
        test_scores_ts,
        multi_label=multi_label,
        n_bins=n_bins,
        class_labels=class_labels,
    )
    return {
        "ece_ts": cal_ts["ece"],
        "rms_ce_ts": cal_ts["rms_ce"],
        "mce_ts": cal_ts["mce"],
        "temperature": temperature,
    }


def evaluate_logistic(
    splits: FeatureSplits[np.ndarray],
    c_values: Sequence[float],
    cfg: DictConfig,
) -> tuple[float, float, float, float, dict[str, float], dict[str, float | None]]:
    """Select C on validation data, refit, and score the test split.

    The label shape selects single-label or multilabel scoring.

    Returns:
        Primary metric and bootstrap bounds, selected ``C``, and raw and scaled calibration.
        Scaled calibration includes the fitted temperature.
        Its values are ``None`` when scaling is disabled or ``merge_val`` is true.
    """
    x_train, y_train = splits.train.features, splits.train.labels
    x_val, y_val = splits.val.features, splits.val.labels
    x_test, y_test = splits.test.features, splits.test.labels
    seed, device, verbose = cfg.seed, cfg.device, cfg.verbose
    n_bootstrap, merge_val = cfg.eval.bootstrap, cfg.eval.merge_val
    calibration_cfg = cfg.eval.get("calibration") or {}
    calibration_n_bins = int(calibration_cfg.get("n_bins_linear", 15))
    temp_scale = bool(calibration_cfg.get("temp_scale", True))
    multi_label = y_train.ndim == 2
    x_train_tensor = torch.from_numpy(x_train)
    x_test_tensor = torch.from_numpy(x_test)

    if multi_label:
        y_train_tensor = torch.from_numpy(y_train).float()
        label_tag = "LogReg-ML"
    else:
        y_train_tensor = torch.from_numpy(y_train).long()
        label_tag = "LogReg"

    best_c = select_logistic_c(
        FeatureSplit(x_train_tensor, y_train_tensor), splits.val, c_values, cfg
    )

    if merge_val:
        x_final_np = np.concatenate([x_train, x_val], axis=0)
        y_final_np = np.concatenate([y_train, y_val], axis=0)
        x_final = torch.from_numpy(x_final_np)
        y_final = (
            torch.from_numpy(y_final_np).float()
            if multi_label
            else torch.from_numpy(y_final_np).long()
        )
    else:
        x_final = x_train_tensor
        y_final = y_train_tensor

    final_model = LogisticRegression(
        C=best_c,
        max_iter=4000,
        tol=1e-6,
        random_state=seed,
        device=device,
        multi_label=multi_label,
    )
    final_model.fit(x_final, y_final)

    if multi_label:
        test_scores = final_model.predict_proba(x_test_tensor)
        metric, lo, hi = bootstrap_map(y_test, test_scores, n_boot=n_bootstrap, seed=seed)
    else:
        test_preds = final_model.predict(x_test_tensor)
        test_scores = final_model.predict_proba(x_test_tensor)
        metric, lo, hi = bootstrap_accuracy(y_test, test_preds, n_boot=n_bootstrap, seed=seed)

    class_labels = None if multi_label else final_model.classes_
    calibration = compute_calibration_metrics(
        y_test,
        test_scores,
        multi_label=multi_label,
        n_bins=calibration_n_bins,
        class_labels=class_labels,
    )

    calibration_ts: dict[str, float | None] = {
        "ece_ts": None,
        "rms_ce_ts": None,
        "mce_ts": None,
        "temperature": None,
    }
    if temp_scale and not merge_val:
        calibration_ts = calibrate_logistic(
            final_model, splits.val, splits.test, calibration_n_bins
        )
    elif temp_scale and merge_val:
        logger.warning(
            "[%s] Skipping temperature scaling because merge_val=true leaves no held-out "
            "calibration split.",
            label_tag,
        )

    if verbose:
        logger.info(
            "[%s] Test score=%.4f (CI %.4f-%.4f) using C=%.4g; train_final=%s test=%s",
            label_tag,
            metric,
            lo,
            hi,
            best_c,
            len(x_final),
            len(x_test),
        )
        logger.info(
            "[%s] Calibration (n_bins=%s) ECE=%.4f RMS-CE=%.4f MCE=%.4f",
            label_tag,
            calibration_n_bins,
            calibration["ece"],
            calibration["rms_ce"],
            calibration["mce"],
        )
        if calibration_ts["temperature"] is not None:
            logger.info(
                "[%s] Post-TS T=%.3f ECE=%.4f RMS-CE=%.4f MCE=%.4f",
                label_tag,
                calibration_ts["temperature"],
                calibration_ts["ece_ts"],
                calibration_ts["rms_ce_ts"],
                calibration_ts["mce_ts"],
            )
    return metric, lo, hi, float(best_c), calibration, calibration_ts


def _resolve_segmentation_runtime_config(
    seg_cfg: DictConfig,
) -> tuple[int, int, float, bool, torch.dtype]:
    """Validate segmentation settings before feature extraction or training."""
    epochs = seg_cfg.get("epochs", 10)
    batch_size = seg_cfg.get("batch_size", 64)
    lr = seg_cfg.get("lr", 1e-3)
    use_cache = seg_cfg.get("cache_features", True)
    cache_dtype_name = seg_cfg.get("cache_dtype", "float16")

    for name, value in (("epochs", epochs), ("batch_size", batch_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"eval.segmentation.{name} must be a positive integer, got {value!r}.")
    if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
        raise ValueError(f"eval.segmentation.lr must be a finite positive number, got {lr!r}.")
    if not isinstance(use_cache, bool):
        raise TypeError(f"eval.segmentation.cache_features must be a boolean, got {use_cache!r}.")
    cache_dtypes = {"float16": torch.float16, "float32": torch.float32}
    if cache_dtype_name not in cache_dtypes:
        raise ValueError(
            "eval.segmentation.cache_dtype must be one of "
            f"{tuple(cache_dtypes)}, got {cache_dtype_name!r}."
        )
    return epochs, batch_size, float(lr), use_cache, cache_dtypes[cache_dtype_name]


def estimate_intrinsic_dimensions(
    X: np.ndarray,
    split_name: str,
    cfg: DictConfig,
    common_meta: ResultMetadata,
    only_metrics: frozenset[str] | None,
) -> dict[str, float]:
    """Compute each estimator independently so one degenerate estimate keeps the others."""
    id_cfg = cfg.eval.intrinsic_dim
    estimators = list(id_cfg.estimators)
    device = id_cfg.get("device") or cfg.device
    max_samples = id_cfg.get("max_samples")
    seed = cfg.seed
    dims: dict[str, float] = {}
    for est_name in estimators:
        if only_metrics is not None and f"id_{est_name}_{split_name}" not in only_metrics:
            continue
        try:
            dims.update(
                compute_intrinsic_dim(
                    X,
                    estimators=[est_name],
                    device=device,
                    max_samples=max_samples,
                    seed=seed,
                )
            )
        except DegenerateManifoldError as exc:  # allow-except: Degenerate features produce NaN.
            logger.warning(
                "[intrinsic-dim] %s split=%s model=%s dataset=%s bands=%s norm=%s: degenerate features, writing NaN. Diagnostic: %s",
                est_name,
                split_name,
                common_meta.get("model"),
                common_meta.get("dataset"),
                common_meta.get("bands"),
                common_meta.get("normalization"),
                exc,
            )
            dims[est_name] = float("nan")
    return dims


def evaluate_intrinsic_dim(
    splits: dict[str, np.ndarray],
    cfg: DictConfig,
    common_meta: ResultMetadata,
    only_metrics: frozenset[str] | None = None,
) -> list[dict]:
    """Return intrinsic-dimension and centered feature-spectrum results for selected splits.

    Each estimator produces an ``id_<estimator>_<split>`` row.
    The five spectrum diagnostics produce ``spectrum_<metric>_<split>`` rows.
    All rows use ``method="intrinsic_dim"``.

    ``only_metrics`` selects unfinished metrics on resume. ``None`` computes everything.
    """
    id_cfg = cfg.eval.intrinsic_dim
    estimators, selected_splits = list(id_cfg.estimators), list(id_cfg.splits)
    device = id_cfg.get("device") or cfg.device
    max_samples, seed, verbose = id_cfg.get("max_samples"), cfg.seed, cfg.verbose
    rows: list[dict] = []
    for split_name in selected_splits:
        if split_name not in splits:
            logger.warning("[intrinsic-dim] unknown split '%s', skipping", split_name)
            continue
        X = splits[split_name]
        if verbose:
            logger.info(
                "[intrinsic-dim] split=%s X%s estimators=%s device=%s",
                split_name,
                X.shape,
                list(estimators),
                device,
            )
        dims = estimate_intrinsic_dimensions(X, split_name, cfg, common_meta, only_metrics)
        for est_name, dim in dims.items():
            rows.append(
                EvaluationResult(
                    **common_meta,
                    method="intrinsic_dim",
                    metric_name=f"id_{est_name}_{split_name}",
                    metric_value=float(dim),
                ).to_row()
            )

        spectrum_names = {f"spectrum_{metric}_{split_name}" for metric in FEATURE_SPECTRUM_METRICS}
        if only_metrics is not None and only_metrics.isdisjoint(spectrum_names):
            continue
        try:
            spectrum = compute_feature_spectrum(X, max_samples=max_samples, seed=seed)
        except DegenerateSpectrumError as exc:  # allow-except: Degenerate features produce NaN.
            logger.warning(
                "[intrinsic-dim] spectrum split=%s model=%s dataset=%s bands=%s norm=%s: degenerate features, writing NaN. Diagnostic: %s",
                split_name,
                common_meta.get("model"),
                common_meta.get("dataset"),
                common_meta.get("bands"),
                common_meta.get("normalization"),
                exc,
            )
            spectrum = {metric: float("nan") for metric in FEATURE_SPECTRUM_METRICS}
        for metric_name, value in spectrum.items():
            if (
                only_metrics is not None
                and f"spectrum_{metric_name}_{split_name}" not in only_metrics
            ):
                continue
            rows.append(
                EvaluationResult(
                    **common_meta,
                    method="intrinsic_dim",
                    metric_name=f"spectrum_{metric_name}_{split_name}",
                    metric_value=value,
                ).to_row()
            )
    return rows


def evaluate_profile(
    model: BenchModel,
    sample_loader: DataLoader,
    cfg: DictConfig,
    common_meta: ResultMetadata,
) -> list[dict]:
    """Measure backbone throughput, memory, and parameter count as CSV rows.

    One row per metric, with ``method="profile"``.

    ``cfg.eval.profile.cpu_throughput.enabled`` adds CPU metrics with a ``_cpu`` suffix.
    Its ``time_budget_s`` setting bounds the extra measurement.
    """
    device = torch.device(cfg.device)
    profile_cfg = cfg.eval.profile
    n_warmup, n_measure = int(profile_cfg.get("n_warmup", 3)), int(profile_cfg.get("n_measure", 20))
    cpu_cfg = profile_cfg.get("cpu_throughput") or {}
    sample = next(iter(sample_loader))["image"].to(device)

    metrics = measure_profile(model, sample, device, n_warmup=n_warmup, n_measure=n_measure)

    if cpu_cfg.get("enabled", False):
        metrics.update(
            measure_cpu_throughput(
                model,
                sample,
                timing=ProfileTiming(
                    batch_size=int(cpu_cfg.get("batch_size", 8)),
                    n_warmup=int(cpu_cfg.get("n_warmup", 1)),
                    n_measure=int(cpu_cfg.get("n_measure", 5)),
                ),
                time_budget_s=float(cpu_cfg.get("time_budget_s", 300.0)),
            )
        )

    rows: list[dict] = []
    for name, value in metrics.items():
        if value is None:
            # None means the measurement is unavailable or exceeded its time budget.
            continue
        rows.append(
            EvaluationResult(
                **common_meta, method="profile", metric_name=name, metric_value=float(value)
            ).to_row()
        )
    return rows


def evaluate_segmentation(
    model: torch.nn.Module,
    loaders: LoaderSplits,
    eval_cfg: DictConfig,
    cfg: DictConfig,
    num_classes: int,
) -> "tuple[torchgeo_bench.segmentation_task.SegMetrics, int, float | None, int | None]":
    """Train a segmentation head on a frozen backbone and evaluate test mIoU.

    Backbone features can be cached to avoid recomputing them each epoch.

    Returns:
        Tuple of (metrics, feature_dim, lr, batch_size).
    """
    train_loader, val_loader, test_loader = loaders.train, loaders.val, loaders.test
    device, seed, verbose = torch.device(cfg.device), cfg.seed, cfg.verbose
    from torchgeo_bench.segmentation_task import build_seg_probe_and_solver

    if "segmentation" not in eval_cfg:
        raise ValueError("Segmentation evaluation config missing for the model.")

    seg_cfg = eval_cfg.segmentation
    epochs, probe_batch_size, lr, use_cache, cache_dtype = _resolve_segmentation_runtime_config(
        seg_cfg
    )

    probe, solver = build_seg_probe_and_solver(model, num_classes, eval_cfg, device, lr)
    collect_confusions = int(eval_cfg.bootstrap) > 0
    if use_cache and probe.freeze_backbone:
        logger.info("Caching backbone features for train and val splits...")
        train_cache = probe.extract_segmentation_features(train_loader, cache_dtype=cache_dtype)
        val_cache = probe.extract_segmentation_features(val_loader, cache_dtype=cache_dtype)
        test_cache = probe.extract_segmentation_features(test_loader, cache_dtype=cache_dtype)
        solver.fit_cached(
            train_cache=train_cache,
            val_cache=val_cache,
            batch_size=probe_batch_size,
            epochs=epochs,
            verbose=verbose,
        )
        eval_result = solver.evaluate_cached(
            test_cache,
            batch_size=probe_batch_size,
            collect_confusions=collect_confusions,
        )
    else:
        solver.fit(train_loader=train_loader, val_loader=val_loader, epochs=epochs, verbose=verbose)
        eval_result = solver.evaluate(
            test_loader,
            collect_confusions=collect_confusions,
        )
    actual_batch_size = (
        probe_batch_size
        if use_cache and probe.freeze_backbone
        else int(train_loader.batch_size or 1)
    )

    if isinstance(eval_result, tuple):
        metrics, confusion_matrices = eval_result
        metrics["ci_lower"], metrics["ci_upper"] = bootstrap_miou(
            confusion_matrices,
            n_boot=int(eval_cfg.bootstrap),
            seed=seed,
        )
    else:
        metrics = eval_result
    return metrics, sum(probe.channels_list), lr, actual_batch_size


def _resolve_output_path(
    cfg: DictConfig, dir_key: str = "results_dir", default_dir: str = DEFAULT_RESULTS_DIR
) -> str:
    """Return explicit ``output``, else the model's CSV in the requested directory.

    An explicit ``output=`` routes metrics, profile, and intrinsic-dimension
    rows to one file; otherwise each kind has its own per-model directory.
    """
    if cfg.get("output"):
        return str(cfg.output)
    name = cfg.model.get("name")
    if not name:
        raise ValueError(
            "model config has no 'name', so no per-model results file can be "
            "derived; set output= explicitly or add a name to the model config."
        )
    return str(model_results_path(cfg.get(dir_key, default_dir), name))


def run_segmentation(
    cfg: DictConfig,
    eval_cfg: DictConfig,
    model: BenchModel,
    loaders: LoaderSplits,
    common_meta: ResultMetadata,
) -> Iterator[list[dict]]:
    """Yield the completed segmentation probe measurement."""
    train_loader, val_loader, test_loader = loaders.train, loaders.val, loaders.test
    train_dataset = train_loader.dataset
    assert isinstance(train_dataset, Sized)
    num_classes = common_meta["num_classes"]
    assert isinstance(val_loader.dataset, Sized)
    assert isinstance(test_loader.dataset, Sized)
    metrics, feat_dim, best_lr, best_bs = evaluate_segmentation(
        model, loaders, eval_cfg, cfg, num_classes
    )

    segmentation_meta: ResultMetadata = {
        **common_meta,
        "merge_val": False,
        "feature_dim": feat_dim,
        "n_train": len(train_dataset),
        "n_val": len(val_loader.dataset),
        "n_test": len(test_loader.dataset),
    }
    row = EvaluationResult(
        **segmentation_meta,
        method=f"seg-{eval_cfg.segmentation.head_type}",
        metric_name="mIoU",
        metric_value=metrics.get("mIoU", float("nan")),
        ci_lower=metrics.get("ci_lower", float("nan")),
        ci_upper=metrics.get("ci_upper", float("nan")),
        best_lr=best_lr,
        best_batch_size=best_bs,
        fw_iou=metrics.get("fw_IoU"),
        precision=metrics.get("precision"),
        recall=metrics.get("recall"),
        f1=metrics.get("f1"),
    ).to_row()
    yield [row]


def run_classification(  # noqa: PLR0913 - keep the CLI failure policy explicit
    cfg: DictConfig,
    plan: DatasetRunPlan,
    model: BenchModel,
    loaders: LoaderSplits,
    common_meta: ResultMetadata,
    *,
    strict: bool = False,
) -> Iterator[tuple[list[dict], list[dict], list[dict]]]:
    """Yield each completed probe or feature measurement for immediate persistence."""
    train_loader, val_loader, test_loader = loaders.train, loaders.val, loaders.test
    model_eval = {} if strict else cfg.model.get("eval") or {}
    knn_k = int(model_eval["knn_k"]) if "knn_k" in model_eval else int(cfg.eval.get("knn_k", 5))
    c_start, c_stop, c_num = model_eval.get("c_range") or cfg.eval.c_range
    c_values_list = (10 ** np.linspace(float(c_start), float(c_stop), int(c_num))).tolist()
    device = torch.device(cfg.device)
    metric_name = plan.metric_name
    x_train, y_train = embed_split(model, train_loader, device, verbose=True, split="train")
    x_val, y_val = embed_split(model, val_loader, device, verbose=True, split="val")
    x_test, y_test = embed_split(model, test_loader, device, verbose=True, split="test")
    feature_dim = x_train.shape[1]
    n_counts = {"train": len(x_train), "val": len(x_val), "test": len(x_test)}

    splits = FeatureSplits(
        FeatureSplit(x_train, y_train), FeatureSplit(x_val, y_val), FeatureSplit(x_test, y_test)
    )
    common_meta = {
        **common_meta,
        "feature_dim": feature_dim,
        "n_train": n_counts["train"],
        "n_val": n_counts["val"],
        "n_test": n_counts["test"],
    }
    cal_n_bins_linear = int((cfg.eval.get("calibration") or {}).get("n_bins_linear", 15))

    if not plan.skip_knn:
        assert plan.knn_device is not None
        knn_score, knn_lo, knn_hi, knn_cal, knn_n_bins = evaluate_knn(
            splits.train, splits.test, cfg, device=plan.knn_device, n_neighbors=knn_k
        )
        row = EvaluationResult(
            **common_meta,
            method=f"knn{knn_k}",
            metric_name=metric_name,
            metric_value=knn_score,
            ci_lower=knn_lo,
            ci_upper=knn_hi,
            ece=knn_cal["ece"],
            rms_ce=knn_cal["rms_ce"],
            mce=knn_cal["mce"],
            calibration_n_bins=knn_n_bins,
        ).to_row()
        yield [row], [], []

    if not plan.skip_linear:
        try:
            lin_score, lin_lo, lin_hi, best_c, lin_cal, lin_cal_ts = evaluate_logistic(
                splits, c_values_list, cfg
            )
        except LinearProbeDivergedError as exc:  # allow-except: Other metrics remain usable.
            if strict:
                raise
            logger.warning(
                "[linear] model=%s dataset=%s bands=%s norm=%s: skipping, no usable C found. Diagnostic: %s",
                common_meta.get("model"),
                common_meta.get("dataset"),
                common_meta.get("bands"),
                common_meta.get("normalization"),
                exc,
            )
        else:
            row = EvaluationResult(
                **common_meta,
                method="linear",
                metric_name=metric_name,
                metric_value=lin_score,
                ci_lower=lin_lo,
                ci_upper=lin_hi,
                best_c=best_c,
                ece=lin_cal["ece"],
                rms_ce=lin_cal["rms_ce"],
                mce=lin_cal["mce"],
                ece_ts=lin_cal_ts["ece_ts"],
                rms_ce_ts=lin_cal_ts["rms_ce_ts"],
                mce_ts=lin_cal_ts["mce_ts"],
                temperature=lin_cal_ts["temperature"],
                calibration_n_bins=cal_n_bins_linear,
            ).to_row()
            yield [row], [], []
    if not plan.skip_id:
        id_rows = evaluate_intrinsic_dim(
            {"train": x_train, "val": x_val, "test": x_test},
            cfg,
            common_meta,
            only_metrics=plan.id_missing_metrics if cfg.resume else None,
        )
        yield [], id_rows, []

    if not plan.skip_profile:
        profile_rows = evaluate_profile(model, train_loader, cfg, common_meta)
        yield [], [], profile_rows


def instantiate_dataset_model(
    cfg: DictConfig,
    model_cfg: DictConfig,
    bench: BenchDataset,
    train_dataset: Dataset,
    device: torch.device,
) -> BenchModel:
    """Construct the model with bands matching the loaded tensor channels."""
    num_channels = train_dataset[0]["image"].shape[0]
    normalization = str(getattr(cfg.dataset, "normalization", "bandspec_zscore"))
    ds_name = bench.name
    bands_resolved = (
        tuple(bench.rgb_bands)
        if cfg.dataset.bands == "rgb"
        else None
        if cfg.dataset.bands in ("all", None)
        else tuple(cfg.dataset.bands)
    )
    bands_list = bench.select_band_specs(bands_resolved)
    if len(bands_list) != num_channels:
        raise ValueError(
            f"BandSpec count {len(bands_list)} != tensor channel count {num_channels} "
            f"for dataset {ds_name}; sample-level canonicalization may have changed shape."
        )

    # Pass BandSpecs outside OmegaConf to preserve the dataclass objects.
    instantiate_kwargs: dict = {
        "bands": bands_list,
        "normalization": normalization,
    }
    if model_cfg.get("mode", None) == "empirical":
        # Empirical RCF whitens against real patches, so it needs the dataset.
        instantiate_kwargs["dataset"] = train_dataset
    # Interpolation belongs to the loader rather than the model constructor.
    model_cfg.pop("interpolation", None)
    model: BenchModel = instantiate(model_cfg, **instantiate_kwargs)
    model.to(device).eval()

    return model


def dataset_metadata(
    cfg: DictConfig,
    ds_name: str,
    ds_cls: type[BenchDataset],
    model_cfg: DictConfig,
    config_hash: str,
) -> ResultMetadata:
    """Collect result metadata before loading data or initializing the model."""
    model_eval = cfg.model.get("eval") or {}
    c_start, c_stop, c_num = model_eval.get("c_range") or cfg.eval.c_range
    normalization = str(getattr(cfg.dataset, "normalization", "bandspec_zscore"))
    bands_value = _normalize_bands_value(getattr(cfg.dataset, "bands", "rgb"))
    effective_image_size = model_cfg.get("image_size", cfg.dataset.get("image_size"))
    effective_interpolation = model_cfg.get(
        "interpolation", cfg.dataset.get("interpolation", "bilinear")
    )
    return {
        "dataset": ds_name,
        "seed": cfg.seed,
        "model": model_cfg._target_,
        "name": model_cfg.name,
        "normalization": normalization,
        "image_size": effective_image_size,
        "interpolation": effective_interpolation,
        "partition": cfg.dataset.partition,
        "bands": bands_value,
        "num_classes": ds_cls.num_classes,
        "config_hash": config_hash,
        "c_range_start": c_start,
        "c_range_stop": c_stop,
        "c_range_num": c_num,
        "merge_val": cfg.eval.merge_val,
        "bootstrap": cfg.eval.bootstrap,
        "res": model_cfg.get("res"),
        "pool": model_cfg.get("pool"),
    }


def run_dataset(
    cfg: DictConfig,
    ds_name: str,
    config_hash: str,
    completed: ResumeState,
    *,
    strict: bool = False,
) -> Iterator[tuple[list[dict], list[dict], list[dict]]]:
    """Load and evaluate one dataset unless resume marks it complete."""
    ds_cls = get_bench_dataset_class(ds_name)

    model_cfg = resolve_model_config(cfg.model, ds_name)
    common_meta = dataset_metadata(cfg, ds_name, ds_cls, model_cfg, config_hash)
    if strict:
        c_start, c_stop, c_num = cfg.eval.c_range
        common_meta.update(
            c_range_start=float(c_start), c_range_stop=float(c_stop), c_range_num=int(c_num)
        )
    model_eval = cfg.model.get("eval", None) if "eval" in cfg.model else None
    eval_cfg = cast(
        DictConfig,
        OmegaConf.merge(OmegaConf.create(model_eval or {}), cfg.eval)
        if strict
        else OmegaConf.merge(cfg.eval, model_eval or {}),
    )
    plan = _plan_dataset_run(cfg, ds_cls, common_meta, completed, eval_cfg)
    if plan.skip_dataset:
        if cfg.verbose:
            logger.info("[%s] Resume preflight: all requested work already complete", ds_name)
        return

    if ds_cls.task != "segmentation" and not plan.skip_knn:
        plan = replace(plan, knn_device=resolve_knn_device(cfg.eval.get("knn_device"), cfg.device))
    train_dataset, train_loader, val_loader, test_loader = get_datasets(
        dataset_name=ds_name,
        partition_name=cfg.dataset.partition,
        batch_size=cfg.dataset.batch_size,
        num_workers=int(cfg.dataset.get("num_workers", 8)),
        return_val=True,
        image_size=model_cfg.get("image_size", cfg.dataset.get("image_size")),
        interpolation=model_cfg.get("interpolation", cfg.dataset.get("interpolation", "bilinear")),
        bands=getattr(cfg.dataset, "bands", "rgb"),
        time_steps=cfg.dataset.get("time_steps", None),
    )

    bench = ds_cls()
    model = instantiate_dataset_model(
        cfg, model_cfg, bench, train_dataset, torch.device(cfg.device)
    )
    loaders = LoaderSplits(train_loader, val_loader, test_loader)
    if ds_cls.task == "segmentation":
        for rows in run_segmentation(cfg, eval_cfg, model, loaders, common_meta):
            yield rows, [], []
        return
    for rows, id_rows, profile_rows in run_classification(
        cfg, plan, model, loaders, common_meta, strict=strict
    ):
        if cfg.resume:
            id_rows = _filter_completed_metric_rows(id_rows, completed.completed_metrics, KEY_COLS)
            profile_rows = _filter_completed_metric_rows(
                profile_rows, completed.completed_metrics, KEY_COLS
            )
        yield rows, id_rows, profile_rows


def load_completed_outputs(
    cfg: DictConfig,
    output_path: str,
    profile_output_path: str,
    intrinsic_dim_output_path: str,
) -> tuple[set[tuple[str, ...]], dict[str, set[tuple[str, ...]]]]:
    """Read each distinct output once, merging side files only into metric resume keys."""
    completed_runs: set[tuple[str, ...]] = set()
    completed_metrics: dict[str, set[tuple[str, ...]]] = {}
    if cfg.resume:
        for path in {output_path, profile_output_path, intrinsic_dim_output_path}:
            if not os.path.exists(path):
                continue
            runs, metrics = load_completed(path)
            if path == output_path:
                completed_runs.update(runs)
            for metric_name, keys in metrics.items():
                completed_metrics.setdefault(metric_name, set()).update(keys)
            logger.info("Resume mode: Found %d existing results in %s", len(runs), path)

    return completed_runs, completed_metrics


def main(cfg: DictConfig, *, strict: bool = False) -> None:
    """Run the benchmark pipeline for all configured datasets and models."""
    torch.manual_seed(cfg.seed)

    # Coordinate models use (lon, lat) inputs and ridge/KNN with k-fold validation.
    if str(cfg.get("mode", "image")) == "coord":
        from torchgeo_bench.coordbench.run import run_coordbench

        run_coordbench(cfg)
        return

    dataset_names = _expand_dataset_list(cfg.dataset.names)
    device = resolve_device(cfg.device)
    cfg.device = str(device)

    output_path = _resolve_output_path(cfg)
    profile_output_path = _resolve_output_path(
        cfg, "profile_results_dir", DEFAULT_PROFILE_RESULTS_DIR
    )
    intrinsic_dim_output_path = _resolve_output_path(
        cfg, "intrinsic_dim_results_dir", DEFAULT_INTRINSIC_DIM_RESULTS_DIR
    )
    output_paths = {output_path, profile_output_path, intrinsic_dim_output_path}
    for path in output_paths:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    completed_runs, completed_metrics = load_completed_outputs(
        cfg, output_path, profile_output_path, intrinsic_dim_output_path
    )
    config_hash = _resume_config_hash(cfg)
    completed = ResumeState(completed_runs, completed_metrics)
    for ds_name in tqdm(dataset_names, desc="Datasets"):
        for all_rows, id_out_rows, profile_out_rows in run_dataset(
            cfg, ds_name, config_hash, completed, strict=strict
        ):
            append_rows_atomic(output_path, all_rows)
            append_rows_atomic(intrinsic_dim_output_path, id_out_rows)
            append_rows_atomic(profile_output_path, profile_out_rows)

    logger.info("Benchmark complete. Results appended to %s", ", ".join(sorted(output_paths)))
