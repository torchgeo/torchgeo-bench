"""Benchmark script for torchgeo-bench."""

import fcntl
import io
import logging
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from rich.progress import track
from sklearn.metrics import accuracy_score, average_precision_score
from torch.utils.data import ConcatDataset, DataLoader
from torchgeo.datasets import DatasetNotFoundError

from torchgeo_bench.calibration import (
    apply_temperature,
    compute_calibration_metrics,
    fit_temperature,
)
from torchgeo_bench.datasets import (
    get_bench_dataset_class,
    get_datasets,
    list_datasets,
)
from torchgeo_bench.intrinsic_dim import DegenerateManifoldError, compute_intrinsic_dim
from torchgeo_bench.knn import KNNClassifier
from torchgeo_bench.linear import LogisticRegression
from torchgeo_bench.model_profile import measure_profile
from torchgeo_bench.models.interface import BenchModel
from torchgeo_bench.segmentation_probe import (
    SegmentationProbe,
)
from torchgeo_bench.segmentation_task import SegmentationSolver, SegMetrics
from torchgeo_bench.segmentation_viz import save_segmentation_viz
from torchgeo_bench.utils import extract_features

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


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


def _normalize_bands_value(bands: object) -> str:
    """Canonicalize the ``cfg.dataset.bands`` value for logging/CSV/resume.

    Hydra hands us either ``"rgb"``/``"all"``, an explicit list (``ListConfig``
    or ``list[str]``), or ``None``.  Reduce all of those to a stable string so
    that the resume key and the CSV column are comparable across runs.

    Args:
        bands: The raw ``cfg.dataset.bands`` value.

    Returns:
        A stable string representation: ``"rgb"``, ``"all"``, or a
        comma-joined explicit band list (e.g. ``"red,green,blue,nir"``).
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
    return set(map(tuple, df[list(key_cols)].fillna("").astype(str).to_numpy()))


def _row_key(row: dict, key_cols: Sequence[str]) -> tuple[str, ...]:
    """Build a normalized resume key tuple from a result row dict."""
    return tuple(str(row.get(col, "")) for col in key_cols)


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
    c_range_start: float
    c_range_stop: float
    c_range_num: int
    merge_val: bool
    bootstrap: int
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


def embed_split(
    model: BenchModel, dataloader: DataLoader, device: torch.device, verbose: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature embeddings and labels from a data split."""
    return extract_features(model, dataloader, device, transforms=None, verbose=verbose)


def evaluate_knn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    n_bootstrap: int,
    verbose: bool = False,
    device: str = "cpu",
    n_neighbors: int = 5,
    calibration_n_bins: int | None = None,
) -> tuple[float, float, float, dict[str, float], int]:
    """Evaluate KNN classifier. Auto-detects single-label vs multi-label from y shape.

    Returns the primary metric with bootstrap CI, a calibration dict
    (``ece``/``rms_ce``/``mce``) computed from ``predict_proba``, and the
    ``n_bins`` actually used (defaults to ``n_neighbors + 1``).
    """
    n_bins = calibration_n_bins if calibration_n_bins is not None else n_neighbors + 1
    multi_label = y_train.ndim == 2
    clf = KNNClassifier(n_neighbors=n_neighbors, device=device, use_fp16=False)
    clf.fit(x_train, y_train)

    if multi_label:
        if verbose:
            logger.info(f"[KNN] Fit KNN5 multilabel (train={len(x_train)}, test={len(x_test)})")
        y_scores = clf.predict_proba(x_test)
        metric, lo, hi = bootstrap_map(y_test, y_scores, n_boot=n_bootstrap, seed=seed)
        if verbose:
            logger.info(f"[KNN] Test micro_mAP={metric:.4f} (CI {lo:.4f}-{hi:.4f})")
    else:
        if verbose:
            logger.info(
                f"[KNN] Fit KNN5 (train={len(x_train)}, test={len(x_test)}, boot={n_bootstrap})"
            )
        preds = clf.predict(x_test)
        y_scores = clf.predict_proba(x_test)
        metric, lo, hi = bootstrap_accuracy(y_test, preds, n_boot=n_bootstrap, seed=seed)
        if verbose:
            logger.info(f"[KNN] Test accuracy={metric:.4f} (CI {lo:.4f}-{hi:.4f})")

    calibration = compute_calibration_metrics(
        y_test, y_scores, multi_label=multi_label, n_bins=n_bins
    )

    if verbose:
        logger.info(
            f"[KNN] Calibration (n_bins={n_bins}) ECE={calibration['ece']:.4f} "
            f"RMS-CE={calibration['rms_ce']:.4f} MCE={calibration['mce']:.4f}"
        )

    return metric, lo, hi, calibration, n_bins


def evaluate_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    c_values: Sequence[float],
    seed: int,
    n_bootstrap: int,
    merge_val: bool,
    device: str,
    verbose: bool = False,
    calibration_n_bins: int = 15,
    temp_scale: bool = True,
) -> tuple[float, float, float, float, dict[str, float], dict[str, float | None]]:
    """Sweep C values, retrain, and evaluate. Auto-detects single/multi-label from y shape.

    Returns the primary metric with bootstrap CI, the selected ``C``, a
    calibration dict from raw ``predict_proba`` on the test split, and a
    second dict with temperature-scaled calibration plus the fitted
    ``temperature`` (all ``None`` when ``temp_scale=False``).
    """
    multi_label = y_train.ndim == 2
    best_c: float | None = None
    best_val_score = -1.0

    x_train_tensor = torch.from_numpy(x_train)
    x_val_tensor = torch.from_numpy(x_val)
    x_test_tensor = torch.from_numpy(x_test)

    if multi_label:
        y_train_tensor = torch.from_numpy(y_train).float()
        label_tag = "LogReg-ML"
    else:
        y_train_tensor = torch.from_numpy(y_train).long()
        label_tag = "LogReg"

    if verbose:
        logger.info(
            f"[{label_tag}] C sweep start over {len(c_values)} values "
            f"(train={len(x_train)}, val={len(x_val)})"
        )
        c_value_iterator = track(c_values, description="C values")
    else:
        c_value_iterator = c_values

    for idx, c in enumerate(c_value_iterator):
        model = LogisticRegression(
            C=c,
            max_iter=2000,
            tol=1e-6,
            random_state=seed,
            device=device,
            multi_label=multi_label,
        )
        model.fit(x_train_tensor, y_train_tensor)

        if multi_label:
            val_scores = model.predict_proba(x_val_tensor)
            val_metric = float(average_precision_score(y_val, val_scores, average="micro"))
        else:
            val_pred = model.predict(x_val_tensor)
            val_metric = accuracy_score(y_val, val_pred)

        if verbose and (idx < 10 or idx % 50 == 0):
            logger.info(f"[{label_tag}] C={c:.4g} val_score={val_metric:.4f}")
        if val_metric > best_val_score:
            best_val_score = val_metric
            best_c = c

    assert best_c is not None, "C sweep failed to select a value"
    if verbose:
        logger.info(f"[{label_tag}] Best C={best_c:.4g} val_score={best_val_score:.4f}")

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

    calibration = compute_calibration_metrics(
        y_test, test_scores, multi_label=multi_label, n_bins=calibration_n_bins
    )

    calibration_ts: dict[str, float | None] = {
        "ece_ts": None,
        "rms_ce_ts": None,
        "mce_ts": None,
        "temperature": None,
    }
    if temp_scale:
        # Fit T on val logits, apply to test logits, recompute calibration.
        # When merge_val=True the final model has seen val during training, but
        # T is a single scalar so the resulting leakage is minimal.
        val_logits = final_model.decision_function(x_val_tensor)
        test_logits = final_model.decision_function(x_test_tensor)
        temperature = fit_temperature(val_logits, y_val, multi_label=multi_label)
        test_scores_ts = apply_temperature(test_logits, temperature, multi_label=multi_label)
        cal_ts = compute_calibration_metrics(
            y_test, test_scores_ts, multi_label=multi_label, n_bins=calibration_n_bins
        )
        calibration_ts = {
            "ece_ts": cal_ts["ece"],
            "rms_ce_ts": cal_ts["rms_ce"],
            "mce_ts": cal_ts["mce"],
            "temperature": temperature,
        }

    if verbose:
        logger.info(
            f"[{label_tag}] Test score={metric:.4f} (CI {lo:.4f}-{hi:.4f}) "
            f"using C={best_c:.4g}; train_final={len(x_final)} test={len(x_test)}"
        )
        logger.info(
            f"[{label_tag}] Calibration (n_bins={calibration_n_bins}) "
            f"ECE={calibration['ece']:.4f} "
            f"RMS-CE={calibration['rms_ce']:.4f} MCE={calibration['mce']:.4f}"
        )
        if temp_scale:
            logger.info(
                f"[{label_tag}] Post-TS T={calibration_ts['temperature']:.3f} "
                f"ECE={calibration_ts['ece_ts']:.4f} "
                f"RMS-CE={calibration_ts['rms_ce_ts']:.4f} "
                f"MCE={calibration_ts['mce_ts']:.4f}"
            )
    return metric, lo, hi, float(best_c), calibration, calibration_ts


def _make_seg_dataloaders(
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    test_loader: DataLoader,
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": test_loader.num_workers,
        "pin_memory": test_loader.pin_memory,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    train_val_loader = DataLoader(
        ConcatDataset([train_dataset, val_dataset]), shuffle=True, **loader_kwargs
    )
    return train_loader, val_loader, train_val_loader


def _build_seg_probe_and_solver(
    model: torch.nn.Module,
    num_classes: int,
    eval_cfg: DictConfig,
    device: torch.device,
    lr: float,
) -> tuple[SegmentationProbe, SegmentationSolver]:
    layer_names = list(eval_cfg.segmentation.layers)
    if not layer_names:
        raise ValueError(
            "Segmentation evaluation requires eval.segmentation.layers to name "
            "spatial backbone layers. Refusing to probe the global backbone output."
        )
    probe = SegmentationProbe(
        backbone=model,
        layer_names=layer_names,
        num_classes=num_classes,
        head_type=eval_cfg.segmentation.head_type,
        freeze_backbone=True,
    )
    criterion = instantiate(eval_cfg.segmentation.criterion)
    ignore_index = _resolve_segmentation_ignore_index(eval_cfg.segmentation, criterion)
    solver = SegmentationSolver(
        model=probe,
        num_classes=num_classes,
        lr=lr,
        device=str(device),
        criterion=criterion,
        lr_scheduler=eval_cfg.segmentation.get("lr_scheduler", "cosine"),
        ignore_index=ignore_index,
    )
    return probe, solver


def _resolve_segmentation_ignore_index(seg_cfg: DictConfig, criterion: torch.nn.Module) -> int:
    """Resolve the ignore index shared by segmentation loss and metrics."""
    explicit = seg_cfg.get("ignore_index", None)
    criterion_value = getattr(criterion, "ignore_index", None)
    if explicit is None:
        return int(criterion_value) if criterion_value is not None else 255
    if criterion_value is not None and int(criterion_value) != int(explicit):
        raise ValueError(
            "Segmentation ignore_index mismatch: "
            f"eval.segmentation.ignore_index={explicit} but "
            f"criterion.ignore_index={criterion_value}."
        )
    return int(explicit)


def evaluate_intrinsic_dim(
    splits: dict[str, np.ndarray],
    estimators: Sequence[str],
    selected_splits: Sequence[str],
    device: str | None,
    max_samples: int | None,
    seed: int,
    common_meta: dict,
    feature_dim: int,
    n_counts: dict[str, int],
    verbose: bool = False,
) -> list[dict]:
    """Compute intrinsic-dimension metrics over selected splits and return CSV rows.

    Each (split, estimator) yields one row with ``method="intrinsic_dim"`` and
    ``metric_name=f"id_{estimator}_{split}"``.
    """
    rows: list[dict] = []
    for split_name in selected_splits:
        if split_name not in splits:
            logger.warning(f"[intrinsic-dim] unknown split '{split_name}', skipping")
            continue
        X = splits[split_name]
        if verbose:
            logger.info(
                f"[intrinsic-dim] split={split_name} X{X.shape} "
                f"estimators={list(estimators)} device={device}"
            )
        # Per-estimator isolation: compute_intrinsic_dim raises on the
        # *first* non-finite dimension (by design — surfaces fp32 bugs).
        # During a long sweep that aborts the whole task and we lose KNN
        # /linear/profile rows too.  Run each estimator separately so a
        # genuinely-degenerate feature manifold (e.g. terramind features
        # with d1==d2 collapsing TwoNN's log-ratio) only loses *that*
        # estimator's row, not the rest of the task.
        dims: dict[str, float] = {}
        for est_name in estimators:
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
            except DegenerateManifoldError as exc:
                logger.warning(
                    f"[intrinsic-dim] {est_name} split={split_name} model={common_meta.get('model')} "
                    f"dataset={common_meta.get('dataset')} bands={common_meta.get('bands')} "
                    f"norm={common_meta.get('normalization')}: degenerate features, writing NaN. "
                    f"Diagnostic: {exc}"
                )
                dims[est_name] = float("nan")
        for est_name, dim in dims.items():
            rows.append(
                EvaluationResult(
                    **common_meta,
                    method="intrinsic_dim",
                    metric_name=f"id_{est_name}_{split_name}",
                    metric_value=float(dim),
                    ci_lower=0.0,
                    ci_upper=0.0,
                    feature_dim=feature_dim,
                    best_c=None,
                    best_lr=None,
                    best_batch_size=None,
                    n_train=n_counts.get("train", 0),
                    n_val=n_counts.get("val", 0),
                    n_test=n_counts.get("test", 0),
                ).to_row()
            )
    return rows


def evaluate_profile(
    model: BenchModel,
    sample_loader: DataLoader,
    device: torch.device,
    n_warmup: int,
    n_measure: int,
    common_meta: dict,
    feature_dim: int,
    n_counts: dict[str, int],
    cpu_throughput_enabled: bool = False,
    cpu_batch_size: int = 8,
    cpu_n_warmup: int = 1,
    cpu_n_measure: int = 5,
    cpu_time_budget_s: float = 300.0,
) -> list[dict]:
    """Measure backbone throughput / memory / GMACs and return CSV rows.

    One row per metric, with ``method="profile"``.

    When ``cpu_throughput_enabled`` is set, *additionally* runs a short
    CPU measurement (smaller batch / fewer iters) and emits the
    throughput / latency / energy / params with a ``_cpu`` suffix.  The
    CPU pass is wall-clock-budgeted via ``cpu_time_budget_s`` so the
    heavyweight ViT-L backbones don't burn an hour on the login node.
    """
    # If the loader is broken there's nothing meaningful to profile; let the
    # error propagate so the failure surfaces in SLURM logs instead of
    # silently appending zero rows and "succeeding" the task.
    sample = next(iter(sample_loader))["image"].to(device)

    metrics = measure_profile(model, sample, device, n_warmup=n_warmup, n_measure=n_measure)

    if cpu_throughput_enabled:
        cpu_metrics = _measure_cpu_throughput(
            model,
            sample,
            cpu_batch_size=cpu_batch_size,
            n_warmup=cpu_n_warmup,
            n_measure=cpu_n_measure,
            time_budget_s=cpu_time_budget_s,
        )
        for k, v in cpu_metrics.items():
            metrics[k + "_cpu"] = v

    rows: list[dict] = []
    for name, value in metrics.items():
        if value is None:
            # value is None only when the underlying probe is structurally
            # unavailable (e.g. CPU device → no peak_gpu_mem, or the CPU
            # pass aborted via the wall-clock budget). Logged inside the
            # measurement helpers; skip the row.
            continue
        rows.append(
            EvaluationResult(
                **common_meta,
                method="profile",
                metric_name=name,
                metric_value=float(value),
                ci_lower=0.0,
                ci_upper=0.0,
                feature_dim=feature_dim,
                best_c=None,
                best_lr=None,
                best_batch_size=None,
                n_train=n_counts.get("train", 0),
                n_val=n_counts.get("val", 0),
                n_test=n_counts.get("test", 0),
            ).to_row()
        )
    return rows


def _measure_cpu_throughput(
    model: BenchModel,
    sample: torch.Tensor,
    *,
    cpu_batch_size: int,
    n_warmup: int,
    n_measure: int,
    time_budget_s: float,
) -> dict[str, float | None]:
    """Run a wall-clock-budgeted CPU pass and return the off-GPU metrics.

    Reports the subset that makes sense on CPU: throughput and latency.
    The model and a fresh batch are moved to CPU for the duration, then
    moved back so the rest of the pipeline can keep using CUDA.  If even
    the first warmup pass exceeds ``time_budget_s`` we return None values
    with a warning rather than waste cluster hours — that's a documented
    soft-fail keyed on a specific named condition, not a generic swallow.
    """
    cpu_dev = torch.device("cpu")
    # rcf/imagestats baselines have no parameters; use the input sample's
    # device as the restoration target since model.to() is a no-op anyway.
    params_iter = iter(model.parameters())
    first_param = next(params_iter, None)
    orig_dev = first_param.device if first_param is not None else sample.device
    cpu_sample = sample[:cpu_batch_size].detach().to(cpu_dev)
    model.to(cpu_dev)
    try:
        t0 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(n_warmup):
                model(cpu_sample)
                if time.perf_counter() - t0 > time_budget_s:
                    logger.warning(
                        f"[profile] CPU warmup exceeded {time_budget_s}s budget on "
                        f"{type(model).__name__}; skipping CPU throughput."
                    )
                    return {
                        "throughput_samples_per_sec": None,
                        "latency_ms_per_batch_p50": None,
                    }
            per_batch_ms: list[float] = []
            t_loop = time.perf_counter()
            for _ in range(n_measure):
                tb = time.perf_counter()
                model(cpu_sample)
                per_batch_ms.append((time.perf_counter() - tb) * 1000.0)
                if time.perf_counter() - t0 > time_budget_s:
                    break
            elapsed = time.perf_counter() - t_loop
        seen = len(per_batch_ms)
        if seen == 0:
            return {"throughput_samples_per_sec": None, "latency_ms_per_batch_p50": None}
        return {
            "throughput_samples_per_sec": (cpu_batch_size * seen) / elapsed,
            "latency_ms_per_batch_p50": median(per_batch_ms),
        }
    finally:
        model.to(orig_dev)


def evaluate_segmentation(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    cfg: DictConfig,
    num_classes: int,
    device: torch.device,
    collect_preds: bool = False,
) -> "tuple[SegMetrics, int, float | None, int | None, torch.Tensor | None]":
    """Evaluate segmentation performance using a frozen-backbone segmentation probe.

    Trains a lightweight segmentation head on top of the frozen backbone and
    evaluates mIoU on the test split. Optionally pre-caches backbone features for
    faster training across epochs.

    Args:
        model: Frozen backbone model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        test_loader: Test DataLoader.
        cfg: Full Hydra config.
        num_classes: Number of segmentation classes.
        device: Torch device.
        collect_preds: If True, collect and return test predictions as (N, H, W) tensor.

    Returns:
        Tuple of (metrics_dict, feature_dim, None, None, preds_or_None).
        ``preds_or_None`` is None when collect_preds is False.
    """
    # Merge model-specific eval config if present
    eval_cfg = cfg.eval
    if "eval" in cfg.model and cfg.model.eval is not None:
        eval_cfg = OmegaConf.merge(eval_cfg, cfg.model.eval)
    if "segmentation" not in eval_cfg:
        raise ValueError("Segmentation evaluation config missing for the model.")

    seg_cfg = eval_cfg.segmentation
    epochs = seg_cfg.epochs
    use_cache = seg_cfg.get("cache_features", True)
    cache_dtype_str = seg_cfg.get("cache_dtype", "float16")
    cache_dtype = torch.float16 if cache_dtype_str == "float16" else torch.float32

    probe, solver = _build_seg_probe_and_solver(model, num_classes, eval_cfg, device, seg_cfg.lr)
    if use_cache and probe.freeze_backbone:
        logger.info("Caching backbone features for train and val splits...")
        train_cache = probe.extract_segmentation_features(train_loader, cache_dtype=cache_dtype)
        val_cache = probe.extract_segmentation_features(val_loader, cache_dtype=cache_dtype)
        test_cache = probe.extract_segmentation_features(test_loader, cache_dtype=cache_dtype)
        solver.fit_cached(
            train_cache=train_cache,
            val_cache=val_cache,
            batch_size=seg_cfg.get("batch_size", 64),
            epochs=epochs,
            verbose=cfg.verbose,
        )
        eval_result = solver.evaluate_cached(
            test_cache,
            batch_size=seg_cfg.get("batch_size", 64),
            collect_preds=collect_preds,
        )
    else:
        solver.fit(
            train_loader=train_loader, val_loader=val_loader, epochs=epochs, verbose=cfg.verbose
        )
        eval_result = solver.evaluate(test_loader, collect_preds=collect_preds)

    if collect_preds:
        metrics, preds = eval_result
    else:
        metrics, preds = eval_result, None
    return metrics, sum(probe.channels_list), None, None, preds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


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
    if not rows:
        return
    df_local = pd.DataFrame(rows)
    fd = os.open(path, os.O_RDWR | os.O_CREAT)
    with os.fdopen(fd, "r+", closefd=True) as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
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
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the benchmark pipeline for all configured datasets and models."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    dataset_names = _expand_dataset_list(cfg.dataset.names)
    device = torch.device(cfg.device)

    output_path = cfg.output
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    all_rows: list[dict] = []
    model_eval = cfg.model.get("eval", None) if "eval" in cfg.model else None
    if model_eval is not None and model_eval.get("c_range", None) is not None:
        c_start, c_stop, c_num = model_eval.c_range
    else:
        c_start, c_stop, c_num = cfg.eval.c_range
    c_values = 10 ** np.linspace(float(c_start), float(c_stop), int(c_num))
    c_values_list = [float(v) for v in c_values.tolist()]

    key_cols = (
        "dataset",
        "method",
        "model",
        "name",
        "normalization",
        "image_size",
        "interpolation",
        "partition",
        "bands",
    )
    completed_runs: set[tuple[str, ...]] = set()
    completed_metrics: dict[str, set[tuple[str, ...]]] = {}
    if cfg.resume and os.path.exists(output_path):
        existing_df = pd.read_csv(cfg.output)
        for col in key_cols:
            if col not in existing_df.columns:
                existing_df[col] = ""
        completed_runs = _completed_run_keys(existing_df, key_cols)
        if "metric_name" in existing_df.columns:
            completed_metrics = {
                str(metric): _completed_run_keys(existing_df, key_cols, str(metric))
                for metric in existing_df["metric_name"].dropna().unique()
            }
        logger.info(f"Resume mode: Found {len(completed_runs)} existing results in {cfg.output}")
        logger.info("Will skip already-computed (dataset, method, model, config) combinations.")

    # Selectable input-normalisation strategy; recorded in the CSV so
    # ablations across strategies are distinguishable.
    normalization = str(getattr(cfg.dataset, "normalization", "bandspec_zscore"))
    bands_value = _normalize_bands_value(getattr(cfg.dataset, "bands", "rgb"))

    for ds_name in track(dataset_names, description="Datasets"):
        try:
            ds_cls = get_bench_dataset_class(ds_name)
        except KeyError:
            logger.warning(f"Skipping dataset {ds_name} (not in registry)")
            continue

        config_tuple = (
            normalization,
            str(getattr(cfg.dataset, "image_size", None)),
            getattr(cfg.dataset, "interpolation", "bilinear"),
            cfg.dataset.partition,
            bands_value,
        )

        # Merge model-specific eval config early so resume key and result rows
        # reflect the actual head_type used, not the global default.
        eval_cfg_merged = OmegaConf.merge(
            cfg.eval,
            cfg.model.eval if "eval" in cfg.model and cfg.model.eval is not None else {},
        )

        knn_k = int(getattr(eval_cfg_merged, "knn_k", 5))
        knn_key = (ds_name, f"knn{knn_k}", cfg.model._target_, cfg.model.name, *config_tuple)
        linear_key = (ds_name, "linear", cfg.model._target_, cfg.model.name, *config_tuple)

        seg_method = f"seg-{eval_cfg_merged.segmentation.head_type}"
        seg_key = (ds_name, seg_method, cfg.model._target_, cfg.model.name, *config_tuple)
        id_key = (ds_name, "intrinsic_dim", cfg.model._target_, cfg.model.name, *config_tuple)
        profile_key = (ds_name, "profile", cfg.model._target_, cfg.model.name, *config_tuple)

        try:
            result = get_datasets(
                dataset_name=ds_name,
                partition_name=cfg.dataset.partition,
                batch_size=cfg.dataset.batch_size,
                num_workers=int(cfg.dataset.get("num_workers", 8)),
                return_val=True,
                image_size=getattr(cfg.dataset, "image_size", None),
                interpolation=getattr(cfg.dataset, "interpolation", "bilinear"),
                bands=getattr(cfg.dataset, "bands", "rgb"),
            )
        except (FileNotFoundError, DatasetNotFoundError) as exc:
            logger.warning(f"Skipping dataset {ds_name} (data not found: {exc})")
            continue
        if result is None or not isinstance(result, tuple) or len(result) != 4:
            logger.warning(f"Skipping dataset {ds_name} (unexpected return)")
            continue
        train_dataset, train_loader, val_loader, test_loader = result

        num_channels = train_dataset[0]["image"].shape[0]
        is_segmentation = ds_cls.task == "segmentation"
        is_multilabel = ds_cls.multilabel
        num_classes = ds_cls.num_classes

        # Build the BandSpec list that matches the actual loaded channels.
        bench_for_bands = ds_cls()
        bands_resolved = (
            tuple(bench_for_bands.rgb_bands)
            if cfg.dataset.bands == "rgb"
            else None
            if cfg.dataset.bands in ("all", None)
            else tuple(cfg.dataset.bands)
        )
        bands_list = bench_for_bands.select_band_specs(bands_resolved)
        assert len(bands_list) == num_channels, (
            f"BandSpec count {len(bands_list)} != tensor channel count {num_channels} "
            f"for dataset {ds_name}; sample-level canonicalization may have changed shape."
        )

        # Resume check for segmentation
        if is_segmentation and cfg.resume and seg_key in completed_runs:
            if cfg.verbose:
                logger.info(f"[{ds_name}] Skipping segmentation (already computed)")
            continue

        # Instantiate Backbone — pass `bands` post-hoc so Hydra never tries
        # to OmegaConf-ify the BandSpec list.  `_convert_="object"` keeps
        # the rest of the model config as plain Python primitives.
        is_rcf_empirical = (
            hasattr(cfg.model, "mode")
            and str(cfg.model._target_).endswith("RCFBench")
            and str(cfg.model.mode) == "empirical"
        )
        instantiate_kwargs: dict = {
            "bands": bands_list,
            "normalization": normalization,
            "_convert_": "object",
        }
        if is_rcf_empirical:
            instantiate_kwargs["dataset"] = train_dataset
        model: BenchModel = instantiate(cfg.model, **instantiate_kwargs)
        model.to(device).eval()

        common_meta = {
            "dataset": ds_name,
            "seed": cfg.seed,
            "model": cfg.model._target_,
            "name": cfg.model.name,
            "normalization": normalization,
            "image_size": getattr(cfg.dataset, "image_size", None),
            "interpolation": getattr(cfg.dataset, "interpolation", "bilinear"),
            "partition": cfg.dataset.partition,
            "bands": bands_value,
            "c_range_start": c_start,
            "c_range_stop": c_stop,
            "c_range_num": c_num,
            "merge_val": cfg.eval.merge_val,
            "bootstrap": cfg.eval.bootstrap,
        }

        if is_segmentation:
            seg_cfg_merged = OmegaConf.merge(
                cfg.eval,
                cfg.model.eval if "eval" in cfg.model and cfg.model.eval is not None else {},
            ).segmentation
            save_viz = seg_cfg_merged.get("save_viz", False)
            metrics, feat_dim, best_lr, best_bs, preds = evaluate_segmentation(
                model,
                train_loader,
                val_loader,
                test_loader,
                cfg,
                num_classes,
                device,
                collect_preds=save_viz,
            )
            all_rows.append(
                EvaluationResult(
                    **common_meta,
                    method=seg_method,
                    metric_name="mIoU",
                    metric_value=metrics.get("mIoU", float("nan")),
                    ci_lower=0.0,
                    ci_upper=0.0,
                    feature_dim=feat_dim,
                    best_c=None,
                    best_lr=best_lr,
                    best_batch_size=best_bs,
                    n_train=len(train_dataset),
                    n_val=len(val_loader.dataset),
                    n_test=len(test_loader.dataset),
                    fw_iou=metrics.get("fw_IoU"),
                    precision=metrics.get("precision"),
                    recall=metrics.get("recall"),
                    f1=metrics.get("f1"),
                ).to_row()
            )
            if save_viz and preds is not None:
                rgb_indices = ds_cls().rgb_indices or [0, 1, 2]
                # Collect images and GT masks from test_loader (cheap pass, no backbone)
                test_imgs, test_gts = [], []
                for _batch in test_loader:
                    if isinstance(_batch, dict):
                        test_imgs.append(_batch["image"])
                        _m = _batch["mask"]
                    else:
                        test_imgs.append(_batch[0])
                        _m = _batch[1]
                    if _m.ndim == 4:
                        _m = _m.squeeze(1)
                    test_gts.append(_m.long())
                test_imgs_t = torch.cat(test_imgs, dim=0)
                test_gts_t = torch.cat(test_gts, dim=0)
                ignore_idx = seg_cfg_merged.get("ignore_index", 255)
                n_viz = seg_cfg_merged.get("n_viz_samples", 8)
                viz_dir = seg_cfg_merged.get("viz_dir", "viz")
                _class_names = list(getattr(train_dataset, "classes", None) or []) or None
                save_segmentation_viz(
                    out_dir=viz_dir,
                    model_name=cfg.model.name,
                    dataset_name=ds_name,
                    images=test_imgs_t,
                    gt_masks=test_gts_t,
                    pred_masks=preds,
                    num_classes=num_classes,
                    rgb_indices=rgb_indices,
                    ignore_index=ignore_idx,
                    n_samples=n_viz,
                    class_names=_class_names,
                )
        else:
            # Classification (single-label or multi-label)
            metric_name = "micro_mAP" if is_multilabel else "accuracy"

            skip_knn = cfg.resume and knn_key in completed_runs
            skip_linear = (cfg.resume and linear_key in completed_runs) or getattr(
                cfg.eval, "skip_linear", False
            )
            id_cfg = getattr(cfg.eval, "intrinsic_dim", None)
            id_enabled = bool(id_cfg and id_cfg.get("enabled", False))
            id_metric_names = (
                [f"id_{est}_{split}" for split in id_cfg.splits for est in id_cfg.estimators]
                if id_enabled
                else []
            )
            skip_id = (not id_enabled) or (
                cfg.resume
                and id_metric_names
                and all(
                    id_key in completed_metrics.get(metric, set()) for metric in id_metric_names
                )
            )
            profile_cfg = getattr(cfg.eval, "profile", None)
            profile_enabled = bool(profile_cfg and profile_cfg.get("enabled", False))
            profile_metric_names = _profile_metric_names(profile_cfg) if profile_enabled else []
            skip_profile = (not profile_enabled) or (
                cfg.resume
                and profile_metric_names
                and all(
                    profile_key in completed_metrics.get(metric, set())
                    for metric in profile_metric_names
                )
            )

            if skip_knn and skip_linear and skip_id and skip_profile:
                continue

            x_train, y_train = embed_split(model, train_loader, device, verbose=cfg.verbose)
            x_val, y_val = embed_split(model, val_loader, device, verbose=cfg.verbose)
            x_test, y_test = embed_split(model, test_loader, device, verbose=cfg.verbose)
            feature_dim = x_train.shape[1]

            cal_cfg = cfg.eval.get("calibration", {}) or {}
            cal_n_bins_knn = cal_cfg.get("n_bins_knn", None)
            cal_n_bins_linear = int(cal_cfg.get("n_bins_linear", 15))
            cal_temp_scale = bool(cal_cfg.get("temp_scale", True))

            if not skip_knn:
                knn_device = cfg.eval.get("knn_device") or cfg.device
                knn_score, knn_lo, knn_hi, knn_cal, knn_n_bins = evaluate_knn(
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    cfg.seed,
                    cfg.eval.bootstrap,
                    verbose=cfg.verbose,
                    device=knn_device,
                    n_neighbors=knn_k,
                    calibration_n_bins=cal_n_bins_knn,
                )
                all_rows.append(
                    EvaluationResult(
                        **common_meta,
                        method=f"knn{knn_k}",
                        metric_name=metric_name,
                        metric_value=knn_score,
                        ci_lower=knn_lo,
                        ci_upper=knn_hi,
                        feature_dim=feature_dim,
                        best_c=None,
                        best_lr=None,
                        best_batch_size=None,
                        n_train=len(x_train),
                        n_val=len(x_val),
                        n_test=len(x_test),
                        ece=knn_cal["ece"],
                        rms_ce=knn_cal["rms_ce"],
                        mce=knn_cal["mce"],
                        calibration_n_bins=knn_n_bins,
                    ).to_row()
                )

            if not skip_linear:
                lin_score, lin_lo, lin_hi, best_c, lin_cal, lin_cal_ts = evaluate_logistic(
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                    x_test,
                    y_test,
                    c_values_list,
                    cfg.seed,
                    cfg.eval.bootstrap,
                    cfg.eval.merge_val,
                    cfg.device,
                    cfg.verbose,
                    calibration_n_bins=cal_n_bins_linear,
                    temp_scale=cal_temp_scale,
                )
                all_rows.append(
                    EvaluationResult(
                        **common_meta,
                        method="linear",
                        metric_name=metric_name,
                        metric_value=lin_score,
                        ci_lower=lin_lo,
                        ci_upper=lin_hi,
                        feature_dim=feature_dim,
                        best_c=best_c,
                        best_lr=None,
                        best_batch_size=None,
                        n_train=len(x_train),
                        n_val=len(x_val),
                        n_test=len(x_test),
                        ece=lin_cal["ece"],
                        rms_ce=lin_cal["rms_ce"],
                        mce=lin_cal["mce"],
                        ece_ts=lin_cal_ts["ece_ts"],
                        rms_ce_ts=lin_cal_ts["rms_ce_ts"],
                        mce_ts=lin_cal_ts["mce_ts"],
                        temperature=lin_cal_ts["temperature"],
                        calibration_n_bins=cal_n_bins_linear,
                    ).to_row()
                )

            if not skip_id:
                id_rows = evaluate_intrinsic_dim(
                    splits={"train": x_train, "val": x_val, "test": x_test},
                    estimators=list(id_cfg.estimators),
                    selected_splits=list(id_cfg.splits),
                    device=id_cfg.get("device", None) or cfg.device,
                    max_samples=id_cfg.get("max_samples", None),
                    seed=cfg.seed,
                    common_meta=common_meta,
                    feature_dim=feature_dim,
                    n_counts={
                        "train": len(x_train),
                        "val": len(x_val),
                        "test": len(x_test),
                    },
                    verbose=cfg.verbose,
                )
                if cfg.resume:
                    id_rows = _filter_completed_metric_rows(id_rows, completed_metrics, key_cols)
                all_rows.extend(id_rows)

            if not skip_profile:
                cpu_cfg = profile_cfg.get("cpu_throughput", {}) if profile_cfg else {}
                profile_rows = evaluate_profile(
                    model=model,
                    sample_loader=train_loader,
                    device=torch.device(cfg.device),
                    n_warmup=int(profile_cfg.get("n_warmup", 3)),
                    n_measure=int(profile_cfg.get("n_measure", 20)),
                    common_meta=common_meta,
                    feature_dim=feature_dim,
                    n_counts={
                        "train": len(x_train),
                        "val": len(x_val),
                        "test": len(x_test),
                    },
                    cpu_throughput_enabled=bool(cpu_cfg.get("enabled", False)),
                    cpu_batch_size=int(cpu_cfg.get("batch_size", 8)),
                    cpu_n_warmup=int(cpu_cfg.get("n_warmup", 1)),
                    cpu_n_measure=int(cpu_cfg.get("n_measure", 5)),
                    cpu_time_budget_s=float(cpu_cfg.get("time_budget_s", 300.0)),
                )
                if cfg.resume:
                    profile_rows = _filter_completed_metric_rows(
                        profile_rows, completed_metrics, key_cols
                    )
                all_rows.extend(profile_rows)

        append_rows_atomic(output_path, all_rows)
        all_rows.clear()

    logger.info(f"Benchmark complete. Results appended to {output_path}")


if __name__ == "__main__":  # pragma: no cover
    # Hydra provides cfg automatically; this call signature is correct.
    main()  # type: ignore[misc]
