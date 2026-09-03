"""Benchmark script for torchgeo-bench."""

import copy
import logging
import math
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch.utils.data import DataLoader

from torchgeo_bench.calibration import (
    apply_temperature,
    compute_calibration_metrics,
    fit_temperature,
)
from torchgeo_bench.config import instantiate
from torchgeo_bench.datasets import (
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
from torchgeo_bench.model_profile import measure_cpu_throughput, measure_profile
from torchgeo_bench.models.interface import BenchModel
from torchgeo_bench.results import (  # noqa: F401  (EvaluationResult re-exported)
    DEFAULT_INTRINSIC_DIM_RESULTS_DIR,
    DEFAULT_PROFILE_RESULTS_DIR,
    DEFAULT_RESULTS_DIR,
    EvaluationResult,
    append_rows_atomic,
    bootstrap_accuracy,
    bootstrap_map,
    bootstrap_miou,
    metric_row,
    model_results_path,
)
from torchgeo_bench.resume import (  # noqa: F401  (re-exported for back-compat)
    KEY_COLS,
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
from torchgeo_bench.settings import RunSettings, merge
from torchgeo_bench.utils import extract_features, resolve_device

if TYPE_CHECKING:
    import torchgeo_bench.segmentation_task
    from torchgeo_bench.settings import EvalSettings, SegmentationSettings

logger = logging.getLogger(__name__)


def resolve_model_config(model_cfg: dict, dataset_name: str) -> dict:
    """Apply a dataset-specific partial override to a model configuration."""
    resolved = copy.deepcopy(model_cfg)
    dataset_overrides = resolved.pop("dataset_overrides", {})
    return merge(resolved, dataset_overrides.get(dataset_name, {}) or {})


def resolve_configured_device(cfg_device: str) -> torch.device:
    """Resolve ``cfg.device``, choosing CPU only for the implicit/explicit ``"auto"``.

    An explicit device request (anything other than ``"auto"``, whether from
    an explicit ``--device`` flag or a ``--config`` YAML) must fail loudly if
    unavailable -- that's what :func:`resolve_device` already does. Only the
    *default* device value (``"auto"``, chosen when ``--device`` is omitted)
    may silently prefer CPU when CUDA isn't installed; a user who actually
    typed ``--device auto`` is asking for that same automatic behavior.

    ``"cuda:0"``, not bare ``"cuda"``, is the auto-selected GPU device: it
    matches the previous hardcoded default's string form exactly, so a run
    that never overrides ``device`` keeps the same resume ``config_hash``
    (which is computed from ``cfg.device`` *after* this resolution) as before
    "auto" existed.
    """
    if cfg_device == "auto":
        return resolve_device("cuda:0" if torch.cuda.is_available() else "cpu")
    return resolve_device(cfg_device)


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


def _validate_dataset_names(names: Sequence[str]) -> None:
    """Fail fast on any unregistered dataset name, listing what is available.

    A typo in ``dataset.names`` used to surface as a per-dataset "Skipping
    dataset ... (not in registry)" warning discovered mid-sweep, after the
    model was already loaded for every dataset before it. Validating the
    full list upfront means a bad name is a startup error, not a silently
    incomplete run.
    """
    available = set(list_datasets())
    unknown = [name for name in names if name not in available]
    if unknown:
        raise ValueError(
            f"Unknown dataset(s): {', '.join(unknown)}. Available: {', '.join(sorted(available))}."
        )


def embed_split(
    model: BenchModel,
    dataloader: DataLoader,
    device: torch.device,
    verbose: bool,
    split: str = "",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature embeddings and labels from a data split."""
    description = f"Extracting ({split})" if split else "Extracting"
    return extract_features(
        model, dataloader, device, transforms=None, verbose=verbose, description=description
    )


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


class LinearProbeDivergedError(RuntimeError):
    """Raised when every candidate C in the sweep produced a non-finite score.

    Distinct from a single bad candidate (handled inline by scoring it -inf so
    the sweep just moves on) -- this means the features themselves are
    unusable for this backbone/dataset pairing at every regularization
    strength tried, so there is no "best_c" to report.
    """


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
    from sklearn.metrics import accuracy_score, average_precision_score

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

    for idx, c in enumerate(c_values):
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
            if not np.all(np.isfinite(val_scores)):
                # Extreme C values can make LogisticRegression's weights diverge,
                # producing non-finite logits/probabilities. average_precision_score
                # raises on that rather than scoring it low, which would otherwise
                # crash the whole C sweep over one bad candidate; treat it as the
                # worst possible score instead so the sweep just moves on to the
                # next C (best_val_score's -1.0 floor already skips it below).
                val_metric = float("-inf")
            else:
                val_metric = float(average_precision_score(y_val, val_scores, average="micro"))
        else:
            val_pred = model.predict(x_val_tensor)
            val_metric = accuracy_score(y_val, val_pred)

        if verbose and (idx < 10 or idx % 50 == 0):
            logger.info(f"[{label_tag}] C={c:.4g} val_score={val_metric:.4f}")
        if val_metric > best_val_score:
            best_val_score = val_metric
            best_c = c

    if best_c is None:
        raise LinearProbeDivergedError(
            f"Every candidate C in {list(c_values)} produced a non-finite val score; "
            "features are unusable for a linear probe at any regularization strength tried."
        )
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
    if temp_scale and not merge_val:
        # Fit T on val logits, apply to test logits, recompute calibration.
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
    elif temp_scale and merge_val:
        logger.warning(
            "[%s] Skipping temperature scaling because merge_val=true leaves no held-out "
            "calibration split.",
            label_tag,
        )

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
        if calibration_ts["temperature"] is not None:
            logger.info(
                f"[{label_tag}] Post-TS T={calibration_ts['temperature']:.3f} "
                f"ECE={calibration_ts['ece_ts']:.4f} "
                f"RMS-CE={calibration_ts['rms_ce_ts']:.4f} "
                f"MCE={calibration_ts['mce_ts']:.4f}"
            )
    return metric, lo, hi, float(best_c), calibration, calibration_ts


def _resolve_segmentation_runtime_config(
    seg_cfg: "SegmentationSettings",
) -> tuple[int, int, float, bool, torch.dtype]:
    """Validate and normalize the segmentation settings used during a run.

    Configuration mistakes should fail before feature extraction or training,
    rather than being silently coerced or failing after an expensive GPU job
    has started. ``seg_cfg``'s fields always exist (a dataclass), but their
    *values* are never runtime-type-checked by Python, so a malformed
    ``--config`` YAML (e.g. ``epochs: "ten"``) still needs this check.
    """
    epochs = seg_cfg.epochs
    batch_size = seg_cfg.batch_size
    lr = seg_cfg.lr
    use_cache = seg_cfg.cache_features
    cache_dtype_name = seg_cfg.cache_dtype

    for name, value in (("epochs", epochs), ("batch_size", batch_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"eval.segmentation.{name} must be a positive integer, got {value!r}.")
    if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
        raise ValueError(f"eval.segmentation.lr must be a finite positive number, got {lr!r}.")
    if not isinstance(use_cache, bool):
        raise ValueError(f"eval.segmentation.cache_features must be a boolean, got {use_cache!r}.")
    cache_dtypes = {"float16": torch.float16, "float32": torch.float32}
    if cache_dtype_name not in cache_dtypes:
        raise ValueError(
            "eval.segmentation.cache_dtype must be one of "
            f"{tuple(cache_dtypes)}, got {cache_dtype_name!r}."
        )
    return epochs, batch_size, float(lr), use_cache, cache_dtypes[cache_dtype_name]


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
    only_metrics: frozenset[str] | None = None,
) -> list[dict]:
    """Compute intrinsic-dimension metrics over selected splits and return CSV rows.

    Each (split, estimator) yields one ``id_<estimator>_<split>`` row. Five
    centered feature-spectrum diagnostics are also emitted per split with a
    ``spectrum_<metric>_<split>`` name. All rows use
    ``method="intrinsic_dim"`` so they share the existing side-output and
    resume path.

    ``only_metrics``, when given, restricts computation to metric names not
    already present on disk -- resuming a run that's missing only the newer
    spectrum rows shouldn't re-run the far more expensive torchid estimators
    just to recompute rows that already exist and would be filtered out
    anyway. ``None`` computes everything (a fresh, non-resumed run).
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
        # Isolate per estimator: compute_intrinsic_dim raises on the first
        # non-finite dimension, which would otherwise cost the other rows too.
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
                metric_row(
                    common_meta,
                    method="intrinsic_dim",
                    metric_name=f"id_{est_name}_{split_name}",
                    metric_value=float(dim),
                    feature_dim=feature_dim,
                    n_counts=n_counts,
                )
            )

        spectrum_names = {f"spectrum_{metric}_{split_name}" for metric in FEATURE_SPECTRUM_METRICS}
        if only_metrics is not None and only_metrics.isdisjoint(spectrum_names):
            continue
        try:
            spectrum = compute_feature_spectrum(X, max_samples=max_samples, seed=seed)
        except DegenerateSpectrumError as exc:
            logger.warning(
                f"[intrinsic-dim] spectrum split={split_name} model={common_meta.get('model')} "
                f"dataset={common_meta.get('dataset')} bands={common_meta.get('bands')} "
                f"norm={common_meta.get('normalization')}: degenerate features, writing NaN. "
                f"Diagnostic: {exc}"
            )
            spectrum = {metric: float("nan") for metric in FEATURE_SPECTRUM_METRICS}
        for metric_name, value in spectrum.items():
            if (
                only_metrics is not None
                and f"spectrum_{metric_name}_{split_name}" not in only_metrics
            ):
                continue
            rows.append(
                metric_row(
                    common_meta,
                    method="intrinsic_dim",
                    metric_name=f"spectrum_{metric_name}_{split_name}",
                    metric_value=value,
                    feature_dim=feature_dim,
                    n_counts=n_counts,
                )
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
    """Measure backbone throughput / memory / params and return CSV rows.

    One row per metric, with ``method="profile"``.

    When ``cpu_throughput_enabled`` is set, *additionally* runs a short
    CPU measurement (smaller batch / fewer iters) and emits the
    throughput / latency with a ``_cpu`` suffix.  The
    CPU pass is wall-clock-budgeted via ``cpu_time_budget_s`` so the
    heavyweight ViT-L backbones don't burn an hour on the login node.
    """
    # A broken loader has nothing to profile; let it raise.
    sample = next(iter(sample_loader))["image"].to(device)

    metrics = measure_profile(model, sample, device, n_warmup=n_warmup, n_measure=n_measure)

    if cpu_throughput_enabled:
        metrics.update(
            measure_cpu_throughput(
                model,
                sample,
                batch_size=cpu_batch_size,
                n_warmup=cpu_n_warmup,
                n_measure=cpu_n_measure,
                time_budget_s=cpu_time_budget_s,
            )
        )

    rows: list[dict] = []
    for name, value in metrics.items():
        if value is None:
            # value is None only when the underlying probe is structurally
            # unavailable (e.g. CPU device → no peak_gpu_mem, or the CPU
            # pass aborted via the wall-clock budget). Logged inside the
            # measurement helpers; skip the row.
            continue
        rows.append(
            metric_row(
                common_meta,
                method="profile",
                metric_name=name,
                metric_value=float(value),
                feature_dim=feature_dim,
                n_counts=n_counts,
            )
        )
    return rows


def evaluate_segmentation(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    eval_cfg: "EvalSettings",
    num_classes: int,
    device: torch.device,
    seed: int,
    collect_preds: bool = False,
    verbose: bool = False,
) -> "tuple[torchgeo_bench.segmentation_task.SegMetrics, int, float | None, int | None, torch.Tensor | None]":
    """Evaluate segmentation performance using a frozen-backbone segmentation probe.

    Trains a lightweight segmentation head on top of the frozen backbone and
    evaluates mIoU on the test split. Optionally pre-caches backbone features
    for faster training across epochs.

    Args:
        model: Frozen backbone model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        test_loader: Test DataLoader.
        eval_cfg: The eval settings, already merged with any model-specific
            ``eval`` overrides.
        num_classes: Number of segmentation classes.
        device: Torch device.
        seed: RNG seed for the mIoU bootstrap when ``eval_cfg.bootstrap`` > 0.
        collect_preds: If True, collect and return test predictions as (N, H, W) tensor.
        verbose: Show training progress.

    Returns:
        Tuple of (metrics, feature_dim, lr, batch_size, preds); ``preds`` is
        None unless ``collect_preds``.
    """
    from torchgeo_bench.segmentation_task import build_seg_probe_and_solver

    seg_cfg = eval_cfg.segmentation
    epochs, probe_batch_size, lr, use_cache, cache_dtype = _resolve_segmentation_runtime_config(
        seg_cfg
    )

    probe, solver = build_seg_probe_and_solver(model, num_classes, seg_cfg, device, lr)
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
            collect_preds=collect_preds,
            collect_confusions=collect_confusions,
        )
    else:
        solver.fit(train_loader=train_loader, val_loader=val_loader, epochs=epochs, verbose=verbose)
        eval_result = solver.evaluate(
            test_loader,
            collect_preds=collect_preds,
            collect_confusions=collect_confusions,
        )
    actual_batch_size = (
        probe_batch_size
        if use_cache and probe.freeze_backbone
        else int(train_loader.batch_size or 1)
    )

    if collect_preds and collect_confusions:
        metrics, preds, confusion_matrices = eval_result
    elif collect_preds:
        metrics, preds = eval_result
        confusion_matrices = None
    elif collect_confusions:
        metrics, confusion_matrices = eval_result
        preds = None
    else:
        metrics, preds = eval_result, None
        confusion_matrices = None
    if confusion_matrices is not None:
        metrics["ci_lower"], metrics["ci_upper"] = bootstrap_miou(
            confusion_matrices,
            n_boot=int(eval_cfg.bootstrap),
            seed=seed,
        )
    return metrics, sum(probe.channels_list), lr, actual_batch_size, preds


def _resolve_output_path(cfg: RunSettings) -> str:
    """Return the CSV to write: explicit ``output``, else the model's own file.

    Per-model files keep a re-run of one model from rewriting every other
    model's rows.  ``output=`` still wins so one-off experiment scripts can
    send their rows to a scratch CSV.
    """
    if cfg.output:
        return str(cfg.output)
    name = cfg.model.get("name")
    if not name:
        raise ValueError(
            "model config has no 'name', so no per-model results file can be "
            "derived; set output= explicitly or add a name to the model config."
        )
    return str(model_results_path(cfg.results_dir, name))


def _resolve_side_output_path(cfg: RunSettings, output_path: str, dir_key: str) -> str:
    """Return the CSV for a one-time measurement kind (profile/intrinsic_dim).

    Mirrors :func:`_resolve_output_path`'s "``output=`` wins" rule: when the
    caller set an explicit ``output``, every row type -- knn/linear/seg as
    well as profile/intrinsic_dim -- lands in that single file, unchanged
    from prior behavior. Only the default per-model routing path splits
    profile/intrinsic_dim into their own directory.
    """
    if cfg.output:
        return output_path
    name = cfg.model.get("name")
    if not name:
        raise ValueError(
            "model config has no 'name', so no per-model results file can be "
            "derived; set output= explicitly or add a name to the model config."
        )
    return str(model_results_path(getattr(cfg, dir_key), name))


def _merge_completed_metrics(
    base: dict[str, set[tuple[str, ...]]], other: dict[str, set[tuple[str, ...]]]
) -> None:
    """Union ``other``'s per-metric resume-key sets into ``base`` in place."""
    for metric_name, keys in other.items():
        base.setdefault(metric_name, set()).update(keys)


def main(cfg: RunSettings) -> None:
    """Run the benchmark pipeline for all configured datasets and models."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Coordinate location-encoder track: a distinct data/probe path (point
    # (lon, lat) -> label, ridge/KNN, k-fold CV) that does not touch the image
    # pipeline below. Dispatched by `mode=coord`.
    if cfg.mode == "coord":
        from torchgeo_bench.coordbench.run import run_coordbench

        run_coordbench(cfg)
        return

    from torchgeo.datasets import DatasetNotFoundError

    dataset_names = _expand_dataset_list(cfg.dataset.names)
    _validate_dataset_names(dataset_names)
    # dataset.names=all is a "run whatever is available" request: a dataset
    # missing its local data is expected and skipped with a warning. An
    # explicit list is a request for *those* datasets specifically, so
    # missing data there must fail clearly instead of silently shrinking
    # the run to whatever happened to be downloaded.
    requested_all_datasets = cfg.dataset.names == "all"
    device = resolve_configured_device(cfg.device)
    cfg.device = str(device)

    output_path = _resolve_output_path(cfg)
    profile_output_path = _resolve_side_output_path(cfg, output_path, "profile_results_dir")
    intrinsic_dim_output_path = _resolve_side_output_path(
        cfg, output_path, "intrinsic_dim_results_dir"
    )
    for path in {output_path, profile_output_path, intrinsic_dim_output_path}:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    all_rows: list[dict] = []
    id_out_rows: list[dict] = []
    profile_out_rows: list[dict] = []
    model_eval = cfg.model.get("eval")
    if model_eval is not None and model_eval.get("c_range") is not None:
        c_start, c_stop, c_num = model_eval["c_range"]
    else:
        c_start, c_stop, c_num = cfg.eval.c_range
    c_values = 10 ** np.linspace(float(c_start), float(c_stop), int(c_num))
    c_values_list = [float(v) for v in c_values.tolist()]

    key_cols = KEY_COLS
    completed_runs: set[tuple[str, ...]] = set()
    completed_metrics: dict[str, set[tuple[str, ...]]] = {}
    if cfg.resume and os.path.exists(output_path):
        completed_runs, completed_metrics = load_completed(output_path)
        logger.info(f"Resume mode: Found {len(completed_runs)} existing results in {output_path}")
        logger.info("Will skip already-computed (dataset, method, model, config) combinations.")
    if cfg.resume:
        # Profile/intrinsic_dim rows may live in their own per-model files
        # (default routing, no explicit output=); merge their completed
        # metrics in too so resume still skips already-measured work.
        for side_path in {profile_output_path, intrinsic_dim_output_path} - {output_path}:
            if os.path.exists(side_path):
                _, side_metrics = load_completed(side_path)
                _merge_completed_metrics(completed_metrics, side_metrics)

    # Selectable input-normalisation strategy; recorded in the CSV so
    # ablations across strategies are distinguishable.
    normalization = str(cfg.dataset.normalization)
    bands_value = _normalize_bands_value(cfg.dataset.bands)
    config_hash = _resume_config_hash(cfg)

    logger.info("Datasets: %d to process", len(dataset_names))
    for i, ds_name in enumerate(dataset_names, start=1):
        logger.info("[%d/%d] Dataset: %s", i, len(dataset_names), ds_name)
        # Every name in dataset_names was already validated against the
        # registry by _validate_dataset_names before this loop started.
        ds_cls = get_bench_dataset_class(ds_name)

        model_cfg = resolve_model_config(cfg.model, ds_name)

        effective_image_size = model_cfg.get("image_size", cfg.dataset.image_size)
        effective_interpolation = model_cfg.get("interpolation", cfg.dataset.interpolation)
        model_res = model_cfg.get("res")
        model_pool = model_cfg.get("pool")

        config_tuple = tuple(
            _canonical_key_cell(v)
            for v in (
                normalization,
                effective_image_size,
                effective_interpolation,
                cfg.dataset.partition,
                bands_value,
                ds_cls.num_classes,
                model_res,
                model_pool,
                config_hash,
            )
        )
        eval_cfg_merged = merge(cfg.eval, model_eval or {})
        knn_k = int(eval_cfg_merged.knn_k)
        seg_method = f"seg-{eval_cfg_merged.segmentation.head_type}"
        plan = _plan_dataset_run(
            cfg=cfg,
            ds_name=ds_name,
            ds_cls=ds_cls,
            knn_k=knn_k,
            seg_method=seg_method,
            config_tuple=config_tuple,
            completed_runs=completed_runs,
            completed_metrics=completed_metrics,
        )
        if plan.skip_dataset:
            if cfg.verbose:
                logger.info(f"[{ds_name}] Resume preflight: all requested work already complete")
            continue

        knn_device: str | None = None
        if ds_cls.task != "segmentation" and not plan.skip_knn:
            knn_device = resolve_knn_device(cfg.eval.knn_device, cfg.device)

        try:
            result = get_datasets(
                dataset_name=ds_name,
                partition_name=cfg.dataset.partition,
                batch_size=cfg.dataset.batch_size,
                num_workers=int(cfg.dataset.num_workers),
                return_val=True,
                image_size=effective_image_size,
                interpolation=effective_interpolation,
                bands=cfg.dataset.bands,
                time_steps=cfg.dataset.time_steps,
                pin_memory=(device.type == "cuda"),
            )
        except (FileNotFoundError, DatasetNotFoundError) as exc:
            if requested_all_datasets:
                logger.warning(f"Skipping dataset {ds_name} (data not found: {exc})")
                continue
            logger.error(
                f"Dataset {ds_name!r} was explicitly requested but its data was not "
                f"found: {exc}. Explicitly requested datasets must have their data "
                "available; pass dataset.names=all to skip unavailable datasets "
                "with a warning instead."
            )
            raise
        train_dataset, train_loader, val_loader, test_loader = result

        num_channels = train_dataset[0]["image"].shape[0]
        is_segmentation = ds_cls.task == "segmentation"
        num_classes = ds_cls.num_classes

        # Build the BandSpec list that matches the actual loaded channels.
        bench = ds_cls()
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

        # `bands` is passed as a kwarg (not via the config) so the BandSpec
        # dataclasses reach the constructor intact.
        instantiate_kwargs: dict = {
            "bands": bands_list,
            "normalization": normalization,
        }
        if model_cfg.get("mode") == "empirical":
            # Empirical RCF whitens against real patches, so it needs the dataset.
            instantiate_kwargs["dataset"] = train_dataset
        # Interpolation belongs to the loader rather than the model constructor.
        model_cfg.pop("interpolation", None)
        model: BenchModel = instantiate(model_cfg, **instantiate_kwargs)
        model.to(device).eval()

        common_meta = {
            "dataset": ds_name,
            "seed": cfg.seed,
            "model": model_cfg["_target_"],
            "name": model_cfg["name"],
            "normalization": normalization,
            "image_size": effective_image_size,
            "interpolation": effective_interpolation,
            "partition": cfg.dataset.partition,
            "bands": bands_value,
            "num_classes": num_classes,
            "config_hash": config_hash,
            "c_range_start": c_start,
            "c_range_stop": c_stop,
            "c_range_num": c_num,
            "merge_val": cfg.eval.merge_val,
            "bootstrap": cfg.eval.bootstrap,
            "res": model_res,
            "pool": model_pool,
        }

        if is_segmentation:
            seg_cfg_merged = eval_cfg_merged.segmentation
            save_viz = seg_cfg_merged.save_viz
            segmentation_meta = {**common_meta, "merge_val": False}
            metrics, feat_dim, best_lr, best_bs, preds = evaluate_segmentation(
                model,
                train_loader,
                val_loader,
                test_loader,
                eval_cfg_merged,
                num_classes,
                device,
                seed=cfg.seed,
                collect_preds=save_viz,
                verbose=cfg.verbose,
            )
            all_rows.append(
                metric_row(
                    segmentation_meta,
                    method=seg_method,
                    metric_name="mIoU",
                    metric_value=metrics.get("mIoU", float("nan")),
                    ci_lower=metrics.get("ci_lower", float("nan")),
                    ci_upper=metrics.get("ci_upper", float("nan")),
                    feature_dim=feat_dim,
                    n_counts={
                        "train": len(train_dataset),
                        "val": len(val_loader.dataset),
                        "test": len(test_loader.dataset),
                    },
                    best_lr=best_lr,
                    best_batch_size=best_bs,
                    fw_iou=metrics.get("fw_IoU"),
                    precision=metrics.get("precision"),
                    recall=metrics.get("recall"),
                    f1=metrics.get("f1"),
                )
            )
            if save_viz and preds is not None:
                from torchgeo_bench.segmentation_viz import (
                    collect_viz_inputs,
                    save_segmentation_viz,
                )

                rgb_indices = bench.rgb_indices or [0, 1, 2]
                test_imgs_t, test_gts_t = collect_viz_inputs(test_loader)
                # Never a real per-model settings field (see
                # SegmentationSettings): the ignore index always comes from
                # the criterion block, so this is the fixed torchmetrics
                # default, not a `.get(..., 255)` fallback that was ever hit.
                ignore_idx = 255
                n_viz = seg_cfg_merged.n_viz_samples
                viz_dir = seg_cfg_merged.viz_dir
                _class_names = list(getattr(train_dataset, "classes", None) or []) or None
                save_segmentation_viz(
                    out_dir=viz_dir,
                    model_name=model_cfg["name"],
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
            metric_name = plan.metric_name
            x_train, y_train = embed_split(model, train_loader, device, verbose=True, split="train")
            x_val, y_val = embed_split(model, val_loader, device, verbose=True, split="val")
            x_test, y_test = embed_split(model, test_loader, device, verbose=True, split="test")
            feature_dim = x_train.shape[1]
            n_counts = {"train": len(x_train), "val": len(x_val), "test": len(x_test)}

            cal_cfg = cfg.eval.calibration
            cal_n_bins_knn = cal_cfg.n_bins_knn
            cal_n_bins_linear = int(cal_cfg.n_bins_linear)
            cal_temp_scale = bool(cal_cfg.temp_scale)

            if not plan.skip_knn:
                assert knn_device is not None
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
                    metric_row(
                        common_meta,
                        method=f"knn{knn_k}",
                        metric_name=metric_name,
                        metric_value=knn_score,
                        ci_lower=knn_lo,
                        ci_upper=knn_hi,
                        feature_dim=feature_dim,
                        n_counts=n_counts,
                        ece=knn_cal["ece"],
                        rms_ce=knn_cal["rms_ce"],
                        mce=knn_cal["mce"],
                        calibration_n_bins=knn_n_bins,
                    )
                )

            if not plan.skip_linear:
                try:
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
                except LinearProbeDivergedError as exc:
                    # A handful of (backbone, dataset) pairings produce feature
                    # magnitudes the probe can't fit at any C in the sweep. That's
                    # a property of this one combination, not the rest of the
                    # benchmark run, so skip just this row rather than losing
                    # every other already-computed metric to an uncaught crash.
                    logger.warning(
                        f"[linear] model={common_meta.get('model')} "
                        f"dataset={common_meta.get('dataset')} bands={common_meta.get('bands')} "
                        f"norm={common_meta.get('normalization')}: skipping, no usable C found. "
                        f"Diagnostic: {exc}"
                    )
                else:
                    all_rows.append(
                        metric_row(
                            common_meta,
                            method="linear",
                            metric_name=metric_name,
                            metric_value=lin_score,
                            ci_lower=lin_lo,
                            ci_upper=lin_hi,
                            feature_dim=feature_dim,
                            n_counts=n_counts,
                            best_c=best_c,
                            ece=lin_cal["ece"],
                            rms_ce=lin_cal["rms_ce"],
                            mce=lin_cal["mce"],
                            ece_ts=lin_cal_ts["ece_ts"],
                            rms_ce_ts=lin_cal_ts["rms_ce_ts"],
                            mce_ts=lin_cal_ts["mce_ts"],
                            temperature=lin_cal_ts["temperature"],
                            calibration_n_bins=cal_n_bins_linear,
                        )
                    )
            if not plan.skip_id:
                id_cfg = cfg.eval.intrinsic_dim
                id_rows = evaluate_intrinsic_dim(
                    splits={"train": x_train, "val": x_val, "test": x_test},
                    estimators=list(id_cfg.estimators),
                    selected_splits=list(id_cfg.splits),
                    device=id_cfg.device or cfg.device,
                    max_samples=id_cfg.max_samples,
                    seed=cfg.seed,
                    common_meta=common_meta,
                    feature_dim=feature_dim,
                    n_counts=n_counts,
                    verbose=cfg.verbose,
                    only_metrics=plan.id_missing_metrics if cfg.resume else None,
                )
                if cfg.resume:
                    id_rows = _filter_completed_metric_rows(id_rows, completed_metrics, key_cols)
                id_out_rows.extend(id_rows)

            if not plan.skip_profile:
                profile_cfg = cfg.eval.profile
                cpu_cfg = profile_cfg.cpu_throughput
                profile_rows = evaluate_profile(
                    model=model,
                    sample_loader=train_loader,
                    device=device,
                    n_warmup=int(profile_cfg.n_warmup),
                    n_measure=int(profile_cfg.n_measure),
                    common_meta=common_meta,
                    feature_dim=feature_dim,
                    n_counts=n_counts,
                    cpu_throughput_enabled=bool(cpu_cfg.enabled),
                    cpu_batch_size=int(cpu_cfg.batch_size),
                    cpu_n_warmup=int(cpu_cfg.n_warmup),
                    cpu_n_measure=int(cpu_cfg.n_measure),
                    cpu_time_budget_s=float(cpu_cfg.time_budget_s),
                )
                if cfg.resume:
                    profile_rows = _filter_completed_metric_rows(
                        profile_rows, completed_metrics, key_cols
                    )
                profile_out_rows.extend(profile_rows)

        append_rows_atomic(output_path, all_rows)
        all_rows.clear()
        append_rows_atomic(intrinsic_dim_output_path, id_out_rows)
        id_out_rows.clear()
        append_rows_atomic(profile_output_path, profile_out_rows)
        profile_out_rows.clear()

    result_paths = ", ".join(sorted({output_path, intrinsic_dim_output_path, profile_output_path}))
    logger.info(f"Benchmark complete. Results appended to {result_paths}")
