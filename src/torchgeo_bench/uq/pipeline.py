"""Hydra entry point and execution pipeline for UQ benchmark runs."""

import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchgeo.datasets.errors import DatasetNotFoundError

from torchgeo_bench.datasets import get_bench_dataset_class, get_datasets, list_datasets
from torchgeo_bench.datasets.base import BandSpec
from torchgeo_bench.linear import LogisticRegression
from torchgeo_bench.main import append_rows_atomic
from torchgeo_bench.models.interface import BenchModel
from torchgeo_bench.uq.corruptions import SKIP_POISSON_GAUSSIAN, CorruptionTransform
from torchgeo_bench.uq.methods import (
    ConformalPredictor,
    DeepEnsemble,
    LaplaceProbe,
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
    sharpness,
)
from torchgeo_bench.uq.splits import stratified_cal_split
from torchgeo_bench.uq.traces import (
    build_config_hash,
    build_conformal_trace_frame,
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

_RESUME_KEY_COLS: tuple[str, ...] = (
    "model",
    "name",
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
)

_CLOUD_PATTERN_MODE_MAP: dict[str, str] = {
    "fixed_across_severity": "fixed",
    "independent_per_severity": "independent",
    "fixed": "fixed",
    "independent": "independent",
}


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


def _expected_metrics(uq_method: str) -> set[str]:
    """Return expected metric names for a UQ method.

    Args:
        uq_method: UQ method name.

    Returns:
        Set of metric names required to mark a resume block as complete.
    """
    if uq_method == "conformal":
        return {"empirical_coverage", "mean_set_size", "raw_aurc", "eaurc", "selective_acc_90"}
    return {
        "ece",
        "nll",
        "brier",
        "predictive_entropy",
        "normalized_predictive_entropy",
        "max_probability",
        "sharpness",
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
            df[col] = ""
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


def _lookup_best_c(prior_results: pd.DataFrame, row_filter: dict[str, Any]) -> float | None:
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

        subset = subset[subset["method"].fillna("").astype(str) == "linear"]
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

    rows: list[dict[str, Any]] = []
    trace_df = None
    trace_link: dict[str, str] = {}
    trace_path: Path | None = None
    trace_block_key: str | None = None
    if method_name == "conformal":
        point_preds, pred_sets = method.predict_sets(X_test, alpha=conformal_alpha)
        set_sizes = pred_sets.sum(axis=1).astype(np.float64)
        conf = 1.0 / np.maximum(set_sizes, 1.0)
        metrics = {
            "empirical_coverage": empirical_coverage(pred_sets, y_test),
            "mean_set_size": mean_set_size(pred_sets),
            "raw_aurc": raw_aurc(conf, point_preds, y_test),
            "eaurc": excess_aurc(conf, point_preds, y_test),
            "selective_acc_90": selective_accuracy(conf, point_preds, y_test, coverage=0.9),
        }
        if trace_ctx and bool(trace_ctx.get("include_conformal", False)):
            trace_block_key = build_trace_block_key(
                run_id=str(trace_ctx["run_id"]),
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
        conf = probs.max(axis=1)
        metrics = {
            "ece": ece(probs, y_test, n_bins=ece_bins),
            "nll": nll(probs, y_test),
            "brier": brier_score(probs, y_test),
            "predictive_entropy": predictive_entropy(probs),
            "normalized_predictive_entropy": normalized_predictive_entropy(probs),
            "max_probability": max_probability(probs),
            "sharpness": sharpness(probs),
            "raw_aurc": raw_aurc(conf, y_pred, y_test),
            "eaurc": excess_aurc(conf, y_pred, y_test),
            "selective_acc_90": selective_accuracy(conf, y_pred, y_test, coverage=0.9),
        }
        if trace_ctx:
            trace_block_key = build_trace_block_key(
                run_id=str(trace_ctx["run_id"]),
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

    prior_results = pd.read_csv(str(cfg.uq.prior_results)) if os.path.exists(str(cfg.uq.prior_results)) else None
    if prior_results is None:
        logger.warning("Prior results file missing at %s; skipping all datasets.", cfg.uq.prior_results)
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
        model: BenchModel = instantiate(cfg.model, **instantiate_kwargs)
        model.to(device).eval()

        X_train, y_train = extract_features(model, train_loader, device, transforms=None, verbose=cfg.verbose)
        X_val, y_val = extract_features(model, val_loader, device, transforms=None, verbose=cfg.verbose)

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

        best_c = _lookup_best_c(
            prior_results,
            {
                "model": cfg.model._target_,
                "name": cfg.model.name,
                "dataset": dataset_name,
                "partition": cfg.dataset.partition,
                "bands": bands_value,
            },
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

        methods: dict[str, Any] = {}
        if "uncalibrated" in cfg.uq.methods:
            methods["uncalibrated"] = Uncalibrated(probe)
        if "temp_scaling" in cfg.uq.methods:
            ts = TemperatureScaling(probe)
            ts.fit(X_cal, y_cal)
            methods["temp_scaling"] = ts
        if "deep_ensemble" in cfg.uq.methods:
            de = DeepEnsemble(n=int(cfg.uq.n_ensemble))
            de.fit(X_final_train, y_final_train, best_c=best_c, seed=cfg.seed)
            methods["deep_ensemble"] = de
        if "laplace" in cfg.uq.methods:
            try:
                la = LaplaceProbe(probe, batch_size=int(cfg.uq.laplace_batch_size))
                la.fit(X_final_train, y_final_train)
                methods["laplace"] = la
            except ModuleNotFoundError as exc:
                logger.warning("Skipping laplace for dataset %s: %s", dataset_name, exc)
        if "conformal" in cfg.uq.methods:
            try:
                conf = ConformalPredictor(probe)
                conf.fit(X_cal, y_cal)
                methods["conformal"] = conf
            except ModuleNotFoundError as exc:
                logger.warning("Skipping conformal for dataset %s: %s", dataset_name, exc)

        common_meta = {
            "model": str(cfg.model._target_),
            "name": str(cfg.model.name),
            "backbone": str(cfg.model.name),
            "dataset": dataset_name,
            "normalization": normalization,
            "image_size": getattr(cfg.dataset, "image_size", None),
            "interpolation": getattr(cfg.dataset, "interpolation", "bilinear"),
            "partition": str(cfg.dataset.partition),
            "bands": bands_value,
            "seed": int(cfg.seed),
        }

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

                for method_name, method in methods.items():
                    key = (
                        str(cfg.model._target_),
                        str(cfg.model.name),
                        str(cfg.seed),
                        dataset_name,
                        normalization,
                        str(getattr(cfg.dataset, "image_size", None)),
                        str(getattr(cfg.dataset, "interpolation", "bilinear")),
                        str(cfg.dataset.partition),
                        bands_value,
                        method_name,
                        str(corruption_type),
                        str(int(severity)),
                    )
                    if cfg.resume and key in completed:
                        if trace_ctx:
                            trace_block_key = build_trace_block_key(
                                run_id=str(trace_ctx["run_id"]),
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
                        conformal_alpha=float(cfg.uq.conformal_alpha),
                        n_cal=len(X_cal),
                        n_train=len(X_final_train),
                        feature_dim=int(X_final_train.shape[1]),
                        best_c=float(best_c),
                        seed=int(cfg.seed),
                        X_test=X_test,
                        y_test=y_test,
                        sample_ids=sample_ids,
                        cloud_pattern_mode=cloud_pattern_mode,
                        trace_ctx=trace_ctx,
                    )

    logger.info("UQ benchmark complete. Results appended to %s", output_path)


if __name__ == "__main__":  # pragma: no cover
    main()  # type: ignore[misc]
