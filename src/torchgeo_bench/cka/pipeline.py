"""Hydra entry point and execution pipeline for CKA benchmark runs."""

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
from omegaconf import DictConfig
from torchgeo.datasets.errors import DatasetNotFoundError

from torchgeo_bench.cka.hooks import HookCollector
from torchgeo_bench.cka.metrics import (
    cosine_drift,
    linear_cka,
    participation_ratio,
    split_half_cka,
    track_b_summary,
)
from torchgeo_bench.datasets import get_bench_dataset_class, get_datasets, list_datasets
from torchgeo_bench.datasets.base import BandSpec
from torchgeo_bench.linear import LogisticRegression
from torchgeo_bench.main import append_rows_atomic
from torchgeo_bench.models.interface import BenchModel
from torchgeo_bench.uq.corruptions import SKIP_POISSON_GAUSSIAN, CorruptionTransform
from torchgeo_bench.uq.splits import stratified_cal_split
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
    "corruption_type",
    "severity",
)

_CLOUD_PATTERN_MODE_MAP: dict[str, str] = {
    "fixed_across_severity": "fixed",
    "independent_per_severity": "independent",
    "fixed": "fixed",
    "independent": "independent",
}


def _is_cka_classification_dataset(ds_cls: type) -> bool:
    """Return whether a dataset class is in scope for CKA runs."""
    return ds_cls.task == "classification" and not bool(getattr(ds_cls, "multilabel", False))


def _expand_dataset_list(names: str | Sequence[str]) -> list[str]:
    """Normalize dataset selectors into an explicit dataset name list."""
    if isinstance(names, str):
        if names == "all":
            return list_datasets()
        return [name.strip() for name in names.split(",") if name.strip()]
    return list(names)


def _normalize_bands_value(bands: object) -> str:
    """Normalize a band selector to a stable CSV-friendly string value."""
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
    """Map pipeline cloud pattern modes to transform-compatible mode values."""
    try:
        return _CLOUD_PATTERN_MODE_MAP[cloud_pattern_mode]
    except KeyError as exc:
        raise ValueError(
            "cka.cloud_pattern_mode must be one of "
            f"{sorted(_CLOUD_PATTERN_MODE_MAP)}."
        ) from exc


def _build_cka_resume_set(
    csv_path: str,
    layer_counts_by_name: dict[str, int],
) -> set[tuple[str, ...]]:
    """Build completed CKA condition keys from an output CSV.

    A key is complete when it has exactly the expected set of layer indices
    ``{0, ..., n_layers - 1}`` for that model name.
    """
    if not os.path.exists(csv_path):
        return set()

    df = pd.read_csv(csv_path)
    for col in (*_RESUME_KEY_COLS, "layer_index"):
        if col not in df.columns:
            df[col] = ""
    df = df.fillna("")

    completed: set[tuple[str, ...]] = set()
    for key_vals, group in df.groupby(list(_RESUME_KEY_COLS), dropna=False):
        key = tuple(str(v) for v in key_vals)
        model_name = str(group["name"].iloc[0])
        expected_n_layers = int(layer_counts_by_name.get(model_name, 4))
        expected = set(range(expected_n_layers))
        present = {
            int(x)
            for x in group["layer_index"].tolist()
            if str(x) != "" and str(x) != "nan"
        }
        if present == expected:
            completed.add(key)
    return completed


def _lookup_best_c(prior_results: pd.DataFrame, row_filter: dict[str, Any]) -> float | None:
    """Resolve ``best_c`` from prior linear-probe results."""
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


def _build_resume_key(common_meta: dict[str, Any], corruption_type: str, severity: int) -> tuple[str, ...]:
    resume_meta = {
        **common_meta,
        "corruption_type": corruption_type,
        "severity": int(severity),
    }
    return tuple(str(resume_meta[col]) for col in _RESUME_KEY_COLS)


def _purge_resume_key_rows(csv_path: str, resume_key: tuple[str, ...]) -> None:
    """Remove all rows for one resume key from an existing CSV."""
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    if df.empty:
        return

    match_mask = np.ones(len(df), dtype=bool)
    for idx, col in enumerate(_RESUME_KEY_COLS):
        col_values = df[col].fillna("").astype(str) if col in df.columns else pd.Series("", index=df.index)
        match_mask &= col_values == str(resume_key[idx])

    if not match_mask.any():
        return
    df_kept = df[~match_mask]
    df_kept.to_csv(csv_path, index=False)


def _validate_collected_activations(
    acts: dict[str, np.ndarray],
    hook_paths: list[str],
    expected_n_samples: int,
    condition_label: str,
) -> None:
    """Validate that all required hook paths produced non-empty aligned arrays."""
    for path in hook_paths:
        if path not in acts:
            raise ValueError(f"Missing collected activations for path {path!r} in {condition_label}.")
        arr = acts[path]
        if arr.ndim != 2:
            raise ValueError(f"Collected activations for path {path!r} must be 2D, got shape {arr.shape}.")
        if arr.shape[0] == 0:
            raise ValueError(f"Collected activations for path {path!r} are empty in {condition_label}.")
        if arr.shape[0] != expected_n_samples:
            raise ValueError(
                f"Collected activations for path {path!r} in {condition_label} have "
                f"{arr.shape[0]} samples, expected {expected_n_samples}."
            )


def _write_sample_parquet(
    *,
    traces_root: str,
    model_name: str,
    dataset_name: str,
    corruption_type: str,
    severity: int,
    X_clean_final: np.ndarray,
    X_corr_final: np.ndarray,
    probe: LogisticRegression,
    y_true: np.ndarray,
) -> None:
    """Append per-sample Track B traces for one condition."""
    probs = probe.predict_proba(torch.from_numpy(X_corr_final).to(dtype=torch.float32))
    y_pred = np.argmax(probs, axis=1).astype(np.int16)
    confidence = probs.max(axis=1).astype(np.float32)
    y_true_arr = np.asarray(y_true, dtype=np.int16)
    correct = (y_pred == y_true_arr).astype(bool)
    drift = np.linalg.norm(X_corr_final - X_clean_final, axis=1).astype(np.float32)

    frame = pd.DataFrame(
        {
            "corruption_type": [str(corruption_type)] * len(drift),
            "severity": np.full(len(drift), int(severity), dtype=np.int16),
            "sample_idx": np.arange(len(drift), dtype=np.int32),
            "drift": drift,
            "confidence": confidence,
            "correct": correct,
            "y_true": y_true_arr,
            "y_pred": y_pred,
        }
    )

    out_path = Path(traces_root) / str(model_name) / f"{dataset_name}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        prev = pd.read_parquet(out_path)
        frame = pd.concat([prev, frame], ignore_index=True)
    frame.to_parquet(out_path, compression="zstd", index=False)


def _build_row(
    *,
    common_meta: dict[str, Any],
    layer_name: str,
    layer_index: int,
    corruption_type: str,
    severity: int,
    cka_value: float,
    cosine_value: float,
    participation_value: float,
    clean_participation_value: float,
    n_test: int,
    feature_dim: int,
    best_c: float,
) -> dict[str, Any]:
    return {
        **common_meta,
        "layer_name": layer_name,
        "layer_index": int(layer_index),
        "corruption_type": str(corruption_type),
        "severity": int(severity),
        "cka": float(cka_value),
        "cosine_drift": float(cosine_value),
        "participation_ratio": float(participation_value),
        "clean_participation_ratio": float(clean_participation_value),
        "n_test": int(n_test),
        "feature_dim": int(feature_dim),
        "best_c": float(best_c),
        "spearman_drift_confidence": float("nan"),
        "spearman_drift_correctness": float("nan"),
        "frac_overconfident_high_drift": float("nan"),
    }


def run_cka(cfg: DictConfig) -> None:
    """Run the Hydra-configured CKA pipeline."""
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    output_path = str(cfg.cka.output)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if bool(getattr(cfg.cka, "write_sample_traces", True)):
        os.makedirs(str(cfg.cka.traces_root), exist_ok=True)

    if not os.path.exists(str(cfg.cka.prior_results)):
        logger.warning(
            "Prior results file missing at %s; skipping all datasets.",
            cfg.cka.prior_results,
        )
        return
    prior_results = pd.read_csv(str(cfg.cka.prior_results))

    layer_counts_by_name = {
        str(model_name): len(paths)
        for model_name, paths in cfg.cka.layers.items()
    }
    completed = _build_cka_resume_set(output_path, layer_counts_by_name) if bool(cfg.resume) else set()
    dataset_names = _expand_dataset_list(cfg.dataset.names)
    device = torch.device(str(cfg.device))
    bands_value = _normalize_bands_value(getattr(cfg.dataset, "bands", "rgb"))
    normalization = str(getattr(cfg.dataset, "normalization", "bandspec_zscore"))
    cloud_pattern_mode = _normalize_cloud_pattern_mode(str(cfg.cka.cloud_pattern_mode))

    for dataset_name in dataset_names:
        try:
            ds_cls = get_bench_dataset_class(dataset_name)
        except KeyError:
            logger.warning("Skipping unknown dataset %s.", dataset_name)
            continue
        if not _is_cka_classification_dataset(ds_cls):
            if ds_cls.task != "classification":
                logger.info("Skipping segmentation dataset %s in CKA pipeline.", dataset_name)
            else:
                logger.info("Skipping multi-label dataset %s in CKA pipeline.", dataset_name)
            continue

        hook_paths = list(cfg.cka.layers.get(str(cfg.model.name), []))
        if not hook_paths:
            logger.warning("Skipping model %s: no CKA layers configured.", cfg.model.name)
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
        bands_resolved: tuple[str, ...] | None
        if cfg.dataset.bands == "rgb":
            bands_resolved = tuple(bench.rgb_bands)
        elif cfg.dataset.bands in ("all", None):
            bands_resolved = None
        else:
            bands_resolved = tuple(cfg.dataset.bands)
        band_specs: list[BandSpec] = bench.select_band_specs(bands_resolved)

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
        cal_size = int(cfg.cka.cal_size)
        if cal_size >= len(X_val):
            logger.warning(
                "Skipping dataset %s: cka.cal_size=%d >= val size=%d.",
                dataset_name,
                cal_size,
                len(X_val),
            )
            continue

        _, _, X_val_rem, y_val_rem = stratified_cal_split(X_val, y_val, cal_size, int(cfg.seed))
        X_final_train = np.concatenate([X_train, X_val_rem], axis=0)
        y_final_train = np.concatenate([y_train, y_val_rem], axis=0)

        best_c = _lookup_best_c(
            prior_results=prior_results,
            row_filter={
                "dataset": dataset_name,
                "name": str(cfg.model.name),
                "partition": str(cfg.dataset.partition),
                "bands": bands_value,
                "normalization": normalization,
                "image_size": str(getattr(cfg.dataset, "image_size", "")),
                "interpolation": str(getattr(cfg.dataset, "interpolation", "")),
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

        probe = LogisticRegression(C=best_c, random_state=int(cfg.seed), device=device, verbose=bool(cfg.verbose))
        probe.fit(
            torch.from_numpy(X_final_train).to(dtype=torch.float32),
            torch.from_numpy(y_final_train).to(dtype=torch.long),
        )

        with HookCollector(model, hook_paths) as collector:
            X_test_clean, y_test = extract_features(
                model,
                test_loader,
                device,
                transforms=None,
                verbose=cfg.verbose,
            )
            clean_acts = collector.collect()

        _validate_collected_activations(
            clean_acts,
            hook_paths,
            expected_n_samples=int(len(y_test)),
            condition_label="clean",
        )
        clean_pr = {path: participation_ratio(clean_acts[path]) for path in hook_paths}

        common_meta = {
            "model": str(cfg.model._target_),
            "name": str(cfg.model.name),
            "dataset": str(dataset_name),
            "normalization": str(normalization),
            "image_size": getattr(cfg.dataset, "image_size", None),
            "interpolation": str(getattr(cfg.dataset, "interpolation", "bilinear")),
            "partition": str(cfg.dataset.partition),
            "bands": str(bands_value),
            "seed": int(cfg.seed),
        }

        clean_key = _build_resume_key(common_meta, "clean", 0)
        if not (bool(cfg.resume) and clean_key in completed):
            if bool(cfg.resume):
                _purge_resume_key_rows(output_path, clean_key)
            clean_rows: list[dict[str, Any]] = []
            for layer_index, path in enumerate(hook_paths):
                cka_value, cosine_value = split_half_cka(clean_acts[path], seed=int(cfg.seed))
                row = _build_row(
                    common_meta=common_meta,
                    layer_name=path,
                    layer_index=layer_index,
                    corruption_type="clean",
                    severity=0,
                    cka_value=cka_value,
                    cosine_value=cosine_value,
                    participation_value=clean_pr[path],
                    clean_participation_value=clean_pr[path],
                    n_test=int(len(y_test)),
                    feature_dim=int(clean_acts[path].shape[1]),
                    best_c=float(best_c),
                )
                if layer_index == len(hook_paths) - 1:
                    row["spearman_drift_confidence"] = float("nan")
                    row["spearman_drift_correctness"] = float("nan")
                    row["frac_overconfident_high_drift"] = 0.0
                clean_rows.append(row)
            append_rows_atomic(output_path, clean_rows)

        X_clean_final = clean_acts[hook_paths[-1]]
        for corruption_type in cfg.cka.corruptions:
            corruption_name = str(corruption_type)
            if corruption_name == "poisson_gaussian" and dataset_name in SKIP_POISSON_GAUSSIAN:
                logger.info(
                    "Skipping corruption=%s for dataset=%s due to skip list.",
                    corruption_name,
                    dataset_name,
                )
                continue
            for severity in cfg.cka.corruption_severities:
                severity_int = int(severity)
                resume_key = _build_resume_key(common_meta, corruption_name, severity_int)
                if bool(cfg.resume) and resume_key in completed:
                    logger.info(
                        "Skipping completed CKA condition: dataset=%s corruption=%s severity=%d.",
                        dataset_name,
                        corruption_name,
                        severity_int,
                    )
                    continue
                if bool(cfg.resume):
                    _purge_resume_key_rows(output_path, resume_key)

                transform = CorruptionTransform(
                    corruption_type=corruption_name,
                    severity=severity_int,
                    seed=int(cfg.seed),
                    band_specs=band_specs,
                    dataset_name=dataset_name,
                    cloud_pattern_mode=cloud_pattern_mode,
                )
                with HookCollector(model, hook_paths) as collector:
                    _, y_test_corr = extract_features(
                        model,
                        test_loader,
                        device,
                        transforms=transform,
                        verbose=cfg.verbose,
                    )
                    corr_acts = collector.collect()

                _validate_collected_activations(
                    corr_acts,
                    hook_paths,
                    expected_n_samples=int(len(y_test_corr)),
                    condition_label=f"{corruption_name}:{severity_int}",
                )

                rows: list[dict[str, Any]] = []
                for layer_index, path in enumerate(hook_paths):
                    row = _build_row(
                        common_meta=common_meta,
                        layer_name=path,
                        layer_index=layer_index,
                        corruption_type=corruption_name,
                        severity=severity_int,
                        cka_value=linear_cka(clean_acts[path], corr_acts[path]),
                        cosine_value=cosine_drift(clean_acts[path], corr_acts[path]),
                        participation_value=participation_ratio(corr_acts[path]),
                        clean_participation_value=clean_pr[path],
                        n_test=int(len(y_test_corr)),
                        feature_dim=int(corr_acts[path].shape[1]),
                        best_c=float(best_c),
                    )
                    if layer_index == len(hook_paths) - 1:
                        summary = track_b_summary(
                            X_clean=X_clean_final,
                            X_corrupted=corr_acts[path],
                            probe=probe,
                            y_true=y_test_corr,
                            confidence_threshold=float(cfg.cka.confidence_threshold),
                        )
                        row.update(summary)
                        if bool(getattr(cfg.cka, "write_sample_traces", True)):
                            _write_sample_parquet(
                                traces_root=str(cfg.cka.traces_root),
                                model_name=str(cfg.model.name),
                                dataset_name=str(dataset_name),
                                corruption_type=corruption_name,
                                severity=severity_int,
                                X_clean_final=X_clean_final,
                                X_corr_final=corr_acts[path],
                                probe=probe,
                                y_true=y_test_corr,
                            )
                    rows.append(row)
                append_rows_atomic(output_path, rows)


@hydra.main(config_path="../conf", config_name="cka_config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Hydra CLI entry point for CKA analysis."""
    run_cka(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
