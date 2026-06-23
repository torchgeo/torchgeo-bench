"""Sample-size sweep pipeline: calibration vs training-set fraction."""

import copy
import hashlib
import json
import logging
import math
import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import hydra
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit

from torchgeo_bench.calibration import compute_calibration_metrics
from torchgeo_bench.datasets import get_bench_dataset_class, get_datasets
from torchgeo_bench.linear import LogisticRegression
from torchgeo_bench.main import _resolve_segmentation_ignore_index, append_rows_atomic, embed_split
from torchgeo_bench.segmentation_probe import CachedFeaturesDataset, SegmentationProbe
from torchgeo_bench.segmentation_task import SegmentationSolver
from torchgeo_bench.uq.metrics import (
    brier_score,
    excess_aurc,
    nll,
    raw_aurc,
    selective_accuracy,
    signed_ece,
)
from torchgeo_bench.utils import extract_features

warnings.filterwarnings("ignore", message="Dataset has no geotransform", category=UserWarning)

logger = logging.getLogger(__name__)

_CLS_METRICS = (
    "accuracy",
    "ece",
    "signed_ece",
    "nll",
    "brier",
    "mean_confidence",
    "overconfidence_gap",
    "mean_wrong_confidence",
    "high_conf_wrong_rate_090",
    "selective_acc_90",
    "raw_aurc",
    "eaurc",
)
_SEG_METRICS = ("miou", "pixel_ece")


@dataclass(frozen=True)
class SampleSizeImageStatsBlockStatus:
    """Completion status for one sample-size image-stats block."""

    row_count: int | None
    expected_count: int
    is_complete: bool


def _compute_epochs(n_sub: int, batch_size: int, target: int, floor: int = 5) -> int:
    """Epochs needed to reach approximately `target` gradient steps."""
    steps_per_epoch = max(1, n_sub // batch_size)
    return max(floor, math.ceil(target / steps_per_epoch))


def _subsample_cache(cache: CachedFeaturesDataset, indices: np.ndarray) -> CachedFeaturesDataset:
    idx_t = torch.from_numpy(indices.astype(np.int64))
    return CachedFeaturesDataset(
        [t[idx_t] for t in cache.layer_tensors],
        cache.masks[idx_t],
    )


def _load_completed(path: str) -> frozenset[tuple]:
    if not os.path.exists(path):
        return frozenset()
    try:
        df = pd.read_csv(path)
        return frozenset(
            zip(
                df["model"],
                df["dataset"],
                df["train_fraction"],
                df["seed"],
                df["metric_name"],
                strict=False,
            )
        )
    except Exception:
        return frozenset()


def _safe_part(value: object) -> str:
    text = str(value)
    return (
        text.replace("/", "_")
        .replace("\\", "_")
        .replace("=", "-")
        .replace(":", "-")
        .replace(" ", "_")
    )


def _fraction_token(value: float) -> str:
    return format(float(value), ".8g")


def _build_image_stats_cfg(cfg: DictConfig) -> dict[str, Any]:
    raw = cfg.sample_size.get("image_stats", {}) or {}
    image_stats_cfg = {
        "enabled": bool(raw.get("enabled", True)),
        "root": str(raw.get("root", "results/sample_size_image_stats")),
        "format": str(raw.get("format", "parquet")),
        "overwrite": bool(raw.get("overwrite", False)),
        "compression": str(raw.get("compression", "zstd")),
    }
    if image_stats_cfg["format"] != "parquet":
        raise ValueError(
            "sample_size.image_stats.format must be 'parquet'; "
            f"got {image_stats_cfg['format']!r}."
        )
    return image_stats_cfg


def _image_stats_block_key(block_meta: dict[str, Any]) -> str:
    payload = json.dumps(block_meta, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _image_stats_block_path(
    *,
    root: str,
    task: str,
    model: str,
    dataset: str,
    train_fraction: float,
    seed: int,
    block_key: str,
) -> Path:
    return (
        Path(root)
        / f"task={_safe_part(task)}"
        / f"model={_safe_part(model)}"
        / f"dataset={_safe_part(dataset)}"
        / f"train_fraction={_fraction_token(train_fraction)}"
        / f"seed={int(seed)}"
        / f"block_key={block_key}.parquet"
    )


def _image_stats_block_status(
    path: Path,
    *,
    expected_count: int,
) -> SampleSizeImageStatsBlockStatus:
    if not path.exists():
        return SampleSizeImageStatsBlockStatus(None, expected_count, False)
    row_count = int(pq.ParquetFile(path).metadata.num_rows)
    return SampleSizeImageStatsBlockStatus(
        row_count=row_count,
        expected_count=expected_count,
        is_complete=row_count == expected_count,
    )


def _write_image_stats_block_atomic(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    expected_count: int,
    overwrite: bool,
    resume: bool,
    compression: str,
) -> bool:
    """Write one sample-size image-stats parquet block atomically."""
    if not rows:
        return False
    if len(rows) != expected_count:
        raise ValueError(
            f"Expected {expected_count} image-stats rows, got {len(rows)} for block at {path}."
        )

    status = _image_stats_block_status(path, expected_count=expected_count)
    if resume and not overwrite and status.is_complete:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}.{uuid4().hex}")
    pd.DataFrame(rows).to_parquet(tmp_path, index=False, compression=compression)
    os.replace(tmp_path, path)
    return True


def _embed_test_split_with_ids(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    verbose: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Extract test embeddings, labels, and optional sample IDs."""
    return extract_features(
        model,
        dataloader,
        device,
        transforms=None,
        verbose=verbose,
        return_sample_ids=True,
    )


def _summary_block_complete(
    *,
    completed: frozenset[tuple],
    model_name: str,
    dataset_name: str,
    fraction: float,
    seed: int,
    metric_names: tuple[str, ...],
) -> bool:
    return all(
        (model_name, dataset_name, fraction, seed, metric) in completed for metric in metric_names
    )


def _merge_image_stats_rows(
    block_meta: dict[str, Any],
    image_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach block-level provenance columns to per-image rows."""
    return [{**block_meta, **row} for row in image_rows]


def _compute_cls_image_stats_rows(
    *,
    y_true: np.ndarray,
    probs: np.ndarray,
    sample_ids: np.ndarray | None,
) -> list[dict[str, Any]]:
    """Build one lean prediction-stats row per classification test image."""
    if probs.ndim != 2:
        raise ValueError(f"probs must be 2D, got shape {probs.shape}")
    if y_true.ndim != 1:
        raise ValueError(f"y_true must be 1D, got shape {y_true.shape}")
    if probs.shape[0] != y_true.shape[0]:
        raise ValueError("probs and y_true must have equal first dimension")
    if sample_ids is not None and sample_ids.shape[0] != y_true.shape[0]:
        raise ValueError("sample_ids and y_true must have equal first dimension")

    n_samples = int(y_true.shape[0])
    preds = probs.argmax(axis=1).astype(np.int64)
    confidence = probs.max(axis=1).astype(np.float64)
    if probs.shape[1] > 1:
        top2 = np.partition(probs, kth=probs.shape[1] - 2, axis=1)[:, -2]
    else:
        top2 = np.zeros(n_samples, dtype=np.float64)
    margin = (confidence - top2).astype(np.float64)
    clipped = np.clip(probs, 1e-12, 1.0)
    entropy = (-clipped * np.log(clipped)).sum(axis=1).astype(np.float64)
    if probs.shape[1] > 1:
        normalized_entropy = (entropy / math.log(float(probs.shape[1]))).astype(np.float64)
    else:
        normalized_entropy = np.zeros(n_samples, dtype=np.float64)
    nll_per_sample = (-np.log(clipped[np.arange(n_samples), y_true])).astype(np.float64)

    rows: list[dict[str, Any]] = []
    for image_index in range(n_samples):
        sample_id: str | None = None
        if sample_ids is not None:
            raw_value = str(sample_ids[image_index]).strip()
            sample_id = raw_value or None
        rows.append(
            {
                "image_index": image_index,
                "sample_id": sample_id,
                "y_true": int(y_true[image_index]),
                "y_pred": int(preds[image_index]),
                "correct": bool(preds[image_index] == y_true[image_index]),
                "confidence": float(confidence[image_index]),
                "margin": float(margin[image_index]),
                "entropy": float(entropy[image_index]),
                "normalized_entropy": float(normalized_entropy[image_index]),
                "nll": float(nll_per_sample[image_index]),
            }
        )
    return rows


def _cls_image_stats_block_meta(
    *,
    model_name: str,
    model_target: str,
    dataset_name: str,
    partition: str,
    bands: str,
    normalization: str,
    image_size: int | None,
    interpolation: str,
    train_fraction: float,
    seed: int,
    n_train_full: int,
    n_train_used: int,
    n_val: int,
    n_test: int,
) -> dict[str, Any]:
    return {
        "task": "classification",
        "model": model_name,
        "model_target": model_target,
        "dataset": dataset_name,
        "partition": partition,
        "bands": bands,
        "normalization": normalization,
        "image_size": image_size,
        "interpolation": interpolation,
        "train_fraction": float(train_fraction),
        "seed": int(seed),
        "n_train_full": int(n_train_full),
        "n_train_used": int(n_train_used),
        "n_val": int(n_val),
        "n_test": int(n_test),
    }


def _seg_image_stats_block_meta(
    *,
    model_name: str,
    model_target: str,
    dataset_name: str,
    partition: str,
    bands: str,
    normalization: str,
    image_size: int | None,
    interpolation: str,
    train_fraction: float,
    seed: int,
    n_train_full: int,
    n_train_used: int,
    n_val: int,
    n_test: int,
    seg_cfg: DictConfig,
    ignore_index: int,
) -> dict[str, Any]:
    return {
        "task": "segmentation",
        "model": model_name,
        "model_target": model_target,
        "dataset": dataset_name,
        "partition": partition,
        "bands": bands,
        "normalization": normalization,
        "image_size": image_size,
        "interpolation": interpolation,
        "train_fraction": float(train_fraction),
        "seed": int(seed),
        "n_train_full": int(n_train_full),
        "n_train_used": int(n_train_used),
        "n_val": int(n_val),
        "n_test": int(n_test),
        "seg_head_type": str(seg_cfg.head_type),
        "seg_layers": ",".join(str(layer) for layer in seg_cfg.layers),
        "seg_lr": float(seg_cfg.lr),
        "seg_batch_size": int(seg_cfg.get("batch_size", 64)),
        "seg_lr_scheduler": str(seg_cfg.get("lr_scheduler", "cosine")),
        "seg_ignore_index": int(ignore_index),
    }


def _compute_cls_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    *,
    n_bins_ece: int,
) -> dict[str, float]:
    """Compute aggregate classification quality and overconfidence metrics."""
    preds = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    wrong = preds != y_true
    acc = float(accuracy_score(y_true, preds))
    cal = compute_calibration_metrics(y_true, probs, multi_label=False, n_bins=n_bins_ece)

    return {
        "accuracy": acc,
        "ece": float(cal["ece"]),
        "signed_ece": float(signed_ece(probs, y_true, n_bins=n_bins_ece)),
        "nll": float(nll(probs, y_true)),
        "brier": float(brier_score(probs, y_true)),
        "mean_confidence": float(conf.mean()),
        "overconfidence_gap": float(conf.mean() - acc),
        "mean_wrong_confidence": float(conf[wrong].mean()) if np.any(wrong) else float("nan"),
        "high_conf_wrong_rate_090": float(np.mean(wrong & (conf >= 0.9))),
        "selective_acc_90": float(selective_accuracy(conf, preds, y_true, coverage=0.9)),
        "raw_aurc": float(raw_aurc(conf, preds, y_true)),
        "eaurc": float(excess_aurc(conf, preds, y_true)),
    }


def _cls_sweep(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    fractions: list[float],
    seeds_cls: int,
    c_values: list[float],
    n_bins_ece: int,
    model_name: str,
    model_target: str,
    dataset_name: str,
    partition: str,
    bands: str,
    normalization: str,
    image_size: int | None,
    interpolation: str,
    device: torch.device,
    test_sample_ids: np.ndarray | None,
    completed: frozenset[tuple],
    image_stats_cfg: dict[str, Any],
) -> list[dict]:
    rows: list[dict] = []
    n_train_full = len(X_train)
    n_val = len(X_val)
    n_test = len(X_test)

    X_val_t = torch.from_numpy(X_val)
    X_test_t = torch.from_numpy(X_test)

    for fraction in fractions:
        for seed in range(seeds_cls):
            metrics_complete = _summary_block_complete(
                completed=completed,
                model_name=model_name,
                dataset_name=dataset_name,
                fraction=fraction,
                seed=seed,
                metric_names=_CLS_METRICS,
            )
            image_stats_meta = _cls_image_stats_block_meta(
                model_name=model_name,
                model_target=model_target,
                dataset_name=dataset_name,
                partition=partition,
                bands=bands,
                normalization=normalization,
                image_size=image_size,
                interpolation=interpolation,
                train_fraction=fraction,
                seed=seed,
                n_train_full=n_train_full,
                n_train_used=(
                    n_train_full
                    if fraction >= 1.0
                    else max(1, int(math.floor(n_train_full * fraction)))
                ),
                n_val=n_val,
                n_test=n_test,
            )
            block_key = _image_stats_block_key(image_stats_meta)
            image_stats_path = _image_stats_block_path(
                root=image_stats_cfg["root"],
                task="classification",
                model=model_name,
                dataset=dataset_name,
                train_fraction=fraction,
                seed=seed,
                block_key=block_key,
            )
            image_stats_complete = True
            if image_stats_cfg["enabled"]:
                image_stats_complete = _image_stats_block_status(
                    image_stats_path,
                    expected_count=n_test,
                ).is_complete

            if metrics_complete and image_stats_complete:
                logger.info(
                    "Skip cls (%s, %s, %.2f, %d) — already done",
                    model_name,
                    dataset_name,
                    fraction,
                    seed,
                )
                continue

            if fraction >= 1.0:
                X_sub, y_sub = X_train, y_train
            else:
                sss = StratifiedShuffleSplit(
                    n_splits=1, test_size=1.0 - fraction, random_state=seed
                )
                train_idx, _ = next(sss.split(X_train, y_train))
                X_sub = X_train[train_idx]
                y_sub = y_train[train_idx]
            n_train_used = len(X_sub)

            X_sub_t = torch.from_numpy(X_sub)
            y_sub_t = torch.from_numpy(y_sub).long()

            best_c: float = c_values[0]
            best_val_score = -1.0
            for c in c_values:
                clf = LogisticRegression(C=c, random_state=seed, device=str(device), verbose=False)
                clf.fit(X_sub_t, y_sub_t)
                val_pred = clf.predict(X_val_t)
                val_acc = float(accuracy_score(y_val, val_pred))
                if val_acc > best_val_score:
                    best_val_score = val_acc
                    best_c = c

            final_clf = LogisticRegression(
                C=best_c, random_state=seed, device=str(device), verbose=False
            )
            final_clf.fit(X_sub_t, y_sub_t)
            probs = final_clf.predict_proba(X_test_t)
            metric_values = _compute_cls_metrics(y_test, probs, n_bins_ece=n_bins_ece)

            base: dict = {
                "model": model_name,
                "dataset": dataset_name,
                "train_fraction": fraction,
                "seed": seed,
                "task": "classification",
                "n_train_full": n_train_full,
                "n_train_used": n_train_used,
                "n_val": n_val,
                "n_test": n_test,
                "best_c": best_c,
            }
            if image_stats_cfg["enabled"]:
                image_stats_meta = _cls_image_stats_block_meta(
                    model_name=model_name,
                    model_target=model_target,
                    dataset_name=dataset_name,
                    partition=partition,
                    bands=bands,
                    normalization=normalization,
                    image_size=image_size,
                    interpolation=interpolation,
                    train_fraction=fraction,
                    seed=seed,
                    n_train_full=n_train_full,
                    n_train_used=n_train_used,
                    n_val=n_val,
                    n_test=n_test,
                )
                block_key = _image_stats_block_key(image_stats_meta)
                image_stats_rows = _merge_image_stats_rows(
                    {
                        **image_stats_meta,
                        "image_stats_block_key": block_key,
                        "best_c": float(best_c),
                    },
                    _compute_cls_image_stats_rows(
                        y_true=y_test,
                        probs=probs,
                        sample_ids=test_sample_ids,
                    ),
                )
                image_stats_path = _image_stats_block_path(
                    root=image_stats_cfg["root"],
                    task="classification",
                    model=model_name,
                    dataset=dataset_name,
                    train_fraction=fraction,
                    seed=seed,
                    block_key=block_key,
                )
                _write_image_stats_block_atomic(
                    image_stats_path,
                    image_stats_rows,
                    expected_count=n_test,
                    overwrite=bool(image_stats_cfg["overwrite"]),
                    resume=bool(image_stats_cfg["resume"]),
                    compression=str(image_stats_cfg["compression"]),
                )
            for metric_name, metric_value in metric_values.items():
                if (model_name, dataset_name, fraction, seed, metric_name) in completed:
                    continue
                rows.append({**base, "metric_name": metric_name, "metric_value": metric_value})

    return rows


def _seg_sweep(
    *,
    probe: SegmentationProbe,
    train_cache: CachedFeaturesDataset,
    val_cache: CachedFeaturesDataset,
    test_cache: CachedFeaturesDataset,
    fractions: list[float],
    seeds_seg: int,
    target_grad_steps: int,
    batch_size: int,
    model_name: str,
    model_target: str,
    dataset_name: str,
    partition: str,
    bands: str,
    normalization: str,
    image_size: int | None,
    interpolation: str,
    num_classes: int,
    device: str,
    seg_cfg: DictConfig,
    completed: frozenset[tuple],
    image_stats_cfg: dict[str, Any],
) -> list[dict]:
    rows: list[dict] = []
    n_train_full = len(train_cache)
    n_val = len(val_cache)
    n_test = len(test_cache)

    criterion_template = instantiate(seg_cfg.criterion)
    ignore_index = _resolve_segmentation_ignore_index(seg_cfg, criterion_template)

    for fraction in fractions:
        for seed in range(seeds_seg):
            n_sub = max(1, int(math.floor(n_train_full * fraction)))
            metrics_complete = _summary_block_complete(
                completed=completed,
                model_name=model_name,
                dataset_name=dataset_name,
                fraction=fraction,
                seed=seed,
                metric_names=_SEG_METRICS,
            )
            image_stats_meta = _seg_image_stats_block_meta(
                model_name=model_name,
                model_target=model_target,
                dataset_name=dataset_name,
                partition=partition,
                bands=bands,
                normalization=normalization,
                image_size=image_size,
                interpolation=interpolation,
                train_fraction=fraction,
                seed=seed,
                n_train_full=n_train_full,
                n_train_used=n_sub,
                n_val=n_val,
                n_test=n_test,
                seg_cfg=seg_cfg,
                ignore_index=ignore_index,
            )
            block_key = _image_stats_block_key(image_stats_meta)
            image_stats_path = _image_stats_block_path(
                root=image_stats_cfg["root"],
                task="segmentation",
                model=model_name,
                dataset=dataset_name,
                train_fraction=fraction,
                seed=seed,
                block_key=block_key,
            )
            image_stats_complete = True
            if image_stats_cfg["enabled"]:
                image_stats_complete = _image_stats_block_status(
                    image_stats_path,
                    expected_count=n_test,
                ).is_complete

            if metrics_complete and image_stats_complete:
                logger.info(
                    "Skip seg (%s, %s, %.2f, %d) — already done",
                    model_name,
                    dataset_name,
                    fraction,
                    seed,
                )
                continue

            rng_np = np.random.default_rng(seed)
            indices = rng_np.choice(n_train_full, size=n_sub, replace=False)
            sub_cache = _subsample_cache(train_cache, indices)

            epochs = _compute_epochs(n_sub, batch_size, target_grad_steps)

            # Fresh head for each (fraction, seed): deep-copy probe, reset head weights
            fresh_probe = copy.deepcopy(probe)
            for m in fresh_probe.head.modules():
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()

            criterion = copy.deepcopy(criterion_template)
            solver = SegmentationSolver(
                model=fresh_probe,
                num_classes=num_classes,
                lr=float(seg_cfg.lr),
                device=device,
                criterion=criterion,
                lr_scheduler=str(seg_cfg.get("lr_scheduler", "cosine")),
                ignore_index=ignore_index,
            )
            solver.fit_cached(
                sub_cache, val_cache=val_cache, batch_size=batch_size, epochs=epochs, verbose=False
            )
            metrics_or_tuple = solver.evaluate_cached(
                test_cache,
                batch_size=batch_size,
                collect_image_stats=bool(image_stats_cfg["enabled"]),
            )
            image_stats_rows: list[dict[str, Any]] = []
            if image_stats_cfg["enabled"]:
                metrics, image_stats_rows = metrics_or_tuple
            else:
                metrics = metrics_or_tuple

            base: dict = {
                "model": model_name,
                "dataset": dataset_name,
                "train_fraction": fraction,
                "seed": seed,
                "task": "segmentation",
                "n_train_full": n_train_full,
                "n_train_used": n_sub,
                "n_val": n_val,
                "n_test": n_test,
                "best_c": float("nan"),
            }
            if image_stats_cfg["enabled"]:
                image_stats_meta = _seg_image_stats_block_meta(
                    model_name=model_name,
                    model_target=model_target,
                    dataset_name=dataset_name,
                    partition=partition,
                    bands=bands,
                    normalization=normalization,
                    image_size=image_size,
                    interpolation=interpolation,
                    train_fraction=fraction,
                    seed=seed,
                    n_train_full=n_train_full,
                    n_train_used=n_sub,
                    n_val=n_val,
                    n_test=n_test,
                    seg_cfg=seg_cfg,
                    ignore_index=ignore_index,
                )
                block_key = _image_stats_block_key(image_stats_meta)
                merged_image_rows = _merge_image_stats_rows(
                    {**image_stats_meta, "image_stats_block_key": block_key},
                    image_stats_rows,
                )
                image_stats_path = _image_stats_block_path(
                    root=image_stats_cfg["root"],
                    task="segmentation",
                    model=model_name,
                    dataset=dataset_name,
                    train_fraction=fraction,
                    seed=seed,
                    block_key=block_key,
                )
                _write_image_stats_block_atomic(
                    image_stats_path,
                    merged_image_rows,
                    expected_count=n_test,
                    overwrite=bool(image_stats_cfg["overwrite"]),
                    resume=bool(image_stats_cfg["resume"]),
                    compression=str(image_stats_cfg["compression"]),
                )
            # SegmentationSolver reports pixel-level calibration error under the
            # "ece" key; we record it as "pixel_ece" to distinguish it from the
            # image-level "ece" used on the classification path.
            for metric_name, key in [("miou", "mIoU"), ("pixel_ece", "ece")]:
                if (model_name, dataset_name, fraction, seed, metric_name) in completed:
                    continue
                rows.append({**base, "metric_name": metric_name, "metric_value": metrics[key]})

    return rows


@hydra.main(config_path="conf", config_name="sample_size_config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the sample-size calibration sweep.

    Args:
        cfg: Hydra configuration.
    """
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))

    output_path = str(cfg.output)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    device = torch.device(str(cfg.device))
    model_target = str(cfg.model._target_)
    model_name = str(cfg.model.get("name", model_target.split(".")[-1]))
    dataset_names = list(cfg.dataset.names)
    fractions = [float(f) for f in cfg.sample_size.fractions]
    seeds_cls = int(cfg.sample_size.seeds_cls)
    seeds_seg = int(cfg.sample_size.seeds_seg)
    target_grad_steps = int(cfg.sample_size.target_grad_steps)
    c_range = [float(c) for c in cfg.sample_size.c_range]
    c_values = [10.0**c for c in c_range]
    n_bins_ece = int(cfg.sample_size.n_bins_ece)
    partition = str(cfg.dataset.partition)
    bands = str(cfg.dataset.bands)
    normalization = str(cfg.dataset.get("normalization", "bandspec_zscore"))
    image_size = getattr(cfg.dataset, "image_size", None)
    interpolation = str(cfg.dataset.get("interpolation", "bilinear"))
    image_stats_cfg = _build_image_stats_cfg(cfg)
    image_stats_cfg["resume"] = bool(cfg.resume)

    completed: frozenset[tuple] = frozenset()
    if bool(cfg.resume):
        completed = _load_completed(output_path)

    for dataset_name in dataset_names:
        try:
            ds_cls = get_bench_dataset_class(dataset_name)
        except KeyError:
            logger.warning("Skipping unknown dataset %s", dataset_name)
            continue

        task = ds_cls.task
        multilabel = getattr(ds_cls, "multilabel", False)

        loaded = get_datasets(
            dataset_name=dataset_name,
            partition_name=partition,
            batch_size=int(cfg.dataset.batch_size),
            num_workers=int(cfg.dataset.get("num_workers", 4)),
            return_val=True,
            image_size=getattr(cfg.dataset, "image_size", None),
            interpolation=str(cfg.dataset.get("interpolation", "bilinear")),
            bands=bands,
        )
        _, train_loader, val_loader, test_loader = loaded

        bench = ds_cls()
        bands_resolved = (
            tuple(bench.rgb_bands)
            if bands == "rgb"
            else None
            if bands in ("all", None)
            else tuple(bands)
        )
        band_specs = bench.select_band_specs(bands_resolved)
        model = instantiate(
            cfg.model, bands=band_specs, normalization=normalization, _convert_="object"
        )
        model.to(device).eval()

        verbose = bool(cfg.verbose)

        if task == "classification" and not multilabel:
            # Skip embedding entirely if all (fraction, seed) combos are already done
            if bool(cfg.resume) and not any(
                not _summary_block_complete(
                    completed=completed,
                    model_name=model_name,
                    dataset_name=dataset_name,
                    fraction=f,
                    seed=s,
                    metric_names=_CLS_METRICS,
                )
                or (
                    image_stats_cfg["enabled"]
                    and not _image_stats_block_status(
                        _image_stats_block_path(
                            root=image_stats_cfg["root"],
                            task="classification",
                            model=model_name,
                            dataset=dataset_name,
                            train_fraction=f,
                            seed=s,
                            block_key=_image_stats_block_key(
                                _cls_image_stats_block_meta(
                                    model_name=model_name,
                                    model_target=model_target,
                                    dataset_name=dataset_name,
                                    partition=partition,
                                    bands=bands,
                                    normalization=normalization,
                                    image_size=image_size,
                                    interpolation=interpolation,
                                    train_fraction=f,
                                    seed=s,
                                    n_train_full=len(train_loader.dataset),
                                    n_train_used=(
                                        len(train_loader.dataset)
                                        if f >= 1.0
                                        else max(1, int(math.floor(len(train_loader.dataset) * f)))
                                    ),
                                    n_val=len(val_loader.dataset),
                                    n_test=len(test_loader.dataset),
                                )
                            ),
                        ),
                        expected_count=len(test_loader.dataset),
                    ).is_complete
                )
                for f in fractions
                for s in range(seeds_cls)
            ):
                logger.info("All cls rows done for %s/%s — skipping", model_name, dataset_name)
                continue

            X_train, y_train = embed_split(model, train_loader, device, verbose)
            X_val, y_val = embed_split(model, val_loader, device, verbose)
            X_test, y_test, test_sample_ids = _embed_test_split_with_ids(
                model,
                test_loader,
                device,
                verbose,
            )

            rows = _cls_sweep(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                fractions=fractions,
                seeds_cls=seeds_cls,
                c_values=c_values,
                n_bins_ece=n_bins_ece,
                model_name=model_name,
                model_target=model_target,
                dataset_name=dataset_name,
                partition=partition,
                bands=bands,
                normalization=normalization,
                image_size=image_size,
                interpolation=interpolation,
                device=device,
                test_sample_ids=test_sample_ids,
                completed=completed,
                image_stats_cfg=image_stats_cfg,
            )
            if rows:
                append_rows_atomic(output_path, rows)
                logger.info("Wrote %d rows for cls %s / %s", len(rows), model_name, dataset_name)

        elif task == "segmentation":
            # Merge the model-specific eval block into the top-level eval config,
            # mirroring main.py. Without this, model configs that ship their own
            # eval.segmentation.layers (e.g. resnet50's FPN layers) are ignored
            # and SegmentationProbe silently falls back to ["backbone_output"].
            seg_eval_cfg = cfg.eval
            if "eval" in cfg.model and cfg.model.eval is not None:
                seg_eval_cfg = OmegaConf.merge(seg_eval_cfg, cfg.model.eval)
            seg_cfg = seg_eval_cfg.segmentation
            if not list(seg_cfg.layers):
                raise ValueError(
                    f"Segmentation sweep for {dataset_name} requires "
                    "eval.segmentation.layers to be set (none found in the "
                    f"top-level config or the {model_name} model config)."
                )

            seg_ignore_index = _resolve_segmentation_ignore_index(
                seg_cfg,
                instantiate(seg_cfg.criterion),
            )
            if bool(cfg.resume) and not any(
                not _summary_block_complete(
                    completed=completed,
                    model_name=model_name,
                    dataset_name=dataset_name,
                    fraction=f,
                    seed=s,
                    metric_names=_SEG_METRICS,
                )
                or (
                    image_stats_cfg["enabled"]
                    and not _image_stats_block_status(
                        _image_stats_block_path(
                            root=image_stats_cfg["root"],
                            task="segmentation",
                            model=model_name,
                            dataset=dataset_name,
                            train_fraction=f,
                            seed=s,
                            block_key=_image_stats_block_key(
                                _seg_image_stats_block_meta(
                                    model_name=model_name,
                                    model_target=model_target,
                                    dataset_name=dataset_name,
                                    partition=partition,
                                    bands=bands,
                                    normalization=normalization,
                                    image_size=image_size,
                                    interpolation=interpolation,
                                    train_fraction=f,
                                    seed=s,
                                    n_train_full=len(train_loader.dataset),
                                    n_train_used=max(
                                        1, int(math.floor(len(train_loader.dataset) * f))
                                    ),
                                    n_val=len(val_loader.dataset),
                                    n_test=len(test_loader.dataset),
                                    seg_cfg=seg_cfg,
                                    ignore_index=seg_ignore_index,
                                )
                            ),
                        ),
                        expected_count=len(test_loader.dataset),
                    ).is_complete
                )
                for f in fractions
                for s in range(seeds_seg)
            ):
                logger.info("All seg rows done for %s/%s — skipping", model_name, dataset_name)
                continue

            seg_probe = SegmentationProbe(
                backbone=model,
                layer_names=list(seg_cfg.layers),
                num_classes=bench.num_classes,
                head_type=str(seg_cfg.head_type),
                freeze_backbone=True,
            )
            train_cache = seg_probe.extract_segmentation_features(train_loader)
            val_cache = seg_probe.extract_segmentation_features(val_loader)
            test_cache = seg_probe.extract_segmentation_features(test_loader)

            rows = _seg_sweep(
                probe=seg_probe,
                train_cache=train_cache,
                val_cache=val_cache,
                test_cache=test_cache,
                fractions=fractions,
                seeds_seg=seeds_seg,
                target_grad_steps=target_grad_steps,
                batch_size=int(seg_cfg.batch_size),
                model_name=model_name,
                model_target=model_target,
                dataset_name=dataset_name,
                partition=partition,
                bands=bands,
                normalization=normalization,
                image_size=image_size,
                interpolation=interpolation,
                num_classes=bench.num_classes,
                device=str(device),
                seg_cfg=seg_cfg,
                completed=completed,
                image_stats_cfg=image_stats_cfg,
            )
            if rows:
                append_rows_atomic(output_path, rows)
                logger.info("Wrote %d rows for seg %s / %s", len(rows), model_name, dataset_name)

        else:
            logger.info(
                "Skipping dataset %s (task=%s, multilabel=%s)", dataset_name, task, multilabel
            )
