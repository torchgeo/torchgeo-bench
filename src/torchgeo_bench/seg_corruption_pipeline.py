"""Segmentation corruption pipeline: train once on clean data, evaluate at each corruption × severity."""

import copy
import logging
import os
import warnings
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from torchgeo_bench.datasets import get_bench_dataset_class, get_datasets
from torchgeo_bench.main import (
    _build_seg_probe_and_solver,
    _resolve_segmentation_ignore_index,
    append_rows_atomic,
)
from torchgeo_bench.segmentation_probe import CachedFeaturesDataset
from torchgeo_bench.uq.corruptions import SKIP_POISSON_GAUSSIAN, CorruptionTransform

warnings.filterwarnings("ignore", message="Dataset has no geotransform", category=UserWarning)

logger = logging.getLogger(__name__)

_SEG_CORRUPTION_METRICS = (
    "miou",
    "fw_iou",
    "ece",
    "rms_ce",
    "mce",
    "precision",
    "recall",
    "f1",
    "ece_ts",
    "rms_ce_ts",
    "mce_ts",
    "temperature",
)


def _load_completed(path: str) -> frozenset[tuple]:
    if not os.path.exists(path):
        return frozenset()
    try:
        import pandas as pd

        df = pd.read_csv(path)
        return frozenset(
            zip(
                df["model"],
                df["dataset"],
                df["corruption_type"],
                df["severity"],
                df["seed"],
                df["metric_name"],
                strict=False,
            )
        )
    except Exception:
        return frozenset()


def _feature_dim(cache: CachedFeaturesDataset) -> int:
    """Return the total feature channel count across all hooked layers."""
    return sum(t.shape[1] for t in cache.layer_tensors)


@hydra.main(config_path="conf", config_name="seg_corruption_config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run the segmentation corruption robustness sweep.

    Args:
        cfg: Hydra configuration.
    """
    torch.manual_seed(int(cfg.seed))

    output_path = str(cfg.output)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    device = torch.device(str(cfg.device))
    seed = int(cfg.seed)
    model_target = str(cfg.model._target_)
    model_name = str(cfg.model.get("name", model_target.split(".")[-1]))
    dataset_names = list(cfg.dataset.names)
    partition = str(cfg.dataset.partition)
    bands = str(cfg.dataset.bands)
    normalization = str(cfg.dataset.get("normalization", "bandspec_zscore"))
    image_size = getattr(cfg.dataset, "image_size", None)
    interpolation = str(cfg.dataset.get("interpolation", "bilinear"))
    corruption_types = list(cfg.corruption.types)
    severities = [int(s) for s in cfg.corruption.severities]

    completed: frozenset[tuple] = frozenset()
    if bool(cfg.resume):
        completed = _load_completed(output_path)

    for dataset_name in dataset_names:
        try:
            ds_cls = get_bench_dataset_class(dataset_name)
        except KeyError:
            logger.warning("Skipping unknown dataset %s", dataset_name)
            continue

        if ds_cls.task != "segmentation":
            logger.warning("Skipping %s: task=%s (not segmentation)", dataset_name, ds_cls.task)
            continue

        loaded = get_datasets(
            dataset_name=dataset_name,
            partition_name=partition,
            batch_size=int(cfg.dataset.batch_size),
            num_workers=int(cfg.dataset.get("num_workers", 4)),
            return_val=True,
            image_size=image_size,
            interpolation=interpolation,
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

        try:
            model = instantiate(
                cfg.model, bands=band_specs, normalization=normalization, _convert_="object"
            )
        except ValueError as exc:
            logger.warning(
                "Skipping %s/%s: model instantiation failed: %s", model_name, dataset_name, exc
            )
            continue
        model.to(device).eval()

        # Merge model-specific eval config (e.g. resnet50 FPN layers) over the base eval config.
        seg_eval_cfg = cfg.eval
        if "eval" in cfg.model and cfg.model.eval is not None:
            seg_eval_cfg = OmegaConf.merge(seg_eval_cfg, cfg.model.eval)
        seg_cfg = seg_eval_cfg.segmentation

        if not list(seg_cfg.layers):
            logger.warning(
                "Skipping segmentation for %s/%s: eval.segmentation.layers is not set.",
                model_name,
                dataset_name,
            )
            continue

        # Check if all corruption × severity blocks are already done.
        if bool(cfg.resume):
            all_done = True
            for ct in corruption_types:
                sev_list = [0] if ct == "clean" else severities
                for sev in sev_list:
                    if ct == "poisson_gaussian" and dataset_name in SKIP_POISSON_GAUSSIAN:
                        continue
                    for metric in _SEG_CORRUPTION_METRICS:
                        if (model_name, dataset_name, ct, sev, seed, metric) not in completed:
                            all_done = False
                            break
                    if not all_done:
                        break
                if not all_done:
                    break
            if all_done:
                logger.info("All rows done for %s/%s — skipping", model_name, dataset_name)
                continue

        probe, solver = _build_seg_probe_and_solver(
            model, bench.num_classes, seg_eval_cfg, device, float(seg_cfg.lr)
        )

        cache_dtype = torch.float16
        cache_dtype_str = str(seg_cfg.get("cache_dtype", "float16"))
        if cache_dtype_str == "float32":
            cache_dtype = torch.float32

        logger.info("Extracting clean train/val features for %s/%s", model_name, dataset_name)
        train_cache = probe.extract_segmentation_features(train_loader, cache_dtype=cache_dtype)
        val_cache = probe.extract_segmentation_features(val_loader, cache_dtype=cache_dtype)

        feat_dim = _feature_dim(train_cache)
        n_test = len(test_loader.dataset)

        epochs = int(seg_cfg.get("epochs", 10))
        batch_size = int(seg_cfg.get("batch_size", 64))

        logger.info(
            "Training segmentation head on clean data for %s/%s (%d epochs)",
            model_name,
            dataset_name,
            epochs,
        )
        solver.fit_cached(train_cache, val_cache=val_cache, batch_size=batch_size, epochs=epochs, verbose=bool(cfg.verbose))

        base_meta = {
            "model": model_name,
            "dataset": dataset_name,
            "seed": seed,
            "normalization": normalization,
            "bands": bands,
            "image_size": image_size,
            "partition": partition,
            "n_test": n_test,
            "feature_dim": feat_dim,
        }

        for corruption_type in corruption_types:
            if corruption_type == "poisson_gaussian" and dataset_name in SKIP_POISSON_GAUSSIAN:
                logger.info(
                    "Skipping poisson_gaussian for %s (in SKIP_POISSON_GAUSSIAN)", dataset_name
                )
                continue

            sev_list = [0] if corruption_type == "clean" else severities

            for severity in sev_list:
                block_key = (model_name, dataset_name, corruption_type, severity, seed)
                if all((*block_key, m) in completed for m in _SEG_CORRUPTION_METRICS):
                    logger.info(
                        "Skip (%s, %s, %s, %d) — already done",
                        model_name,
                        dataset_name,
                        corruption_type,
                        severity,
                    )
                    continue

                if corruption_type == "clean":
                    transform = None
                else:
                    transform = CorruptionTransform(
                        corruption_type=corruption_type,
                        severity=severity,
                        seed=seed,
                        band_specs=band_specs,
                        dataset_name=dataset_name,
                    )

                logger.info(
                    "Extracting test features: %s/%s corruption=%s severity=%d",
                    model_name,
                    dataset_name,
                    corruption_type,
                    severity,
                )
                test_cache = probe.extract_segmentation_features(
                    test_loader, cache_dtype=cache_dtype, transform=transform
                )

                metrics: dict = solver.evaluate_cached(test_cache, batch_size=batch_size)

                temperature, cal_ts = solver.evaluate_cached_temperature_scaled(
                    test_cache, val_cache, batch_size=batch_size
                )

                all_metrics = {
                    "miou": metrics["mIoU"],
                    "fw_iou": metrics["fw_IoU"],
                    "ece": metrics["ece"],
                    "rms_ce": metrics["rms_ce"],
                    "mce": metrics["mce"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1": metrics["f1"],
                    "ece_ts": cal_ts["ece_ts"],
                    "rms_ce_ts": cal_ts["rms_ce_ts"],
                    "mce_ts": cal_ts["mce_ts"],
                    "temperature": temperature,
                }

                rows = []
                for metric_name, metric_value in all_metrics.items():
                    if (*block_key, metric_name) in completed:
                        continue
                    rows.append(
                        {
                            **base_meta,
                            "corruption_type": corruption_type,
                            "severity": severity,
                            "metric_name": metric_name,
                            "metric_value": metric_value,
                        }
                    )

                if rows:
                    append_rows_atomic(output_path, rows)
                    logger.info(
                        "Wrote %d rows for (%s, %s, %s, %d)",
                        len(rows),
                        model_name,
                        dataset_name,
                        corruption_type,
                        severity,
                    )
