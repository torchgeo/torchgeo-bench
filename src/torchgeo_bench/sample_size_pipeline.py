"""Sample-size sweep pipeline: calibration vs training-set fraction."""

import copy
import logging
import math
import os
import warnings

import hydra
import numpy as np
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
from torchgeo_bench.uq.metrics import nll

warnings.filterwarnings("ignore", message="Dataset has no geotransform", category=UserWarning)

logger = logging.getLogger(__name__)

_CLS_METRICS = ("accuracy", "ece", "nll")
_SEG_METRICS = ("miou", "pixel_ece")


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
        import pandas as pd

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
    dataset_name: str,
    device: torch.device,
    completed: frozenset[tuple],
) -> list[dict]:
    rows: list[dict] = []
    n_train_full = len(X_train)
    n_val = len(X_val)
    n_test = len(X_test)

    X_val_t = torch.from_numpy(X_val)
    X_test_t = torch.from_numpy(X_test)

    for fraction in fractions:
        for seed in range(seeds_cls):
            if all(
                (model_name, dataset_name, fraction, seed, m) in completed
                for m in _CLS_METRICS
            ):
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
            preds = probs.argmax(axis=1)
            acc = float(accuracy_score(y_test, preds))
            cal = compute_calibration_metrics(y_test, probs, multi_label=False, n_bins=n_bins_ece)
            ece_val = cal["ece"]
            nll_val = float(nll(probs, y_test))

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
            for metric_name, metric_value in [
                ("accuracy", acc),
                ("ece", ece_val),
                ("nll", nll_val),
            ]:
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
    dataset_name: str,
    num_classes: int,
    device: str,
    seg_cfg: DictConfig,
    completed: frozenset[tuple],
) -> list[dict]:
    rows: list[dict] = []
    n_train_full = len(train_cache)
    n_val = len(val_cache)
    n_test = len(test_cache)

    criterion_template = instantiate(seg_cfg.criterion)
    ignore_index = _resolve_segmentation_ignore_index(seg_cfg, criterion_template)

    for fraction in fractions:
        for seed in range(seeds_seg):
            if all(
                (model_name, dataset_name, fraction, seed, m) in completed
                for m in _SEG_METRICS
            ):
                logger.info(
                    "Skip seg (%s, %s, %.2f, %d) — already done",
                    model_name,
                    dataset_name,
                    fraction,
                    seed,
                )
                continue

            rng_np = np.random.default_rng(seed)
            n_sub = max(1, int(math.floor(n_train_full * fraction)))
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
            metrics = solver.evaluate_cached(test_cache, batch_size=batch_size)

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
            # SegmentationSolver reports pixel-level calibration error under the
            # "ece" key; we record it as "pixel_ece" to distinguish it from the
            # image-level "ece" used on the classification path.
            for metric_name, key in [("miou", "mIoU"), ("pixel_ece", "ece")]:
                rows.append(
                    {**base, "metric_name": metric_name, "metric_value": metrics[key]}
                )

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
        normalization = str(cfg.dataset.get("normalization", "bandspec_zscore"))
        model = instantiate(
            cfg.model, bands=band_specs, normalization=normalization, _convert_="object"
        )
        model.to(device).eval()

        verbose = bool(cfg.verbose)

        if task == "classification" and not multilabel:
            # Skip embedding entirely if all (fraction, seed) combos are already done
            if bool(cfg.resume) and not any(
                (model_name, dataset_name, f, s, m) not in completed
                for f in fractions
                for s in range(seeds_cls)
                for m in _CLS_METRICS
            ):
                logger.info("All cls rows done for %s/%s — skipping", model_name, dataset_name)
                continue

            X_train, y_train = embed_split(model, train_loader, device, verbose)
            X_val, y_val = embed_split(model, val_loader, device, verbose)
            X_test, y_test = embed_split(model, test_loader, device, verbose)

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
                dataset_name=dataset_name,
                device=device,
                completed=completed,
            )
            if rows:
                append_rows_atomic(output_path, rows)
                logger.info(
                    "Wrote %d rows for cls %s / %s", len(rows), model_name, dataset_name
                )

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
                dataset_name=dataset_name,
                num_classes=bench.num_classes,
                device=str(device),
                seg_cfg=seg_cfg,
                completed=completed,
            )
            if rows:
                append_rows_atomic(output_path, rows)
                logger.info(
                    "Wrote %d rows for seg %s / %s", len(rows), model_name, dataset_name
                )

        else:
            logger.info(
                "Skipping dataset %s (task=%s, multilabel=%s)", dataset_name, task, multilabel
            )
