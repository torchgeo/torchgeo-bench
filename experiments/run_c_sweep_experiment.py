#!/usr/bin/env python
"""C-sweep analysis for linear probing.

Extracts features once per (model, dataset), then sweeps the L2
regularization strength ``C`` of :class:`torchgeo_bench.linear.LogisticRegression`
and records train/val/test accuracy for each ``C`` value.

Usage:
    python experiments/run_c_sweep_experiment.py
    python experiments/run_c_sweep_experiment.py --devices 3
"""

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
from _runner import add_devices_argument, default_output
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from torchgeo_bench.datasets import get_bench_dataset_class, get_datasets
from torchgeo_bench.datasets.base import BandSpec
from torchgeo_bench.linear import LogisticRegression
from torchgeo_bench.utils import extract_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUTPUT = default_output(__file__)
SEED = 0
IMAGE_SIZE = 224

DATASETS = ["m-bigearthnet", "m-brick-kiln", "m-eurosat", "m-forestnet", "m-pv4ger", "m-so2sat"]

MODEL_CONFIGS = {
    "resnet18": {
        "_target_": "torchgeo_bench.models.timm.TimmPatchBenchModel",
        "model_name": "resnet18",
        "pretrained": True,
        "global_pool": "avg",
        "name": "resnet18",
    },
    "resnet50": {
        "_target_": "torchgeo_bench.models.timm.TimmPatchBenchModel",
        "model_name": "resnet50",
        "pretrained": True,
        "global_pool": "avg",
        "name": "resnet50",
    },
    "convnext_large_dinov3": {
        "_target_": "torchgeo_bench.models.timm.TimmPatchBenchModel",
        "model_name": "convnext_large.dinov3_lvd1689m",
        "pretrained": True,
        "global_pool": "avg",
        "name": "convnext_large_dinov3",
    },
    "dinov3": {
        "_target_": "torchgeo_bench.models.timm.TimmPatchBenchModel",
        "model_name": "vit_large_patch16_dinov3.lvd1689m",
        "pretrained": True,
        "global_pool": "avg",
        "use_cls_token": False,
        "name": "vit_large_patch16_dinov3",
    },
    "dinov3sat": {
        "_target_": "torchgeo_bench.models.timm.TimmPatchBenchModel",
        "model_name": "vit_large_patch16_dinov3.sat493m",
        "pretrained": True,
        "global_pool": "avg",
        "use_cls_token": False,
        "name": "vit_large_patch16_dinov3sat",
    },
}

C_VALUES = np.sort(np.unique(np.append(np.logspace(-7, 2, 40), 1.0)))


def instantiate_model(model_cfg: dict, bands: list[BandSpec]) -> torch.nn.Module:
    """Instantiate a model from its ``MODEL_CONFIGS`` entry."""
    target = model_cfg["_target_"]
    module_name, class_name = target.rsplit(".", 1)
    module = __import__(module_name, fromlist=[class_name])
    cls = getattr(module, class_name)

    kwargs = {k: v for k, v in model_cfg.items() if k not in ("_target_", "name")}
    kwargs["bands"] = bands
    return cls(**kwargs)


def run_c_sweep(
    model_name: str,
    dataset_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
) -> list[dict]:
    """Train ``LogisticRegression`` for each ``C`` and return per-C metrics."""
    rows = []
    x_train_t = torch.from_numpy(x_train)
    y_train_t = torch.from_numpy(y_train).long()
    x_val_t = torch.from_numpy(x_val)
    x_test_t = torch.from_numpy(x_test)

    for c in tqdm(C_VALUES, desc=f"  C-sweep ({model_name})", leave=False):
        clf = LogisticRegression(
            C=float(c),
            max_iter=2000,
            tol=1e-6,
            random_state=SEED,
            device=device,
        )
        clf.fit(x_train_t, y_train_t)

        train_preds = clf.predict(x_train_t)
        val_preds = clf.predict(x_val_t)
        test_preds = clf.predict(x_test_t)

        rows.append(
            {
                "dataset": dataset_name,
                "model": model_name,
                "C": float(c),
                "log10_C": float(np.log10(c)),
                "train_acc": float(accuracy_score(y_train, train_preds)),
                "val_acc": float(accuracy_score(y_val, val_preds)),
                "test_acc": float(accuracy_score(y_test, test_preds)),
                "feature_dim": int(x_train.shape[1]),
                "n_train": int(len(x_train)),
                "n_val": int(len(x_val)),
                "n_test": int(len(x_test)),
            }
        )

    return rows


def run_dataset_sweep(dataset_name: str, device: torch.device, all_rows: list[dict]) -> list[dict]:
    """Run the C sweep for one dataset, appending rows in-place to ``all_rows``."""
    bench_cls = get_bench_dataset_class(dataset_name)
    if bench_cls.multilabel:
        logger.warning(
            "Skipping %s: this script is single-label only (uses accuracy_score + "
            "single-label LogisticRegression).",
            dataset_name,
        )
        return all_rows

    completed_models = {r["model"] for r in all_rows if r.get("dataset") == dataset_name}
    if completed_models:
        logger.info(
            "Resume: %d models already computed for %s: %s",
            len(completed_models),
            dataset_name,
            sorted(completed_models),
        )

    logger.info("Loading %s dataset...", dataset_name)
    _train_dataset, train_loader, val_loader, test_loader = get_datasets(
        dataset_name=dataset_name,
        partition_name="default",
        batch_size=64,
        return_val=True,
        image_size=IMAGE_SIZE,
        interpolation="bilinear",
    )
    bands = bench_cls().select_band_specs(tuple(bench_cls.rgb_bands))

    for model_name, model_cfg in MODEL_CONFIGS.items():
        if model_name in completed_models:
            logger.info("=== %s/%s === skipping, already computed", dataset_name, model_name)
            continue

        logger.info("=== %s/%s ===", dataset_name, model_name)
        logger.info("  Loading model %s...", model_name)
        model = instantiate_model(model_cfg, bands)
        model.to(device).eval()

        logger.info("  Extracting features...")
        x_train, y_train = extract_features(model, train_loader, device, verbose=False)
        x_val, y_val = extract_features(model, val_loader, device, verbose=False)
        x_test, y_test = extract_features(model, test_loader, device, verbose=False)
        logger.info(
            "  Features: train=%s, val=%s, test=%s",
            x_train.shape,
            x_val.shape,
            x_test.shape,
        )

        del model
        torch.cuda.empty_cache()

        rows = run_c_sweep(
            model_name,
            dataset_name,
            x_train,
            y_train,
            x_val,
            y_val,
            x_test,
            y_test,
            device,
        )
        all_rows.extend(rows)
        pd.DataFrame(all_rows).to_csv(OUTPUT, index=False)
        logger.info("  Saved %d rows to %s", len(all_rows), OUTPUT)

    return all_rows


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="C-sweep for linear probing")
    add_devices_argument(parser)
    args = parser.parse_args()

    if len(args.devices) > 1:
        logger.warning(
            "This script runs in-process; using only the first of --devices=%s.",
            args.devices,
        )
    device = torch.device(f"cuda:{args.devices[0]}")

    os.makedirs(os.path.dirname(OUTPUT) or ".", exist_ok=True)

    all_rows: list[dict] = []
    if os.path.exists(OUTPUT):
        all_rows = pd.read_csv(OUTPUT).to_dict("records")
        logger.info("Resume: loaded %d existing rows from %s", len(all_rows), OUTPUT)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    logger.info("Running C sweep on %d datasets -> %s", len(DATASETS), OUTPUT)

    for dataset_name in DATASETS:
        all_rows = run_dataset_sweep(dataset_name, device, all_rows)

    logger.info("Done. Final results: %s", OUTPUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
