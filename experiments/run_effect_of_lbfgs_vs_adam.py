#!/usr/bin/env python
"""Compare LBFGS vs Adam for linear probing — speed and final accuracy.

Both branches use :class:`torchgeo_bench.linear.LogisticRegression` (the
same class the main pipeline uses for its linear probe), so the comparison
is apples-to-apples on regularization and the training objective::

    loss = (1/n) * CrossEntropy + (1/n) * 0.5/C * ||W||^2

For each (model, dataset, fit_config) we record wall-clock fit time,
number of optimizer iterations, and train/val/test accuracy.

Configs swept (per (model, dataset)):
    - LBFGS: one fit per ``C``, ``lr=1.0`` (LBFGS uses strong-Wolfe line
      search, so ``lr`` is mostly a starting step; we keep the default).
    - Adam:  one fit per (``C``, ``lr``) on a small LR grid.

Usage:
    python experiments/run_effect_of_lbfgs_vs_adam.py
    python experiments/run_effect_of_lbfgs_vs_adam.py --devices 3
"""

import argparse
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from _runner import add_devices_argument, default_output
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from torchgeo_bench.datasets import get_bench_dataset_class, get_datasets
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
    "dinov3sat": {
        "_target_": "torchgeo_bench.models.timm.TimmPatchBenchModel",
        "model_name": "vit_large_patch16_dinov3.sat493m",
        "pretrained": True,
        "global_pool": "avg",
        "use_cls_token": False,
        "auto_resize": True,
        "name": "vit_large_patch16_dinov3sat",
    },
}

# Mirrors the main pipeline's c_range default (small, focused grid).
C_VALUES = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
ADAM_LRS = [1e-3, 1e-2, 1e-1, 1.0]

# Match main.py's linear-probe call (LogisticRegression(C=c, max_iter=2000, tol=1e-6)).
MAX_ITER = 2000
TOL = 1e-6


def instantiate_model(model_cfg: dict, bands: list) -> torch.nn.Module:
    """Instantiate a model from its ``MODEL_CONFIGS`` entry."""
    target = model_cfg["_target_"]
    module_name, class_name = target.rsplit(".", 1)
    module = __import__(module_name, fromlist=[class_name])
    cls = getattr(module, class_name)

    kwargs = {k: v for k, v in model_cfg.items() if k not in ("_target_", "name")}
    kwargs["bands"] = bands
    return cls(**kwargs)


def build_configs() -> list[dict]:
    """Build a flat list of ``(solver, C, lr)`` fit configurations."""
    configs: list[dict] = []
    for c in C_VALUES:
        configs.append({"solver": "lbfgs", "C": float(c), "lr": 1.0})
    for c in C_VALUES:
        for lr in ADAM_LRS:
            configs.append({"solver": "adam", "C": float(c), "lr": float(lr)})
    return configs


def fit_one(
    cfg: dict,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    x_test: torch.Tensor,
    device: torch.device,
) -> tuple[LogisticRegression, float, np.ndarray, np.ndarray, np.ndarray]:
    """Fit one config and return ``(clf, fit_seconds, train_pred, val_pred, test_pred)``."""
    clf = LogisticRegression(
        C=cfg["C"],
        lr=cfg["lr"],
        solver=cfg["solver"],
        max_iter=MAX_ITER,
        tol=TOL,
        random_state=SEED,
        device=device,
    )

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    clf.fit(x_train, y_train)
    if device.type == "cuda":
        torch.cuda.synchronize()
    fit_seconds = time.perf_counter() - start

    train_pred = clf.predict(x_train)
    val_pred = clf.predict(x_val)
    test_pred = clf.predict(x_test)
    return clf, fit_seconds, train_pred, val_pred, test_pred


def run_dataset(
    dataset_name: str,
    configs: list[dict],
    device: torch.device,
    all_rows: list[dict],
) -> list[dict]:
    """Run all (model, config) fits for one dataset, appending rows to ``all_rows``."""
    completed = {
        (r["model"], r["solver"], float(r["C"]), float(r["lr"]))
        for r in all_rows
        if r.get("dataset") == dataset_name
    }
    if completed:
        logger.info("Resume: %d existing rows for %s", len(completed), dataset_name)

    logger.info("Loading %s dataset...", dataset_name)
    bench = get_bench_dataset_class(dataset_name)()
    if bench.multilabel:
        logger.info(
            "=== %s === skipping (multi-label not supported by this comparison)", dataset_name
        )
        return all_rows
    bands_list = bench.select_band_specs(tuple(bench.rgb_bands))
    train_dataset, train_loader, val_loader, test_loader = get_datasets(
        dataset_name=dataset_name,
        partition_name="default",
        batch_size=64,
        return_val=True,
        image_size=IMAGE_SIZE,
        interpolation="bilinear",
    )
    num_channels = train_dataset[0]["image"].shape[0]
    assert len(bands_list) == num_channels, (
        f"BandSpec count {len(bands_list)} != tensor channel count {num_channels} for {dataset_name}"
    )

    for model_name, model_cfg in MODEL_CONFIGS.items():
        remaining = [
            c
            for c in configs
            if (model_name, c["solver"], float(c["C"]), float(c["lr"])) not in completed
        ]
        if not remaining:
            logger.info(
                "=== %s/%s === all %d configs already computed, skipping",
                dataset_name,
                model_name,
                len(configs),
            )
            continue

        logger.info(
            "=== %s/%s === %d/%d configs remaining",
            dataset_name,
            model_name,
            len(remaining),
            len(configs),
        )

        logger.info("  Loading model %s...", model_name)
        model = instantiate_model(model_cfg, bands_list)
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
        if device.type == "cuda":
            torch.cuda.empty_cache()

        x_train_t = torch.from_numpy(x_train)
        y_train_t = torch.from_numpy(y_train).long()
        x_val_t = torch.from_numpy(x_val)
        x_test_t = torch.from_numpy(x_test)

        for cfg in tqdm(remaining, desc=f"  fits ({model_name})", leave=False):
            clf, fit_seconds, train_pred, val_pred, test_pred = fit_one(
                cfg,
                x_train_t,
                y_train_t,
                x_val_t,
                x_test_t,
                device,
            )
            all_rows.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    "solver": cfg["solver"],
                    "C": float(cfg["C"]),
                    "log10_C": float(np.log10(cfg["C"])),
                    "lr": float(cfg["lr"]),
                    "fit_seconds": float(fit_seconds),
                    "n_iter": int(clf.n_iter_),
                    "train_acc": float(accuracy_score(y_train, train_pred)),
                    "val_acc": float(accuracy_score(y_val, val_pred)),
                    "test_acc": float(accuracy_score(y_test, test_pred)),
                    "feature_dim": int(x_train.shape[1]),
                    "n_train": int(len(x_train)),
                    "n_val": int(len(x_val)),
                    "n_test": int(len(x_test)),
                    "max_iter": MAX_ITER,
                    "tol": TOL,
                    "device": str(device),
                }
            )

        pd.DataFrame(all_rows).to_csv(OUTPUT, index=False)
        logger.info("  Saved %d rows to %s", len(all_rows), OUTPUT)

    return all_rows


def main() -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="LBFGS vs Adam linear-probing speed test")
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

    configs = build_configs()
    logger.info(
        "Configs per (model, dataset): %d LBFGS + %d Adam = %d total",
        len(C_VALUES),
        len(C_VALUES) * len(ADAM_LRS),
        len(configs),
    )

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    logger.info("Running LBFGS-vs-Adam sweep on %d datasets -> %s", len(DATASETS), OUTPUT)

    for dataset_name in DATASETS:
        all_rows = run_dataset(dataset_name, configs, device, all_rows)

    logger.info("Done. Final results: %s", OUTPUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
