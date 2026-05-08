#!/usr/bin/env python
"""Compare LBFGS vs Adam for linear probing — speed and final accuracy.

Both branches use :class:`torchgeo_bench.linear.LogisticRegression` (the
same class the main pipeline uses for its linear probe), so the comparison
is apples-to-apples on regularization and the training objective::

    loss = (1/n) * CrossEntropy + (1/n) * 0.5/C * ||W||^2

For each (model, dataset, fit_config) we record wall-clock fit time,
number of optimizer iterations, and train/val/test accuracy. With these
columns you can plot e.g. accuracy-vs-time Pareto curves to see which
solver hits a target accuracy fastest.

Configs swept (per (model, dataset)):
    - LBFGS: one fit per ``C``, ``lr=1.0`` (LBFGS uses strong-Wolfe line
      search, so ``lr`` is mostly a starting step; we keep the default).
    - Adam:  one fit per (``C``, ``lr``) on a small LR grid.

Usage:
    python experiments/run_effect_of_lbfgs_vs_adam.py
    python experiments/run_effect_of_lbfgs_vs_adam.py --dataset m-eurosat m-forestnet
    python experiments/run_effect_of_lbfgs_vs_adam.py --device cuda:3
"""

import argparse
import logging
import os
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from torchgeo_bench.datasets import get_datasets
from torchgeo_bench.linear import LogisticRegression
from torchgeo_bench.utils import extract_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

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

# Mirrors the main pipeline's c_range default (small, focused grid). Adjust
# via --c-values if you want a wider sweep.
DEFAULT_C_VALUES = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
DEFAULT_ADAM_LRS = [1e-3, 1e-2, 1e-1, 1.0]
DEFAULT_DATASETS = ["m-eurosat", "m-forestnet"]

# Match main.py's linear-probe call (LogisticRegression(C=c, max_iter=2000, tol=1e-6)).
MAX_ITER = 2000
TOL = 1e-6


def instantiate_model(model_cfg: dict, num_channels: int) -> torch.nn.Module:
    """Instantiate a model from its ``MODEL_CONFIGS`` entry."""
    target = model_cfg["_target_"]
    module_name, class_name = target.rsplit(".", 1)
    module = __import__(module_name, fromlist=[class_name])
    cls = getattr(module, class_name)

    kwargs = {k: v for k, v in model_cfg.items() if k not in ("_target_", "name")}
    kwargs["num_channels"] = num_channels
    return cls(**kwargs)


def parse_dataset_names(raw_names: list[str]) -> list[str]:
    """Parse comma- and space-separated dataset name lists."""
    out: list[str] = []
    for raw in raw_names:
        for name in raw.split(","):
            clean = name.strip()
            if clean and clean not in out:
                out.append(clean)
    return out


def parse_float_list(raw: str | None, default: list[float]) -> list[float]:
    """Parse a comma-separated float list (e.g. ``"1e-3,1e-2,1"``)."""
    if raw is None:
        return list(default)
    values = [float(x) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError(f"Empty float list: {raw!r}")
    return values


def build_configs(c_values: list[float], adam_lrs: list[float]) -> list[dict]:
    """Build a flat list of ``(solver, C, lr)`` fit configurations."""
    configs: list[dict] = []
    for c in c_values:
        configs.append({"solver": "lbfgs", "C": float(c), "lr": 1.0})
    for c in c_values:
        for lr in adam_lrs:
            configs.append({"solver": "adam", "C": float(c), "lr": float(lr)})
    return configs


def fit_one(
    cfg: dict,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    x_test: torch.Tensor,
    device: torch.device,
    seed: int,
) -> tuple[LogisticRegression, float, np.ndarray, np.ndarray, np.ndarray]:
    """Fit one config and return ``(clf, fit_seconds, train_pred, val_pred, test_pred)``."""
    clf = LogisticRegression(
        C=cfg["C"],
        lr=cfg["lr"],
        solver=cfg["solver"],
        max_iter=MAX_ITER,
        tol=TOL,
        random_state=seed,
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


def run_dataset(dataset_name: str, configs: list[dict], args: argparse.Namespace) -> str:
    """Run all (model, config) fits for one dataset and write its CSV."""
    output_path = os.path.join(args.output_dir, f"effect_of_lbfgs_vs_adam_{dataset_name}.csv")
    device = torch.device(args.device)

    completed: set[tuple] = set()
    all_rows: list[dict] = []
    if os.path.exists(output_path):
        existing = pd.read_csv(output_path)
        all_rows = existing.to_dict("records")
        completed = {(r["model"], r["solver"], float(r["C"]), float(r["lr"])) for r in all_rows}
        logger.info("Resume: found %d existing rows in %s", len(existing), output_path)

    logger.info("Loading %s dataset...", dataset_name)
    train_dataset, train_loader, val_loader, test_loader = get_datasets(
        dataset_name=dataset_name,
        partition_name="default",
        batch_size=64,
        return_val=True,
        image_size=args.image_size,
        interpolation="bilinear",
    )
    num_channels = train_dataset[0]["image"].shape[0]

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
        model = instantiate_model(model_cfg, num_channels)
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
                args.seed,
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

        df = pd.DataFrame(all_rows)
        df.to_csv(output_path, index=False)
        logger.info("  Saved %d rows to %s", len(all_rows), output_path)

    logger.info("Done. Final results for %s: %s", dataset_name, output_path)
    return output_path


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="LBFGS vs Adam linear-probing speed test")
    parser.add_argument(
        "--dataset",
        "--datasets",
        dest="datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help=(
            f"Dataset name(s), space- or comma-separated (default: {', '.join(DEFAULT_DATASETS)})."
        ),
    )
    parser.add_argument("--device", default="cuda:0", help="PyTorch device.")
    parser.add_argument(
        "--output-dir",
        default="results/effect_of_lbfgs_vs_adam",
        help="Output directory (one CSV per dataset).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--c-values",
        default=None,
        help=(
            "Comma-separated C values for both solvers "
            f"(default: {','.join(repr(c) for c in DEFAULT_C_VALUES)})."
        ),
    )
    parser.add_argument(
        "--adam-lrs",
        default=None,
        help=(
            "Comma-separated learning rates for the Adam branch "
            f"(default: {','.join(repr(lr) for lr in DEFAULT_ADAM_LRS)})."
        ),
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dataset_names = parse_dataset_names(args.datasets)
    if not dataset_names:
        raise ValueError("Provide at least one dataset name via --dataset.")

    c_values = parse_float_list(args.c_values, DEFAULT_C_VALUES)
    adam_lrs = parse_float_list(args.adam_lrs, DEFAULT_ADAM_LRS)
    configs = build_configs(c_values, adam_lrs)
    logger.info(
        "Configs per (model, dataset): %d LBFGS + %d Adam = %d total",
        len(c_values),
        len(c_values) * len(adam_lrs),
        len(configs),
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    logger.info("Running LBFGS-vs-Adam sweep on datasets: %s", ", ".join(dataset_names))

    output_paths = [run_dataset(name, configs, args) for name in dataset_names]
    logger.info("Finished sweeps: %s", ", ".join(output_paths))


if __name__ == "__main__":
    main()
