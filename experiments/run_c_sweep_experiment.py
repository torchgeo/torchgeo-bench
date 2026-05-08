#!/usr/bin/env python
"""C-sweep analysis for linear probing.

Extracts features once per (model, dataset), then sweeps the L2
regularization strength ``C`` of :class:`torchgeo_bench.linear.LogisticRegression`
and records train/val/test accuracy for each ``C`` value.

Usage:
    python experiments/run_c_sweep_experiment.py --dataset m-eurosat
    python experiments/run_c_sweep_experiment.py --dataset m-so2sat --device cuda:3
    python experiments/run_c_sweep_experiment.py --dataset m-eurosat,m-forestnet
"""

import argparse
import logging
import os

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


def run_c_sweep(
    model_name: str,
    dataset_name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    c_values: np.ndarray,
    device: str,
    seed: int = 0,
) -> list[dict]:
    """Train ``LogisticRegression`` for each ``C`` and return per-C metrics."""
    rows = []
    x_train_t = torch.from_numpy(x_train)
    y_train_t = torch.from_numpy(y_train).long()
    x_val_t = torch.from_numpy(x_val)
    x_test_t = torch.from_numpy(x_test)

    for c in tqdm(c_values, desc=f"  C-sweep ({model_name})", leave=False):
        clf = LogisticRegression(
            C=float(c),
            max_iter=2000,
            tol=1e-6,
            random_state=seed,
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


def run_dataset_sweep(dataset_name: str, args: argparse.Namespace) -> str:
    """Run the C sweep for one dataset and write its CSV."""
    output_path = os.path.join(args.output_dir, f"c_sweep_{dataset_name}.csv")
    device = torch.device(args.device)

    completed_models: set[str] = set()
    all_rows: list[dict] = []
    if os.path.exists(output_path):
        existing_df = pd.read_csv(output_path)
        all_rows = existing_df.to_dict("records")
        completed_models = set(existing_df["model"].unique())
        logger.info(
            "Resume: found %d completed models in %s: %s",
            len(completed_models),
            output_path,
            completed_models,
        )

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
        if model_name in completed_models:
            logger.info("=== %s/%s === skipping, already computed", dataset_name, model_name)
            continue

        logger.info("=== %s/%s ===", dataset_name, model_name)
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
            C_VALUES,
            args.device,
            args.seed,
        )
        all_rows.extend(rows)

        df = pd.DataFrame(all_rows)
        df.to_csv(output_path, index=False)
        logger.info("  Saved %d rows to %s", len(all_rows), output_path)

    logger.info("Done. Final results for %s: %s", dataset_name, output_path)
    return output_path


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="C-sweep for linear probing")
    parser.add_argument(
        "--dataset",
        "--datasets",
        dest="datasets",
        nargs="+",
        default=["m-eurosat"],
        help="Dataset name(s), space- or comma-separated (default: m-eurosat).",
    )
    parser.add_argument("--device", default="cuda:0", help="PyTorch device.")
    parser.add_argument(
        "--output-dir",
        default="results/c_sweep_results",
        help="Output directory.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    dataset_names = parse_dataset_names(args.datasets)
    if not dataset_names:
        raise ValueError("Provide at least one dataset name via --dataset.")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    logger.info("Running C sweep for datasets: %s", ", ".join(dataset_names))

    output_paths = [run_dataset_sweep(name, args) for name in dataset_names]
    logger.info("Finished C sweeps: %s", ", ".join(output_paths))


if __name__ == "__main__":
    main()
