#!/usr/bin/env python3
"""Generate Logistic Regression benchmark data comparing sklearn vs PyTorch.

This script runs benchmarks for:
1. Varying dataset sizes (fixed features=256)
2. Varying feature dimensionality (fixed samples=20,000)

Results are saved to results/fast_probing/ as .npy files.
"""

import logging
import time
from pathlib import Path

import numpy as np
import torch
from benchmark_utils import get_dataset
from sklearn.linear_model import LogisticRegression as SkLogReg
from tqdm import tqdm

from src.linear import LogisticRegression as TorchLR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_size_experiment(
    results_dir: Path,
    size_multipliers: list[int],
    n_classes: int = 20,
    n_features: int = 256,
    max_iter: int = 200,
    repeats: int = 5,
    device: str | None = None,
) -> None:
    """Run benchmark varying dataset sizes.

    Args:
        results_dir: Directory to save results.
        size_multipliers: List of multipliers for dataset size (samples = 100 * multiplier).
        n_classes: Number of classes.
        n_features: Number of features (fixed).
        max_iter: Maximum iterations for logistic regression.
        repeats: Number of repeat runs for timing.
        device: Device for PyTorch model (e.g., "cuda:0"). Auto-detected if None.
    """
    logger.info("Running size experiment (varying dataset sizes)")

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    sizes: list[int] = []
    sk_times: list[float] = []
    sk_std: list[float] = []
    sk_acc_mean: list[float] = []
    sk_acc_std: list[float] = []
    torch_times: list[float] = []
    torch_std: list[float] = []
    torch_acc_mean: list[float] = []
    torch_acc_std: list[float] = []

    for size_multiplier in tqdm(size_multipliers, desc="Dataset sizes"):
        X, y = get_dataset(
            n_samples=100 * size_multiplier, n_classes=n_classes, n_features=n_features
        )
        X_torch = torch.from_numpy(X)
        y_torch = torch.from_numpy(y)

        sk_runs: list[float] = []
        torch_runs: list[float] = []
        sk_acc_runs: list[float] = []
        torch_acc_runs: list[float] = []

        for r in range(repeats):
            # sklearn timing + accuracy
            t0 = time.perf_counter()
            sk_model = SkLogReg(max_iter=max_iter, solver="lbfgs", random_state=r)
            sk_model.fit(X, y)
            sk_runs.append(time.perf_counter() - t0)
            sk_preds = sk_model.predict(X)
            sk_acc_runs.append(float((sk_preds == y).mean()))

            # torch timing + accuracy
            torch_model = TorchLR(max_iter=max_iter, solver="lbfgs", device=device)
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            torch_model.fit(X_torch, y_torch)
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            torch_runs.append(time.perf_counter() - t1)
            torch_preds = torch_model.predict(X_torch)
            torch_acc_runs.append(float((torch_preds == y).mean()))

        sizes.append(X.shape[0])
        sk_times.append(float(np.mean(sk_runs)))
        sk_std.append(float(np.std(sk_runs, ddof=0)))
        sk_acc_mean.append(float(np.mean(sk_acc_runs)))
        sk_acc_std.append(float(np.std(sk_acc_runs, ddof=0)))
        torch_times.append(float(np.mean(torch_runs)))
        torch_std.append(float(np.std(torch_runs, ddof=0)))
        torch_acc_mean.append(float(np.mean(torch_acc_runs)))
        torch_acc_std.append(float(np.std(torch_acc_runs, ddof=0)))

    # Save results
    np.save(results_dir / "lr_sizes.npy", np.array(sizes))
    np.save(results_dir / "lr_sk_times.npy", np.array(sk_times))
    np.save(results_dir / "lr_sk_std.npy", np.array(sk_std))
    np.save(results_dir / "lr_sk_acc_mean.npy", np.array(sk_acc_mean))
    np.save(results_dir / "lr_sk_acc_std.npy", np.array(sk_acc_std))
    np.save(results_dir / "lr_torch_times.npy", np.array(torch_times))
    np.save(results_dir / "lr_torch_std.npy", np.array(torch_std))
    np.save(results_dir / "lr_torch_acc_mean.npy", np.array(torch_acc_mean))
    np.save(results_dir / "lr_torch_acc_std.npy", np.array(torch_acc_std))

    logger.info(f"Size experiment results saved to {results_dir}")


def run_feature_experiment(
    results_dir: Path,
    feature_dims: list[int],
    n_samples: int = 20_000,
    n_classes: int = 20,
    max_iter: int = 200,
    repeats: int = 5,
    device: str | None = None,
) -> None:
    """Run benchmark varying feature dimensionality.

    Args:
        results_dir: Directory to save results.
        feature_dims: List of feature dimensions to test.
        n_samples: Number of samples (fixed).
        n_classes: Number of classes.
        max_iter: Maximum iterations for logistic regression.
        repeats: Number of repeat runs for timing.
        device: Device for PyTorch model. Auto-detected if None.
    """
    logger.info(f"Running feature experiment (n_samples={n_samples:,})")

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    feat_dims: list[int] = []
    feat_sk_times: list[float] = []
    feat_sk_std: list[float] = []
    feat_sk_acc_mean: list[float] = []
    feat_sk_acc_std: list[float] = []
    feat_torch_times: list[float] = []
    feat_torch_std: list[float] = []
    feat_torch_acc_mean: list[float] = []
    feat_torch_acc_std: list[float] = []

    for n_features in tqdm(feature_dims, desc="Feature dims"):
        X, y = get_dataset(n_samples=n_samples, n_classes=n_classes, n_features=n_features)
        X_torch = torch.from_numpy(X)
        y_torch = torch.from_numpy(y)

        sk_runs: list[float] = []
        torch_runs: list[float] = []
        sk_acc_runs: list[float] = []
        torch_acc_runs: list[float] = []

        for r in range(repeats):
            # sklearn timing + accuracy
            t0 = time.perf_counter()
            sk_model = SkLogReg(max_iter=max_iter, solver="lbfgs", random_state=r)
            sk_model.fit(X, y)
            sk_runs.append(time.perf_counter() - t0)
            sk_preds = sk_model.predict(X)
            sk_acc_runs.append(float((sk_preds == y).mean()))

            # torch timing + accuracy
            torch_model = TorchLR(max_iter=max_iter, solver="lbfgs", device=device)
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            t1 = time.perf_counter()
            torch_model.fit(X_torch, y_torch)
            if device.startswith("cuda"):
                torch.cuda.synchronize(device)
            torch_runs.append(time.perf_counter() - t1)
            torch_preds = torch_model.predict(X_torch)
            torch_acc_runs.append(float((torch_preds == y).mean()))

        feat_dims.append(n_features)
        feat_sk_times.append(float(np.mean(sk_runs)))
        feat_sk_std.append(float(np.std(sk_runs, ddof=0)))
        feat_sk_acc_mean.append(float(np.mean(sk_acc_runs)))
        feat_sk_acc_std.append(float(np.std(sk_acc_runs, ddof=0)))
        feat_torch_times.append(float(np.mean(torch_runs)))
        feat_torch_std.append(float(np.std(torch_runs, ddof=0)))
        feat_torch_acc_mean.append(float(np.mean(torch_acc_runs)))
        feat_torch_acc_std.append(float(np.std(torch_acc_runs, ddof=0)))

    # Save results
    np.save(results_dir / "lr_feat_dims.npy", np.array(feat_dims))
    np.save(results_dir / "lr_feat_sk_times.npy", np.array(feat_sk_times))
    np.save(results_dir / "lr_feat_sk_std.npy", np.array(feat_sk_std))
    np.save(results_dir / "lr_feat_sk_acc_mean.npy", np.array(feat_sk_acc_mean))
    np.save(results_dir / "lr_feat_sk_acc_std.npy", np.array(feat_sk_acc_std))
    np.save(results_dir / "lr_feat_torch_times.npy", np.array(feat_torch_times))
    np.save(results_dir / "lr_feat_torch_std.npy", np.array(feat_torch_std))
    np.save(results_dir / "lr_feat_torch_acc_mean.npy", np.array(feat_torch_acc_mean))
    np.save(results_dir / "lr_feat_torch_acc_std.npy", np.array(feat_torch_acc_std))

    logger.info(f"Feature experiment results saved to {results_dir}")


def main() -> None:
    """Run all logistic regression benchmark experiments."""
    results_dir = Path("results/fast_probing")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Experiment 1: Varying dataset sizes
    run_size_experiment(
        results_dir=results_dir,
        size_multipliers=[10, 20, 50, 100, 200, 500, 1000],
        n_classes=20,
        n_features=256,
        max_iter=200,
        repeats=5,
    )

    # Experiment 2: Varying feature dimensionality
    run_feature_experiment(
        results_dir=results_dir,
        feature_dims=[32, 64, 128, 256, 512, 1024, 2048],
        n_samples=20_000,
        n_classes=20,
        max_iter=200,
        repeats=5,
    )

    logger.info("All LR experiments complete!")


if __name__ == "__main__":
    main()
