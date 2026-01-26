#!/usr/bin/env python3
"""Generate KNN benchmark data comparing sklearn vs FaissKNN.

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
from faissknn import FaissKNNClassifier
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm

from benchmark_utils import get_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_size_experiment(
    results_dir: Path,
    size_multipliers: list[int],
    n_classes: int = 20,
    n_features: int = 256,
    n_neighbors: int = 5,
    repeats: int = 5,
    device: str | None = None,
) -> None:
    """Run benchmark varying dataset sizes.

    Args:
        results_dir: Directory to save results.
        size_multipliers: List of multipliers for dataset size (samples = 100 * multiplier).
        n_classes: Number of classes.
        n_features: Number of features (fixed).
        n_neighbors: Number of neighbors for KNN.
        repeats: Number of repeat runs for timing.
        device: Device for FaissKNN GPU (e.g., "cuda:0"). Auto-detected if None.
    """
    logger.info("Running size experiment (varying dataset sizes)")

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    use_gpu = device.startswith("cuda")
    logger.info(f"Using device: {device} (GPU enabled: {use_gpu})")
    sk_times: list[float] = []
    sk_std: list[float] = []
    sk_acc_mean: list[float] = []
    sk_acc_std: list[float] = []
    faiss_cpu_times: list[float] = []
    faiss_cpu_std: list[float] = []
    faiss_cpu_acc_mean: list[float] = []
    faiss_cpu_acc_std: list[float] = []
    faiss_gpu_times: list[float] = []
    faiss_gpu_std: list[float] = []
    faiss_gpu_acc_mean: list[float] = []
    faiss_gpu_acc_std: list[float] = []


    for size_multiplier in tqdm(size_multipliers, desc="Dataset sizes"):
        X, y = get_dataset(
            n_samples=100 * size_multiplier, n_classes=n_classes, n_features=n_features
        )

        sk_runs: list[float] = []
        faiss_cpu_runs: list[float] = []
        faiss_gpu_runs: list[float] = []
        sk_acc_runs: list[float] = []
        faiss_cpu_acc_runs: list[float] = []
        faiss_gpu_acc_runs: list[float] = []

        for _ in range(repeats):
            # sklearn KNN
            t0 = time.perf_counter()
            sk_model = KNeighborsClassifier(n_neighbors=n_neighbors)
            sk_model.fit(X, y)
            sk_preds = sk_model.predict(X)
            sk_runs.append(time.perf_counter() - t0)
            sk_acc_runs.append(float((sk_preds == y).mean()))

            # FaissKNN CPU
            t1 = time.perf_counter()
            faiss_cpu_model = FaissKNNClassifier(n_neighbors=n_neighbors, device="cpu")
            faiss_cpu_model.fit(X, y)
            faiss_cpu_preds = faiss_cpu_model.predict(X)
            faiss_cpu_runs.append(time.perf_counter() - t1)
            faiss_cpu_acc_runs.append(float((faiss_cpu_preds == y).mean()))

            # FaissKNN GPU
            if use_gpu:
                torch.cuda.synchronize()
                t2 = time.perf_counter()
                faiss_gpu_model = FaissKNNClassifier(n_neighbors=n_neighbors, device="cuda")
                faiss_gpu_model.fit(X, y)
                faiss_gpu_preds = faiss_gpu_model.predict(X)
                torch.cuda.synchronize()
                faiss_gpu_runs.append(time.perf_counter() - t2)
                faiss_gpu_acc_runs.append(float((faiss_gpu_preds == y).mean()))

        sizes.append(X.shape[0])
        sk_times.append(float(np.mean(sk_runs)))
        sk_std.append(float(np.std(sk_runs, ddof=0)))
        sk_acc_mean.append(float(np.mean(sk_acc_runs)))
        sk_acc_std.append(float(np.std(sk_acc_runs, ddof=0)))
        faiss_cpu_times.append(float(np.mean(faiss_cpu_runs)))
        faiss_cpu_std.append(float(np.std(faiss_cpu_runs, ddof=0)))
        faiss_cpu_acc_mean.append(float(np.mean(faiss_cpu_acc_runs)))
        faiss_cpu_acc_std.append(float(np.std(faiss_cpu_acc_runs, ddof=0)))

        if use_gpu:
            faiss_gpu_times.append(float(np.mean(faiss_gpu_runs)))
            faiss_gpu_std.append(float(np.std(faiss_gpu_runs, ddof=0)))
            faiss_gpu_acc_mean.append(float(np.mean(faiss_gpu_acc_runs)))
            faiss_gpu_acc_std.append(float(np.std(faiss_gpu_acc_runs, ddof=0)))

    # Save results
    np.save(results_dir / "knn_sizes.npy", np.array(sizes))
    np.save(results_dir / "knn_sk_times.npy", np.array(sk_times))
    np.save(results_dir / "knn_sk_std.npy", np.array(sk_std))
    np.save(results_dir / "knn_sk_acc_mean.npy", np.array(sk_acc_mean))
    np.save(results_dir / "knn_sk_acc_std.npy", np.array(sk_acc_std))
    np.save(results_dir / "knn_faiss_cpu_times.npy", np.array(faiss_cpu_times))
    np.save(results_dir / "knn_faiss_cpu_std.npy", np.array(faiss_cpu_std))
    np.save(results_dir / "knn_faiss_cpu_acc_mean.npy", np.array(faiss_cpu_acc_mean))
    np.save(results_dir / "knn_faiss_cpu_acc_std.npy", np.array(faiss_cpu_acc_std))

    if use_gpu:
        np.save(results_dir / "knn_faiss_gpu_times.npy", np.array(faiss_gpu_times))
        np.save(results_dir / "knn_faiss_gpu_std.npy", np.array(faiss_gpu_std))
        np.save(results_dir / "knn_faiss_gpu_acc_mean.npy", np.array(faiss_gpu_acc_mean))
        np.save(results_dir / "knn_faiss_gpu_acc_std.npy", np.array(faiss_gpu_acc_std))

    logger.info(f"Size experiment results saved to {results_dir}")


def run_feature_experiment(
    results_dir: Path,
    feature_dims: list[int],
    n_samples: int = 20_000,
    n_classes: int = 20,
    n_neighbors: int = 5,
    repeats: int = 5,
    device: str | None = None,
) -> None:
    """Run benchmark varying feature dimensionality.

    Args:
        results_dir: Directory to save results.
        feature_dims: List of feature dimensions to test.
        n_samples: Number of samples (fixed).
        n_classes: Number of classes.
        n_neighbors: Number of neighbors for KNN.
        repeats: Number of repeat runs for timing.
        device: Device for FaissKNN GPU (e.g., "cuda:0"). Auto-detected if None.
    """
    logger.info(f"Running feature experiment (n_samples={n_samples:,})")

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    use_gpu = device.startswith("cuda")

    feat_dims: list[int] = []
    feat_sk_times: list[float] = []
    feat_sk_std: list[float] = []
    feat_sk_acc_mean: list[float] = []
    feat_sk_acc_std: list[float] = []
    feat_faiss_cpu_times: list[float] = []
    feat_faiss_cpu_std: list[float] = []
    feat_faiss_cpu_acc_mean: list[float] = []
    feat_faiss_cpu_acc_std: list[float] = []
    feat_faiss_gpu_times: list[float] = []
    feat_faiss_gpu_std: list[float] = []
    feat_faiss_gpu_acc_mean: list[float] = []
    feat_faiss_gpu_acc_std: list[float] = []


    for n_features in tqdm(feature_dims, desc="Feature dims"):
        X, y = get_dataset(n_samples=n_samples, n_classes=n_classes, n_features=n_features)

        sk_runs: list[float] = []
        faiss_cpu_runs: list[float] = []
        faiss_gpu_runs: list[float] = []
        sk_acc_runs: list[float] = []
        faiss_cpu_acc_runs: list[float] = []
        faiss_gpu_acc_runs: list[float] = []

        for _ in range(repeats):
            # sklearn KNN
            t0 = time.perf_counter()
            sk_model = KNeighborsClassifier(n_neighbors=n_neighbors)
            sk_model.fit(X, y)
            sk_preds = sk_model.predict(X)
            sk_runs.append(time.perf_counter() - t0)
            sk_acc_runs.append(float((sk_preds == y).mean()))

            # FaissKNN CPU
            t1 = time.perf_counter()
            faiss_cpu_model = FaissKNNClassifier(n_neighbors=n_neighbors, device="cpu")
            faiss_cpu_model.fit(X, y)
            faiss_cpu_preds = faiss_cpu_model.predict(X)
            faiss_cpu_runs.append(time.perf_counter() - t1)
            faiss_cpu_acc_runs.append(float((faiss_cpu_preds == y).mean()))

            # FaissKNN GPU
            if use_gpu:
                torch.cuda.synchronize(device)
                t2 = time.perf_counter()
                faiss_gpu_model = FaissKNNClassifier(n_neighbors=n_neighbors, device=device)
                faiss_gpu_model.fit(X, y)
                faiss_gpu_preds = faiss_gpu_model.predict(X)
                torch.cuda.synchronize(device)
                faiss_gpu_runs.append(time.perf_counter() - t2)
                faiss_gpu_acc_runs.append(float((faiss_gpu_preds == y).mean()))

        feat_dims.append(n_features)
        feat_sk_times.append(float(np.mean(sk_runs)))
        feat_sk_std.append(float(np.std(sk_runs, ddof=0)))
        feat_sk_acc_mean.append(float(np.mean(sk_acc_runs)))
        feat_sk_acc_std.append(float(np.std(sk_acc_runs, ddof=0)))
        feat_faiss_cpu_times.append(float(np.mean(faiss_cpu_runs)))
        feat_faiss_cpu_std.append(float(np.std(faiss_cpu_runs, ddof=0)))
        feat_faiss_cpu_acc_mean.append(float(np.mean(faiss_cpu_acc_runs)))
        feat_faiss_cpu_acc_std.append(float(np.std(faiss_cpu_acc_runs, ddof=0)))

        if use_gpu:
            feat_faiss_gpu_times.append(float(np.mean(faiss_gpu_runs)))
            feat_faiss_gpu_std.append(float(np.std(faiss_gpu_runs, ddof=0)))
            feat_faiss_gpu_acc_mean.append(float(np.mean(faiss_gpu_acc_runs)))
            feat_faiss_gpu_acc_std.append(float(np.std(faiss_gpu_acc_runs, ddof=0)))

    # Save results
    np.save(results_dir / "knn_feat_dims.npy", np.array(feat_dims))
    np.save(results_dir / "knn_feat_sk_times.npy", np.array(feat_sk_times))
    np.save(results_dir / "knn_feat_sk_std.npy", np.array(feat_sk_std))
    np.save(results_dir / "knn_feat_sk_acc_mean.npy", np.array(feat_sk_acc_mean))
    np.save(results_dir / "knn_feat_sk_acc_std.npy", np.array(feat_sk_acc_std))
    np.save(results_dir / "knn_feat_faiss_cpu_times.npy", np.array(feat_faiss_cpu_times))
    np.save(results_dir / "knn_feat_faiss_cpu_std.npy", np.array(feat_faiss_cpu_std))
    np.save(results_dir / "knn_feat_faiss_cpu_acc_mean.npy", np.array(feat_faiss_cpu_acc_mean))
    np.save(results_dir / "knn_feat_faiss_cpu_acc_std.npy", np.array(feat_faiss_cpu_acc_std))

    if use_gpu:
        np.save(results_dir / "knn_feat_faiss_gpu_times.npy", np.array(feat_faiss_gpu_times))
        np.save(results_dir / "knn_feat_faiss_gpu_std.npy", np.array(feat_faiss_gpu_std))
        np.save(results_dir / "knn_feat_faiss_gpu_acc_mean.npy", np.array(feat_faiss_gpu_acc_mean))
        np.save(results_dir / "knn_feat_faiss_gpu_acc_std.npy", np.array(feat_faiss_gpu_acc_std))

    logger.info(f"Feature experiment results saved to {results_dir}")


def main() -> None:
    """Run all KNN benchmark experiments."""
    results_dir = Path("results/fast_probing")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Experiment 1: Varying dataset sizes
    run_size_experiment(
        results_dir=results_dir,
        size_multipliers=[10, 20, 50, 100, 200, 500, 1000],
        n_classes=20,
        n_features=256,
        n_neighbors=5,
        repeats=5,
    )

    # Experiment 2: Varying feature dimensionality
    run_feature_experiment(
        results_dir=results_dir,
        feature_dims=[32, 64, 128, 256, 512, 1024, 2048],
        n_samples=20_000,
        n_classes=20,
        n_neighbors=5,
        repeats=5,
    )

    logger.info("All KNN experiments complete!")


if __name__ == "__main__":
    main()
