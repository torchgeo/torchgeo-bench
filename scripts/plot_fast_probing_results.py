#!/usr/bin/env python3
"""Generate plots and LaTeX tables from fast probing benchmark results.

This script loads pre-computed benchmark data from results/fast_probing/
and generates:
1. Plots saved to paper/figures/
2. LaTeX tables saved to paper/tables/

Run after generate_knn_benchmark_data.py and generate_lr_benchmark_data.py.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Directories
RESULTS_DIR = Path("results/fast_probing")
FIG_DIR = Path("paper/figures")
TABLE_DIR = Path("paper/tables")

# Plot styling (no seaborn style)
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": True,
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)


def load_npy(filename: str) -> np.ndarray:
    """Load a .npy file from results directory."""
    return np.load(RESULTS_DIR / filename)


# =============================================================================
# KNN Plots
# =============================================================================


def plot_knn_time_vs_size() -> None:
    """Plot KNN time vs dataset size."""
    sizes = load_npy("knn_sizes.npy")
    sk_times = load_npy("knn_sk_times.npy")
    sk_std = load_npy("knn_sk_std.npy")
    faiss_cpu_times = load_npy("knn_faiss_cpu_times.npy")
    faiss_cpu_std = load_npy("knn_faiss_cpu_std.npy")
    faiss_gpu_times = load_npy("knn_faiss_gpu_times.npy")
    faiss_gpu_std = load_npy("knn_faiss_gpu_std.npy")

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(sizes, sk_times, label="sklearn KNN", marker="o")
    ax.fill_between(sizes, sk_times - sk_std, sk_times + sk_std, alpha=0.2)

    ax.plot(sizes, faiss_cpu_times, label="FaissKNN (CPU)", marker="s")
    ax.fill_between(
        sizes, faiss_cpu_times - faiss_cpu_std, faiss_cpu_times + faiss_cpu_std, alpha=0.2
    )

    ax.plot(sizes, faiss_gpu_times, label="FaissKNN (GPU)", marker="^")
    ax.fill_between(
        sizes, faiss_gpu_times - faiss_gpu_std, faiss_gpu_times + faiss_gpu_std, alpha=0.2
    )

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Dataset Size (samples)")
    ax.set_ylabel("Fit + Predict Time (s)")
    ax.legend(loc="best")
    fig.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"knn_time.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved knn_time.{png,pdf}")


def plot_knn_accuracy_vs_size() -> None:
    """Plot KNN accuracy vs dataset size."""
    sizes = load_npy("knn_sizes.npy")
    sk_acc = load_npy("knn_sk_acc_mean.npy")
    faiss_cpu_acc = load_npy("knn_faiss_cpu_acc_mean.npy")
    faiss_gpu_acc = load_npy("knn_faiss_gpu_acc_mean.npy")

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(sizes, sk_acc, label="sklearn KNN", marker="o")
    ax.plot(sizes, faiss_cpu_acc, label="FaissKNN (CPU)", marker="s")
    ax.plot(sizes, faiss_gpu_acc, label="FaissKNN (GPU)", marker="^")

    ax.set_xscale("log")
    ax.set_xlabel("Dataset Size (samples)")
    ax.set_ylabel("Training Accuracy")
    ax.legend(loc="best")
    fig.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"knn_accuracy.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved knn_accuracy.{png,pdf}")


def plot_knn_time_vs_features() -> None:
    """Plot KNN time vs feature dimensionality."""
    feat_dims = load_npy("knn_feat_dims.npy")
    sk_times = load_npy("knn_feat_sk_times.npy")
    sk_std = load_npy("knn_feat_sk_std.npy")
    faiss_cpu_times = load_npy("knn_feat_faiss_cpu_times.npy")
    faiss_cpu_std = load_npy("knn_feat_faiss_cpu_std.npy")
    faiss_gpu_times = load_npy("knn_feat_faiss_gpu_times.npy")
    faiss_gpu_std = load_npy("knn_feat_faiss_gpu_std.npy")

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(feat_dims, sk_times, label="sklearn KNN", marker="o")
    ax.fill_between(feat_dims, sk_times - sk_std, sk_times + sk_std, alpha=0.2)

    ax.plot(feat_dims, faiss_cpu_times, label="FaissKNN (CPU)", marker="s")
    ax.fill_between(
        feat_dims, faiss_cpu_times - faiss_cpu_std, faiss_cpu_times + faiss_cpu_std, alpha=0.2
    )

    ax.plot(feat_dims, faiss_gpu_times, label="FaissKNN (GPU)", marker="^")
    ax.fill_between(
        feat_dims, faiss_gpu_times - faiss_gpu_std, faiss_gpu_times + faiss_gpu_std, alpha=0.2
    )

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Feature Dimensionality")
    ax.set_ylabel("Fit + Predict Time (s)")
    ax.legend(loc="best")
    fig.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"knn_time_vs_features.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved knn_time_vs_features.{png,pdf}")


def plot_knn_accuracy_vs_features() -> None:
    """Plot KNN accuracy vs feature dimensionality."""
    feat_dims = load_npy("knn_feat_dims.npy")
    sk_acc = load_npy("knn_feat_sk_acc_mean.npy")
    faiss_cpu_acc = load_npy("knn_feat_faiss_cpu_acc_mean.npy")
    faiss_gpu_acc = load_npy("knn_feat_faiss_gpu_acc_mean.npy")

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(feat_dims, sk_acc, label="sklearn KNN", marker="o")
    ax.plot(feat_dims, faiss_cpu_acc, label="FaissKNN (CPU)", marker="s")
    ax.plot(feat_dims, faiss_gpu_acc, label="FaissKNN (GPU)", marker="^")

    ax.set_xscale("log")
    ax.set_xlabel("Feature Dimensionality")
    ax.set_ylabel("Training Accuracy")
    ax.legend(loc="best")
    fig.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"knn_accuracy_vs_features.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved knn_accuracy_vs_features.{png,pdf}")


# =============================================================================
# Logistic Regression Plots
# =============================================================================


def plot_lr_time_vs_size() -> None:
    """Plot LR time vs dataset size."""
    sizes = load_npy("lr_sizes.npy")
    sk_times = load_npy("lr_sk_times.npy")
    sk_std = load_npy("lr_sk_std.npy")
    torch_times = load_npy("lr_torch_times.npy")
    torch_std = load_npy("lr_torch_std.npy")

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(sizes, sk_times, label="sklearn LR", marker="o", color="C0")
    ax.fill_between(sizes, sk_times - sk_std, sk_times + sk_std, alpha=0.2, color="C0")

    ax.plot(sizes, torch_times, label="Torch LR (GPU)", marker="s", color="C3")
    ax.fill_between(sizes, torch_times - torch_std, torch_times + torch_std, alpha=0.2, color="C3")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Dataset Size (samples)")
    ax.set_ylabel("Fit Time (s)")
    ax.legend(loc="best")
    fig.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"lr_time.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved lr_time.{png,pdf}")


def plot_lr_accuracy_vs_size() -> None:
    """Plot LR accuracy vs dataset size."""
    sizes = load_npy("lr_sizes.npy")
    sk_acc = load_npy("lr_sk_acc_mean.npy")
    torch_acc = load_npy("lr_torch_acc_mean.npy")

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(sizes, sk_acc, label="sklearn LR", marker="o", color="C0")
    ax.plot(sizes, torch_acc, label="Torch LR (GPU)", marker="s", color="C3")

    ax.set_xscale("log")
    ax.set_xlabel("Dataset Size (samples)")
    ax.set_ylabel("Training Accuracy")
    ax.legend(loc="best")
    fig.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"lr_accuracy.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved lr_accuracy.{png,pdf}")


def plot_lr_time_vs_features() -> None:
    """Plot LR time vs feature dimensionality."""
    feat_dims = load_npy("lr_feat_dims.npy")
    sk_times = load_npy("lr_feat_sk_times.npy")
    sk_std = load_npy("lr_feat_sk_std.npy")
    torch_times = load_npy("lr_feat_torch_times.npy")
    torch_std = load_npy("lr_feat_torch_std.npy")

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(feat_dims, sk_times, label="sklearn LR", marker="o", color="C0")
    ax.fill_between(feat_dims, sk_times - sk_std, sk_times + sk_std, alpha=0.2, color="C0")

    ax.plot(feat_dims, torch_times, label="Torch LR (GPU)", marker="s", color="C3")
    ax.fill_between(
        feat_dims, torch_times - torch_std, torch_times + torch_std, alpha=0.2, color="C3"
    )

    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xlabel("Feature Dimensionality")
    ax.set_ylabel("Fit Time (s)")
    ax.legend(loc="best")
    fig.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"lr_time_vs_features.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved lr_time_vs_features.{png,pdf}")


def plot_lr_accuracy_vs_features() -> None:
    """Plot LR accuracy vs feature dimensionality."""
    feat_dims = load_npy("lr_feat_dims.npy")
    sk_acc = load_npy("lr_feat_sk_acc_mean.npy")
    sk_std = load_npy("lr_feat_sk_acc_std.npy")
    torch_acc = load_npy("lr_feat_torch_acc_mean.npy")
    torch_std = load_npy("lr_feat_torch_acc_std.npy")

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(feat_dims, sk_acc, label="sklearn LR", marker="o", color="C0")
    ax.fill_between(feat_dims, sk_acc - sk_std, sk_acc + sk_std, alpha=0.2, color="C0")

    ax.plot(feat_dims, torch_acc, label="Torch LR (GPU)", marker="s", color="C3")
    ax.fill_between(feat_dims, torch_acc - torch_std, torch_acc + torch_std, alpha=0.2, color="C3")

    ax.set_xscale("log")
    ax.set_xlabel("Feature Dimensionality")
    ax.set_ylabel("Training Accuracy")
    ax.legend(loc="best")
    fig.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(FIG_DIR / f"lr_accuracy_vs_features.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved lr_accuracy_vs_features.{png,pdf}")


# =============================================================================
# LaTeX Tables
# =============================================================================


def generate_knn_size_table() -> str:
    """Generate LaTeX table for KNN size experiment."""
    sizes = load_npy("knn_sizes.npy")
    sk_times = load_npy("knn_sk_times.npy")
    sk_std = load_npy("knn_sk_std.npy")
    faiss_cpu_times = load_npy("knn_faiss_cpu_times.npy")
    faiss_cpu_std = load_npy("knn_faiss_cpu_std.npy")
    faiss_gpu_times = load_npy("knn_faiss_gpu_times.npy")
    faiss_gpu_std = load_npy("knn_faiss_gpu_std.npy")
    sk_acc = load_npy("knn_sk_acc_mean.npy")
    faiss_cpu_acc = load_npy("knn_faiss_cpu_acc_mean.npy")
    faiss_gpu_acc = load_npy("knn_faiss_gpu_acc_mean.npy")

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{KNN Performance: Varying Dataset Size (256 features)}",
        r"\label{tab:knn_size}",
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"Samples & sklearn (s) & Faiss CPU (s) & Faiss GPU (s) & Speedup & Acc Match \\",
        r"\midrule",
    ]

    for i, n in enumerate(sizes):
        sk_t = sk_times[i]
        sk_s = sk_std[i]
        cpu_t = faiss_cpu_times[i]
        cpu_s = faiss_cpu_std[i]
        gpu_t = faiss_gpu_times[i]
        gpu_s = faiss_gpu_std[i]

        best_faiss = min(cpu_t, gpu_t)
        speedup = sk_t / best_faiss

        match = abs(sk_acc[i] - faiss_cpu_acc[i]) < 0.001 and abs(sk_acc[i] - faiss_gpu_acc[i]) < 0.001
        match_str = r"\checkmark" if match else r"\texttimes"

        lines.append(
            f"{int(n):,} & {sk_t:.2f}$\\pm${sk_s:.2f} & {cpu_t:.2f}$\\pm${cpu_s:.2f} & {gpu_t:.2f}$\\pm${gpu_s:.2f} & {speedup:.1f}$\\times$ & {match_str} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_knn_features_table() -> str:
    """Generate LaTeX table for KNN features experiment."""
    feat_dims = load_npy("knn_feat_dims.npy")
    sk_times = load_npy("knn_feat_sk_times.npy")
    sk_std = load_npy("knn_feat_sk_std.npy")
    faiss_cpu_times = load_npy("knn_feat_faiss_cpu_times.npy")
    faiss_cpu_std = load_npy("knn_feat_faiss_cpu_std.npy")
    faiss_gpu_times = load_npy("knn_feat_faiss_gpu_times.npy")
    faiss_gpu_std = load_npy("knn_feat_faiss_gpu_std.npy")
    sk_acc = load_npy("knn_feat_sk_acc_mean.npy")
    faiss_cpu_acc = load_npy("knn_feat_faiss_cpu_acc_mean.npy")
    faiss_gpu_acc = load_npy("knn_feat_faiss_gpu_acc_mean.npy")

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{KNN Performance: Varying Feature Dimensionality (20,000 samples)}",
        r"\label{tab:knn_features}",
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"Features & sklearn (s) & Faiss CPU (s) & Faiss GPU (s) & Speedup & Acc Match \\",
        r"\midrule",
    ]

    for i, n_feat in enumerate(feat_dims):
        sk_t = sk_times[i]
        sk_s = sk_std[i]
        cpu_t = faiss_cpu_times[i]
        cpu_s = faiss_cpu_std[i]
        gpu_t = faiss_gpu_times[i]
        gpu_s = faiss_gpu_std[i]

        best_faiss = min(cpu_t, gpu_t)
        speedup = sk_t / best_faiss

        match = abs(sk_acc[i] - faiss_cpu_acc[i]) < 0.001 and abs(sk_acc[i] - faiss_gpu_acc[i]) < 0.001
        match_str = r"\checkmark" if match else r"\texttimes"

        lines.append(
            f"{int(n_feat)} & {sk_t:.2f}$\\pm${sk_s:.2f} & {cpu_t:.2f}$\\pm${cpu_s:.2f} & {gpu_t:.2f}$\\pm${gpu_s:.2f} & {speedup:.1f}$\\times$ & {match_str} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_lr_size_table() -> str:
    """Generate LaTeX table for LR size experiment."""
    sizes = load_npy("lr_sizes.npy")
    sk_times = load_npy("lr_sk_times.npy")
    sk_std = load_npy("lr_sk_std.npy")
    torch_times = load_npy("lr_torch_times.npy")
    torch_std = load_npy("lr_torch_std.npy")
    sk_acc = load_npy("lr_sk_acc_mean.npy")
    torch_acc = load_npy("lr_torch_acc_mean.npy")

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Logistic Regression Performance: Varying Dataset Size (256 features)}",
        r"\label{tab:lr_size}",
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"Samples & sklearn (s) & Torch GPU (s) & Speedup & Acc $\Delta$ \\",
        r"\midrule",
    ]

    for i, n in enumerate(sizes):
        sk_t = sk_times[i]
        t_t = torch_times[i]
        speedup = sk_t / t_t if t_t > 0 else np.nan
        acc_delta = torch_acc[i] - sk_acc[i]

        lines.append(
            f"{int(n):,} & {sk_t:.2f}$\\pm${sk_std[i]:.2f} & {t_t:.2f}$\\pm${torch_std[i]:.2f} & {speedup:.1f}$\\times$ & {acc_delta:+.2f} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_lr_features_table() -> str:
    """Generate LaTeX table for LR features experiment."""
    feat_dims = load_npy("lr_feat_dims.npy")
    sk_times = load_npy("lr_feat_sk_times.npy")
    sk_std = load_npy("lr_feat_sk_std.npy")
    torch_times = load_npy("lr_feat_torch_times.npy")
    torch_std = load_npy("lr_feat_torch_std.npy")
    sk_acc = load_npy("lr_feat_sk_acc_mean.npy")
    torch_acc = load_npy("lr_feat_torch_acc_mean.npy")

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Logistic Regression Performance: Varying Feature Dimensionality (20,000 samples)}",
        r"\label{tab:lr_features}",
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"Features & sklearn (s) & Torch GPU (s) & Speedup & Acc $\Delta$ \\",
        r"\midrule",
    ]

    for i, n_feat in enumerate(feat_dims):
        sk_t = sk_times[i]
        sk_s = sk_std[i]
        t_t = torch_times[i]
        t_s = torch_std[i]
        speedup = sk_t / t_t if t_t > 0 else np.nan
        acc_delta = torch_acc[i] - sk_acc[i]

        lines.append(
            f"{int(n_feat)} & {sk_t:.2f}$\\pm${sk_s:.2f} & {t_t:.2f}$\\pm${t_s:.2f} & {speedup:.1f}$\\times$ & {acc_delta:+.2f} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_knn_results_table() -> str:
    """Generate combined LaTeX table for all KNN experiments."""
    # Size experiment data
    sizes = load_npy("knn_sizes.npy")
    sk_times = load_npy("knn_sk_times.npy")
    sk_std = load_npy("knn_sk_std.npy")
    faiss_cpu_times = load_npy("knn_faiss_cpu_times.npy")
    faiss_cpu_std = load_npy("knn_faiss_cpu_std.npy")
    faiss_gpu_times = load_npy("knn_faiss_gpu_times.npy")
    faiss_gpu_std = load_npy("knn_faiss_gpu_std.npy")
    sk_acc = load_npy("knn_sk_acc_mean.npy")
    faiss_cpu_acc = load_npy("knn_faiss_cpu_acc_mean.npy")
    faiss_gpu_acc = load_npy("knn_faiss_gpu_acc_mean.npy")

    # Feature experiment data
    feat_dims = load_npy("knn_feat_dims.npy")
    feat_sk_times = load_npy("knn_feat_sk_times.npy")
    feat_sk_std = load_npy("knn_feat_sk_std.npy")
    feat_faiss_cpu_times = load_npy("knn_feat_faiss_cpu_times.npy")
    feat_faiss_cpu_std = load_npy("knn_feat_faiss_cpu_std.npy")
    feat_faiss_gpu_times = load_npy("knn_feat_faiss_gpu_times.npy")
    feat_faiss_gpu_std = load_npy("knn_feat_faiss_gpu_std.npy")
    feat_sk_acc = load_npy("knn_feat_sk_acc_mean.npy")
    feat_faiss_cpu_acc = load_npy("knn_feat_faiss_cpu_acc_mean.npy")
    feat_faiss_gpu_acc = load_npy("knn_feat_faiss_gpu_acc_mean.npy")

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{KNN probe performance for varying dataset size and feature dimensionality.}",
        r"\label{tab:knn_results}",
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Setting & sklearn (s) & Faiss CPU (s) & Faiss GPU (s) & Speedup & Acc Match \\",
        r"\midrule",
        r"\multicolumn{6}{l}{\textbf{Varying dataset size} (256 features)} \\",
        r"\midrule",
    ]

    # Size experiment rows
    for i, n in enumerate(sizes):
        sk_t = sk_times[i]
        sk_s = sk_std[i]
        cpu_t = faiss_cpu_times[i]
        cpu_s = faiss_cpu_std[i]
        gpu_t = faiss_gpu_times[i]
        gpu_s = faiss_gpu_std[i]

        best_faiss = min(cpu_t, gpu_t)
        speedup = sk_t / best_faiss

        match = abs(sk_acc[i] - faiss_cpu_acc[i]) < 0.001 and abs(sk_acc[i] - faiss_gpu_acc[i]) < 0.001
        match_str = r"\checkmark" if match else r"\texttimes"

        lines.append(
            f"{int(n):,} samples & {sk_t:.2f}$\\pm${sk_s:.2f} & {cpu_t:.2f}$\\pm${cpu_s:.2f} & {gpu_t:.2f}$\\pm${gpu_s:.2f} & {speedup:.1f}$\\times$ & {match_str} \\\\"
        )

    lines.append(r"\midrule\midrule")
    lines.append(r"\multicolumn{6}{l}{\textbf{Varying feature dimensionality} (20,000 samples)} \\")
    lines.append(r"\midrule")

    # Feature experiment rows
    for i, n_feat in enumerate(feat_dims):
        sk_t = feat_sk_times[i]
        sk_s = feat_sk_std[i]
        cpu_t = feat_faiss_cpu_times[i]
        cpu_s = feat_faiss_cpu_std[i]
        gpu_t = feat_faiss_gpu_times[i]
        gpu_s = feat_faiss_gpu_std[i]

        best_faiss = min(cpu_t, gpu_t)
        speedup = sk_t / best_faiss

        match = abs(feat_sk_acc[i] - feat_faiss_cpu_acc[i]) < 0.001 and abs(feat_sk_acc[i] - feat_faiss_gpu_acc[i]) < 0.001
        match_str = r"\checkmark" if match else r"\texttimes"

        lines.append(
            f"{int(n_feat)} features & {sk_t:.2f}$\\pm${sk_s:.2f} & {cpu_t:.2f}$\\pm${cpu_s:.2f} & {gpu_t:.2f}$\\pm${gpu_s:.2f} & {speedup:.1f}$\\times$ & {match_str} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def generate_lr_results_table() -> str:
    """Generate combined LaTeX table for all LR experiments."""
    # Size experiment data
    sizes = load_npy("lr_sizes.npy")
    sk_times = load_npy("lr_sk_times.npy")
    sk_std_arr = load_npy("lr_sk_std.npy")
    torch_times = load_npy("lr_torch_times.npy")
    torch_std_arr = load_npy("lr_torch_std.npy")
    sk_acc = load_npy("lr_sk_acc_mean.npy")
    torch_acc = load_npy("lr_torch_acc_mean.npy")

    # Feature experiment data
    feat_dims = load_npy("lr_feat_dims.npy")
    feat_sk_times = load_npy("lr_feat_sk_times.npy")
    feat_sk_std = load_npy("lr_feat_sk_std.npy")
    feat_torch_times = load_npy("lr_feat_torch_times.npy")
    feat_torch_std = load_npy("lr_feat_torch_std.npy")
    feat_sk_acc = load_npy("lr_feat_sk_acc_mean.npy")
    feat_torch_acc = load_npy("lr_feat_torch_acc_mean.npy")

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Multinomial logistic regression (linear probe) performance for varying dataset size and feature dimensionality.}",
        r"\label{tab:lr_results}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Setting & sklearn (s) & Torch GPU (s) & Speedup & Acc $\Delta$ \\",
        r"\midrule",
        r"\multicolumn{5}{l}{\textbf{Varying dataset size} (256 features)} \\",
        r"\midrule",
    ]

    # Size experiment rows
    for i, n in enumerate(sizes):
        sk_t = sk_times[i]
        sk_s = sk_std_arr[i]
        t_t = torch_times[i]
        t_s = torch_std_arr[i]
        speedup = sk_t / t_t if t_t > 0 else np.nan
        acc_delta = torch_acc[i] - sk_acc[i]

        lines.append(
            f"{int(n):,} samples & {sk_t:.2f}$\\pm${sk_s:.2f} & {t_t:.2f}$\\pm${t_s:.2f} & {speedup:.1f}$\\times$ & {acc_delta:+.2f} \\\\"
        )

    lines.append(r"\midrule\midrule")
    lines.append(r"\multicolumn{5}{l}{\textbf{Varying feature dimensionality} (20,000 samples)} \\")
    lines.append(r"\midrule")

    # Feature experiment rows
    for i, n_feat in enumerate(feat_dims):
        sk_t = feat_sk_times[i]
        sk_s = feat_sk_std[i]
        t_t = feat_torch_times[i]
        t_s = feat_torch_std[i]
        speedup = sk_t / t_t if t_t > 0 else np.nan
        acc_delta = feat_torch_acc[i] - feat_sk_acc[i]

        lines.append(
            f"{int(n_feat)} features & {sk_t:.2f}$\\pm${sk_s:.2f} & {t_t:.2f}$\\pm${t_s:.2f} & {speedup:.1f}$\\times$ & {acc_delta:+.2f} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    return "\n".join(lines)


def save_tables() -> None:
    """Generate and save all LaTeX tables."""
    tables = {
        "knn_size.tex": generate_knn_size_table(),
        "knn_features.tex": generate_knn_features_table(),
        "lr_size.tex": generate_lr_size_table(),
        "lr_features.tex": generate_lr_features_table(),
        "knn_results.tex": generate_knn_results_table(),
        "lr_results.tex": generate_lr_results_table(),
    }

    for filename, content in tables.items():
        path = TABLE_DIR / filename
        path.write_text(content)
        logger.info(f"Saved {path}")


def main() -> None:
    """Generate all plots and tables."""
    # Ensure directories exist
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    # KNN plots
    logger.info("Generating KNN plots...")
    plot_knn_time_vs_size()
    plot_knn_accuracy_vs_size()
    plot_knn_time_vs_features()
    plot_knn_accuracy_vs_features()

    # LR plots
    logger.info("Generating LR plots...")
    plot_lr_time_vs_size()
    plot_lr_accuracy_vs_size()
    plot_lr_time_vs_features()
    plot_lr_accuracy_vs_features()

    # LaTeX tables
    logger.info("Generating LaTeX tables...")
    save_tables()

    logger.info("All plots and tables generated successfully!")


if __name__ == "__main__":
    main()
