"""Bootstrap confidence intervals for benchmark metrics."""

import numpy as np
import torch


def bootstrap_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 1000,
    ci: float = 95.0,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Return accuracy and its bootstrap confidence interval."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    indices = rng.integers(0, n, size=(n_boot, n))
    scores = (y_true[indices] == y_pred[indices]).mean(axis=1).astype(np.float32)
    mean = float((y_true == y_pred).mean())
    lower = (100 - ci) / 2
    upper = 100 - lower
    return mean, float(np.percentile(scores, lower)), float(np.percentile(scores, upper))


def bootstrap_map(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    n_boot: int = 1000,
    ci: float = 95.0,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Return micro-averaged mean average precision and its bootstrap interval."""
    from sklearn.metrics import average_precision_score

    rng = np.random.default_rng(seed)
    n = len(y_true)
    mean = float(average_precision_score(y_true, y_scores, average="micro"))
    scores: list[float] = []
    for _ in range(n_boot):
        indices = rng.integers(0, n, size=n)
        sampled_labels = y_true[indices]
        if sampled_labels.sum() == 0:
            continue
        scores.append(average_precision_score(sampled_labels, y_scores[indices], average="micro"))
    if not scores:
        return mean, mean, mean
    values = np.array(scores, dtype=np.float32)
    lower = (100 - ci) / 2
    upper = 100 - lower
    return mean, float(np.percentile(values, lower)), float(np.percentile(values, upper))


def bootstrap_miou(
    confusion_matrices: torch.Tensor,
    n_boot: int = 1000,
    ci: float = 95.0,
    seed: int | None = None,
) -> tuple[float, float]:
    """Return a dataset-level mIoU bootstrap interval."""
    if n_boot < 1:
        return float("nan"), float("nan")
    if confusion_matrices.ndim != 3 or confusion_matrices.shape[0] == 0:
        raise ValueError("Expected non-empty per-image confusion matrices with shape (N, C, C).")
    if confusion_matrices.shape[1] != confusion_matrices.shape[2]:
        raise ValueError("Confusion matrices must be square.")

    confusions = confusion_matrices.to(dtype=torch.float64, device="cpu")
    generator = torch.Generator().manual_seed(seed if seed is not None else 0)
    n_samples = len(confusions)
    scores = torch.empty(n_boot, dtype=torch.float64)
    for index in range(n_boot):
        sample = torch.randint(n_samples, (n_samples,), generator=generator)
        confusion = confusions[sample].sum(dim=0)
        intersection = confusion.diagonal()
        union = confusion.sum(dim=0) + confusion.sum(dim=1) - intersection
        present = union > 0
        scores[index] = (
            (intersection[present] / union[present]).mean() if present.any() else float("nan")
        )

    lower = (100.0 - ci) / 2.0
    return float(torch.nanquantile(scores, lower / 100.0)), float(
        torch.nanquantile(scores, 1 - lower / 100.0)
    )
