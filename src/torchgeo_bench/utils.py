"""Feature extraction utilities for model benchmarking."""

import numpy as np
import torch
from rich.progress import track
from torch.utils.data import DataLoader


def extract_features(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str | torch.device,
    transforms: object | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature embeddings and labels from a dataloader.

    Args:
        model: Model to use for feature extraction.
        dataloader: DataLoader yielding dicts with ``"image"`` and ``"label"`` keys.
        device: Device to run inference on.
        transforms: Optional transform applied to images before the model.
        verbose: Whether to display a progress bar.

    Returns:
        Tuple of (features, labels) as NumPy arrays.
    """
    x_all = []
    y_all = []

    iterator = (
        track(dataloader, total=len(dataloader), description="Extracting")
        if verbose
        else dataloader
    )

    for batch in iterator:
        images = batch["image"].to(device)
        if "label" not in batch:
            raise KeyError(
                "Batch is missing 'label' key. extract_features() is a classification "
                "utility; for segmentation use "
                "SegmentationProbe.extract_segmentation_features() instead."
            )
        labels = batch["label"].numpy()

        if transforms is not None:
            images = transforms(images)

        with torch.no_grad(), torch.inference_mode():
            features = model(images)
            if isinstance(features, torch.Tensor):
                features = features.cpu().numpy()
            else:
                if "norm" in features:
                    features = features["norm"].cpu().numpy()
                elif "global_pool" in features:
                    features = features["global_pool"].cpu().numpy()
                elif "head.global_pool" in features:
                    features = features["head.global_pool"].cpu().numpy()
                    if features.ndim == 3 and features.shape[1] == 1:
                        features = features[:, 0, :]
                else:
                    raise ValueError(f"Unexpected features format: {features.keys()}")

            if features.ndim == 1:
                features = features[np.newaxis, :]

            if features.ndim == 3:
                features = np.mean(features, axis=1, keepdims=False)

        x_all.append(features)
        y_all.append(labels)

    x_all = np.concatenate(x_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    return x_all, y_all


def stratified_subsample_indices(
    y: np.ndarray,
    fraction: float,
    seed: int,
    min_per_class: int = 1,
) -> np.ndarray:
    """Return sorted positions of a stratified, nested subsample of ``y``.

    For each class, one permutation of that class's positions is drawn from an
    RNG seeded only by ``seed`` (never by the model or the fraction), and the
    first ``min(n_c, max(min_per_class, round(fraction * n_c)))`` positions are
    kept. Because every fraction reads a prefix of the same per-class
    permutation, subsamples are automatically **nested**
    (``f=0.05 ⊂ f=0.1 ⊂ … ⊂ f=1.0``) and **identical across models**. The
    floor guarantees every class survives at every fraction.

    Args:
        y: 1D integer label array in fixed dataset order.
        fraction: Label budget in ``(0, 1]``.
        seed: Per-repeat seed (e.g. ``cfg.seed + repeat_index``).
        min_per_class: Minimum kept examples per class.

    Returns:
        A sorted 1D ``int64`` array of positions into ``y``.

    Raises:
        ValueError: If ``fraction`` is outside ``(0, 1]``.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}.")
    n = len(y)
    if fraction == 1.0:
        return np.arange(n, dtype=np.int64)

    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for cls in np.unique(y):
        positions = np.flatnonzero(y == cls)
        n_c = len(positions)
        keep = min(n_c, max(min_per_class, round(fraction * n_c)))
        perm = rng.permutation(positions)
        selected.append(perm[:keep])
    return np.sort(np.concatenate(selected)).astype(np.int64)
