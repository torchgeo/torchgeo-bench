"""Feature extraction utilities for model benchmarking."""

import logging
from collections.abc import Callable

import numpy as np
import torch
from rich.progress import track
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def resolve_device(requested_device: str | torch.device) -> torch.device:
    """Resolve a requested device, falling back to CPU if CUDA is unavailable."""
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    return device


def extract_features(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str | torch.device,
    transforms: Callable[[torch.Tensor], torch.Tensor] | None = None,
    verbose: bool = True,
    description: str = "Extracting",
) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature embeddings and labels from a dataloader.

    Args:
        model: Model to use for feature extraction.
        dataloader: DataLoader yielding dicts with ``"image"`` and ``"label"`` keys.
        device: Device to run inference on.
        transforms: Optional transform applied to images before the model.
        verbose: Whether to display a progress bar.
        description: Progress bar label.

    Returns:
        Tuple of (features, labels) as NumPy arrays.
    """
    x_all = []
    y_all = []

    iterator = (
        track(dataloader, total=len(dataloader), description=description) if verbose else dataloader
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
            features = pooled_features(features)

        x_all.append(features)
        y_all.append(labels)

    x_all = np.concatenate(x_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)

    return x_all, y_all


def pooled_features(features: torch.Tensor | dict[str, torch.Tensor]) -> np.ndarray:
    """Convert supported model outputs to pooled NumPy features."""
    if isinstance(features, torch.Tensor):
        array = features.cpu().numpy()
    elif "norm" in features:
        array = features["norm"].cpu().numpy()
    elif "global_pool" in features:
        array = features["global_pool"].cpu().numpy()
    elif "head.global_pool" in features:
        array = features["head.global_pool"].cpu().numpy()
        if array.ndim == 3 and array.shape[1] == 1:
            array = array[:, 0, :]
    else:
        raise ValueError(f"Unexpected features format: {features.keys()}")

    if array.ndim == 1:
        array = array[np.newaxis, :]

    if array.ndim == 3:
        array = np.mean(array, axis=1, keepdims=False)
    return array
