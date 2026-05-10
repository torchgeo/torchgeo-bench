"""Feature extraction utilities for model benchmarking."""

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


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

    iterator = tqdm(dataloader, total=len(dataloader)) if verbose else dataloader

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
                    features = features["head.global_pool"].cpu().numpy().squeeze()
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
