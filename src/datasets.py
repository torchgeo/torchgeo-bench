"""Dataset utilities for torchgeo-bench.

This module provides a unified interface to load GeoBench datasets using
the lightweight GeoBenchDataset class.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .geobench_dataset import GeoBenchDataset

NUM_CLASSES_PER_DATASET = {
    "m-forestnet": 12,
    "m-eurosat": 10,
    "m-pv4ger": 2,
    "m-brick-kiln": 2,
    "m-so2sat": 17,
    # "m-bigearthnet": None,  # TODO: Handle BigEarthNet separately
}

PARTITION_NAMES = [
    "0.01x_train",
    "0.02x_train",
    "0.05x_train",
    "0.10x_train",
    "0.20x_train",
    "0.50x_train",
    "1.00x_train",
    "default",
]

# Default GeoBench data root - override via environment variable if needed
DEFAULT_GEOBENCH_ROOT = "data/classification_v1.0"


def get_datasets(
    dataset_name: str = "m-forestnet",
    partition_name: str = "default",
    batch_size: int = 32,
    normalization: str = "mean_stdev",
    return_val: bool = False,
    only_return_datasets: bool = False,
    geobench_root: str | None = None,
    num_workers: int = 8,
    image_size: int | None = None,
    interpolation: str = "bicubic",
):
    """Load GeoBench dataset splits and dataloaders.

    Args:
        dataset_name: Dataset identifier (e.g., 'm-eurosat', 'm-forestnet')
        partition_name: Partition to use (e.g., 'default', '0.01x_train')
        batch_size: Batch size for dataloaders
        normalization: Normalization method ('mean_stdev', 'min_max', 'percentile_2_98', or 'none')
        return_val: If True, return 4-tuple including validation split
        only_return_datasets: If True, return only datasets without dataloaders
        geobench_root: Path to classification_v1.0 directory (uses DEFAULT if None)
        num_workers: Number of dataloader workers

    Returns:
        If return_val=True: (train_dataset, train_loader, val_loader, test_loader)
        If return_val=False: (train_dataset, train_loader, test_loader)
    """
    if geobench_root is None:
        import os

        geobench_root = os.getenv("GEOBENCH_ROOT", DEFAULT_GEOBENCH_ROOT)

    # Map normalization argument
    if normalization == "mean_stdev":
        normalize = True
    elif normalization == "min_max":
        normalize = "min_max"
    elif normalization == "percentile_2_98":
        normalize = "percentile_2_98"
    else:
        normalize = False

    # Optional resize transform
    resize_transform = None
    if image_size is not None:
        interp_mode = {
            "bicubic": "bicubic",
            "bilinear": "bilinear",
            "nearest": "nearest",
        }.get(interpolation, "bicubic")

        def _resize(sample: dict) -> dict:
            img: torch.Tensor = sample["image"]
            h, w = img.shape[-2], img.shape[-1]
            if h != image_size or w != image_size:
                # Use float32 already; add batch dim for F.interpolate
                img = img.unsqueeze(0)
                img = F.interpolate(
                    img,
                    size=(image_size, image_size),
                    mode=interp_mode,
                    align_corners=False if interp_mode in ("bicubic", "bilinear") else None,
                )
                sample["image"] = img.squeeze(0)
            return sample

        resize_transform = _resize

    # Compose transform(s) if needed (currently single resize)
    transform_callable = resize_transform
    train_dataset = GeoBenchDataset(
        root=geobench_root,
        dataset_name=dataset_name,
        split="train",
        partition=partition_name,
        bands=("red", "green", "blue"),
        normalize=normalize,
        transform=transform_callable,
    )

    valid_dataset = GeoBenchDataset(
        root=geobench_root,
        dataset_name=dataset_name,
        split="valid",
        partition="default",  # validation always uses default partition
        bands=("red", "green", "blue"),
        normalize=normalize,
        transform=transform_callable,
    )

    test_dataset = GeoBenchDataset(
        root=geobench_root,
        dataset_name=dataset_name,
        split="test",
        partition="default",  # test always uses default partition
        bands=("red", "green", "blue"),
        normalize=normalize,
        transform=transform_callable,
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    if only_return_datasets:
        if return_val:
            return train_dataset, valid_dataset, test_dataset
        else:
            return train_dataset, test_dataset

    if return_val:
        return train_dataset, train_dataloader, val_dataloader, test_dataloader
    else:
        return train_dataset, train_dataloader, test_dataloader
