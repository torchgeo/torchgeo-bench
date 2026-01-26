"""Dataset utilities for torchgeo-bench.

This module provides a unified interface to load GeoBench datasets using
the V1 GeoBenchDataset class and the V2 geobench_v2 package.
"""

import os
import warnings
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .geobench_dataset import GeoBenchDataset

try:
    import geobench_v2.datasets as gb_v2

    HAS_V2 = True
except ImportError:
    HAS_V2 = False


NUM_CLASSES_PER_DATASET = {
    "m-forestnet": 12,
    "m-eurosat": 10,
    "m-pv4ger": 2,
    "m-brick-kiln": 2,
    "m-so2sat": 17,
    # "m-bigearthnet": None,  # TODO: Handle BigEarthNet separately
    "benv2": 19,
    "treesatai": 13,
    "so2sat": 17,
    "forestnet": 12,
    "caffe": 4,
    "cloudsen12": 4,
    "burn_scars": 2,
    "dynamic_earthnet": 7,
    "flair2": 13,
    "fotw": 2,
    "kuro_siwo": 4,
    "pastis": 19,
    "spacenet2": 3,
    "spacenet7": 3,
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

DEFAULT_GEOBENCH_ROOT = "data/classification_v1.0"
DEFAULT_GEOBENCH_V2_ROOT = "/mnt/SSD2/nils/datasets/GEO-Bench-2/"

# V2 Dataset Registry
V2_DATASETS = {
    "benv2",
    "biomassters",
    "burn_scars",
    "caffe",
    "cloudsen12",
    "dynamic_earthnet",
    "everwatch",
    "flair2",
    "forestnet",
    "fotw",
    "kuro_siwo",
    "pastis",
    "spacenet2",
    "spacenet7",
    "substation",
    "treesatai",
    "wind_turbine",
    "so2sat",
}

# V2 Task Types
V2_TASK_TYPES = {
    "benv2": "classification",
    "forestnet": "classification",
    "so2sat": "classification",
    "eurosat": "classification",
    "treesatai": "classification",
    # Default others to segmentation
    "biomassters": "segmentation",
    "burn_scars": "segmentation",
    "caffe": "segmentation",
    "cloudsen12": "segmentation",
    "dynamic_earthnet": "segmentation",
    "flair2": "segmentation",
    "fotw": "segmentation",
    "kuro_siwo": "segmentation",
    "pastis": "segmentation",
    "spacenet2": "segmentation",
    "spacenet7": "segmentation",
}


def _get_v2_class_name(dataset_name: str) -> str:
    """Helper to convert dataset snake_case name to CamelCase class name."""
    if dataset_name == "benv2":
        return "GeoBenchBENV2"
    if dataset_name == "so2sat":
        return "GeoBenchSo2Sat"
    if dataset_name == "flair2":
        return "GeoBenchFLAIR2"
    if dataset_name == "spacenet2":
        return "GeoBenchSpaceNet2"
    if dataset_name == "spacenet7":
        return "GeoBenchSpaceNet7"

    camel_name = "".join(x.title() for x in dataset_name.split("_"))
    return f"GeoBench{camel_name}"


def _get_datasets_v2(
    dataset_name: str,
    partition_name: str,
    batch_size: int,
    return_val: bool,
    only_return_datasets: bool,
    root: str,
    num_workers: int,
    bands: tuple,
    transform: Optional[Callable],
    normalization: str,
):
    """Handles loading logic for V2 datasets."""
    if not HAS_V2:
        raise ImportError(
            f"Cannot load V2 dataset '{dataset_name}': geobench_v2 package not found."
        )

    if partition_name != "default":
        warnings.warn(
            f"Partitions are not supported in GeoBench V2. Ignoring partition '{partition_name}'.",
            UserWarning,
        )

    class_name = _get_v2_class_name(dataset_name)
    dataset_cls = getattr(gb_v2, class_name, None)
    if dataset_cls is None:
        raise ValueError(f"Could not find V2 dataset class '{class_name}' in geobench_v2.datasets.")

    # currently only support mean-stdev normalization for V2, which happens by default
    def load_split(split):
        ds = dataset_cls(
            root=os.path.join(root, dataset_name),
            split=split,
            transforms=transform,
            band_order=bands,
        )
        ds.task_type = V2_TASK_TYPES.get(dataset_name, "segmentation")
        return ds

    train_dataset = load_split("train")
    valid_dataset = load_split("val")
    test_dataset = load_split("test")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    if only_return_datasets:
        return (
            (train_dataset, valid_dataset, test_dataset)
            if return_val
            else (train_dataset, test_dataset)
        )
    if return_val:
        return train_dataset, train_loader, val_loader, test_loader
    return train_dataset, train_loader, test_loader


def _get_datasets_v1(
    dataset_name: str,
    partition_name: str,
    batch_size: int,
    return_val: bool,
    only_return_datasets: bool,
    root: str,
    num_workers: int,
    bands: tuple,
    transform: Optional[Callable],
    normalize_arg: Union[bool, str],
):
    """Handles loading logic for V1 datasets."""

    train_dataset = GeoBenchDataset(
        root=root,
        dataset_name=dataset_name,
        split="train",
        partition=partition_name,
        bands=bands,
        normalize=normalize_arg,
        transform=transform,
    )

    valid_dataset = GeoBenchDataset(
        root=root,
        dataset_name=dataset_name,
        split="valid",
        partition="default",
        bands=bands,
        normalize=normalize_arg,
        transform=transform,
    )

    test_dataset = GeoBenchDataset(
        root=root,
        dataset_name=dataset_name,
        split="test",
        partition="default",
        bands=bands,
        normalize=normalize_arg,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    if only_return_datasets:
        return (
            (train_dataset, valid_dataset, test_dataset)
            if return_val
            else (train_dataset, test_dataset)
        )
    if return_val:
        return train_dataset, train_loader, val_loader, test_loader
    return train_dataset, train_loader, test_loader


def get_datasets(
    dataset_name: str = "m-forestnet",
    partition_name: str = "default",
    batch_size: int = 32,
    normalization: str = "mean_stdev",
    return_val: bool = False,
    only_return_datasets: bool = False,
    geobench_root: str | None = None,
    geobench_v2_root: str | None = None,
    num_workers: int = 8,
    image_size: int | None = None,
    interpolation: str = "bicubic",
):
    """Load GeoBench dataset splits and dataloaders (supports V1 and V2)."""

    if geobench_root is None:
        geobench_root = os.getenv("GEOBENCH_ROOT", DEFAULT_GEOBENCH_ROOT)

    if geobench_v2_root is None:
        geobench_v2_root = os.getenv("GEOBENCH_V2_ROOT", DEFAULT_GEOBENCH_V2_ROOT)
    if normalization == "mean_stdev":
        normalize_v1 = True
    elif normalization in ["min_max", "percentile_2_98"]:
        normalize_v1 = normalization
    else:
        normalize_v1 = False

    resize_transform = None
    if image_size is not None:
        interp_mode = {
            "bicubic": "bicubic",
            "bilinear": "bilinear",
            "nearest": "nearest",
        }.get(interpolation, "bicubic")

        def _resize(sample: dict) -> dict:
            """Resize image in sample to desired size."""
            img: torch.Tensor = sample["image"]
            h, w = img.shape[-2], img.shape[-1]
            if h != image_size or w != image_size:
                img = img.unsqueeze(0)
                img = F.interpolate(
                    img,
                    size=(image_size, image_size),
                    mode=interp_mode,
                    align_corners=False if interp_mode in ("bicubic", "bilinear") else None,
                )
                sample["image"] = img.squeeze(0)
            if "mask" in sample:
                # assuming mask is single-channel with class indices
                mask: torch.Tensor = sample["mask"].float()
                h_m, w_m = mask.shape[-2], mask.shape[-1]
                if h_m != image_size or w_m != image_size:
                    mask = mask.unsqueeze(0).unsqueeze(0)
                    mask = F.interpolate(
                        mask,
                        size=(image_size, image_size),
                        mode="nearest",
                    )
                    sample["mask"] = mask.squeeze(0).squeeze(0).long()
            return sample

        resize_transform = _resize

    bands = ("red", "green", "blue")

    if dataset_name in V2_DATASETS:
        return _get_datasets_v2(
            dataset_name=dataset_name,
            partition_name=partition_name,
            batch_size=batch_size,
            return_val=return_val,
            only_return_datasets=only_return_datasets,
            root=geobench_v2_root,
            num_workers=num_workers,
            bands=bands,
            transform=resize_transform,
            normalization=normalization,
        )
    else:
        return _get_datasets_v1(
            dataset_name=dataset_name,
            partition_name=partition_name,
            batch_size=batch_size,
            return_val=return_val,
            only_return_datasets=only_return_datasets,
            root=geobench_root,
            num_workers=num_workers,
            bands=bands,
            transform=resize_transform,
            normalize_arg=normalize_v1,  # V1 uses mapped bool/string
        )
