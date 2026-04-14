"""Dataset utilities for torchgeo-bench.

This module provides a unified interface to load GeoBench datasets using
the V1 GeoBenchDataset class and the V2 geobench_v2 package.
"""

import os
import warnings
from collections.abc import Callable

import geobench_v2.datasets as gb_v2
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..geobench_dataset import GeoBenchDataset

NUM_CLASSES_PER_DATASET = {
    "m-forestnet": 12,
    "m-eurosat": 10,
    "m-pv4ger": 2,
    "m-brick-kiln": 2,
    "m-so2sat": 17,
    "m-bigearthnet": 43,
    "benv2": 19,
    "treesatai": 13,
    "so2sat": 17,
    "forestnet": 12,
    "caffe": 4,
    "cloudsen12": 4,
    "burn_scars": 3,  # 0=background, 1=burn, 2=cloud
    "dynamic_earthnet": 7,
    "flair2": 13,
    "fotw": 4,  # 0=background, 1=field, 2=boundary, 3=other
    "kuro_siwo": 4,
    "pastis": 20,  # 0-18 crops + background
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
DEFAULT_GEOBENCH_V2_ROOT = "data/geobenchv2"

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
    bands: tuple[str, ...] | None,
    transform: Callable | None,
    normalization: str,
):
    """Handles loading logic for V2 datasets."""
    del normalization
    if partition_name != "default":
        warnings.warn(
            f"Partitions are not supported in GeoBench V2. Ignoring partition '{partition_name}'.",
            UserWarning,
            stacklevel=2,
        )

    class_name = _get_v2_class_name(dataset_name)
    dataset_cls = getattr(gb_v2, class_name, None)
    if dataset_cls is None:
        raise ValueError(f"Could not find V2 dataset class '{class_name}' in geobench_v2.datasets.")

    # currently only support mean-stdev normalization for V2, which happens by default
    def load_split(split: str) -> gb_v2.GeoBenchBaseDataset:
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
    bands: tuple[str, ...] | None,
    transform: Callable | None,
    normalize_arg: bool | str,
):
    """Handle loading logic for V1 datasets."""
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


def is_dataset_available(
    dataset_name: str,
    geobench_root: str | None = None,
    geobench_v2_root: str | None = None,
) -> bool:
    """Check whether the data directory for a dataset exists on disk.

    Args:
        dataset_name: Name of the dataset (e.g., ``"m-eurosat"``).
        geobench_root: Root directory for V1 data.
        geobench_v2_root: Root directory for V2 data.

    Returns:
        True if the dataset directory exists.
    """
    if geobench_root is None:
        geobench_root = os.getenv("GEOBENCH_ROOT", DEFAULT_GEOBENCH_ROOT)
    if geobench_v2_root is None:
        geobench_v2_root = os.getenv("GEOBENCH_V2_ROOT", DEFAULT_GEOBENCH_V2_ROOT)

    if dataset_name in V2_DATASETS:
        return os.path.isdir(os.path.join(geobench_v2_root, dataset_name))
    return os.path.isdir(os.path.join(geobench_root, dataset_name))


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
    bands: str | tuple[str, ...] | None = "rgb",
):
    """Load GeoBench dataset splits and dataloaders (supports V1 and V2).

    Args:
        dataset_name: Name of the dataset (e.g., ``"m-eurosat"``).
        partition_name: Partition name (e.g., ``"default"``, ``"0.01x_train"``).
        batch_size: Batch size for the dataloaders.
        normalization: Normalization strategy (``"mean_stdev"``, ``"min_max"``,
            ``"percentile_2_98"``).
        return_val: If True, also return a validation dataloader.
        only_return_datasets: If True, return raw Dataset objects instead of
            DataLoaders.
        geobench_root: Root directory for V1 data. Defaults to ``GEOBENCH_ROOT``
            env var.
        geobench_v2_root: Root directory for V2 data. Defaults to
            ``GEOBENCH_V2_ROOT`` env var.
        num_workers: Number of dataloader worker processes.
        image_size: If set, resize images to this square size.
        interpolation: Interpolation mode for resizing (``"bicubic"``,
            ``"bilinear"``, ``"nearest"``).
        bands: Band selection. Options:
            - ``"rgb"`` (default): Load red, green, blue bands only.
            - ``"all"`` or None: Load all available bands (multispectral).
            - tuple of band names: e.g., ``("red", "green", "blue", "nir")``.
    """
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

    # Resolve bands parameter
    # Convert OmegaConf ListConfig or other iterables to tuple
    if bands == "rgb":
        bands_tuple: tuple[str, ...] | None = ("red", "green", "blue")
    elif bands == "all" or bands is None:
        bands_tuple = None  # None means load all available bands
    elif isinstance(bands, str):
        raise ValueError(
            f"Invalid bands parameter: {bands}. Use 'rgb', 'all', None, or list of band names."
        )
    else:
        # Handle list, tuple, or OmegaConf ListConfig
        try:
            bands_tuple = tuple(bands)  # type: ignore[arg-type]
        except TypeError:
            raise ValueError(
                f"Invalid bands parameter: {bands}. Use 'rgb', 'all', None, or list of band names."
            ) from None

    if dataset_name in V2_DATASETS:
        return _get_datasets_v2(
            dataset_name=dataset_name,
            partition_name=partition_name,
            batch_size=batch_size,
            return_val=return_val,
            only_return_datasets=only_return_datasets,
            root=geobench_v2_root,
            num_workers=num_workers,
            bands=bands_tuple,
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
            bands=bands_tuple,
            transform=resize_transform,
            normalize_arg=normalize_v1,  # V1 uses mapped bool/string
        )
