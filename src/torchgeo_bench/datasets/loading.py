"""High-level dataset loading helpers and registry for torchgeo-bench.

This module owns the public ``get_datasets`` API used by
``torchgeo_bench.main`` and the registry that maps dataset names to their
:class:`~.base.BenchDataset` subclass.  All band resolution, resize
transforms and DataLoader construction live here so the per-dataset wrappers
stay focused on declaring metadata.

Wrapper modules (and torch) import lazily: ``list_datasets`` reads only the
registry spec below, and ``get_bench_dataset_class`` imports just the one
module that defines the requested dataset.
"""

import logging
import warnings
from importlib import import_module
from typing import TYPE_CHECKING

from .base import BenchDataset

if TYPE_CHECKING:
    from collections.abc import Iterable

    import torch
    from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# Dataset name -> (submodule, class name).  Kept as strings so that importing
# this module stays cheap; ``get_bench_dataset_class`` resolves entries on
# demand and verifies the class's ``name`` attribute matches its key.
_REGISTRY_SPEC: dict[str, tuple[str, str]] = {
    # V1 classification
    "m-eurosat": ("m_eurosat", "MEurosat"),
    "m-forestnet": ("m_forestnet", "MForestnet"),
    "m-so2sat": ("m_so2sat", "MSo2Sat"),
    "m-pv4ger": ("m_pv4ger", "MPv4ger"),
    "m-brick-kiln": ("m_brick_kiln", "MBrickKiln"),
    "m-bigearthnet": ("m_bigearthnet", "MBigEarthNet"),
    # V2 classification
    "benv2": ("benv2", "BENV2"),
    "treesatai": ("treesatai", "TreeSatAI"),
    "so2sat": ("so2sat", "So2Sat"),
    "forestnet": ("forestnet", "Forestnet"),
    # V2 segmentation
    "caffe": ("caffe", "CaFFe"),
    "burn_scars": ("burn_scars", "BurnScars"),
    "cloudsen12": ("cloudsen12", "CloudSEN12"),
    "dynamic_earthnet": ("dynamic_earthnet", "DynamicEarthNet"),
    "flair2": ("flair2", "FLAIR2"),
    "fotw": ("fotw", "FieldsOfTheWorld"),
    "kuro_siwo": ("kuro_siwo", "KuroSiwo"),
    "pastis": ("pastis", "PASTIS"),
    "spacenet2": ("spacenet2", "SpaceNet2"),
    "spacenet7": ("spacenet7", "SpaceNet7"),
    # torchgeo template
    "eurosat": ("eurosat", "EuroSAT"),
    "eurosat-spatial": ("eurosat", "EuroSATSpatial"),
    "resisc45": ("resisc45", "RESISC45"),
}


def get_bench_dataset_class(name: str) -> type[BenchDataset]:
    """Look up a dataset by name and return its :class:`BenchDataset` class.

    Args:
        name: Dataset identifier (e.g. ``"m-eurosat"``, ``"burn_scars"``).

    Returns:
        The registered :class:`BenchDataset` subclass.

    Raises:
        KeyError: If *name* is not in the registry.
    """
    if name not in _REGISTRY_SPEC:
        available = ", ".join(sorted(_REGISTRY_SPEC))
        raise KeyError(f"Unknown dataset '{name}'. Available: {available}")
    module_name, class_name = _REGISTRY_SPEC[name]
    cls: type[BenchDataset] = getattr(import_module(f".{module_name}", __package__), class_name)
    if cls.name != name:
        raise RuntimeError(
            f"Registry mismatch: entry '{name}' resolved to {class_name} "
            f"whose declared name is '{cls.name}'."
        )
    return cls


def list_datasets() -> list[str]:
    """Return sorted names of all registered benchmark datasets."""
    return sorted(_REGISTRY_SPEC)


class _ResizeTransform:
    """Sample-level transform that resizes ``image`` (and ``mask``)."""

    valid_modes = ("area", "bicubic", "bilinear", "nearest")

    def __init__(self, image_size: int, interp_mode: str) -> None:
        if interp_mode not in self.valid_modes:
            raise ValueError(
                f"interpolation must be one of {self.valid_modes}, got {interp_mode!r}."
            )
        self.image_size = image_size
        self.interp_mode = interp_mode
        self.align_corners = False if interp_mode in ("bicubic", "bilinear") else None

    def __call__(self, sample: dict) -> dict:
        import torch.nn.functional as F

        image_size = self.image_size
        img: torch.Tensor = sample["image"]
        h, w = img.shape[-2], img.shape[-1]
        if h != image_size or w != image_size:
            squeeze_batch = img.ndim == 3
            resize_input = img.unsqueeze(0) if squeeze_batch else img
            img = F.interpolate(
                resize_input,
                size=(image_size, image_size),
                mode=self.interp_mode,
                align_corners=self.align_corners,
            )
            if squeeze_batch:
                img = img.squeeze(0)
            sample["image"] = img
        if "mask" in sample:
            mask: torch.Tensor = sample["mask"].float()
            h_m, w_m = mask.shape[-2], mask.shape[-1]
            if h_m != image_size or w_m != image_size:
                if mask.ndim == 2:
                    resize_mask = mask.unsqueeze(0).unsqueeze(0)
                    squeeze_dims = 2
                elif mask.ndim == 3:
                    resize_mask = mask.unsqueeze(0)
                    squeeze_dims = 1
                else:
                    resize_mask = mask
                    squeeze_dims = 0
                mask = F.interpolate(
                    resize_mask,
                    size=(image_size, image_size),
                    mode="nearest",
                )
                for _ in range(squeeze_dims):
                    mask = mask.squeeze(0)
                mask = mask.long()
                sample["mask"] = mask
        return sample


def _make_loader(
    ds: "Dataset", *, batch_size: int, shuffle: bool, num_workers: int
) -> "DataLoader":
    import torch
    from torch.utils.data import DataLoader

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def get_datasets(
    dataset_name: str = "m-forestnet",
    partition_name: str = "default",
    batch_size: int = 32,
    return_val: bool = False,
    num_workers: int = 8,
    image_size: int | None = None,
    interpolation: str = "bilinear",
    bands: "str | Iterable[str] | None" = "rgb",
    time_steps: int | None = None,
) -> tuple:
    """Load benchmark dataset splits and dataloaders.

    Datasets always emit raw float32 values; per-channel normalization is
    the model's responsibility (see :class:`~torchgeo_bench.models.interface.BenchModel`).

    Args:
        dataset_name: Identifier registered in ``_REGISTRY_SPEC``.
        partition_name: Partition name (only honoured by datasets where
            :attr:`~.base.BenchDataset.supports_partitions` is ``True``).
        batch_size: Batch size for the returned dataloaders.
        return_val: If ``True``, also return a validation dataloader.
        num_workers: Number of dataloader worker processes.
        image_size: If set, resize images (and masks, with nearest) to this
            square size at sample time.
        interpolation: Resize interpolation for images (``"bicubic"``,
            ``"bilinear"``, ``"nearest"``).
        bands: ``"rgb"`` (use the dataset's ``rgb_bands``), ``"all"`` /
            ``None`` (load all bands), or an explicit iterable of band names.
        time_steps: Number of acquisition dates per sample.  Only accepted by
            multi-temporal wrappers (PASTIS); ``None`` keeps each dataset's
            own default.

    Returns:
        Either ``(train_dataset, train_loader, test_loader)`` or, when
        ``return_val=True``, ``(train_dataset, train_loader, val_loader,
        test_loader)``.

    Raises:
        KeyError: If ``dataset_name`` is not registered.
    """
    cls = get_bench_dataset_class(dataset_name)
    bench = cls()

    if partition_name != "default" and not bench.supports_partitions:
        warnings.warn(
            f"Dataset '{dataset_name}' does not support custom partitions. "
            f"Ignoring partition '{partition_name}'.",
            UserWarning,
            stacklevel=2,
        )

    if bands == "rgb":
        bands_tuple: tuple[str, ...] | None = tuple(bench.rgb_bands)
    elif bands == "all" or bands is None:
        bands_tuple = None
    elif isinstance(bands, str):
        raise ValueError(
            f"Invalid bands parameter: {bands!r}. Use 'rgb', 'all', None, "
            "or an iterable of band names."
        )
    else:
        bands_tuple = tuple(bands)

    transform = _ResizeTransform(image_size, interpolation) if image_size is not None else None
    train_partition = partition_name if bench.supports_partitions else "default"

    common: dict = {"bands": bands_tuple, "transform": transform}
    if time_steps is not None:
        # Only multi-temporal wrappers accept this; others would not know what
        # to do with a time axis, so passing it to them is a config error.
        common["time_steps"] = time_steps
    train_ds = bench.get_dataset("train", partition=train_partition, **common)
    val_ds = bench.get_dataset("val", partition="default", **common)
    test_ds = bench.get_dataset("test", partition="default", **common)

    train_loader = _make_loader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = _make_loader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = _make_loader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    if return_val:
        return train_ds, train_loader, val_loader, test_loader
    return train_ds, train_loader, test_loader


__all__ = [
    "get_bench_dataset_class",
    "get_datasets",
    "list_datasets",
]
