"""High-level dataset loading helpers and registry for torchgeo-bench.

This module owns the public ``get_datasets`` API used by
``torchgeo_bench.main`` and the registry that maps dataset names to their
:class:`~.base.BenchDataset` subclass.  All band resolution, resize
transforms and DataLoader construction live here so the per-dataset wrappers
stay focused on declaring metadata.
"""

import logging
import warnings
from collections.abc import Callable, Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .advance import ADVANCE
from .base import BenchDataset
from .benv2 import BENV2
from .burn_scars import BurnScars
from .caffe import CaFFe
from .cloudsen12 import CloudSEN12
from .dynamic_earthnet import DynamicEarthNet
from .eurosat import EuroSAT, EuroSATSpatial
from .flair2 import FLAIR2
from .forestnet import Forestnet
from .fotw import FieldsOfTheWorld
from .kuro_siwo import KuroSiwo
from .m_bigearthnet import MBigEarthNet
from .m_brick_kiln import MBrickKiln
from .m_eurosat import MEurosat
from .m_forestnet import MForestnet
from .m_pv4ger import MPv4ger
from .m_so2sat import MSo2Sat
from .pastis import PASTIS
from .resisc45 import RESISC45
from .sen12ms_cr import SEN12MS, SEN12MSCRC1, SEN12MSCRC2, SEN12MSCRC3, SEN12MSCRC4, SEN12MSCRC5
from .so2sat import So2Sat
from .spacenet2 import SpaceNet2
from .spacenet7 import SpaceNet7
from .treesatai import TreeSatAI

logger = logging.getLogger(__name__)


_REGISTRY: dict[str, type[BenchDataset]] = {
    cls.name: cls
    for cls in [
        # V1 classification
        MEurosat,
        MForestnet,
        MSo2Sat,
        MPv4ger,
        MBrickKiln,
        MBigEarthNet,
        # V2 classification
        BENV2,
        TreeSatAI,
        So2Sat,
        Forestnet,
        # V2 segmentation
        CaFFe,
        BurnScars,
        CloudSEN12,
        DynamicEarthNet,
        FLAIR2,
        FieldsOfTheWorld,
        KuroSiwo,
        PASTIS,
        SpaceNet2,
        SpaceNet7,
        # torchgeo template
        ADVANCE,
        EuroSAT,
        EuroSATSpatial,
        RESISC45,
        SEN12MS,
        SEN12MSCRC1,
        SEN12MSCRC2,
        SEN12MSCRC3,
        SEN12MSCRC4,
        SEN12MSCRC5,
    ]
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
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"Unknown dataset '{name}'. Available: {available}")
    return _REGISTRY[name]


def list_datasets() -> list[str]:
    """Return sorted names of all registered benchmark datasets."""
    return sorted(_REGISTRY)


def _make_resize_transform(
    image_size: int | None,
    interpolation: str,
) -> Callable[[dict], dict] | None:
    """Build a sample-level transform that resizes ``image`` (and ``mask``)."""
    if image_size is None:
        return None

    interp_mode = (
        interpolation if interpolation in ("bicubic", "bilinear", "nearest") else "bicubic"
    )
    align_corners = False if interp_mode in ("bicubic", "bilinear") else None

    def _resize(sample: dict) -> dict:
        img: torch.Tensor = sample["image"]
        h, w = img.shape[-2], img.shape[-1]
        if h != image_size or w != image_size:
            img = F.interpolate(
                img.unsqueeze(0),
                size=(image_size, image_size),
                mode=interp_mode,
                align_corners=align_corners,
            ).squeeze(0)
            sample["image"] = img
        if "mask" in sample:
            mask: torch.Tensor = sample["mask"].float()
            h_m, w_m = mask.shape[-2], mask.shape[-1]
            if h_m != image_size or w_m != image_size:
                mask = (
                    F.interpolate(
                        mask.unsqueeze(0).unsqueeze(0),
                        size=(image_size, image_size),
                        mode="nearest",
                    )
                    .squeeze(0)
                    .squeeze(0)
                    .long()
                )
                sample["mask"] = mask
        return sample

    return _resize


def _make_loader(ds: Dataset, *, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def get_datasets(
    dataset_name: str = "m-forestnet",
    partition_name: str = "default",
    batch_size: int = 32,
    return_val: bool = False,
    num_workers: int = 8,
    image_size: int | None = None,
    interpolation: str = "bicubic",
    bands: str | Iterable[str] | None = "rgb",
) -> tuple:
    """Load benchmark dataset splits and dataloaders.

    Datasets always emit raw float32 values; per-channel normalization is
    the model's responsibility (see :class:`~torchgeo_bench.models.interface.BenchModel`).

    Args:
        dataset_name: Identifier registered in :data:`_REGISTRY`.
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

    transform = _make_resize_transform(image_size, interpolation)
    train_partition = partition_name if bench.supports_partitions else "default"

    common = {"bands": bands_tuple, "transform": transform}
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
