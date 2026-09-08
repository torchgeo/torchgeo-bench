"""GeoBench V2 adapters with raw sensor values and shared wrapper defaults."""

import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import ClassVar, Literal

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from .base import BenchDataset

logger = logging.getLogger(__name__)

V2_ROOT = Path("data/geobenchv2")


_V2_REGISTRY: dict[str, str] = {
    "benv2": "GeoBenchBENV2",
    "burn_scars": "GeoBenchBurnScars",
    "caffe": "GeoBenchCaFFe",
    "cloudsen12": "GeoBenchCloudSen12",
    "dynamic_earthnet": "GeoBenchDynamicEarthNet",
    "flair2": "GeoBenchFLAIR2",
    "forestnet": "GeoBenchForestnet",
    "fotw": "GeoBenchFieldsOfTheWorld",
    "kuro_siwo": "GeoBenchKuroSiwo",
    "pastis": "GeoBenchPASTIS",
    "so2sat": "GeoBenchSo2Sat",
    "spacenet2": "GeoBenchSpaceNet2",
    "spacenet7": "GeoBenchSpaceNet7",
    "treesatai": "GeoBenchTreeSatAI",
}

# KuroSiwo expects "val"; the other upstream loaders expect "validation".
_V2_VAL_AS_VAL: frozenset[str] = frozenset({"kuro_siwo"})


def list_v2_datasets() -> list[str]:
    """Return the sorted set of dataset names handled by the V2 adapter."""
    return sorted(_V2_REGISTRY)


class _ChainedTransform:
    """Canonicalize an upstream V2 sample, then apply the framework transform."""

    def __init__(self, canonicalize: Callable[[dict], dict], transform: Callable | None) -> None:
        self.canonicalize = canonicalize
        self.transform = transform

    def __call__(self, sample: dict) -> dict:
        sample = self.canonicalize(sample)
        if self.transform is None:
            return sample
        if "image" in sample:
            return self.transform(sample)
        image_keys = [k for k in sample if k.startswith("image_")]
        for index, key in enumerate(image_keys):
            wrapped = {"image": sample[key]}
            if index == 0 and "mask" in sample:
                wrapped["mask"] = sample["mask"]
            wrapped = self.transform(wrapped)
            sample[key] = wrapped["image"]
            if "mask" in wrapped:
                sample["mask"] = wrapped["mask"]
        return sample


class GeoBenchv2(Dataset):
    """Load a GeoBench V2 dataset through its upstream class.

    Args:
        root: Path to the GeoBench V2 collection root (the directory
            containing per-dataset subdirectories, e.g. ``data/geobenchv2``).
        dataset_name: One of :func:`list_v2_datasets`.
        split: ``"train"``, ``"val"``, or ``"test"``.
        band_order: Bands to load in upstream-expected shape (a flat ``list``
            for single-modality datasets, or ``dict[modality, list[str]]`` for
            multi-modality ones).
        transforms: Optional sample transform forwarded to the upstream class.
        **kwargs: Additional keyword arguments forwarded to the upstream class.
    """

    def __init__(
        self,
        root: str | Path,
        dataset_name: str,
        split: str,
        *,
        band_order: object | None = None,
        transforms: Callable | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if dataset_name not in _V2_REGISTRY:
            raise KeyError(
                f"Unknown GeoBench V2 dataset '{dataset_name}'. "
                f"Available: {', '.join(list_v2_datasets())}"
            )
        # Import GeoBench V2 only when needed to keep CLI startup fast.
        import geobench_v2.datasets as _gb_v2

        cls: type[Dataset] = getattr(_gb_v2, _V2_REGISTRY[dataset_name])
        upstream_split = (
            "val"
            if split == "val" and dataset_name in _V2_VAL_AS_VAL
            else "validation"
            if split == "val"
            else split
        )

        forward: dict[str, object] = {
            "root": Path(root) / dataset_name,
            "split": upstream_split,
            "transforms": transforms,
        }
        if band_order is not None:
            forward["band_order"] = band_order
        forward.update(kwargs)

        self._inner: Dataset = cls(**forward)
        self.dataset_name = dataset_name
        self.split = split

    def __len__(self) -> int:
        return len(self._inner)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> dict:
        return self._inner[idx]


class _V2Dataset(BenchDataset):
    """Base class for every GeoBench V2 wrapper.

    Wrappers declare metadata and set ``band_order_strategy = "by_sensor"`` when the upstream loader expects bands grouped by sensor.

    Override :meth:`canonicalize_sample` to adapt sample keys or temporal dimensions. Use :attr:`upstream_kwargs` to customize upstream loader arguments.
    """

    band_order_strategy: Literal["flat", "by_sensor"] = "flat"

    #: Extra upstream loader arguments; these override the defaults in ``get_dataset`` (including ``return_stacked_image``).
    upstream_kwargs: ClassVar[dict[str, object]] = {}

    @classmethod
    def data_root(cls) -> Path:
        return V2_ROOT

    def build_band_order(self, bands: tuple[str, ...] | None) -> object:
        """Translate canonical band names into the upstream loader's shape."""
        specs = self.select_band_specs(bands)
        if self.band_order_strategy == "by_sensor":
            grouped: dict[str, list[str]] = {}
            for spec in specs:
                grouped.setdefault(spec.sensor, []).append(spec.source_name)
            return grouped
        return [spec.source_name for spec in specs]

    def canonicalize_sample(self, sample: dict) -> dict:
        """Adapt an upstream sample to ``image`` and ``label``/``mask`` keys.

        The default leaves samples unchanged. Wrappers override this for change-detection pairs (``image_a``/``image_b``) or temporal stacks.
        """
        return sample

    def get_dataset(
        self,
        split: str,
        *,
        partition: str = "default",
        bands: tuple[str, ...] | None = None,
        transform: Callable | None = None,
    ) -> Dataset:
        """Return raw sensor values for a split.

        ``data_normalizer=nn.Identity`` disables upstream normalization; per-channel normalization belongs on :class:`~torchgeo_bench.models.interface.BenchModel`.
        """
        del partition
        band_order = self.build_band_order(bands)

        kwargs: dict[str, object] = {
            "data_normalizer": nn.Identity,
            # Download missing tortilla files from aialliance/<name> on Hugging Face.
            # Set GEOBENCH_V2_NO_DOWNLOAD=1 for offline runs.
            "download": os.environ.get("GEOBENCH_V2_NO_DOWNLOAD") != "1",
        }
        if self.band_order_strategy == "by_sensor":
            kwargs["return_stacked_image"] = True
        kwargs.update(self.upstream_kwargs)

        return GeoBenchv2(
            root=self.data_root(),
            dataset_name=self.name,
            split=split,
            band_order=band_order,
            transforms=_ChainedTransform(self.canonicalize_sample, transform),
            **kwargs,
        )


class _OffsetMaskV2Dataset(_V2Dataset):
    """Restore SpaceNet's native two-class building masks.

    Upstream adds 1 to ``{0: no-building, 1: building}``, reserving an unused background class 0. :meth:`canonicalize_sample` removes this offset so the probe learns only the two real classes.

    See https://github.com/The-AI-Alliance/GEO-Bench-2 (``geobench_v2/datasets/spacenet2.py`` and ``spacenet7.py``: ``sample["mask"] = ... + 1``).
    """

    def canonicalize_sample(self, sample: dict) -> dict:
        """Reverse GeoBench's ``+1`` offset: mask ``1 -> 0``, ``2 -> 1``.

        ``clamp(min=0)`` maps any reserved upstream class 0 to no-building rather than creating a negative class ID. This runs before resizing.
        """
        mask = sample.get("mask")
        if mask is not None:
            mask = torch.as_tensor(mask)
            sample["mask"] = (mask - 1).clamp_(min=0)
        return sample
