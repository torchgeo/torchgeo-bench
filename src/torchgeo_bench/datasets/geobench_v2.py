"""GeoBench V2 runtime readers with raw values and explicit source policies."""

import logging
from collections.abc import Callable
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset

from torchgeo_bench.bands import BandSpec

from .input import ResolvedInput, Split
from .spec import DatasetSpec, V2Source

logger = logging.getLogger(__name__)


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
        spec: Definition containing the upstream source identity and split policy.
        split: ``"train"``, ``"val"``, or ``"test"``.
        band_specs: Requested band metadata, in output channel order.
        band_order: Bands to load in upstream-expected shape (a flat ``list``
            for single-modality datasets, or ``dict[modality, list[str]]`` for
            multi-modality ones).
        sensor_order: Override the upstream stack order for custom canonicalizers.
        transforms: Optional sample transform forwarded to the upstream class.
        **kwargs: Additional keyword arguments forwarded to the upstream class.
    """

    def __init__(  # noqa: PLR0913 - adapter construction with resolved band metadata.
        self,
        spec: DatasetSpec,
        split: str,
        *,
        band_specs: tuple[BandSpec, ...],
        band_order: list[str] | dict[str, list[str]] | None = None,
        sensor_order: tuple[str, ...] | None = None,
        transforms: Callable | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        source = spec.source
        assert isinstance(source, V2Source)
        # Import GeoBench V2 only when needed to keep CLI startup fast.
        import geobench_v2.datasets as _gb_v2
        from geobench_v2.datasets.base import GeoBenchBaseDataset

        cls = getattr(_gb_v2, source.upstream_class)
        upstream_split = source.validation_split if split == "val" else split

        forward: dict[str, object] = {
            "root": Path(source.root) / spec.storage_name,
            "split": upstream_split,
            "transforms": transforms,
        }
        if band_order is not None:
            forward["band_order"] = band_order
        forward.update(kwargs)

        self._inner: GeoBenchBaseDataset = cls(**forward)
        self.dataset_name = spec.name
        self.split = split
        self.band_specs = band_specs
        # Upstream preserves within-sensor band order, but stacks using its resolved
        # band_order, not necessarily the dictionary supplied to its constructor.
        upstream_order = self._inner.band_order
        if sensor_order is not None:
            self._sensor_order = sensor_order
        elif isinstance(upstream_order, dict):
            self._sensor_order = tuple(upstream_order)
        else:
            self._sensor_order = ()
        emitted = list(range(len(band_specs)))
        if self._sensor_order:
            emitted = [
                index
                for sensor in self._sensor_order
                for index, spec in enumerate(band_specs)
                if spec.sensor == sensor
            ]
        if sorted(emitted) != list(range(len(band_specs))):
            raise ValueError(f"{spec.name}: backend sensor order does not match requested bands")
        self._channel_indices = [emitted.index(index) for index in range(len(band_specs))]

    def __len__(self) -> int:
        return len(self._inner)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> dict:
        sample = self._inner[index]
        if "image" not in sample and self._sensor_order:
            # Upstream PASTIS stacks along dim=0 even for TCHW. Keep its acquisition
            # selection and transforms, then concatenate along the channel axis here.
            sample["image"] = torch.cat(
                [sample.pop(f"image_{sensor}") for sensor in self._sensor_order], dim=-3
            )
        image = sample["image"]
        if image.ndim not in (3, 4) or image.shape[-3] != len(self.band_specs):
            raise ValueError(
                f"{self.dataset_name}: expected CHW or TCHW image with "
                f"{len(self.band_specs)} channels, got {tuple(image.shape)}"
            )
        if self._channel_indices != list(range(len(self.band_specs))):
            sample["image"] = image.index_select(
                -3, torch.tensor(self._channel_indices, device=image.device)
            )
        return sample


def build_band_order(
    source: V2Source, bands: tuple[BandSpec, ...]
) -> list[str] | dict[str, list[str]]:
    """Translate the single resolved band sequence into the upstream request."""
    if source.band_order_strategy == "by_sensor":
        grouped: dict[str, list[str]] = {}
        for band in bands:
            grouped.setdefault(band.sensor, []).append(band.source_name)
        return grouped
    return [band.source_name for band in bands]


def canonicalize_sample(sample: dict, *, source: V2Source) -> dict:
    """Apply the declared existing acquisition or label policy before resizing."""
    match source.sample_adapter:
        case "later_acquisition":
            if "image" not in sample and "image_b" in sample:
                sample["image"] = sample.pop("image_b")
                sample.pop("image_a", None)
        case "post_sar_dem":
            # Upstream cannot stack SAR and DEM with different channel counts.
            keys = {"sar": "image_post", "dem": "image_dem"}
            assert source.canonical_sensor_order is not None
            modalities = [
                sample.pop(keys[sensor])
                for sensor in source.canonical_sensor_order
                if keys[sensor] in sample
            ]
            if modalities:
                sample["image"] = (
                    modalities[0] if len(modalities) == 1 else torch.cat(modalities, dim=0)
                )
        case "offset_mask":
            # SpaceNet upstream adds 1 to native {0, 1}; reserved zero stays background.
            mask = sample.get("mask")
            if mask is not None:
                sample["mask"] = (torch.as_tensor(mask) - 1).clamp_(min=0)
    return sample


def load_v2_split(
    spec: DatasetSpec,
    split: Split,
    *,
    inputs: ResolvedInput,
    transform: Callable | None = None,
) -> Dataset:
    """Construct the common V2 reader from immutable, explicit source policies."""
    source = spec.source
    assert isinstance(source, V2Source)
    kwargs: dict[str, object] = {"data_normalizer": nn.Identity, "download": False}
    if source.band_order_strategy == "by_sensor":
        kwargs["return_stacked_image"] = True
    if source.return_stacked_image is not None:
        kwargs["return_stacked_image"] = source.return_stacked_image
    if source.time_step is not None:
        kwargs["time_step"] = list(source.time_step)
    if inputs.time_steps is not None:
        kwargs["num_time_steps"] = inputs.time_steps
        kwargs["temporal_output_format"] = "TCHW"
        if inputs.time_steps > 1:
            kwargs["return_stacked_image"] = False
    return GeoBenchv2(
        spec=spec,
        split=split,
        band_specs=inputs.bands,
        band_order=build_band_order(source, inputs.bands),
        sensor_order=source.canonical_sensor_order,
        transforms=_ChainedTransform(partial(canonicalize_sample, source=source), transform),
        **kwargs,
    )
