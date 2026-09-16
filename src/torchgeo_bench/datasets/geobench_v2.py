"""Explicit GeoBench V2 construction and source-to-canonical sample conversion."""

from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Literal, NotRequired, TypedDict

import torch
from torch import nn
from torch.utils.data import Dataset

from torchgeo_bench.bands import BandSpec

from .input import ResolvedInput, Split
from .spec import DatasetSpec, V2Source
from .transforms import CanonicalTransform, Resize, canonical_image, canonical_target

type BandOrder = list[str] | dict[str, list[str]]


class _SourceOptions(TypedDict):
    root: Path
    split: str
    band_order: BandOrder
    data_normalizer: type[nn.Identity]
    transforms: None
    download: Literal[False]
    return_stacked_image: NotRequired[Literal[False]]
    time_step: NotRequired[list[Literal["post"]]]
    num_time_steps: NotRequired[int]
    temporal_output_format: NotRequired[Literal["TCHW"]]
    temporal_aggregation: NotRequired[None]
    temporal_setting: NotRequired[Literal["single"]]
    label_type: NotRequired[Literal["semantic_seg"]]
    return_window: NotRequired[list[Literal["win_a", "win_b"]]]


def _band_order(source: V2Source, bands: tuple[BandSpec, ...]) -> BandOrder:
    if source.band_order_strategy == "flat":
        return [band.source_name for band in bands]
    grouped: dict[str, list[str]] = {}
    for band in bands:
        grouped.setdefault(band.sensor, []).append(band.source_name)
    return grouped


def _channel_indices(order: BandOrder, bands: tuple[BandSpec, ...]) -> tuple[int, ...]:
    """Match actual emitted source bands, including duplicates and within-sensor order."""
    if isinstance(order, dict):
        emitted = [(sensor, name) for sensor, names in order.items() for name in names]
        requested = [(band.sensor, band.source_name) for band in bands]
    else:
        emitted = [("", name) for name in order]
        requested = [("", band.source_name) for band in bands]
    if Counter(emitted) != Counter(requested):
        raise ValueError("Backend band order does not match requested bands")
    positions: dict[tuple[str, str], deque[int]] = defaultdict(deque)
    for index, band in enumerate(emitted):
        positions[band].append(index)
    return tuple(positions[band].popleft() for band in requested)


class _V2Samples(Dataset):
    """Convert known upstream components before canonical validation and resizing."""

    def __init__(
        self,
        inner: Dataset,
        spec: DatasetSpec,
        inputs: ResolvedInput,
        order: BandOrder,
        resize: Resize | None,
    ) -> None:
        self._inner = inner
        self.spec = spec
        self.inputs = inputs
        self.band_specs = inputs.bands
        self.order = order
        self.indices = _channel_indices(order, inputs.bands)
        self.transform = CanonicalTransform(spec, inputs, resize)

    def __len__(self) -> int:
        return len(self._inner)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> dict:
        sample = dict(self._inner[index])
        source = self.spec.source
        assert isinstance(source, V2Source)
        canonical_target(sample, self.spec)
        if source.sample_adapter == "offset_mask":
            # Upstream adds one to native {0, 1}; reserved zero remains background.
            sample["mask"] = sample["mask"].clamp_min(1) - 1
        if isinstance(self.order, dict):
            keys = (
                {"sar": "image_post", "dem": "image_dem"}
                if source.sample_adapter == "post_sar_dem"
                else {sensor: f"image_{sensor}" for sensor in self.order}
            )
            images = [
                canonical_image(
                    self._component(sample, keys[sensor]),
                    self.inputs,
                    name=f"{self.spec.name} {keys[sensor]}",
                    channels=len(names),
                )
                for sensor, names in self.order.items()
            ]
            if len({image.shape[-2:] for image in images}) > 1:
                if not source.align_to_output:
                    raise ValueError(f"{self.spec.name}: source sensor grids must already match")
                resize = self.transform.resize
                if resize is None:
                    raise ValueError(
                        f"{self.spec.name}: different sensor grids require image_size for alignment"
                    )
                # Previously each sensor was resized directly onto the requested grid
                # before stacking. Avoid a new intermediate grid/double interpolation.
                images = [resize.image(image) for image in images]
            image = torch.cat(images, dim=-3)
        else:
            key = "image_b" if source.sample_adapter == "later_acquisition" else "image"
            image = canonical_image(self._component(sample, key), self.inputs, name=self.spec.name)
            if source.sample_adapter == "later_acquisition":
                sample.pop("image_a", None)
        if self.indices != tuple(range(len(self.indices))):
            image = image[list(self.indices)] if image.ndim == 3 else image[:, list(self.indices)]
        sample["image"] = image
        return self.transform(sample)

    def _component(self, sample: dict, key: str) -> object:
        if key not in sample:
            raise ValueError(f"{self.spec.name}: missing required source component {key!r}")
        return sample.pop(key)


def load_v2_split(
    spec: DatasetSpec,
    split: Split,
    *,
    inputs: ResolvedInput,
    resize: Resize | None = None,
) -> Dataset:
    """Construct only supported source options; leave acquisition sampling upstream."""
    source = spec.source
    if not isinstance(source, V2Source):
        raise TypeError("load_v2_split requires a V2Source")
    import geobench_v2.datasets as upstream

    cls = getattr(upstream, source.upstream_class)
    order = _band_order(source, inputs.bands)
    available = cls.dataset_band_config.modalities
    if isinstance(order, dict):
        valid = all(
            sensor in available and all(name in available[sensor].default_order for name in names)
            for sensor, names in order.items()
        )
    else:
        names = {name for config in available.values() for name in config.default_order}
        valid = all(name in names for name in order)
    if not valid:
        raise ValueError(
            f"{spec.name}: requested source bands are unsupported by {source.upstream_class}"
        )
    options: _SourceOptions = {
        "root": Path(source.root) / spec.storage_name,
        "split": source.validation_split if split == "val" else split,
        "band_order": order,
        "data_normalizer": nn.Identity,
        "transforms": None,
        "download": False,
    }
    if source.band_order_strategy == "by_sensor" or source.sample_adapter == "later_acquisition":
        options["return_stacked_image"] = False
    if source.sample_adapter == "post_sar_dem":
        options["time_step"] = ["post"]
    if source.sample_adapter == "later_acquisition":
        options["return_window"] = ["win_a", "win_b"]
        options["label_type"] = "semantic_seg"
    if source.upstream_class == "GeoBenchSpaceNet2":
        options["label_type"] = "semantic_seg"
    if source.upstream_class == "GeoBenchDynamicEarthNet":
        options["temporal_setting"] = "single"
    if spec.capabilities.multi_temporal:
        options["num_time_steps"] = inputs.num_time_steps
        options["temporal_output_format"] = "TCHW"
        options["temporal_aggregation"] = None
        options["label_type"] = "semantic_seg"
    inner = cls(**options)
    return _V2Samples(inner, spec, inputs, inner.band_order, resize)
