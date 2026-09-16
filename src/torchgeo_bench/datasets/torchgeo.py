"""Runtime-only factories for TorchGeo datasets."""

from dataclasses import dataclass

import torch
import torchgeo.datasets
from torch.utils.data import Dataset

from .input import ResolvedInput, Split
from .spec import DatasetSpec, TorchGeoSource
from .transforms import CanonicalTransform, Resize


@dataclass(frozen=True)
class _RESISC45Transform:
    """Select from the fixed RGB source order before canonical validation."""

    canonical: CanonicalTransform
    indices: tuple[int, ...]

    def __call__(self, sample: dict) -> dict:
        image = torch.as_tensor(sample["image"])
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("RESISC45: expected source CHW image with 3 channels")
        if self.indices != (0, 1, 2):
            sample = {**sample, "image": image[list(self.indices)]}
        return self.canonical(sample)


def load_torchgeo_split(
    spec: DatasetSpec,
    split: Split,
    *,
    inputs: ResolvedInput,
    resize: Resize | None = None,
) -> Dataset:
    """Load a requested TorchGeo split with downloading explicitly disabled."""
    source = spec.source
    if not isinstance(source, TorchGeoSource):
        raise TypeError("load_torchgeo_split requires a TorchGeoSource")
    cls = getattr(torchgeo.datasets, source.upstream_class)
    if source.upstream_class == "RESISC45":
        source_names = ("R", "G", "B")
        if any(band.source_name not in source_names for band in inputs.bands):
            raise ValueError("RESISC45: unsupported source bands")
        indices = tuple(source_names.index(band.source_name) for band in inputs.bands)
        transform = _RESISC45Transform(
            CanonicalTransform(spec, inputs, resize),
            indices,
        )
        return cls(root=source.root, split=split, transforms=transform, download=False)
    if any(band.source_name not in cls.all_band_names for band in inputs.bands):
        raise ValueError(f"{spec.name}: unsupported source bands")
    return cls(
        root=source.root,
        split=split,
        bands=tuple(band.source_name for band in inputs.bands),
        transforms=CanonicalTransform(spec, inputs, resize),
        download=False,
    )
