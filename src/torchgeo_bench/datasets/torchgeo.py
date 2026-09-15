"""Runtime-only factories for TorchGeo datasets."""

from collections.abc import Callable

import torch
import torchgeo.datasets
from torch.utils.data import Dataset
from torchvision.transforms import Compose

from .input import ResolvedInput, Split
from .spec import DatasetSpec, TorchGeoSource


def load_torchgeo_split(
    spec: DatasetSpec,
    split: Split,
    *,
    inputs: ResolvedInput,
    transform: Callable | None = None,
) -> Dataset:
    """Load a requested TorchGeo split with downloading explicitly disabled."""
    source = spec.source
    assert isinstance(source, TorchGeoSource)
    cls = getattr(torchgeo.datasets, source.upstream_class)
    if source.upstream_class == "RESISC45":
        indices = [spec.bands.index(band) for band in inputs.bands]
        select = _make_band_select(indices, len(spec.bands))
        if select is not None:
            transform = select if transform is None else Compose([select, transform])
        return cls(root=source.root, split=split, transforms=transform, download=False)
    return cls(
        root=source.root,
        split=split,
        bands=tuple(band.source_name for band in inputs.bands),
        transforms=transform,
        download=False,
    )


def _make_band_select(indices: list[int], n_bands: int) -> Callable[[dict], dict] | None:
    """Select channels before resizing; skip the transform for identity selection."""
    if indices == list(range(n_bands)):
        return None
    index = torch.tensor(indices)

    def _select(sample: dict) -> dict:
        sample["image"] = sample["image"].index_select(-3, index)
        return sample

    return _select
