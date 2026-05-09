"""GeoBench V2 dataset adapter and per-wrapper base class.

Each :class:`~torchgeo_bench.datasets.base.BenchDataset` subclass that wraps
a GeoBench V2 dataset inherits from :class:`_V2Dataset`, which dispatches to
the matching ``geobench_v2.datasets.GeoBench<X>`` upstream class.
"""

import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import geobench_v2.datasets as _gb_v2
import torch.nn as nn
from torch.utils.data import Dataset

from .base import BenchDataset

logger = logging.getLogger(__name__)

V2_ROOT = Path(os.environ.get("GEOBENCH_V2_ROOT", "data/geobenchv2"))


# Map dataset name → upstream class name on ``geobench_v2.datasets``.
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

# A few upstream classes only accept "val" instead of "validation".
_V2_VAL_AS_VAL: frozenset[str] = frozenset({"kuro_siwo"})


def list_v2_datasets() -> list[str]:
    """Return the sorted set of dataset names handled by the V2 adapter."""
    return sorted(_V2_REGISTRY)


def _resolve_class(dataset_name: str) -> type[Dataset]:
    return getattr(_gb_v2, _V2_REGISTRY[dataset_name])


class GeoBenchv2(Dataset):
    """Thin :class:`Dataset` adapter around any GeoBench V2 upstream class.

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
        cls = _resolve_class(dataset_name)
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

    Concrete subclasses just declare metadata; ``get_dataset`` is fully
    implemented here and dispatches to :class:`GeoBenchv2`. Multi-modality
    wrappers (those whose bands span multiple sensors and whose upstream class
    expects ``band_order`` as a ``dict``) opt in by setting
    ``band_order_strategy = "by_sensor"``. Wrappers that need to remap the
    upstream sample dict (e.g. ``KuroSiwo`` collapsing a temporal axis) override
    :meth:`canonicalize_sample`.
    """

    band_order_strategy: Literal["flat", "by_sensor"] = "flat"

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
        """Map an upstream sample dict onto the framework's canonical schema.

        Default implementation is a no-op. Subclasses override when the upstream
        loader yields keys other than ``image`` and ``label``/``mask`` (e.g.
        ``image_a``/``image_b`` for change-detection, or temporal stacks).
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
        """Return a :class:`GeoBenchv2` for the given split (raw values).

        Forces ``data_normalizer=nn.Identity`` so the upstream class emits
        raw sensor values; per-channel normalization belongs on
        :class:`~torchgeo_bench.models.interface.BenchModel`.
        """
        del partition
        band_order = self.build_band_order(bands)
        canonicalize = self.canonicalize_sample

        def chained(sample: dict) -> dict:
            sample = canonicalize(sample)  # rename image_b → "image" first
            if transform is not None:
                if "image" in sample:
                    sample = transform(sample)  # _resize now finds "image" safely
                else:
                    # treesatai applies its transforms on the per-modality dict
                    # (image_aerial / image_s2 / image_s1) *before* stacking, so
                    # the framework's resize transform must operate on each
                    # modality independently here.
                    for key in [k for k in sample if k.startswith("image_")]:
                        wrapped = transform({"image": sample[key]})
                        sample[key] = wrapped["image"]
            return sample

        kwargs: dict[str, object] = {
            "data_normalizer": nn.Identity,
            # No-op if the tortilla file is already present; otherwise pulls
            # it from the upstream HF mirror (aialliance/<name>) on first use.
            "download": True,
        }
        if self.band_order_strategy == "by_sensor":
            kwargs["return_stacked_image"] = True

        return GeoBenchv2(
            root=self.data_root(),
            dataset_name=self.name,
            split=split,
            band_order=band_order,
            transforms=chained,
            **kwargs,
        )
