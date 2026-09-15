"""Immutable, dependency-light scientific definitions and source policies."""

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Literal

from torchgeo_bench.bands import BandSpec

type Split = Literal["train", "val", "test"]
type Task = Literal["classification", "segmentation"]


@dataclass(frozen=True)
class SplitSizes(Mapping[str, int]):
    """Sample counts for the default partition, not for requested subsets."""

    train: int
    val: int
    test: int

    def __getitem__(self, key: str) -> int:
        if key not in ("train", "val", "test"):
            raise KeyError(key)
        return getattr(self, key)

    def __iter__(self) -> Iterator[str]:
        return iter(("train", "val", "test"))

    def __len__(self) -> int:
        return 3


@dataclass(frozen=True)
class DatasetCapabilities:
    """Input options supported by the source without changing split membership."""

    supports_partitions: bool = False
    multi_temporal: bool = False


@dataclass(frozen=True)
class GeographySpec:
    """Coordinate reuse or an explicit explanation for unavailable coordinates."""

    alias_of: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class V1Source:
    """Local JSON shards with the retained custom JSON-metadata HDF5 fallback."""

    kind: Literal["v1"] = field(default="v1", init=False)
    root: str = "data/classification_v1.0_wds"
    hdf5_root: str = "data/classification_v1.0"
    storage_name: str | None = None
    validation_split: Literal["valid"] = "valid"


@dataclass(frozen=True)
class V2Source:
    """Upstream V2 class and explicit channel, acquisition, and label policies."""

    upstream_class: str
    kind: Literal["v2"] = field(default="v2", init=False)
    root: str = "data/geobenchv2"
    storage_name: str | None = None
    validation_split: Literal["val", "validation"] = "validation"
    band_order_strategy: Literal["flat", "by_sensor"] = "flat"
    sample_adapter: Literal["identity", "later_acquisition", "post_sar_dem", "offset_mask"] = (
        "identity"
    )
    return_stacked_image: bool | None = None
    time_step: tuple[str, ...] | None = None
    canonical_sensor_order: tuple[str, ...] | None = None


@dataclass(frozen=True)
class TorchGeoSource:
    """TorchGeo reader identity, independent of shared archive identity."""

    upstream_class: Literal["EuroSAT", "EuroSATSpatial", "RESISC45"]
    root: str
    kind: Literal["torchgeo"] = field(default="torchgeo", init=False)
    storage_name: str | None = None
    validation_split: Literal["val"] = "val"
    download_checksum: bool = False


type DatasetSource = V1Source | V2Source | TorchGeoSource


@dataclass(frozen=True)
class DatasetSpec:
    """Authoritative metadata for one benchmark identity, without a reader.

    All collections are immutable. Bands retain their scientific values and order;
    selectors return the same frozen BandSpec objects, never inferred metadata.
    Source classes and sample adapters are imported only when loading a split.
    """

    name: str
    task: Task
    num_classes: int
    bands: tuple[BandSpec, ...]
    rgb_bands: tuple[str, ...]
    split_sizes: SplitSizes
    source: DatasetSource
    multilabel: bool = False
    capabilities: DatasetCapabilities = DatasetCapabilities()
    geography: GeographySpec = GeographySpec()

    @property
    def num_channels(self) -> int:
        """Return the number of declared channels."""
        return len(self.bands)

    @property
    def rgb_indices(self) -> tuple[int, ...]:
        """Return declared RGB selector positions, including current gray/SAR sets."""
        names = tuple(b.name for b in self.bands)
        return tuple(names.index(name) for name in self.rgb_bands if name in names)

    @property
    def target_key(self) -> Literal["label", "mask"]:
        """Return the canonical target key without changing labels or masks."""
        return "mask" if self.task == "segmentation" else "label"

    @property
    def storage_name(self) -> str:
        """Return the source identity, which may be shared by distinct benchmarks."""
        return self.source.storage_name or self.name

    def select_band_specs(self, bands: Iterable[str] | None) -> tuple[BandSpec, ...]:
        """Select original band objects in the requested order."""
        if bands is None:
            return self.bands
        by_name = {band.name: band for band in self.bands}
        result = []
        for name in bands:
            if name not in by_name:
                raise ValueError(
                    f"{self.name}: unknown band {name!r}; available: {sorted(by_name)}"
                )
            result.append(by_name[name])
        return tuple(result)

    def resolve_band_specs(self, selection: str | Iterable[str]) -> tuple[BandSpec, ...]:
        """Resolve rgb, all, or ordered explicit band names without loading data."""
        if not isinstance(selection, str):
            return self.select_band_specs(selection)
        if selection == "rgb":
            return self.select_band_specs(self.rgb_bands)
        if selection == "all":
            return self.bands
        raise ValueError(
            f"Unknown band selection {selection!r}; use rgb, all, or explicit band names"
        )
