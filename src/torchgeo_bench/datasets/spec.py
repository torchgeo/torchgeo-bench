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

    def validate(self, source: "DatasetSource") -> None:
        """Check that declared options can actually be honored by this backend."""
        if type(self.supports_partitions) is not bool or type(self.multi_temporal) is not bool:
            raise TypeError("capabilities must contain boolean supported-option flags")
        if self.supports_partitions and not isinstance(source, V1Source):
            raise ValueError("Only V1 sources support partitions")
        temporal = isinstance(source, V2Source) and source.upstream_class == "GeoBenchPASTIS"
        if self.multi_temporal != temporal:
            raise ValueError("Only the PASTIS source supports multi-temporal inputs")


@dataclass(frozen=True)
class GeographySpec:
    """Coordinate reuse or an explicit explanation for unavailable coordinates."""

    alias_of: str | None = None
    reason: str | None = None


@dataclass(frozen=True)
class V1Source:
    """Local JSON-metadata shards acquired through the explicit verified download."""

    kind: Literal["v1"] = field(default="v1", init=False)
    root: str = "data/classification_v1.0_wds"
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
    align_to_output: bool = False


@dataclass(frozen=True)
class TorchGeoSource:
    """TorchGeo reader identity, independent of shared archive identity."""

    upstream_class: Literal["EuroSAT", "EuroSATSpatial", "RESISC45"]
    root: str
    kind: Literal["torchgeo"] = field(default="torchgeo", init=False)
    storage_name: str | None = None
    validation_split: Literal["val"] = "val"
    download_checksum: bool = False

    def validate(self) -> None:
        """Reject unsupported TorchGeo reader identities and options."""
        if self.upstream_class not in ("EuroSAT", "EuroSATSpatial", "RESISC45"):
            raise ValueError(f"Unsupported TorchGeo source {self.upstream_class!r}")
        if self.validation_split != "val" or type(self.download_checksum) is not bool:
            raise ValueError("Unsupported TorchGeo source options")


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

    def validate_source(self) -> None:
        """Reject unsupported source/capability combinations before importing readers."""
        if self.task not in ("classification", "segmentation"):
            raise ValueError(f"Unsupported task {self.task!r}")
        if type(self.multilabel) is not bool or (self.multilabel and self.task != "classification"):
            raise ValueError("multilabel must be boolean and requires classification")
        if type(self.num_classes) is not int or self.num_classes < 1:
            raise ValueError("num_classes must be a positive integer")
        if not isinstance(self.capabilities, DatasetCapabilities):
            raise TypeError("capabilities must be DatasetCapabilities")
        source = self.source
        if not isinstance(source, V1Source | V2Source | TorchGeoSource):
            raise TypeError(f"Unsupported dataset source {type(source).__name__}")
        self.capabilities.validate(source)
        if isinstance(source, V1Source):
            if source.validation_split != "valid":
                raise ValueError("V1 validation_split must be 'valid'")
        elif isinstance(source, TorchGeoSource):
            source.validate()
        else:
            _validate_v2_source(source)

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


def _validate_v2_source(source: V2Source) -> None:
    flat = {
        "GeoBenchBurnScars",
        "GeoBenchCaFFe",
        "GeoBenchCloudSen12",
        "GeoBenchFLAIR2",
        "GeoBenchForestnet",
        "GeoBenchFieldsOfTheWorld",
        "GeoBenchSpaceNet7",
    }
    grouped = {
        "GeoBenchBENV2",
        "GeoBenchDynamicEarthNet",
        "GeoBenchKuroSiwo",
        "GeoBenchPASTIS",
        "GeoBenchSo2Sat",
        "GeoBenchSpaceNet2",
        "GeoBenchTreeSatAI",
    }
    if source.upstream_class not in flat | grouped:
        raise ValueError(f"Unsupported V2 source {source.upstream_class!r}")
    strategy = "by_sensor" if source.upstream_class in grouped else "flat"
    adapter = {
        "GeoBenchFieldsOfTheWorld": "later_acquisition",
        "GeoBenchKuroSiwo": "post_sar_dem",
        "GeoBenchSpaceNet2": "offset_mask",
        "GeoBenchSpaceNet7": "offset_mask",
    }.get(source.upstream_class, "identity")
    if source.band_order_strategy != strategy or source.sample_adapter != adapter:
        raise ValueError(f"{source.upstream_class}: unsupported band/sample source policy")
    alignment = source.upstream_class in {
        "GeoBenchTreeSatAI",
        "GeoBenchSpaceNet2",
        "GeoBenchPASTIS",
        "GeoBenchDynamicEarthNet",
    }
    if type(source.align_to_output) is not bool or source.align_to_output != alignment:
        raise ValueError(f"{source.upstream_class}: unsupported sensor alignment policy")
    if source.validation_split not in ("val", "validation"):
        raise ValueError("Unsupported V2 validation_split")
    if adapter == "post_sar_dem":
        if (
            source.return_stacked_image is not False
            or source.time_step != ("post",)
            or source.canonical_sensor_order != ("sar", "dem")
        ):
            raise ValueError("KuroSiwo requires unstacked post-event SAR/DEM source options")
    elif any(
        option is not None
        for option in (source.return_stacked_image, source.time_step, source.canonical_sensor_order)
    ):
        raise ValueError(f"{source.upstream_class}: unsupported source options")
