"""Immutable runtime metadata for a resolved dataset split."""

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal

from torchgeo_bench.bands import BandSpec

from .catalog import get_dataset_spec
from .spec import DatasetSpec, Split, Task

if TYPE_CHECKING:
    from torch.utils.data import Dataset

__all__ = ["LoadedSplit", "ResolvedInput", "Split", "resolve_input"]

# Version 2 replaces the channel-order-only version with explicit source/input metadata.
DATASET_INPUT_PROTOCOL_VERSION = 2


def _band_selection(bands: str | Iterable[str] | None) -> str | tuple[str, ...]:
    if bands is None:
        return "all"
    if isinstance(bands, str):
        return bands
    if not isinstance(bands, Iterable) or isinstance(bands, dict | set | frozenset):
        raise TypeError(
            "bands must be rgb, default, all, None, or an ordered iterable of band names"
        )
    selection = tuple(bands)
    if any(not isinstance(name, str) for name in selection):
        raise TypeError("band names must be strings")
    return selection


def validate_input_options(spec: DatasetSpec, partition: str, time_steps: int | None) -> None:
    """Reject unsupported source options without importing readers or touching data."""
    spec.validate_source()
    if not isinstance(partition, str):
        raise TypeError("partition must be a string")
    if not partition.strip():
        raise ValueError("partition must not be blank")
    if partition != "default" and not spec.capabilities.supports_partitions:
        raise ValueError(f"Dataset {spec.name!r} does not support custom partitions.")
    if time_steps is not None:
        if type(time_steps) is not int:
            raise TypeError("time_steps must be an integer or None")
        if time_steps < 1:
            raise ValueError("time_steps must be positive")
        if not spec.capabilities.multi_temporal:
            raise ValueError(f"{spec.name} is not multi-temporal; drop time_steps.")


@dataclass(frozen=True)
class ResolvedInput:
    """Describe the channels and temporal layout actually emitted by a source.

    Band objects are the dataset's original immutable metadata, in tensor order.
    ``time_steps=None`` preserves the source's single-acquisition default. Explicit
    one-step requests also emit CHW: the supported temporal source squeezes that axis.
    """

    spec: DatasetSpec
    bands: tuple[BandSpec, ...]
    selection: str | tuple[str, ...]
    time_steps: int | None = None
    partition: str = "default"

    @property
    def band_names(self) -> tuple[str, ...]:
        """Return short channel names in emitted tensor order."""
        return tuple(band.name for band in self.bands)

    @property
    def description(self) -> dict[str, Any]:
        """Return versioned, JSON-serializable scientific input definitions only.

        Source adapter versions describe deliberate reader/acquisition policies, not
        the environment's installed dependency versions. Only selected BandSpecs are
        included; changing unused channel metadata does not invalidate an RGB run.
        """
        source = asdict(self.spec.source)
        source.pop("download_checksum", None)
        source["storage_name"] = self.spec.storage_name
        return {
            "protocol_version": DATASET_INPUT_PROTOCOL_VERSION,
            "dataset": self.spec.name,
            "task": self.spec.task,
            "num_classes": self.spec.num_classes,
            "multilabel": self.spec.multilabel,
            "target_key": self.spec.target_key,
            "source": source,
            "split_policy": {
                "train": "train",
                "val": self.spec.source.validation_split,
                "test": "test",
                "partition": self.partition,
            },
            "selection": self.selection
            if isinstance(self.selection, str)
            else list(self.selection),
            "bands": [asdict(band) for band in self.bands],
            "layout": self.layout,
            "time_steps": self.time_steps,
            "num_time_steps": self.num_time_steps,
        }

    @property
    def fingerprint(self) -> str:
        """Hash the canonical input description identically in preflight and loading."""
        encoded = json.dumps(
            self.description, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(encoded.encode()).hexdigest()

    @property
    def layout(self) -> Literal["CHW", "TCHW"]:
        """Return the unbatched image layout, including single-step squeezing."""
        return "TCHW" if self.time_steps is not None and self.time_steps > 1 else "CHW"

    @property
    def num_time_steps(self) -> int:
        """Return the number of temporal slots per sample, including the default."""
        return self.time_steps if self.time_steps is not None else 1


def resolve_input(
    dataset_name: DatasetSpec | str,
    *,
    bands: str | Iterable[str] | None = "rgb",
    partition: str = "default",
    time_steps: int | None = None,
) -> ResolvedInput:
    """Resolve and validate a dataset input without source imports or data access.

    ``rgb`` remains the default and requires genuine RGB. ``default`` explicitly
    selects the dataset's reduced inputs, including CaFFe gray and KuroSiwo vv/vh.
    Pass the result as ``load_split(..., inputs=resolved)`` with the same options
    to reuse these exact BandSpec objects instead of selecting again.
    """
    spec = get_dataset_spec(dataset_name) if isinstance(dataset_name, str) else dataset_name
    if not isinstance(spec, DatasetSpec):
        raise TypeError("dataset_name must be a DatasetSpec or canonical name")
    validate_input_options(spec, partition, time_steps)
    selection = _band_selection(bands)
    selected = spec.resolve_band_specs(selection)
    if not selected:
        raise ValueError("bands must select at least one channel")
    return ResolvedInput(spec, selected, selection, time_steps, partition)


@dataclass(frozen=True)
class LoadedSplit:
    """A dataset and its authoritative input, identity, and target metadata.

    This runtime result owns no DataLoader and must not be serialized into run
    configuration or resume hashes. Callers choose their own batching policy.
    """

    dataset: "Dataset"
    spec: DatasetSpec
    split: Split
    partition: str
    input: ResolvedInput

    @property
    def dataset_name(self) -> str:
        """Return the authoritative benchmark identity."""
        return self.spec.name

    @property
    def task(self) -> Task:
        """Return the authoritative task."""
        return self.spec.task

    @property
    def num_classes(self) -> int:
        """Return the declared target count."""
        return self.spec.num_classes

    @property
    def multilabel(self) -> bool:
        """Return whether labels are multi-hot vectors."""
        return self.spec.multilabel

    @property
    def bands(self) -> tuple[BandSpec, ...]:
        """Return the original BandSpec objects in emitted channel order."""
        return self.input.bands

    @property
    def target_key(self) -> Literal["label", "mask"]:
        """Return the existing canonical target key without changing target values."""
        return self.spec.target_key
