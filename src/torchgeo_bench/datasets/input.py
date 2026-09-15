"""Immutable runtime metadata for a resolved dataset split."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from torchgeo_bench.bands import BandSpec

from .spec import DatasetSpec, Split, Task

if TYPE_CHECKING:
    from torch.utils.data import Dataset

__all__ = ["LoadedSplit", "ResolvedInput", "Split"]


@dataclass(frozen=True)
class ResolvedInput:
    """Describe the channels and temporal layout actually emitted by a source.

    Band objects are the dataset's original immutable metadata, in tensor order.
    ``time_steps=None`` preserves the source's single-acquisition default. Explicit
    one-step requests also emit CHW: the supported temporal source squeezes that axis.
    """

    bands: tuple[BandSpec, ...]
    selection: str | tuple[str, ...]
    time_steps: int | None = None

    @property
    def layout(self) -> Literal["CHW", "TCHW"]:
        """Return the unbatched image layout, including single-step squeezing."""
        return "TCHW" if self.time_steps is not None and self.time_steps > 1 else "CHW"

    @property
    def num_time_steps(self) -> int:
        """Return the number of temporal slots per sample, including the default."""
        return self.time_steps if self.time_steps is not None else 1


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
