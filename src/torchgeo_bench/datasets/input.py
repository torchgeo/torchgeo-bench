"""Immutable runtime metadata for a resolved dataset split."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from torchgeo_bench.bands import BandSpec

if TYPE_CHECKING:
    from torch.utils.data import Dataset

type Split = Literal["train", "val", "test"]


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
    dataset_name: str
    split: Split
    partition: str
    input: ResolvedInput
    task: Literal["classification", "segmentation"]
    num_classes: int
    multilabel: bool

    @property
    def bands(self) -> tuple[BandSpec, ...]:
        """Return the original BandSpec objects in emitted channel order."""
        return self.input.bands

    @property
    def target_key(self) -> Literal["label", "mask"]:
        """Return the existing canonical target key without changing target values."""
        return "mask" if self.task == "segmentation" else "label"
