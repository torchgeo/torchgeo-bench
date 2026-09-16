"""Input resolution and single-split loading from immutable definitions."""

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, cast

from .catalog import download_command, get_dataset_spec
from .input import LoadedSplit, ResolvedInput, Split
from .spec import DatasetSpec, TorchGeoSource, V1Source, V2Source

if TYPE_CHECKING:
    from torch.utils.data import Dataset

    from .transforms import Resize

logger = logging.getLogger(__name__)


def _validate_split_options(
    bench: DatasetSpec, split: str, partition: str, time_steps: int | None
) -> Split:
    if not isinstance(split, str) or split not in ("train", "val", "test"):
        raise ValueError(f"Unknown split {split!r}. Expected train, val, or test.")
    if not isinstance(partition, str):
        raise TypeError("partition must be a string")
    if not partition.strip():
        raise ValueError("partition must not be blank")
    if partition != "default" and not bench.capabilities.supports_partitions:
        raise ValueError(f"Dataset {bench.name!r} does not support custom partitions.")
    if time_steps is not None:
        if type(time_steps) is not int:
            raise TypeError("time_steps must be an integer or None")
        if time_steps < 1:
            raise ValueError("time_steps must be positive")
        if not bench.capabilities.multi_temporal:
            raise ValueError(f"{bench.name} is not multi-temporal; drop time_steps.")
    return cast(Split, split)


def _resolve_input(
    bench: DatasetSpec, bands: str | Iterable[str] | None, time_steps: int | None
) -> ResolvedInput:
    selection: str | tuple[str, ...]
    if bands is None:
        selection = "all"
    elif isinstance(bands, str):
        selection = bands
    else:
        if not isinstance(bands, Iterable) or isinstance(bands, dict | set | frozenset):
            raise TypeError("bands must be rgb, all, None, or an ordered iterable of band names")
        selection = tuple(bands)
        if any(not isinstance(name, str) for name in selection):
            raise TypeError("band names must be strings")
    specs = tuple(bench.resolve_band_specs(selection))
    if not specs:
        raise ValueError("bands must select at least one channel")
    return ResolvedInput(specs, selection, time_steps)


def _validate_resize_options(image_size: int | None, interpolation: str) -> None:
    if image_size is not None:
        if type(image_size) is not int:
            raise TypeError("image_size must be an integer or None")
        if image_size < 1:
            raise ValueError("image_size must be positive")
    if interpolation not in ("area", "bicubic", "bilinear", "nearest"):
        raise ValueError(
            f"interpolation must be one of area, bicubic, bilinear, nearest; got {interpolation!r}"
        )


def _load_source(
    spec: DatasetSpec,
    split: Split,
    *,
    inputs: ResolvedInput,
    partition: str,
    resize: "Resize | None",
) -> "Dataset":
    match spec.source:
        case V1Source():
            from .geobench_v1 import load_v1_split
            from .transforms import CanonicalTransform

            return load_v1_split(
                spec,
                split,
                inputs=inputs,
                partition=partition,
                transform=CanonicalTransform(spec, inputs, resize),
            )
        case V2Source():
            from .geobench_v2 import load_v2_split

            return load_v2_split(spec, split, inputs=inputs, resize=resize)
        case TorchGeoSource():
            from .torchgeo import load_torchgeo_split

            return load_torchgeo_split(spec, split, inputs=inputs, resize=resize)
        case _:
            raise TypeError(f"Unsupported dataset source {type(spec.source).__name__}")


def load_split(  # noqa: PLR0913 - explicit public input options, without batching policy.
    dataset_name: DatasetSpec | str,
    split: str,
    *,
    partition: str = "default",
    image_size: int | None = None,
    interpolation: str = "bilinear",
    bands: str | Iterable[str] | None = "rgb",
    time_steps: int | None = None,
) -> LoadedSplit:
    """Load exactly one split and its resolved metadata, without batching.

    Datasets always emit raw float32 values; per-channel normalization is
    the model's responsibility (see :class:`~torchgeo_bench.models.interface.BenchModel`).
    No samples are read to infer metadata, and no unrelated split is constructed.

    Args:
        dataset_name: Canonical catalog name or an immutable dataset definition.
        split: ``"train"``, ``"val"``, or ``"test"``.
        partition: Partition for this split (only supported by datasets where
            ``capabilities.supports_partitions`` is ``True``).
        image_size: If set, resize images (and masks, with nearest) to this
            square size at sample time.
        interpolation: Resize interpolation for images (``"area"``, ``"bicubic"``,
            ``"bilinear"``, ``"nearest"``).
        bands: ``"rgb"`` (use the dataset's ``rgb_bands``), ``"all"`` /
            ``None`` (load all bands), or an explicit iterable of band names.
        time_steps: Number of acquisition dates per sample.  Only accepted by
            multi-temporal sources (PASTIS); ``None`` keeps the source default.

    Returns:
        The requested Dataset and immutable metadata describing its output.

    Raises:
        KeyError: If ``dataset_name`` is not registered.
        TypeError: If an input option has the wrong type.
        ValueError: If an option is invalid or unsupported by this dataset.
        FileNotFoundError: If a required dataset file is missing.
    """
    bench = get_dataset_spec(dataset_name) if isinstance(dataset_name, str) else dataset_name
    if not isinstance(bench, DatasetSpec):
        raise TypeError("dataset_name must be a DatasetSpec or canonical name")
    bench.validate_source()
    validated_split = _validate_split_options(bench, split, partition, time_steps)
    inputs = _resolve_input(bench, bands, time_steps)
    _validate_resize_options(image_size, interpolation)

    from torchgeo.datasets import DatasetNotFoundError

    from .transforms import Resize

    resize = Resize(image_size, interpolation) if image_size is not None else None
    try:
        dataset = _load_source(
            bench, validated_split, inputs=inputs, partition=partition, resize=resize
        )
    except (
        FileNotFoundError,
        DatasetNotFoundError,
    ) as error:  # allow-except: add download command.
        raise FileNotFoundError(
            f"Required files for {bench.name!r} split {split!r} are missing. "
            f"Run `{download_command(bench)}`."
        ) from error

    return LoadedSplit(
        dataset=dataset,
        spec=bench,
        split=validated_split,
        partition=partition,
        input=inputs,
    )


__all__ = ["load_split"]
