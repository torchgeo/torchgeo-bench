"""Input resolution and single-split loading from immutable definitions."""

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, cast

from .catalog import download_command, get_dataset_spec
from .input import (
    LoadedSplit,
    ResolvedInput,
    Split,
    _band_selection,
    resolve_input,
    validate_input_options,
)
from .spec import DatasetSpec, TorchGeoSource, V1Source, V2Source

if TYPE_CHECKING:
    from torch.utils.data import Dataset

    from .transforms import Resize

logger = logging.getLogger(__name__)


def _validate_split(split: str) -> Split:
    if not isinstance(split, str) or split not in ("train", "val", "test"):
        raise ValueError(f"Unknown split {split!r}. Expected train, val, or test.")
    return cast(Split, split)


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
    inputs: ResolvedInput | None = None,
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
        bands: ``"rgb"`` (genuine RGB only), ``"default"`` (the dataset's
            reduced/default inputs), ``"all"`` / ``None`` (all bands), or
            an explicit ordered iterable of names. Non-RGB datasets require
            an explicit selection; there is no implicit fallback.
        time_steps: Number of acquisition dates per sample.  Only accepted by
            multi-temporal sources (PASTIS); ``None`` keeps the source default.
        inputs: Optional metadata-only preflight result from ``resolve_input``.
            Its dataset, bands selector, partition and time_steps must match
            the supplied options. Reused unchanged, without selecting again.

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
    validated_split = _validate_split(split)
    if inputs is None:
        inputs = resolve_input(bench, bands=bands, partition=partition, time_steps=time_steps)
    else:
        if not isinstance(inputs, ResolvedInput):
            raise TypeError("inputs must be a ResolvedInput or None")
        validate_input_options(bench, partition, time_steps)
        if (
            inputs.spec != bench
            or inputs.selection != _band_selection(bands)
            or inputs.partition != partition
            or inputs.time_steps != time_steps
        ):
            raise ValueError("Preflight inputs disagree with requested dataset/input options")
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
