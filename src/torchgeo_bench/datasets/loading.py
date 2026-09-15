"""Dataset registry, input resolution, resizing, and single-split loading.

Wrapper modules and torch load only when needed; listing datasets does not import them.
"""

import logging
from collections.abc import Iterable
from importlib import import_module
from typing import TYPE_CHECKING, Literal, cast

from .base import BenchDataset
from .input import LoadedSplit, ResolvedInput, Split

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


# Store (submodule, class name, task) so catalogs need no wrapper imports.
_REGISTRY_SPEC: dict[str, tuple[str, str, Literal["classification", "segmentation"]]] = {
    # V1 classification
    "m-eurosat": ("m_eurosat", "MEurosat", "classification"),
    "m-forestnet": ("m_forestnet", "MForestnet", "classification"),
    "m-so2sat": ("m_so2sat", "MSo2Sat", "classification"),
    "m-pv4ger": ("m_pv4ger", "MPv4ger", "classification"),
    "m-brick-kiln": ("m_brick_kiln", "MBrickKiln", "classification"),
    "m-bigearthnet": ("m_bigearthnet", "MBigEarthNet", "classification"),
    # V2 classification
    "benv2": ("benv2", "BENV2", "classification"),
    "treesatai": ("treesatai", "TreeSatAI", "classification"),
    "so2sat": ("so2sat", "So2Sat", "classification"),
    "forestnet": ("forestnet", "Forestnet", "classification"),
    # V2 segmentation
    "caffe": ("caffe", "CaFFe", "segmentation"),
    "burn_scars": ("burn_scars", "BurnScars", "segmentation"),
    "cloudsen12": ("cloudsen12", "CloudSEN12", "segmentation"),
    "dynamic_earthnet": ("dynamic_earthnet", "DynamicEarthNet", "segmentation"),
    "flair2": ("flair2", "FLAIR2", "segmentation"),
    "fotw": ("fotw", "FieldsOfTheWorld", "segmentation"),
    "kuro_siwo": ("kuro_siwo", "KuroSiwo", "segmentation"),
    "pastis": ("pastis", "PASTIS", "segmentation"),
    "spacenet2": ("spacenet2", "SpaceNet2", "segmentation"),
    "spacenet7": ("spacenet7", "SpaceNet7", "segmentation"),
    # torchgeo datasets
    "eurosat": ("eurosat", "EuroSAT", "classification"),
    "eurosat-spatial": ("eurosat", "EuroSATSpatial", "classification"),
    "resisc45": ("resisc45", "RESISC45", "classification"),
}


def get_bench_dataset_class(name: str) -> type[BenchDataset]:
    """Look up a dataset by name and return its :class:`BenchDataset` class.

    Args:
        name: Dataset identifier (e.g. ``"m-eurosat"``, ``"burn_scars"``).

    Returns:
        The registered :class:`BenchDataset` subclass.

    Raises:
        KeyError: If *name* is not in the registry.
    """
    if name not in _REGISTRY_SPEC:
        available = ", ".join(sorted(_REGISTRY_SPEC))
        raise KeyError(f"Unknown dataset '{name}'. Available: {available}")
    module_name, class_name, _ = _REGISTRY_SPEC[name]
    cls: type[BenchDataset] = getattr(import_module(f".{module_name}", __package__), class_name)
    if cls.name != name:
        raise RuntimeError(
            f"Registry mismatch: entry '{name}' resolved to {class_name} "
            f"whose declared name is '{cls.name}'."
        )
    return cls


def list_datasets() -> list[str]:
    """Return sorted names of all registered benchmark datasets."""
    return sorted(_REGISTRY_SPEC)


def get_dataset_task(name: str) -> Literal["classification", "segmentation"]:
    """Return a registered dataset's task without importing its wrapper.

    Raises:
        KeyError: If *name* is not in the registry.
    """
    return _REGISTRY_SPEC[name][2]


def download_command(name: str) -> str:
    """Return the command that downloads one registered dataset."""
    if name.startswith("m-"):
        return f"torchgeo-bench download geobench_v1 --datasets {name}"
    if name in {"eurosat", "eurosat-spatial", "resisc45"}:
        return f"torchgeo-bench download {name.removesuffix('-spatial')}"
    return f"torchgeo-bench download geobench_v2 --datasets {name}"


class _ResizeTransform:
    """Sample-level transform that resizes ``image`` (and ``mask``)."""

    valid_modes = ("area", "bicubic", "bilinear", "nearest")

    def __init__(self, image_size: int, interp_mode: str) -> None:
        if interp_mode not in self.valid_modes:
            raise ValueError(
                f"interpolation must be one of {self.valid_modes}, got {interp_mode!r}."
            )
        self.image_size = image_size
        self.interp_mode = interp_mode
        self.align_corners = False if interp_mode in ("bicubic", "bilinear") else None

    def __call__(self, sample: dict) -> dict:
        import torch.nn.functional as F

        image_size = self.image_size
        img: torch.Tensor = sample["image"]
        h, w = img.shape[-2], img.shape[-1]
        if h != image_size or w != image_size:
            squeeze_batch = img.ndim == 3
            resize_input = img.unsqueeze(0) if squeeze_batch else img
            img = F.interpolate(
                resize_input,
                size=(image_size, image_size),
                mode=self.interp_mode,
                align_corners=self.align_corners,
            )
            if squeeze_batch:
                img = img.squeeze(0)
            sample["image"] = img
        if "mask" in sample:
            mask: torch.Tensor = sample["mask"].float()
            h_m, w_m = mask.shape[-2], mask.shape[-1]
            if h_m != image_size or w_m != image_size:
                if mask.ndim == 2:
                    resize_mask = mask.unsqueeze(0).unsqueeze(0)
                    squeeze_dims = 2
                elif mask.ndim == 3:
                    resize_mask = mask.unsqueeze(0)
                    squeeze_dims = 1
                else:
                    resize_mask = mask
                    squeeze_dims = 0
                mask = F.interpolate(
                    resize_mask,
                    size=(image_size, image_size),
                    mode="nearest",
                )
                for _ in range(squeeze_dims):
                    mask = mask.squeeze(0)
                mask = mask.long()
                sample["mask"] = mask
        return sample


def _validate_split_options(
    bench: BenchDataset, split: str, partition: str, time_steps: int | None
) -> Split:
    if not isinstance(split, str) or split not in ("train", "val", "test"):
        raise ValueError(f"Unknown split {split!r}. Expected train, val, or test.")
    if not isinstance(partition, str):
        raise TypeError("partition must be a string")
    if not partition.strip():
        raise ValueError("partition must not be blank")
    if partition != "default" and not bench.supports_partitions:
        raise ValueError(f"Dataset {bench.name!r} does not support custom partitions.")
    if time_steps is not None:
        if type(time_steps) is not int:
            raise TypeError("time_steps must be an integer or None")
        if time_steps < 1:
            raise ValueError("time_steps must be positive")
        if not bench.multi_temporal:
            raise ValueError(f"{bench.name} is not multi-temporal; drop time_steps.")
    return cast(Split, split)


def _resolve_input(
    bench: BenchDataset, bands: str | Iterable[str] | None, time_steps: int | None
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


def _resize_transform(image_size: int | None, interpolation: str) -> _ResizeTransform | None:
    if image_size is not None:
        if type(image_size) is not int:
            raise TypeError("image_size must be an integer or None")
        if image_size < 1:
            raise ValueError("image_size must be positive")
    if interpolation not in _ResizeTransform.valid_modes:
        raise ValueError(
            f"interpolation must be one of {_ResizeTransform.valid_modes}, got {interpolation!r}."
        )
    return _ResizeTransform(image_size, interpolation) if image_size is not None else None


def load_split(  # noqa: PLR0913 - explicit public input options, without batching policy.
    dataset_name: str,
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
        dataset_name: Identifier registered in ``_REGISTRY_SPEC``.
        split: ``"train"``, ``"val"``, or ``"test"``.
        partition: Partition for this split (only supported by datasets where
            :attr:`~.base.BenchDataset.supports_partitions` is ``True``).
        image_size: If set, resize images (and masks, with nearest) to this
            square size at sample time.
        interpolation: Resize interpolation for images (``"area"``, ``"bicubic"``,
            ``"bilinear"``, ``"nearest"``).
        bands: ``"rgb"`` (use the dataset's ``rgb_bands``), ``"all"`` /
            ``None`` (load all bands), or an explicit iterable of band names.
        time_steps: Number of acquisition dates per sample.  Only accepted by
            multi-temporal wrappers (PASTIS); ``None`` keeps the source default.

    Returns:
        The requested Dataset and immutable metadata describing its output.

    Raises:
        KeyError: If ``dataset_name`` is not registered.
        TypeError: If an input option has the wrong type.
        ValueError: If an option is invalid or unsupported by this dataset.
        FileNotFoundError: If a required dataset file is missing.
    """
    from torchgeo.datasets import DatasetNotFoundError

    cls = get_bench_dataset_class(dataset_name)
    bench = cls()

    validated_split = _validate_split_options(bench, split, partition, time_steps)
    inputs = _resolve_input(bench, bands, time_steps)
    transform = _resize_transform(image_size, interpolation)
    try:
        dataset = bench._load_split(
            validated_split, inputs=inputs, partition=partition, transform=transform
        )
    except (
        FileNotFoundError,
        DatasetNotFoundError,
    ) as error:  # allow-except: add download command.
        raise FileNotFoundError(
            f"Required files for {dataset_name!r} split {split!r} are missing. "
            f"Run `{download_command(dataset_name)}`."
        ) from error

    return LoadedSplit(
        dataset=dataset,
        dataset_name=dataset_name,
        split=validated_split,
        partition=partition,
        input=inputs,
        task=bench.task,
        num_classes=bench.num_classes,
        multilabel=bench.multilabel,
    )


__all__ = [
    "get_bench_dataset_class",
    "get_dataset_task",
    "list_datasets",
    "load_split",
]
