"""Strict settings for synthetic backbone and probe compute measurements."""

from typing import Any, Literal, Self

from pydantic import Field, StrictBool, StrictInt, StrictStr, field_validator

from .config_schema import (
    ModelConfig,
    RuntimeConfig,
    SegmentationConfig,
    StrictModel,
)
from .datasets import list_datasets
from .presets import ModelPreset, load_model_preset, merge_settings

type BandConfig = Literal["rgb", "s2"]
type Head = Literal["linear", "conv_block", "fpn", "dpt", "patch_linear"]


def _band_configs() -> list[BandConfig]:
    """Return independently mutable default input selections."""
    return ["rgb", "s2"]


def _heads() -> list[Head]:
    """Return the default multi-scale segmentation heads."""
    return ["fpn", "dpt"]


class FlopsRuntimeConfig(StrictModel):
    """Device and reproducible model initialization."""

    device: StrictStr = "cuda"
    seed: StrictInt = 0
    verbose: StrictBool = True

    @field_validator("device")
    @classmethod
    def validate_device(cls, value: str) -> str:
        """Validate the device without importing Torch."""
        return RuntimeConfig.validate_device(value)


class FlopsInputConfig(StrictModel):
    """Synthetic input shape and metadata, never actual dataset samples.

    ``s2`` retains the historical meaning: all bands from ``band_source``.
    The default CloudSen12 source contains twelve Sentinel-2 optical bands.
    """

    band_source: StrictStr = "cloudsen12"
    band_configs: list[BandConfig] = Field(default_factory=_band_configs, min_length=1)
    image_size: StrictInt = Field(default=224, gt=0)
    normalization: Literal["dataset", "model", "minmax", "minmax_zscore", "none"] = "dataset"

    @field_validator("band_source")
    @classmethod
    def validate_source(cls, value: str) -> str:
        """Reject blank metadata sources."""
        if not value.strip():
            raise ValueError("band_source must not be blank")
        if value not in list_datasets():
            raise ValueError(f"unknown band_source: {value!r}")
        return value

    @field_validator("band_configs")
    @classmethod
    def validate_bands(cls, value: list[BandConfig]) -> list[BandConfig]:
        """Reject repeated measurements."""
        if len(set(value)) != len(value):
            raise ValueError("band_configs must not contain duplicates")
        return value


class FlopsClassificationConfig(StrictModel):
    """Classification probe accounting on the measured embedding width."""

    head: Literal["linear", "mlp"] = "linear"
    num_classes: StrictInt = Field(default=10, gt=0)


class FlopsSegmentationConfig(StrictModel):
    """Head sweep and shared benchmark probe settings.

    An empty head or band selection disables segmentation. ``probe.layers``
    inherits model/dataset defaults unless explicitly supplied, including ``[]``.
    Each selected head replaces ``probe.head`` for its measurement.
    """

    heads: list[Head] = Field(default_factory=_heads)
    band_configs: list[BandConfig] = Field(default_factory=_band_configs)
    num_classes: StrictInt = Field(default=4, gt=0)
    probe: SegmentationConfig = Field(default_factory=SegmentationConfig)

    @field_validator("heads", "band_configs")
    @classmethod
    def validate_unique(cls, value: list[str]) -> list[str]:
        """Reject repeated sweep cells while allowing an empty selection."""
        if len(set(value)) != len(value):
            raise ValueError("selections must not contain duplicates")
        return value


class FlopsTimingConfig(StrictModel):
    """Batch-sensitive timing; FLOPs always counts a single sample."""

    batch_size: StrictInt = Field(default=64, gt=0)
    n_warmup: StrictInt = Field(default=3, ge=0)
    n_measure: StrictInt = Field(default=20, gt=0)


class FlopsOutputConfig(StrictModel):
    """Append-only CSV with the established per-cell resume keys."""

    file: StrictStr = "results/compute_cost.csv"
    resume: StrictBool = True

    @field_validator("file")
    @classmethod
    def validate_file(cls, value: str) -> str:
        """Reject blank output paths."""
        if not value.strip():
            raise ValueError("output.file must not be blank")
        return value


class FlopsConfig(StrictModel):
    """Complete synthetic compute measurement, independent of image evaluation."""

    schema_version: Literal[1] = 1
    model: ModelConfig
    runtime: FlopsRuntimeConfig = Field(default_factory=FlopsRuntimeConfig)
    input: FlopsInputConfig = Field(default_factory=FlopsInputConfig)
    classification: FlopsClassificationConfig = Field(default_factory=FlopsClassificationConfig)
    segmentation: FlopsSegmentationConfig = Field(default_factory=FlopsSegmentationConfig)
    timing: FlopsTimingConfig = Field(default_factory=FlopsTimingConfig)
    output: FlopsOutputConfig = Field(default_factory=FlopsOutputConfig)

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_version(cls, value: object) -> object:
        """Keep booleans distinct from schema version one."""
        if isinstance(value, bool):
            raise ValueError("schema_version must be the integer 1")  # noqa: TRY004 - Pydantic validation contract
        return value

    def model_dump_yaml(self) -> dict[str, Any]:
        """Serialize supplied values without overriding omitted preset defaults."""
        return self.model_dump(mode="json", exclude_unset=True)

    def resolve(self) -> tuple[Self, ModelPreset]:
        """Apply model/dataset defaults beneath explicit YAML or CLI settings."""
        preset = load_model_preset(self.model, seed=self.runtime.seed).for_dataset(
            self.input.band_source
        )
        if preset.track != "image":
            raise ValueError(f"{self.model.name!r} is a coordinate encoder; use 'coord'")
        # Explicit constructor options also take precedence over dataset overrides.
        preset = preset.model_copy(update={"kwargs": {**preset.kwargs, **self.model.kwargs}})
        inputs = preset.input.model_dump(exclude_unset=True)
        # Native-size presets have no synthetic dataset dimensions; retain the 224 default.
        if inputs.get("image_size") is None:
            inputs.pop("image_size", None)
        defaults = {
            "input": {
                key: value
                for key, value in inputs.items()
                if key in {"image_size", "normalization"}
            },
            "segmentation": {"probe": preset.segmentation.model_dump(exclude_unset=True)},
        }
        return self.model_validate(merge_settings(defaults, self.model_dump_yaml())), preset
