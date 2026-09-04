# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Strict YAML configuration for the core image benchmark."""

import pathlib
import re
from typing import Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)
from yaml.constructor import ConstructorError


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        """Construct a mapping and reject duplicate or unhashable keys."""
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            try:
                duplicate = key in mapping
            except TypeError as error:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found an unhashable key",
                    key_node.start_mark,
                ) from error
            if duplicate:
                raise ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


_UniqueKeyLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)|\.[0-9_]+(?:[eE][-+]?[0-9]+)?
        |[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)


class StrictModel(BaseModel):
    """Base for configuration sections with strict fields and no extras."""

    model_config = ConfigDict(
        extra="forbid", strict=True, validate_default=True, allow_inf_nan=False
    )


class ModelConfig(StrictModel):
    """Selected model preset."""

    name: StrictStr = Field(min_length=1)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        """Reject whitespace-only model names."""
        if not value.strip():
            raise ValueError("model name must not be blank")
        return value


class InputConfig(StrictModel):
    """Input bands, resizing, dataset partition, and normalization."""

    bands: StrictStr | list[StrictStr] = "rgb"
    partition: StrictStr = "default"
    time_steps: StrictInt | None = Field(default=None, gt=0)
    image_size: StrictInt | None = Field(default=224, gt=0)
    interpolation: Literal["area", "bilinear", "bicubic", "nearest"] = "bilinear"
    normalization: Literal["dataset", "model", "minmax", "none"] = "dataset"

    @field_validator("partition")
    @classmethod
    def validate_partition(cls, value: str) -> str:
        """Reject whitespace-only partition names."""
        if not value.strip():
            raise ValueError("partition must not be blank")
        return value

    @field_validator("bands")
    @classmethod
    def validate_bands(cls, value: StrictStr | list[StrictStr]) -> StrictStr | list[StrictStr]:
        """Reject empty band specifications."""
        if isinstance(value, str) and not value.strip():
            raise ValueError("bands must not be empty")
        if isinstance(value, list) and (
            not value or any(not band.strip() for band in value) or len(set(value)) != len(value)
        ):
            raise ValueError("bands must contain non-empty names")
        return value


class LinearConfig(StrictModel):
    """Linear probe hyperparameters."""

    c_log10_start: StrictFloat = -6.0
    c_log10_stop: StrictFloat = 4.0
    c_count: StrictInt = Field(default=40, gt=0)
    refit_train_val: StrictBool = True

    @model_validator(mode="after")
    def validate_range(self) -> "LinearConfig":
        """Require an ascending regularization range."""
        if self.c_log10_stop < self.c_log10_start:
            raise ValueError("c_log10_stop must be greater than or equal to c_log10_start")
        return self


class CalibrationConfig(StrictModel):
    """Classification calibration metrics."""

    n_bins_knn: StrictInt | None = Field(default=None, gt=0)
    n_bins_linear: StrictInt = Field(default=15, gt=0)
    temp_scale: StrictBool = False


class ClassificationConfig(StrictModel):
    """KNN, linear probe, and bootstrap settings."""

    methods: list[Literal["knn", "linear"]] = Field(
        default_factory=lambda: ["knn", "linear"], min_length=1
    )
    knn_k: StrictInt = Field(default=5, gt=0)
    knn_device: StrictStr | None = None
    linear: LinearConfig = Field(default_factory=LinearConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    bootstrap_samples: StrictInt = Field(default=200, ge=0)

    @field_validator("methods")
    @classmethod
    def validate_methods(
        cls, value: list[Literal["knn", "linear"]]
    ) -> list[Literal["knn", "linear"]]:
        """Reject repeated method selections."""
        if len(set(value)) != len(value):
            raise ValueError("methods must not contain duplicates")
        return value

    @model_validator(mode="after")
    def validate_calibration(self) -> "ClassificationConfig":
        """Require held-out validation when temperature scaling is enabled."""
        if self.calibration.temp_scale and (
            "linear" not in self.methods or self.linear.refit_train_val
        ):
            raise ValueError("temp_scale requires linear selected and refit_train_val=false")
        return self

    @field_validator("knn_device")
    @classmethod
    def validate_knn_device(cls, value: str | None) -> str | None:
        """Reject malformed KNN device strings without importing Torch."""
        if value is None or value in {"cpu", "cuda"} or re.fullmatch(r"cuda:[0-9]+", value):
            return value
        raise ValueError("knn_device must be 'cpu', 'cuda', or 'cuda:<index>'")


class SegmentationConfig(StrictModel):
    """Segmentation probe settings."""

    head: Literal["linear", "conv_block", "fpn", "dpt", "patch_linear"] = "fpn"
    layers: list[StrictStr] = Field(default_factory=list)
    learning_rate: StrictFloat = Field(default=1e-3, gt=0)
    epochs: StrictInt = Field(default=10, gt=0)
    batch_size: StrictInt = Field(default=64, gt=0)
    temporal_pool: Literal["mean", "max"] = "mean"
    scheduler: Literal["cosine", "none"] = "cosine"
    ignore_index: StrictInt = 255
    cache_features: StrictBool = True
    cache_dtype: Literal["float16", "float32"] = "float16"


class RuntimeConfig(StrictModel):
    """Execution settings."""

    device: StrictStr = "cuda:0"
    batch_size: StrictInt = Field(default=64, gt=0)
    workers: StrictInt = Field(default=4, ge=0)
    seed: StrictInt = 0
    verbose: StrictBool = False

    @field_validator("device")
    @classmethod
    def validate_device(cls, value: str) -> str:
        """Reject malformed device strings without importing Torch."""
        if value == "auto" or re.fullmatch(r"(cpu|cuda(?::[0-9]+)?)", value):
            return value
        raise ValueError("device must be 'auto', 'cpu', 'cuda', or 'cuda:<index>'")


class OutputConfig(StrictModel):
    """Result storage settings."""

    directory: StrictStr = "results/models"
    file: StrictStr | None = None
    resume: StrictBool = False

    @field_validator("directory", "file")
    @classmethod
    def validate_paths(cls, value: str | None) -> str | None:
        """Reject blank paths while allowing a null optional file."""
        if value is not None and not value.strip():
            raise ValueError("paths must not be blank")
        return value


class RunConfig(StrictModel):
    """Complete core image benchmark configuration."""

    schema_version: Literal[1] = 1
    model: ModelConfig
    datasets: list[StrictStr] = Field(min_length=1)
    input: InputConfig = Field(default_factory=InputConfig)
    classification: ClassificationConfig = Field(default_factory=ClassificationConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @field_validator("schema_version", mode="before")
    @classmethod
    def reject_bool_schema_version(cls, value: object) -> object:
        """Reject ``true`` because booleans are integer subclasses in Python."""
        if isinstance(value, bool):
            raise ValueError("schema_version must be the integer 1")
        return value

    @field_validator("datasets")
    @classmethod
    def validate_datasets(cls, value: list[StrictStr]) -> list[StrictStr]:
        """Reject empty or repeated dataset names."""
        if any(not name.strip() for name in value):
            raise ValueError("datasets must contain non-empty names")
        if len(set(value)) != len(value):
            raise ValueError("datasets must not contain duplicates")
        return value

    def model_dump_yaml(self) -> dict[str, Any]:
        """Return a YAML-serializable resolved configuration."""
        return self.model_dump(mode="json")


def load_yaml(path: str | pathlib.Path) -> dict[str, Any]:
    """Load one YAML mapping with safe tags and unique keys."""
    with pathlib.Path(path).open(encoding="utf-8") as file:
        value = yaml.load(file, Loader=_UniqueKeyLoader)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top level must be a YAML mapping")
    return value


def load_run_config(path: str | pathlib.Path) -> RunConfig:
    """Load and strictly validate a core image benchmark configuration."""
    return RunConfig.model_validate(load_yaml(path), strict=True)


def validate_run_config(value: dict[str, Any]) -> RunConfig:
    """Validate a mapping after explicit CLI overrides are applied."""
    return RunConfig.model_validate(value, strict=True)
