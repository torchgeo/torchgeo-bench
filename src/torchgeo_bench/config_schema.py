# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Validated configuration for the core image benchmark.

This module is deliberately independent of the benchmark runner.  It parses
YAML and validates the public configuration contract without importing Torch,
TorchGeo, or any model implementation.
"""

import math
import pathlib
import re
from typing import Literal

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
)
from yaml.constructor import ConstructorError


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicate mapping keys."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict:
        """Construct a mapping and reject duplicate keys."""
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in mapping:
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
        r"""^(?:
        [-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\.[0-9_]+(?:[eE][-+]?[0-9]+)?
        |[-+]?\.(?:inf|Inf|INF)
        |\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)


class StrictModel(BaseModel):
    """Base for configuration sections with no implicit coercion."""

    model_config = ConfigDict(extra="forbid", strict=True)


class ModelConfig(StrictModel):
    """Model preset selected by the benchmark."""

    preset: StrictStr = Field(min_length=1)


class InputConfig(StrictModel):
    """Image decoding, resizing, band selection, and normalization."""

    bands: StrictStr | list[StrictStr] = "rgb"
    image_size: StrictInt | None = Field(default=224, gt=0)
    interpolation: Literal["area", "bilinear", "bicubic", "nearest"] = "bilinear"
    normalization: Literal[
        "bandspec_zscore", "model_native", "minmax", "minmax_zscore", "identity"
    ] = "bandspec_zscore"

    @staticmethod
    def _nonempty_bands(value: StrictStr | list[StrictStr]) -> StrictStr | list[StrictStr]:
        if isinstance(value, str) and not value:
            raise ValueError("bands must not be empty")
        if isinstance(value, list) and (not value or any(not band for band in value)):
            raise ValueError("bands must contain at least one non-empty band")
        return value

    _validate_bands = field_validator("bands")(_nonempty_bands)


class ClassificationConfig(StrictModel):
    """Classification probe settings."""

    methods: list[Literal["knn", "linear"]] = Field(
        default_factory=lambda: ["knn", "linear"], min_length=1
    )
    knn_neighbors: StrictInt = Field(default=5, gt=0)
    linear_c_range: list[StrictInt | StrictFloat] = Field(
        default_factory=lambda: [-6, 4, 40], min_length=3, max_length=3
    )
    merge_train_val: StrictBool = True
    bootstrap_samples: StrictInt = Field(default=200, ge=0)

    @staticmethod
    def _valid_c_range(value: list[StrictInt | StrictFloat]) -> list[StrictInt | StrictFloat]:
        values = [float(item) for item in value]
        if not all(math.isfinite(item) for item in values):
            raise ValueError("linear_c_range values must be finite")
        if values[1] < values[0] or values[2] < 1 or values[2] != int(values[2]):
            raise ValueError("linear_c_range must be [start, stop, positive count]")
        return value

    _validate_c_range = field_validator("linear_c_range")(_valid_c_range)


class SegmentationConfig(StrictModel):
    """Segmentation probe settings."""

    enabled: StrictBool = False
    head: Literal["fpn", "dpt"] = "fpn"
    layers: list[StrictStr] = Field(default_factory=list)
    learning_rate: StrictFloat = Field(default=1e-3, gt=0)
    epochs: StrictInt = Field(default=10, gt=0)
    batch_size: StrictInt = Field(default=64, gt=0)
    temporal_pool: Literal["mean", "max"] = "mean"
    scheduler: Literal["cosine", "none"] = "cosine"
    cache_features: StrictBool = True
    cache_dtype: Literal["float16", "float32"] = "float16"


class RuntimeConfig(StrictModel):
    """Execution settings shared by image evaluation methods."""

    device: StrictStr = "auto"
    batch_size: StrictInt = Field(default=64, gt=0)
    workers: StrictInt = Field(default=4, ge=0)
    seed: StrictInt = 0


class OutputConfig(StrictModel):
    """Result storage settings."""

    directory: StrictStr = "results"
    resume: StrictBool = False


class RunConfig(StrictModel):
    """Complete validated configuration for a core image benchmark."""

    schema_version: Literal[1] = 1
    model: ModelConfig
    datasets: list[StrictStr] = Field(min_length=1)
    input: InputConfig = Field(default_factory=InputConfig)
    classification: ClassificationConfig = Field(default_factory=ClassificationConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @field_validator("datasets")
    @staticmethod
    def _nonempty_datasets(value: list[StrictStr]) -> list[StrictStr]:
        if any(not name for name in value):
            raise ValueError("datasets must contain non-empty names")
        return value

    def model_dump_yaml(self) -> dict:
        """Return a YAML-serializable resolved configuration."""
        return self.model_dump(mode="json")


def load_yaml(path: str | pathlib.Path) -> dict:
    """Load one YAML mapping, rejecting duplicate keys and unsafe tags."""
    with pathlib.Path(path).open(encoding="utf-8") as file:
        value = yaml.load(file, Loader=_UniqueKeyLoader)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: top level must be a YAML mapping")
    return value


def load_run_config(path: str | pathlib.Path) -> RunConfig:
    """Load and strictly validate a core image benchmark configuration."""
    return RunConfig.model_validate(load_yaml(path), strict=True)


def validate_run_config(value: dict) -> RunConfig:
    """Validate a mapping after explicit CLI overrides are applied."""
    return RunConfig.model_validate(value, strict=True)
