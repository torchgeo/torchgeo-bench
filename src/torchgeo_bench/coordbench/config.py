"""Strict, lightweight configuration for coordinate encoder evaluation."""

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, StrictBool, StrictFloat, StrictInt, StrictStr, field_validator

from torchgeo_bench.config.presets import ModelPreset, load_model_preset
from torchgeo_bench.config.schema import (
    Device,
    KnnDevice,
    Methods,
    ModelConfig,
    OutputPath,
    SchemaVersion,
    StrictModel,
    default_methods,
    load_yaml,
)
from torchgeo_bench.coordbench.catalog import validate_datasets


class CoordRuntimeConfig(StrictModel):
    """Coordinate encoder and linear-probe execution settings."""

    device: Device = "cpu"
    seed: StrictInt = Field(default=0, ge=0, le=2**64 - 1)


class CoordOutputConfig(StrictModel):
    """Coordinate result CSV and resume settings."""

    file: OutputPath = "results/coordbench_results.csv"
    resume: StrictBool = False


class CoordEvaluationConfig(StrictModel):
    """Random/spatial cross-validation and coordinate probe settings."""

    methods: Methods = Field(default_factory=default_methods, min_length=1)
    split: Literal["random", "spatial", "both"] = "random"
    folds: StrictInt = Field(default=5, ge=2)
    cell_deg: StrictFloat = Field(default=10.0, gt=0)
    knn_k: StrictInt = Field(default=5, gt=0)
    knn_device: KnnDevice = "cpu"


class CoordConfig(StrictModel):
    """Complete coordinate benchmark configuration, independent of image settings."""

    schema_version: SchemaVersion = 1
    model: ModelConfig
    datasets: list[StrictStr] = Field(default_factory=lambda: ["all"], min_length=1)
    evaluation: CoordEvaluationConfig = Field(default_factory=CoordEvaluationConfig)
    runtime: CoordRuntimeConfig = Field(default_factory=CoordRuntimeConfig)
    output: CoordOutputConfig = Field(default_factory=CoordOutputConfig)

    @field_validator("datasets")
    @classmethod
    def validate_datasets(cls, value: list[str]) -> list[str]:
        """Require distinct, non-blank names and an exclusive all selection."""
        return validate_datasets(value)

    def model_dump_yaml(self) -> dict[str, Any]:
        """Return a reusable YAML mapping with explicit effective defaults."""
        return self.model_dump(mode="json")


def load_coord_config(path: str | Path) -> CoordConfig:
    """Load one strictly validated coordinate YAML configuration."""
    return CoordConfig.model_validate(load_yaml(path))


def resolve_coord_preset(config: CoordConfig) -> ModelPreset:
    """Resolve a coordinate preset without importing encoder implementations."""
    preset = load_model_preset(config.model, seed=config.runtime.seed)
    if config.model.target is None and preset.track != "coord":
        raise ValueError(f"{config.model.name!r} is an image model; use 'run'")
    return preset
