"""Strict, lightweight configuration for coordinate encoder evaluation."""

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, StrictBool, StrictFloat, StrictInt, StrictStr, field_validator

from torchgeo_bench.config_schema import (
    ClassificationConfig,
    ModelConfig,
    RuntimeConfig,
    StrictModel,
    load_yaml,
)
from torchgeo_bench.coordbench.catalog import FAMILY_BENCHMARKS
from torchgeo_bench.presets import ModelPreset, load_model_preset


class CoordRuntimeConfig(StrictModel):
    """Coordinate encoder and linear-probe execution settings."""

    device: StrictStr = "cpu"
    seed: StrictInt = Field(default=0, ge=0, le=2**64 - 1)

    @field_validator("device")
    @classmethod
    def validate_device(cls, value: str) -> str:
        """Validate a Torch device without importing Torch."""
        return RuntimeConfig.validate_device(value)


class CoordOutputConfig(StrictModel):
    """Coordinate result CSV and resume settings."""

    file: StrictStr = "results/coordbench_results.csv"
    resume: StrictBool = False

    @field_validator("file")
    @classmethod
    def validate_file(cls, value: str) -> str:
        """Require a non-blank result path."""
        if not value.strip():
            raise ValueError("output.file must not be blank")
        return value


def _default_methods() -> list[Literal["knn", "linear"]]:
    """Return the supported default probe selection."""
    return ["knn", "linear"]


class CoordEvaluationConfig(StrictModel):
    """Random/spatial cross-validation and coordinate probe settings."""

    methods: list[Literal["knn", "linear"]] = Field(default_factory=_default_methods, min_length=1)
    split: Literal["random", "spatial", "both"] = "random"
    folds: StrictInt = Field(default=5, ge=2)
    cell_deg: StrictFloat = Field(default=10.0, gt=0)
    knn_k: StrictInt = Field(default=5, gt=0)
    knn_device: StrictStr = "cpu"

    @field_validator("methods")
    @classmethod
    def validate_methods(
        cls, value: list[Literal["knn", "linear"]]
    ) -> list[Literal["knn", "linear"]]:
        """Reject repeated method selections."""
        return ClassificationConfig.validate_methods(value)

    @field_validator("knn_device")
    @classmethod
    def validate_knn_device(cls, value: str) -> str:
        """Validate the KNN device without importing its backend."""
        ClassificationConfig.validate_knn_device(value)
        return value


class CoordConfig(StrictModel):
    """Complete coordinate benchmark configuration, independent of image settings."""

    schema_version: Literal[1] = 1
    model: ModelConfig
    datasets: list[StrictStr] = Field(default_factory=lambda: ["all"], min_length=1)
    evaluation: CoordEvaluationConfig = Field(default_factory=CoordEvaluationConfig)
    runtime: CoordRuntimeConfig = Field(default_factory=CoordRuntimeConfig)
    output: CoordOutputConfig = Field(default_factory=CoordOutputConfig)

    @field_validator("schema_version", mode="before")
    @classmethod
    def validate_version(cls, value: object) -> object:
        """Require integer schema versions without numeric coercion."""
        if type(value) is not int:
            raise ValueError("schema_version must be the integer 1")
        return value

    @field_validator("datasets")
    @classmethod
    def validate_datasets(cls, value: list[str]) -> list[str]:
        """Require distinct, non-blank names and an exclusive all selection."""
        if any(not name.strip() for name in value) or len(set(value)) != len(value):
            raise ValueError("datasets must contain distinct non-empty names")
        if "all" in value and len(value) != 1:
            raise ValueError("'all' cannot be combined with other datasets")
        known = {"all", *FAMILY_BENCHMARKS}
        known.update(name for names in FAMILY_BENCHMARKS.values() for name in names)
        unknown = sorted(set(value) - known)
        if unknown:
            raise ValueError(f"Unknown coordinate datasets: {unknown}")
        return value

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
