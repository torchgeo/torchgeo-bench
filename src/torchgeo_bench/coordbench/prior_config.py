"""Strict, lightweight configuration for supervised coordinate priors."""

from pathlib import Path
from typing import Any, Literal

from pydantic import Field, StrictBool, StrictFloat, StrictInt, StrictStr, field_validator

from torchgeo_bench.config.schema import OutputPath, SchemaVersion, StrictModel, load_yaml
from torchgeo_bench.coordbench.catalog import validate_datasets

type PriorName = Literal["uniform", "frequency", "grid", "nearest", "kde"]

PRIOR_NAMES: tuple[PriorName, ...] = ("uniform", "frequency", "grid", "nearest", "kde")


class CoordPriorEvaluationConfig(StrictModel):
    """Prior estimators and random/spatial holdout settings."""

    methods: list[PriorName] = Field(default_factory=lambda: list(PRIOR_NAMES), min_length=1)
    split: Literal["random", "spatial", "both"] = "random"
    folds: StrictInt = Field(default=5, ge=2)
    cell_deg: StrictFloat = Field(default=10.0, gt=0)
    grid_cell_size: StrictFloat = Field(default=10.0, gt=0)
    smoothing: StrictFloat = Field(default=0.0, ge=0)
    nearest_k: StrictInt = Field(default=5, gt=0)
    nearest_weights: Literal["uniform", "distance"] = "uniform"
    kde_bandwidth: StrictFloat = Field(default=10.0, gt=0)

    @field_validator("methods")
    @classmethod
    def validate_methods(cls, value: list[PriorName]) -> list[PriorName]:
        """Reject repeated prior selections."""
        if len(set(value)) != len(value):
            raise ValueError("methods must not contain duplicates")
        return value


class CoordPriorRuntimeConfig(StrictModel):
    """Seed for CPU-only coordinate priors."""

    seed: StrictInt = Field(default=0, ge=0, le=2**64 - 1)


class CoordPriorOutputConfig(StrictModel):
    """Separate prior result CSV and resume settings."""

    file: OutputPath = "results/coordbench_priors.csv"
    resume: StrictBool = False


class CoordPriorConfig(StrictModel):
    """Supervised prior configuration, independent of frozen encoder settings."""

    schema_version: SchemaVersion = 1
    datasets: list[StrictStr] = Field(default_factory=lambda: ["all"], min_length=1)
    evaluation: CoordPriorEvaluationConfig = Field(default_factory=CoordPriorEvaluationConfig)
    runtime: CoordPriorRuntimeConfig = Field(default_factory=CoordPriorRuntimeConfig)
    output: CoordPriorOutputConfig = Field(default_factory=CoordPriorOutputConfig)

    @field_validator("datasets")
    @classmethod
    def validate_datasets(cls, value: list[str]) -> list[str]:
        """Require known, distinct names or the exclusive all selector."""
        return validate_datasets(value)

    def model_dump_yaml(self) -> dict[str, Any]:
        """Return reusable YAML with explicit effective defaults."""
        return self.model_dump(mode="json")


def load_coord_prior_config(path: str | Path) -> CoordPriorConfig:
    """Load strictly validated prior YAML without importing runtime dependencies."""
    return CoordPriorConfig.model_validate(load_yaml(path))
