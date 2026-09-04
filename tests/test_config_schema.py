# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Tests for the core image configuration schema."""

import sys
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from torchgeo_bench.config_schema import RunConfig, load_run_config, validate_run_config


def valid_config() -> dict:
    """Return the smallest valid image benchmark configuration."""
    return {"model": {"preset": "timm/resnet50"}, "datasets": ["m-eurosat"]}


def test_defaults_and_strict_types() -> None:
    config = validate_run_config(valid_config())

    assert config.runtime.batch_size == 64
    assert config.input.normalization == "bandspec_zscore"
    assert config.classification.merge_train_val is True
    assert isinstance(config.runtime.batch_size, int)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("runtime.batch_size", "64"),
        ("runtime.workers", True),
        ("output.resume", "false"),
        ("model.preset", 50),
    ],
)
def test_wrong_types_are_rejected(path: str, value: object) -> None:
    section, field = path.split(".")
    config = valid_config()
    config[section] = {field: value}

    with pytest.raises(ValidationError):
        validate_run_config(config)


def test_false_and_null_values_are_preserved() -> None:
    config = valid_config()
    config["output"] = {"directory": "results", "resume": False}
    config["input"] = {"image_size": None, "normalization": "identity"}

    parsed = validate_run_config(config)

    assert parsed.output.resume is False
    assert parsed.input.image_size is None
    assert parsed.input.normalization == "identity"


def test_unknown_nested_field_is_rejected() -> None:
    config = valid_config()
    config["runtime"] = {"batch_size": 8, "batc_size": 4}

    with pytest.raises(ValidationError, match="batc_size"):
        validate_run_config(config)


def test_empty_selections_are_rejected() -> None:
    with pytest.raises(ValidationError):
        validate_run_config({"model": {"preset": "x"}, "datasets": []})
    with pytest.raises(ValidationError):
        validate_run_config({"model": {"preset": "x"}, "datasets": [""]})
    with pytest.raises(ValidationError):
        validate_run_config({"model": {"preset": "x"}, "datasets": ["x"], "input": {"bands": []}})


def test_invalid_ranges_are_rejected() -> None:
    config = valid_config()
    config["classification"] = {"linear_c_range": [4, -6, 40]}

    with pytest.raises(ValidationError):
        validate_run_config(config)


def test_yaml_loads_bare_exponent_as_float(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "model:\n  preset: rcf\ndatasets: [m-eurosat]\nsegmentation:\n  learning_rate: 1e-3\n",
        encoding="utf-8",
    )

    config = load_run_config(path)

    assert config.segmentation.learning_rate == 1e-3


def test_duplicate_yaml_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text("model: {preset: rcf}\nmodel: {preset: other}\n", encoding="utf-8")

    with pytest.raises(yaml.constructor.ConstructorError, match="duplicate key"):
        load_run_config(path)


def test_schema_does_not_import_ml_frameworks() -> None:
    assert not {"torch", "torchgeo", "pandas", "numpy"} & set(sys.modules)


def test_round_trip_dump_is_valid() -> None:
    config = RunConfig.model_validate(valid_config(), strict=True)
    restored = RunConfig.model_validate(config.model_dump(mode="json"), strict=True)

    assert restored == config
