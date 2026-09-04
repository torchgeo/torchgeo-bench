# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Tests for the strict core image configuration schema."""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from torchgeo_bench.config_schema import RunConfig, load_run_config, validate_run_config


def valid_config() -> dict:
    """Return the smallest valid image benchmark configuration."""
    return {"model": {"name": "timm/resnet50"}, "datasets": ["m-eurosat"]}


def test_defaults_and_modern_shape() -> None:
    config = validate_run_config(valid_config())
    assert config.runtime.device == "cuda:0"
    assert config.input.normalization == "dataset"
    assert config.classification.linear.c_count == 40
    assert config.output.directory == "results/models"


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("runtime", "batch_size", "64"),
        ("runtime", "workers", True),
        ("output", "resume", "false"),
        ("model", "name", 50),
        ("classification", "bootstrap_samples", 1.0),
    ],
)
def test_wrong_types_are_rejected(section: str, field: str, value: object) -> None:
    config = valid_config()
    config[section] = {field: value}
    with pytest.raises(ValidationError):
        validate_run_config(config)


def test_false_and_null_values_are_preserved() -> None:
    config = valid_config()
    config["output"] = {"directory": "results", "file": None, "resume": False}
    config["input"] = {"image_size": None, "normalization": "none"}
    parsed = validate_run_config(config)
    assert parsed.output.resume is False
    assert parsed.output.file is None
    assert parsed.input.image_size is None


def test_unknown_nested_field_is_rejected() -> None:
    config = valid_config()
    config["runtime"] = {"batch_size": 8, "batc_size": 4}
    with pytest.raises(ValidationError, match="batc_size"):
        validate_run_config(config)


def test_empty_or_duplicate_selections_are_rejected() -> None:
    for datasets in ([], [""], ["x", "x"]):
        config = valid_config()
        config["datasets"] = datasets
        with pytest.raises(ValidationError):
            validate_run_config(config)
    for bands in (" ", [], ["red", ""]):
        config = valid_config()
        config["input"] = {"bands": bands}
        with pytest.raises(ValidationError):
            validate_run_config(config)
    config = valid_config()
    config["classification"] = {"methods": ["knn", "knn"]}
    with pytest.raises(ValidationError):
        validate_run_config(config)


def test_invalid_ranges_and_calibration_are_rejected() -> None:
    config = valid_config()
    config["classification"] = {"linear": {"c_log10_start": 4.0, "c_log10_stop": -6.0}}
    with pytest.raises(ValidationError):
        validate_run_config(config)
    config["classification"] = {
        "calibration": {"temp_scale": True},
        "linear": {"refit_train_val": True},
    }
    with pytest.raises(ValidationError, match="refit_train_val"):
        validate_run_config(config)
    config["classification"] = {"linear": {"c_log10_start": float("inf")}}
    with pytest.raises(ValidationError, match="finite"):
        validate_run_config(config)


def test_invalid_device_is_rejected() -> None:
    config = valid_config()
    config["runtime"] = {"device": "gpu:0"}
    with pytest.raises(ValidationError, match="device"):
        validate_run_config(config)


def test_yaml_loads_bare_exponent_as_float(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "model:\n  name: rcf\ndatasets: [m-eurosat]\nsegmentation:\n  learning_rate: 1e-3\n",
        encoding="utf-8",
    )
    assert load_run_config(path).segmentation.learning_rate == 1e-3


def test_duplicate_yaml_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text("model: {name: rcf}\nmodel: {name: other}\n", encoding="utf-8")
    with pytest.raises(yaml.constructor.ConstructorError, match="duplicate key"):
        load_run_config(path)


def test_unhashable_yaml_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "unhashable.yaml"
    path.write_text("[a, b]: value\n", encoding="utf-8")
    with pytest.raises(yaml.constructor.ConstructorError, match="unhashable key"):
        load_run_config(path)


def test_non_mapping_yaml_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "scalar.yaml"
    path.write_text("null\n", encoding="utf-8")
    with pytest.raises(ValueError, match="top level"):
        load_run_config(path)


def test_schema_rejects_bool_schema_version() -> None:
    config = valid_config()
    config["schema_version"] = True
    with pytest.raises(ValidationError):
        validate_run_config(config)


def test_schema_does_not_import_ml_frameworks() -> None:
    code = (
        "import sys; import torchgeo_bench.config_schema; "
        "print([name for name in ('torch', 'torchgeo', 'pandas', 'numpy') if name in sys.modules])"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"


def test_round_trip_dump_is_valid() -> None:
    config = RunConfig.model_validate(valid_config(), strict=True)
    restored = RunConfig.model_validate(config.model_dump(mode="json"), strict=True)
    assert restored == config
    assert config.model_dump_yaml()["model"]["name"] == "timm/resnet50"
