# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Implementation of the image benchmark command."""

import argparse
import json
from typing import Any

import yaml

from .. import commands
from ..config_schema import RunConfig, load_yaml, validate_run_config
from ..presets import load_model_preset


def _set(overrides: dict[str, Any], section: str, key: str, value: Any) -> None:
    """Set one explicit CLI override in a nested mapping."""
    if "." in section:
        parent, child = section.split(".", maxsplit=1)
        overrides.setdefault(parent, {}).setdefault(child, {})[key] = value
    else:
        overrides.setdefault(section, {})[key] = value


def _run_mapping(args: argparse.Namespace) -> dict[str, Any]:  # noqa: C901 - flat flag table
    """Translate supplied run flags into the schema's mapping."""
    overrides: dict[str, Any] = {}
    if getattr(args, "model", None) is not None:
        overrides["model"] = {"name": args.model}
    if getattr(args, "datasets", None) is not None:
        overrides["datasets"] = args.datasets
    for name in ("device", "batch_size", "workers", "seed", "verbose"):
        if hasattr(args, name):
            _set(overrides, "runtime", name, getattr(args, name))
    for name in (
        "bands",
        "image_size",
        "interpolation",
        "normalization",
        "partition",
        "time_steps",
    ):
        if hasattr(args, name):
            value = getattr(args, name)
            if name == "bands" and value not in {"rgb", "all"}:
                value = [band.strip() for band in value.split(",")]
            _set(overrides, "input", name, value)
    for name in ("methods", "knn_k", "knn_device", "bootstrap_samples"):
        if hasattr(args, name):
            _set(overrides, "classification", name, getattr(args, name))
    for flag, section, key in (
        ("refit_train_val", "classification.linear", "refit_train_val"),
        ("temp_scale", "classification.calibration", "temp_scale"),
        ("resume", "output", "resume"),
        ("output", "output", "file"),
        ("results_dir", "output", "directory"),
    ):
        if hasattr(args, flag):
            _set(overrides, section, key, getattr(args, flag))
    return overrides


def _apply_overrides(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Apply known CLI fields with explicit section handling."""
    result = dict(base)
    for key, value in overrides.items():
        if key == "model":
            result[key] = value
            continue
        if key in {"model", "input", "classification", "runtime", "output"}:
            section = dict(result.get(key, {}))
            for nested_key, nested_value in value.items():
                if isinstance(nested_value, dict) and isinstance(section.get(nested_key), dict):
                    merged = dict(section[nested_key])
                    merged.update(nested_value)
                    section[nested_key] = merged
                else:
                    section[nested_key] = nested_value
            result[key] = section
        else:
            result[key] = value
    return result


def _load_run(
    args: argparse.Namespace, model_names: list[str], datasets: tuple[str, ...]
) -> RunConfig:
    """Load, validate, and return the selected image configuration."""
    config_path = getattr(args, "config", None)
    base = load_yaml(config_path) if config_path else {}
    values = _apply_overrides(base, _run_mapping(args))
    if getattr(args, "config_help", False):
        print(json.dumps(RunConfig.model_json_schema(), indent=2))
        raise SystemExit(0)
    config = validate_run_config(values)
    unknown_model = config.model.target is None and config.model.name not in model_names
    unknown_datasets = [name for name in config.datasets if name != "all" and name not in datasets]
    if unknown_model:
        load_model_preset(config.model)
    if unknown_datasets:
        raise ValueError(
            f"unknown model or dataset: model={config.model.name}, datasets={unknown_datasets}"
        )
    if isinstance(config.input.bands, str) and config.input.bands not in {"rgb", "all"}:
        raise ValueError("input.bands must be rgb, all, or a YAML list of band names")
    if load_model_preset(config.model).track != "image":
        raise ValueError(f"{config.model.name!r} is a coordinate encoder; use 'coord'")
    return config


def run(args: argparse.Namespace, model_names: list[str], datasets: tuple[str, ...]) -> None:
    """Validate and execute one image benchmark."""
    config = _load_run(args, model_names, datasets)
    if getattr(args, "dry_run", False):
        print(yaml.safe_dump(config.model_dump_yaml(), sort_keys=False), end="")
        return
    commands._image_runtime.run(config)
