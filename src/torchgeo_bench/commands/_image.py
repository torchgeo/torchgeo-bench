# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Implementation of the image benchmark command."""

import argparse
import json

import yaml

from .. import commands
from ..config_schema import RunConfig, validate_run_config
from ..presets import load_model_preset
from ._config import (
    FlagOverride,
    comma_separated_bands,
    load_config_or_exit,
    load_from_flags,
)

_FLAG_OVERRIDES = (
    FlagOverride("model", ("model", "name"), replace_roots=("model",)),
    FlagOverride("datasets", ("datasets",)),
    FlagOverride("device", ("runtime", "device")),
    FlagOverride("batch_size", ("runtime", "batch_size")),
    FlagOverride("workers", ("runtime", "workers")),
    FlagOverride("seed", ("runtime", "seed")),
    FlagOverride("verbose", ("runtime", "verbose")),
    FlagOverride("bands", ("input", "bands"), comma_separated_bands),
    FlagOverride("image_size", ("input", "image_size")),
    FlagOverride("interpolation", ("input", "interpolation")),
    FlagOverride("normalization", ("input", "normalization")),
    FlagOverride("partition", ("input", "partition")),
    FlagOverride("time_steps", ("input", "time_steps")),
    FlagOverride("methods", ("classification", "methods")),
    FlagOverride("knn_k", ("classification", "knn_k")),
    FlagOverride("knn_device", ("classification", "knn_device")),
    FlagOverride("bootstrap_samples", ("classification", "bootstrap_samples")),
    FlagOverride("refit_train_val", ("classification", "linear", "refit_train_val")),
    FlagOverride("temp_scale", ("classification", "calibration", "temp_scale")),
    FlagOverride("resume", ("output", "resume")),
    FlagOverride("output", ("output", "file")),
    FlagOverride("results_dir", ("output", "directory")),
)


def _load_run(args: argparse.Namespace, datasets: tuple[str, ...]) -> RunConfig:
    """Load, validate, and return the selected image configuration."""
    config = load_from_flags(args, _FLAG_OVERRIDES, validate_run_config)
    preset = load_model_preset(config.model)
    unknown_datasets = [name for name in config.datasets if name != "all" and name not in datasets]
    if unknown_datasets:
        raise ValueError(
            f"unknown model or dataset: model={config.model.name}, datasets={unknown_datasets}"
        )
    if isinstance(config.input.bands, str) and config.input.bands not in {"rgb", "all"}:
        raise ValueError("input.bands must be rgb, all, or a YAML list of band names")
    if preset.track != "image":
        raise ValueError(f"{config.model.name!r} is a coordinate encoder; use 'coord'")
    return config


def run(args: argparse.Namespace, datasets: tuple[str, ...]) -> None:
    """Validate and execute one image benchmark."""
    if getattr(args, "config_help", False):
        print(json.dumps(RunConfig.model_json_schema(), indent=2))
        raise SystemExit(0)
    config = load_config_or_exit(args, lambda value: _load_run(value, datasets))
    if getattr(args, "dry_run", False):
        print(yaml.safe_dump(config.model_dump_yaml(), sort_keys=False), end="")
        return
    commands._image_runtime.run(config)
