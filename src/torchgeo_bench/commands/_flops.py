"""Validate synthetic compute measurements before importing the runtime."""

import argparse
import json
import logging
from typing import Any

import yaml

from .. import commands
from ..config_schema import _UniqueKeyLoader, load_yaml
from ..flops_config import FlopsConfig
from ..presets import merge_settings

_FLAG_FIELDS = {
    "model": ("model", "name"),
    "model_target": ("model", "target"),
    "model_kwargs": ("model", "kwargs"),
    "device": ("runtime", "device"),
    "seed": ("runtime", "seed"),
    "verbose": ("runtime", "verbose"),
    "band_source": ("input", "band_source"),
    "band_configs": ("input", "band_configs"),
    "image_size": ("input", "image_size"),
    "normalization": ("input", "normalization"),
    "probe_head": ("classification", "head"),
    "probe_num_classes": ("classification", "num_classes"),
    "seg_heads": ("segmentation", "heads"),
    "seg_band_configs": ("segmentation", "band_configs"),
    "seg_num_classes": ("segmentation", "num_classes"),
    "timing_batch_size": ("timing", "batch_size"),
    "n_warmup": ("timing", "n_warmup"),
    "n_measure": ("timing", "n_measure"),
    "output": ("output", "file"),
    "resume": ("output", "resume"),
}


def _flag_values(args: argparse.Namespace) -> dict[str, Any]:
    """Translate only explicitly supplied flags into validated sections."""
    values: dict[str, Any] = {}
    for flag, (section, field) in _FLAG_FIELDS.items():
        if hasattr(args, flag):
            value = getattr(args, flag)
            if flag == "model_kwargs":
                value = yaml.load(value, Loader=_UniqueKeyLoader)
            values.setdefault(section, {})[field] = value
    probe = {
        field: getattr(args, flag)
        for flag, field in (("seg_layers", "layers"), ("temporal_pool", "temporal_pool"))
        if hasattr(args, flag)
    }
    if probe:
        values.setdefault("segmentation", {})["probe"] = probe
    return values


def load_config(args: argparse.Namespace) -> FlopsConfig:
    """Load strict YAML, apply supplied flags, and validate preset resolution."""
    path = getattr(args, "config", None)
    base = load_yaml(path) if path else {}
    if hasattr(args, "model"):
        base.pop("model", None)
    config = FlopsConfig.model_validate(merge_settings(base, _flag_values(args)))
    config.resolve()
    return config


def run(args: argparse.Namespace) -> None:
    """Execute the explicit command; report expected input errors without a traceback."""
    if getattr(args, "config_help", False):
        print(json.dumps(FlopsConfig.model_json_schema(), indent=2))
        return
    try:
        config = load_config(args)
    except (OSError, ValueError, yaml.YAMLError) as error:  # allow-except: CLI configuration errors
        raise SystemExit(f"error: {error}") from error
    if getattr(args, "dry_run", False):
        print(yaml.safe_dump(config.model_dump_yaml(), sort_keys=False), end="")
        return
    logging.basicConfig(
        level=logging.INFO if config.runtime.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    commands.run_flops(config)


flops = run
