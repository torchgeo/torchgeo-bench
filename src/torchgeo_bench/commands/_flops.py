"""Validate synthetic compute measurements before importing the runtime."""

import argparse
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
    config = FlopsConfig.model_validate(merge_settings(base, _flag_values(args)))
    config.resolve()
    return config


def run(args: argparse.Namespace) -> None:
    """Execute the explicit command; report expected input errors without a traceback."""
    if getattr(args, "config_help", False):
        print(yaml.safe_dump(FlopsConfig.model_json_schema(), sort_keys=False), end="")
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


_LEGACY_FIELDS = {
    **{name: name for name in _FLAG_FIELDS},
    "seg_head_types": "seg_heads",
    "eval.segmentation.layers": "seg_layers",
    "eval.segmentation.temporal_pool": "temporal_pool",
}


def _legacy_namespace(args: argparse.Namespace) -> argparse.Namespace:
    """Translate the retiring parser's finite FLOPs vocabulary."""
    flags: dict[str, Any] = {}
    kwargs: dict[str, Any] = {}
    for override in getattr(args, "overrides", []):
        key, separator, raw = override.partition("=")
        if separator and key in {
            "model.features",
            "model.kernel_size",
            "model.mode",
            "model.stats_mode",
            "model.seed",
        }:
            kwargs[key.removeprefix("model.")] = yaml.safe_load(raw)
            continue
        if not separator or key not in _LEGACY_FIELDS:
            raise SystemExit(
                f"error: unsupported FLOPs override {override!r}; use explicit flags or --config YAML"
            )
        flags[_LEGACY_FIELDS[key]] = yaml.safe_load(raw)
    if kwargs:
        flags["model_kwargs"] = yaml.safe_dump(kwargs)
    flags.update(
        {
            key: value
            for key, value in vars(args).items()
            if key in _FLAG_FIELDS and value is not None
        }
    )
    if "model" not in flags:
        raise SystemExit("error: No model selected; pass --model/-m or use --config YAML")
    normalization = flags.get("normalization")
    aliases = {"bandspec_zscore": "dataset", "model_native": "model", "identity": "none"}
    if isinstance(normalization, str) and normalization in aliases:
        flags["normalization"] = aliases[normalization]
    flags["dry_run"] = getattr(args, "print_config", False)
    return argparse.Namespace(**flags)


def flops(args: argparse.Namespace) -> None:
    """Keep legacy command callers working until canonical CLI registration."""
    try:
        converted = _legacy_namespace(args)
    except yaml.YAMLError as error:  # allow-except: legacy YAML input errors
        raise SystemExit(f"error: {error}") from error
    run(converted)
