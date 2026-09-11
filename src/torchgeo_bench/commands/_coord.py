"""Typed YAML and explicit flag boundary for coordinate evaluation."""

import argparse
from typing import Any

import yaml

from torchgeo_bench.config_schema import load_yaml
from torchgeo_bench.coordbench.config import CoordConfig, resolve_coord_preset
from torchgeo_bench.presets import merge_settings


def _coord_mapping(args: argparse.Namespace) -> dict[str, Any]:
    """Translate explicitly supplied flags into coordinate settings."""
    values: dict[str, Any] = {}
    if hasattr(args, "model"):
        values["model"] = {"name": args.model}
    if hasattr(args, "datasets"):
        values["datasets"] = args.datasets
    for section, names in (
        ("evaluation", ("methods", "split", "folds", "cell_deg", "knn_k", "knn_device")),
        ("runtime", ("device", "seed")),
        ("output", ("resume",)),
    ):
        for name in names:
            if hasattr(args, name):
                values.setdefault(section, {})[name] = getattr(args, name)
    if hasattr(args, "output"):
        values.setdefault("output", {})["file"] = args.output
    return values


def load_config(args: argparse.Namespace) -> CoordConfig:
    """Apply flag precedence and validate configuration before importing the runtime."""
    path = getattr(args, "config", None)
    values = load_yaml(path) if path is not None else {}
    overrides = _coord_mapping(args)
    if "model" in overrides:
        values.pop("model", None)
    config = CoordConfig.model_validate(merge_settings(values, overrides))
    resolve_coord_preset(config)
    return config


def run(args: argparse.Namespace) -> None:
    """Validate YAML/flags, print a dry run, or execute coordinate evaluation."""
    config = load_config(args)
    if getattr(args, "dry_run", False):
        print(yaml.safe_dump(config.model_dump_yaml(), sort_keys=False), end="")
        return
    from torchgeo_bench.coordbench.run import run_coordbench

    run_coordbench(config)
