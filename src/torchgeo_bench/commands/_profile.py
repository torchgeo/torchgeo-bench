# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Standalone real-batch profile command."""

import argparse

import yaml

from .. import commands
from ..profile_config import resolve_profile_config
from .profile_arguments import load_profile_config


def profile(args: argparse.Namespace) -> None:
    """Profile one selected model and dataset."""
    try:
        config, _ = resolve_profile_config(load_profile_config(args))
    except (OSError, yaml.YAMLError, ValueError) as error:
        raise SystemExit(f"error: {error}") from error
    if getattr(args, "dry_run", False):
        print(yaml.safe_dump(config.model_dump_yaml(), sort_keys=False), end="")
        return
    commands._profile_runtime.run(config)
