# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Run command handler."""

import argparse

from omegaconf import OmegaConf

from .. import commands
from ..config import list_model_configs, model_config_path
from ..datasets import list_datasets
from ._common import compose, setup_logging


def run(args: argparse.Namespace) -> None:
    """Run a benchmark or a lightweight run command query."""
    if args.list_models:
        print('\n'.join(list_model_configs()))
        return
    if args.list_datasets:
        print('\n'.join(list_datasets()))
        return
    if args.model_help is not None:
        try:
            print(model_config_path(args.model_help).read_text(), end='')
        except ValueError as err:
            raise SystemExit(f'error: {err}') from err
        return
    cfg = compose(args, config_name='config', default_model='rcf')
    if args.print_config:
        print(OmegaConf.to_yaml(cfg), end='')
        return
    setup_logging(bool(cfg.verbose))
    commands.run_benchmark(cfg)
