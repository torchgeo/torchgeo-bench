# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Flops command handler."""

import argparse

from omegaconf import OmegaConf

from .. import commands
from ._common import compose, setup_logging


def flops(args: argparse.Namespace) -> None:
    """Run the compute-cost pipeline."""
    cfg = compose(args, config_name='flops_config', default_model=None)
    if args.print_config:
        print(OmegaConf.to_yaml(cfg), end='')
        return
    setup_logging(verbose=True)
    commands.run_flops(cfg)
