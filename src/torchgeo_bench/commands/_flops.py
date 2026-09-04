"""Flops command handler."""

import argparse

from torchgeo_bench.cli import _compose, _setup_logging


def flops(args: argparse.Namespace) -> None:
    """Run the compute-cost pipeline."""
    cfg = _compose(args, config_name='flops_config', default_model=None)
    if args.print_config:
        from omegaconf import OmegaConf

        print(OmegaConf.to_yaml(cfg), end='')
        return
    _setup_logging(verbose=True)
    from torchgeo_bench.flops_pipeline import main as run_flops

    run_flops(cfg)
