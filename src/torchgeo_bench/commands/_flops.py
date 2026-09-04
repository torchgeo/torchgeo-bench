"""Flops command handler."""

import argparse

from torchgeo_bench.commands._common import compose, setup_logging


def flops(args: argparse.Namespace) -> None:
    """Run the compute-cost pipeline."""
    cfg = compose(args, config_name="flops_config", default_model=None)
    if args.print_config:
        from omegaconf import OmegaConf

        print(OmegaConf.to_yaml(cfg), end="")
        return
    setup_logging(verbose=True)
    from torchgeo_bench.commands._flops_runtime import run as run_flops

    run_flops(cfg)
