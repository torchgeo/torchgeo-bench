"""Run command handler."""

import argparse

from torchgeo_bench.commands._common import compose, setup_logging


def run(args: argparse.Namespace) -> None:
    """Run a benchmark or a lightweight run command query."""
    if args.list_models:
        from torchgeo_bench.config import list_model_configs

        print("\n".join(list_model_configs()))
        return
    if args.list_datasets:
        from torchgeo_bench.datasets import list_datasets

        print("\n".join(list_datasets()))
        return
    if args.model_help is not None:
        from torchgeo_bench.config import model_config_path

        try:
            print(model_config_path(args.model_help).read_text(), end="")
        except ValueError as err:
            raise SystemExit(f"error: {err}") from err
        return
    cfg = compose(args, config_name="config", default_model="rcf")
    if args.print_config:
        from omegaconf import OmegaConf

        print(OmegaConf.to_yaml(cfg), end="")
        return
    setup_logging(bool(cfg.verbose))
    from torchgeo_bench.commands._run_runtime import run as run_benchmark

    run_benchmark(cfg)
