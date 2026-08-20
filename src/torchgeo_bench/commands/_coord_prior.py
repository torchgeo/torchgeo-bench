"""Typed YAML and explicit flag boundary for supervised coordinate priors."""

import argparse

import yaml

from torchgeo_bench.coordbench.prior_config import CoordPriorConfig

from ._config import FlagOverride, load_config_or_exit, load_from_flags

_FLAG_OVERRIDES = (
    FlagOverride("datasets", ("datasets",)),
    FlagOverride("methods", ("evaluation", "methods")),
    FlagOverride("split", ("evaluation", "split")),
    FlagOverride("folds", ("evaluation", "folds")),
    FlagOverride("cell_deg", ("evaluation", "cell_deg")),
    FlagOverride("grid_cell_size", ("evaluation", "grid_cell_size")),
    FlagOverride("smoothing", ("evaluation", "smoothing")),
    FlagOverride("nearest_k", ("evaluation", "nearest_k")),
    FlagOverride("nearest_weights", ("evaluation", "nearest_weights")),
    FlagOverride("kde_bandwidth", ("evaluation", "kde_bandwidth")),
    FlagOverride("seed", ("runtime", "seed")),
    FlagOverride("resume", ("output", "resume")),
    FlagOverride("output", ("output", "file")),
)


def load_config(args: argparse.Namespace) -> CoordPriorConfig:
    """Apply flags over YAML and validate before importing the runtime."""
    return load_from_flags(args, _FLAG_OVERRIDES, CoordPriorConfig.model_validate)


def run(args: argparse.Namespace) -> None:
    """Validate configuration, print a dry run, or execute CPU prior evaluation."""
    config = load_config_or_exit(args, load_config)
    if getattr(args, "dry_run", False):
        print(yaml.safe_dump(config.model_dump_yaml(), sort_keys=False), end="")
        return
    from torchgeo_bench.coordbench.prior_run import run_coordbench_priors

    run_coordbench_priors(config)


coord_prior = run
