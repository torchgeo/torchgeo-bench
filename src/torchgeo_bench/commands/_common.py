# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Lightweight helpers shared by command handlers."""

import argparse
import logging
from typing import Any


def setup_logging(verbose: bool = False) -> None:
    """Configure command logging without importing benchmark modules."""
    from rich.logging import RichHandler

    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format='%(message)s',
        datefmt='[%X]',
        handlers=[RichHandler(rich_tracebacks=True, markup=True)],
    )


def flag_overrides(args: argparse.Namespace) -> list[str]:
    """Translate convenience flags into config overrides."""
    overrides: list[str] = []
    for attr, key in [
        ('model', 'model'),
        ('device', 'device'),
        ('output', 'output'),
        ('seed', 'seed'),
        ('partition', 'dataset.partition'),
        ('batch_size', 'dataset.batch_size'),
        ('image_size', 'dataset.image_size'),
        ('normalization', 'dataset.normalization'),
        ('bootstrap', 'eval.bootstrap'),
    ]:
        value = getattr(args, attr, None)
        if value is not None:
            overrides.append(f'{key}={value}')
    datasets = getattr(args, 'datasets', None)
    if datasets is not None:
        overrides.append(
            f'dataset.names={"all" if datasets == "all" else f"[{datasets}]"}'
        )
    bands = getattr(args, 'bands', None)
    if bands is not None:
        overrides.append(
            f'dataset.bands={bands if bands in ("rgb", "all") else f"[{bands}]"}'
        )
    if getattr(args, 'resume', False):
        overrides.append('resume=true')
    if getattr(args, 'skip_linear', False):
        overrides.append('eval.skip_linear=true')
    if getattr(args, 'verbose', False):
        overrides.append('verbose=true')
    return overrides


def compose(
    args: argparse.Namespace, *, config_name: str, default_model: str | None
) -> Any:
    """Compose a config from positional overrides and convenience flags."""
    from omegaconf.errors import OmegaConfBaseException

    from torchgeo_bench.config import compose_config

    try:
        return compose_config(
            [*args.overrides, *flag_overrides(args)],
            config_name=config_name,
            default_model=default_model,
        )
    except (OmegaConfBaseException, ValueError) as err:
        raise SystemExit(f'error: bad config override: {err}') from err
