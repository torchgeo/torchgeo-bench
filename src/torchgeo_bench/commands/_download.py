# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Download command handler."""

import argparse
from pathlib import Path
from typing import Any

from .. import commands
from ._common import setup_logging


def download(args: argparse.Namespace) -> None:
    """Download named datasets or one legacy dataset collection."""
    setup_logging(verbose=True)
    targets = list(args.target)
    collections = {'geobench_v1', 'geobench_v2', 'eurosat', 'resisc45'}
    if any(target in collections for target in targets):
        if len(targets) != 1:
            raise SystemExit(
                'error: legacy collection targets cannot be mixed with dataset names'
            )
        target = targets[0]
    else:
        if args.datasets is not None:
            raise SystemExit(
                'error: --datasets is only supported for legacy GeoBench targets'
            )
        try:
            commands.download_module.download_datasets(targets, Path(args.output_dir))
        except ValueError as err:
            raise SystemExit(f'error: {err}') from err
        return
    names = None
    if args.datasets is not None:
        names = [name.strip() for name in args.datasets.split(',') if name.strip()]
        if not names:
            raise SystemExit('error: --datasets must contain at least one dataset name')
    output_dir = Path(args.output_dir)
    _download_legacy(commands.download_module, target, output_dir, names)


def _download_legacy(
    module: Any, target: str, output_dir: Path, names: list[str] | None
) -> None:
    """Dispatch one legacy collection target."""
    if target == 'geobench_v1':
        module.download_geobench_v1(output_dir, datasets=names)
        return
    if target == 'geobench_v2':
        module.download_geobench_v2(output_dir, datasets=names)
        return
    if names is not None:
        raise SystemExit('error: --datasets is only supported for GeoBench downloads')
    if target == 'eurosat':
        module.download_eurosat(output_dir)
        return
    module.download_resisc45(output_dir)
