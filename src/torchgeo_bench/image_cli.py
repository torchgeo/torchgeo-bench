# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Discoverable CLI for the core image benchmark."""

import argparse
import pathlib
import sys
from typing import Any

_DATASETS = (
    'm-eurosat',
    'm-forestnet',
    'm-so2sat',
    'm-pv4ger',
    'm-brick-kiln',
    'm-bigearthnet',
    'benv2',
    'treesatai',
    'so2sat',
    'forestnet',
    'caffe',
    'burn_scars',
    'cloudsen12',
    'dynamic_earthnet',
    'flair2',
    'fotw',
    'kuro_siwo',
    'pastis',
    'spacenet2',
    'spacenet7',
    'eurosat',
    'eurosat-spatial',
    'resisc45',
)
_SEGMENTATION_DATASETS = frozenset(
    {
        'caffe',
        'burn_scars',
        'cloudsen12',
        'dynamic_earthnet',
        'flair2',
        'fotw',
        'kuro_siwo',
        'pastis',
        'spacenet2',
        'spacenet7',
    }
)


def _model_names() -> list[str]:
    """Return preset names from packaged YAML files without importing models."""
    root = pathlib.Path(__file__).parent / 'conf' / 'model'
    return sorted(
        path.relative_to(root).with_suffix('').as_posix()
        for path in root.rglob('*.yaml')
    )


def _model_detail(name: str) -> str:
    """Return the packaged model preset for a catalog detail request."""
    path = pathlib.Path(__file__).parent / 'conf' / 'model' / f'{name}.yaml'
    return path.read_text(encoding='utf-8')


def _dataset_detail(name: str) -> str:
    """Return lightweight metadata for a dataset catalog detail request."""
    task = 'segmentation' if name in _SEGMENTATION_DATASETS else 'classification'
    return f'name: {name}\ntask: {task}\n'


def _parser() -> argparse.ArgumentParser:
    """Build the CLI parser without importing numerical dependencies."""
    parser = argparse.ArgumentParser(prog='torchgeo-bench')
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser(
        'run', help='Run image benchmarks', argument_default=argparse.SUPPRESS
    )
    run.add_argument('--config', type=pathlib.Path, help='YAML configuration file')
    run.add_argument('--model', help='Model preset name')
    run.add_argument(
        '--dataset', action='append', dest='datasets', help='Dataset (repeatable)'
    )
    run.add_argument('--device')
    run.add_argument('--batch-size', type=int)
    run.add_argument('--workers', type=int)
    run.add_argument('--seed', type=int)
    run.add_argument('--bands')
    run.add_argument('--image-size', type=_image_size, metavar='PX|none')
    run.add_argument('--normalization', choices=('dataset', 'model', 'minmax', 'none'))
    run.add_argument('--partition')
    run.add_argument('--time-steps', type=int)
    run.add_argument('--methods', nargs='+', choices=('knn', 'linear'))
    run.add_argument('--knn-k', type=int)
    run.add_argument('--knn-device')
    run.add_argument('--bootstrap-samples', type=int)
    run.add_argument('--refit-train-val', action=argparse.BooleanOptionalAction)
    run.add_argument('--temp-scale', action=argparse.BooleanOptionalAction)
    run.add_argument('--resume', action=argparse.BooleanOptionalAction)
    run.add_argument('--verbose', action=argparse.BooleanOptionalAction)
    run.add_argument('--dry-run', action='store_true')
    run.add_argument(
        '--config-help', action='store_true', help='Print the JSON schema and exit'
    )
    for name, help_text in (
        ('models', 'List model presets or show one preset'),
        ('datasets', 'List datasets or show one dataset'),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument('name', nargs='?')
    return parser


def _set(overrides: dict[str, Any], section: str, key: str, value: Any) -> None:
    """Set one explicit CLI override in a nested mapping."""
    if '.' in section:
        parent, child = section.split('.', maxsplit=1)
        overrides.setdefault(parent, {}).setdefault(child, {})[key] = value
    else:
        overrides.setdefault(section, {})[key] = value


def _image_size(value: str) -> int | None:
    """Parse a positive image size or the explicit ``none`` value."""
    if value == 'none':
        return None
    size = int(value)
    if size <= 0:
        raise argparse.ArgumentTypeError('image size must be positive or none')
    return size


def _run(args: argparse.Namespace) -> None:
    """Validate and execute one image benchmark."""
    from torchgeo_bench.commands import _image

    _image.run(args, _model_names(), _DATASETS)


def main(argv: list[str] | None = None) -> None:
    """Run the image benchmark CLI."""
    args = _parser().parse_args(sys.argv[1:] if argv is None else argv)
    if args.command == 'run':
        try:
            _run(args)
        except ValueError as error:
            print(f'error: {error}', file=sys.stderr)
            raise SystemExit(2) from error
    elif args.command == 'models':
        names = _model_names()
        if args.name is None:
            print('\n'.join(names))
        elif args.name not in names:
            raise SystemExit(f'unknown model {args.name!r}')
        else:
            print(_model_detail(args.name), end='')
    elif args.command == 'datasets':
        if args.name is None:
            print('\n'.join(_DATASETS))
        elif args.name not in _DATASETS:
            raise SystemExit(f'unknown dataset {args.name!r}')
        else:
            print(_dataset_detail(args.name), end='')
    else:
        raise SystemExit(f'{args.command} is not implemented by the image CLI yet')


if __name__ == '__main__':
    main()
