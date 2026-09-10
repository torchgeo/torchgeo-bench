# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Discoverable CLI for the core image benchmark."""

import argparse
import pathlib
import sys
from collections.abc import Callable, Sequence

from . import commands
from .cli import add_profile_arguments

_DATASETS = (
    "m-eurosat",
    "m-forestnet",
    "m-so2sat",
    "m-pv4ger",
    "m-brick-kiln",
    "m-bigearthnet",
    "benv2",
    "treesatai",
    "so2sat",
    "forestnet",
    "caffe",
    "burn_scars",
    "cloudsen12",
    "dynamic_earthnet",
    "flair2",
    "fotw",
    "kuro_siwo",
    "pastis",
    "spacenet2",
    "spacenet7",
    "eurosat",
    "eurosat-spatial",
    "resisc45",
)
_SEGMENTATION_DATASETS = frozenset(
    {
        "caffe",
        "burn_scars",
        "cloudsen12",
        "dynamic_earthnet",
        "flair2",
        "fotw",
        "kuro_siwo",
        "pastis",
        "spacenet2",
        "spacenet7",
    }
)


def _model_names() -> list[str]:
    """Return preset names from packaged YAML files without importing models."""
    root = pathlib.Path(__file__).parent / "conf" / "model"
    return sorted(
        path.relative_to(root).with_suffix("").as_posix() for path in root.rglob("*.yaml")
    )


def _model_detail(name: str) -> str:
    """Return the packaged model preset for a catalog detail request."""
    path = pathlib.Path(__file__).parent / "conf" / "model" / f"{name}.yaml"
    return path.read_text(encoding="utf-8")


def _dataset_detail(name: str) -> str:
    """Return lightweight metadata for a dataset catalog detail request."""
    task = "segmentation" if name in _SEGMENTATION_DATASETS else "classification"
    return f"name: {name}\ntask: {task}\n"


def _parser() -> argparse.ArgumentParser:
    """Build the CLI parser without importing numerical dependencies."""
    parser = argparse.ArgumentParser(prog="torchgeo-bench")
    subcommands = parser.add_subparsers(dest="command", required=True)
    run = subcommands.add_parser(
        "run", help="Run image benchmarks", argument_default=argparse.SUPPRESS
    )
    run.add_argument("--config", type=pathlib.Path, help="YAML configuration file")
    run.add_argument("--model", help="Model preset name")
    run.add_argument("--dataset", action="append", dest="datasets", help="Dataset (repeatable)")
    run.add_argument("--device")
    run.add_argument("--batch-size", type=int)
    run.add_argument("--workers", type=int)
    run.add_argument("--seed", type=int)
    run.add_argument("--bands", help="rgb, all, or comma-separated band names")
    run.add_argument("--interpolation", choices=("area", "bilinear", "bicubic", "nearest"))
    run.add_argument("--image-size", type=_image_size, metavar="PX|none")
    run.add_argument("--normalization", choices=("dataset", "model", "minmax", "none"))
    run.add_argument("--partition")
    run.add_argument("--time-steps", type=int)
    run.add_argument("--methods", nargs="+", choices=("knn", "linear"))
    run.add_argument("--knn-k", type=int)
    run.add_argument("--knn-device")
    run.add_argument("--bootstrap-samples", type=int)
    run.add_argument("--refit-train-val", action=argparse.BooleanOptionalAction)
    run.add_argument("--temp-scale", action=argparse.BooleanOptionalAction)
    run.add_argument("--resume", action=argparse.BooleanOptionalAction)
    run.add_argument("--verbose", action=argparse.BooleanOptionalAction)
    run.add_argument(
        "--dry-run", action="store_true", help="Validate and print reusable YAML without running"
    )
    run.add_argument("--config-help", action="store_true", help="Print the JSON schema and exit")
    for name, help_text in (
        ("models", "List model presets or show one preset"),
        ("datasets", "List datasets or show one dataset"),
    ):
        command = subcommands.add_parser(name, help=help_text)
        command.add_argument("name", nargs="?")
    download = subcommands.add_parser("download", help="Download benchmark datasets")
    download.add_argument("target", nargs="+")
    download.add_argument("--output-dir", default="data")
    download.add_argument("--datasets")
    profile = subcommands.add_parser("profile", help="Measure one real inference batch")
    add_profile_arguments(profile)
    return parser


def _image_size(value: str) -> int | None:
    """Parse a positive image size or the explicit ``none`` value."""
    if value == "none":
        return None
    size = int(value)
    if size <= 0:
        raise argparse.ArgumentTypeError("image size must be positive or none")
    return size


def _run(args: argparse.Namespace) -> None:
    """Validate and execute one image benchmark."""
    commands._image.run(args, _model_names(), _DATASETS)


def _show_catalog(
    name: str | None, choices: Sequence[str], detail: Callable[[str], str], kind: str
) -> None:
    """Print a catalog or one entry without loading its runtime."""
    if name is None:
        print("\n".join(choices))
    elif name not in choices:
        raise SystemExit(f"unknown {kind} {name!r}")
    else:
        print(detail(name), end="")


def main(argv: list[str] | None = None) -> None:
    """Run the image benchmark CLI."""
    args = _parser().parse_args(sys.argv[1:] if argv is None else argv)
    if args.command == "run":
        try:
            _run(args)
        except ValueError as error:  # allow-except: report configuration errors to the CLI user
            print(f"error: {error}", file=sys.stderr)
            raise SystemExit(2) from error
    elif args.command == "models":
        _show_catalog(args.name, _model_names(), _model_detail, "model")
    elif args.command == "datasets":
        _show_catalog(args.name, _DATASETS, _dataset_detail, "dataset")
    elif args.command == "download":
        commands.download(args)
    elif args.command == "profile":
        commands.profile(args)
    else:
        raise SystemExit(f"{args.command} is not implemented by the image CLI yet")


if __name__ == "__main__":
    main()
