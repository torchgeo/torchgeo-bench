"""Explicit arguments and YAML loading for standalone profiling."""

import argparse
from typing import Any

from ..config_schema import load_yaml
from ..presets import NORMALIZATIONS, merge_settings
from ..profile_config import ProfileConfig


def _image_size(value: str) -> int | None:
    """Parse an explicit resize request, including disabled resizing."""
    if value.lower() == "none":
        return None
    try:
        size = int(value)
    except ValueError as error:  # allow-except: convert invalid sizes into argparse input errors
        raise argparse.ArgumentTypeError("image-size must be a positive integer or none") from error
    if size <= 0:
        raise argparse.ArgumentTypeError("image-size must be a positive integer or none")
    return size


def add_profile_arguments(parser: argparse.ArgumentParser) -> None:
    """Register profile flags without overriding YAML with parser defaults."""
    parser.set_defaults(_profile_explicit=True)
    parser.add_argument("--config", default=argparse.SUPPRESS, help="Standalone profile YAML")
    parser.add_argument("-m", "--model", default=argparse.SUPPRESS, help="Image model preset")
    parser.add_argument("-d", "--dataset", default=argparse.SUPPRESS, help="One dataset name")
    parser.add_argument(
        "--device", default=argparse.SUPPRESS, help="auto, cpu, cuda, or cuda:<index>"
    )
    parser.add_argument("--batch-size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--warmup", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--measurements", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument(
        "--bands", default=argparse.SUPPRESS, help="rgb, all, or comma-separated band names"
    )
    parser.add_argument("--partition", default=argparse.SUPPRESS)
    parser.add_argument(
        "--image-size", type=_image_size, metavar="PX|none", default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--interpolation",
        choices=("area", "bilinear", "bicubic", "nearest"),
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--normalization", choices=tuple(NORMALIZATIONS), default=argparse.SUPPRESS)
    parser.add_argument(
        "--precision", choices=("float32", "float16", "bfloat16"), default=argparse.SUPPRESS
    )
    parser.add_argument(
        "--count-flops", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS
    )
    parser.add_argument("--dry-run", action="store_true", default=argparse.SUPPRESS)


def _input_mapping(args: argparse.Namespace) -> dict[str, Any]:
    """Map explicit preprocessing flags, accepting old parser normalization names."""
    values: dict[str, Any] = {}
    legacy = not getattr(args, "_profile_explicit", False)
    aliases = {value: key for key, value in NORMALIZATIONS.items()}
    for name in ("bands", "partition", "image_size", "interpolation", "normalization"):
        if not hasattr(args, name):
            continue
        value = getattr(args, name)
        if legacy and value is None:
            continue
        if name == "bands" and isinstance(value, str) and value not in {"rgb", "all"}:
            value = [band.strip() for band in value.split(",")]
        if name == "normalization" and isinstance(value, str):
            value = aliases.get(value, value)
        values[name] = value
    return values


def load_profile_config(args: argparse.Namespace) -> ProfileConfig:
    """Load strict YAML, then apply only supplied command-line settings."""
    path = getattr(args, "config", None)
    base = load_yaml(path) if path is not None else {}
    overrides: dict[str, Any] = {}
    if getattr(args, "model", None) is not None:
        # Selecting a preset replaces a custom target and its constructor options.
        base = {**base, "model": {"name": args.model}}
    for name in ("dataset", "warmup", "measurements", "precision", "count_flops"):
        if hasattr(args, name):
            overrides[name] = getattr(args, name)
    runtime = {
        name: getattr(args, name)
        for name in ("device", "batch_size", "seed")
        if hasattr(args, name)
    }
    if runtime:
        overrides["runtime"] = runtime
    inputs = _input_mapping(args)
    if inputs:
        overrides["input"] = inputs
    return ProfileConfig.model_validate(merge_settings(base, overrides))
