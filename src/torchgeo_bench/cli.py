"""Command-line interface for ``torchgeo-bench``.

Subcommands: ``run`` (benchmark), ``flops`` (compute-cost measurement), and
``download`` (datasets).  This module imports only the standard library so
``torchgeo-bench --help`` is instant; torch and friends load only once a
command actually starts doing work.
"""

import argparse
import sys

_RUN_EPILOG = """\
examples:
  torchgeo-bench run -m timm/resnet50 -d m-eurosat
  torchgeo-bench run -m torchgeo/scalemae_large_fmow -d m-eurosat,m-so2sat --device cuda:1
  torchgeo-bench run -m rcf dataset.batch_size=128 eval.knn_k=10

Any key=value pair overrides the config (values parse as YAML, e.g.
dataset.names=[m-eurosat]). Flags are shorthand for common overrides and win
over positional key=value pairs. Use --print-config to see the merged result.
"""


def _add_override_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "overrides",
        nargs="*",
        metavar="key=value",
        help="Config overrides, e.g. dataset.batch_size=128 (values parse as YAML)",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="torchgeo-bench",
        description="Benchmark geospatial foundation models on GeoBench datasets.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser(
        "run",
        help="Run KNN / linear-probe / segmentation benchmarks",
        epilog=_RUN_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    run.add_argument("-m", "--model", default=None, help="Model config, e.g. timm/resnet50 (default: rcf)")
    run.add_argument("-d", "--datasets", default=None, help="Comma-separated dataset names, or 'all'")
    run.add_argument("--device", default=None, help="Torch device, e.g. cuda:1 or cpu")
    run.add_argument("-o", "--output", default=None, help="Results CSV path (default: results/all_results.csv)")
    run.add_argument("--resume", action="store_true", help="Skip (dataset, method, config) combos already in the output CSV")
    run.add_argument("--seed", type=int, default=None, help="Random seed (default: 0)")
    run.add_argument("--partition", default=None, help="GeoBench partition (default: 'default')")
    run.add_argument("--bands", default=None, help="rgb | all | comma-separated band names")
    run.add_argument("--batch-size", type=int, default=None, help="Dataloader batch size (default: 64)")
    run.add_argument("--image-size", type=int, default=None, help="Resize edge in px (default: 224)")
    run.add_argument(
        "--normalization",
        choices=["bandspec_zscore", "model_native", "minmax", "minmax_zscore", "identity"],
        default=None,
        help="Input normalization strategy (default: bandspec_zscore)",
    )
    run.add_argument("--skip-linear", action="store_true", help="Skip the linear probe (KNN only)")
    run.add_argument("--bootstrap", type=int, default=None, help="Bootstrap resamples for CIs (default: 200)")
    run.add_argument("-v", "--verbose", action="store_true", help="Verbose progress logging")
    run.add_argument("--print-config", action="store_true", help="Print the merged config and exit")
    run.add_argument("--list-models", action="store_true", help="List available model configs and exit")
    _add_override_arg(run)
    run.set_defaults(func=_cmd_run)

    flops = sub.add_parser("flops", help="Measure per-sample compute cost (GFLOPs)")
    flops.add_argument("-m", "--model", required=False, default=None, help="Model config (required)")
    flops.add_argument("--device", default=None, help="Torch device")
    flops.add_argument("-o", "--output", default=None, help="Results CSV path")
    flops.add_argument("--print-config", action="store_true", help="Print the merged config and exit")
    _add_override_arg(flops)
    flops.set_defaults(func=_cmd_flops)

    download = sub.add_parser("download", help="Download benchmark datasets")
    download.add_argument("target", choices=["geobench_v1", "geobench_v2", "eurosat"], help="What to download")
    download.add_argument("-o", "--output-dir", default="data", help="Benchmark data root (default: data)")
    download.add_argument("--datasets", default=None, help="(GeoBench only) comma-separated dataset names")
    download.set_defaults(func=_cmd_download)

    return parser


def _setup_logging() -> None:
    import logging

    from rich.logging import RichHandler

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, markup=True)],
    )


def _flag_overrides(args: argparse.Namespace) -> list[str]:
    """Translate convenience flags into config dotlist overrides."""
    overrides = []
    if args.model is not None:
        overrides.append(f"model={args.model}")
    if args.datasets is not None:
        names = args.datasets if args.datasets == "all" else f"[{args.datasets}]"
        overrides.append(f"dataset.names={names}")
    if args.device is not None:
        overrides.append(f"device={args.device}")
    if args.output is not None:
        overrides.append(f"output={args.output}")
    if getattr(args, "resume", False):
        overrides.append("resume=true")
    if getattr(args, "seed", None) is not None:
        overrides.append(f"seed={args.seed}")
    if getattr(args, "partition", None) is not None:
        overrides.append(f"dataset.partition={args.partition}")
    if getattr(args, "bands", None) is not None:
        bands = args.bands if args.bands in ("rgb", "all") else f"[{args.bands}]"
        overrides.append(f"dataset.bands={bands}")
    if getattr(args, "batch_size", None) is not None:
        overrides.append(f"dataset.batch_size={args.batch_size}")
    if getattr(args, "image_size", None) is not None:
        overrides.append(f"dataset.image_size={args.image_size}")
    if getattr(args, "normalization", None) is not None:
        overrides.append(f"dataset.normalization={args.normalization}")
    if getattr(args, "skip_linear", False):
        overrides.append("eval.skip_linear=true")
    if getattr(args, "bootstrap", None) is not None:
        overrides.append(f"eval.bootstrap={args.bootstrap}")
    if getattr(args, "verbose", False):
        overrides.append("verbose=true")
    return overrides


def _compose(args: argparse.Namespace, *, config_name: str, default_model: str | None):
    from torchgeo_bench.config import compose_config

    try:
        return compose_config(
            [*args.overrides, *_flag_overrides(args)],
            config_name=config_name,
            default_model=default_model,
        )
    except ValueError as err:
        raise SystemExit(f"error: {err}") from err


def _cmd_run(args: argparse.Namespace) -> None:
    if args.list_models:
        from torchgeo_bench.config import list_model_configs

        print("\n".join(list_model_configs()))
        return
    cfg = _compose(args, config_name="config", default_model="rcf")
    if args.print_config:
        from omegaconf import OmegaConf

        print(OmegaConf.to_yaml(cfg), end="")
        return
    _setup_logging()
    from torchgeo_bench.main import main

    main(cfg)


def _cmd_flops(args: argparse.Namespace) -> None:
    cfg = _compose(args, config_name="flops_config", default_model=None)
    if args.print_config:
        from omegaconf import OmegaConf

        print(OmegaConf.to_yaml(cfg), end="")
        return
    _setup_logging()
    from torchgeo_bench.flops_pipeline import main

    main(cfg)


def _cmd_download(args: argparse.Namespace) -> None:
    from pathlib import Path

    _setup_logging()
    from torchgeo_bench.download import (
        download_eurosat,
        download_geobench_v1,
        download_geobench_v2,
    )

    names = None
    if args.datasets is not None:
        names = [n.strip() for n in args.datasets.split(",") if n.strip()]
        if not names:
            raise SystemExit("error: --datasets must contain at least one dataset name")

    output_dir = Path(args.output_dir)
    if args.target == "geobench_v1":
        download_geobench_v1(output_dir, datasets=names)
    elif args.target == "geobench_v2":
        download_geobench_v2(output_dir, datasets=names)
    else:
        if names is not None:
            raise SystemExit("error: --datasets is only supported for GeoBench downloads")
        download_eurosat(output_dir)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``torchgeo-bench`` console script."""
    argv = list(sys.argv[1:] if argv is None else argv)
    # key=value override tokens can appear anywhere after the subcommand;
    # pull them out before argparse so they mix freely with flags.
    overrides: list[str] = []
    if argv and argv[0] in ("run", "flops"):
        rest = [argv[0]]
        for token in argv[1:]:
            if "=" in token and not token.startswith("-"):
                overrides.append(token)
            else:
                rest.append(token)
        argv = rest
    parser = _build_parser()
    args = parser.parse_args(argv)
    if hasattr(args, "overrides"):
        args.overrides = [*args.overrides, *overrides]
    args.func(args)


if __name__ == "__main__":
    main()
