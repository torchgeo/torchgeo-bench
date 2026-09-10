"""Command-line interface for ``torchgeo-bench``.

Subcommands: ``run`` (benchmark), ``coord`` (CoordBench location-encoder
track), ``profile``/``intrinsic-dim`` (thin aliases over ``run`` that enable
one additive measurement pass), ``flops`` (compute-cost measurement), and
``download`` (datasets). This module imports only the standard library so
``torchgeo-bench --help`` is instant; torch and friends load only once a
command actually starts doing work.

Config precedence (lowest to highest): built-in Python defaults, then
``--config PATH`` (a YAML file for uncommon settings), then explicit CLI
flags. There is no positional ``key=value`` override syntax.
"""

import argparse
import sys

_RUN_EPILOG = """\
examples:
  torchgeo-bench run -m timm/resnet50 -d m-eurosat
  torchgeo-bench run -m torchgeo/scalemae_large_fmow -d m-eurosat,m-so2sat --device cuda:1
  torchgeo-bench run -m rcf --batch-size 128 --config my_settings.yaml
  torchgeo-bench run -m timm/resnet50 -d burn_scars --seg-head dpt --seg-epochs 20 \\
      --seg-lr 5e-4 --seg-scheduler none --seg-batch-size 32 --num-workers 8
  torchgeo-bench run -m timm/vit/vit_base_patch16_224 --use-cls-token \\
      --model-input-normalization imagenet --model-name vit_base_cls_imagenet

Common options are flags; anything else goes in a YAML file passed via
--config (flags still win over it). Use --print-config to see the merged
result.
"""


def _add_common_flags(parser: argparse.ArgumentParser) -> None:
    """Flags shared by every config-composing subcommand."""
    parser.add_argument(
        "--config",
        metavar="PATH",
        default=argparse.SUPPRESS,
        help="YAML file of uncommon settings, merged under CLI flags",
    )
    parser.add_argument(
        "--print-config", action="store_true", help="Print the merged config and exit"
    )


def _add_run_flags(parser: argparse.ArgumentParser) -> None:
    """Flags shared by ``run``, ``profile``, and ``intrinsic-dim``."""
    parser.add_argument(
        "-m",
        "--model",
        default=argparse.SUPPRESS,
        help="Model config, e.g. timm/resnet50 (default: rcf)",
    )
    parser.add_argument(
        "-d",
        "--datasets",
        default=argparse.SUPPRESS,
        help="Comma-separated dataset names, or 'all'",
    )
    parser.add_argument(
        "--device",
        default=argparse.SUPPRESS,
        help="Torch device, e.g. cuda:1 or cpu (default: auto -- cuda if available, else cpu)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=argparse.SUPPRESS,
        help="Results CSV path (default: results/models/<model name>.csv)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Skip (dataset, method, config) combos already in the output CSV",
    )
    parser.add_argument(
        "--seed", type=int, default=argparse.SUPPRESS, help="Random seed (default: 0)"
    )
    parser.add_argument(
        "--partition", default=argparse.SUPPRESS, help="GeoBench partition (default: 'default')"
    )
    parser.add_argument(
        "--bands", default=argparse.SUPPRESS, help="rgb | all | comma-separated band names"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=argparse.SUPPRESS,
        help="Dataloader batch size (default: 64)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=argparse.SUPPRESS,
        help="Dataloader worker processes, 0 disables multiprocessing (default: 4)",
    )
    parser.add_argument(
        "--image-size", type=int, default=argparse.SUPPRESS, help="Resize edge in px (default: 224)"
    )
    parser.add_argument(
        "--time-steps",
        type=int,
        default=argparse.SUPPRESS,
        help="Dates per sample for multi-temporal datasets, e.g. pastis (default: dataset's own)",
    )
    parser.add_argument(
        "--interpolation",
        choices=["area", "bilinear", "bicubic", "nearest"],
        default=argparse.SUPPRESS,
        help="Resize interpolation, used if --image-size is set (default: bilinear)",
    )
    parser.add_argument(
        "--normalization",
        choices=["bandspec_zscore", "model_native", "minmax", "minmax_zscore", "identity"],
        default=argparse.SUPPRESS,
        help="Input normalization strategy (default: bandspec_zscore)",
    )
    parser.add_argument(
        "--skip-linear",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Skip the linear probe (KNN only)",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=argparse.SUPPRESS,
        help="Bootstrap resamples for CIs (default: 200)",
    )
    parser.add_argument(
        "--merge-val",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Merge train+val for the final logistic fit (default: true)",
    )
    parser.add_argument(
        "--knn-device",
        default=argparse.SUPPRESS,
        help="FAISS device for KNN, e.g. cpu or cuda (default: inherit --device)",
    )
    parser.add_argument(
        "--seg-head",
        default=argparse.SUPPRESS,
        help="Segmentation head type, e.g. fpn or dpt (default: fpn)",
    )
    parser.add_argument(
        "--seg-epochs",
        type=int,
        default=argparse.SUPPRESS,
        help="Segmentation probe training epochs (default: 10)",
    )
    parser.add_argument(
        "--seg-lr",
        type=float,
        default=argparse.SUPPRESS,
        help="Segmentation probe learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--seg-scheduler",
        choices=["cosine", "none"],
        default=argparse.SUPPRESS,
        help="Segmentation probe LR scheduler (default: cosine)",
    )
    parser.add_argument(
        "--seg-batch-size",
        type=int,
        default=argparse.SUPPRESS,
        help="Segmentation probe training batch size (default: 64)",
    )
    parser.add_argument(
        "--seg-cache",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Pre-extract and cache frozen backbone features for probe training (default: true)",
    )
    parser.add_argument(
        "--seg-cache-dtype",
        choices=["float16", "float32"],
        default=argparse.SUPPRESS,
        help="Storage dtype for cached segmentation features (default: float16)",
    )
    parser.add_argument(
        "--use-cls-token",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="For ViT-family timm models, use the CLS token instead of averaging "
        "spatial tokens (default: false)",
    )
    parser.add_argument(
        "--model-input-normalization",
        choices=["bands_zscore", "imagenet", "timm_default", "none"],
        default=argparse.SUPPRESS,
        help="timm model-side input normalization (default: bands_zscore)",
    )
    parser.add_argument(
        "--model-name",
        default=argparse.SUPPRESS,
        help="Override the model's display/result-file name, e.g. to distinguish "
        "an ablation variant (default: the model config's own name)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Verbose progress logging",
    )
    _add_common_flags(parser)
    parser.add_argument(
        "--list-models", action="store_true", help="List available model configs and exit"
    )
    parser.add_argument(
        "--list-datasets", action="store_true", help="List available dataset names and exit"
    )
    parser.add_argument(
        "--model-help",
        metavar="MODEL",
        default=None,
        help="Print a model config's YAML (available settings) and exit",
    )


def _add_coord_flags(parser: argparse.ArgumentParser) -> None:
    """Flags for the ``coord`` (CoordBench location-encoder) subcommand."""
    parser.add_argument(
        "-m", "--model", default=argparse.SUPPRESS, help="Coord model, e.g. sincos (required)"
    )
    parser.add_argument(
        "--device",
        default=argparse.SUPPRESS,
        help="Torch device, e.g. cuda:1 or cpu (default: auto -- cuda if available, else cpu)",
    )
    parser.add_argument(
        "-o", "--output", default=argparse.SUPPRESS, help="Results CSV path (default: coord.output)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Skip (benchmark, task, method, split) combos already in the output CSV",
    )
    parser.add_argument(
        "--seed", type=int, default=argparse.SUPPRESS, help="Random seed (default: 0)"
    )
    parser.add_argument(
        "--names",
        default=argparse.SUPPRESS,
        help="all | comma-separated families/benchmark names (default: all)",
    )
    parser.add_argument(
        "--methods",
        default=argparse.SUPPRESS,
        help="Comma-separated subset of {knn, linear} (default: knn,linear)",
    )
    parser.add_argument(
        "--split",
        choices=["random", "spatial", "both"],
        default=argparse.SUPPRESS,
        help="CV split mode (default: random)",
    )
    parser.add_argument(
        "--folds", type=int, default=argparse.SUPPRESS, help="CV fold count (default: 5)"
    )
    parser.add_argument(
        "--cell-deg",
        type=float,
        default=argparse.SUPPRESS,
        help="Spatial-block grid-cell size in degrees (default: 10.0)",
    )
    parser.add_argument(
        "--knn-k", type=int, default=argparse.SUPPRESS, help="KNN neighbours (default: 5)"
    )
    parser.add_argument(
        "--knn-device", default=argparse.SUPPRESS, help="FAISS device for KNN (default: cpu)"
    )
    _add_common_flags(parser)


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
    _add_run_flags(run)
    run.set_defaults(func=_cmd_run)

    profile = sub.add_parser(
        "profile",
        help="Run the benchmark with the compute-profile pass enabled "
        "(throughput/latency/params, on top of knn/linear)",
    )
    _add_run_flags(profile)
    profile.set_defaults(func=_cmd_profile)

    intrinsic_dim = sub.add_parser(
        "intrinsic-dim",
        help="Run the benchmark with the intrinsic-dimension pass enabled (on top of knn/linear)",
    )
    _add_run_flags(intrinsic_dim)
    intrinsic_dim.set_defaults(func=_cmd_intrinsic_dim)

    coord = sub.add_parser("coord", help="Run the CoordBench location-encoder track")
    _add_coord_flags(coord)
    coord.set_defaults(func=_cmd_coord)

    flops = sub.add_parser("flops", help="Measure per-sample compute cost (GFLOPs)")
    flops.add_argument("-m", "--model", default=argparse.SUPPRESS, help="Model config (required)")
    flops.add_argument(
        "--device",
        default=argparse.SUPPRESS,
        help="Torch device, e.g. cuda:1 or cpu (default: auto -- cuda if available, else cpu)",
    )
    flops.add_argument("-o", "--output", default=argparse.SUPPRESS, help="Results CSV path")
    _add_common_flags(flops)
    flops.set_defaults(func=_cmd_flops)

    download = sub.add_parser("download", help="Download benchmark datasets")
    download.add_argument(
        "target",
        choices=["geobench_v1", "geobench_v2", "eurosat", "resisc45"],
        help="What to download",
    )
    download.add_argument(
        "-o", "--output-dir", default="data", help="Benchmark data root (default: data)"
    )
    download.add_argument(
        "--datasets", default=None, help="(GeoBench only) comma-separated dataset names"
    )
    download.set_defaults(func=_cmd_download)

    return parser


def _setup_logging(verbose: bool = False) -> None:
    import logging

    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _flag_overrides(args: argparse.Namespace) -> dict:
    """Translate explicit run/profile/intrinsic-dim flags into a nested settings mapping.

    Only flags actually supplied on the command line appear on ``args``
    (every optional flag defaults to ``argparse.SUPPRESS``), so a
    ``hasattr`` check distinguishes "not given" from a value that happens to
    equal a settings default. ``-m/--model`` (which *selects* the model YAML
    to load) is not included here -- it is passed to ``compose_config``
    directly -- but a handful of flags override specific keys *within* the
    selected model config (e.g. ``--use-cls-token``, ``--model-name``) and
    are collected into a ``model`` mapping like any other settings group.
    """
    overrides: dict = {}
    dataset: dict = {}
    eval_: dict = {}
    segmentation: dict = {}
    model: dict = {}
    if hasattr(args, "datasets"):
        names = args.datasets if args.datasets == "all" else _split_csv(args.datasets)
        dataset["names"] = names
    if hasattr(args, "device"):
        overrides["device"] = args.device
    if hasattr(args, "output"):
        overrides["output"] = args.output
    if getattr(args, "resume", False):
        overrides["resume"] = True
    if hasattr(args, "seed"):
        overrides["seed"] = args.seed
    if hasattr(args, "partition"):
        dataset["partition"] = args.partition
    if hasattr(args, "bands"):
        dataset["bands"] = args.bands if args.bands in ("rgb", "all") else _split_csv(args.bands)
    if hasattr(args, "batch_size"):
        dataset["batch_size"] = args.batch_size
    if hasattr(args, "num_workers"):
        dataset["num_workers"] = args.num_workers
    if hasattr(args, "image_size"):
        dataset["image_size"] = args.image_size
    if hasattr(args, "time_steps"):
        dataset["time_steps"] = args.time_steps
    if hasattr(args, "interpolation"):
        dataset["interpolation"] = args.interpolation
    if hasattr(args, "normalization"):
        dataset["normalization"] = args.normalization
    if getattr(args, "skip_linear", False):
        eval_["skip_linear"] = True
    if hasattr(args, "bootstrap"):
        eval_["bootstrap"] = args.bootstrap
    if hasattr(args, "merge_val"):
        eval_["merge_val"] = args.merge_val
    if hasattr(args, "knn_device"):
        eval_["knn_device"] = args.knn_device
    if hasattr(args, "seg_head"):
        segmentation["head_type"] = args.seg_head
    if hasattr(args, "seg_epochs"):
        segmentation["epochs"] = args.seg_epochs
    if hasattr(args, "seg_lr"):
        segmentation["lr"] = args.seg_lr
    if hasattr(args, "seg_scheduler"):
        segmentation["lr_scheduler"] = args.seg_scheduler
    if hasattr(args, "seg_batch_size"):
        segmentation["batch_size"] = args.seg_batch_size
    if hasattr(args, "seg_cache"):
        segmentation["cache_features"] = args.seg_cache
    if hasattr(args, "seg_cache_dtype"):
        segmentation["cache_dtype"] = args.seg_cache_dtype
    if hasattr(args, "use_cls_token"):
        model["use_cls_token"] = args.use_cls_token
    if hasattr(args, "model_input_normalization"):
        model["input_normalization"] = args.model_input_normalization
    if hasattr(args, "model_name"):
        model["name"] = args.model_name
    if getattr(args, "verbose", False):
        overrides["verbose"] = True
    if segmentation:
        eval_["segmentation"] = segmentation
    if dataset:
        overrides["dataset"] = dataset
    if eval_:
        overrides["eval"] = eval_
    if model:
        overrides["model"] = model
    return overrides


def _coord_flag_overrides(args: argparse.Namespace) -> dict:
    """Translate ``coord`` flags into a nested settings mapping; always forces mode=coord."""
    overrides: dict = {"mode": "coord"}
    coord: dict = {}
    if hasattr(args, "device"):
        overrides["device"] = args.device
    if hasattr(args, "output"):
        coord["output"] = args.output
    if getattr(args, "resume", False):
        overrides["resume"] = True
    if hasattr(args, "seed"):
        overrides["seed"] = args.seed
    if hasattr(args, "names"):
        coord["names"] = args.names if args.names == "all" else _split_csv(args.names)
    if hasattr(args, "methods"):
        coord["methods"] = _split_csv(args.methods)
    if hasattr(args, "split"):
        coord["split"] = args.split
    if hasattr(args, "folds"):
        coord["folds"] = args.folds
    if hasattr(args, "cell_deg"):
        coord["cell_deg"] = args.cell_deg
    if hasattr(args, "knn_k"):
        coord["knn_k"] = args.knn_k
    if hasattr(args, "knn_device"):
        coord["knn_device"] = args.knn_device
    if coord:
        overrides["coord"] = coord
    return overrides


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _compose(
    args: argparse.Namespace,
    *,
    config_name: str,
    default_model: str | None,
    forced_overrides: dict | None = None,
    coord: bool = False,
):
    from torchgeo_bench.config import compose_config
    from torchgeo_bench.settings import merge

    overrides = _coord_flag_overrides(args) if coord else _flag_overrides(args)
    if forced_overrides:
        overrides = merge(overrides, forced_overrides)
    try:
        return compose_config(
            overrides,
            config_name=config_name,
            default_model=default_model,
            model=getattr(args, "model", None),
            config_path=getattr(args, "config", None),
        )
    except ValueError as err:
        raise SystemExit(f"error: {err}") from err


def _print_config(cfg) -> None:
    import yaml

    from torchgeo_bench.settings import to_dict

    print(yaml.safe_dump(to_dict(cfg), sort_keys=False), end="")


def _cmd_run(args: argparse.Namespace) -> None:
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
    cfg = _compose(args, config_name="config", default_model="rcf")
    if args.print_config:
        _print_config(cfg)
        return
    _setup_logging(bool(cfg.verbose))
    from torchgeo_bench.main import main

    main(cfg)


def _cmd_profile(args: argparse.Namespace) -> None:
    if args.list_models or args.list_datasets or args.model_help is not None:
        _cmd_run(args)
        return
    cfg = _compose(
        args,
        config_name="config",
        default_model="rcf",
        forced_overrides={"eval": {"profile": {"enabled": True}}},
    )
    if args.print_config:
        _print_config(cfg)
        return
    _setup_logging(bool(cfg.verbose))
    from torchgeo_bench.main import main

    main(cfg)


def _cmd_intrinsic_dim(args: argparse.Namespace) -> None:
    if args.list_models or args.list_datasets or args.model_help is not None:
        _cmd_run(args)
        return
    cfg = _compose(
        args,
        config_name="config",
        default_model="rcf",
        forced_overrides={"eval": {"intrinsic_dim": {"enabled": True}}},
    )
    if args.print_config:
        _print_config(cfg)
        return
    _setup_logging(bool(cfg.verbose))
    from torchgeo_bench.main import main

    main(cfg)


def _cmd_coord(args: argparse.Namespace) -> None:
    cfg = _compose(args, config_name="config", default_model=None, coord=True)
    if args.print_config:
        _print_config(cfg)
        return
    _setup_logging(bool(cfg.verbose))
    from torchgeo_bench.main import main

    main(cfg)


def _cmd_flops(args: argparse.Namespace) -> None:
    cfg = _compose(args, config_name="flops_config", default_model=None)
    if args.print_config:
        _print_config(cfg)
        return
    _setup_logging(verbose=True)
    from torchgeo_bench.flops_pipeline import main

    main(cfg)


def _cmd_download(args: argparse.Namespace) -> None:
    from pathlib import Path

    _setup_logging(verbose=True)
    from torchgeo_bench.download import (
        download_eurosat,
        download_geobench_v1,
        download_geobench_v2,
        download_resisc45,
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
        if args.target == "eurosat":
            download_eurosat(output_dir)
        else:
            download_resisc45(output_dir)


def main(argv: list[str] | None = None) -> None:
    """Entry point for the ``torchgeo-bench`` console script."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
