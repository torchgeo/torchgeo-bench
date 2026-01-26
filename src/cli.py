"""Command-line interface for torchgeo-bench."""

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def download_command(args: argparse.Namespace) -> int:
    """Execute the download command."""
    # Import here to avoid loading heavy dependencies for --help
    from src.download import (
        GEOBENCH_V2_DATASETS,
        download_geobench_v1,
        download_geobench_v2,
    )

    if args.force:
        print("Force mode enabled: existing files will be re-downloaded")

    try:
        if args.version == "v1":
            download_geobench_v1(args.output_dir, args.force)
        elif args.version == "v2":
            # Parse datasets argument
            datasets = None
            if args.datasets and args.datasets != "all":
                datasets = [d.strip() for d in args.datasets.split(",")]
            download_geobench_v2(args.output_dir, datasets, args.force)
        return 0
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return 1


def run_command(args: argparse.Namespace) -> int:
    """Execute the run command."""
    import subprocess
    from pathlib import Path

    # Pass through all arguments after 'run'
    hydra_args = args.hydra_args if args.hydra_args else []

    # Find the torchgeo_bench.py script
    script_dir = Path(__file__).parent.parent
    script_path = script_dir / "torchgeo_bench.py"

    if not script_path.exists():
        logger.error(f"Could not find torchgeo_bench.py at {script_path}")
        return 1

    # Run the script directly with subprocess to preserve Hydra's argument handling
    try:
        cmd = [sys.executable, str(script_path)] + hydra_args
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except Exception as e:
        logger.error(f"Benchmark run failed: {e}")
        return 1


def main() -> int:
    """Main CLI entry point."""
    # Special handling for "run" command - pass everything after "run" to Hydra
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        # Extract hydra args (everything after "run")
        hydra_args = sys.argv[2:]

        # Create a minimal args object for run_command
        args = argparse.Namespace(hydra_args=hydra_args)
        return run_command(args)

    # For other commands, use standard argparse
    parser = argparse.ArgumentParser(
        prog="torchgeo-bench",
        description="Lightweight benchmarking framework for geospatial foundation models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Download command
    download_parser = subparsers.add_parser(
        "download",
        help="Download and extract GeoBench datasets from Hugging Face",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    download_parser.add_argument(
        "--version",
        type=str,
        choices=["v1", "v2"],
        default="v1",
        help="GeoBench version to download",
    )
    download_parser.add_argument(
        "--datasets",
        type=str,
        default="all",
        help="For v2: comma-separated dataset names or 'all' (default: all)",
    )
    download_parser.add_argument(
        "--output-dir",
        type=Path,
        default="data/",
        help="Directory to download and extract the dataset",
    )
    download_parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download of files even if they already exist",
    )

    # Run command - just show basic help since actual parsing is done above
    run_parser = subparsers.add_parser(
        "run",
        help="Run benchmark experiments with Hydra configuration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    args = parser.parse_args()

    if args.command == "download":
        return download_command(args)
    elif args.command == "run":
        assert False, "This should never be reached due to special handling above."
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
