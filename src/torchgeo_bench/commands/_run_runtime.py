"""Heavy runtime for the run command."""

import torchgeo_bench.main as benchmark_main


def run(config: object) -> None:
    """Execute the benchmark runtime."""
    benchmark_main.main(config)
