# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Heavy runtime for the run command."""

from omegaconf import DictConfig

import torchgeo_bench.main as benchmark_main


def run(config: DictConfig) -> None:
    """Execute the benchmark runtime."""
    benchmark_main.main(config)


run_benchmark = run
