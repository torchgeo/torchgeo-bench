# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Heavy runtime for the flops command."""

from omegaconf import DictConfig

import torchgeo_bench.flops_pipeline as flops_pipeline


def run(config: DictConfig) -> None:
    """Execute the FLOP profiling runtime."""
    flops_pipeline.main(config)


run_flops = run
