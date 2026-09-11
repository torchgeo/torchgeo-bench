# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Heavy runtime for the flops command."""

import torchgeo_bench.flops_pipeline as flops_pipeline
from torchgeo_bench.flops_config import FlopsConfig


def run(config: FlopsConfig) -> None:
    """Execute the FLOP profiling runtime."""
    flops_pipeline.main(config)


run_flops = run
