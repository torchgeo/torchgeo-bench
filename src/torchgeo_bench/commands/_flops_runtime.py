"""Heavy runtime for the flops command."""

import torchgeo_bench.flops_pipeline as flops_pipeline


def run(config: object) -> None:
    """Execute the FLOP profiling runtime."""
    flops_pipeline.main(config)
