# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Standalone real-batch profile command."""

import argparse


def profile(args: argparse.Namespace) -> None:
    """Profile one selected model and dataset."""
    from torchgeo_bench.commands._profile_runtime import run

    run(args)
