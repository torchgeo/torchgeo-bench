# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Standalone real-batch profile command."""

import argparse

from .. import commands


def profile(args: argparse.Namespace) -> None:
    """Profile one selected model and dataset."""
    commands._profile_runtime.run(args)
