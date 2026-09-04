# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Standalone real-batch profile command."""

import argparse

from . import _profile_runtime as runtime


def profile(args: argparse.Namespace) -> None:
    """Profile one selected model and dataset."""
    runtime.run(args)
