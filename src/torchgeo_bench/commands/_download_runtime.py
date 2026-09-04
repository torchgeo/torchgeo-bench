# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Heavy runtime for dataset downloads."""

import importlib

download_module = importlib.import_module('torchgeo_bench.download')
