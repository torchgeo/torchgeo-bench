"""Benchmark model implementations and exports."""

from .bench_models import ImageStatsBench, RCFBench
from .interface import BenchModel
from .olmoearth import OlmoEarthBenchModel
from .timm import TimmPatchBenchModel
from .torchgeo_models import (
    TorchGeoDOFABench,
    TorchGeoEarthLocBench,
    TorchGeoResNetBench,
    TorchGeoScaleMAEBench,
    TorchGeoSwinBench,
)

__all__: list[str] = [
    "BenchModel",
    "RCFBench",
    "ImageStatsBench",
    "TimmPatchBenchModel",
    "OlmoEarthBenchModel",
    "TorchGeoDOFABench",
    "TorchGeoEarthLocBench",
    "TorchGeoResNetBench",
    "TorchGeoScaleMAEBench",
    "TorchGeoSwinBench",
]
