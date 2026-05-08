"""Benchmark model implementations and exports."""

from .image_stats import ImageStatsBench
from .interface import BenchModel
from .olmoearth import OlmoEarthBenchModel
from .rcf import RCFBench
from .sam3 import SAM3Encoder
from .segmentation_heads import ConvBlockHead, DPTHead, FPNHead, LinearHead
from .terratorch_models import (
    TerraTorchClayBench,
    TerraTorchPrithviBench,
    TerraTorchTerraMindBench,
)
from .timm import TimmPatchBenchModel
from .torchgeo_models import (
    TorchGeoCromaBench,
    TorchGeoDOFABench,
    TorchGeoEarthLocBench,
    TorchGeoPanopticonBench,
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
    "SAM3Encoder",
    "TorchGeoCromaBench",
    "TorchGeoDOFABench",
    "TorchGeoEarthLocBench",
    "TorchGeoPanopticonBench",
    "TorchGeoResNetBench",
    "TorchGeoScaleMAEBench",
    "TorchGeoSwinBench",
    "TerraTorchPrithviBench",
    "TerraTorchClayBench",
    "TerraTorchTerraMindBench",
    "LinearHead",
    "ConvBlockHead",
    "FPNHead",
    "DPTHead",
]
