"""Benchmark model implementations and exports."""

from ._input_units import InputUnit
from ._normalization import NormalizationStrategy
from .image_stats import ImageStatsBench
from .interface import BenchModel
from .olmoearth import OlmoEarthBenchModel
from .rcf import RCFBench
from .sam3 import SAM3Encoder
from .segmentation_heads import ConvBlockHead, DPTHead, FPNHead, LinearHead, PatchLinearHead
from .timm import TimmPatchBenchModel
from .torchgeo_models import (
    TorchGeoCromaBench,
    TorchGeoDEOBench,
    TorchGeoDOFABench,
    TorchGeoEarthLocBench,
    TorchGeoPanopticonBench,
    TorchGeoResNetBench,
    TorchGeoScaleMAEBench,
    TorchGeoSwinBench,
)
from .universat import UniverSatBenchModel

__all__: list[str] = [
    "BenchModel",
    "InputUnit",
    "NormalizationStrategy",
    "RCFBench",
    "ImageStatsBench",
    "TimmPatchBenchModel",
    "UniverSatBenchModel",
    "OlmoEarthBenchModel",
    "SAM3Encoder",
    "TorchGeoCromaBench",
    "TorchGeoDEOBench",
    "TorchGeoDOFABench",
    "TorchGeoEarthLocBench",
    "TorchGeoPanopticonBench",
    "TorchGeoResNetBench",
    "TorchGeoScaleMAEBench",
    "TorchGeoSwinBench",
    "LinearHead",
    "PatchLinearHead",
    "ConvBlockHead",
    "FPNHead",
    "DPTHead",
]
