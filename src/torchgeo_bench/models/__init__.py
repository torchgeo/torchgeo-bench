"""Benchmark model implementations and exports."""

from ._input_units import InputUnit
from ._normalization import NormalizationStrategy
from .image_stats import ImageStatsBench
from .interface import BenchModel
from .olmoearth import OlmoEarthBenchModel
from .rcf import RCFBench
from .sam3 import SAM3Encoder
from .segmentation_heads import ConvBlockHead, DPTHead, FPNHead, LinearHead, PatchLinearHead
from .terratorch_models import (
    TerraTorchClayBench,
    TerraTorchPrithviBench,
    TerraTorchTerraMindBench,
)
from .timm import TimmPatchBenchModel
from .torchgeo_croma_panopticon import TorchGeoCromaBench, TorchGeoPanopticonBench
from .torchgeo_deo import TorchGeoDEOBench
from .torchgeo_dofa_earthloc import TorchGeoDOFABench, TorchGeoEarthLocBench
from .torchgeo_resnet_swin import TorchGeoResNetBench, TorchGeoSwinBench
from .torchgeo_scalemae import TorchGeoScaleMAEBench
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
    "TerraTorchPrithviBench",
    "TerraTorchClayBench",
    "TerraTorchTerraMindBench",
    "LinearHead",
    "PatchLinearHead",
    "ConvBlockHead",
    "FPNHead",
    "DPTHead",
]
