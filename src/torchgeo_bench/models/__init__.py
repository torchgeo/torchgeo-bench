"""Benchmark model implementations and exports.

Submodules import lazily so instantiating one model config doesn't pay the
import cost of every backend (timm, torchgeo.models, terratorch, ...).
"""

_EXPORTS: dict[str, str] = {
    "InputUnit": "torchgeo_bench.models._input_units",
    "NormalizationStrategy": "torchgeo_bench.models._normalization",
    "ImageStatsBench": "torchgeo_bench.models.image_stats",
    "BenchModel": "torchgeo_bench.models.interface",
    "OlmoEarthBenchModel": "torchgeo_bench.models.olmoearth",
    "RCFBench": "torchgeo_bench.models.rcf",
    "SAM3Encoder": "torchgeo_bench.models.sam3",
    "ConvBlockHead": "torchgeo_bench.models.segmentation_heads",
    "DPTHead": "torchgeo_bench.models.segmentation_heads",
    "FPNHead": "torchgeo_bench.models.segmentation_heads",
    "LinearHead": "torchgeo_bench.models.segmentation_heads",
    "PatchLinearHead": "torchgeo_bench.models.segmentation_heads",
    "TerraTorchClayBench": "torchgeo_bench.models.terratorch_models",
    "TerraTorchPrithviBench": "torchgeo_bench.models.terratorch_models",
    "TerraTorchTerraMindBench": "torchgeo_bench.models.terratorch_models",
    "TimmPatchBenchModel": "torchgeo_bench.models.timm",
    "TorchGeoCromaBench": "torchgeo_bench.models.torchgeo_models",
    "TorchGeoDOFABench": "torchgeo_bench.models.torchgeo_models",
    "TorchGeoEarthLocBench": "torchgeo_bench.models.torchgeo_models",
    "TorchGeoPanopticonBench": "torchgeo_bench.models.torchgeo_models",
    "TorchGeoResNetBench": "torchgeo_bench.models.torchgeo_models",
    "TorchGeoScaleMAEBench": "torchgeo_bench.models.torchgeo_models",
    "TorchGeoSwinBench": "torchgeo_bench.models.torchgeo_models",
}

__all__: list[str] = list(_EXPORTS)


def __getattr__(name: str) -> object:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_name), name)
