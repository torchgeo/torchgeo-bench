"""Explicit constructors for the first migrated benchmark models."""

from dataclasses import dataclass

from torch import nn

from torchgeo_bench.datasets.base import BandSpec

from .rcf import RCFBench
from .timm import TimmPatchBenchModel


@dataclass(frozen=True)
class TimmModelConfig:
    """Validated settings for a timm benchmark model."""

    model_name: str
    pretrained: bool = True
    global_pool: str | None = 'avg'
    input_normalization: str = 'bands_zscore'


@dataclass(frozen=True)
class RCFModelConfig:
    """Validated settings for an RCF benchmark model."""

    features: int = 512
    kernel_size: int = 3
    mode: str = 'gaussian'
    stats_mode: str = 'mean'
    seed: int | None = None


def build_timm_model(
    config: TimmModelConfig,
    bands: list[BandSpec],
    *,
    normalization: str = 'bandspec_zscore',
) -> TimmPatchBenchModel:
    """Construct a timm model from explicit, validated settings."""
    return TimmPatchBenchModel(
        bands=bands,
        model_name=config.model_name,
        pretrained=config.pretrained,
        global_pool=config.global_pool,
        input_normalization=config.input_normalization,
        normalization=normalization,
    )


def build_rcf_model(
    config: RCFModelConfig,
    bands: list[BandSpec],
    *,
    normalization: str = 'bandspec_zscore',
) -> RCFBench:
    """Construct an RCF model from explicit, validated settings."""
    return RCFBench(
        bands=bands,
        features=config.features,
        kernel_size=config.kernel_size,
        mode=config.mode,
        stats_mode=config.stats_mode,
        seed=config.seed,
        normalization=normalization,
    )


def build_model(
    name: str,
    bands: list[BandSpec],
    *,
    normalization: str = 'bandspec_zscore',
    timm: TimmModelConfig | None = None,
    rcf: RCFModelConfig | None = None,
) -> nn.Module:
    """Construct a migrated model by its public preset name.

    ``timm`` and ``rcf`` are mutually exclusive explicit configuration
    objects. They are optional so the small dispatcher can reject settings
    that do not belong to the selected model family.
    """
    if name == 'rcf':
        if timm is not None:
            raise ValueError("timm settings cannot be used with model 'rcf'")
        return build_rcf_model(rcf or RCFModelConfig(), bands, normalization=normalization)
    if name.startswith('timm/'):
        if rcf is not None:
            raise ValueError(f"rcf settings cannot be used with model {name!r}")
        config = timm or TimmModelConfig(model_name=name.removeprefix('timm/'))
        if config.model_name != name.removeprefix('timm/'):
            raise ValueError(f"timm model_name does not match preset {name!r}")
        return build_timm_model(config, bands, normalization=normalization)
    raise ValueError(f"Model preset {name!r} is not migrated to explicit construction")
