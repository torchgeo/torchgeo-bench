"""Explicit constructors for the first migrated benchmark models."""

from dataclasses import dataclass

from torchgeo.datasets import NonGeoDataset

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
    normalize: bool = False
    auto_resize: bool = False
    target_size: int | None = None
    use_cls_token: bool = False

    def __post_init__(self) -> None:
        """Validate settings before importing or constructing a backbone."""
        if not self.model_name:
            raise ValueError('model_name must not be empty')
        if self.target_size is not None and self.target_size <= 0:
            raise ValueError('target_size must be positive')
        if self.global_pool not in (None, '', 'avg', 'max', 'avgmax', 'catavgmax'):
            raise ValueError(f'unsupported global_pool: {self.global_pool!r}')


@dataclass(frozen=True)
class RCFModelConfig:
    """Validated settings for an RCF benchmark model."""

    features: int = 512
    kernel_size: int = 3
    mode: str = 'gaussian'
    stats_mode: str = 'mean'
    seed: int | None = None
    dataset: NonGeoDataset | None = None

    def __post_init__(self) -> None:
        """Validate settings before constructing the filter bank."""
        if self.features <= 0 or self.features % 2:
            raise ValueError('features must be a positive even number')
        if self.kernel_size <= 0:
            raise ValueError('kernel_size must be positive')
        if self.mode not in ('gaussian', 'empirical'):
            raise ValueError("mode must be 'gaussian' or 'empirical'")
        if self.stats_mode not in ('mean', 'stdev', 'all'):
            raise ValueError("stats_mode must be 'mean', 'stdev', or 'all'")
        if self.mode == 'empirical' and self.dataset is None:
            raise ValueError("dataset must be provided for empirical mode")


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
        normalize=config.normalize,
        global_pool=config.global_pool,
        auto_resize=config.auto_resize,
        target_size=config.target_size,
        use_cls_token=config.use_cls_token,
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
        dataset=config.dataset,
        normalization=normalization,
    )
