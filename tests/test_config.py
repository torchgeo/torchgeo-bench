"""Configuration composition and model-construction contracts."""

from torchgeo_bench.config import compose_config, instantiate, list_model_configs
from torchgeo_bench.datasets import get_bench_dataset_class


def test_instantiate_preserves_bandspec_objects() -> None:
    """Kwargs bypass OmegaConf, so BandSpec dataclasses reach the constructor intact."""
    cfg = compose_config(["model=rcf"])

    bench = get_bench_dataset_class("m-eurosat")()
    bands = bench.select_band_specs(tuple(bench.rgb_bands))
    model = instantiate(cfg.model, bands=bands)

    assert len(model.bands) == len(bands)
    assert all(actual is expected for actual, expected in zip(model.bands, bands, strict=True))
    assert model.num_channels == len(bands)


def test_double_plus_override_sets_the_real_key() -> None:
    cfg = compose_config(["model=rcf", "++model.pool=cls", "+model.gsd=1.0"])
    assert "+model" not in cfg
    assert cfg.model.pool == "cls"
    assert float(cfg.model.gsd) == 1.0


def test_model_names_are_posix_on_every_platform() -> None:
    names = list_model_configs()
    assert "torchgeo/scalemae_large_fmow" in names
    assert not any("\\" in name for name in names)
