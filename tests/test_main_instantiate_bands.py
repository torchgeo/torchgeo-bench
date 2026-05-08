"""Unit test for the Hydra `instantiate(cfg.model, bands=..., _convert_="object")` contract."""

from hydra import compose, initialize_config_module
from hydra.utils import instantiate

from torchgeo_bench.datasets.base import BandSpec
from torchgeo_bench.models.interface import BenchModel


def _bands() -> list[BandSpec]:
    return [
        BandSpec(
            sensor="s2",
            name=f"b{i}",
            source_name=f"B{i}",
            mean=10.0,
            std=2.0,
            min=0.0,
            max=255.0,
        )
        for i in range(3)
    ]


def test_instantiate_preserves_bandspec_objects():
    """`_convert_="object"` keeps BandSpec dataclasses intact (not OmegaConf'd)."""
    with initialize_config_module(config_module="torchgeo_bench.conf", version_base="1.3"):
        cfg = compose(config_name="config", overrides=["model=rcf"])

    bands = _bands()
    model = instantiate(cfg.model, bands=bands, _convert_="object")

    assert isinstance(model, BenchModel)
    assert isinstance(model.bands, list)
    assert all(isinstance(b, BandSpec) for b in model.bands), (
        f"BandSpec identity lost; got types {[type(b).__name__ for b in model.bands]}"
    )
    assert model.num_channels == len(bands)
