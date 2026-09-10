"""Unit test for the `instantiate(cfg.model, bands=...)` contract."""

from torchgeo_bench.config import compose_config, instantiate
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
    """Kwargs bypass config plumbing, so BandSpec dataclasses reach the constructor intact."""
    cfg = compose_config(model="rcf")

    bands = _bands()
    model = instantiate(cfg.model, bands=bands)

    assert isinstance(model, BenchModel)
    assert isinstance(model.bands, list)
    assert all(isinstance(b, BandSpec) for b in model.bands), (
        f"BandSpec identity lost; got types {[type(b).__name__ for b in model.bands]}"
    )
    assert model.num_channels == len(bands)
