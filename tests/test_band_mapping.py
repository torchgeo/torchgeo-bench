"""Tests for the BandSpec -> model-band mapping helper."""

import torch

from torchgeo_bench.datasets.base import BandSpec
from torchgeo_bench.models._band_mapping import (
    canonical_band_name,
    map_to_model_bands,
    wavelengths_um,
)


def _band(name: str, wl: float | None = None) -> BandSpec:
    return BandSpec(
        sensor="s2",
        name=name,
        source_name=name,
        mean=0.0,
        std=1.0,
        min=0.0,
        max=1.0,
        wavelength_um=wl,
    )


class TestCanonicalBandName:
    def test_aliases(self) -> None:
        assert canonical_band_name("Red") == "red"
        assert canonical_band_name("B04") == "red"
        assert canonical_band_name("04") == "red"
        assert canonical_band_name("04 - Red") == "red"
        assert canonical_band_name("blue") == "blue"
        assert canonical_band_name("B02") == "blue"
        assert canonical_band_name("nir") == "nir"
        assert canonical_band_name("B8A") == "nir_narrow"
        assert canonical_band_name("VV") == "vv"

    def test_geobench_v1_aliases(self) -> None:
        # GeoBench V1 datasets use these long-form names; ensure they resolve
        # to canonical short names so band-mapping doesn't silently zero-fill.
        assert canonical_band_name("coastal_aerosol") == "coastal"
        assert canonical_band_name("red_edge_1") == "rededge1"
        assert canonical_band_name("red_edge_2") == "rededge2"
        assert canonical_band_name("red_edge_3") == "rededge3"
        assert canonical_band_name("red_edge_4") == "nir_narrow"
        assert canonical_band_name("water_vapour") == "watervapor"
        assert canonical_band_name("water_vapor") == "watervapor"
        assert canonical_band_name("swir_cirrus") == "cirrus"
        assert canonical_band_name("swir_1") == "swir1"
        assert canonical_band_name("swir_2") == "swir2"

    def test_unknown_falls_through(self) -> None:
        assert canonical_band_name("custom_xyz") == "custom_xyz"


class TestMapToModelBands:
    def test_rgb_to_six_band_zerofills(self) -> None:
        src = [_band("red"), _band("green"), _band("blue")]
        x = torch.arange(3 * 4 * 4, dtype=torch.float32).reshape(1, 3, 4, 4)
        target = ["blue", "green", "red", "nir_narrow", "swir1", "swir2"]
        out, missing = map_to_model_bands(x, src, target)
        assert out.shape == (1, 6, 4, 4)
        # red came from src[0], green from src[1], blue from src[2]
        assert torch.equal(out[:, 0], x[:, 2])  # blue
        assert torch.equal(out[:, 1], x[:, 1])  # green
        assert torch.equal(out[:, 2], x[:, 0])  # red
        # nir_narrow / swir1 / swir2 missing -> zero
        assert torch.equal(out[:, 3], torch.zeros(1, 4, 4))
        assert missing == [False, False, False, True, True, True]

    def test_alias_resolution(self) -> None:
        src = [_band("B04"), _band("B03"), _band("B02")]
        x = torch.zeros(2, 3, 2, 2)
        x[:, 0] = 7  # B04 == red
        target = ["red", "green", "blue"]
        out, missing = map_to_model_bands(x, src, target)
        assert torch.equal(out[:, 0], x[:, 0])
        assert missing == [False, False, False]

    def test_channel_count_mismatch_raises(self) -> None:
        src = [_band("red"), _band("green")]
        x = torch.zeros(1, 3, 4, 4)
        try:
            map_to_model_bands(x, src, ["red"])
        except ValueError:
            return
        raise AssertionError("expected ValueError for channel-count mismatch")


class TestWavelengthsUm:
    def test_default_fill(self) -> None:
        bands = [_band("red", 0.665), _band("vv", None)]
        wls = wavelengths_um(bands, default_um=1.5)
        assert wls == [0.665, 1.5]
