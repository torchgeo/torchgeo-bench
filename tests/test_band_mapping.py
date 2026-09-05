"""Tests for the BandSpec -> model-band mapping helper."""

import pytest
import torch

from torchgeo_bench.datasets.base import BandSpec
from torchgeo_bench.models._band_mapping import (
    S2_WAVELENGTHS_UM,
    BandMappingPolicy,
    canonical_band_name,
    map_to_model_bands,
    select_src_bands,
    wavelengths_um,
)
from torchgeo_bench.models.torchgeo_models import _resolve_dofa_wavelengths


def _band(name: str, wl: float | None = None, sensor: str = "s2") -> BandSpec:
    return BandSpec(
        sensor=sensor,
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
    def test_rgb_to_six_band_missing_raises_by_default(self) -> None:
        src = [_band("red"), _band("green"), _band("blue")]
        x = torch.arange(3 * 4 * 4, dtype=torch.float32).reshape(1, 3, 4, 4)
        target = ["blue", "green", "red", "nir_narrow", "swir1", "swir2"]
        with pytest.raises(ValueError, match="Missing required model band"):
            map_to_model_bands(x, src, target)

    def test_rgb_to_six_band_zerofills_when_explicitly_allowed(self) -> None:
        src = [_band("red"), _band("green"), _band("blue")]
        x = torch.arange(3 * 4 * 4, dtype=torch.float32).reshape(1, 3, 4, 4)
        target = ["blue", "green", "red", "nir_narrow", "swir1", "swir2"]
        out, missing = map_to_model_bands(
            x, src, target, policy=BandMappingPolicy(allow_missing=True)
        )
        assert out.shape == (1, 6, 4, 4)
        # red came from src[0], green from src[1], blue from src[2]
        assert torch.equal(out[:, 0], x[:, 2])  # blue
        assert torch.equal(out[:, 1], x[:, 1])  # green
        assert torch.equal(out[:, 2], x[:, 0])  # red
        # nir_narrow / swir1 / swir2 missing -> zero
        assert torch.equal(out[:, 3], torch.zeros(1, 4, 4))
        assert missing == [False, False, False, True, True, True]

    def test_missing_coastal_falls_back_to_blue_by_default(self) -> None:
        """A dataset with no coastal-aerosol band (most GeoBench S2 datasets)
        must not hard-fail a coastal-requiring model (CROMA) -- blue is the
        spectrally nearest available band, so substitute it instead of
        zero-filling or raising."""
        src = [_band("red"), _band("green"), _band("blue")]
        x = torch.arange(3 * 4 * 4, dtype=torch.float32).reshape(1, 3, 4, 4)
        out, missing = map_to_model_bands(x, src, ["coastal", "blue"])
        assert torch.equal(out[:, 0], x[:, 2])  # coastal <- blue (src[2])
        assert torch.equal(out[:, 1], x[:, 2])  # blue <- blue
        assert missing == [False, False]  # fallback-filled, not zero-filled

    def test_band_fallbacks_can_be_disabled(self) -> None:
        src = [_band("red"), _band("green"), _band("blue")]
        x = torch.zeros(1, 3, 4, 4)
        with pytest.raises(ValueError, match="Missing required model band"):
            map_to_model_bands(x, src, ["coastal"], policy=BandMappingPolicy(band_fallbacks={}))

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
        with pytest.raises(ValueError, match="images has 3 channels"):
            map_to_model_bands(x, src, ["red"])

    def test_preferred_sensor_wins_slot(self) -> None:
        src = [_band("red", sensor="aerial"), _band("B04")]
        x = torch.zeros(1, 2, 2, 2)
        x[:, 0] = 1.0
        x[:, 1] = 2.0
        out, _ = map_to_model_bands(
            x, src, ["red"], policy=BandMappingPolicy(preferred_sensors=("s2",))
        )
        assert torch.equal(out[:, 0], x[:, 1])


class TestSelectSrcBands:
    def test_full_match_preserves_target_order(self) -> None:
        src = [_band("B02"), _band("B03"), _band("B04")]
        indices, selected = select_src_bands(src, ["red", "green", "blue"])
        assert selected == ["red", "green", "blue"]
        assert indices == [2, 1, 0]

    def test_partial_match_drops_missing_targets(self) -> None:
        src = [_band("red"), _band("green"), _band("blue")]
        target = ["coastal", "blue", "green", "red", "nir"]
        indices, selected = select_src_bands(src, target)
        assert selected == ["blue", "green", "red"]
        assert indices == [2, 1, 0]

    def test_duplicate_canonical_name_prefers_sensor(self) -> None:
        src = [
            _band("red", sensor="aerial"),
            _band("green", sensor="aerial"),
            _band("blue", sensor="aerial"),
            _band("nir", sensor="aerial"),
            _band("B02"),
            _band("B03"),
            _band("B04"),
            _band("B08"),
        ]
        indices, selected = select_src_bands(
            src, ["blue", "green", "red", "nir"], preferred_sensors=("s2",)
        )
        assert selected == ["blue", "green", "red", "nir"]
        assert indices == [4, 5, 6, 7]

    def test_duplicate_without_preference_keeps_first_occurrence(self) -> None:
        src = [_band("red", sensor="aerial"), _band("B04")]
        indices, _ = select_src_bands(src, ["red"])
        assert indices == [0]

    def test_no_match_raises(self) -> None:
        src = [_band("vv", sensor="sar"), _band("vh", sensor="sar")]
        with pytest.raises(ValueError, match="none of the target bands"):
            select_src_bands(src, ["red", "green", "blue"])


class TestWavelengthsUm:
    def test_missing_raises_by_default(self) -> None:
        bands = [_band("red", 0.665), _band("vv", None)]
        with pytest.raises(ValueError, match="Missing wavelengths"):
            wavelengths_um(bands)

    def test_default_fill_when_explicitly_provided(self) -> None:
        bands = [_band("red", 0.665), _band("vv", None)]
        wls = wavelengths_um(bands, default_um=1.5)
        assert wls == [0.665, 1.5]

    def test_falls_back_to_s2_wavelength_by_canonical_name(self) -> None:
        """A Landsat dataset's nir/swir bands with no declared wavelength_um
        (m_forestnet.py) must not hard-fail -- most datasets here are
        Sentinel-2, so its true wavelength is a reasonable default."""
        bands = [
            _band("nir", None, sensor="landsat"),
            _band("swir_1", None, sensor="landsat"),
            _band("swir_2", None, sensor="landsat"),
        ]
        assert wavelengths_um(bands) == [
            S2_WAVELENGTHS_UM["nir"],
            S2_WAVELENGTHS_UM["swir1"],
            S2_WAVELENGTHS_UM["swir2"],
        ]

    def test_sar_still_raises_without_s2_default_or_explicit_default(self) -> None:
        """SAR has no canonical S2 wavelength -- must still raise, not
        silently invent an optical wavelength for radar backscatter."""
        bands = [_band("red", 0.665), _band("vv", None, sensor="sar")]
        with pytest.raises(ValueError, match="Missing wavelengths"):
            wavelengths_um(bands)


class TestDofaWavelengths:
    def test_derived_from_selected_bands(self) -> None:
        bands = [_band("red", 0.665), _band("green", 0.56), _band("nir", 0.842)]
        assert _resolve_dofa_wavelengths(bands, None) == [0.665, 0.56, 0.842]

    def test_manual_override_length_must_match_channels(self) -> None:
        bands = [_band("red", 0.665), _band("green", 0.56)]
        with pytest.raises(ValueError, match="wavelengths"):
            _resolve_dofa_wavelengths(bands, [0.665, 0.56, 0.49])
