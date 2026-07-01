"""Smoke tests for :class:`OlmoEarthBenchModel`.

The full GeoBench v1+v2 sweep originally couldn't run OlmoEarth because
the ``[olmoearth]`` extra wasn't installed.  These tests both prevent
that regression (by importing the wrapper and checking each variant
loads) and validate the wrapper actually produces sensible embeddings.
"""

from importlib.util import find_spec
from unittest import mock

import pytest
import torch

from torchgeo_bench.datasets.base import BandSpec

olmoearth_available = find_spec("olmoearth_pretrain_minimal") is not None
requires_olmoearth = pytest.mark.skipif(
    not olmoearth_available,
    reason="olmoearth-pretrain-minimal not installed (pip install 'torchgeo-bench[olmoearth]')",
)


def _rgb_bands() -> list[BandSpec]:
    return [
        BandSpec(
            sensor="s2",
            name=n,
            source_name=n.upper(),
            mean=1500.0,
            std=600.0,
            min=0.0,
            max=10000.0,
        )
        for n in ("red", "green", "blue")
    ]


def _s2_bands() -> list[BandSpec]:
    from torchgeo_bench.models.olmoearth import OLMOEARTH_S2_BANDS

    return [
        BandSpec(
            sensor="s2",
            name=b.lower(),
            source_name=b,
            mean=1500.0,
            std=600.0,
            min=0.0,
            max=10000.0,
        )
        for b in OLMOEARTH_S2_BANDS
    ]


# Map variant -> expected embedding dim (from the HF weights configs).
EXPECTED_DIM = {"nano": 128, "tiny": 192, "small": 384, "base": 768, "large": 1024}


@requires_olmoearth
@pytest.mark.parametrize("size", ["nano", "tiny"])  # base/large are too heavy for CI
def test_rgb_forward_pass_shape(size: str) -> None:
    """All-RGB input must produce a 2-D embedding of the expected width."""
    from torchgeo_bench.models.olmoearth import OlmoEarthBenchModel

    model = OlmoEarthBenchModel(bands=_rgb_bands(), model_size=size, normalization="identity")
    model.eval()
    x = torch.rand(2, 3, 64, 64) * 3000.0  # raw S2-like values
    out = model.forward_patch_features(x)
    assert out.shape == (2, EXPECTED_DIM[size])
    assert torch.isfinite(out).all()


@requires_olmoearth
def test_s2_forward_pass_shape() -> None:
    """12-channel S2 input goes through the multispectral path."""
    from torchgeo_bench.models.olmoearth import OlmoEarthBenchModel

    model = OlmoEarthBenchModel(bands=_s2_bands(), model_size="nano", normalization="identity")
    model.eval()
    x = torch.rand(2, 12, 64, 64) * 3000.0
    out = model.forward_patch_features(x)
    assert out.shape == (2, EXPECTED_DIM["nano"])
    assert torch.isfinite(out).all()


@requires_olmoearth
def test_all_variants_are_loadable() -> None:
    """ModelID enum exposes all variants used by this benchmark.

    Prevents the regression where the wrapper silently lost a variant
    after an upstream rename.
    """
    from olmoearth_pretrain_minimal import ModelID

    names = {attr for attr in dir(ModelID) if attr.startswith("OLMOEARTH_V1_")}
    assert names == {
        "OLMOEARTH_V1_NANO",
        "OLMOEARTH_V1_TINY",
        "OLMOEARTH_V1_BASE",
        "OLMOEARTH_V1_LARGE",
        "OLMOEARTH_V1_1_NANO",
        "OLMOEARTH_V1_1_TINY",
        "OLMOEARTH_V1_1_BASE",
        "OLMOEARTH_V1_2_NANO",
        "OLMOEARTH_V1_2_TINY",
        "OLMOEARTH_V1_2_SMALL",
        "OLMOEARTH_V1_2_BASE",
    }


@requires_olmoearth
def test_reflectance_input_is_rescaled_to_dn() -> None:
    """Datasets like m-so2sat / so2sat deliver S2 reflectance in [0, ~2.8],
    not raw DN.  The wrapper must detect this and rescale to DN before
    OlmoEarth's Normalizer sees the values — otherwise the normalizer's
    DN-fitted mean/std produce near-zero normalized inputs and embeddings
    collapse.

    We verify the scale-detection path picks ``REFLECTANCE_0_1`` for
    so2sat-style band stats and that the forward pass produces non-degenerate
    embeddings.
    """
    from torchgeo_bench.models._input_units import InputUnit
    from torchgeo_bench.models.olmoearth import OlmoEarthBenchModel

    # so2sat-style band stats: optical reflectance with max ~2.8
    refl_bands = [
        BandSpec(
            sensor="s2",
            name=n,
            source_name=n.upper(),
            mean=0.13,
            std=0.07,
            min=0.0001,
            max=2.8,
            wavelength_um=0.5,
        )
        for n in ("red", "green", "blue")
    ]
    model = OlmoEarthBenchModel(bands=refl_bands, model_size="nano", normalization="identity")
    assert model._sensor_groups[0]["input_unit"] == InputUnit.REFLECTANCE_0_1
    model.eval()
    x = torch.rand(2, 3, 64, 64) * 2.5  # reflectance-like values
    out = model.forward_patch_features(x)
    assert out.shape == (2, EXPECTED_DIM["nano"])
    assert torch.isfinite(out).all()
    # Embeddings should have non-trivial variance — collapsed-to-zero
    # embeddings would have std ~ 0.
    assert out.std() > 1e-4


@requires_olmoearth
def test_rejects_unknown_sensor() -> None:
    """A BandSpec with a sensor name we have no OlmoEarth layout for must
    fail loudly."""
    from torchgeo_bench.models.olmoearth import OlmoEarthBenchModel

    weird_bands = [
        BandSpec(
            sensor="totally_unknown_sensor",
            name="band1",
            source_name="BAND1",
            mean=1500.0,
            std=600.0,
            min=0.0,
            max=10000.0,
        )
    ]
    with pytest.raises(ValueError, match="no layout for sensor"):
        OlmoEarthBenchModel(bands=weird_bands, model_size="nano", normalization="identity")


@requires_olmoearth
def test_rejects_unknown_band_name() -> None:
    """A BandSpec name we have no OlmoEarth-position mapping for must fail
    loudly so we don't quietly zero-fill every channel."""
    from torchgeo_bench.models.olmoearth import OlmoEarthBenchModel

    weird_bands = [
        BandSpec(
            sensor="s2",
            name="totally_made_up",
            source_name="MADE_UP",
            mean=1500.0,
            std=600.0,
            min=0.0,
            max=10000.0,
            wavelength_um=0.5,
        )
    ]
    with pytest.raises(ValueError, match="can't map BandSpec names"):
        OlmoEarthBenchModel(bands=weird_bands, model_size="nano", normalization="identity")


@requires_olmoearth
def test_landsat_modality_routing() -> None:
    """Landsat input picks Modality.LANDSAT, not SENTINEL2_L2A.  The mask
    should have 2 band-sets, the sample field should be 'landsat'.
    input_res must auto-detect to 30 m."""
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import Modality

    from torchgeo_bench.models.olmoearth import OlmoEarthBenchModel

    names = ("blue", "green", "red", "nir", "swir_1", "swir_2")
    landsat_bands = [
        BandSpec(
            sensor="landsat",
            name=n,
            source_name=n.upper(),
            mean=80.0,
            std=20.0,
            min=0.0,
            max=255.0,
        )
        for n in names
    ]
    model = OlmoEarthBenchModel(bands=landsat_bands, model_size="nano", normalization="identity")
    g = model._sensor_groups[0]
    assert g["modality"] == Modality.LANDSAT
    assert g["sample_field"] == "landsat"
    assert g["channels"] == 11
    assert g["num_band_sets"] == 2
    assert model.input_res == 30
    model.eval()
    x = torch.rand(2, 6, 64, 64) * 200.0  # uint8-ish Landsat
    out = model.forward_patch_features(x)
    assert out.shape == (2, EXPECTED_DIM["nano"])
    assert torch.isfinite(out).all()


@requires_olmoearth
def test_aerial_falls_back_to_s2() -> None:
    """olmoearth-pretrain-minimal's encoder doesn't ship a NAIP branch,
    so aerial-RGB datasets (m-pv4ger, treesatai aerial) have to route
    through the S2 modality with non-RGB S2 positions zero-filled.
    """
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import Modality

    from torchgeo_bench.models.olmoearth import OlmoEarthBenchModel

    naip_bands = [
        BandSpec(
            sensor="aerial",
            name=n,
            source_name=n.capitalize(),
            mean=120.0,
            std=40.0,
            min=0.0,
            max=255.0,
        )
        for n in ("red", "green", "blue")
    ]
    model = OlmoEarthBenchModel(bands=naip_bands, model_size="nano", normalization="identity")
    g = model._sensor_groups[0]
    assert g["modality"] == Modality.SENTINEL2_L2A
    assert g["sample_field"] == "sentinel2_l2a"
    assert g["channels"] == 12
    # red -> B04 (idx 2), green -> B03 (idx 1), blue -> B02 (idx 0)
    assert g["dst_indices"] == [2, 1, 0]
    model.eval()
    x = torch.rand(2, 3, 64, 64) * 200.0
    out = model.forward_patch_features(x)
    assert out.shape == (2, EXPECTED_DIM["nano"])
    assert torch.isfinite(out).all()


@requires_olmoearth
def test_partial_s2_10band_forward_pass() -> None:
    """10-band S2 input (m-so2sat-style, no B01/B09) routes through the
    S2 modality; B01/B09 are imputed from the nearest present band
    (blue / B8A) — matching helios' m-so2sat imputes — not zero-filled."""
    from torchgeo_bench.models.olmoearth import OlmoEarthBenchModel

    names = ["b02", "b03", "b04", "b08", "b05", "b06", "b07", "b8a", "b11", "b12"]
    bands = [
        BandSpec(
            sensor="s2",
            name=n,
            source_name=n.upper(),
            mean=1500.0,
            std=600.0,
            min=0.0,
            max=10000.0,
        )
        for n in names
    ]
    model = OlmoEarthBenchModel(bands=bands, model_size="nano", normalization="identity")
    g = model._sensor_groups[0]
    model.eval()
    assert g["channels"] == 12
    assert g["num_band_sets"] == 3
    # B02..B12 map to positions 0..9; B01/B09 (positions 10/11) are absent.
    assert set(g["dst_indices"]) == set(range(10))
    # B01 coastal (10) <- B02 blue (0); B09 water vapour (11) <- B8A (7).
    assert g["impute_ops"] == [(0, 10), (7, 11)]
    x = torch.rand(2, 10, 64, 64) * 3000.0
    out = model.forward_patch_features(x)
    assert out.shape == (2, EXPECTED_DIM["nano"])
    assert torch.isfinite(out).all()


@requires_olmoearth
def test_forestnet_landsat_imputes_missing_bands() -> None:
    """m-forestnet ships only 6 of 11 Landsat channels.  The 5 missing
    OlmoEarth positions must be imputed from the most spectrally similar
    present band (matching helios) rather than left zero-filled."""
    from torchgeo_bench.models.olmoearth import OlmoEarthBenchModel

    names = ("blue", "green", "red", "nir", "swir_1", "swir_2")
    landsat_bands = [
        BandSpec(
            sensor="landsat",
            name=n,
            source_name=n.upper(),
            mean=80.0,
            std=20.0,
            min=0.0,
            max=255.0,
        )
        for n in names
    ]
    model = OlmoEarthBenchModel(bands=landsat_bands, model_size="nano", normalization="identity")
    g = model._sensor_groups[0]
    # pan<-green(3), coastal<-blue(2), cirrus/tirs1/tirs2<-swir2(7); each
    # source channel (3, 2, 7) is one of the present bands.
    assert g["impute_ops"] == [(3, 0), (2, 1), (7, 8), (7, 9), (7, 10)]
    assert all(src in set(g["dst_indices"]) for src, _ in g["impute_ops"])
    model.eval()
    x = torch.rand(2, 6, 64, 64) * 200.0
    out = model.forward_patch_features(x)
    assert out.shape == (2, EXPECTED_DIM["nano"])
    assert torch.isfinite(out).all()


@requires_olmoearth
def test_landsat_dataset_stats_normalization() -> None:
    """norm_from_pretrained=False normalizes each band with its BandSpec stats
    (helios-style ±2σ no-clip), bypassing the DN rescale + pretrained
    Normalizer — required for GeoBench's uint8 Landsat scale."""
    from torchgeo_bench.models.olmoearth import OlmoEarthBenchModel

    names = ("blue", "green", "red", "nir", "swir_1", "swir_2")
    bands = [
        BandSpec(
            sensor="landsat",
            name=n,
            source_name=n.upper(),
            mean=80.0,
            std=20.0,
            min=0.0,
            max=255.0,
        )
        for n in names
    ]
    model = OlmoEarthBenchModel(
        bands=bands, model_size="nano", normalization="identity", norm_from_pretrained=False
    )
    assert model.norm_from_pretrained is False
    g = model._sensor_groups[0]
    assert len(g["src_means"]) == 6 and len(g["src_stds"]) == 6
    model.eval()
    x = torch.rand(2, 6, 64, 64) * 200.0  # uint8-scale Landsat
    out = model.forward_patch_features(x)
    assert out.shape == (2, EXPECTED_DIM["nano"])
    assert torch.isfinite(out).all()


@requires_olmoearth
def test_auto_normalization_default_per_sensor() -> None:
    """Default norm_from_pretrained='auto' routes Landsat to dataset stats and
    S2 to the pretrained normalizer, so one shared config is correct for both.
    Both must produce finite embeddings without an explicit override."""
    from torchgeo_bench.models._input_units import InputUnit
    from torchgeo_bench.models.olmoearth import _DATASET_STATS_SENSORS, OlmoEarthBenchModel

    # Landsat (uint8) — 'auto' should pick dataset stats.
    ls = [
        BandSpec(sensor="landsat", name=n, source_name=n.upper(), mean=80.0, std=20.0, min=0.0, max=255.0)
        for n in ("blue", "green", "red", "nir", "swir_1", "swir_2")
    ]
    ls_model = OlmoEarthBenchModel(bands=ls, model_size="nano", normalization="identity")
    assert ls_model.norm_from_pretrained == "auto"  # default
    assert ls_model._sensor_groups[0]["sensor"] in _DATASET_STATS_SENSORS
    ls_model.eval()
    ls_out = ls_model.forward_patch_features(torch.rand(2, 6, 64, 64) * 200.0)
    assert ls_out.shape == (2, EXPECTED_DIM["nano"]) and torch.isfinite(ls_out).all()

    # S2 (DN) — 'auto' should keep the pretrained normalizer (rescale to DN).
    s2 = [
        BandSpec(sensor="s2", name=n, source_name=n.upper(), mean=1500.0, std=600.0, min=0.0, max=10000.0)
        for n in ("red", "green", "blue")
    ]
    s2_model = OlmoEarthBenchModel(bands=s2, model_size="nano", normalization="identity")
    assert s2_model._sensor_groups[0]["sensor"] not in _DATASET_STATS_SENSORS
    s2_model.eval()
    s2_out = s2_model.forward_patch_features(torch.rand(2, 3, 64, 64) * 3000.0)
    assert s2_out.shape == (2, EXPECTED_DIM["nano"]) and torch.isfinite(s2_out).all()
    # sanity: input-unit detection still runs on the S2 (pretrained) path
    assert s2_model._sensor_groups[0]["input_unit"] == InputUnit.S2_DN


@requires_olmoearth
@pytest.mark.parametrize("size", ["nano", "small"])
def test_v1_2_variants_forward_pass(size: str) -> None:
    """OlmoEarth v1.2 (Nano/Tiny/Small/Base) must load and run; Small is the
    new 384-d size introduced in v1.2."""
    from torchgeo_bench.models.olmoearth import OlmoEarthBenchModel

    model = OlmoEarthBenchModel(
        bands=_rgb_bands(), model_size=size, version="v1_2", normalization="identity"
    )
    model.eval()
    x = torch.rand(2, 3, 64, 64) * 3000.0
    out = model.forward_patch_features(x)
    assert out.shape == (2, EXPECTED_DIM[size])
    assert torch.isfinite(out).all()


@requires_olmoearth
def test_mixed_s2_sar_forward_pass() -> None:
    """Mixed S2 + SAR input (m-so2sat) routes to two separate modalities:
    SENTINEL2_L2A and SENTINEL1.  Both sample fields are populated in
    MaskedOlmoEarthSample simultaneously."""
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import Modality

    from torchgeo_bench.models.olmoearth import OlmoEarthBenchModel

    # Minimal m-so2sat-style: 3 S2 (reflectance) + 2 SAR (Lee-filtered).
    mixed_bands = [
        BandSpec(
            sensor="s2",
            name="blue",
            source_name="B02",
            mean=0.13,
            std=0.05,
            min=0.0001,
            max=2.8,
            wavelength_um=0.49,
        ),
        BandSpec(
            sensor="s2",
            name="green",
            source_name="B03",
            mean=0.12,
            std=0.05,
            min=0.0001,
            max=2.8,
            wavelength_um=0.56,
        ),
        BandSpec(
            sensor="s2",
            name="red",
            source_name="B04",
            mean=0.11,
            std=0.07,
            min=0.0001,
            max=2.8,
            wavelength_um=0.665,
        ),
        BandSpec(
            sensor="sar",
            name="vv_lee",
            source_name="VV_LEE",
            mean=0.34,
            std=11.8,
            min=0.0,
            max=9950.0,
        ),
        BandSpec(
            sensor="sar",
            name="vh_lee",
            source_name="VH_LEE",
            mean=0.06,
            std=5.4,
            min=0.0,
            max=10867.0,
        ),
    ]
    model = OlmoEarthBenchModel(bands=mixed_bands, model_size="nano", normalization="identity")
    assert len(model._sensor_groups) == 2
    s2_group = next(g for g in model._sensor_groups if g["sensor"] == "s2")
    sar_group = next(g for g in model._sensor_groups if g["sensor"] == "sar")
    assert s2_group["modality"] == Modality.SENTINEL2_L2A
    assert sar_group["modality"] == Modality.SENTINEL1
    assert sar_group["sample_field"] == "sentinel1"
    assert sar_group["channels"] == 2
    # SAR bands are passthrough — no s2-DN rescaling.
    assert sar_group["input_unit"] is None
    # S2/SAR coregistered to 10 m grid.
    assert model.input_res == 10
    model.eval()
    x = torch.rand(2, 5, 64, 64)
    x[:, :3] *= 2.5  # S2 reflectance scale
    x[:, 3:] *= 5000  # SAR Lee-filtered scale
    out = model.forward_patch_features(x)
    assert out.shape == (2, EXPECTED_DIM["nano"])
    assert torch.isfinite(out).all()


@requires_olmoearth
def test_invalid_model_size_raises() -> None:
    """Unknown model_size values must fail during constructor model-id resolution."""
    from torchgeo_bench.models.olmoearth import OlmoEarthBenchModel

    with pytest.raises(AttributeError, match="OLMOEARTH_V1_XLARGE"):
        OlmoEarthBenchModel(bands=_rgb_bands(), model_size="xlarge", normalization="identity")


@requires_olmoearth
def test_invalid_model_size_at_construction_not_forward() -> None:
    """Invalid model_size must fail in __init__ before any model loading call."""
    import olmoearth_pretrain_minimal as oepm

    from torchgeo_bench.models.olmoearth import OlmoEarthBenchModel

    with (
        mock.patch.object(
            oepm,
            "load_model_from_id",
            side_effect=AssertionError("load_model_from_id should not be called"),
        ),
        pytest.raises(AttributeError, match="OLMOEARTH_V1_XLARGE"),
    ):
        OlmoEarthBenchModel(bands=_rgb_bands(), model_size="xlarge", normalization="identity")
