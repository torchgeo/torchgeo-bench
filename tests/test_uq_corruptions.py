import pytest
import torch

from torchgeo_bench.datasets.base import BandSpec
from torchgeo_bench.uq.corruptions import CorruptionTransform, SKIP_POISSON_GAUSSIAN
from torchgeo_bench.uq.viz_corruptions import generate_grid


def _require_cloud_dependency() -> None:
    pytest.importorskip("satellite_cloud_generator")


def _bands(names: tuple[str, ...] = ("red", "green", "blue")) -> list[BandSpec]:
    return [
        BandSpec(
            sensor="s2",
            name=name,
            source_name=name.upper(),
            mean=float(100 + i * 5),
            std=float(10 + i),
            min=0.0,
            max=255.0,
            wavelength_um=float(0.4 + i * 0.1),
        )
        for i, name in enumerate(names)
    ]


def _mixed_sensor_bands() -> list[BandSpec]:
    return [
        BandSpec(
            sensor="s2",
            name="red",
            source_name="B04",
            mean=120.0,
            std=12.0,
            min=0.0,
            max=255.0,
            wavelength_um=0.665,
        ),
        BandSpec(
            sensor="s2",
            name="green",
            source_name="B03",
            mean=110.0,
            std=11.0,
            min=0.0,
            max=255.0,
            wavelength_um=0.560,
        ),
        BandSpec(
            sensor="s2",
            name="blue",
            source_name="B02",
            mean=95.0,
            std=10.0,
            min=0.0,
            max=255.0,
            wavelength_um=0.490,
        ),
        BandSpec(
            sensor="sar",
            name="vv",
            source_name="VV",
            mean=-12.0,
            std=2.0,
            min=-30.0,
            max=3.0,
            wavelength_um=None,
        ),
        BandSpec(
            sensor="sar",
            name="vh",
            source_name="VH",
            mean=-17.0,
            std=2.5,
            min=-35.0,
            max=0.0,
            wavelength_um=None,
        ),
    ]


def test_cloud_output_shape():
    _require_cloud_dependency()
    x = torch.rand((2, 3, 16, 16), dtype=torch.float32) * 255.0
    tfm = CorruptionTransform("cloud", severity=1, seed=11, band_specs=_bands(), dataset_name="m-eurosat")
    y = tfm(x)
    assert y.shape == x.shape


def test_cloud_dtype_preserved():
    _require_cloud_dependency()
    x = torch.rand((2, 3, 16, 16), dtype=torch.float16) * 255.0
    tfm = CorruptionTransform("cloud", severity=1, seed=11, band_specs=_bands(), dataset_name="m-eurosat")
    y = tfm(x)
    assert y.dtype == x.dtype


def test_cloud_values_clamped():
    _require_cloud_dependency()
    x = torch.rand((2, 3, 16, 16), dtype=torch.float32) * 512.0
    tfm = CorruptionTransform("cloud", severity=1, seed=11, band_specs=_bands(), dataset_name="m-eurosat")
    y = tfm(x)
    assert float(y.min()) >= 0.0
    assert float(y.max()) <= 255.0


def test_cloud_determinism():
    _require_cloud_dependency()
    x = torch.rand((2, 3, 16, 16), dtype=torch.float32) * 255.0
    tfm_a = CorruptionTransform("cloud", severity=3, seed=17, band_specs=_bands(), dataset_name="m-eurosat")
    tfm_b = CorruptionTransform("cloud", severity=3, seed=17, band_specs=_bands(), dataset_name="m-eurosat")
    y_a = tfm_a(x)
    y_b = tfm_b(x)
    assert torch.allclose(y_a, y_b)


def test_cloud_counter_increments():
    _require_cloud_dependency()
    x1 = torch.rand((4, 3, 16, 16), dtype=torch.float32) * 255.0
    x2 = torch.rand((2, 3, 16, 16), dtype=torch.float32) * 255.0
    tfm = CorruptionTransform("cloud", severity=2, seed=31, band_specs=_bands(), dataset_name="m-eurosat")
    _ = tfm(x1)
    assert tfm._n_images_seen == 4
    y2 = tfm(x2)

    tfm_reset = CorruptionTransform("cloud", severity=2, seed=31, band_specs=_bands(), dataset_name="m-eurosat")
    y2_reset = tfm_reset(x2)
    assert not torch.allclose(y2, y2_reset)


def test_cloud_mixed_sensor_keeps_non_optical_channels():
    _require_cloud_dependency()
    band_specs = _mixed_sensor_bands()
    x = torch.rand((2, len(band_specs), 16, 16), dtype=torch.float32)
    x[:, 0:3] *= 255.0
    x[:, 3] = -25.0 + 20.0 * x[:, 3]
    x[:, 4] = -30.0 + 25.0 * x[:, 4]

    tfm = CorruptionTransform(
        "cloud",
        severity=4,
        seed=23,
        band_specs=band_specs,
        dataset_name="m-eurosat",
    )
    y = tfm(x)
    assert torch.allclose(x[:, 3:], y[:, 3:])
    assert not torch.allclose(x[:, :3], y[:, :3])


def test_cloud_severity_changes_statistics():
    _require_cloud_dependency()
    x = torch.rand((1, 3, 32, 32), dtype=torch.float32) * 255.0
    tfm_low = CorruptionTransform("cloud", severity=1, seed=91, band_specs=_bands(), dataset_name="m-eurosat")
    tfm_high = CorruptionTransform(
        "cloud",
        severity=5,
        seed=91,
        band_specs=_bands(),
        dataset_name="m-eurosat",
    )
    y_low = tfm_low(x)
    y_high = tfm_high(x)

    delta_low = torch.mean(torch.abs(y_low - x))
    delta_high = torch.mean(torch.abs(y_high - x))
    assert float(delta_high) > float(delta_low)


def test_cloud_missing_calibration_raises():
    _require_cloud_dependency()
    x = torch.rand((1, 3, 16, 16), dtype=torch.float32) * 255.0
    tfm = CorruptionTransform(
        "cloud",
        severity=1,
        seed=7,
        band_specs=_bands(),
        dataset_name="unknown-dataset",
    )
    with pytest.raises(ValueError, match="No cloud calibration"):
        _ = tfm(x)


def test_cloud_pattern_mode_validation():
    with pytest.raises(ValueError, match="cloud_pattern_mode"):
        _ = CorruptionTransform(
            "cloud",
            severity=1,
            seed=7,
            band_specs=_bands(),
            dataset_name="m-eurosat",
            cloud_pattern_mode="bad_mode",
        )


def test_cloud_pattern_mode_changes_realization():
    _require_cloud_dependency()
    x = torch.rand((2, 3, 16, 16), dtype=torch.float32) * 255.0
    tfm_fixed = CorruptionTransform(
        "cloud",
        severity=3,
        seed=17,
        band_specs=_bands(),
        dataset_name="m-eurosat",
        cloud_pattern_mode="fixed_across_severity",
    )
    tfm_independent = CorruptionTransform(
        "cloud",
        severity=3,
        seed=17,
        band_specs=_bands(),
        dataset_name="m-eurosat",
        cloud_pattern_mode="independent_per_severity",
    )
    y_fixed = tfm_fixed(x)
    y_independent = tfm_independent(x)
    assert not torch.allclose(y_fixed, y_independent)


def test_poisson_gaussian_output_shape():
    x = torch.rand((2, 3, 16, 16), dtype=torch.float32) * 255.0
    tfm = CorruptionTransform("poisson_gaussian", severity=1, seed=21, band_specs=_bands())
    y = tfm(x)
    assert y.shape == x.shape


def test_poisson_gaussian_values_clamped():
    x = torch.rand((2, 3, 16, 16), dtype=torch.float32) * 255.0
    tfm = CorruptionTransform("poisson_gaussian", severity=1, seed=21, band_specs=_bands())
    y = tfm(x)
    assert float(y.min()) >= 0.0
    assert float(y.max()) <= 255.0


def test_skip_poisson_gaussian_constant():
    assert "m-so2sat" in SKIP_POISSON_GAUSSIAN
    assert "so2sat" in SKIP_POISSON_GAUSSIAN


def test_viz_corruptions_runs(tmp_path):
    _require_cloud_dependency()
    _ = __import__("matplotlib")
    _ = __import__("PIL")
    samples = torch.rand((4, 3, 32, 32), dtype=torch.float32) * 255.0
    out = generate_grid(
        dataset_name="m-eurosat",
        samples=samples,
        band_specs=_bands(),
        out_dir=tmp_path,
        n_samples=2,
        cloud_pattern_mode="independent_per_severity",
    )
    assert out.suffix == ".png"
    assert out.exists()
    assert out.stat().st_size > 0


def test_viz_corruptions_handles_dataset_space_rgb_indices(tmp_path):
    _require_cloud_dependency()
    _ = __import__("matplotlib")
    _ = __import__("PIL")
    samples = torch.rand((2, 3, 32, 32), dtype=torch.float32) * 255.0
    out = generate_grid(
        dataset_name="forestnet",
        samples=samples,
        band_specs=_bands(("b04", "b03", "b02")),
        out_dir=tmp_path,
        n_samples=2,
        rgb_indices=[3, 2, 1],
    )
    assert out.exists()
    assert out.stat().st_size > 0
