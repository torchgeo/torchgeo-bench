import json

import pytest
import torch

from torchgeo_bench.datasets.base import BandSpec
from torchgeo_bench.uq.corruptions import (
    CLOUD_DATASET_CALIBRATIONS,
    CorruptionTransform,
    SKIP_POISSON_GAUSSIAN,
    _resolve_cloud_calibration,
)
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


def _so2sat_rgb_bands() -> list[BandSpec]:
    return [
        BandSpec(
            sensor="s2",
            name="b04",
            source_name="B04",
            mean=0.1138,
            std=0.0733,
            min=0.0001,
            max=2.8,
            wavelength_um=0.665,
        ),
        BandSpec(
            sensor="s2",
            name="b03",
            source_name="B03",
            mean=0.1172,
            std=0.052,
            min=0.0001,
            max=2.8,
            wavelength_um=0.56,
        ),
        BandSpec(
            sensor="s2",
            name="b02",
            source_name="B02",
            mean=0.1295,
            std=0.0414,
            min=0.0001,
            max=2.8,
            wavelength_um=0.49,
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


@pytest.mark.parametrize("dataset_name", ["advance", "resisc45"])
def test_cloud_calibration_registered_for_advance_and_resisc45(dataset_name):
    assert dataset_name in CLOUD_DATASET_CALIBRATIONS
    optical_indices, lower, upper, _ = _resolve_cloud_calibration(
        dataset_name=dataset_name,
        band_specs=_bands(),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert optical_indices == [0, 1, 2]
    assert lower.shape == (3, 1, 1)
    assert upper.shape == (3, 1, 1)


def test_cloud_calibration_uses_zero_haze_floor():
    for calibration in CLOUD_DATASET_CALIBRATIONS.values():
        for severity in range(1, 6):
            assert calibration.severity_presets[severity].min_lvl == (0.0, 0.0)


def test_cloud_calibration_uses_stronger_opacity_for_bright_rgb_datasets():
    advance_max = CLOUD_DATASET_CALIBRATIONS["advance"].severity_presets[5].max_lvl[0]
    resisc_max = CLOUD_DATASET_CALIBRATIONS["resisc45"].severity_presets[5].max_lvl[0]
    so2sat_max = CLOUD_DATASET_CALIBRATIONS["so2sat"].severity_presets[5].max_lvl[0]
    assert advance_max > so2sat_max
    assert resisc_max > so2sat_max


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
        cloud_pattern_mode="fixed",
    )
    tfm_independent = CorruptionTransform(
        "cloud",
        severity=3,
        seed=17,
        band_specs=_bands(),
        dataset_name="m-eurosat",
        cloud_pattern_mode="independent",
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


def test_poisson_gaussian_severity_progression():
    x = torch.rand((8, 3, 128, 128), dtype=torch.float32) * 255.0
    deltas: list[float] = []
    for severity in [1, 2, 3, 4, 5]:
        tfm = CorruptionTransform("poisson_gaussian", severity=severity, seed=21, band_specs=_bands())
        y = tfm(x)
        deltas.append(float(torch.mean(torch.abs(y - x))))

    non_decreasing_steps = sum(next_delta >= delta - 1e-3 for delta, next_delta in zip(deltas, deltas[1:]))
    assert non_decreasing_steps >= 3
    assert deltas[-1] > deltas[0] * 1.35


def test_poisson_gaussian_determinism_and_seed_diversity():
    x = torch.rand((2, 3, 64, 64), dtype=torch.float32) * 255.0
    tfm_a = CorruptionTransform("poisson_gaussian", severity=3, seed=123, band_specs=_bands())
    tfm_b = CorruptionTransform("poisson_gaussian", severity=3, seed=123, band_specs=_bands())
    tfm_c = CorruptionTransform("poisson_gaussian", severity=3, seed=124, band_specs=_bands())

    y_a = tfm_a(x)
    y_b = tfm_b(x)
    y_c = tfm_c(x)
    assert torch.allclose(y_a, y_b)
    assert not torch.allclose(y_a, y_c)


def test_poisson_gaussian_non_degenerate_on_low_signal():
    x = torch.zeros((4, 3, 64, 64), dtype=torch.float32)
    tfm = CorruptionTransform("poisson_gaussian", severity=3, seed=8, band_specs=_bands())
    y = tfm(x)

    assert float(y.min()) >= 0.0
    assert float(y.max()) <= 255.0
    assert float(torch.mean(torch.abs(y - x))) > 0.05


def test_poisson_gaussian_mixed_range_bands_clamped():
    band_specs = _mixed_sensor_bands()
    x = torch.rand((2, len(band_specs), 32, 32), dtype=torch.float32)
    x[:, 0:3] *= 255.0
    x[:, 3] = -30.0 + 33.0 * x[:, 3]
    x[:, 4] = -35.0 + 35.0 * x[:, 4]

    tfm = CorruptionTransform("poisson_gaussian", severity=4, seed=43, band_specs=band_specs)
    y = tfm(x)

    mins = torch.tensor([band.min for band in band_specs], dtype=torch.float32).view(1, -1, 1, 1)
    maxs = torch.tensor([band.max for band in band_specs], dtype=torch.float32).view(1, -1, 1, 1)
    assert bool((y >= mins).all())
    assert bool((y <= maxs).all())
    assert float(torch.mean(torch.abs(y - x))) > 0.0


def test_skip_poisson_gaussian_constant():
    assert "m-so2sat" not in SKIP_POISSON_GAUSSIAN
    assert "so2sat" not in SKIP_POISSON_GAUSSIAN
    assert SKIP_POISSON_GAUSSIAN == frozenset()


def test_poisson_gaussian_so2sat_progression():
    min_val = 0.0001
    max_val = 2.8
    x = min_val + torch.rand((8, 3, 128, 128), dtype=torch.float32) * (max_val - min_val)
    deltas: list[float] = []
    for severity in [1, 2, 3, 4, 5]:
        tfm = CorruptionTransform(
            "poisson_gaussian",
            severity=severity,
            seed=21,
            band_specs=_so2sat_rgb_bands(),
            dataset_name="so2sat",
        )
        y = tfm(x)
        deltas.append(float(torch.mean(torch.abs(y - x))))

    non_decreasing_steps = sum(next_delta >= delta - 1e-4 for delta, next_delta in zip(deltas, deltas[1:]))
    assert non_decreasing_steps >= 3
    assert deltas[-1] > deltas[0] * 1.5


def test_poisson_gaussian_so2sat_determinism():
    min_val = 0.0001
    max_val = 2.8
    x = min_val + torch.rand((4, 3, 64, 64), dtype=torch.float32) * (max_val - min_val)
    tfm_a = CorruptionTransform(
        "poisson_gaussian",
        severity=3,
        seed=123,
        band_specs=_so2sat_rgb_bands(),
        dataset_name="so2sat",
    )
    tfm_b = CorruptionTransform(
        "poisson_gaussian",
        severity=3,
        seed=123,
        band_specs=_so2sat_rgb_bands(),
        dataset_name="so2sat",
    )
    tfm_c = CorruptionTransform(
        "poisson_gaussian",
        severity=3,
        seed=124,
        band_specs=_so2sat_rgb_bands(),
        dataset_name="so2sat",
    )

    y_a = tfm_a(x)
    y_b = tfm_b(x)
    y_c = tfm_c(x)
    assert torch.allclose(y_a, y_b)
    assert not torch.allclose(y_a, y_c)


def test_poisson_gaussian_so2sat_bounds():
    min_val = 0.0001
    max_val = 2.8
    x = min_val + torch.rand((4, 3, 64, 64), dtype=torch.float32) * (max_val - min_val)
    tfm = CorruptionTransform(
        "poisson_gaussian",
        severity=5,
        seed=333,
        band_specs=_so2sat_rgb_bands(),
        dataset_name="so2sat",
    )
    y = tfm(x)
    assert float(y.min()) >= min_val - 1e-6
    assert float(y.max()) <= max_val + 1e-6


def test_motion_blur_output_shape():
    x = torch.rand((2, 3, 32, 32), dtype=torch.float32) * 255.0
    tfm = CorruptionTransform("motion_blur", severity=1, seed=0, band_specs=_bands())
    y = tfm(x)
    assert y.shape == x.shape


def test_motion_blur_dtype_preserved():
    x = torch.rand((2, 3, 32, 32), dtype=torch.float32) * 255.0
    tfm = CorruptionTransform("motion_blur", severity=3, seed=0, band_specs=_bands())
    assert tfm(x).dtype == x.dtype


def test_motion_blur_values_clamped():
    x = torch.rand((2, 3, 32, 32), dtype=torch.float32) * 255.0
    tfm = CorruptionTransform("motion_blur", severity=5, seed=0, band_specs=_bands())
    y = tfm(x)
    assert float(y.min()) >= 0.0
    assert float(y.max()) <= 255.0


def test_motion_blur_is_deterministic():
    x = torch.rand((4, 3, 64, 64), dtype=torch.float32) * 255.0
    tfm_a = CorruptionTransform("motion_blur", severity=3, seed=0, band_specs=_bands())
    tfm_b = CorruptionTransform("motion_blur", severity=3, seed=0, band_specs=_bands())
    assert torch.allclose(tfm_a(x), tfm_b(x))


def test_motion_blur_severity_progression():
    x = torch.rand((4, 3, 224, 224), dtype=torch.float32) * 255.0
    deltas: list[float] = []
    for severity in [1, 2, 3, 4, 5]:
        tfm = CorruptionTransform("motion_blur", severity=severity, seed=0, band_specs=_bands())
        deltas.append(float(torch.mean(torch.abs(tfm(x) - x))))
    assert all(b >= a for a, b in zip(deltas, deltas[1:]))
    assert deltas[-1] > deltas[0]


def test_motion_blur_blurs_horizontally_not_vertically():
    # Step edge in the horizontal direction — blur must smooth it
    x = torch.zeros((1, 3, 32, 32), dtype=torch.float32)
    x[:, :, :, 16:] = 255.0  # right half is white
    tfm = CorruptionTransform("motion_blur", severity=3, seed=0, band_specs=_bands())
    y = tfm(x)
    # Columns near the step edge should be smoothed (neither 0 nor 255)
    assert float(y[:, :, :, 12:20].min()) > 0.0
    assert float(y[:, :, :, 12:20].max()) < 255.0
    # Rows should be unchanged — no vertical blur (all rows identical)
    assert torch.allclose(y[0, 0, 0, :], y[0, 0, 1, :])


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
        cloud_pattern_mode="independent",
    )
    assert out.suffix == ".png"
    assert out.exists()
    assert out.stat().st_size > 0
    stats_path = tmp_path / "m-eurosat_corruptions_stats.json"
    assert stats_path.exists()
    with stats_path.open("r", encoding="utf-8") as file:
        stats = json.load(file)
    assert stats["dataset"] == "m-eurosat"
    assert "1" in stats["severity_stats"]
    assert "mean_cloud_alpha_clouded" in stats["severity_stats"]["1"]


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
