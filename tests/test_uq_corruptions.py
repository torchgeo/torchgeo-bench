import torch

from torchgeo_bench.datasets.base import BandSpec
from torchgeo_bench.uq.corruptions import CorruptionTransform, SKIP_POISSON_GAUSSIAN
from torchgeo_bench.uq.viz_corruptions import generate_grid


def _bands(n: int = 3) -> list[BandSpec]:
    return [
        BandSpec(
            sensor="s2",
            name=f"b{i}",
            source_name=f"B{i}",
            mean=float(100 + i * 5),
            std=float(10 + i),
            min=0.0,
            max=255.0,
        )
        for i in range(n)
    ]


def test_cloud_shadow_output_shape():
    x = torch.rand((2, 3, 16, 16), dtype=torch.float32) * 255.0
    tfm = CorruptionTransform("cloud_shadow", severity=1, seed=11, band_specs=_bands())
    y = tfm(x)
    assert y.shape == x.shape


def test_cloud_shadow_dtype_preserved():
    x = torch.rand((2, 3, 16, 16), dtype=torch.float16) * 255.0
    tfm = CorruptionTransform("cloud_shadow", severity=1, seed=11, band_specs=_bands())
    y = tfm(x)
    assert y.dtype == x.dtype


def test_cloud_shadow_values_clamped():
    x = torch.rand((2, 3, 16, 16), dtype=torch.float32) * 512.0
    tfm = CorruptionTransform("cloud_shadow", severity=1, seed=11, band_specs=_bands())
    y = tfm(x)
    assert float(y.min()) >= 0.0
    assert float(y.max()) <= 255.0


def test_cloud_shadow_determinism():
    x = torch.rand((2, 3, 16, 16), dtype=torch.float32) * 255.0
    tfm_a = CorruptionTransform("cloud_shadow", severity=3, seed=17, band_specs=_bands())
    tfm_b = CorruptionTransform("cloud_shadow", severity=3, seed=17, band_specs=_bands())
    y_a = tfm_a(x)
    y_b = tfm_b(x)
    assert torch.allclose(y_a, y_b)


def test_cloud_shadow_counter_increments():
    x1 = torch.rand((4, 3, 16, 16), dtype=torch.float32) * 255.0
    x2 = torch.rand((2, 3, 16, 16), dtype=torch.float32) * 255.0
    tfm = CorruptionTransform("cloud_shadow", severity=2, seed=31, band_specs=_bands())
    _ = tfm(x1)
    assert tfm._n_images_seen == 4
    y2 = tfm(x2)

    tfm_reset = CorruptionTransform("cloud_shadow", severity=2, seed=31, band_specs=_bands())
    y2_reset = tfm_reset(x2)
    assert not torch.allclose(y2, y2_reset)


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
    _ = __import__("matplotlib")
    _ = __import__("PIL")
    samples = torch.rand((4, 3, 32, 32), dtype=torch.float32) * 255.0
    out = generate_grid(
        dataset_name="m-eurosat",
        samples=samples,
        band_specs=_bands(),
        out_dir=tmp_path,
        n_samples=2,
    )
    assert out.suffix == ".png"
    assert out.exists()
    assert out.stat().st_size > 0


def test_viz_corruptions_handles_dataset_space_rgb_indices(tmp_path):
    _ = __import__("matplotlib")
    _ = __import__("PIL")
    samples = torch.rand((2, 3, 32, 32), dtype=torch.float32) * 255.0
    out = generate_grid(
        dataset_name="benv2",
        samples=samples,
        band_specs=_bands(),
        out_dir=tmp_path,
        n_samples=2,
        rgb_indices=[3, 2, 1],
    )
    assert out.exists()
    assert out.stat().st_size > 0
