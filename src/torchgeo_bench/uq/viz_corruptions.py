"""Standalone visualizer for cloud and noise corruption severities."""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from torchgeo_bench.datasets import get_bench_dataset_class, get_datasets
from torchgeo_bench.datasets.base import BandSpec
from torchgeo_bench.uq.corruptions import CorruptionTransform, SKIP_POISSON_GAUSSIAN

logger = logging.getLogger(__name__)


def _import_plotting():
    """Import optional plotting dependencies.

    Returns:
        The imported ``matplotlib.pyplot`` module.

    Raises:
        ModuleNotFoundError: If matplotlib or Pillow is not installed.
    """
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required for viz_corruptions. Install `torchgeo-bench[uq,viz]` "
            "or run `uv sync --extra uq --extra viz`."
        ) from exc
    try:
        import PIL  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Pillow is required for viz_corruptions. Install `torchgeo-bench[uq,viz]` "
            "or run `uv sync --extra uq --extra viz`."
        ) from exc
    return plt


def _resolve_display_indices(sample: torch.Tensor, rgb_indices: list[int]) -> list[int]:
    """Return safe 3-channel indices for rendering.

    Args:
        sample: Input sample tensor with shape ``(C, H, W)``.
        rgb_indices: Preferred channel indices for RGB visualization.

    Returns:
        Exactly three valid channel indices into ``sample``.
    """
    n_channels = int(sample.shape[0])
    if n_channels <= 0:
        raise ValueError("Sample must contain at least one channel.")

    valid = [idx for idx in rgb_indices if 0 <= idx < n_channels]
    if not valid:
        valid = [0]

    while len(valid) < 3:
        valid.append(valid[-1])
    return valid[:3]


def _to_rgb(sample: torch.Tensor, rgb_indices: list[int]) -> np.ndarray:
    """Convert a ``(C, H, W)`` sample into a display-ready RGB image.

    Args:
        sample: Input sample tensor with shape ``(C, H, W)``.
        rgb_indices: Preferred channel indices for RGB visualization.

    Returns:
        Normalized RGB array with shape ``(H, W, 3)``.
    """
    idx = _resolve_display_indices(sample, rgb_indices)

    rgb = sample[idx].detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)
    out = np.empty_like(rgb)
    for chan in range(3):
        lo, hi = np.percentile(rgb[..., chan], [1, 99])
        if hi <= lo:
            out[..., chan] = 0.0
        else:
            out[..., chan] = np.clip((rgb[..., chan] - lo) / (hi - lo), 0.0, 1.0)
    return out


def _load_test_samples(
    dataset_name: str,
    n_samples: int,
) -> tuple[torch.Tensor, list[BandSpec], list[int]]:
    """Load test samples and RGB/band metadata for visualization.

    Args:
        dataset_name: Benchmark dataset name.
        n_samples: Number of test samples to return.

    Returns:
        Tuple ``(samples, band_specs, rgb_indices)``.
    """
    ds_cls = get_bench_dataset_class(dataset_name)
    bench = ds_cls()
    bands_resolved = tuple(bench.rgb_bands)
    band_specs = bench.select_band_specs(bands_resolved)
    loaded_band_names = [spec.name for spec in band_specs]
    rgb_indices = [loaded_band_names.index(name) for name in bench.rgb_bands if name in loaded_band_names]

    loaded = get_datasets(
        dataset_name=dataset_name,
        partition_name="default",
        batch_size=max(16, n_samples),
        num_workers=0,
        return_val=True,
        image_size=224,
        interpolation="bilinear",
        bands="rgb",
    )
    if loaded is None:
        raise RuntimeError(f"Failed to load dataset {dataset_name}")
    _, _, _, test_loader = loaded
    batch = next(iter(test_loader))
    images = batch["image"]
    return images[:n_samples], band_specs, rgb_indices


def generate_grid(
    dataset_name: str,
    samples: torch.Tensor,
    band_specs: list[BandSpec],
    out_dir: str | Path,
    n_samples: int = 4,
    rgb_indices: list[int] | None = None,
    seed: int = 0,
) -> Path:
    """Render and save an 11-column corruption grid for a dataset.

    Args:
        dataset_name: Benchmark dataset name.
        samples: Input samples tensor with shape ``(N, C, H, W)``.
        band_specs: Per-channel band metadata for corruption synthesis.
        out_dir: Output directory for the PNG image.
        n_samples: Number of rows to render from ``samples``.
        rgb_indices: Optional RGB channel indices for display.
        seed: Base random seed for deterministic corruption generation.

    Returns:
        Path to the generated PNG file.
    """
    plt = _import_plotting()

    if rgb_indices is None:
        rgb_indices = [0, 1, 2]

    samples = samples[:n_samples].detach().clone()
    n_rows = samples.shape[0]
    n_cols = 11
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.4 * n_cols, 2.4 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    headers = [
        "clean",
        "cloud s1",
        "cloud s2",
        "cloud s3",
        "cloud s4",
        "cloud s5",
        "noise s1",
        "noise s2",
        "noise s3",
        "noise s4",
        "noise s5",
    ]
    for col, header in enumerate(headers):
        axes[0, col].set_title(header)

    noise_skipped = dataset_name in SKIP_POISSON_GAUSSIAN
    for row in range(n_rows):
        clean = samples[row]
        variants: list[torch.Tensor | None] = [clean]

        for severity in [1, 2, 3, 4, 5]:
            cloud_t = CorruptionTransform("cloud_shadow", severity, seed + row * 100, band_specs)
            variants.append(cloud_t(clean.unsqueeze(0))[0])

        for severity in [1, 2, 3, 4, 5]:
            if noise_skipped:
                variants.append(None)
            else:
                noise_t = CorruptionTransform(
                    "poisson_gaussian", severity, seed + row * 100 + 7, band_specs
                )
                variants.append(noise_t(clean.unsqueeze(0))[0])

        for col, img in enumerate(variants):
            ax = axes[row, col]
            ax.axis("off")
            if img is None:
                ax.text(0.5, 0.5, "—", ha="center", va="center", fontsize=18)
            else:
                ax.imshow(_to_rgb(img, rgb_indices))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset_name}_corruptions.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for the corruption visualizer.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(description="Visualize UQ corruption severities.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g. m-eurosat.")
    parser.add_argument("--n-samples", type=int, default=4, help="Number of test samples to render.")
    parser.add_argument("--out", type=Path, default=Path("viz/corruptions"), help="Output directory.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for corruption generation.")
    return parser


def main() -> int:
    """Run the corruption-visualization CLI.

    Returns:
        Process exit code.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = _build_parser()
    args = parser.parse_args()
    samples, band_specs, rgb_indices = _load_test_samples(args.dataset, args.n_samples)
    out_path = generate_grid(
        dataset_name=args.dataset,
        samples=samples,
        band_specs=band_specs,
        out_dir=args.out,
        n_samples=args.n_samples,
        rgb_indices=rgb_indices,
        seed=args.seed,
    )
    logger.info("Saved corruption grid to %s", out_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
