"""Standalone visualizer for cloud and noise corruption severities."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from torchgeo_bench.datasets import get_bench_dataset_class, get_datasets
from torchgeo_bench.datasets.base import BandSpec
from torchgeo_bench.uq.corruptions import (
    SKIP_POISSON_GAUSSIAN,
    CorruptionTransform,
    _resolve_cloud_calibration,
)

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


def _to_rgb_display(
    sample: torch.Tensor,
    rgb_indices: list[int],
    *,
    low: np.ndarray | None = None,
    high: np.ndarray | None = None,
    low_pct: float = 1.0,
    high_pct: float = 99.5,
    compress: float = 6.0,
    gamma: float = 2.2,
) -> np.ndarray:
    """Convert ``(C, H, W)`` sample into display RGB for MSI imagery.

    Uses per-image percentile clipping, then simple tone mapping to keep
    cloud highlights from saturating while lifting dark midtones.
    """
    idx = _resolve_display_indices(sample, rgb_indices)
    rgb = sample[idx].detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)

    if low is None or high is None:
        low, high = _compute_display_bounds(
            sample,
            idx,
            low_pct=low_pct,
            high_pct=high_pct,
        )

    rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)
    out = np.clip((rgb - low.reshape(1, 1, 3)) / np.maximum(high - low, 1e-6).reshape(1, 1, 3), 0.0, 1.0)
    out = np.log1p(compress * out) / np.log1p(compress)
    return np.clip(out, 0.0, 1.0) ** (1.0 / gamma)


def _compute_display_bounds(
    sample: torch.Tensor,
    rgb_indices: list[int],
    *,
    low_pct: float = 1.0,
    high_pct: float = 99.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute one fixed display range for a sample."""
    idx = _resolve_display_indices(sample, rgb_indices)
    rgb = sample[idx].detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)

    low = np.zeros(3, dtype=np.float32)
    high = np.ones(3, dtype=np.float32)
    for chan in range(3):
        vals = rgb[..., chan]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        lo = float(np.percentile(vals, low_pct))
        hi = float(np.percentile(vals, high_pct))
        if hi <= lo:
            lo = float(np.min(vals))
            hi = float(np.max(vals))
            if hi <= lo:
                hi = lo + 1e-6
        low[chan] = lo
        high[chan] = hi
    return low, high


def _summarize_cloud_batches(
    dataset_name: str,
    clean_samples: torch.Tensor,
    band_specs: list[BandSpec],
    cloud_batches: dict[int, torch.Tensor],
    cloud_masks: dict[int, torch.Tensor],
) -> dict[str, object]:
    """Summarize cloud coverage, alpha, and brightness shifts for calibration."""
    optical_indices, lower, upper, _ = _resolve_cloud_calibration(
        dataset_name=dataset_name,
        band_specs=band_specs,
        device=clean_samples.device,
        dtype=clean_samples.dtype,
    )
    denom = (upper - lower).clamp(min=1e-6)

    severity_stats: dict[str, dict[str, float]] = {}
    for severity, corrupted_samples in cloud_batches.items():
        coverage_fractions: list[float] = []
        thick_coverage_fractions: list[float] = []
        mean_cloud_alphas: list[float] = []
        mean_luma_shifts: list[float] = []
        median_luma_shifts: list[float] = []

        for row in range(clean_samples.shape[0]):
            alpha = cloud_masks[severity][row]
            clouded = alpha > 1e-6
            thick_cloud = alpha > 0.10

            coverage_fractions.append(float(clouded.float().mean()))
            thick_coverage_fractions.append(float(thick_cloud.float().mean()))

            if not bool(clouded.any()):
                mean_cloud_alphas.append(0.0)
                mean_luma_shifts.append(0.0)
                median_luma_shifts.append(0.0)
                continue

            clean_optical = ((clean_samples[row, optical_indices] - lower) / denom).clamp(0.0, 1.0)
            corrupted_optical = ((corrupted_samples[row, optical_indices] - lower) / denom).clamp(
                0.0,
                1.0,
            )
            delta_luma = corrupted_optical.mean(dim=0) - clean_optical.mean(dim=0)

            mean_cloud_alphas.append(float(alpha[clouded].mean()))
            mean_luma_shifts.append(float(delta_luma[clouded].mean()))
            median_luma_shifts.append(float(delta_luma[clouded].median()))

        count = float(len(coverage_fractions))
        severity_stats[str(severity)] = {
            "coverage_fraction_mean": float(sum(coverage_fractions) / count),
            "thick_coverage_fraction_mean": float(sum(thick_coverage_fractions) / count),
            "mean_cloud_alpha_clouded": float(sum(mean_cloud_alphas) / count),
            "mean_luma_shift_clouded": float(sum(mean_luma_shifts) / count),
            "median_luma_shift_clouded": float(sum(median_luma_shifts) / count),
        }

    return {
        "dataset": dataset_name,
        "n_samples": int(clean_samples.shape[0]),
        "severity_stats": severity_stats,
    }


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
    if images.shape[0] <= n_samples:
        return images[:n_samples], band_specs, rgb_indices

    # Prefer informative samples (higher spatial contrast in display bands)
    # so corruption progression is visible in the saved grid.
    idx = _resolve_display_indices(images[0], rgb_indices)
    rgb = images[:, idx]
    scores = rgb.std(dim=(1, 2, 3))
    topk = torch.topk(scores, k=n_samples, largest=True).indices
    return images[topk], band_specs, rgb_indices


def generate_grid(
    dataset_name: str,
    samples: torch.Tensor,
    band_specs: list[BandSpec],
    out_dir: str | Path,
    n_samples: int = 4,
    rgb_indices: list[int] | None = None,
    seed: int = 0,
    cloud_pattern_mode: str = "fixed_across_severity",
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
        cloud_pattern_mode: Cloud RNG mode. Use ``fixed_across_severity`` to
            keep one cloud pattern per image across severities, or
            ``independent_per_severity`` for different patterns per severity.

    Returns:
        Path to the generated PNG file.
    """
    plt = _import_plotting()

    if rgb_indices is None:
        rgb_indices = [0, 1, 2]

    samples = samples[:n_samples].detach().clone()
    n_rows = samples.shape[0]
    display_indices = _resolve_display_indices(samples[0], rgb_indices)

    cloud_batches: dict[int, torch.Tensor] = {}
    cloud_masks: dict[int, torch.Tensor] = {}
    for severity in [1, 2, 3, 4, 5]:
        cloud_t = CorruptionTransform(
            "cloud",
            severity,
            seed,
            band_specs,
            dataset_name=dataset_name,
            cloud_pattern_mode=cloud_pattern_mode,
        )
        cloud_batches[severity], cloud_masks[severity] = cloud_t.apply_cloud_with_mask(samples)

    noise_skipped = dataset_name in SKIP_POISSON_GAUSSIAN
    noise_batches: dict[int, torch.Tensor | None] = {}
    for severity in [1, 2, 3, 4, 5]:
        if noise_skipped:
            noise_batches[severity] = None
            continue
        noise_t = CorruptionTransform(
            "poisson_gaussian",
            severity,
            seed,
            band_specs,
            dataset_name=dataset_name,
        )
        noise_batches[severity] = noise_t(samples)

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

    for row in range(n_rows):
        clean = samples[row]
        display_low, display_high = _compute_display_bounds(clean, display_indices)
        variants: list[torch.Tensor | None] = [clean]

        for severity in [1, 2, 3, 4, 5]:
            variants.append(cloud_batches[severity][row])

        for severity in [1, 2, 3, 4, 5]:
            variants.append(None if noise_batches[severity] is None else noise_batches[severity][row])

        for col, img in enumerate(variants):
            ax = axes[row, col]
            ax.axis("off")
            if img is None:
                ax.text(0.5, 0.5, "—", ha="center", va="center", fontsize=18)
            else:
                ax.imshow(
                    _to_rgb_display(
                        img,
                        display_indices,
                        low=display_low,
                        high=display_high,
                    )
                )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset_name}_corruptions.png"
    stats_path = out_dir / f"{dataset_name}_corruptions_stats.json"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    stats = _summarize_cloud_batches(
        dataset_name=dataset_name,
        clean_samples=samples,
        band_specs=band_specs,
        cloud_batches=cloud_batches,
        cloud_masks=cloud_masks,
    )
    with stats_path.open("w", encoding="utf-8") as file:
        json.dump(stats, file, indent=2, sort_keys=True)
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
    parser.add_argument(
        "--cloud-pattern-mode",
        type=str,
        choices=["fixed_across_severity", "independent_per_severity"],
        default="fixed_across_severity",
        help="Cloud pattern RNG mode across severities.",
    )
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
        cloud_pattern_mode=args.cloud_pattern_mode,
    )
    logger.info("Saved corruption grid to %s", out_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
