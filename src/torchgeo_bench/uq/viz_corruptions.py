"""Standalone visualizer for cloud and noise corruption severities."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from torchgeo_bench.datasets import get_bench_dataset_class, get_datasets
from torchgeo_bench.datasets.base import BandSpec
from torchgeo_bench.models._normalization import NormalizationStrategy, build_normalizer
from torchgeo_bench.uq.corruptions import (
    SKIP_POISSON_GAUSSIAN,
    CorruptionTransform,
    _resolve_cloud_calibration,
)

logger = logging.getLogger(__name__)

DISPLAY_LOW_PCT = 1.0
DISPLAY_HIGH_PCT = 99.5
DISPLAY_SAMPLE_COUNT = 512
DISPLAY_SAMPLE_PIXELS = 1024
CACHE_VERSION = 2
PERCENTILE_CACHE_NAME = "torchgeo_bench_display_percentiles.json"

DISPLAY_PERCENTILES_BY_DATASET: dict[str, tuple[float, float]] = {
    "so2sat": (0.5, 99.0),
    "m-so2sat": (0.5, 99.0),
    "eurosat": (0.5, 99.0),
    "m-eurosat": (0.5, 99.0),
    "eurosat-spatial": (0.5, 99.0),
}

TONE_MAPPED_DATASETS: frozenset[str] = frozenset(
    {
        "so2sat",
        "m-so2sat",
        "eurosat",
        "m-eurosat",
        "eurosat-spatial",
    }
)


def _dataset_display_percentiles(dataset_name: str) -> tuple[float, float]:
    return DISPLAY_PERCENTILES_BY_DATASET.get(dataset_name, (DISPLAY_LOW_PCT, DISPLAY_HIGH_PCT))


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


def _resolve_dataset_cache_dir(dataset_name: str) -> Path:
    ds_cls = get_bench_dataset_class(dataset_name)
    bench = ds_cls()
    root = Path(bench.data_root())
    dataset_root = root / dataset_name
    return dataset_root if dataset_root.exists() else root


def _percentile_cache_path(dataset_root: Path) -> Path:
    return dataset_root / PERCENTILE_CACHE_NAME


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


def _to_rgb_display_linear(
    sample: torch.Tensor,
    rgb_indices: list[int],
    *,
    low: np.ndarray,
    high: np.ndarray,
    gamma: float = 1.0,
) -> np.ndarray:
    idx = _resolve_display_indices(sample, rgb_indices)
    rgb = sample[idx].detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)
    out = np.clip((rgb - low.reshape(1, 1, 3)) / np.maximum(high - low, 1e-6).reshape(1, 1, 3), 0.0, 1.0)
    if gamma != 1.0:
        out = np.clip(out, 0.0, 1.0) ** (1.0 / gamma)
    return out


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


def _sample_rgb_pixels(
    sample: torch.Tensor,
    rgb_indices: list[int],
    *,
    max_pixels: int,
    rng: np.random.Generator,
) -> np.ndarray:
    idx = _resolve_display_indices(sample, rgb_indices)
    rgb = sample[idx].detach().cpu().numpy().astype(np.float32)
    _, height, width = rgb.shape
    flat = rgb.reshape(3, height * width).transpose(1, 0)
    if flat.shape[0] <= max_pixels:
        return flat
    pick = rng.choice(flat.shape[0], size=max_pixels, replace=False)
    return flat[pick]


def _sample_luma_pixels(
    sample: torch.Tensor,
    rgb_indices: list[int],
    *,
    max_pixels: int,
    rng: np.random.Generator,
) -> np.ndarray:
    idx = _resolve_display_indices(sample, rgb_indices)
    rgb = sample[idx].detach().cpu().numpy().astype(np.float32)
    _, height, width = rgb.shape
    luma = rgb.mean(axis=0).reshape(height * width)
    if luma.shape[0] <= max_pixels:
        return luma
    pick = rng.choice(luma.shape[0], size=max_pixels, replace=False)
    return luma[pick]


def _load_cache_samples(
    dataset_name: str,
    n_samples: int,
) -> tuple[torch.Tensor, list[BandSpec], list[int]]:
    ds_cls = get_bench_dataset_class(dataset_name)
    bench = ds_cls()
    bands_resolved = tuple(bench.rgb_bands)
    band_specs = bench.select_band_specs(bands_resolved)
    loaded_band_names = [spec.name for spec in band_specs]
    rgb_indices = [loaded_band_names.index(name) for name in bench.rgb_bands if name in loaded_band_names]

    loaded = get_datasets(
        dataset_name=dataset_name,
        partition_name="default",
        batch_size=64,
        num_workers=0,
        return_val=True,
        image_size=224,
        interpolation="bilinear",
        bands="rgb",
    )
    if loaded is None:
        raise RuntimeError(f"Failed to load dataset {dataset_name}")
    _, _, _, test_loader = loaded

    images: list[torch.Tensor] = []
    for batch in test_loader:
        images.append(batch["image"])
        total = sum(item.shape[0] for item in images)
        if total >= n_samples:
            break
    if not images:
        raise RuntimeError(f"No samples available for dataset {dataset_name}")
    stacked = torch.cat(images, dim=0)
    return stacked[:n_samples], band_specs, rgb_indices


def _compute_display_percentiles(
    dataset_name: str,
    band_specs: list[BandSpec],
    rgb_indices: list[int],
    *,
    seed: int,
    cloud_pattern_mode: str,
    n_samples: int,
    max_pixels: int,
) -> dict[str, object]:
    low_pct, high_pct = _dataset_display_percentiles(dataset_name)
    rng = np.random.default_rng(seed)
    samples, _, _ = _load_cache_samples(dataset_name, n_samples)
    samples = samples.detach().clone()

    normalizer = build_normalizer(NormalizationStrategy.BANDSPEC_ZSCORE, bands=band_specs)
    noise_skipped = dataset_name in SKIP_POISSON_GAUSSIAN

    cloud_batches: dict[int, torch.Tensor] = {}
    for severity in [1, 2, 3, 4, 5]:
        cloud_t = CorruptionTransform(
            "cloud",
            severity,
            seed,
            band_specs,
            dataset_name=dataset_name,
            cloud_pattern_mode=cloud_pattern_mode,
        )
        cloud_batches[severity] = cloud_t(samples)

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

    raw_pixels: list[np.ndarray] = []
    norm_pixels: list[np.ndarray] = []
    delta_pixels: list[np.ndarray] = []

    for row in range(samples.shape[0]):
        clean = samples[row]
        clean_norm = normalizer(clean.unsqueeze(0)).squeeze(0)

        raw_pixels.append(
            _sample_rgb_pixels(clean, rgb_indices, max_pixels=max_pixels, rng=rng)
        )
        norm_pixels.append(
            _sample_rgb_pixels(clean_norm, rgb_indices, max_pixels=max_pixels, rng=rng)
        )

        for severity in [1, 2, 3, 4, 5]:
            clouded = cloud_batches[severity][row]
            clouded_norm = normalizer(clouded.unsqueeze(0)).squeeze(0)
            raw_pixels.append(
                _sample_rgb_pixels(clouded, rgb_indices, max_pixels=max_pixels, rng=rng)
            )
            norm_pixels.append(
                _sample_rgb_pixels(clouded_norm, rgb_indices, max_pixels=max_pixels, rng=rng)
            )
            delta = clouded_norm - clean_norm
            delta_pixels.append(
                _sample_luma_pixels(delta, rgb_indices, max_pixels=max_pixels, rng=rng)
            )

        for severity in [1, 2, 3, 4, 5]:
            noised = noise_batches[severity]
            if noised is None:
                continue
            noised_row = noised[row]
            noised_norm = normalizer(noised_row.unsqueeze(0)).squeeze(0)
            raw_pixels.append(
                _sample_rgb_pixels(noised_row, rgb_indices, max_pixels=max_pixels, rng=rng)
            )
            norm_pixels.append(
                _sample_rgb_pixels(noised_norm, rgb_indices, max_pixels=max_pixels, rng=rng)
            )
            delta = noised_norm - clean_norm
            delta_pixels.append(
                _sample_luma_pixels(delta, rgb_indices, max_pixels=max_pixels, rng=rng)
            )

    raw_stack = np.concatenate(raw_pixels, axis=0)
    norm_stack = np.concatenate(norm_pixels, axis=0)
    delta_stack = np.concatenate(delta_pixels, axis=0)

    raw_low = np.percentile(raw_stack, low_pct, axis=0).astype(np.float32)
    raw_high = np.percentile(raw_stack, high_pct, axis=0).astype(np.float32)
    norm_low = np.percentile(norm_stack, low_pct, axis=0).astype(np.float32)
    norm_high = np.percentile(norm_stack, high_pct, axis=0).astype(np.float32)
    delta_low = float(np.percentile(delta_stack, low_pct))
    delta_high = float(np.percentile(delta_stack, high_pct))
    delta_bound = max(abs(delta_low), abs(delta_high), 1e-6)

    return {
        "version": CACHE_VERSION,
        "dataset": dataset_name,
        "n_samples": n_samples,
        "low_pct": low_pct,
        "high_pct": high_pct,
        "raw": {"low": raw_low.tolist(), "high": raw_high.tolist()},
        "normalized": {"low": norm_low.tolist(), "high": norm_high.tolist()},
        "delta_luma": {"bound": float(delta_bound)},
    }


def _load_or_compute_percentiles(
    dataset_name: str,
    band_specs: list[BandSpec],
    rgb_indices: list[int],
    *,
    seed: int,
    cloud_pattern_mode: str,
) -> dict[str, object]:
    dataset_root = _resolve_dataset_cache_dir(dataset_name)
    dataset_root.mkdir(parents=True, exist_ok=True)
    cache_path = _percentile_cache_path(dataset_root)
    low_pct, high_pct = _dataset_display_percentiles(dataset_name)
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as file:
            cached = json.load(file)
        if (
            cached.get("version") == CACHE_VERSION
            and cached.get("dataset") == dataset_name
            and cached.get("low_pct") == low_pct
            and cached.get("high_pct") == high_pct
        ):
            return cached

    stats = _compute_display_percentiles(
        dataset_name,
        band_specs,
        rgb_indices,
        seed=seed,
        cloud_pattern_mode=cloud_pattern_mode,
        n_samples=DISPLAY_SAMPLE_COUNT,
        max_pixels=DISPLAY_SAMPLE_PIXELS,
    )
    with cache_path.open("w", encoding="utf-8") as file:
        json.dump(stats, file, indent=2, sort_keys=True)
    return stats


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
    cloud_pattern_mode: str = "fixed",
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
        cloud_pattern_mode: Cloud RNG mode. Use ``fixed`` to
            keep one cloud pattern per image across severities, or
            ``independent`` for different patterns per severity.

    Returns:
        Path to the generated PNG file.
    """
    plt = _import_plotting()

    if rgb_indices is None:
        rgb_indices = [0, 1, 2]

    samples = samples[:n_samples].detach().clone()
    n_rows = samples.shape[0]
    display_indices = _resolve_display_indices(samples[0], rgb_indices)

    percentiles = _load_or_compute_percentiles(
        dataset_name,
        band_specs,
        display_indices,
        seed=seed,
        cloud_pattern_mode=cloud_pattern_mode,
    )
    raw_low = np.array(percentiles["raw"]["low"], dtype=np.float32)
    raw_high = np.array(percentiles["raw"]["high"], dtype=np.float32)
    norm_low = np.array(percentiles["normalized"]["low"], dtype=np.float32)
    norm_high = np.array(percentiles["normalized"]["high"], dtype=np.float32)
    delta_bound = float(percentiles["delta_luma"]["bound"])

    normalizer = build_normalizer(NormalizationStrategy.BANDSPEC_ZSCORE, bands=band_specs)
    use_tone_map = dataset_name in TONE_MAPPED_DATASETS

    def _render_rgb(sample: torch.Tensor, *, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        if use_tone_map:
            return _to_rgb_display(
                sample,
                display_indices,
                low=low,
                high=high,
                compress=6.0,
                gamma=2.2,
            )
        return _to_rgb_display_linear(
            sample,
            display_indices,
            low=low,
            high=high,
            gamma=1.0,
        )

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

    def _render_grid(
        variants_by_row: list[list[torch.Tensor | None]],
        *,
        out_path: Path,
        cmap: str | None = None,
        vmin: float | None = None,
        vmax: float | None = None,
        draw_dash_for_none: bool = True,
        is_rgb: bool = True,
    ) -> None:
        n_cols = len(variants_by_row[0])
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.4 * n_cols, 2.4 * n_rows))
        if n_rows == 1:
            axes = np.expand_dims(axes, axis=0)
        for col, header in enumerate(headers):
            axes[0, col].set_title(header)
        for row in range(n_rows):
            for col, img in enumerate(variants_by_row[row]):
                ax = axes[row, col]
                ax.axis("off")
                if img is None:
                    if draw_dash_for_none:
                        ax.text(0.5, 0.5, "—", ha="center", va="center", fontsize=18)
                    continue
                if is_rgb:
                    ax.imshow(img)
                else:
                    ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

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

    raw_variants: list[list[np.ndarray | None]] = []
    norm_variants: list[list[np.ndarray | None]] = []
    delta_variants: list[list[np.ndarray | None]] = []
    alpha_variants: list[list[np.ndarray | None]] = []

    for row in range(n_rows):
        clean = samples[row]
        clean_norm = normalizer(clean.unsqueeze(0)).squeeze(0)

        row_raw: list[np.ndarray | None] = [
            _render_rgb(clean, low=raw_low, high=raw_high)
        ]
        row_norm: list[np.ndarray | None] = [
            _render_rgb(clean_norm, low=norm_low, high=norm_high)
        ]
        row_delta: list[np.ndarray | None] = [
            np.zeros((clean.shape[1], clean.shape[2]), dtype=np.float32)
        ]
        row_alpha: list[np.ndarray | None] = [None]

        for severity in [1, 2, 3, 4, 5]:
            clouded = cloud_batches[severity][row]
            clouded_norm = normalizer(clouded.unsqueeze(0)).squeeze(0)
            row_raw.append(
                _render_rgb(clouded, low=raw_low, high=raw_high)
            )
            row_norm.append(
                _render_rgb(clouded_norm, low=norm_low, high=norm_high)
            )
            delta_luma = (clouded_norm - clean_norm).mean(dim=0).detach().cpu().numpy().astype(np.float32)
            row_delta.append(delta_luma)
            row_alpha.append(cloud_masks[severity][row].detach().cpu().numpy().astype(np.float32))

        for severity in [1, 2, 3, 4, 5]:
            noised = None if noise_batches[severity] is None else noise_batches[severity][row]
            if noised is None:
                row_raw.append(None)
                row_norm.append(None)
                row_delta.append(None)
                row_alpha.append(None)
                continue
            noised_norm = normalizer(noised.unsqueeze(0)).squeeze(0)
            row_raw.append(
                _render_rgb(noised, low=raw_low, high=raw_high)
            )
            row_norm.append(
                _render_rgb(noised_norm, low=norm_low, high=norm_high)
            )
            delta_luma = (noised_norm - clean_norm).mean(dim=0).detach().cpu().numpy().astype(np.float32)
            row_delta.append(delta_luma)
            row_alpha.append(None)

        raw_variants.append(row_raw)
        norm_variants.append(row_norm)
        delta_variants.append(row_delta)
        alpha_variants.append(row_alpha)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{dataset_name}_corruptions_raw.png"
    norm_path = out_dir / f"{dataset_name}_corruptions_norm.png"
    delta_path = out_dir / f"{dataset_name}_corruptions_delta.png"
    alpha_path = out_dir / f"{dataset_name}_corruptions_alpha.png"
    stats_path = out_dir / f"{dataset_name}_corruptions_stats.json"

    _render_grid(raw_variants, out_path=out_path, is_rgb=True)
    _render_grid(norm_variants, out_path=norm_path, is_rgb=True)
    _render_grid(
        delta_variants,
        out_path=delta_path,
        cmap="coolwarm",
        vmin=-delta_bound,
        vmax=delta_bound,
        is_rgb=False,
    )
    _render_grid(
        alpha_variants,
        out_path=alpha_path,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        is_rgb=False,
    )
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


def generate_histograms(
    dataset_name: str,
    samples: torch.Tensor,
    band_specs: list[BandSpec],
    out_dir: str | Path,
    n_samples: int = 4,
    rgb_indices: list[int] | None = None,
    seed: int = 0,
    cloud_pattern_mode: str = "fixed",
    n_bins: int = 60,
) -> Path:
    """Render pixel-value distribution histograms across corruption severities.

    Produces one figure per corruption type (cloud, noise), each showing
    overlaid per-channel histograms for clean + severities 1–5 in
    z-score-normalized space. Channels are plotted in separate rows.

    Args:
        dataset_name: Benchmark dataset name.
        samples: Input samples tensor with shape ``(N, C, H, W)``.
        band_specs: Per-channel band metadata.
        out_dir: Output directory for the PNG files.
        n_samples: Number of samples to use from ``samples``.
        rgb_indices: Optional RGB channel indices to plot (defaults to [0,1,2]).
        seed: Base random seed for corruption generation.
        cloud_pattern_mode: Cloud RNG mode.
        n_bins: Number of histogram bins.

    Returns:
        Path to the cloud histogram PNG (noise histogram written alongside it).
    """
    plt = _import_plotting()

    if rgb_indices is None:
        rgb_indices = [0, 1, 2]

    samples = samples[:n_samples].detach().clone()
    display_indices = _resolve_display_indices(samples[0], rgb_indices)
    channel_names = [band_specs[i].name for i in display_indices]

    normalizer = build_normalizer(NormalizationStrategy.BANDSPEC_ZSCORE, bands=band_specs)

    # Collect normalized pixels per severity for each corruption type.
    # Shape: list of (N*H*W,) arrays, indexed by severity 0=clean,1..5=corrupted.
    def _collect_pixels(batches: dict[int, torch.Tensor | None]) -> dict[int, list[np.ndarray]]:
        """Return {severity: [ch0_pixels, ch1_pixels, ch2_pixels]}."""
        result: dict[int, list[np.ndarray]] = {}
        clean_norm = normalizer(samples)
        result[0] = [
            clean_norm[:, display_indices[c]].detach().cpu().numpy().ravel()
            for c in range(3)
        ]
        for sev, batch in batches.items():
            if batch is None:
                continue
            normed = normalizer(batch)
            result[sev] = [
                normed[:, display_indices[c]].detach().cpu().numpy().ravel()
                for c in range(3)
            ]
        return result

    # --- Cloud ---
    cloud_batches: dict[int, torch.Tensor] = {}
    for severity in [1, 2, 3, 4, 5]:
        cloud_t = CorruptionTransform(
            "cloud", severity, seed, band_specs,
            dataset_name=dataset_name, cloud_pattern_mode=cloud_pattern_mode,
        )
        cloud_batches[severity] = cloud_t(samples)

    # --- Noise ---
    noise_skipped = dataset_name in SKIP_POISSON_GAUSSIAN
    noise_batches: dict[int, torch.Tensor | None] = {}
    for severity in [1, 2, 3, 4, 5]:
        if noise_skipped:
            noise_batches[severity] = None
            continue
        noise_t = CorruptionTransform(
            "poisson_gaussian", severity, seed, band_specs, dataset_name=dataset_name,
        )
        noise_batches[severity] = noise_t(samples)

    cloud_pixels = _collect_pixels(cloud_batches)
    noise_pixels = _collect_pixels(noise_batches)

    severity_colors = {
        0: "#333333",  # clean
        1: "#2196F3",
        2: "#4CAF50",
        3: "#FF9800",
        4: "#F44336",
        5: "#9C27B0",
    }
    severity_labels = {0: "clean", 1: "sev 1", 2: "sev 2", 3: "sev 3", 4: "sev 4", 5: "sev 5"}

    def _render_histogram_figure(
        pixels_by_severity: dict[int, list[np.ndarray]],
        corruption_label: str,
        out_path: Path,
    ) -> None:
        n_channels = 3
        fig, axes = plt.subplots(
            n_channels, 1, figsize=(9, 3 * n_channels), sharex=False, sharey=False
        )
        if n_channels == 1:
            axes = [axes]

        # Determine per-channel x-range from clean pixels.
        clean_ch = pixels_by_severity[0]

        for ch in range(n_channels):
            ax = axes[ch]
            # Bin range: cover clean + all severities' range.
            all_vals = np.concatenate(
                [pxs[ch] for pxs in pixels_by_severity.values() if pxs is not None]
            )
            all_vals = all_vals[np.isfinite(all_vals)]
            lo = float(np.percentile(all_vals, 0.5))
            hi = float(np.percentile(all_vals, 99.5))
            if hi <= lo:
                hi = lo + 1.0
            bins = np.linspace(lo, hi, n_bins + 1)

            for sev in sorted(pixels_by_severity.keys()):
                pxs = pixels_by_severity[sev]
                if pxs is None:
                    continue
                vals = pxs[ch]
                vals = vals[np.isfinite(vals)]
                counts, edges = np.histogram(vals, bins=bins)
                centers = 0.5 * (edges[:-1] + edges[1:])
                density = counts / max(counts.max(), 1)
                lw = 2.0 if sev == 0 else 1.4
                alpha = 0.9 if sev == 0 else 0.75
                ax.plot(
                    centers, density,
                    color=severity_colors[sev],
                    label=severity_labels[sev],
                    linewidth=lw,
                    alpha=alpha,
                )

            ax.set_ylabel("rel. freq.")
            ax.set_xlabel(f"{channel_names[ch]} (z-score)")
            ax.set_title(f"{channel_names[ch]} — {corruption_label}")
            ax.legend(fontsize=8, ncol=3, loc="upper right")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        fig.suptitle(f"{dataset_name} · {corruption_label} · pixel distributions (z-score)", fontsize=11)
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cloud_hist_path = out_dir / f"{dataset_name}_hist_cloud.png"
    _render_histogram_figure(cloud_pixels, "cloud", cloud_hist_path)

    if not noise_skipped:
        noise_hist_path = out_dir / f"{dataset_name}_hist_noise.png"
        _render_histogram_figure(noise_pixels, "noise", noise_hist_path)

    return cloud_hist_path


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
        choices=["fixed", "independent"],
        default="fixed",
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
    hist_path = generate_histograms(
        dataset_name=args.dataset,
        samples=samples,
        band_specs=band_specs,
        out_dir=args.out,
        n_samples=args.n_samples,
        rgb_indices=rgb_indices,
        seed=args.seed,
        cloud_pattern_mode=args.cloud_pattern_mode,
    )
    logger.info("Saved corruption histograms to %s", hist_path)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
