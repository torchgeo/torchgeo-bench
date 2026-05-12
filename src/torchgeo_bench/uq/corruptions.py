"""Image corruption transforms used for UQ robustness evaluation."""

import math

import numpy as np
import torch
import torch.nn.functional as F
from opensimplex import OpenSimplex

from torchgeo_bench.datasets.base import BandSpec

SKIP_POISSON_GAUSSIAN: frozenset[str] = frozenset({"m-so2sat", "so2sat"})
CLOUD_SIGMAS: list[float] = [2, 4, 6, 8, 10]
SENSOR_NOISE_PARAMS: dict[str, tuple[float, float]] = {
    "s2": (8e-5, 0.02),
    "landsat": (1e-4, 0.05),
    "aerial": (5e-5, 0.02),
}
NOISE_SCALES: list[float] = [1.0, 2.0, 4.0, 8.0, 16.0]


def _gaussian_blur2d(mask: torch.Tensor, sigma_px: float) -> torch.Tensor:
    """Apply a Gaussian blur to a 2D mask.

    Args:
        mask: Input mask with shape ``(H, W)``.
        sigma_px: Gaussian sigma in pixels.

    Returns:
        Blurred mask with shape ``(H, W)`` and values in ``[0, 1]``.
    """
    if sigma_px <= 0.0:
        return mask
    radius = int(max(1, math.ceil(3.0 * sigma_px)))
    size = 2 * radius + 1
    coords = torch.arange(-radius, radius + 1, device=mask.device, dtype=mask.dtype)
    kernel_1d = torch.exp(-(coords**2) / (2.0 * sigma_px**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d).view(1, 1, size, size)
    out = F.conv2d(mask.view(1, 1, *mask.shape), kernel_2d, padding=radius)
    return out.view_as(mask).clamp(0.0, 1.0)


class CorruptionTransform:
    """Apply cloud/shadow or Poisson-Gaussian corruptions to image batches.

    Args:
        corruption_type: One of ``"cloud_shadow"`` or ``"poisson_gaussian"``.
        severity: Corruption severity in ``[1, 5]``.
        seed: Base seed used for deterministic per-image corruption.
        band_specs: Per-channel statistics and sensor metadata.
    """

    def __init__(
        self,
        corruption_type: str,
        severity: int,
        seed: int,
        band_specs: list[BandSpec],
    ) -> None:
        if corruption_type not in {"cloud_shadow", "poisson_gaussian"}:
            raise ValueError(
                f"Unknown corruption_type={corruption_type!r}; expected cloud_shadow or "
                "poisson_gaussian"
            )
        if severity < 1 or severity > 5:
            raise ValueError(f"severity must be in [1, 5], got {severity}")
        if not band_specs:
            raise ValueError("band_specs must be non-empty")

        self.corruption_type = corruption_type
        self.severity = severity
        self.seed = int(seed)
        self.band_specs = list(band_specs)
        self._n_images_seen = 0

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """Apply corruption to a batch of images.

        Args:
            images: Image batch with shape ``(B, C, H, W)``.

        Returns:
            Corrupted image batch with the same shape and dtype as input.
        """
        if images.ndim != 4:
            raise ValueError(f"images must be 4D (B, C, H, W), got shape {tuple(images.shape)}")
        if images.shape[1] != len(self.band_specs):
            raise ValueError(
                f"Expected {len(self.band_specs)} channels from band_specs, got {images.shape[1]}"
            )

        in_dtype = images.dtype
        out = images.detach().clone().to(dtype=torch.float32)
        bsz = int(out.shape[0])

        for i in range(bsz):
            global_idx = self._n_images_seen + i
            if self.corruption_type == "cloud_shadow":
                out[i] = self._apply_cloud_shadow(out[i], global_idx)
            else:
                out[i] = self._apply_poisson_gaussian(out[i], global_idx)

        self._n_images_seen += bsz
        return out.to(dtype=in_dtype)

    def _apply_cloud_shadow(self, image: torch.Tensor, global_idx: int) -> torch.Tensor:
        """Apply cloud and shifted shadow corruption to one image.

        Args:
            image: Input image with shape ``(C, H, W)``.
            global_idx: Global image index used for deterministic seeding.

        Returns:
            Corrupted image with shape ``(C, H, W)``.
        """
        c, h, w = image.shape
        simplex = OpenSimplex(seed=self.seed + global_idx)

        spatial_scale = max(8.0, max(h, w) / (0.75 + 0.25 * self.severity))
        noise = np.empty((h, w), dtype=np.float32)
        for yy in range(h):
            for xx in range(w):
                noise[yy, xx] = simplex.noise2(x=xx / spatial_scale, y=yy / spatial_scale)

        coverage_by_severity = [0.12, 0.2, 0.28, 0.36, 0.45]
        cloud_frac = coverage_by_severity[self.severity - 1]
        threshold = float(np.quantile(noise, 1.0 - cloud_frac))
        cloud_mask = torch.from_numpy((noise >= threshold).astype(np.float32)).to(image.device)
        cloud_mask = _gaussian_blur2d(cloud_mask, sigma_px=0.5 + 0.35 * self.severity)

        shadow_dx = int(round(1 + self.severity))
        shadow_dy = int(round(1 + 0.5 * self.severity))
        shadow_mask = torch.zeros_like(cloud_mask)
        y_src = slice(0, h - shadow_dy)
        y_dst = slice(shadow_dy, h)
        x_src = slice(0, w - shadow_dx)
        x_dst = slice(shadow_dx, w)
        shadow_mask[y_dst, x_dst] = cloud_mask[y_src, x_src]
        shadow_mask = _gaussian_blur2d(shadow_mask, sigma_px=0.4 + 0.25 * self.severity)

        cloud_sigma = float(CLOUD_SIGMAS[self.severity - 1])
        out = image.clone()
        for chan in range(c):
            band = self.band_specs[chan]
            cloud_value = band.mean + cloud_sigma * band.std
            out_chan = out[chan]
            out_chan = out_chan * (1.0 - cloud_mask) + cloud_mask * cloud_value
            out_chan = out_chan * (1.0 - shadow_mask) + shadow_mask * (0.4 * out_chan)
            out[chan] = out_chan.clamp(min=0.0, max=band.max)
        return out

    def _apply_poisson_gaussian(self, image: torch.Tensor, global_idx: int) -> torch.Tensor:
        """Apply sensor-aware Poisson-Gaussian noise to one image.

        Args:
            image: Input image with shape ``(C, H, W)``.
            global_idx: Global image index used for deterministic seeding.

        Returns:
            Corrupted image with shape ``(C, H, W)``.
        """
        c, _, _ = image.shape
        device = image.device
        dtype = image.dtype
        scale = NOISE_SCALES[self.severity - 1]

        alpha_list: list[float] = []
        sigma_list: list[float] = []
        max_list: list[float] = []
        for band in self.band_specs:
            sensor = band.sensor if band.sensor in SENSOR_NOISE_PARAMS else "aerial"
            alpha_frac, sigma_frac = SENSOR_NOISE_PARAMS[sensor]
            alpha_list.append(max(alpha_frac * band.mean * scale, 1e-12))
            sigma_list.append(max(sigma_frac * band.std * scale, 0.0))
            max_list.append(band.max)

        alpha = torch.tensor(alpha_list, device=device, dtype=dtype).view(c, 1, 1)
        sigma = torch.tensor(sigma_list, device=device, dtype=dtype).view(c, 1, 1)
        max_vals = torch.tensor(max_list, device=device, dtype=dtype).view(c, 1, 1)

        gen = torch.Generator(device=device)
        gen.manual_seed(self.seed + global_idx)

        shot = torch.poisson((image.clamp(min=0.0) / alpha).clamp(min=0.0), generator=gen) * alpha
        readout = sigma * torch.randn(image.shape, device=device, dtype=dtype, generator=gen)
        out = shot + readout
        out = torch.maximum(out, torch.zeros_like(out))
        out = torch.minimum(out, max_vals)
        return out
