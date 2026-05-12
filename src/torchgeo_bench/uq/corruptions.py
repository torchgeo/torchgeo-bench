"""Image corruption transforms used for UQ robustness evaluation."""

from dataclasses import dataclass

import torch

from torchgeo_bench.datasets.base import BandSpec

SKIP_POISSON_GAUSSIAN: frozenset[str] = frozenset({"m-so2sat", "so2sat"})
SENSOR_NOISE_PARAMS: dict[str, tuple[float, float]] = {
    "s2": (8e-5, 0.02),
    "landsat": (1e-4, 0.05),
    "aerial": (5e-5, 0.02),
}
NOISE_SCALES: list[float] = [1.0, 2.0, 4.0, 8.0, 16.0]


@dataclass(frozen=True)
class CloudSeverityPreset:
    """Cloud synthesis settings for one severity."""

    min_lvl: tuple[float, float]
    max_lvl: tuple[float, float]
    clear_threshold: float
    decay_factor: float = 1.25
    locality_degree: int = 1
    channel_offset: int = 0
    blur_scaling: float = 0.0
    cloud_color: bool = True
    channel_magnitude_shift: float = 0.02


@dataclass(frozen=True)
class CloudDatasetCalibration:
    """Dataset-local calibration for cloud synthesis."""

    optical_band_names: tuple[str, ...]
    lower_fracs: tuple[float, ...]
    upper_fracs: tuple[float, ...]
    severity_presets: dict[int, CloudSeverityPreset]


CLOUD_SEVERITY_PRESETS: dict[int, CloudSeverityPreset] = {
    # SatelliteCloudGenerator uses clear_threshold as cloud coverage gate:
    # lower threshold -> more retained cloud mask (0.0 is maximal cloud coverage).
    # Severity is therefore driven primarily by decreasing clear_threshold.
    1: CloudSeverityPreset(min_lvl=(0.00, 0.02), max_lvl=(0.45, 0.60), clear_threshold=0.98),
    2: CloudSeverityPreset(min_lvl=(0.00, 0.02), max_lvl=(0.45, 0.60), clear_threshold=0.9),
    3: CloudSeverityPreset(min_lvl=(0.00, 0.02), max_lvl=(0.45, 0.60), clear_threshold=0.8),
    4: CloudSeverityPreset(min_lvl=(0.00, 0.02), max_lvl=(0.50, 0.65), clear_threshold=0.6),
    5: CloudSeverityPreset(min_lvl=(0.00, 0.02), max_lvl=(0.55, 0.72), clear_threshold=0.4),
}

CLOUD_DATASET_CALIBRATIONS: dict[str, CloudDatasetCalibration] = {
    "m-eurosat": CloudDatasetCalibration(
        optical_band_names=("red", "green", "blue"),
        lower_fracs=(0.02, 0.02, 0.02),
        upper_fracs=(0.98, 0.98, 0.98),
        severity_presets=CLOUD_SEVERITY_PRESETS,
    ),
    "m-forestnet": CloudDatasetCalibration(
        optical_band_names=("red", "green", "blue"),
        lower_fracs=(0.02, 0.02, 0.02),
        upper_fracs=(0.98, 0.98, 0.98),
        severity_presets=CLOUD_SEVERITY_PRESETS,
    ),
    "m-so2sat": CloudDatasetCalibration(
        optical_band_names=("red", "green", "blue"),
        lower_fracs=(0.02, 0.02, 0.02),
        upper_fracs=(0.98, 0.98, 0.98),
        severity_presets=CLOUD_SEVERITY_PRESETS,
    ),
    "m-pv4ger": CloudDatasetCalibration(
        optical_band_names=("red", "green", "blue"),
        lower_fracs=(0.02, 0.02, 0.02),
        upper_fracs=(0.98, 0.98, 0.98),
        severity_presets=CLOUD_SEVERITY_PRESETS,
    ),
    "m-brick-kiln": CloudDatasetCalibration(
        optical_band_names=("red", "green", "blue"),
        lower_fracs=(0.02, 0.02, 0.02),
        upper_fracs=(0.98, 0.98, 0.98),
        severity_presets=CLOUD_SEVERITY_PRESETS,
    ),
    "forestnet": CloudDatasetCalibration(
        optical_band_names=("b04", "b03", "b02"),
        lower_fracs=(0.02, 0.02, 0.02),
        upper_fracs=(0.98, 0.98, 0.98),
        severity_presets=CLOUD_SEVERITY_PRESETS,
    ),
    "so2sat": CloudDatasetCalibration(
        optical_band_names=("b04", "b03", "b02"),
        lower_fracs=(0.02, 0.02, 0.02),
        upper_fracs=(0.98, 0.98, 0.98),
        severity_presets=CLOUD_SEVERITY_PRESETS,
    ),
    "eurosat": CloudDatasetCalibration(
        optical_band_names=("red", "green", "blue"),
        lower_fracs=(0.02, 0.02, 0.02),
        upper_fracs=(0.98, 0.98, 0.98),
        severity_presets=CLOUD_SEVERITY_PRESETS,
    ),
    "eurosat-spatial": CloudDatasetCalibration(
        optical_band_names=("red", "green", "blue"),
        lower_fracs=(0.02, 0.02, 0.02),
        upper_fracs=(0.98, 0.98, 0.98),
        severity_presets=CLOUD_SEVERITY_PRESETS,
    ),
}


def _load_satellite_cloud_generator():
    """Load SatelliteCloudGenerator dependency for cloud synthesis."""
    try:
        import satellite_cloud_generator as scg
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "satellite-cloud-generator is required for cloud corruption. "
            "Install with `torchgeo-bench[uq]` or `uv sync --extra uq`."
        ) from exc
    return scg


def _seed_devices(device: torch.device) -> list[int]:
    """Return CUDA device list suitable for ``torch.random.fork_rng``."""
    if device.type != "cuda":
        return []
    if device.index is not None:
        return [device.index]
    return [torch.cuda.current_device()]


def _tensor_range(band_specs: list[BandSpec], device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-channel min/max tensors with shape ``(C, 1, 1)``."""
    min_vals = torch.tensor([band.min for band in band_specs], device=device, dtype=dtype).view(-1, 1, 1)
    max_vals = torch.tensor([band.max for band in band_specs], device=device, dtype=dtype).view(-1, 1, 1)
    return min_vals, max_vals


def _resolve_cloud_calibration(
    *,
    dataset_name: str,
    band_specs: list[BandSpec],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[list[int], torch.Tensor, torch.Tensor, CloudDatasetCalibration]:
    """Resolve optical channels plus synthesis bounds for a dataset."""
    if dataset_name not in CLOUD_DATASET_CALIBRATIONS:
        supported = ", ".join(sorted(CLOUD_DATASET_CALIBRATIONS))
        raise ValueError(
            f"No cloud calibration registered for dataset {dataset_name!r}. "
            f"Supported datasets: {supported}"
        )
    calibration = CLOUD_DATASET_CALIBRATIONS[dataset_name]
    if len(calibration.optical_band_names) != len(calibration.lower_fracs) or len(
        calibration.optical_band_names
    ) != len(calibration.upper_fracs):
        raise ValueError(f"Invalid cloud calibration for dataset {dataset_name!r}: size mismatch.")

    by_name = {band.name: (idx, band) for idx, band in enumerate(band_specs)}
    optical_indices: list[int] = []
    lower_vals: list[float] = []
    upper_vals: list[float] = []

    for band_name, lo_frac, hi_frac in zip(
        calibration.optical_band_names,
        calibration.lower_fracs,
        calibration.upper_fracs,
        strict=True,
    ):
        if band_name not in by_name:
            available = ", ".join(sorted(by_name))
            raise ValueError(
                f"Dataset {dataset_name!r}: missing calibrated optical band {band_name!r}. "
                f"Available loaded bands: {available}"
            )
        if not 0.0 <= lo_frac < hi_frac <= 1.0:
            raise ValueError(
                f"Dataset {dataset_name!r}: invalid calibration fractions for {band_name!r}: "
                f"lower={lo_frac}, upper={hi_frac}"
            )
        idx, band = by_name[band_name]
        span = band.max - band.min
        if span <= 0:
            raise ValueError(
                f"Dataset {dataset_name!r}: non-positive band range for {band_name!r}: "
                f"min={band.min}, max={band.max}"
            )

        optical_indices.append(idx)
        lower_vals.append(band.min + lo_frac * span)
        upper_vals.append(band.min + hi_frac * span)

    lower = torch.tensor(lower_vals, device=device, dtype=dtype).view(-1, 1, 1)
    upper = torch.tensor(upper_vals, device=device, dtype=dtype).view(-1, 1, 1)
    return optical_indices, lower, upper, calibration


class CorruptionTransform:
    """Apply cloud or Poisson-Gaussian corruptions to image batches.

    Args:
        corruption_type: One of ``"cloud"`` or ``"poisson_gaussian"``.
        severity: Corruption severity in ``[1, 5]``.
        seed: Base seed used for deterministic per-image corruption.
        band_specs: Per-channel statistics and sensor metadata.
        dataset_name: Dataset key used by cloud calibration.
        cloud_pattern_mode: Cloud RNG mode. ``"fixed_across_severity"`` reuses
            the same per-image cloud realization across severities. ``"independent_per_severity"``
            samples a different realization for each severity.
    """

    def __init__(
        self,
        corruption_type: str,
        severity: int,
        seed: int,
        band_specs: list[BandSpec],
        dataset_name: str | None = None,
        cloud_pattern_mode: str = "fixed_across_severity",
    ) -> None:
        if corruption_type not in {"cloud", "poisson_gaussian"}:
            raise ValueError(
                f"Unknown corruption_type={corruption_type!r}; expected cloud or poisson_gaussian"
            )
        if severity < 1 or severity > 5:
            raise ValueError(f"severity must be in [1, 5], got {severity}")
        if not band_specs:
            raise ValueError("band_specs must be non-empty")
        if corruption_type == "cloud" and not dataset_name:
            raise ValueError("dataset_name is required for cloud corruption.")
        if cloud_pattern_mode not in {"fixed_across_severity", "independent_per_severity"}:
            raise ValueError(
                "cloud_pattern_mode must be one of "
                "{'fixed_across_severity', 'independent_per_severity'}"
            )

        self.corruption_type = corruption_type
        self.severity = severity
        self.seed = int(seed)
        self.band_specs = list(band_specs)
        self.dataset_name = dataset_name
        self.cloud_pattern_mode = cloud_pattern_mode
        self._n_images_seen = 0
        self._scg = None

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
            if self.corruption_type == "cloud":
                out[i] = self._apply_cloud(out[i], global_idx)
            else:
                out[i] = self._apply_poisson_gaussian(out[i], global_idx)

        self._n_images_seen += bsz
        return out.to(dtype=in_dtype)

    def _apply_cloud(self, image: torch.Tensor, global_idx: int) -> torch.Tensor:
        """Apply optical-only cloud synthesis to one image."""
        assert self.dataset_name is not None
        device = image.device
        dtype = image.dtype

        optical_indices, lower, upper, calibration = _resolve_cloud_calibration(
            dataset_name=self.dataset_name,
            band_specs=self.band_specs,
            device=device,
            dtype=dtype,
        )
        if self.severity not in calibration.severity_presets:
            raise ValueError(
                f"Dataset {self.dataset_name!r}: missing severity preset for severity={self.severity}."
            )
        preset = calibration.severity_presets[self.severity]
        if self._scg is None:
            self._scg = _load_satellite_cloud_generator()

        out = image.clone()
        optical = out[optical_indices]
        denom = (upper - lower).clamp(min=1e-6)
        synth_input = ((optical - lower) / denom).clamp(0.0, 1.0).unsqueeze(0)

        synth_seed = self.seed + global_idx
        if self.cloud_pattern_mode == "independent_per_severity":
            synth_seed += 1_000_000 * self.severity
        with torch.random.fork_rng(devices=_seed_devices(device), enabled=True):
            torch.manual_seed(synth_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(synth_seed)
            synth_output = self._scg.add_cloud(
                synth_input,
                min_lvl=preset.min_lvl,
                max_lvl=preset.max_lvl,
                clear_threshold=preset.clear_threshold,
                noise_type="perlin",
                const_scale=True,
                decay_factor=preset.decay_factor,
                locality_degree=preset.locality_degree,
                channel_offset=preset.channel_offset,
                blur_scaling=preset.blur_scaling,
                cloud_color=preset.cloud_color,
                channel_magnitude_shift=preset.channel_magnitude_shift,
            )

        synth_output = synth_output.squeeze(0).clamp(0.0, 1.0)
        out[optical_indices] = synth_output * denom + lower

        min_vals, max_vals = _tensor_range(self.band_specs, device=device, dtype=dtype)
        out[optical_indices] = out[optical_indices].clamp(
            min=min_vals[optical_indices], max=max_vals[optical_indices]
        )
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
        for band in self.band_specs:
            sensor = band.sensor if band.sensor in SENSOR_NOISE_PARAMS else "aerial"
            alpha_frac, sigma_frac = SENSOR_NOISE_PARAMS[sensor]
            alpha_list.append(max(alpha_frac * max(band.mean, 1e-6) * scale, 1e-12))
            sigma_list.append(max(sigma_frac * max(band.std, 1e-6) * scale, 0.0))

        alpha = torch.tensor(alpha_list, device=device, dtype=dtype).view(c, 1, 1)
        sigma = torch.tensor(sigma_list, device=device, dtype=dtype).view(c, 1, 1)
        min_vals, max_vals = _tensor_range(self.band_specs, device=device, dtype=dtype)

        gen = torch.Generator(device=device)
        gen.manual_seed(self.seed + global_idx)

        shot = torch.poisson((image.clamp(min=0.0) / alpha).clamp(min=0.0), generator=gen) * alpha
        readout = sigma * torch.randn(image.shape, device=device, dtype=dtype, generator=gen)
        out = shot + readout
        out = torch.maximum(out, min_vals)
        out = torch.minimum(out, max_vals)
        return out
