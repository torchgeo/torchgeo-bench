"""Image corruption transforms used for UQ robustness evaluation."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from torchgeo_bench.datasets.base import BandSpec

MOTION_BLUR_KERNEL_SIZES: dict[int, int] = {1: 3, 2: 5, 3: 9, 4: 15, 5: 21}

SKIP_POISSON_GAUSSIAN: frozenset[str] = frozenset()


@dataclass(frozen=True)
class _NoiseSeverityPreset:
    """Sensor noise settings for one severity level."""

    photon_count: float
    read_std_frac: float


@dataclass(frozen=True)
class _NoiseSensorCalibration:
    """Per-sensor calibration for Poisson-Gaussian corruption."""

    severity_presets: dict[int, _NoiseSeverityPreset]


@dataclass(frozen=True)
class _NoiseDatasetOverride:
    """Dataset-local override ladder for Poisson-Gaussian corruption."""

    severity_presets: dict[int, _NoiseSeverityPreset]


def _build_noise_severity_presets(levels: tuple[tuple[float, float], ...]) -> dict[int, _NoiseSeverityPreset]:
    """Build five calibrated severity presets from ``(photon_count, read_std_frac)`` pairs."""
    if len(levels) != 5:
        raise ValueError(f"Expected five noise severity levels, got {len(levels)}")

    presets = {
        severity: _NoiseSeverityPreset(photon_count=photon_count, read_std_frac=read_std_frac)
        for severity, (photon_count, read_std_frac) in enumerate(levels, start=1)
    }
    for severity, preset in presets.items():
        if preset.photon_count <= 0:
            raise ValueError(
                f"Noise preset severity {severity} must use positive photon_count, got "
                f"{preset.photon_count}"
            )
        if preset.read_std_frac < 0:
            raise ValueError(
                f"Noise preset severity {severity} must use non-negative read_std_frac, got "
                f"{preset.read_std_frac}"
            )
    return presets


NOISE_SENSOR_CALIBRATIONS: dict[str, _NoiseSensorCalibration] = {
    # Lower photon counts increase shot noise; read_std_frac controls additive readout noise
    # in normalized [0, 1] space before mapping back to native band ranges.
    "s2": _NoiseSensorCalibration(
        severity_presets=_build_noise_severity_presets(
            (
                (6000.0, 0.01),
                (3500.0, 0.014),
                (1800.0, 0.02),
                (900.0, 0.03),
                (450.0, 0.07),
            )
        )
    ),
    "landsat": _NoiseSensorCalibration(
        severity_presets=_build_noise_severity_presets(
            (
                (5950.0, 0.0030),
                (3825.0, 0.0045),
                (2125.0, 0.0065),
                (1020.0, 0.0095),
                (467.5, 0.0140),
            )
        )
    ),
    "aerial": _NoiseSensorCalibration(
        severity_presets=_build_noise_severity_presets(
            (
                (3600.0, 0.0018),
                (826.47, 0.15135),
                (189.74, 0.30090),
                (43.56, 0.45045),
                (10.0, 0.60000),
            )
        )
    ),
}


NOISE_DATASET_OVERRIDES: dict[str, _NoiseDatasetOverride] = {
    "so2sat": _NoiseDatasetOverride(
        severity_presets=_build_noise_severity_presets(
            (
                (5737.5, 0.002188),
                (3825.0, 0.0089141),
                (2231.25, 0.01),
                (1147.5, 0.02),
                (200.0, 0.07),
            )
        )
    ),
    "m-so2sat": _NoiseDatasetOverride(
        severity_presets=_build_noise_severity_presets(
            (
                (5737.5, 0.002188),
                (3825.0, 0.0089141),
                (2231.25, 0.01),
                (1147.5, 0.02),
                (200.0, 0.07),
            )
        )
    ),
}


@dataclass(frozen=True)
class CloudSeverityPreset:
    """Cloud synthesis settings for one severity."""

    min_lvl: tuple[float, float]
    max_lvl: tuple[float, float]
    clear_threshold: float
    decay_factor: float = 2.0
    locality_degree: int = 1
    channel_offset: int = 1
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


CLOUD_CLEAR_THRESHOLDS: dict[int, float] = {
    # SatelliteCloudGenerator uses clear_threshold as a coverage gate:
    # lower threshold -> more retained cloud mask (0.0 is maximal coverage).
    1: 0.9,
    2: 0.65,
    3: 0.4,
    4: 0.25,
    5: 0,
}


def _fixed_cloud_preset(max_lvl: float, clear_threshold: float) -> CloudSeverityPreset:
    """Build a cloud preset with zero haze floor outside retained clouds."""
    return CloudSeverityPreset(
        min_lvl=(0.0, 0.0),
        max_lvl=(max_lvl, max_lvl),
        clear_threshold=clear_threshold,
    )


def _build_cloud_severity_presets(max_lvls: tuple[float, ...]) -> dict[int, CloudSeverityPreset]:
    """Build a five-level opacity ladder while keeping coverage global."""
    if len(max_lvls) != 5:
        raise ValueError(f"Expected five cloud opacity levels, got {len(max_lvls)}")
    return {
        severity: _fixed_cloud_preset(max_lvl=max_lvl, clear_threshold=CLOUD_CLEAR_THRESHOLDS[severity])
        for severity, max_lvl in enumerate(max_lvls, start=1)
    }

CLOUD_DATASET_CALIBRATIONS: dict[str, CloudDatasetCalibration] = {
    "m-eurosat": CloudDatasetCalibration(
        optical_band_names=("red", "green", "blue"),
        lower_fracs=(0.02, 0.02, 0.02),
        # S2 bands have max=28000 but typical pixels cluster around mean~1000 DN.
        # upper_fracs narrowed so norm_mean≈0.30, matching aerial datasets and giving
        # realistic cloud contrast (~0.5–3.5×) rather than the 20–27× contrast from 0.98.
        upper_fracs=(0.0673, 0.0789, 0.0873),
        severity_presets=_build_cloud_severity_presets((0.45, 0.60, 0.75, 0.96, 1.35)),
    ),
    "m-forestnet": CloudDatasetCalibration(
        optical_band_names=("red", "green", "blue"),
        lower_fracs=(0.02, 0.02, 0.02),
        upper_fracs=(0.98, 0.98, 0.98),
        severity_presets=_build_cloud_severity_presets((0.48, 0.56, 0.64, 0.72, 0.90)),
    ),
    "m-so2sat": CloudDatasetCalibration(
        optical_band_names=("red", "green", "blue"),
        lower_fracs=(0.02, 0.02, 0.02),
        # S2 bands max=2.8 but typical pixels at mean≈0.11–0.13; same calibration logic as so2sat.
        upper_fracs=(0.0887, 0.0927, 0.1074),
        severity_presets=_build_cloud_severity_presets((0.45, 0.60, 0.75, 0.96, 1.35)),
    ),
    "m-pv4ger": CloudDatasetCalibration(
        optical_band_names=("red", "green", "blue"),
        lower_fracs=(0.02, 0.02, 0.02),
        upper_fracs=(0.98, 0.98, 0.98),
        severity_presets=_build_cloud_severity_presets((0.8, 1.0, 1.0, 1.25, 2.0)),
    ),
    "m-brick-kiln": CloudDatasetCalibration(
        optical_band_names=("red", "green", "blue"),
        lower_fracs=(0.02, 0.02, 0.02),
        upper_fracs=(0.98, 0.98, 0.98),
        severity_presets=_build_cloud_severity_presets((0.52, 0.60, 0.68, 0.76, 0.90)),
    ),
    "forestnet": CloudDatasetCalibration(
        optical_band_names=("b04", "b03", "b02"),
        lower_fracs=(0.02, 0.02, 0.02),
        upper_fracs=(0.98, 0.98, 0.98),
        severity_presets=_build_cloud_severity_presets((0.46, 0.54, 0.62, 0.70, 0.90)),
    ),
    "so2sat": CloudDatasetCalibration(
        optical_band_names=("b04", "b03", "b02"),
        lower_fracs=(0.02, 0.02, 0.02),
        # S2 bands max=2.8 but typical pixels at mean≈0.11–0.13 (b04/b03/b02 order).
        # upper_fracs narrowed so norm_mean≈0.30, giving realistic cloud contrast.
        upper_fracs=(0.0887, 0.0927, 0.1074),
        severity_presets=_build_cloud_severity_presets((0.45, 0.60, 0.75, 0.96, 1.35)),
    ),
    "eurosat": CloudDatasetCalibration(
        optical_band_names=("red", "green", "blue"),
        lower_fracs=(0.02, 0.02, 0.02),
        # Same S2 scale as m-eurosat (max=28000, mean~950–1120 DN); apply same recalibration.
        upper_fracs=(0.0673, 0.0789, 0.0873),
        severity_presets=_build_cloud_severity_presets((0.45, 0.60, 0.75, 0.96, 1.35)),
    ),
    "eurosat-spatial": CloudDatasetCalibration(
        optical_band_names=("red", "green", "blue"),
        lower_fracs=(0.02, 0.02, 0.02),
        upper_fracs=(0.0673, 0.0789, 0.0873),
        severity_presets=_build_cloud_severity_presets((0.45, 0.60, 0.75, 0.96, 1.35)),
    ),
    "advance": CloudDatasetCalibration(
        optical_band_names=("red", "green", "blue"),
        lower_fracs=(0.02, 0.02, 0.02),
        upper_fracs=(0.98, 0.98, 0.98),
        severity_presets=_build_cloud_severity_presets((0.8, 1.0, 1.0, 1.25, 2.0)),
    ),
    "resisc45": CloudDatasetCalibration(
        optical_band_names=("red", "green", "blue"),
        lower_fracs=(0.02, 0.02, 0.02),
        upper_fracs=(0.98, 0.98, 0.98),
        severity_presets=_build_cloud_severity_presets((0.8, 1.0, 1.0, 1.25, 2.0)),
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
    """Apply cloud, Poisson-Gaussian, or motion-blur corruptions to image batches.

    Args:
        corruption_type: One of ``"cloud"``, ``"poisson_gaussian"``, or ``"motion_blur"``.
        severity: Corruption severity in ``[1, 5]``.
        seed: Base seed used for deterministic per-image corruption.
        band_specs: Per-channel statistics and sensor metadata.
        dataset_name: Dataset key used by cloud calibration.
        cloud_pattern_mode: Cloud RNG mode. ``"fixed"`` reuses
            the same per-image cloud realization across severities. ``"independent"``
            samples a different realization for each severity.
    """

    def __init__(
        self,
        corruption_type: str,
        severity: int,
        seed: int,
        band_specs: list[BandSpec],
        dataset_name: str | None = None,
        cloud_pattern_mode: str = "fixed",
    ) -> None:
        if corruption_type not in {"cloud", "poisson_gaussian", "motion_blur"}:
            raise ValueError(
                f"Unknown corruption_type={corruption_type!r}; "
                f"expected cloud, poisson_gaussian, or motion_blur"
            )
        if severity < 1 or severity > 5:
            raise ValueError(f"severity must be in [1, 5], got {severity}")
        if not band_specs:
            raise ValueError("band_specs must be non-empty")
        if corruption_type == "cloud" and not dataset_name:
            raise ValueError("dataset_name is required for cloud corruption.")
        if cloud_pattern_mode not in {"fixed", "independent"}:
            raise ValueError(
                "cloud_pattern_mode must be one of "
                "{'fixed', 'independent'}"
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
        out, _ = self._apply_batch(images, return_cloud_masks=False)
        return out

    def apply_cloud_with_mask(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply cloud corruption and return the mean per-pixel cloud alpha mask."""
        if self.corruption_type != "cloud":
            raise ValueError("apply_cloud_with_mask is only valid for cloud corruption.")
        out, cloud_masks = self._apply_batch(images, return_cloud_masks=True)
        assert cloud_masks is not None
        return out, cloud_masks

    def _apply_batch(
        self,
        images: torch.Tensor,
        *,
        return_cloud_masks: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply the configured corruption to a batch."""
        if images.ndim != 4:
            raise ValueError(f"images must be 4D (B, C, H, W), got shape {tuple(images.shape)}")
        if images.shape[1] != len(self.band_specs):
            raise ValueError(
                f"Expected {len(self.band_specs)} channels from band_specs, got {images.shape[1]}"
            )

        in_dtype = images.dtype
        out = images.detach().clone().to(dtype=torch.float32)
        bsz = int(out.shape[0])
        cloud_masks = None
        if return_cloud_masks:
            _, _, height, width = out.shape
            cloud_masks = torch.zeros((bsz, height, width), device=out.device, dtype=torch.float32)

        if self.corruption_type == "motion_blur":
            out = self._apply_motion_blur(out)
        else:
            for i in range(bsz):
                global_idx = self._n_images_seen + i
                if self.corruption_type == "cloud":
                    out[i], cloud_mask = self._apply_cloud(
                        out[i],
                        global_idx,
                        return_cloud_mask=return_cloud_masks,
                    )
                    if cloud_masks is not None and cloud_mask is not None:
                        cloud_masks[i] = cloud_mask
                else:
                    out[i] = self._apply_poisson_gaussian(out[i], global_idx)

        self._n_images_seen += bsz
        return out.to(dtype=in_dtype), cloud_masks

    def _apply_cloud(
        self,
        image: torch.Tensor,
        global_idx: int,
        *,
        return_cloud_mask: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
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
        if self.cloud_pattern_mode == "independent":
            synth_seed += 1_000_000 * self.severity
        with torch.random.fork_rng(devices=_seed_devices(device), enabled=True):
            torch.manual_seed(synth_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(synth_seed)
            synth_result = self._scg.add_cloud(
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
                return_cloud=return_cloud_mask,
            )

        cloud_mask = None
        if return_cloud_mask:
            synth_output, cloud = synth_result
            cloud_mask = cloud.squeeze(0).mean(dim=0).to(dtype=torch.float32).clamp(0.0, 1.0)
        else:
            synth_output = synth_result
        synth_output = synth_output.squeeze(0).clamp(0.0, 1.0)
        out[optical_indices] = synth_output * denom + lower

        min_vals, max_vals = _tensor_range(self.band_specs, device=device, dtype=dtype)
        out[optical_indices] = out[optical_indices].clamp(
            min=min_vals[optical_indices], max=max_vals[optical_indices]
        )
        return out, cloud_mask

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
        min_vals, max_vals = _tensor_range(self.band_specs, device=device, dtype=dtype)
        span_vals = (max_vals - min_vals).clamp(min=1e-6)
        image_norm = ((image - min_vals) / span_vals).clamp(0.0, 1.0)

        photon_counts_list: list[float] = []
        read_std_frac_list: list[float] = []
        dataset_override = None
        if self.dataset_name is not None:
            dataset_override = NOISE_DATASET_OVERRIDES.get(self.dataset_name)
        override_preset = None
        if dataset_override is not None:
            override_preset = dataset_override.severity_presets[self.severity]

        for band in self.band_specs:
            if override_preset is not None:
                preset = override_preset
            else:
                sensor = band.sensor if band.sensor in NOISE_SENSOR_CALIBRATIONS else "aerial"
                preset = NOISE_SENSOR_CALIBRATIONS[sensor].severity_presets[self.severity]
            photon_counts_list.append(preset.photon_count)
            read_std_frac_list.append(preset.read_std_frac)

        photon_counts = torch.tensor(photon_counts_list, device=device, dtype=dtype).view(c, 1, 1)
        read_std_frac = torch.tensor(read_std_frac_list, device=device, dtype=dtype).view(c, 1, 1)

        gen = torch.Generator(device=device)
        gen.manual_seed(self.seed + global_idx)

        shot_counts = (image_norm * photon_counts).clamp(min=0.0)
        shot_norm = torch.poisson(shot_counts, generator=gen) / photon_counts
        readout_norm = read_std_frac * torch.randn(image.shape, device=device, dtype=dtype, generator=gen)
        out = (shot_norm + readout_norm) * span_vals + min_vals
        out = torch.maximum(out, min_vals)
        out = torch.minimum(out, max_vals)
        return out

    def _apply_motion_blur(self, images: torch.Tensor) -> torch.Tensor:
        """Apply horizontal motion blur to a batch.

        Args:
            images: Float batch with shape ``(B, C, H, W)``.

        Returns:
            Blurred batch with the same shape as input.
        """
        b, c, h, w = images.shape
        k = MOTION_BLUR_KERNEL_SIZES[self.severity]
        device = images.device

        # 1×k depthwise kernel — one uniform row, normalised
        kernel = torch.zeros(c, 1, 1, k, device=device, dtype=torch.float32)
        kernel[:, 0, 0, :] = 1.0 / k

        # Reflect-pad left/right only; kernel height is 1 so no vertical padding needed
        padded = F.pad(images, (k // 2, k // 2, 0, 0), mode="reflect")
        out = F.conv2d(padded, kernel, groups=c)

        min_vals, max_vals = _tensor_range(self.band_specs, device=device, dtype=images.dtype)
        return out.clamp(min_vals, max_vals)
