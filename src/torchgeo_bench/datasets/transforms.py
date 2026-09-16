"""Validation and resizing at the raw, canonical sample boundary."""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .input import ResolvedInput
from .spec import DatasetSpec

INTERPOLATION_MODES = ("area", "bicubic", "bilinear", "nearest")


def integer_target(value: object, *, name: str) -> torch.Tensor:
    """Convert integral targets without silently truncating floats."""
    target = value if isinstance(value, torch.Tensor) else torch.from_numpy(np.asarray(value))
    if target.is_complex() or (
        target.is_floating_point()
        and not bool((torch.isfinite(target) & (target == target.trunc())).all())
    ):
        raise ValueError(f"{name}: target must contain finite integer values")
    converted = target.long()
    if (target.is_floating_point() and bool(((target < -(2**63)) | (target >= 2**63)).any())) or (
        target.dtype == torch.uint64 and bool((converted < 0).any())
    ):
        raise ValueError(f"{name}: target is outside the supported integer range")
    return converted


def canonical_target(sample: dict, spec: DatasetSpec) -> None:
    """Validate the declared target, retaining all existing class and ignore values."""
    key = spec.target_key
    if key not in sample:
        raise ValueError(f"{spec.name}: missing required {key!r}")
    if spec.multilabel:
        label = torch.as_tensor(sample[key])
        if label.shape != (spec.num_classes,) or label.is_complex():
            raise ValueError(f"{spec.name}: expected label vector of length {spec.num_classes}")
        label = label.float()
        if not bool(torch.isfinite(label).all()):
            raise ValueError(f"{spec.name}: label vector must be finite")
        sample[key] = label
    else:
        target = integer_target(sample[key], name=f"{spec.name} {key}")
        if key == "mask" and target.ndim == 3 and target.shape[0] == 1:
            target = target.squeeze(0)
        expected_rank = 2 if key == "mask" else 0
        if target.ndim != expected_rank:
            shape = "H,W mask (or singleton 1,H,W)" if key == "mask" else "scalar label"
            raise ValueError(f"{spec.name}: expected {shape}, got {tuple(target.shape)}")
        if target.numel() == 0:
            raise ValueError(f"{spec.name}: expected nonempty {key}")
        sample[key] = target


def canonical_image(
    value: object, inputs: ResolvedInput, *, name: str, channels: int | None = None
) -> torch.Tensor:
    """Check exactly the resolved layout and channels before converting raw dtype."""
    image = torch.as_tensor(value)
    count = len(inputs.bands) if channels is None else channels
    rank = 4 if inputs.layout == "TCHW" else 3
    if image.ndim != rank or image.shape[-3] != count:
        raise ValueError(
            f"{name}: expected {inputs.layout} image with {count} channels, "
            f"got {tuple(image.shape)}"
        )
    if rank == 4 and image.shape[0] != inputs.num_time_steps:
        raise ValueError(f"{name}: expected {inputs.num_time_steps} time steps")
    if min(image.shape[-2:]) < 1 or image.is_complex():
        raise ValueError(f"{name}: expected nonempty real imagery")
    return image.float()


@dataclass(frozen=True)
class Resize:
    """Resize canonical imagery and categorical H,W masks without altering metadata."""

    image_size: int
    interpolation: str = "bilinear"

    def __post_init__(self) -> None:
        if type(self.image_size) is not int:
            raise TypeError("image_size must be an integer")
        if self.image_size < 1:
            raise ValueError("image_size must be positive")
        if self.interpolation not in INTERPOLATION_MODES:
            raise ValueError(f"interpolation must be one of {INTERPOLATION_MODES}")

    def image(self, image: torch.Tensor) -> torch.Tensor:
        """Resize raw CHW/TCHW imagery, preserving temporal slots."""
        size = (self.image_size, self.image_size)
        if image.shape[-2:] == size:
            return image
        temporal = image.ndim == 4
        result = F.interpolate(
            image if temporal else image.unsqueeze(0),
            size=size,
            mode=self.interpolation,
            align_corners=False if self.interpolation in ("bicubic", "bilinear") else None,
        )
        return result if temporal else result.squeeze(0)

    def __call__(self, sample: dict) -> dict:
        """Resize canonical image/mask fields only."""
        sample["image"] = self.image(sample["image"])
        if "mask" in sample:
            mask = sample["mask"]
            if mask.shape == (self.image_size, self.image_size):
                return sample
            # Indexing implements nearest interpolation without floating-point
            # round-trips that can corrupt large integer class/ignore IDs.
            height, width = mask.shape
            rows = torch.arange(self.image_size, device=mask.device) * height // self.image_size
            cols = torch.arange(self.image_size, device=mask.device) * width // self.image_size
            sample["mask"] = mask[rows[:, None], cols]
        return sample


@dataclass(frozen=True)
class CanonicalTransform:
    """Validate source-converted samples before common resizing."""

    spec: DatasetSpec
    inputs: ResolvedInput
    resize: Resize | None = None

    def __call__(self, sample: dict) -> dict:
        """Check raw images and targets, then apply optional common resizing."""
        sample = dict(sample)
        if "image" not in sample:
            raise ValueError(f"{self.spec.name}: missing required 'image'")
        sample["image"] = canonical_image(sample["image"], self.inputs, name=self.spec.name)
        canonical_target(sample, self.spec)
        return self.resize(sample) if self.resize is not None else sample
