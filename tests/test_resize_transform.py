"""Resize images without mixing temporal dimensions or interpolating class IDs."""

import pytest
import torch

from torchgeo_bench.datasets import ResolvedInput, get_dataset_spec
from torchgeo_bench.datasets.transforms import CanonicalTransform, Resize


def test_invalid_interpolation_is_rejected() -> None:
    with pytest.raises(ValueError, match="interpolation must be one of"):
        Resize(224, "bilnear")


def test_area_resize_averages_pixels() -> None:
    image = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
    resized = Resize(2, "area")({"image": image})
    torch.testing.assert_close(resized["image"], torch.tensor([[[2.5, 4.5], [10.5, 12.5]]]))


def test_temporal_images_and_singleton_masks_keep_their_axes_and_values() -> None:
    image = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).reshape(2, 3, 8, 8)
    mask = torch.arange(64).reshape(1, 8, 8) % 3
    spec = get_dataset_spec("pastis")
    inputs = ResolvedInput(spec.resolve_band_specs("rgb"), "rgb", 2)
    resized = CanonicalTransform(spec, inputs, Resize(4))({"image": image, "mask": mask})

    expected_image = image.reshape(2, 3, 4, 2, 4, 2).mean(dim=(3, 5))
    torch.testing.assert_close(resized["image"], expected_image)
    torch.testing.assert_close(resized["mask"], mask[0, ::2, ::2])
    assert resized["mask"].dtype == torch.long
