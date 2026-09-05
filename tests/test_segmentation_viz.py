"""Segmentation image rendering for the supported display-band layouts."""

import numpy as np
import pytest
import torch

from torchgeo_bench.segmentation_viz import render_sample_grid


@pytest.mark.parametrize(
    ("channels", "indices", "expected_indices"),
    [(1, [0], [0, 0, 0]), (2, [0, 1], [0, 1, 1]), (3, [2, 1, 0], [2, 1, 0])],
)
def test_sample_grid_has_rgb_panels(channels, indices, expected_indices) -> None:
    ramp = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    bands = torch.stack([ramp, ramp.flip(0), ramp.flip(1)])[:channels]
    masks = torch.zeros((1, 3, 4), dtype=torch.long)
    grid = render_sample_grid(
        bands.unsqueeze(0), masks, masks, num_classes=4, rgb_indices=indices, n_samples=1
    )
    assert grid.shape == (24 + 3, 4 * 4, 3)
    assert grid.dtype == np.uint8
    image_panel = grid[24:, :4]
    for channel, source in enumerate(expected_indices):
        expected = (bands[source].numpy() / 11 * 255).astype(np.uint8)
        np.testing.assert_array_equal(image_panel[:, :, channel], expected)
