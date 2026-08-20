"""Tests for the band-statistics contributor utility."""

import pytest

from scripts.compute_band_statistics import _format_stat


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0, "0"),
        (-0.49, "-0.49"),
        (0.99, "0.99"),
        (-12.75, "-12.75"),
        (33.25, "33.25"),
        (255.0, "255"),
    ],
)
def test_format_stat_preserves_fractional_extrema(value: float, expected: str) -> None:
    assert _format_stat(value) == expected
