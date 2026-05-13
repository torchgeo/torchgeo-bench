import numpy as np
import pytest

from torchgeo_bench.uq.reliability import (
    build_reliability_frame,
    reliability_bins_equal_mass,
    reliability_bins_equal_width,
)


def test_reliability_bins_equal_width_expected_values():
    confidence = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
    correct = np.array([0, 1, 1, 1], dtype=np.float64)

    rel_df = reliability_bins_equal_width(confidence, correct, bins=2)
    assert rel_df["n_bin"].tolist() == [2, 2]
    assert np.isclose(rel_df["mean_conf"].iloc[0], 0.15)
    assert np.isclose(rel_df["accuracy"].iloc[0], 0.5)
    assert np.isclose(rel_df["mean_conf"].iloc[1], 0.85)
    assert np.isclose(rel_df["accuracy"].iloc[1], 1.0)


def test_reliability_bins_equal_mass_row_counts():
    confidence = np.array([0.9, 0.2, 0.1, 0.8, 0.4], dtype=np.float64)
    correct = np.array([1, 0, 1, 1, 0], dtype=np.float64)

    rel_df = reliability_bins_equal_mass(confidence, correct, bins=3)
    assert len(rel_df) == 3
    assert int(rel_df["n_bin"].sum()) == len(confidence)


def test_build_reliability_frame_invalid_mode():
    confidence = np.array([0.1, 0.9], dtype=np.float64)
    correct = np.array([0, 1], dtype=np.float64)

    with pytest.raises(ValueError):
        build_reliability_frame(confidence=confidence, correct=correct, bins=2, binning="invalid")
