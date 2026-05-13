import numpy as np
import pytest

from torchgeo_bench.uq.error_pr import compute_error_pr


def test_compute_error_pr_basic():
    is_error = np.array([0, 0, 1, 1, 1], dtype=np.int64)
    uncertainty = np.array([0.1, 0.2, 0.8, 0.9, 0.7], dtype=np.float64)

    result = compute_error_pr(is_error=is_error, uncertainty=uncertainty)
    assert 0.0 <= float(result["ap"]) <= 1.0
    assert 0.0 <= float(result["auroc"]) <= 1.0
    assert float(result["ap"]) > 0.8
    assert len(result["precision"]) == len(result["recall"])


def test_compute_error_pr_requires_both_classes():
    is_error = np.array([0, 0, 0], dtype=np.int64)
    uncertainty = np.array([0.1, 0.2, 0.3], dtype=np.float64)

    with pytest.raises(ValueError):
        compute_error_pr(is_error=is_error, uncertainty=uncertainty)
