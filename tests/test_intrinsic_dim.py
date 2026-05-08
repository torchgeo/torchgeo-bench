"""Tests for intrinsic-dimension wrapper around torchid."""

import logging
from importlib.util import find_spec
from unittest import mock

import numpy as np
import pytest
import torch

from torchgeo_bench.intrinsic_dim import (
    SUPPORTED_ESTIMATORS,
    _resolve_device,
    _subsample,
    compute_intrinsic_dim,
)

torchid_available = find_spec("torchid") is not None
requires_torchid = pytest.mark.skipif(
    not torchid_available, reason="torchid not installed (requires Python >=3.13)"
)


# ---- pure-python helpers (no torchid required) ---------------------------


class TestResolveDevice:
    def test_none_uses_cuda_when_available(self) -> None:
        with mock.patch.object(torch.cuda, "is_available", return_value=True):
            assert _resolve_device(None).type == "cuda"

    def test_none_falls_back_to_cpu(self) -> None:
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            assert _resolve_device(None).type == "cpu"

    def test_explicit_cpu(self) -> None:
        assert _resolve_device("cpu").type == "cpu"

    def test_cuda_unavailable_falls_back_to_cpu(self, caplog: pytest.LogCaptureFixture) -> None:
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=False),
            caplog.at_level(logging.WARNING),
        ):
            dev = _resolve_device("cuda")
        assert dev.type == "cpu"
        assert any("CUDA requested" in r.message for r in caplog.records)

    def test_torch_device_passthrough(self) -> None:
        d = torch.device("cpu")
        assert _resolve_device(d) == d


class TestSubsample:
    def test_no_subsample_when_under_cap(self) -> None:
        X = np.arange(20).reshape(10, 2)
        out = _subsample(X, max_samples=100, seed=0)
        assert out is X  # unchanged ref

    def test_no_subsample_when_max_is_none(self) -> None:
        X = np.arange(20).reshape(10, 2)
        out = _subsample(X, max_samples=None, seed=0)
        assert out is X

    def test_subsamples_to_exact_size(self) -> None:
        X = np.arange(200).reshape(100, 2)
        out = _subsample(X, max_samples=10, seed=0)
        assert out.shape == (10, 2)

    def test_seed_determinism(self) -> None:
        X = np.arange(200).reshape(100, 2)
        a = _subsample(X, max_samples=10, seed=42)
        b = _subsample(X, max_samples=10, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self) -> None:
        X = np.arange(2000).reshape(1000, 2)
        a = _subsample(X, max_samples=10, seed=1)
        b = _subsample(X, max_samples=10, seed=2)
        assert not np.array_equal(a, b)


# ---- compute_intrinsic_dim: argument validation (no torchid needed) ------


class TestComputeBasic:
    def test_rejects_non_2d(self) -> None:
        with pytest.raises(ValueError, match="2D"):
            compute_intrinsic_dim(np.zeros((10,)), estimators=["TwoNN"])

    def test_empty_estimator_list_returns_empty(self) -> None:
        out = compute_intrinsic_dim(np.zeros((10, 3)), estimators=[])
        assert out == {}

    def test_supported_estimators_constant(self) -> None:
        for name in ("TwoNN", "MLE", "lPCA"):
            assert name in SUPPORTED_ESTIMATORS


# ---- error paths (mocked torchid) ----------------------------------------


class TestErrorHandling:
    def test_unknown_estimator_returns_nan(self, caplog: pytest.LogCaptureFixture) -> None:
        X = np.random.RandomState(0).randn(100, 5).astype(np.float32)
        with caplog.at_level(logging.WARNING):
            out = compute_intrinsic_dim(
                X, estimators=["NotARealEstimator"], device="cpu", max_samples=None
            )
        assert "NotARealEstimator" in out
        assert np.isnan(out["NotARealEstimator"])
        assert any("NotARealEstimator" in r.message for r in caplog.records)

    def test_failing_estimator_continues_others(self, caplog: pytest.LogCaptureFixture) -> None:
        """A failing estimator should log+nan, not abort the loop."""

        class _Boom:
            def fit(self, X: torch.Tensor) -> "_Boom":  # noqa: ARG002
                raise RuntimeError("boom")

        class _Good:
            def fit(self, X: torch.Tensor) -> "_Good":  # noqa: ARG002
                self.dimension_ = 3.5
                return self

        fake_module = mock.MagicMock()
        fake_module.Boom = _Boom
        fake_module.Good = _Good
        fake_pkg = mock.MagicMock()
        fake_pkg.estimators = fake_module

        X = np.random.RandomState(0).randn(50, 4).astype(np.float32)
        with (
            mock.patch.dict(
                "sys.modules", {"torchid": fake_pkg, "torchid.estimators": fake_module}
            ),
            caplog.at_level(logging.WARNING),
        ):
            out = compute_intrinsic_dim(
                X, estimators=["Boom", "Good"], device="cpu", max_samples=None
            )
        assert np.isnan(out["Boom"])
        assert out["Good"] == pytest.approx(3.5)
        assert any("Boom" in r.message for r in caplog.records)

    def test_missing_torchid_raises_importerror(self) -> None:
        from torchgeo_bench import intrinsic_dim as mod

        X = np.random.RandomState(0).randn(50, 4).astype(np.float32)
        # Force ImportError inside _load_estimator regardless of install state.
        with mock.patch.object(mod, "_load_estimator", side_effect=ImportError("no torchid")):
            out = compute_intrinsic_dim(X, estimators=["TwoNN"], device="cpu", max_samples=None)
        # ImportError is caught per-estimator → nan, not raised.
        assert np.isnan(out["TwoNN"])


# ---- real torchid integration (requires py>=3.13) ------------------------


@requires_torchid
class TestRealTorchid:
    @pytest.fixture(autouse=True)
    def _seed(self) -> None:
        torch.manual_seed(0)
        np.random.seed(0)

    @staticmethod
    def _swiss_roll(n: int) -> np.ndarray:
        """2D manifold embedded in 3D — true intrinsic dim = 2."""
        rng = np.random.default_rng(0)
        t = rng.uniform(1.5, 4.5, size=n) * np.pi
        h = rng.uniform(0, 5, size=n)
        X = np.stack([t * np.cos(t), h, t * np.sin(t)], axis=1)
        return X.astype(np.float32)

    @staticmethod
    def _uniform_cube(n: int, d: int) -> np.ndarray:
        rng = np.random.default_rng(0)
        return rng.uniform(0, 1, size=(n, d)).astype(np.float32)

    def test_swiss_roll_two_nn_close_to_2(self) -> None:
        X = self._swiss_roll(2000)
        out = compute_intrinsic_dim(X, estimators=["TwoNN"], device="cpu", max_samples=None)
        assert abs(out["TwoNN"] - 2.0) < 0.5

    def test_swiss_roll_mle_close_to_2(self) -> None:
        X = self._swiss_roll(2000)
        out = compute_intrinsic_dim(X, estimators=["MLE"], device="cpu", max_samples=None)
        assert abs(out["MLE"] - 2.0) < 0.5

    def test_uniform_cube_lpca_matches_ambient(self) -> None:
        X = self._uniform_cube(1000, d=5)
        out = compute_intrinsic_dim(X, estimators=["lPCA"], device="cpu", max_samples=None)
        # lPCA on full-rank cube yields ambient dim
        assert out["lPCA"] == pytest.approx(5.0, abs=0.1)

    def test_multiple_estimators_returned(self) -> None:
        X = self._uniform_cube(800, d=4)
        out = compute_intrinsic_dim(
            X, estimators=["TwoNN", "MLE", "lPCA"], device="cpu", max_samples=None
        )
        assert set(out) == {"TwoNN", "MLE", "lPCA"}
        for v in out.values():
            assert np.isfinite(v)

    def test_subsampling_determinism(self) -> None:
        X = self._uniform_cube(5000, d=3)
        a = compute_intrinsic_dim(X, estimators=["TwoNN"], device="cpu", max_samples=500, seed=7)
        b = compute_intrinsic_dim(X, estimators=["TwoNN"], device="cpu", max_samples=500, seed=7)
        assert a == b

    def test_cpu_explicit(self) -> None:
        X = self._uniform_cube(500, d=3)
        out = compute_intrinsic_dim(X, estimators=["TwoNN"], device="cpu", max_samples=None)
        assert np.isfinite(out["TwoNN"])

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_path(self) -> None:
        X = self._uniform_cube(500, d=3)
        out = compute_intrinsic_dim(X, estimators=["TwoNN"], device="cuda", max_samples=None)
        assert np.isfinite(out["TwoNN"])

    def test_auto_device_runs(self) -> None:
        X = self._uniform_cube(500, d=3)
        out = compute_intrinsic_dim(X, estimators=["TwoNN"], device=None, max_samples=None)
        assert np.isfinite(out["TwoNN"])
