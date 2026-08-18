"""Tests for intrinsic-dimension wrapper around torchid."""

import logging
from importlib.util import find_spec
from unittest import mock

import numpy as np
import pytest
import torch

from torchgeo_bench.intrinsic_dim import (
    FEATURE_SPECTRUM_METRICS,
    DegenerateManifoldError,
    DegenerateSpectrumError,
    _drop_zero_distance_rows,
    _load_estimator,
    _resolve_device,
    _subsample,
    compute_feature_spectrum,
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

    def test_cuda_unavailable_falls_back_to_cpu(self, caplog: pytest.LogCaptureFixture) -> None:
        with (
            mock.patch.object(torch.cuda, "is_available", return_value=False),
            caplog.at_level(logging.WARNING),
        ):
            dev = _resolve_device("cuda")
        assert dev.type == "cpu"
        assert any("CUDA requested" in r.message for r in caplog.records)


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


# ---- compute_intrinsic_dim: argument validation (no torchid needed) ------


class TestComputeBasic:
    def test_rejects_non_2d(self) -> None:
        with pytest.raises(ValueError, match="2D"):
            compute_intrinsic_dim(np.zeros((10,)), estimators=["TwoNN"])

    def test_empty_estimator_list_returns_empty(self) -> None:
        out = compute_intrinsic_dim(np.zeros((10, 3)), estimators=[])
        assert out == {}


class TestFeatureSpectrum:
    def test_isotropic_features(self) -> None:
        X = np.vstack([np.eye(4), -np.eye(4)])

        metrics = compute_feature_spectrum(X, max_samples=None)

        assert set(metrics) == set(FEATURE_SPECTRUM_METRICS)
        assert metrics["effective_rank"] == pytest.approx(4.0)
        assert metrics["participation_ratio"] == pytest.approx(4.0)
        assert metrics["pc1_variance_ratio"] == pytest.approx(0.25)
        assert metrics["pc10_variance_ratio"] == pytest.approx(1.0)
        assert metrics["spectral_anisotropy"] == pytest.approx(0.0)

    def test_rank_one_features(self) -> None:
        samples = np.arange(-3, 4, dtype=np.float64)
        direction = np.array([1.0, -2.0, 3.0, 0.5])
        X = np.outer(samples, direction)

        metrics = compute_feature_spectrum(X, max_samples=None)

        assert metrics["effective_rank"] == pytest.approx(1.0)
        assert metrics["participation_ratio"] == pytest.approx(1.0)
        assert metrics["pc1_variance_ratio"] == pytest.approx(1.0)
        assert metrics["pc10_variance_ratio"] == pytest.approx(1.0)
        assert metrics["spectral_anisotropy"] == pytest.approx(1.0)

    def test_single_feature_uses_zero_anisotropy_convention(self) -> None:
        X = np.arange(5, dtype=np.float64)[:, None]

        metrics = compute_feature_spectrum(X, max_samples=None)

        assert metrics["effective_rank"] == pytest.approx(1.0)
        assert metrics["spectral_anisotropy"] == pytest.approx(0.0)

    def test_known_low_rank_spectrum(self) -> None:
        X = np.zeros((6, 12), dtype=np.float64)
        X[0:2, 0] = [3.0, -3.0]
        X[2:4, 1] = [2.0, -2.0]
        X[4:6, 2] = [1.0, -1.0]
        proportions = np.array([9.0, 4.0, 1.0]) / 14.0

        metrics = compute_feature_spectrum(X + 100.0, max_samples=None)

        expected_effective_rank = np.exp(-np.sum(proportions * np.log(proportions)))
        expected_participation_ratio = 1.0 / np.sum(proportions**2)
        expected_anisotropy = (12 * proportions[0] - 1) / 11
        assert metrics["effective_rank"] == pytest.approx(expected_effective_rank)
        assert metrics["participation_ratio"] == pytest.approx(expected_participation_ratio)
        assert metrics["pc1_variance_ratio"] == pytest.approx(proportions[0])
        assert metrics["pc10_variance_ratio"] == pytest.approx(1.0)
        assert metrics["spectral_anisotropy"] == pytest.approx(expected_anisotropy)

    @pytest.mark.parametrize(
        "X,match",
        [
            (np.ones(5), "2D"),
            (np.empty((2, 0)), "at least one feature"),
            (np.ones((5, 3)), "zero total variance"),
            (np.ones((1, 3)), "at least two samples"),
            (np.array([[0.0, np.nan], [1.0, 2.0]]), "finite values"),
        ],
    )
    def test_degenerate_inputs_fail_clearly(self, X: np.ndarray, match: str) -> None:
        error = DegenerateSpectrumError if "zero total variance" in match else ValueError
        with pytest.raises(error, match=match):
            compute_feature_spectrum(X, max_samples=None)

    def test_subsampling_is_deterministic(self) -> None:
        X = np.random.default_rng(0).normal(size=(100, 8))
        first = compute_feature_spectrum(X, max_samples=20, seed=17)
        second = compute_feature_spectrum(X, max_samples=20, seed=17)

        assert first == second

    def test_subsampling_requires_two_samples(self) -> None:
        X = np.random.default_rng(0).normal(size=(10, 3))
        with pytest.raises(ValueError, match="at least 2"):
            compute_feature_spectrum(X, max_samples=1)


# ---- error paths (mocked torchid) ----------------------------------------


class TestErrorHandling:
    @requires_torchid
    def test_unknown_estimator_raises(self) -> None:
        """Estimator lookup failure raises instead of writing NaN."""
        X = np.random.RandomState(0).randn(100, 5).astype(np.float32)
        with pytest.raises(ValueError, match="Unknown torchid estimator"):
            compute_intrinsic_dim(
                X, estimators=["NotARealEstimator"], device="cpu", max_samples=None
            )

    @requires_torchid
    def test_failing_estimator_propagates(self) -> None:
        """A torchid-internal exception propagates instead of being
        written as NaN.

        Patches the torchid estimators registry rather than swapping the
        whole module so ``torchid.primitives`` (used by the
        zero-distance dedup) keeps working."""
        import torchid.estimators as real_estimators

        class _Boom:
            def fit(self, X: torch.Tensor) -> "_Boom":  # noqa: ARG002
                raise RuntimeError("boom")

        X = np.random.RandomState(0).randn(50, 4).astype(np.float32)
        with (
            mock.patch.object(real_estimators, "Boom", _Boom, create=True),
            pytest.raises(RuntimeError, match="boom"),
        ):
            compute_intrinsic_dim(X, estimators=["Boom"], device="cpu", max_samples=None)


# ---- _load_estimator ---------------------------------------------------------


class TestLoadEstimator:
    @requires_torchid
    def test_known_estimator_returns_class(self) -> None:
        cls = _load_estimator("TwoNN")
        assert callable(cls)

    def test_missing_torchid_raises_import_error(self) -> None:
        import builtins

        real_import = builtins.__import__

        def _mock(name, *a, **kw):
            if name == "torchid":
                raise ImportError("mocked")
            return real_import(name, *a, **kw)

        with (
            mock.patch.object(builtins, "__import__", side_effect=_mock),
            pytest.raises(ImportError, match="torchid is required"),
        ):
            _load_estimator("TwoNN")


# ---- _drop_zero_distance_rows ------------------------------------------------


class TestDropZeroDistanceRows:
    def test_no_duplicates_all_rows_kept(self) -> None:
        torch.manual_seed(0)
        X = torch.randn(20, 4)
        out = _drop_zero_distance_rows(X)
        assert out.shape[0] == 20

    def test_exact_duplicates_rows_dropped(self) -> None:
        torch.manual_seed(1)
        X = torch.randn(10, 4)
        X[3] = X[1].clone()  # inject duplicate
        out = _drop_zero_distance_rows(X)
        assert out.shape[0] < 10

    def test_output_has_no_zero_distance(self) -> None:
        torch.manual_seed(2)
        X = torch.randn(15, 4)
        X[5] = X[2].clone()
        out = _drop_zero_distance_rows(X)
        # After dropping, no two rows should share zero d1
        if out.shape[0] >= 2:
            from torchgeo_bench.intrinsic_dim import _two_nearest_distances

            d = _two_nearest_distances(out)
            assert (d[:, 0] > 0).all()

    def test_logging_on_drop(self, caplog: pytest.LogCaptureFixture) -> None:
        torch.manual_seed(3)
        X = torch.randn(10, 4)
        X[0] = X[1].clone()
        with caplog.at_level(logging.INFO):
            _drop_zero_distance_rows(X)
        assert any("dropped" in r.message for r in caplog.records)


# ---- DegenerateManifoldError --------------------------------------------------


class TestDegenerateManifoldError:
    @requires_torchid
    def test_raised_on_non_finite_dimension(self) -> None:
        """Mock a torchid estimator that returns NaN to trigger the error."""
        import torchid.estimators as real_estimators

        class _NaNEstimator:
            dimension_: float = float("nan")

            def fit(self, X: torch.Tensor) -> "_NaNEstimator":  # noqa: ARG002
                return self

        X = np.random.RandomState(0).randn(50, 4).astype(np.float32)
        with (
            mock.patch.object(real_estimators, "NaNEst", _NaNEstimator, create=True),
            pytest.raises(DegenerateManifoldError, match="non-finite"),
        ):
            compute_intrinsic_dim(X, estimators=["NaNEst"], device="cpu", max_samples=None)


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

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_path(self) -> None:
        X = self._uniform_cube(500, d=3)
        out = compute_intrinsic_dim(X, estimators=["TwoNN"], device="cuda", max_samples=None)
        assert np.isfinite(out["TwoNN"])
