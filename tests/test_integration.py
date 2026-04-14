"""Integration tests for the full torchgeo-bench benchmark pipeline.

These tests run the actual CLI workflow end-to-end: load a real dataset,
extract features with a real model, and evaluate with KNN / logistic
regression.  They are slow (10–60 s each) and require data on disk, so
they are skipped by default.

Run with::

    pytest -m slow tests/test_integration.py
    pytest -m slow                           # all slow tests
    pytest -m slow -k forestnet              # just forestnet tests
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

GEOBENCH_ROOT = Path(os.getenv("GEOBENCH_ROOT", "data/classification_v1.0"))
GEOBENCH_V2_ROOT = Path(os.getenv("GEOBENCH_V2_ROOT", "data/geobenchv2"))

# Re-usable skip condition
_skip_no_v1 = pytest.mark.skipif(
    not GEOBENCH_ROOT.exists(), reason=f"GeoBench V1 data not found at {GEOBENCH_ROOT}"
)


def _run_bench(*overrides: str, timeout: int = 300) -> subprocess.CompletedProcess:
    """Run ``python -m torchgeo_bench`` with the given Hydra overrides."""
    cmd = [sys.executable, "-m", "torchgeo_bench", *overrides]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(Path(__file__).resolve().parent.parent),
    )


# ---------------------------------------------------------------------------
# Full pipeline: m-forestnet + resnet18
# ---------------------------------------------------------------------------


@pytest.mark.slow
@_skip_no_v1
class TestForestnetResNet18Pipeline:
    """End-to-end benchmark on m-forestnet with timm resnet18."""

    @pytest.fixture(autouse=True)
    def _output_csv(self, tmp_path):
        self.output = str(tmp_path / "integration_results.csv")

    def _run(self, *extra: str) -> subprocess.CompletedProcess:
        return _run_bench(
            "model=timm/resnet18",
            "dataset.names=[m-forestnet]",
            "dataset.partition=0.01x_train",
            f"output={self.output}",
            "eval.bootstrap=10",
            "device=cuda:0",
            *extra,
        )

    def test_knn_and_linear(self):
        """Full run: KNN + linear probe produces valid CSV output."""
        result = self._run()
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"

        df = pd.read_csv(self.output)
        assert len(df) >= 2, f"Expected ≥2 rows (knn + linear), got {len(df)}"
        assert set(df["method"]).issuperset({"knn5", "linear"})
        assert (df["metric_value"] > 0).all(), "All metrics should be > 0"
        assert (df["metric_value"] <= 1).all(), "All metrics should be ≤ 1"
        assert (df["dataset"] == "m-forestnet").all()
        assert (df["name"] == "resnet18").all()

    def test_knn_only(self):
        """Skip linear probe and verify only KNN results."""
        result = self._run("eval.skip_linear=true")
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"

        df = pd.read_csv(self.output)
        assert len(df) == 1
        assert df.iloc[0]["method"] == "knn5"
        assert 0 < df.iloc[0]["metric_value"] <= 1

    def test_resume_skips_completed(self):
        """Resume mode skips already-computed results."""
        # First run
        r1 = self._run("eval.skip_linear=true")
        assert r1.returncode == 0

        # Second run with resume — should skip and still succeed
        r2 = self._run("eval.skip_linear=true", "resume=true")
        assert r2.returncode == 0

        # CSV should still have exactly 1 row (not duplicated)
        df = pd.read_csv(self.output)
        assert len(df) == 1


# ---------------------------------------------------------------------------
# Full pipeline: m-eurosat + resnet18
# ---------------------------------------------------------------------------


@pytest.mark.slow
@_skip_no_v1
class TestEurosatResNet18Pipeline:
    """End-to-end benchmark on m-eurosat with timm resnet18."""

    @pytest.fixture(autouse=True)
    def _output_csv(self, tmp_path):
        self.output = str(tmp_path / "eurosat_results.csv")

    def test_full_run(self):
        """Full pipeline on m-eurosat produces reasonable accuracy."""
        result = _run_bench(
            "model=timm/resnet18",
            "dataset.names=[m-eurosat]",
            "dataset.partition=0.01x_train",
            f"output={self.output}",
            "eval.bootstrap=10",
            "device=cuda:0",
        )
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"

        df = pd.read_csv(self.output)
        linear = df[df["method"] == "linear"]
        assert len(linear) == 1
        # Even 1% partition should beat random (10 classes → 0.10)
        assert linear.iloc[0]["metric_value"] > 0.15, (
            f"Linear accuracy too low: {linear.iloc[0]['metric_value']}"
        )


# ---------------------------------------------------------------------------
# torchgeo model with normalization override
# ---------------------------------------------------------------------------


@pytest.mark.slow
@_skip_no_v1
class TestTorchGeoModelNormalization:
    """Verify torchgeo models use their own normalization (none) override."""

    @pytest.fixture(autouse=True)
    def _output_csv(self, tmp_path):
        self.output = str(tmp_path / "tgeo_norm_results.csv")

    def test_torchgeo_resnet_uses_none_normalization(self):
        """torchgeo ResNet18 MoCo should record normalization=none."""
        result = _run_bench(
            "model=torchgeo/resnet18_s2rgb_moco",
            "dataset.names=[m-eurosat]",
            "dataset.partition=0.01x_train",
            f"output={self.output}",
            "eval.bootstrap=10",
            "eval.skip_linear=true",
            "device=cuda:0",
        )
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"

        df = pd.read_csv(self.output)
        assert len(df) == 1
        assert df.iloc[0]["normalization"] == "none", (
            f"Expected normalization=none, got {df.iloc[0]['normalization']}"
        )


# ---------------------------------------------------------------------------
# Multi-dataset run
# ---------------------------------------------------------------------------


@pytest.mark.slow
@_skip_no_v1
class TestMultiDatasetRun:
    """Verify running on multiple datasets in a single invocation."""

    @pytest.fixture(autouse=True)
    def _output_csv(self, tmp_path):
        self.output = str(tmp_path / "multi_ds_results.csv")

    def test_two_datasets(self):
        """Run on m-eurosat + m-forestnet together."""
        result = _run_bench(
            "model=rcf",
            "dataset.names=[m-eurosat,m-forestnet]",
            "dataset.partition=0.01x_train",
            f"output={self.output}",
            "eval.bootstrap=10",
            "eval.skip_linear=true",
            "device=cuda:0",
        )
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"

        df = pd.read_csv(self.output)
        datasets = set(df["dataset"])
        assert datasets == {"m-eurosat", "m-forestnet"}, f"Got datasets: {datasets}"
        assert len(df) == 2  # 1 knn row per dataset
