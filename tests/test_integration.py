"""Integration tests for the full torchgeo-bench benchmark pipeline.

These tests run the actual CLI workflow end-to-end: load a real dataset,
extract features with a real model, and evaluate with KNN / logistic
regression.  They are slow (10–60 s each) and require data on disk, so
they are skipped by default.

Run with::

    pytest -m slow tests/test_integration.py
    pytest -m slow                           # all slow tests
    pytest -m slow -k forestnet              # just forestnet tests

Expected values below were measured with seed=0 and are perfectly
reproducible across runs.  Tolerance is set to ±0.02 to absorb minor
library-version differences without masking real regressions.
"""

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

GEOBENCH_ROOT = Path(os.getenv("GEOBENCH_ROOT", "data/classification_v1.0"))

# Re-usable skip condition
_skip_no_v1 = pytest.mark.skipif(
    not GEOBENCH_ROOT.exists(), reason=f"GeoBench V1 data not found at {GEOBENCH_ROOT}"
)

# Tolerance for metric comparisons (absorbs minor library version diffs)
_TOL = 0.02


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
    """End-to-end benchmark on m-forestnet with timm resnet18 (1% partition)."""

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

    # Expected (seed=0, 1% partition): knn5=0.3112, linear=0.4622

    def test_knn_and_linear(self):
        """Full run: KNN + linear probe with tight performance check."""
        result = self._run()
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"

        df = pd.read_csv(self.output)
        assert set(df["method"]).issuperset({"knn5", "linear"})
        assert (df["dataset"] == "m-forestnet").all()
        assert (df["name"] == "resnet18").all()

        knn = df[df["method"] == "knn5"].iloc[0]["metric_value"]
        linear = df[df["method"] == "linear"].iloc[0]["metric_value"]
        assert knn == pytest.approx(0.311, abs=_TOL), f"KNN={knn}"
        assert linear == pytest.approx(0.462, abs=_TOL), f"Linear={linear}"

    def test_knn_only(self):
        """Skip linear probe and verify only KNN results."""
        result = self._run("eval.skip_linear=true")
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"

        df = pd.read_csv(self.output)
        assert len(df) == 1
        assert df.iloc[0]["method"] == "knn5"
        assert df.iloc[0]["metric_value"] == pytest.approx(0.311, abs=_TOL)

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
    """End-to-end benchmark on m-eurosat with timm resnet18 (1% partition)."""

    @pytest.fixture(autouse=True)
    def _output_csv(self, tmp_path):
        self.output = str(tmp_path / "eurosat_results.csv")

    # Expected (seed=0, 1% partition): knn5=0.279, linear=0.910

    def test_full_run(self):
        """Full pipeline on m-eurosat with tight performance check."""
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
        knn = df[df["method"] == "knn5"].iloc[0]["metric_value"]
        linear = df[df["method"] == "linear"].iloc[0]["metric_value"]
        assert knn == pytest.approx(0.279, abs=_TOL), f"KNN={knn}"
        assert linear == pytest.approx(0.910, abs=_TOL), f"Linear={linear}"


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


# ---------------------------------------------------------------------------
# Model baselines (default partition, seed=0)
# ---------------------------------------------------------------------------
#
# Verified deterministic (0 diff across runs).  Expected values:
#
# model                  | dataset     | knn5   | linear
# rcf                    | m-eurosat   | 0.6110 | 0.7690
# rcf                    | m-forestnet | 0.2276 | 0.4693
# imagestats             | m-eurosat   | 0.5970 | 0.7180
# mobilenetv3_small_100  | m-eurosat   | 0.8150 | 0.9330
# mobilenetv3_small_100  | m-forestnet | 0.3565 | 0.4975
# resnet18               | m-eurosat   | 0.8580 | 0.9290
# resnet18               | m-forestnet | 0.3666 | 0.5277


@pytest.mark.slow
@_skip_no_v1
class TestRCFBaseline:
    """RCF (random convolutional features) — fastest model, no downloads."""

    @pytest.fixture(autouse=True)
    def _output_csv(self, tmp_path):
        self.output = str(tmp_path / "rcf_results.csv")

    def test_eurosat(self):
        """RCF on m-eurosat default partition."""
        result = _run_bench(
            "model=rcf",
            "dataset.names=[m-eurosat]",
            f"output={self.output}",
            "eval.bootstrap=10",
            "device=cuda:0",
        )
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
        df = pd.read_csv(self.output)
        knn = df[df["method"] == "knn5"].iloc[0]["metric_value"]
        linear = df[df["method"] == "linear"].iloc[0]["metric_value"]
        assert knn == pytest.approx(0.611, abs=_TOL), f"KNN={knn}"
        assert linear == pytest.approx(0.769, abs=_TOL), f"Linear={linear}"

    def test_forestnet(self):
        """RCF on m-forestnet default partition."""
        result = _run_bench(
            "model=rcf",
            "dataset.names=[m-forestnet]",
            f"output={self.output}",
            "eval.bootstrap=10",
            "device=cuda:0",
        )
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
        df = pd.read_csv(self.output)
        knn = df[df["method"] == "knn5"].iloc[0]["metric_value"]
        linear = df[df["method"] == "linear"].iloc[0]["metric_value"]
        assert knn == pytest.approx(0.228, abs=_TOL), f"KNN={knn}"
        assert linear == pytest.approx(0.469, abs=_TOL), f"Linear={linear}"


@pytest.mark.slow
@_skip_no_v1
class TestImageStatsBaseline:
    """ImageStats — trivial per-channel statistics baseline."""

    @pytest.fixture(autouse=True)
    def _output_csv(self, tmp_path):
        self.output = str(tmp_path / "imagestats_results.csv")

    def test_eurosat(self):
        """ImageStats on m-eurosat default partition."""
        result = _run_bench(
            "model=imagestats",
            "dataset.names=[m-eurosat]",
            f"output={self.output}",
            "eval.bootstrap=10",
            "device=cuda:0",
        )
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
        df = pd.read_csv(self.output)
        knn = df[df["method"] == "knn5"].iloc[0]["metric_value"]
        linear = df[df["method"] == "linear"].iloc[0]["metric_value"]
        assert knn == pytest.approx(0.597, abs=_TOL), f"KNN={knn}"
        assert linear == pytest.approx(0.718, abs=_TOL), f"Linear={linear}"


@pytest.mark.slow
@_skip_no_v1
class TestMobileNetV3Baseline:
    """MobileNetV3-Small — small pretrained CNN."""

    @pytest.fixture(autouse=True)
    def _output_csv(self, tmp_path):
        self.output = str(tmp_path / "mobilenet_results.csv")

    def test_eurosat(self):
        """MobileNetV3-Small on m-eurosat default partition."""
        result = _run_bench(
            "model=timm/mobilenetv3_small_100",
            "dataset.names=[m-eurosat]",
            f"output={self.output}",
            "eval.bootstrap=10",
            "device=cuda:0",
        )
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
        df = pd.read_csv(self.output)
        knn = df[df["method"] == "knn5"].iloc[0]["metric_value"]
        linear = df[df["method"] == "linear"].iloc[0]["metric_value"]
        assert knn == pytest.approx(0.815, abs=_TOL), f"KNN={knn}"
        assert linear == pytest.approx(0.933, abs=_TOL), f"Linear={linear}"

    def test_forestnet(self):
        """MobileNetV3-Small on m-forestnet default partition."""
        result = _run_bench(
            "model=timm/mobilenetv3_small_100",
            "dataset.names=[m-forestnet]",
            f"output={self.output}",
            "eval.bootstrap=10",
            "device=cuda:0",
        )
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
        df = pd.read_csv(self.output)
        knn = df[df["method"] == "knn5"].iloc[0]["metric_value"]
        linear = df[df["method"] == "linear"].iloc[0]["metric_value"]
        assert knn == pytest.approx(0.357, abs=_TOL), f"KNN={knn}"
        assert linear == pytest.approx(0.498, abs=_TOL), f"Linear={linear}"


@pytest.mark.slow
@_skip_no_v1
class TestResNet18FullPartition:
    """ResNet18 on full default partition — strongest timm baseline."""

    @pytest.fixture(autouse=True)
    def _output_csv(self, tmp_path):
        self.output = str(tmp_path / "resnet18_full_results.csv")

    def test_eurosat(self):
        """ResNet18 on m-eurosat default partition."""
        result = _run_bench(
            "model=timm/resnet18",
            "dataset.names=[m-eurosat]",
            f"output={self.output}",
            "eval.bootstrap=10",
            "device=cuda:0",
        )
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
        df = pd.read_csv(self.output)
        knn = df[df["method"] == "knn5"].iloc[0]["metric_value"]
        linear = df[df["method"] == "linear"].iloc[0]["metric_value"]
        assert knn == pytest.approx(0.858, abs=_TOL), f"KNN={knn}"
        assert linear == pytest.approx(0.929, abs=_TOL), f"Linear={linear}"

    def test_forestnet(self):
        """ResNet18 on m-forestnet default partition."""
        result = _run_bench(
            "model=timm/resnet18",
            "dataset.names=[m-forestnet]",
            f"output={self.output}",
            "eval.bootstrap=10",
            "device=cuda:0",
        )
        assert result.returncode == 0, f"CLI failed:\n{result.stderr}"
        df = pd.read_csv(self.output)
        knn = df[df["method"] == "knn5"].iloc[0]["metric_value"]
        linear = df[df["method"] == "linear"].iloc[0]["metric_value"]
        assert knn == pytest.approx(0.367, abs=_TOL), f"KNN={knn}"
        assert linear == pytest.approx(0.528, abs=_TOL), f"Linear={linear}"
