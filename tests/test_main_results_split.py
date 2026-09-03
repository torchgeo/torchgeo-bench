"""Fast offline tests for the default (no explicit ``output=``) per-model routing.

Profile and intrinsic-dim rows are one-time model+hardware measurements, so
the default routing path (no explicit ``output=``) sends them to their own
per-model files under ``profile_results_dir`` / ``intrinsic_dim_results_dir``,
separate from the ``results_dir`` metrics file.
"""

from pathlib import Path
from unittest import mock

import pandas as pd

from torchgeo_bench.config import compose_config
from torchgeo_bench.main import main
from torchgeo_bench.results import model_results_path
from torchgeo_bench.settings import RunSettings, merge

from .test_main_fast import _synthetic_embeddings, _synthetic_loaders


def _compose_default_routing_cfg(tmp_path: Path, overrides: dict | None = None) -> RunSettings:
    """Compose a config with no explicit ``output=``, routed at ``tmp_path``."""
    base = {
        "dataset": {
            "names": ["m-eurosat"],
            "partition": "default",
            "batch_size": 4,
            "num_workers": 0,
        },
        "eval": {"bootstrap": 5, "c_range": [-2, -1, 2]},
        "device": "cpu",
        "results_dir": str(tmp_path / "models"),
        "profile_results_dir": str(tmp_path / "profiles"),
        "intrinsic_dim_results_dir": str(tmp_path / "intrinsic_dim"),
    }
    if overrides:
        base = merge(base, overrides)
    return compose_config(base, model="rcf")


def test_default_routing_splits_profile_and_intrinsic_dim_into_own_files(tmp_path: Path):
    cfg = _compose_default_routing_cfg(
        tmp_path,
        overrides={
            "eval": {
                "skip_linear": True,
                "intrinsic_dim": {
                    "enabled": True,
                    "estimators": ["twonn"],
                    "splits": ["train"],
                    "max_samples": 100,
                },
                "profile": {"enabled": True, "n_warmup": 1, "n_measure": 1},
            },
        },
    )
    profile_metrics = {
        "params_m": 0.01,
        "throughput_samples_per_sec": 100.0,
        "latency_ms_per_batch_p50": 5.0,
    }

    def _mock_compute(*args, **kwargs):
        return {str(kwargs["estimators"][0]): 5.0}

    with (
        mock.patch("torchgeo_bench.main.get_datasets", return_value=_synthetic_loaders()),
        mock.patch("torchgeo_bench.main.embed_split", side_effect=_synthetic_embeddings()),
        mock.patch(
            "torchgeo_bench.main.evaluate_knn",
            return_value=(0.5, 0.45, 0.55, {"ece": 0.05, "rms_ce": 0.07, "mce": 0.1}, 6),
        ),
        mock.patch("torchgeo_bench.main.compute_intrinsic_dim", side_effect=_mock_compute),
        mock.patch("torchgeo_bench.main.measure_profile", return_value=profile_metrics),
    ):
        main(cfg)

    metrics_path = model_results_path(tmp_path / "models", "rcf")
    profile_path = model_results_path(tmp_path / "profiles", "rcf")
    id_path = model_results_path(tmp_path / "intrinsic_dim", "rcf")

    assert metrics_path.exists()
    metrics_df = pd.read_csv(metrics_path)
    assert set(metrics_df["method"]) == {"knn5"}

    assert profile_path.exists()
    profile_df = pd.read_csv(profile_path)
    assert set(profile_df["method"]) == {"profile"}
    for name in profile_metrics:
        assert name in profile_df["metric_name"].values

    assert id_path.exists()
    id_df = pd.read_csv(id_path)
    assert set(id_df["method"]) == {"intrinsic_dim"}
    assert "id_twonn_train" in id_df["metric_name"].values


def test_default_routing_resume_reads_all_three_files(tmp_path: Path):
    """resume=true must merge completed_metrics across all 3 per-model files."""
    cfg = _compose_default_routing_cfg(
        tmp_path,
        overrides={
            "resume": True,
            "eval": {
                "skip_linear": True,
                "intrinsic_dim": {
                    "enabled": True,
                    "estimators": ["twonn"],
                    "splits": ["train"],
                    "max_samples": 100,
                },
                "profile": {"enabled": True, "n_warmup": 1, "n_measure": 1},
            },
        },
    )

    def _mock_compute(*args, **kwargs):
        return {str(kwargs["estimators"][0]): 5.0}

    profile_metrics = {
        "params_m": 0.01,
        "throughput_samples_per_sec": 100.0,
        "latency_ms_per_batch_p50": 5.0,
    }

    # First run: creates all three files.
    with (
        mock.patch("torchgeo_bench.main.get_datasets", return_value=_synthetic_loaders()),
        mock.patch("torchgeo_bench.main.embed_split", side_effect=_synthetic_embeddings()),
        mock.patch(
            "torchgeo_bench.main.evaluate_knn",
            return_value=(0.5, 0.45, 0.55, {"ece": 0.05, "rms_ce": 0.07, "mce": 0.1}, 6),
        ),
        mock.patch("torchgeo_bench.main.compute_intrinsic_dim", side_effect=_mock_compute),
        mock.patch("torchgeo_bench.main.measure_profile", return_value=profile_metrics),
    ):
        main(cfg)

    metrics_path = model_results_path(tmp_path / "models", "rcf")
    profile_path = model_results_path(tmp_path / "profiles", "rcf")
    id_path = model_results_path(tmp_path / "intrinsic_dim", "rcf")
    metrics_rows_after_first = len(pd.read_csv(metrics_path))
    profile_rows_after_first = len(pd.read_csv(profile_path))
    id_rows_after_first = len(pd.read_csv(id_path))

    # Second run with resume=true: nothing should be recomputed/duplicated.
    with (
        mock.patch("torchgeo_bench.main.get_datasets", return_value=_synthetic_loaders()),
        mock.patch("torchgeo_bench.main.evaluate_knn") as knn_mock,
        mock.patch("torchgeo_bench.main.compute_intrinsic_dim") as id_mock,
        mock.patch("torchgeo_bench.main.measure_profile") as profile_mock,
    ):
        main(cfg)

    knn_mock.assert_not_called()
    id_mock.assert_not_called()
    profile_mock.assert_not_called()

    assert len(pd.read_csv(metrics_path)) == metrics_rows_after_first
    assert len(pd.read_csv(profile_path)) == profile_rows_after_first
    assert len(pd.read_csv(id_path)) == id_rows_after_first
