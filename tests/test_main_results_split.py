"""Fast offline tests for the default (no explicit ``output=``) per-model routing.

Profile and intrinsic-dim rows are one-time model+hardware measurements, so
the default routing path (no explicit ``output=``) sends them to their own
per-model files under ``profile_results_dir`` / ``intrinsic_dim_results_dir``,
separate from the ``results_dir`` metrics file.
"""

from collections.abc import Sequence
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest
from omegaconf import DictConfig

from torchgeo_bench.config import compose_config
from torchgeo_bench.main import main
from torchgeo_bench.results import model_results_path

from .test_main_fast import _chainable_model_mock, _synthetic_embeddings, _synthetic_loaders


def _compose_default_routing_cfg(
    tmp_path: Path, overrides: Sequence[str] | None = None
) -> DictConfig:
    """Compose a config with no explicit ``output=``, routed at ``tmp_path``."""
    extra = list(overrides or [])
    return compose_config(
        [
            "model=rcf",
            "dataset.names=[m-eurosat]",
            "dataset.partition=default",
            "dataset.batch_size=4",
            "dataset.num_workers=0",
            "eval.bootstrap=5",
            "eval.c_range=[-2,-1,2]",
            "device=cpu",
            f"results_dir={tmp_path / 'models'}",
            f"profile_results_dir={tmp_path / 'profiles'}",
            f"intrinsic_dim_results_dir={tmp_path / 'intrinsic_dim'}",
            *extra,
        ]
    )


@pytest.mark.parametrize("explicit_output", [False, True])
def test_routing_splits_by_kind_unless_output_is_explicit(tmp_path: Path, explicit_output: bool):
    cfg = _compose_default_routing_cfg(
        tmp_path,
        overrides=[
            "eval.skip_linear=true",
            "eval.intrinsic_dim.enabled=true",
            "eval.intrinsic_dim.estimators=[twonn]",
            "eval.intrinsic_dim.splits=[train]",
            "eval.intrinsic_dim.max_samples=100",
            "eval.profile.enabled=true",
            "eval.profile.n_warmup=1",
            "eval.profile.n_measure=1",
        ],
    )
    if explicit_output:
        cfg.output = str(tmp_path / "all.csv")
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
    if explicit_output:
        assert not any(path.exists() for path in (metrics_path, profile_path, id_path))
        metrics_path = profile_path = id_path = Path(cfg.output)
    all_methods = {"knn5", "profile", "intrinsic_dim"}

    assert metrics_path.exists()
    metrics_df = pd.read_csv(metrics_path)
    assert set(metrics_df["method"]) == (all_methods if explicit_output else {"knn5"})

    assert profile_path.exists()
    profile_df = pd.read_csv(profile_path)
    assert set(profile_df["method"]) == (all_methods if explicit_output else {"profile"})
    for name in profile_metrics:
        assert name in profile_df["metric_name"].values

    assert id_path.exists()
    id_df = pd.read_csv(id_path)
    assert set(id_df["method"]) == (all_methods if explicit_output else {"intrinsic_dim"})
    assert "id_twonn_train" in id_df["metric_name"].values


@pytest.mark.parametrize("explicit_output", [False, True])
def test_completed_intrinsic_dim_survives_profile_failure(
    tmp_path: Path, explicit_output: bool
) -> None:
    cfg = _compose_default_routing_cfg(
        tmp_path,
        overrides=[
            "eval.skip_linear=true",
            "eval.intrinsic_dim.enabled=true",
            "eval.intrinsic_dim.estimators=[twonn]",
            "eval.intrinsic_dim.splits=[train]",
            "eval.profile.enabled=true",
        ],
    )
    metrics_path = model_results_path(tmp_path / "models", "rcf")
    id_path = model_results_path(tmp_path / "intrinsic_dim", "rcf")
    if explicit_output:
        cfg.output = str(tmp_path / "all.csv")
        metrics_path = id_path = Path(cfg.output)

    with (
        mock.patch("torchgeo_bench.main.get_datasets", return_value=_synthetic_loaders()),
        mock.patch("torchgeo_bench.main.instantiate", return_value=_chainable_model_mock()),
        mock.patch("torchgeo_bench.main.embed_split", side_effect=_synthetic_embeddings()),
        mock.patch(
            "torchgeo_bench.main.evaluate_knn",
            return_value=(0.5, 0.45, 0.55, {"ece": 0.05, "rms_ce": 0.07, "mce": 0.1}, 6),
        ),
        mock.patch("torchgeo_bench.main.compute_intrinsic_dim", return_value={"twonn": 5.0}),
        mock.patch(
            "torchgeo_bench.main.measure_profile", side_effect=RuntimeError("profile failed")
        ),
        pytest.raises(RuntimeError, match="profile failed"),
    ):
        main(cfg)

    metrics_df = pd.read_csv(metrics_path)
    assert int((metrics_df["method"] == "knn5").sum()) == 1
    id_df = pd.read_csv(id_path)
    row = id_df[id_df["metric_name"] == "id_twonn_train"].iloc[0]
    assert row["method"] == "intrinsic_dim"
    assert row["metric_value"] == 5.0
    assert not model_results_path(tmp_path / "profiles", "rcf").exists()


def test_default_routing_resume_reads_all_three_files(tmp_path: Path):
    """resume=true must merge completed_metrics across all 3 per-model files."""
    cfg = _compose_default_routing_cfg(
        tmp_path,
        overrides=[
            "resume=true",
            "eval.skip_linear=true",
            "eval.intrinsic_dim.enabled=true",
            "eval.intrinsic_dim.estimators=[twonn]",
            "eval.intrinsic_dim.splits=[train]",
            "eval.intrinsic_dim.max_samples=100",
            "eval.profile.enabled=true",
            "eval.profile.n_warmup=1",
            "eval.profile.n_measure=1",
        ],
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
