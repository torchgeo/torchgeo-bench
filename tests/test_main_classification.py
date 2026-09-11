"""Offline tests for the classification runner."""

from pathlib import Path
from unittest import mock

import pandas as pd
import pytest
from omegaconf import OmegaConf
from torchgeo.datasets import DatasetNotFoundError

from tests.support.runner import (
    _chainable_model_mock,
    _compose_cfg,
    _resume_row,
    _synthetic_embeddings,
    _synthetic_loaders,
)
from torchgeo_bench.main import LinearProbeDivergedError, main, resolve_model_config
from torchgeo_bench.resume import _resume_config_hash


def test_model_dataset_overrides_are_isolated_and_fall_back() -> None:
    model_cfg = OmegaConf.create(
        {
            "_target_": "example.Model",
            "name": "example",
            "image_size": 224,
            "res": 1.0,
            "pool": "cls",
            "dataset_overrides": {
                "m-eurosat": {"image_size": 64, "res": 3.5},
                "forestnet": {"image_size": 128, "pool": "mean"},
            },
        }
    )

    eurosat = resolve_model_config(model_cfg, "m-eurosat")
    fallback = resolve_model_config(model_cfg, "unlisted")
    forestnet = resolve_model_config(model_cfg, "forestnet")

    assert (eurosat.image_size, eurosat.res, eurosat.pool) == (64, 3.5, "cls")
    assert (fallback.image_size, fallback.res, fallback.pool) == (224, 1.0, "cls")
    assert (forestnet.image_size, forestnet.res, forestnet.pool) == (128, 1.0, "mean")
    assert "dataset_overrides" not in eurosat
    assert model_cfg.image_size == 224


def test_dataset_override_routes_recipe_and_changes_resume_key(tmp_path: Path) -> None:
    out = tmp_path / "out.csv"
    cfg = _compose_cfg(out, overrides=["eval.skip_linear=true"])
    OmegaConf.update(
        cfg,
        "model.dataset_overrides",
        {
            "m-eurosat": {
                "image_size": 64,
                "interpolation": "area",
                "res": 3.5,
                "pool": "cls",
            }
        },
        force_add=True,
    )
    model = _chainable_model_mock()

    with (
        mock.patch(
            "torchgeo_bench.main.get_datasets", return_value=_synthetic_loaders()
        ) as data_mock,
        mock.patch("torchgeo_bench.main.instantiate", return_value=model) as instantiate_mock,
        mock.patch("torchgeo_bench.main.embed_split", side_effect=_synthetic_embeddings()),
        mock.patch(
            "torchgeo_bench.main.evaluate_knn",
            return_value=(0.5, 0.45, 0.55, {"ece": 0.05, "rms_ce": 0.07, "mce": 0.1}, 6),
        ),
    ):
        main(cfg)

    assert data_mock.call_args.kwargs["image_size"] == 64
    assert data_mock.call_args.kwargs["interpolation"] == "area"
    instantiated_cfg = instantiate_mock.call_args.args[0]
    assert instantiated_cfg.image_size == 64
    assert instantiated_cfg.res == 3.5
    assert instantiated_cfg.pool == "cls"
    assert "interpolation" not in instantiated_cfg

    first_row = pd.read_csv(out).iloc[0]
    assert first_row["res"] == 3.5
    assert first_row["pool"] == "cls"

    changed_cfg = _compose_cfg(out, overrides=["resume=true", "eval.skip_linear=true"])
    OmegaConf.update(
        changed_cfg,
        "model.dataset_overrides",
        {
            "m-eurosat": {
                "image_size": 64,
                "interpolation": "area",
                "res": 3.5,
                "pool": "mean",
            }
        },
        force_add=True,
    )
    with (
        mock.patch("torchgeo_bench.main.get_datasets", return_value=_synthetic_loaders()),
        mock.patch("torchgeo_bench.main.instantiate", return_value=model),
        mock.patch("torchgeo_bench.main.embed_split", side_effect=_synthetic_embeddings()),
        mock.patch(
            "torchgeo_bench.main.evaluate_knn",
            return_value=(0.5, 0.45, 0.55, {"ece": 0.05, "rms_ce": 0.07, "mce": 0.1}, 6),
        ) as knn_mock,
    ):
        main(changed_cfg)

    knn_mock.assert_called_once()
    rows = pd.read_csv(out)
    assert set(rows["pool"]) == {"cls", "mean"}


def test_knn_row_preserves_metrics_and_metadata(tmp_path: Path) -> None:
    out = tmp_path / "out.csv"
    cfg = _compose_cfg(out, overrides=["eval.skip_linear=true"])

    with (
        mock.patch("torchgeo_bench.main.get_datasets", return_value=_synthetic_loaders()),
        mock.patch("torchgeo_bench.main.embed_split", side_effect=_synthetic_embeddings()),
        mock.patch(
            "torchgeo_bench.main.evaluate_knn",
            return_value=(0.5, 0.45, 0.55, {"ece": 0.05, "rms_ce": 0.07, "mce": 0.1}, 6),
        ),
    ):
        main(cfg)

    df = pd.read_csv(out)
    assert list(df["method"]) == ["knn5"]
    row = df.iloc[0]
    assert row["metric_name"] == "accuracy"
    assert row["dataset"] == "m-eurosat"
    assert (row["metric_value"], row["ci_lower"], row["ci_upper"]) == (0.5, 0.45, 0.55)
    assert row["ece"] == 0.05
    assert row["model"] == cfg.model._target_
    assert row["num_classes"] == 10
    assert row["partition"] == cfg.dataset.partition
    assert row["bands"] == cfg.dataset.bands
    assert row["config_hash"] == _resume_config_hash(cfg)


def test_implicit_gpu_knn_fallback_reaches_evaluator_as_cpu(tmp_path: Path, monkeypatch):
    import torchgeo_bench.knn as knn

    out = tmp_path / "out.csv"
    cfg = _compose_cfg(
        out,
        overrides=["device=cuda:0", "eval.knn_device=null", "eval.skip_linear=true"],
    )
    model = _chainable_model_mock()
    monkeypatch.setattr(knn, "gpu_faiss_available", lambda: False)

    with (
        mock.patch("torchgeo_bench.main.get_datasets", return_value=_synthetic_loaders()),
        mock.patch("torchgeo_bench.main.instantiate", return_value=model),
        mock.patch("torchgeo_bench.main.embed_split", side_effect=_synthetic_embeddings()),
        mock.patch(
            "torchgeo_bench.main.evaluate_knn",
            return_value=(0.5, 0.45, 0.55, {"ece": 0.05, "rms_ce": 0.07, "mce": 0.1}, 6),
        ) as knn_mock,
    ):
        main(cfg)

    assert knn_mock.call_args.kwargs["device"] == "cpu"


def test_explicit_gpu_knn_without_gpu_faiss_fails_before_data_loading(tmp_path: Path, monkeypatch):
    import torchgeo_bench.knn as knn

    out = tmp_path / "out.csv"
    cfg = _compose_cfg(
        out,
        overrides=["device=cuda:0", "eval.knn_device=cuda", "eval.skip_linear=true"],
    )
    monkeypatch.setattr(knn, "gpu_faiss_available", lambda: False)

    with (
        mock.patch("torchgeo_bench.main.get_datasets") as data_mock,
        pytest.raises(RuntimeError, match="explicit KNN device 'cuda'"),
    ):
        main(cfg)

    data_mock.assert_not_called()


def test_linear_row_emitted(tmp_path: Path):
    out = tmp_path / "out.csv"
    cfg = _compose_cfg(out)

    with (
        mock.patch("torchgeo_bench.main.get_datasets", return_value=_synthetic_loaders()),
        mock.patch("torchgeo_bench.main.embed_split", side_effect=_synthetic_embeddings()),
        mock.patch(
            "torchgeo_bench.main.evaluate_knn",
            return_value=(0.5, 0.45, 0.55, {"ece": 0.05, "rms_ce": 0.07, "mce": 0.1}, 6),
        ),
        mock.patch(
            "torchgeo_bench.main.evaluate_logistic",
            return_value=(
                0.6,
                0.52,
                0.66,
                0.1,
                {"ece": 0.04, "rms_ce": 0.06, "mce": 0.09},
                {"ece_ts": 0.04, "rms_ce_ts": 0.06, "mce_ts": 0.09, "temperature": 0.8},
            ),
        ),
    ):
        main(cfg)

    df = pd.read_csv(out)
    assert "linear" in df["method"].values
    row = df[df["method"] == "linear"].iloc[0]
    assert row["metric_name"] == "accuracy"


@pytest.mark.parametrize(
    ("error", "strict"),
    [
        (RuntimeError("linear probe failed"), False),
        (RuntimeError("linear probe failed"), True),
        (LinearProbeDivergedError("linear probe failed"), True),
    ],
)
def test_completed_knn_survives_later_linear_failure(
    tmp_path: Path, *, error: RuntimeError, strict: bool
) -> None:
    out = tmp_path / "out.csv"
    cfg = _compose_cfg(out)

    with (
        mock.patch("torchgeo_bench.main.get_datasets", return_value=_synthetic_loaders()),
        mock.patch("torchgeo_bench.main.instantiate", return_value=_chainable_model_mock()),
        mock.patch("torchgeo_bench.main.embed_split", side_effect=_synthetic_embeddings()),
        mock.patch(
            "torchgeo_bench.main.evaluate_knn",
            return_value=(0.5, 0.45, 0.55, {"ece": 0.05, "rms_ce": 0.07, "mce": 0.1}, 6),
        ),
        mock.patch("torchgeo_bench.main.evaluate_logistic", side_effect=error),
        pytest.raises(RuntimeError, match="linear probe failed"),
    ):
        main(cfg, strict=strict)

    df = pd.read_csv(out)
    assert list(df["method"]) == ["knn5"]
    row = df.iloc[0]
    assert (row["metric_value"], row["ci_lower"], row["ci_upper"]) == (0.5, 0.45, 0.55)
    assert row["ece"] == 0.05
    assert row["config_hash"] == _resume_config_hash(cfg)


def test_resume_skips_completed_knn_row(tmp_path: Path):
    out = tmp_path / "out.csv"
    cfg = _compose_cfg(out, overrides=["resume=true", "eval.skip_linear=true"])
    pd.DataFrame([_resume_row(cfg, method="knn5", metric_name="accuracy")]).to_csv(out, index=False)

    with (
        mock.patch("torchgeo_bench.main.get_datasets", return_value=_synthetic_loaders()),
        mock.patch("torchgeo_bench.main.evaluate_knn") as knn_mock,
    ):
        main(cfg)

    knn_mock.assert_not_called()
    df = pd.read_csv(out)
    assert int((df["method"] == "knn5").sum()) == 1


def test_resume_complete_preflight_skips_data_loading_and_model_init(tmp_path: Path):
    out = tmp_path / "out.csv"
    cfg = _compose_cfg(out, overrides=["resume=true"])
    pd.DataFrame(
        [
            _resume_row(cfg, method="knn5", metric_name="accuracy"),
            _resume_row(cfg, method="linear", metric_name="accuracy"),
        ]
    ).to_csv(out, index=False)

    with (
        mock.patch("torchgeo_bench.main.get_datasets") as get_datasets_mock,
        mock.patch("torchgeo_bench.main.instantiate") as instantiate_mock,
    ):
        main(cfg)

    get_datasets_mock.assert_not_called()
    instantiate_mock.assert_not_called()
    assert len(pd.read_csv(out)) == 2


def test_resume_partial_completion_still_runs_missing_work(tmp_path: Path):
    out = tmp_path / "out.csv"
    cfg = _compose_cfg(out, overrides=["resume=true"])
    pd.DataFrame([_resume_row(cfg, method="knn5", metric_name="accuracy")]).to_csv(out, index=False)
    model = _chainable_model_mock()

    with (
        mock.patch(
            "torchgeo_bench.main.get_datasets", return_value=_synthetic_loaders()
        ) as data_mock,
        mock.patch("torchgeo_bench.main.instantiate", return_value=model) as instantiate_mock,
        mock.patch("torchgeo_bench.main.embed_split", side_effect=_synthetic_embeddings()),
        mock.patch("torchgeo_bench.main.evaluate_knn") as knn_mock,
        mock.patch(
            "torchgeo_bench.main.evaluate_logistic",
            return_value=(
                0.6,
                0.52,
                0.66,
                0.1,
                {"ece": 0.04, "rms_ce": 0.06, "mce": 0.09},
                {"ece_ts": 0.04, "rms_ce_ts": 0.06, "mce_ts": 0.09, "temperature": 0.8},
            ),
        ) as linear_mock,
    ):
        main(cfg)

    data_mock.assert_called_once()
    instantiate_mock.assert_called_once()
    knn_mock.assert_not_called()
    linear_mock.assert_called_once()
    df = pd.read_csv(out)
    assert int((df["method"] == "knn5").sum()) == 1
    assert int((df["method"] == "linear").sum()) == 1


def test_non_resume_still_runs_even_with_matching_existing_rows(tmp_path: Path):
    out = tmp_path / "out.csv"
    cfg = _compose_cfg(out, overrides=["eval.skip_linear=true"])
    pd.DataFrame([_resume_row(cfg, method="knn5", metric_name="accuracy")]).to_csv(out, index=False)
    model = _chainable_model_mock()

    with (
        mock.patch(
            "torchgeo_bench.main.get_datasets", return_value=_synthetic_loaders()
        ) as data_mock,
        mock.patch("torchgeo_bench.main.instantiate", return_value=model) as instantiate_mock,
        mock.patch("torchgeo_bench.main.embed_split", side_effect=_synthetic_embeddings()),
        mock.patch(
            "torchgeo_bench.main.evaluate_knn",
            return_value=(0.5, 0.45, 0.55, {"ece": 0.05, "rms_ce": 0.07, "mce": 0.1}, 6),
        ) as knn_mock,
    ):
        main(cfg)

    data_mock.assert_called_once()
    instantiate_mock.assert_called_once()
    knn_mock.assert_called_once()
    assert int((pd.read_csv(out)["method"] == "knn5").sum()) == 2


@pytest.mark.parametrize("strict", [False, True])
def test_model_eval_overrides_do_not_change_classification_resume_semantics(
    tmp_path: Path, *, strict: bool
) -> None:
    out = tmp_path / "out.csv"
    cfg = _compose_cfg(
        out,
        overrides=[
            "resume=true",
            "model=timm/resnet50",
            "eval.knn_device=cpu",
            "eval.merge_val=false",
            "eval.calibration.n_bins_linear=15",
            "+model.eval.knn_k=7",
            "+model.eval.skip_linear=true",
            "+model.eval.bootstrap=99",
            "+model.eval.merge_val=true",
            "+model.eval.knn_device=meta-device",
            "+model.eval.calibration.n_bins_linear=99",
            "+model.eval.c_range=[-5,-4,2]",
        ],
    )
    method = "knn5" if strict else "knn7"
    pd.DataFrame([_resume_row(cfg, method=method, metric_name="accuracy")]).to_csv(out, index=False)
    model = _chainable_model_mock()

    with (
        mock.patch(
            "torchgeo_bench.main.get_datasets", return_value=_synthetic_loaders()
        ) as data_mock,
        mock.patch("torchgeo_bench.main.instantiate", return_value=model) as instantiate_mock,
        mock.patch("torchgeo_bench.main.embed_split", side_effect=_synthetic_embeddings()),
        mock.patch("torchgeo_bench.main.evaluate_knn") as knn_mock,
        mock.patch(
            "torchgeo_bench.main.evaluate_logistic",
            return_value=(
                0.6,
                0.52,
                0.66,
                0.1,
                {"ece": 0.04, "rms_ce": 0.06, "mce": 0.09},
                {"ece_ts": 0.04, "rms_ce_ts": 0.06, "mce_ts": 0.09, "temperature": 0.8},
            ),
        ) as linear_mock,
    ):
        main(cfg, strict=strict)

    data_mock.assert_called_once()
    instantiate_mock.assert_called_once()
    knn_mock.assert_not_called()
    linear_mock.assert_called_once()

    linear_call = linear_mock.call_args
    expected_start, expected_stop, _ = cfg.eval.c_range if strict else cfg.model.eval.c_range
    assert linear_call.args[1] == pytest.approx([10**expected_start, 10**expected_stop])
    evaluation_cfg = linear_call.args[2].eval
    assert evaluation_cfg.bootstrap == cfg.eval.bootstrap
    assert evaluation_cfg.merge_val is cfg.eval.merge_val
    assert evaluation_cfg.calibration.n_bins_linear == cfg.eval.calibration.n_bins_linear

    df = pd.read_csv(out)
    linear_row = df[df["method"] == "linear"].iloc[0]
    assert linear_row["bootstrap"] == cfg.eval.bootstrap
    assert bool(linear_row["merge_val"]) is bool(cfg.eval.merge_val)
    assert int((df["method"] == "linear").sum()) == 1
    assert int((df["method"] == method).sum()) == 1
    assert linear_row["c_range_start"] == expected_start
    assert linear_row["c_range_stop"] == expected_stop


def test_resume_skips_when_image_size_read_as_float(tmp_path: Path):
    """Resume must match CSV floats such as ``224.0`` to integer config values such as ``224``."""
    out = tmp_path / "out.csv"
    cfg = _compose_cfg(out, overrides=["resume=true", "eval.skip_linear=true"])
    df = pd.DataFrame([_resume_row(cfg, method="knn5", metric_name="accuracy")])
    df["image_size"] = df["image_size"].astype(float)
    df.to_csv(out, index=False)

    with (
        mock.patch("torchgeo_bench.main.get_datasets", return_value=_synthetic_loaders()),
        mock.patch("torchgeo_bench.main.evaluate_knn") as knn_mock,
    ):
        main(cfg)

    knn_mock.assert_not_called()
    assert int((pd.read_csv(out)["method"] == "knn5").sum()) == 1


def test_resume_recomputes_legacy_row_without_num_classes(tmp_path: Path):
    """Rows from an older label schema must not satisfy the current resume key."""
    out = tmp_path / "out.csv"
    cfg = _compose_cfg(out, overrides=["resume=true", "eval.skip_linear=true"])
    legacy_row = _resume_row(cfg, method="knn5", metric_name="accuracy")
    legacy_row.pop("num_classes")
    pd.DataFrame([legacy_row]).to_csv(out, index=False)
    model = _chainable_model_mock()

    with (
        mock.patch("torchgeo_bench.main.get_datasets", return_value=_synthetic_loaders()),
        mock.patch("torchgeo_bench.main.instantiate", return_value=model),
        mock.patch("torchgeo_bench.main.embed_split", side_effect=_synthetic_embeddings()),
        mock.patch(
            "torchgeo_bench.main.evaluate_knn",
            return_value=(0.5, 0.45, 0.55, {"ece": 0.05, "rms_ce": 0.07, "mce": 0.1}, 6),
        ) as knn_mock,
    ):
        main(cfg)

    knn_mock.assert_called_once()
    df = pd.read_csv(out)
    assert len(df) == 2
    assert pd.isna(df.iloc[0]["num_classes"])
    assert int(df.iloc[1]["num_classes"]) == 10


def test_resume_reruns_when_evaluation_config_changes(tmp_path: Path):
    """A stale row must not suppress a run with a different random seed."""
    out = tmp_path / "out.csv"
    old_cfg = _compose_cfg(out, overrides=["eval.skip_linear=true"])
    cfg = _compose_cfg(out, overrides=["resume=true", "seed=7", "eval.skip_linear=true"])
    pd.DataFrame([_resume_row(old_cfg, method="knn5", metric_name="accuracy")]).to_csv(
        out, index=False
    )

    with (
        mock.patch("torchgeo_bench.main.get_datasets", return_value=_synthetic_loaders()),
        mock.patch("torchgeo_bench.main.embed_split", side_effect=_synthetic_embeddings()),
        mock.patch(
            "torchgeo_bench.main.evaluate_knn",
            return_value=(0.5, 0.45, 0.55, {"ece": 0.05, "rms_ce": 0.07, "mce": 0.1}, 6),
        ) as knn_mock,
    ):
        main(cfg)

    knn_mock.assert_called_once()


@pytest.mark.parametrize(
    "error", [FileNotFoundError("missing dataset"), DatasetNotFoundError("missing")]
)
def test_missing_dataset_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    out = tmp_path / "out.csv"
    cfg = _compose_cfg(out)

    def missing_dataset(**_kwargs: object) -> None:
        raise error

    monkeypatch.setattr("torchgeo_bench.main.get_datasets", missing_dataset)
    with pytest.raises(type(error)) as exc_info:
        main(cfg)
    assert exc_info.value is error
    assert not out.exists()


def test_unknown_dataset_raises(tmp_path: Path) -> None:
    cfg = _compose_cfg(tmp_path / "out.csv", overrides=["dataset.names=[unknown-dataset]"])
    with pytest.raises(KeyError, match="unknown-dataset"):
        main(cfg)
