"""Offline tests for the classification runner."""

from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, Dataset
from torchgeo.datasets import DatasetNotFoundError

from torchgeo_bench.config_schema import RunConfig
from torchgeo_bench.main import LinearProbeDivergedError, main
from torchgeo_bench.presets import NORMALIZATIONS, ModelPreset, merge_settings, resolve_run_config
from torchgeo_bench.resume import _resume_config_hash


class _DictTensorDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor) -> None:
        self._images = images
        self._labels = labels

    def __len__(self) -> int:
        return int(self._images.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "image": self._images[index],
            "label": self._labels[index],
        }


def _compose_cfg(output_path: Path, overrides: dict | None = None) -> RunConfig:
    return RunConfig.model_validate(
        merge_settings(
            {
                "model": {"name": "rcf"},
                "datasets": ["m-eurosat"],
                "runtime": {"batch_size": 4, "workers": 0, "device": "cpu"},
                "classification": {
                    "bootstrap_samples": 5,
                    "linear": {"c_log10_start": -2.0, "c_log10_stop": -1.0, "c_count": 2},
                },
                "output": {"file": str(output_path)},
            },
            overrides or {},
        )
    )


def _synthetic_loaders(
    n_train: int = 16,
    n_val: int = 8,
    n_test: int = 8,
    n_classes: int = 10,
    channels: int = 3,
) -> tuple[_DictTensorDataset, DataLoader, DataLoader, DataLoader]:
    rng = torch.Generator().manual_seed(0)
    train_images = torch.rand(n_train, channels, 64, 64, generator=rng) * 3000.0
    val_images = torch.rand(n_val, channels, 64, 64, generator=rng) * 3000.0
    test_images = torch.rand(n_test, channels, 64, 64, generator=rng) * 3000.0

    train_labels = torch.randint(0, n_classes, (n_train,), generator=rng)
    val_labels = torch.randint(0, n_classes, (n_val,), generator=rng)
    test_labels = torch.randint(0, n_classes, (n_test,), generator=rng)

    train_dataset = _DictTensorDataset(train_images, train_labels)
    val_dataset = _DictTensorDataset(val_images, val_labels)
    test_dataset = _DictTensorDataset(test_images, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)
    return train_dataset, train_loader, val_loader, test_loader


def _synthetic_embeddings() -> list[tuple[np.ndarray, np.ndarray]]:
    """Embeddings in the order of the train, validation, and test calls."""
    rng = np.random.default_rng(0)
    x_train = rng.standard_normal((16, 8), dtype=np.float32)
    y_train = rng.integers(0, 10, size=(16,), dtype=np.int64)
    x_val = rng.standard_normal((8, 8), dtype=np.float32)
    y_val = rng.integers(0, 10, size=(8,), dtype=np.int64)
    x_test = rng.standard_normal((8, 8), dtype=np.float32)
    y_test = rng.integers(0, 10, size=(8,), dtype=np.int64)
    return [(x_train, y_train), (x_val, y_val), (x_test, y_test)]


def _resume_row(cfg: RunConfig, *, method: str, metric_name: str) -> dict[str, object]:
    """Seed the CSV with a row matching this configuration."""
    config_hash = _resume_config_hash(cfg)
    cfg, preset = resolve_run_config(cfg, "m-eurosat")
    return {
        "dataset": "m-eurosat",
        "method": method,
        "model": preset.target,
        "name": preset.name,
        "normalization": NORMALIZATIONS[cfg.input.normalization],
        "image_size": cfg.input.image_size,
        "interpolation": cfg.input.interpolation,
        "partition": cfg.input.partition,
        "bands": cfg.input.bands,
        "num_classes": 10,
        "config_hash": config_hash,
        "metric_name": metric_name,
        "metric_value": 0.1,
    }


def _chainable_model_mock() -> mock.Mock:
    model = mock.Mock()
    model.to.return_value = model
    model.eval.return_value = model
    return model


def test_model_dataset_overrides_are_isolated_and_fall_back() -> None:
    model_cfg = ModelPreset.model_validate(
        {
            "target": "example.Model",
            "name": "example",
            "input": {"image_size": 224},
            "kwargs": {"res": 1.0, "pool": "cls"},
            "dataset_overrides": {
                "m-eurosat": {"input": {"image_size": 64}, "kwargs": {"res": 3.5}},
                "forestnet": {"input": {"image_size": 128}, "kwargs": {"pool": "mean"}},
            },
        }
    )

    eurosat = model_cfg.for_dataset("m-eurosat")
    fallback = model_cfg.for_dataset("unlisted")
    forestnet = model_cfg.for_dataset("forestnet")

    assert (eurosat.input.image_size, eurosat.kwargs["res"], eurosat.kwargs["pool"]) == (
        64,
        3.5,
        "cls",
    )
    assert (fallback.input.image_size, fallback.kwargs["res"], fallback.kwargs["pool"]) == (
        224,
        1.0,
        "cls",
    )
    assert (forestnet.input.image_size, forestnet.kwargs["res"], forestnet.kwargs["pool"]) == (
        128,
        1.0,
        "mean",
    )
    assert not eurosat.dataset_overrides
    assert model_cfg.input.image_size == 224


def test_dataset_override_routes_recipe_and_changes_resume_key(tmp_path: Path) -> None:
    out = tmp_path / "out.csv"
    cfg = _compose_cfg(
        out,
        overrides={
            "classification": {"methods": ["knn"]},
            "model": {
                "name": "torchgeo/scalemae_large_fmow",
                "kwargs": {"res": 3.5, "pool": "cls"},
            },
            "input": {"image_size": 64, "interpolation": "area"},
        },
    )
    model = _chainable_model_mock()

    with (
        mock.patch(
            "torchgeo_bench.main.get_datasets", return_value=_synthetic_loaders()
        ) as data_mock,
        mock.patch("torchgeo_bench.main.build_model", return_value=model) as instantiate_mock,
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
    assert instantiated_cfg.kwargs["res"] == 3.5
    assert instantiated_cfg.kwargs["pool"] == "cls"
    assert "interpolation" not in instantiated_cfg.kwargs
    assert "image_size" not in instantiated_cfg.kwargs

    first_row = pd.read_csv(out).iloc[0]
    assert first_row["res"] == 3.5
    assert first_row["pool"] == "cls"

    changed_cfg = cfg.model_copy(deep=True)
    changed_cfg.output.resume = True
    changed_cfg.model.kwargs["pool"] = "mean"
    with (
        mock.patch("torchgeo_bench.main.get_datasets", return_value=_synthetic_loaders()),
        mock.patch("torchgeo_bench.main.build_model", return_value=model),
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


def test_knn_row_emitted(tmp_path: Path):
    out = tmp_path / "out.csv"
    cfg = _compose_cfg(out, overrides={"classification": {"methods": ["knn"]}})

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
    assert "knn5" in df["method"].values
    row = df[df["method"] == "knn5"].iloc[0]
    assert row["metric_name"] == "accuracy"
    assert row["dataset"] == "m-eurosat"


def test_implicit_gpu_knn_fallback_reaches_evaluator_as_cpu(tmp_path: Path, monkeypatch):
    import torchgeo_bench.knn as knn

    out = tmp_path / "out.csv"
    cfg = _compose_cfg(
        out,
        overrides={
            "runtime": {"device": "cuda:0"},
            "classification": {"knn_device": None, "methods": ["knn"]},
        },
    )
    model = _chainable_model_mock()
    monkeypatch.setattr(knn, "gpu_faiss_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    with (
        mock.patch("torchgeo_bench.main.get_datasets", return_value=_synthetic_loaders()),
        mock.patch("torchgeo_bench.main.build_model", return_value=model),
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
        overrides={"classification": {"knn_device": "cuda", "methods": ["knn"]}},
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
        mock.patch("torchgeo_bench.main.build_model", return_value=_chainable_model_mock()),
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
    cfg = _compose_cfg(
        out, overrides={"output": {"resume": True}, "classification": {"methods": ["knn"]}}
    )
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
    cfg = _compose_cfg(out, overrides={"output": {"resume": True}})
    pd.DataFrame(
        [
            _resume_row(cfg, method="knn5", metric_name="accuracy"),
            _resume_row(cfg, method="linear", metric_name="accuracy"),
        ]
    ).to_csv(out, index=False)

    with (
        mock.patch("torchgeo_bench.main.get_datasets") as get_datasets_mock,
        mock.patch("torchgeo_bench.main.build_model") as instantiate_mock,
    ):
        main(cfg)

    get_datasets_mock.assert_not_called()
    instantiate_mock.assert_not_called()
    assert len(pd.read_csv(out)) == 2


def test_resume_partial_completion_still_runs_missing_work(tmp_path: Path):
    out = tmp_path / "out.csv"
    cfg = _compose_cfg(out, overrides={"output": {"resume": True}})
    pd.DataFrame([_resume_row(cfg, method="knn5", metric_name="accuracy")]).to_csv(out, index=False)
    model = _chainable_model_mock()

    with (
        mock.patch(
            "torchgeo_bench.main.get_datasets", return_value=_synthetic_loaders()
        ) as data_mock,
        mock.patch("torchgeo_bench.main.build_model", return_value=model) as instantiate_mock,
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
    cfg = _compose_cfg(out, overrides={"classification": {"methods": ["knn"]}})
    pd.DataFrame([_resume_row(cfg, method="knn5", metric_name="accuracy")]).to_csv(out, index=False)
    model = _chainable_model_mock()

    with (
        mock.patch(
            "torchgeo_bench.main.get_datasets", return_value=_synthetic_loaders()
        ) as data_mock,
        mock.patch("torchgeo_bench.main.build_model", return_value=model) as instantiate_mock,
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
        overrides={
            "output": {"resume": True},
            "model": {"name": "olmoearth_v1_2_base"},
            "classification": {
                "knn_device": "cpu",
                "knn_k": 7,
                "linear": {"refit_train_val": False, "c_log10_start": -5.0, "c_log10_stop": -4.0},
                "calibration": {"n_bins_linear": 15},
            },
        },
    )
    method = "knn7"
    pd.DataFrame([_resume_row(cfg, method=method, metric_name="accuracy")]).to_csv(out, index=False)
    model = _chainable_model_mock()

    with (
        mock.patch(
            "torchgeo_bench.main.get_datasets", return_value=_synthetic_loaders()
        ) as data_mock,
        mock.patch("torchgeo_bench.main.build_model", return_value=model) as instantiate_mock,
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
    expected_start = cfg.classification.linear.c_log10_start
    expected_stop = cfg.classification.linear.c_log10_stop
    assert linear_call.args[1] == pytest.approx([10**expected_start, 10**expected_stop])
    evaluation_cfg = linear_call.args[2].classification
    assert evaluation_cfg.bootstrap_samples == cfg.classification.bootstrap_samples
    assert evaluation_cfg.linear.refit_train_val is cfg.classification.linear.refit_train_val
    assert evaluation_cfg.calibration.n_bins_linear == cfg.classification.calibration.n_bins_linear

    df = pd.read_csv(out)
    linear_row = df[df["method"] == "linear"].iloc[0]
    assert linear_row["bootstrap"] == cfg.classification.bootstrap_samples
    assert bool(linear_row["merge_val"]) is cfg.classification.linear.refit_train_val
    assert int((df["method"] == "linear").sum()) == 1
    assert int((df["method"] == method).sum()) == 1
    assert linear_row["c_range_start"] == expected_start
    assert linear_row["c_range_stop"] == expected_stop


def test_resume_skips_when_image_size_read_as_float(tmp_path: Path):
    """Resume must match CSV floats such as ``224.0`` to integer config values such as ``224``."""
    out = tmp_path / "out.csv"
    cfg = _compose_cfg(
        out, overrides={"output": {"resume": True}, "classification": {"methods": ["knn"]}}
    )
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
    cfg = _compose_cfg(
        out, overrides={"output": {"resume": True}, "classification": {"methods": ["knn"]}}
    )
    legacy_row = _resume_row(cfg, method="knn5", metric_name="accuracy")
    legacy_row.pop("num_classes")
    pd.DataFrame([legacy_row]).to_csv(out, index=False)
    model = _chainable_model_mock()

    with (
        mock.patch("torchgeo_bench.main.get_datasets", return_value=_synthetic_loaders()),
        mock.patch("torchgeo_bench.main.build_model", return_value=model),
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
    old_cfg = _compose_cfg(out, overrides={"classification": {"methods": ["knn"]}})
    cfg = _compose_cfg(
        out,
        overrides={
            "output": {"resume": True},
            "runtime": {"seed": 7},
            "classification": {"methods": ["knn"]},
        },
    )
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
    cfg = _compose_cfg(tmp_path / "out.csv", overrides={"datasets": ["unknown-dataset"]})
    with pytest.raises(KeyError, match="unknown-dataset"):
        main(cfg)


def test_csv_row_has_required_columns(tmp_path: Path):
    out = tmp_path / "out.csv"
    cfg = _compose_cfg(out, overrides={"classification": {"methods": ["knn"]}})

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
    required = {
        "dataset",
        "method",
        "model",
        "metric_name",
        "metric_value",
        "partition",
        "bands",
        "num_classes",
    }
    assert required.issubset(set(df.columns))
