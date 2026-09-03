"""Fast offline tests for classification orchestration in ``torchgeo_bench.main``."""

from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader, Dataset
from torchgeo.datasets import DatasetNotFoundError

from torchgeo_bench.config import compose_config
from torchgeo_bench.main import main, resolve_model_config
from torchgeo_bench.results import ResultSchemaError, metric_row
from torchgeo_bench.resume import _resume_config_hash
from torchgeo_bench.settings import RunSettings, merge


class _DictTensorDataset(Dataset):
    """Small dataset wrapper that emits ``{"image", "label"}`` samples."""

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


def _compose_cfg(
    output_path: Path, overrides: dict | None = None, model: str = "rcf"
) -> RunSettings:
    """Compose the config for fast offline main-path tests."""
    base = {
        "dataset": {
            "names": ["m-eurosat"],
            "partition": "default",
            "batch_size": 4,
            "num_workers": 0,
        },
        "eval": {"bootstrap": 5, "c_range": [-2, -1, 2]},
        "device": "cpu",
        "output": str(output_path),
    }
    if overrides:
        base = merge(base, overrides)
    return compose_config(base, model=model)


def _synthetic_loaders(
    n_train: int = 16,
    n_val: int = 8,
    n_test: int = 8,
    n_classes: int = 10,
    channels: int = 3,
) -> tuple[_DictTensorDataset, DataLoader, DataLoader, DataLoader]:
    """Return train dataset + train/val/test loaders matching benchmark contract."""
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
    """Return deterministic (X, y) tuples for train/val/test ``embed_split`` calls."""
    rng = np.random.default_rng(0)
    x_train = rng.standard_normal((16, 8), dtype=np.float32)
    y_train = rng.integers(0, 10, size=(16,), dtype=np.int64)
    x_val = rng.standard_normal((8, 8), dtype=np.float32)
    y_val = rng.integers(0, 10, size=(8,), dtype=np.int64)
    x_test = rng.standard_normal((8, 8), dtype=np.float32)
    y_test = rng.integers(0, 10, size=(8,), dtype=np.int64)
    return [(x_train, y_train), (x_val, y_val), (x_test, y_test)]


def _resume_row(
    cfg: RunSettings, *, method: str, metric_name: str, num_classes: int = 10
) -> dict[str, object]:
    """Build a full ``EvaluationResult``-shaped row for pre-seeding output files.

    Mirrors ``main.py``'s real ``common_meta`` construction so the row has
    every column ``append_rows_atomic`` now strictly requires to match --
    a partial/legacy-shaped row is a schema mismatch, not something resume
    silently tolerates.
    """
    c_start, c_stop, c_num = cfg.eval.c_range
    common_meta = {
        "dataset": "m-eurosat",
        "seed": cfg.seed,
        "model": cfg.model["_target_"],
        "name": cfg.model["name"],
        "normalization": cfg.dataset.normalization,
        "image_size": cfg.dataset.image_size,
        "interpolation": cfg.dataset.interpolation,
        "partition": cfg.dataset.partition,
        "bands": cfg.dataset.bands,
        "num_classes": num_classes,
        "config_hash": _resume_config_hash(cfg),
        "c_range_start": c_start,
        "c_range_stop": c_stop,
        "c_range_num": c_num,
        "merge_val": cfg.eval.merge_val,
        "bootstrap": cfg.eval.bootstrap,
        "res": None,
        "pool": None,
    }
    return metric_row(
        common_meta,
        method=method,
        metric_name=metric_name,
        metric_value=0.1,
        feature_dim=8,
        n_counts={"train": 16, "val": 8, "test": 8},
    )


def _chainable_model_mock() -> mock.Mock:
    """Return a mock model whose ``to().eval()`` chain returns itself."""
    model = mock.Mock()
    model.to.return_value = model
    model.eval.return_value = model
    return model


def test_model_dataset_overrides_are_isolated_and_fall_back() -> None:
    model_cfg = {
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

    eurosat = resolve_model_config(model_cfg, "m-eurosat")
    fallback = resolve_model_config(model_cfg, "unlisted")
    forestnet = resolve_model_config(model_cfg, "forestnet")

    assert (eurosat["image_size"], eurosat["res"], eurosat["pool"]) == (64, 3.5, "cls")
    assert (fallback["image_size"], fallback["res"], fallback["pool"]) == (224, 1.0, "cls")
    assert (forestnet["image_size"], forestnet["res"], forestnet["pool"]) == (128, 1.0, "mean")
    assert "dataset_overrides" not in eurosat
    assert model_cfg["image_size"] == 224


def test_dataset_override_routes_recipe_and_changes_resume_key(tmp_path: Path) -> None:
    out = tmp_path / "out.csv"
    cfg = _compose_cfg(out, overrides={"eval": {"skip_linear": True}})
    cfg.model["dataset_overrides"] = {
        "m-eurosat": {
            "image_size": 64,
            "interpolation": "area",
            "res": 3.5,
            "pool": "cls",
        }
    }
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
    assert instantiated_cfg["image_size"] == 64
    assert instantiated_cfg["res"] == 3.5
    assert instantiated_cfg["pool"] == "cls"
    assert "interpolation" not in instantiated_cfg

    first_row = pd.read_csv(out).iloc[0]
    assert first_row["res"] == 3.5
    assert first_row["pool"] == "cls"

    changed_cfg = _compose_cfg(out, overrides={"resume": True, "eval": {"skip_linear": True}})
    changed_cfg.model["dataset_overrides"] = {
        "m-eurosat": {
            "image_size": 64,
            "interpolation": "area",
            "res": 3.5,
            "pool": "mean",
        }
    }
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


def test_get_datasets_pin_memory_false_for_cpu(tmp_path: Path):
    """A CPU run must never pin dataloader memory."""
    out = tmp_path / "out.csv"
    cfg = _compose_cfg(out, overrides={"eval": {"skip_linear": True}})

    with (
        mock.patch(
            "torchgeo_bench.main.get_datasets", return_value=_synthetic_loaders()
        ) as data_mock,
        mock.patch("torchgeo_bench.main.embed_split", side_effect=_synthetic_embeddings()),
        mock.patch(
            "torchgeo_bench.main.evaluate_knn",
            return_value=(0.5, 0.45, 0.55, {"ece": 0.05, "rms_ce": 0.07, "mce": 0.1}, 6),
        ),
    ):
        main(cfg)

    assert data_mock.call_args.kwargs["pin_memory"] is False


def test_get_datasets_pin_memory_true_for_cuda_device(tmp_path: Path, monkeypatch):
    """pin_memory must follow the resolved run device, not global CUDA
    availability -- so a CPU run doesn't pin memory merely because some other
    GPU exists on the machine, and a CUDA run does pin it."""
    out = tmp_path / "out.csv"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    cfg = _compose_cfg(out, overrides={"device": "cuda:0", "eval": {"skip_linear": True}})
    model = _chainable_model_mock()

    with (
        mock.patch(
            "torchgeo_bench.main.get_datasets", return_value=_synthetic_loaders()
        ) as data_mock,
        mock.patch("torchgeo_bench.main.instantiate", return_value=model),
        mock.patch("torchgeo_bench.main.embed_split", side_effect=_synthetic_embeddings()),
        mock.patch(
            "torchgeo_bench.main.evaluate_knn",
            return_value=(0.5, 0.45, 0.55, {"ece": 0.05, "rms_ce": 0.07, "mce": 0.1}, 6),
        ),
    ):
        main(cfg)

    assert data_mock.call_args.kwargs["pin_memory"] is True


def test_knn_row_emitted(tmp_path: Path):
    out = tmp_path / "out.csv"
    cfg = _compose_cfg(out, overrides={"eval": {"skip_linear": True}})

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
        overrides={"device": "cuda:0", "eval": {"knn_device": None, "skip_linear": True}},
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
        overrides={"device": "cuda:0", "eval": {"knn_device": "cuda", "skip_linear": True}},
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


def test_resume_skips_completed_knn_row(tmp_path: Path):
    out = tmp_path / "out.csv"
    cfg = _compose_cfg(out, overrides={"resume": True, "eval": {"skip_linear": True}})
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
    cfg = _compose_cfg(out, overrides={"resume": True})
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
    cfg = _compose_cfg(out, overrides={"resume": True})
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
    cfg = _compose_cfg(out, overrides={"eval": {"skip_linear": True}})
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


def test_model_eval_overrides_do_not_change_classification_resume_semantics(tmp_path: Path):
    out = tmp_path / "out.csv"
    cfg = _compose_cfg(
        out,
        overrides={
            "resume": True,
            "eval": {
                "knn_device": "cpu",
                "merge_val": False,
                "calibration": {"n_bins_linear": 15},
            },
        },
        model="timm/resnet50",
    )
    cfg.model["eval"] = merge(
        cfg.model["eval"],
        {
            "knn_k": 7,
            "skip_linear": True,
            "bootstrap": 99,
            "merge_val": True,
            "knn_device": "meta-device",
            "calibration": {"n_bins_linear": 99},
        },
    )
    pd.DataFrame([_resume_row(cfg, method="knn7", metric_name="accuracy")]).to_csv(out, index=False)
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

    linear_call = linear_mock.call_args
    assert linear_call.args[8] == cfg.eval.bootstrap
    assert linear_call.args[9] is cfg.eval.merge_val
    assert linear_call.kwargs["calibration_n_bins"] == cfg.eval.calibration.n_bins_linear

    df = pd.read_csv(out)
    linear_row = df[df["method"] == "linear"].iloc[0]
    assert linear_row["bootstrap"] == cfg.eval.bootstrap
    assert bool(linear_row["merge_val"]) is bool(cfg.eval.merge_val)
    assert int((df["method"] == "knn7").sum()) == 1
    assert int((df["method"] == "linear").sum()) == 1


def test_resume_skips_when_image_size_read_as_float(tmp_path: Path):
    """Regression for the resume-key int/float mismatch (image_size).

    A populated results CSV with any missing ``image_size`` cell is typed by
    pandas as ``float64``, so the default ``224`` round-trips as ``"224.0"``
    while the config-side key is ``"224"``. Resume must still treat the row
    as complete instead of recomputing and appending a duplicate.
    """
    out = tmp_path / "out.csv"
    cfg = _compose_cfg(out, overrides={"resume": True, "eval": {"skip_linear": True}})
    df = pd.DataFrame([_resume_row(cfg, method="knn5", metric_name="accuracy")])
    # Reproduce the float64 dtype a real (partially-NaN) results CSV exhibits.
    df["image_size"] = df["image_size"].astype(float)
    df.to_csv(out, index=False)

    with (
        mock.patch("torchgeo_bench.main.get_datasets", return_value=_synthetic_loaders()),
        mock.patch("torchgeo_bench.main.evaluate_knn") as knn_mock,
    ):
        main(cfg)

    knn_mock.assert_not_called()
    assert int((pd.read_csv(out)["method"] == "knn5").sum()) == 1


def test_resume_legacy_row_without_num_classes_requires_migration(tmp_path: Path):
    """A results CSV from an older schema (missing a column ``EvaluationResult``
    now has) is not something resume silently heals: it fails clearly instead
    of writing a duplicate row alongside a mismatched-schema legacy one.

    The resume-key check itself still treats the legacy row as "not
    complete" (a missing key column backfills to "" for matching purposes,
    which cannot equal the current run's real ``num_classes``), so ``knn``
    still runs -- it's the *append* of the newly computed row that must
    refuse to silently coexist with the old schema.
    """
    out = tmp_path / "out.csv"
    cfg = _compose_cfg(out, overrides={"resume": True, "eval": {"skip_linear": True}})
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
        pytest.raises(ResultSchemaError, match="schema mismatch"),
    ):
        main(cfg)

    knn_mock.assert_called_once()


def test_resume_reruns_when_evaluation_config_changes(tmp_path: Path):
    """A stale row must not suppress a run with a different random seed."""
    out = tmp_path / "out.csv"
    old_cfg = _compose_cfg(out, overrides={"eval": {"skip_linear": True}})
    cfg = _compose_cfg(out, overrides={"resume": True, "seed": 7, "eval": {"skip_linear": True}})
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


def test_dataset_not_found_skips_when_names_is_all(tmp_path: Path):
    """dataset.names=all is a "run whatever is available" request: missing
    local data for one dataset is expected and skipped with a warning."""
    out = tmp_path / "out.csv"
    cfg = _compose_cfg(out, overrides={"dataset": {"names": "all"}})

    with mock.patch(
        "torchgeo_bench.main.get_datasets", side_effect=DatasetNotFoundError("missing")
    ):
        main(cfg)

    assert not out.exists()


def test_dataset_not_found_fails_clearly_for_explicit_names(tmp_path: Path):
    """An explicitly requested dataset must fail loudly if its data is
    missing, not silently shrink the run -- unlike dataset.names=all."""
    out = tmp_path / "out.csv"
    cfg = _compose_cfg(out)  # default names=[m-eurosat], an explicit list

    with (
        mock.patch("torchgeo_bench.main.get_datasets", side_effect=DatasetNotFoundError("missing")),
        pytest.raises(DatasetNotFoundError),
    ):
        main(cfg)

    assert not out.exists()


def test_unknown_dataset_name_fails_fast_with_available_names(tmp_path: Path):
    """A typo in dataset.names must be a startup error, not a silent per-dataset
    skip discovered mid-sweep after the model was already loaded."""
    out = tmp_path / "out.csv"
    cfg = _compose_cfg(out, overrides={"dataset": {"names": ["bogus-dataset", "m-eurosat"]}})

    with mock.patch("torchgeo_bench.main.get_datasets") as get_datasets_mock:
        with pytest.raises(ValueError, match="Unknown dataset\\(s\\): bogus-dataset"):
            main(cfg)
        get_datasets_mock.assert_not_called()

    assert not out.exists()


def test_csv_row_has_required_columns(tmp_path: Path):
    out = tmp_path / "out.csv"
    cfg = _compose_cfg(out, overrides={"eval": {"skip_linear": True}})

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
