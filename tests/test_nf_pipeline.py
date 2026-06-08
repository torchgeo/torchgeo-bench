import importlib.util

import numpy as np
import pandas as pd
import pytest
from omegaconf import OmegaConf


@pytest.fixture(scope="module")
def _tiny_xy():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((90, 16)).astype(np.float32)
    y = np.repeat([0, 1, 2], 30).astype(np.int64)
    return X, y


def test_nf_pipeline_writes_nf_results_csv(tmp_path, monkeypatch, _tiny_xy):
    if importlib.util.find_spec("zuko") is None or importlib.util.find_spec("optuna") is None:
        pytest.skip("zuko/optuna not installed")
    X, y = _tiny_xy

    monkeypatch.setattr("torchgeo_bench.nf_pipeline.extract_features", lambda *a, **kw: (X, y))
    monkeypatch.setattr(
        "torchgeo_bench.nf_pipeline.get_bench_dataset_class",
        lambda _: type("D", (), {"task": "classification", "multilabel": False})(),
    )
    monkeypatch.setattr("torchgeo_bench.nf_pipeline.get_datasets", lambda **_: (None, None, None, None))
    monkeypatch.setattr("torchgeo_bench.nf_pipeline.instantiate", lambda *a, **kw: object())

    from torchgeo_bench.nf_pipeline import main as nf_main

    output_csv = str(tmp_path / "nf_results.csv")
    cfg = OmegaConf.create({
        "seed": 0, "device": "cpu", "verbose": False, "resume": False,
        "model": {"_target_": "dummy.T", "name": "resnet50"},
        "dataset": {"names": ["m-eurosat"], "partition": "default",
                    "batch_size": 2, "num_workers": 0, "bands": "rgb",
                    "interpolation": "bilinear", "normalization": "bandspec_zscore"},
        "nf": {"output": output_csv, "n_trials": 2, "bootstrap": 10,
               "epochs": 2, "batch_size": 32},
    })
    nf_main.__wrapped__(cfg)

    df = pd.read_csv(output_csv)
    required_cols = {"model", "name", "dataset", "metric_name", "metric_value",
                     "best_lr", "best_wd", "val_nll", "n_trials"}
    assert required_cols.issubset(set(df.columns))
    assert set(df["metric_name"]) == {"accuracy", "nll", "ece", "brier"}


def test_nf_pipeline_resume_skips_extraction(tmp_path, monkeypatch):
    if importlib.util.find_spec("zuko") is None or importlib.util.find_spec("optuna") is None:
        pytest.skip("zuko/optuna not installed")

    rows = [
        {"model": "dummy.T", "name": "resnet50", "dataset": "m-eurosat",
         "partition": "default", "bands": "rgb",
         "metric_name": m, "metric_value": 0.5, "best_lr": 1e-3, "best_wd": 1e-4,
         "val_nll": 1.0, "n_trials": 2, "seed": 0}
        for m in ["accuracy", "nll", "ece", "brier"]
    ]
    csv_path = tmp_path / "nf_results.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    extract_calls: list[int] = []
    monkeypatch.setattr(
        "torchgeo_bench.nf_pipeline.extract_features",
        lambda *a, **kw: extract_calls.append(1) or (np.zeros((4, 4), np.float32), np.zeros(4, np.int64)),
    )
    monkeypatch.setattr(
        "torchgeo_bench.nf_pipeline.get_bench_dataset_class",
        lambda _: type("D", (), {"task": "classification", "multilabel": False})(),
    )
    monkeypatch.setattr("torchgeo_bench.nf_pipeline.get_datasets", lambda **_: (None, None, None, None))
    monkeypatch.setattr("torchgeo_bench.nf_pipeline.instantiate", lambda *a, **kw: object())

    from torchgeo_bench.nf_pipeline import main as nf_main

    cfg = OmegaConf.create({
        "seed": 0, "device": "cpu", "verbose": False, "resume": True,
        "model": {"_target_": "dummy.T", "name": "resnet50"},
        "dataset": {"names": ["m-eurosat"], "partition": "default",
                    "batch_size": 2, "num_workers": 0, "bands": "rgb",
                    "interpolation": "bilinear", "normalization": "bandspec_zscore"},
        "nf": {"output": str(csv_path), "n_trials": 2, "bootstrap": 10,
               "epochs": 2, "batch_size": 32},
    })
    nf_main.__wrapped__(cfg)
    assert len(extract_calls) == 0
