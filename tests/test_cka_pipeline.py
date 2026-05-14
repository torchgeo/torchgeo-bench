from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from torchgeo_bench.cka.pipeline import _build_cka_resume_set, _lookup_best_c, run_cka
from torchgeo_bench.datasets.base import BandSpec


def _base_row(layer_index: int, corruption_type: str = "poisson_gaussian", severity: int = 1) -> dict:
    return {
        "model": "m.t",
        "name": "resnet50",
        "seed": 42,
        "dataset": "m-eurosat",
        "normalization": "bandspec_zscore",
        "image_size": 224,
        "interpolation": "bilinear",
        "partition": "default",
        "bands": "rgb",
        "corruption_type": corruption_type,
        "severity": severity,
        "layer_index": layer_index,
    }


def test_build_cka_resume_set_empty_csv(tmp_path):
    csv_path = tmp_path / "cka_results.csv"
    done = _build_cka_resume_set(str(csv_path), {"resnet50": 4})
    assert done == set()


def test_build_cka_resume_set_complete_key(tmp_path):
    csv_path = tmp_path / "cka_results.csv"
    pd.DataFrame([_base_row(i) for i in range(4)]).to_csv(csv_path, index=False)
    done = _build_cka_resume_set(str(csv_path), {"resnet50": 4})
    key = (
        "m.t",
        "resnet50",
        "42",
        "m-eurosat",
        "bandspec_zscore",
        "224",
        "bilinear",
        "default",
        "rgb",
        "poisson_gaussian",
        "1",
    )
    assert key in done


def test_build_cka_resume_set_partial_key(tmp_path):
    csv_path = tmp_path / "cka_results.csv"
    pd.DataFrame([_base_row(i) for i in range(3)]).to_csv(csv_path, index=False)
    done = _build_cka_resume_set(str(csv_path), {"resnet50": 4})
    key = (
        "m.t",
        "resnet50",
        "42",
        "m-eurosat",
        "bandspec_zscore",
        "224",
        "bilinear",
        "default",
        "rgb",
        "poisson_gaussian",
        "1",
    )
    assert key not in done


def test_lookup_best_c_found():
    df = pd.DataFrame(
        [
            {
                "method": "linear",
                "best_c": 0.01,
                "dataset": "m-eurosat",
                "name": "resnet50",
                "partition": "default",
                "bands": "rgb",
            }
        ]
    )
    got = _lookup_best_c(
        df,
        {
            "dataset": "m-eurosat",
            "name": "resnet50",
            "partition": "default",
            "bands": "rgb",
        },
    )
    assert got == 0.01


def test_lookup_best_c_missing_returns_none():
    df = pd.DataFrame([{"method": "linear", "best_c": 0.01, "dataset": "m-forestnet"}])
    got = _lookup_best_c(df, {"dataset": "m-eurosat", "name": "resnet50"})
    assert got is None


class _DummyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Module()
        self.backbone.layer1 = nn.Identity()
        self.backbone.layer2 = nn.Identity()
        self.backbone.layer3 = nn.Identity()
        self.backbone.layer4 = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _DummyBench:
    task = "classification"
    multilabel = False
    rgb_bands = ("red", "green", "blue")
    bands = [
        BandSpec(
            sensor="s2",
            name="red",
            source_name="red",
            mean=0.0,
            std=1.0,
            min=0.0,
            max=1.0,
            wavelength_um=0.665,
        ),
        BandSpec(
            sensor="s2",
            name="green",
            source_name="green",
            mean=0.0,
            std=1.0,
            min=0.0,
            max=1.0,
            wavelength_um=0.56,
        ),
        BandSpec(
            sensor="s2",
            name="blue",
            source_name="blue",
            mean=0.0,
            std=1.0,
            min=0.0,
            max=1.0,
            wavelength_um=0.49,
        ),
    ]

    def select_band_specs(self, bands):  # noqa: ANN001
        del bands
        return list(self.bands)


def _make_cfg(
    tmp_path: Path,
    *,
    hook_paths: list[str] | None = None,
    corruptions: list[str] | None = None,
    severities: list[int] | None = None,
) -> object:
    if hook_paths is None:
        hook_paths = [
            "backbone.layer1",
            "backbone.layer2",
            "backbone.layer3",
            "backbone.layer4",
        ]
    if corruptions is None:
        corruptions = []
    if severities is None:
        severities = [1]

    return OmegaConf.create(
        {
            "seed": 42,
            "device": "cpu",
            "verbose": False,
            "resume": True,
            "dataset": {
                "names": ["m-eurosat"],
                "partition": "default",
                "batch_size": 2,
                "num_workers": 0,
                "normalization": "bandspec_zscore",
                "bands": "rgb",
                "image_size": 224,
                "interpolation": "bilinear",
            },
            "model": {"_target_": "m.t", "name": "resnet50"},
            "cka": {
                "output": str(tmp_path / "cka_results.csv"),
                "prior_results": str(tmp_path / "all_results.csv"),
                "traces_root": str(tmp_path / "cka_traces"),
                "write_sample_traces": True,
                "corruptions": corruptions,
                "corruption_severities": severities,
                "cloud_pattern_mode": "fixed_across_severity",
                "cal_size": 4,
                "confidence_threshold": 0.9,
                "layers": {"resnet50": hook_paths},
            },
        }
    )


def _write_prior_results(tmp_path: Path) -> None:
    pd.DataFrame(
        [
            {
                "method": "linear",
                "best_c": 0.01,
                "dataset": "m-eurosat",
                "name": "resnet50",
                "partition": "default",
                "bands": "rgb",
                "normalization": "bandspec_zscore",
                "image_size": 224,
                "interpolation": "bilinear",
            }
        ]
    ).to_csv(tmp_path / "all_results.csv", index=False)


def _patch_common(monkeypatch, clean_acts, corr_acts_seq):  # noqa: ANN001
    monkeypatch.setattr("torchgeo_bench.cka.pipeline.get_bench_dataset_class", lambda _: _DummyBench)
    monkeypatch.setattr(
        "torchgeo_bench.cka.pipeline.get_datasets",
        lambda **_: (object(), object(), object(), object()),
    )
    monkeypatch.setattr("torchgeo_bench.cka.pipeline.instantiate", lambda *_, **__: _DummyModel())

    state = {"calls": 0}
    corr_acts_list = list(corr_acts_seq)

    def _fake_collect(self):  # noqa: ANN001
        del self
        idx = state["calls"]
        state["calls"] += 1
        if idx == 0:
            return clean_acts
        return corr_acts_list[idx - 1]

    monkeypatch.setattr("torchgeo_bench.cka.pipeline.HookCollector.collect", _fake_collect)

    call_state = {"n_corrupted": 0}

    def _fake_extract(model, dataloader, device, transforms=None, verbose=False):  # noqa: ANN001
        del model, dataloader, device, verbose
        if transforms is None:
            if state.get("feature_phase", 0) < 2:
                state["feature_phase"] = state.get("feature_phase", 0) + 1
                X = np.arange(80, dtype=np.float32).reshape(10, 8)
                y = np.array([0, 1] * 5, dtype=np.int64)
                return X, y
            X = np.arange(80, dtype=np.float32).reshape(10, 8)
            y = np.array([0, 1] * 5, dtype=np.int64)
            return X, y
        call_state["n_corrupted"] += 1
        X = np.arange(80, dtype=np.float32).reshape(10, 8) + float(call_state["n_corrupted"])
        y = np.array([0, 1] * 5, dtype=np.int64)
        return X, y

    monkeypatch.setattr("torchgeo_bench.cka.pipeline.extract_features", _fake_extract)
    return call_state


def test_invalid_hook_path_fails_fast(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path, hook_paths=["backbone.bad_layer"], corruptions=[])
    _write_prior_results(tmp_path)
    monkeypatch.setattr("torchgeo_bench.cka.pipeline.get_bench_dataset_class", lambda _: _DummyBench)
    monkeypatch.setattr(
        "torchgeo_bench.cka.pipeline.get_datasets",
        lambda **_: (object(), object(), object(), object()),
    )
    monkeypatch.setattr("torchgeo_bench.cka.pipeline.instantiate", lambda *_, **__: _DummyModel())
    monkeypatch.setattr(
        "torchgeo_bench.cka.pipeline.extract_features",
        lambda *_, **__: (np.arange(80, dtype=np.float32).reshape(10, 8), np.array([0, 1] * 5, dtype=np.int64)),
    )
    with pytest.raises((AttributeError, ValueError)) as exc_info:
        run_cka(cfg)
    assert "backbone.bad_layer" in str(exc_info.value)
    assert not Path(cfg.cka.output).exists()


def test_run_cka_clean_pass_writes_rows(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path, corruptions=[])
    _write_prior_results(tmp_path)
    clean_acts = {
        "backbone.layer1": np.arange(80, dtype=np.float32).reshape(10, 8),
        "backbone.layer2": np.arange(80, dtype=np.float32).reshape(10, 8) + 1.0,
        "backbone.layer3": np.arange(80, dtype=np.float32).reshape(10, 8) + 2.0,
        "backbone.layer4": np.arange(80, dtype=np.float32).reshape(10, 8) + 3.0,
    }
    _patch_common(monkeypatch, clean_acts, [])
    run_cka(cfg)
    df = pd.read_csv(cfg.cka.output)
    assert len(df) == 4
    assert set(df["corruption_type"]) == {"clean"}
    assert set(df["severity"]) == {0}
    assert np.isfinite(df["cka"].to_numpy(dtype=np.float64)).all()
    assert np.isfinite(df["cosine_drift"].to_numpy(dtype=np.float64)).all()
    assert np.allclose(
        df["participation_ratio"].to_numpy(dtype=np.float64),
        df["clean_participation_ratio"].to_numpy(dtype=np.float64),
    )


def test_run_cka_empty_collect_hard_fails(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path, corruptions=[])
    _write_prior_results(tmp_path)
    clean_acts = {
        "backbone.layer1": np.empty((0, 8), dtype=np.float32),
        "backbone.layer2": np.arange(80, dtype=np.float32).reshape(10, 8),
        "backbone.layer3": np.arange(80, dtype=np.float32).reshape(10, 8),
        "backbone.layer4": np.arange(80, dtype=np.float32).reshape(10, 8),
    }
    _patch_common(monkeypatch, clean_acts, [])
    with pytest.raises(ValueError):
        run_cka(cfg)
    assert not Path(cfg.cka.output).exists()


def test_run_cka_corrupted_loop_writes_rows(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path, corruptions=["poisson_gaussian"], severities=[1, 2])
    _write_prior_results(tmp_path)
    clean = np.arange(80, dtype=np.float32).reshape(10, 8)
    clean_acts = {
        "backbone.layer1": clean + 0.0,
        "backbone.layer2": clean + 1.0,
        "backbone.layer3": clean + 2.0,
        "backbone.layer4": clean + 3.0,
    }
    corr_acts_seq = [
        {
            "backbone.layer1": clean + 0.2,
            "backbone.layer2": clean + 1.2,
            "backbone.layer3": clean + 2.2,
            "backbone.layer4": clean + 3.2,
        },
        {
            "backbone.layer1": clean + 0.4,
            "backbone.layer2": clean + 1.4,
            "backbone.layer3": clean + 2.4,
            "backbone.layer4": clean + 3.4,
        },
    ]
    _patch_common(monkeypatch, clean_acts, corr_acts_seq)
    run_cka(cfg)
    df = pd.read_csv(cfg.cka.output)
    assert len(df) == 12
    assert np.isfinite(df["cka"].to_numpy(dtype=np.float64)).all()
    assert np.isfinite(df["cosine_drift"].to_numpy(dtype=np.float64)).all()
    for _, g in df.groupby(["corruption_type", "severity"]):
        assert sorted(g["layer_index"].tolist()) == [0, 1, 2, 3]


def test_run_cka_resume_skips_complete_key(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path, corruptions=["poisson_gaussian"], severities=[1, 2])
    _write_prior_results(tmp_path)

    seeded = pd.DataFrame([_base_row(i, severity=1) for i in range(4)])
    seeded["cka"] = 0.1
    seeded["cosine_drift"] = 0.1
    seeded["participation_ratio"] = 1.0
    seeded["clean_participation_ratio"] = 1.0
    seeded["n_test"] = 10
    seeded["feature_dim"] = 8
    seeded["best_c"] = 0.01
    seeded.to_csv(cfg.cka.output, index=False)

    clean = np.arange(80, dtype=np.float32).reshape(10, 8)
    clean_acts = {
        "backbone.layer1": clean + 0.0,
        "backbone.layer2": clean + 1.0,
        "backbone.layer3": clean + 2.0,
        "backbone.layer4": clean + 3.0,
    }
    corr_acts_seq = [
        {
            "backbone.layer1": clean + 0.3,
            "backbone.layer2": clean + 1.3,
            "backbone.layer3": clean + 2.3,
            "backbone.layer4": clean + 3.3,
        }
    ]
    call_state = _patch_common(monkeypatch, clean_acts, corr_acts_seq)
    run_cka(cfg)
    assert call_state["n_corrupted"] == 1


def test_run_cka_resume_reruns_partial_key(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path, corruptions=["poisson_gaussian"], severities=[1])
    _write_prior_results(tmp_path)

    partial = pd.DataFrame([_base_row(i, severity=1) for i in range(2)])
    partial["cka"] = 0.1
    partial["cosine_drift"] = 0.1
    partial["participation_ratio"] = 1.0
    partial["clean_participation_ratio"] = 1.0
    partial["n_test"] = 10
    partial["feature_dim"] = 8
    partial["best_c"] = 0.01
    partial.to_csv(cfg.cka.output, index=False)

    clean = np.arange(80, dtype=np.float32).reshape(10, 8)
    clean_acts = {
        "backbone.layer1": clean + 0.0,
        "backbone.layer2": clean + 1.0,
        "backbone.layer3": clean + 2.0,
        "backbone.layer4": clean + 3.0,
    }
    corr_acts_seq = [
        {
            "backbone.layer1": clean + 0.5,
            "backbone.layer2": clean + 1.5,
            "backbone.layer3": clean + 2.5,
            "backbone.layer4": clean + 3.5,
        }
    ]
    _patch_common(monkeypatch, clean_acts, corr_acts_seq)
    run_cka(cfg)

    df = pd.read_csv(cfg.cka.output)
    key_rows = df[
        (df["corruption_type"] == "poisson_gaussian")
        & (df["severity"] == 1)
        & (df["dataset"] == "m-eurosat")
    ]
    assert len(key_rows) == 4


def test_poisson_skip_list_honored(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path, corruptions=["poisson_gaussian"], severities=[1])
    _write_prior_results(tmp_path)
    clean = np.arange(80, dtype=np.float32).reshape(10, 8)
    clean_acts = {
        "backbone.layer1": clean + 0.0,
        "backbone.layer2": clean + 1.0,
        "backbone.layer3": clean + 2.0,
        "backbone.layer4": clean + 3.0,
    }
    call_state = _patch_common(monkeypatch, clean_acts, [])
    monkeypatch.setattr("torchgeo_bench.cka.pipeline.SKIP_POISSON_GAUSSIAN", frozenset({"m-eurosat"}))
    run_cka(cfg)
    assert call_state["n_corrupted"] == 0
    df = pd.read_csv(cfg.cka.output)
    assert set(df["corruption_type"]) == {"clean"}


def test_track_b_columns_nan_on_non_final_layers(tmp_path, monkeypatch):
    cfg = _make_cfg(
        tmp_path,
        hook_paths=["backbone.layer1", "backbone.layer2"],
        corruptions=["poisson_gaussian"],
        severities=[1],
    )
    _write_prior_results(tmp_path)
    clean = np.arange(80, dtype=np.float32).reshape(10, 8)
    clean_acts = {"backbone.layer1": clean + 0.0, "backbone.layer2": clean + 1.0}
    corr_acts_seq = [{"backbone.layer1": clean + 0.2, "backbone.layer2": clean + 1.2}]
    _patch_common(monkeypatch, clean_acts, corr_acts_seq)
    monkeypatch.setattr(
        "torchgeo_bench.cka.pipeline.track_b_summary",
        lambda *_, **__: {
            "spearman_drift_confidence": 0.1,
            "spearman_drift_correctness": -0.2,
            "frac_overconfident_high_drift": 0.3,
        },
    )
    run_cka(cfg)
    df = pd.read_csv(cfg.cka.output)
    non_final = df[df["layer_index"] == 0]
    assert non_final["spearman_drift_confidence"].isna().all()
    assert non_final["spearman_drift_correctness"].isna().all()
    assert non_final["frac_overconfident_high_drift"].isna().all()


def test_track_b_columns_finite_on_final_layer(tmp_path, monkeypatch):
    cfg = _make_cfg(
        tmp_path,
        hook_paths=["backbone.layer1", "backbone.layer2"],
        corruptions=["poisson_gaussian"],
        severities=[1],
    )
    _write_prior_results(tmp_path)
    clean = np.arange(80, dtype=np.float32).reshape(10, 8)
    clean_acts = {"backbone.layer1": clean + 0.0, "backbone.layer2": clean + 1.0}
    corr_acts_seq = [{"backbone.layer1": clean + 0.2, "backbone.layer2": clean + 1.2}]
    _patch_common(monkeypatch, clean_acts, corr_acts_seq)
    monkeypatch.setattr(
        "torchgeo_bench.cka.pipeline.track_b_summary",
        lambda *_, **__: {
            "spearman_drift_confidence": 0.1,
            "spearman_drift_correctness": -0.2,
            "frac_overconfident_high_drift": 0.3,
        },
    )
    run_cka(cfg)
    df = pd.read_csv(cfg.cka.output)
    final_row = df[(df["corruption_type"] == "poisson_gaussian") & (df["layer_index"] == 1)].iloc[0]
    assert np.isfinite(float(final_row["spearman_drift_confidence"]))
    assert np.isfinite(float(final_row["spearman_drift_correctness"]))
    assert np.isfinite(float(final_row["frac_overconfident_high_drift"]))


def test_parquet_written_with_correct_schema(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path, corruptions=["poisson_gaussian"], severities=[1])
    _write_prior_results(tmp_path)
    clean = np.arange(80, dtype=np.float32).reshape(10, 8)
    clean_acts = {
        "backbone.layer1": clean + 0.0,
        "backbone.layer2": clean + 1.0,
        "backbone.layer3": clean + 2.0,
        "backbone.layer4": clean + 3.0,
    }
    corr_acts_seq = [
        {
            "backbone.layer1": clean + 0.2,
            "backbone.layer2": clean + 1.2,
            "backbone.layer3": clean + 2.2,
            "backbone.layer4": clean + 3.2,
        }
    ]
    _patch_common(monkeypatch, clean_acts, corr_acts_seq)
    run_cka(cfg)

    parquet_path = Path(cfg.cka.traces_root) / "resnet50" / "m-eurosat.parquet"
    assert parquet_path.exists()
    frame = pd.read_parquet(parquet_path)
    assert list(frame.columns) == [
        "corruption_type",
        "severity",
        "sample_idx",
        "drift",
        "confidence",
        "correct",
        "y_true",
        "y_pred",
    ]
    assert frame["drift"].dtype == np.float32
    assert frame["confidence"].dtype == np.float32
    assert frame["correct"].dtype == bool
    assert frame["y_true"].dtype == np.int16
    assert frame["y_pred"].dtype == np.int16


def test_parquet_appends_across_conditions(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path, corruptions=["cloud", "poisson_gaussian"], severities=[1])
    _write_prior_results(tmp_path)
    clean = np.arange(80, dtype=np.float32).reshape(10, 8)
    clean_acts = {
        "backbone.layer1": clean + 0.0,
        "backbone.layer2": clean + 1.0,
        "backbone.layer3": clean + 2.0,
        "backbone.layer4": clean + 3.0,
    }
    corr_acts_seq = [
        {
            "backbone.layer1": clean + 0.2,
            "backbone.layer2": clean + 1.2,
            "backbone.layer3": clean + 2.2,
            "backbone.layer4": clean + 3.2,
        },
        {
            "backbone.layer1": clean + 0.3,
            "backbone.layer2": clean + 1.3,
            "backbone.layer3": clean + 2.3,
            "backbone.layer4": clean + 3.3,
        },
    ]
    _patch_common(monkeypatch, clean_acts, corr_acts_seq)
    run_cka(cfg)

    parquet_path = Path(cfg.cka.traces_root) / "resnet50" / "m-eurosat.parquet"
    frame = pd.read_parquet(parquet_path)
    assert set(frame["corruption_type"].unique()) == {"cloud", "poisson_gaussian"}


def test_clean_condition_track_b_clean_policy(tmp_path, monkeypatch):
    cfg = _make_cfg(tmp_path, corruptions=[])
    _write_prior_results(tmp_path)
    clean = np.arange(80, dtype=np.float32).reshape(10, 8)
    clean_acts = {
        "backbone.layer1": clean + 0.0,
        "backbone.layer2": clean + 1.0,
        "backbone.layer3": clean + 2.0,
        "backbone.layer4": clean + 3.0,
    }
    _patch_common(monkeypatch, clean_acts, [])
    run_cka(cfg)
    df = pd.read_csv(cfg.cka.output)
    final_clean = df[(df["corruption_type"] == "clean") & (df["layer_index"] == 3)].iloc[0]
    assert np.isnan(float(final_clean["spearman_drift_confidence"]))
    assert np.isnan(float(final_clean["spearman_drift_correctness"]))
    assert np.isclose(float(final_clean["frac_overconfident_high_drift"]), 0.0)
