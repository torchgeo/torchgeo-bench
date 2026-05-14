import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from torchgeo_bench.uq.pipeline import (
    _build_resume_set,
    _expected_metrics,
    _is_uq_classification_dataset,
    _lookup_best_c,
    _normalize_cloud_pattern_mode,
    _run_uq_block,
)
from torchgeo_bench.uq.pipeline import (
    main as uq_main,
)


def _base_row(metric_name: str) -> dict[str, object]:
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
        "uq_method": "uncalibrated",
        "corruption_type": "clean",
        "severity": 0,
        "metric_name": metric_name,
        "metric_value": 0.1,
    }


def test_build_resume_set_empty_csv(tmp_path):
    csv_path = tmp_path / "uq_results.csv"
    assert _build_resume_set(str(csv_path)) == set()


def test_build_resume_set_complete_key(tmp_path):
    csv_path = tmp_path / "uq_results.csv"
    rows = [_base_row(metric) for metric in sorted(_expected_metrics("uncalibrated"))]
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    done = _build_resume_set(str(csv_path))
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
        "uncalibrated",
        "clean",
        "0",
    )
    assert key in done


def test_build_resume_set_partial_key(tmp_path):
    csv_path = tmp_path / "uq_results.csv"
    metrics = sorted(_expected_metrics("uncalibrated"))
    rows = [_base_row(metric) for metric in metrics[:-1]]
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    done = _build_resume_set(str(csv_path))
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
        "uncalibrated",
        "clean",
        "0",
    )
    assert key not in done


def test_expected_metrics_for_method():
    assert _expected_metrics("uncalibrated") == {
        "accuracy",
        "ece",
        "nll",
        "brier",
        "predictive_entropy",
        "normalized_predictive_entropy",
        "max_probability",
        "raw_aurc",
        "eaurc",
        "selective_acc_90",
    }
    assert _expected_metrics("conformal") == {
        "accuracy",
        "empirical_coverage",
        "mean_set_size",
    }


def test_is_uq_classification_dataset():
    class _SingleLabelCls:
        task = "classification"
        multilabel = False

    class _MultiLabelCls:
        task = "classification"
        multilabel = True

    class _Segmentation:
        task = "segmentation"
        multilabel = False

    assert _is_uq_classification_dataset(_SingleLabelCls)
    assert not _is_uq_classification_dataset(_MultiLabelCls)
    assert not _is_uq_classification_dataset(_Segmentation)


def test_lookup_best_c_returns_direct_match():
    prior = pd.DataFrame(
        [
            {
                "model": "m.t",
                "name": "resnet50",
                "dataset": "sen12ms",
                "partition": "default",
                "bands": "rgb",
                "method": "linear",
                "best_c": 0.25,
            }
        ]
    )
    got = _lookup_best_c(
        prior,
        {
            "model": "m.t",
            "name": "resnet50",
            "dataset": "sen12ms",
            "partition": "default",
            "bands": "rgb",
        },
    )
    assert got == 0.25


def test_lookup_best_c_no_match_no_alias():
    prior = pd.DataFrame(
        [
            {
                "model": "m.t",
                "name": "resnet50",
                "dataset": "sen12ms",
                "partition": "default",
                "bands": "rgb",
                "method": "linear",
                "best_c": 0.25,
            }
        ]
    )
    got = _lookup_best_c(
        prior,
        {
            "model": "m.t",
            "name": "resnet50",
            "dataset": "sen12ms_cr_c1",
            "partition": "default",
            "bands": "rgb",
        },
    )
    assert got is None


def test_lookup_best_c_falls_back_to_alias():
    prior = pd.DataFrame(
        [
            {
                "model": "m.t",
                "name": "resnet50",
                "dataset": "sen12ms",
                "partition": "default",
                "bands": "rgb",
                "method": "linear",
                "best_c": 0.75,
            }
        ]
    )
    got = _lookup_best_c(
        prior,
        {
            "model": "m.t",
            "name": "resnet50",
            "dataset": "sen12ms_cr_c3",
            "partition": "default",
            "bands": "rgb",
        },
        alias_dataset="sen12ms",
    )
    assert got == 0.75


def test_lookup_best_c_direct_takes_precedence():
    prior = pd.DataFrame(
        [
            {
                "model": "m.t",
                "name": "resnet50",
                "dataset": "sen12ms",
                "partition": "default",
                "bands": "rgb",
                "method": "linear",
                "best_c": 0.75,
            },
            {
                "model": "m.t",
                "name": "resnet50",
                "dataset": "sen12ms_cr_c3",
                "partition": "default",
                "bands": "rgb",
                "method": "linear",
                "best_c": 1.5,
            },
        ]
    )
    got = _lookup_best_c(
        prior,
        {
            "model": "m.t",
            "name": "resnet50",
            "dataset": "sen12ms_cr_c3",
            "partition": "default",
            "bands": "rgb",
        },
        alias_dataset="sen12ms",
    )
    assert got == 1.5


def test_lookup_best_c_alias_logs_info(caplog):
    prior = pd.DataFrame(
        [
            {
                "model": "m.t",
                "name": "resnet50",
                "dataset": "sen12ms",
                "partition": "default",
                "bands": "rgb",
                "method": "linear",
                "best_c": 0.5,
            }
        ]
    )
    with caplog.at_level("INFO"):
        got = _lookup_best_c(
            prior,
            {
                "model": "m.t",
                "name": "resnet50",
                "dataset": "sen12ms_cr_c4",
                "partition": "default",
                "bands": "rgb",
            },
            alias_dataset="sen12ms",
        )
    assert got == 0.5
    assert "using alias" in caplog.text.lower()


def test_lookup_best_c_falls_back_to_alias_sweep_c_format():
    prior = pd.DataFrame(
        [
            {"dataset": "sen12ms", "model": "resnet50", "C": 0.9, "val_acc": 0.80},
            {"dataset": "sen12ms", "model": "resnet50", "C": 0.7, "val_acc": 0.85},
        ]
    )
    got = _lookup_best_c(
        prior,
        {"dataset": "sen12ms_cr_c2", "name": "resnet50"},
        alias_dataset="sen12ms",
    )
    assert got == 0.7


def test_pipeline_passes_alias_to_lookup(monkeypatch, tmp_path):
    csv_path = tmp_path / "all_results.csv"
    pd.DataFrame([{"method": "linear", "best_c": 0.3}]).to_csv(csv_path, index=False)

    class _DummyBench:
        task = "classification"
        multilabel = False
        prior_results_alias = "sen12ms"
        rgb_bands = ["red", "green", "blue"]

        def select_band_specs(self, bands):  # noqa: ANN001
            return []

    class _DummyModel:
        def to(self, _device):  # noqa: ANN001
            return self

        def eval(self):
            return self

    seen_alias: list[str | None] = []

    def _fake_lookup(prior_results, row_filter, *, alias_dataset=None):  # noqa: ARG001
        seen_alias.append(alias_dataset)
        return None

    monkeypatch.setattr("torchgeo_bench.uq.pipeline.get_bench_dataset_class", lambda _name: _DummyBench)
    monkeypatch.setattr(
        "torchgeo_bench.uq.pipeline.get_datasets",
        lambda **kwargs: (object(), object(), object(), object()),
    )
    monkeypatch.setattr("torchgeo_bench.uq.pipeline.instantiate", lambda *args, **kwargs: _DummyModel())
    monkeypatch.setattr(
        "torchgeo_bench.uq.pipeline.extract_features",
        lambda *args, **kwargs: (
            np.zeros((8, 4), dtype=np.float32),
            np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64),
        ),
    )
    monkeypatch.setattr("torchgeo_bench.uq.pipeline._lookup_best_c", _fake_lookup)

    cfg = OmegaConf.create(
        {
            "seed": 1,
            "resume": False,
            "device": "cpu",
            "verbose": False,
            "model": {"_target_": "dummy.Target", "name": "resnet50"},
            "dataset": {
                "names": ["sen12ms_cr_c1"],
                "partition": "default",
                "batch_size": 2,
                "num_workers": 0,
                "bands": "rgb",
                "interpolation": "bilinear",
                "normalization": "bandspec_zscore",
            },
            "uq": {
                "output": str(tmp_path / "uq_results.csv"),
                "prior_results": str(csv_path),
                "methods": ["uncalibrated"],
                "corruptions": ["clean"],
                "corruption_severities": [1],
                "cal_size": 2,
                "ece_bins": 10,
                "ece_binning": "equal_width",
                "conformal_alpha": 0.1,
                "n_ensemble": 2,
                "laplace_batch_size": 16,
                "cloud_pattern_mode": "fixed_across_severity",
            },
        }
    )
    uq_main.__wrapped__(cfg)  # type: ignore[attr-defined]
    assert seen_alias == ["sen12ms"]


def test_run_uq_block_writes_csv(tmp_path, monkeypatch):
    csv_path = tmp_path / "uq_results.csv"
    X_test = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    y_test = np.array([0, 1, 2, 1], dtype=np.int64)

    def _fake_extract(*args, **kwargs):  # noqa: ARG001
        return X_test, y_test

    monkeypatch.setattr("torchgeo_bench.uq.pipeline.extract_features", _fake_extract)

    class _DummyUncalibrated:
        def predict_proba(self, X: np.ndarray) -> np.ndarray:
            logits = X.copy()
            logits = logits - logits.max(axis=1, keepdims=True)
            exps = np.exp(logits)
            return exps / exps.sum(axis=1, keepdims=True)

    common_meta = {
        "model": "m.t",
        "name": "resnet50",
        "backbone": "resnet50",
        "dataset": "m-eurosat",
        "normalization": "bandspec_zscore",
        "image_size": 224,
        "interpolation": "bilinear",
        "partition": "default",
        "bands": "rgb",
        "seed": 42,
    }

    rows = _run_uq_block(
        method_name="uncalibrated",
        method=_DummyUncalibrated(),
        output_path=str(csv_path),
        common_meta=common_meta,
        corruption_type="clean",
        severity=0,
        ece_bins=15,
        ece_binning="equal_width",
        conformal_alpha=0.1,
        n_cal=40,
        n_train=160,
        feature_dim=3,
        best_c=1.0,
        seed=42,
        model=object(),  # type: ignore[arg-type]
        test_loader=object(),  # type: ignore[arg-type]
        verbose=False,
    )

    df = pd.read_csv(csv_path)
    assert len(rows) == 10
    assert set(df["metric_name"]) == _expected_metrics("uncalibrated")
    assert np.isfinite(df["metric_value"].to_numpy(dtype=np.float64)).all()


def test_run_uq_block_conformal_writes_reduced_metrics(tmp_path):
    csv_path = tmp_path / "uq_results.csv"
    X_test = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    y_test = np.array([0, 1, 2, 1], dtype=np.int64)

    class _DummyConformal:
        def predict_sets(self, X: np.ndarray, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
            assert alpha == 0.1
            point_preds = np.array([0, 0, 2, 1], dtype=np.int64)
            pred_sets = np.array(
                [
                    [True, False, False],
                    [True, True, False],
                    [False, False, True],
                    [False, True, False],
                ],
                dtype=bool,
            )
            assert X.shape == X_test.shape
            return point_preds, pred_sets

    common_meta = {
        "model": "m.t",
        "name": "resnet50",
        "backbone": "resnet50",
        "dataset": "m-eurosat",
        "normalization": "bandspec_zscore",
        "image_size": 224,
        "interpolation": "bilinear",
        "partition": "default",
        "bands": "rgb",
        "seed": 42,
    }

    rows = _run_uq_block(
        method_name="conformal",
        method=_DummyConformal(),
        output_path=str(csv_path),
        common_meta=common_meta,
        corruption_type="clean",
        severity=0,
        ece_bins=15,
        ece_binning="equal_width",
        conformal_alpha=0.1,
        n_cal=40,
        n_train=160,
        feature_dim=3,
        best_c=1.0,
        seed=42,
        X_test=X_test,
        y_test=y_test,
    )

    df = pd.read_csv(csv_path)
    assert len(rows) == 3
    assert set(df["metric_name"]) == _expected_metrics("conformal")


def test_normalize_cloud_pattern_mode():
    assert _normalize_cloud_pattern_mode("fixed_across_severity") == "fixed"
    assert _normalize_cloud_pattern_mode("independent_per_severity") == "independent"
    assert _normalize_cloud_pattern_mode("fixed") == "fixed"
    assert _normalize_cloud_pattern_mode("independent") == "independent"


def test_normalize_cloud_pattern_mode_invalid():
    with np.testing.assert_raises(ValueError):
        _normalize_cloud_pattern_mode("invalid")
