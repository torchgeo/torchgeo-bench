import numpy as np
import pandas as pd

from torchgeo_bench.uq.pipeline import (
    _build_resume_set,
    _expected_metrics,
    _is_uq_classification_dataset,
    _normalize_cloud_pattern_mode,
    _run_uq_block,
)


def _base_row(metric_name: str) -> dict[str, object]:
    return {
        "model": "m.t",
        "name": "resnet50",
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
        "ece",
        "nll",
        "brier",
        "predictive_entropy",
        "sharpness",
        "raw_aurc",
        "eaurc",
        "selective_acc_90",
    }
    assert _expected_metrics("conformal") == {
        "empirical_coverage",
        "mean_set_size",
        "raw_aurc",
        "eaurc",
        "selective_acc_90",
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
    assert len(rows) == 8
    assert set(df["metric_name"]) == _expected_metrics("uncalibrated")
    assert np.isfinite(df["metric_value"].to_numpy(dtype=np.float64)).all()


def test_normalize_cloud_pattern_mode():
    assert _normalize_cloud_pattern_mode("fixed_across_severity") == "fixed"
    assert _normalize_cloud_pattern_mode("independent_per_severity") == "independent"
    assert _normalize_cloud_pattern_mode("fixed") == "fixed"
    assert _normalize_cloud_pattern_mode("independent") == "independent"


def test_normalize_cloud_pattern_mode_invalid():
    with np.testing.assert_raises(ValueError):
        _normalize_cloud_pattern_mode("invalid")
