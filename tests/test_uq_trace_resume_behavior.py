import numpy as np
import pandas as pd

from torchgeo_bench.uq.pipeline import _run_uq_block


class _DummyUncalibrated:
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits = X.copy()
        logits = logits - logits.max(axis=1, keepdims=True)
        exps = np.exp(logits)
        return exps / exps.sum(axis=1, keepdims=True)


def _common_meta() -> dict[str, object]:
    return {
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


def _trace_ctx(tmp_path):
    return {
        "run_id": "run-1",
        "trace_root": str(tmp_path / "uq_traces"),
        "manifest_path": str(tmp_path / "uq_traces" / "run_id=run-1" / "manifest.csv"),
        "trace_format": "csv",
        "schema_version": "v1",
        "config_hash": "abc123",
        "git_sha": "deadbeef",
        "compression": "zstd",
        "overwrite": False,
        "include_conformal": False,
    }


def test_run_uq_block_does_not_duplicate_complete_trace_manifest(tmp_path):
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

    kwargs = {
        "method_name": "uncalibrated",
        "method": _DummyUncalibrated(),
        "output_path": str(csv_path),
        "common_meta": _common_meta(),
        "corruption_type": "clean",
        "severity": 0,
        "ece_bins": 15,
        "conformal_alpha": 0.1,
        "n_cal": 40,
        "n_train": 160,
        "feature_dim": 3,
        "best_c": 1.0,
        "seed": 42,
        "X_test": X_test,
        "y_test": y_test,
        "trace_ctx": _trace_ctx(tmp_path),
    }

    _run_uq_block(**kwargs)
    _run_uq_block(**kwargs)

    manifest_path = tmp_path / "uq_traces" / "run_id=run-1" / "manifest.csv"
    manifest_df = pd.read_csv(manifest_path)
    assert len(manifest_df) == 1

    trace_path = manifest_df["trace_path"].iloc[0]
    trace_df = pd.read_csv(trace_path)
    assert len(trace_df) == len(y_test)
