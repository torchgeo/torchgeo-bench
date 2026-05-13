import numpy as np
import pandas as pd

from torchgeo_bench.uq.pipeline import _run_uq_block
from torchgeo_bench.uq.traces import read_trace_row_count, scan_traces


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
        "backbone": "resnet50",
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
        "trace_dataset_root": str(tmp_path / "uq_traces"),
        "config_hash": "abc123",
        "git_sha": "deadbeef",
        "created_at_utc": "2026-05-13T00:00:00Z",
        "compression": "zstd",
        "overwrite": False,
        "include_conformal": False,
    }


def test_run_uq_block_does_not_duplicate_complete_trace_fragment(tmp_path):
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
        "sample_ids": np.array(["s0", "s1", "s2", "s3"], dtype=object),
        "trace_ctx": _trace_ctx(tmp_path),
    }

    _run_uq_block(**kwargs)
    _run_uq_block(**kwargs)

    results_df = pd.read_csv(csv_path)
    block_keys = results_df["trace_block_key"].dropna().astype(str).unique().tolist()
    assert len(block_keys) == 1

    trace_df = scan_traces(tmp_path / "uq_traces", block_keys=block_keys).sort_values("sample_idx")
    assert len(trace_df) == len(y_test)
    assert trace_df["sample_id"].tolist() == ["s0", "s1", "s2", "s3"]

    trace_path = next((tmp_path / "uq_traces").rglob("*.parquet"))
    assert read_trace_row_count(trace_path) == len(y_test)
