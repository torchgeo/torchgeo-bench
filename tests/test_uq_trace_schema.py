import numpy as np

from torchgeo_bench.uq.traces import TRACE_REQUIRED_COLUMNS, build_probabilistic_trace_frame


def _common_meta() -> dict[str, object]:
    return {
        "model": "torchgeo_bench.models.TimmBench",
        "name": "resnet50",
        "dataset": "m-eurosat",
        "partition": "default",
        "bands": "rgb",
        "normalization": "bandspec_zscore",
        "image_size": 224,
        "interpolation": "bilinear",
        "seed": 42,
    }


def test_build_probabilistic_trace_frame_schema_and_values():
    y_true = np.array([0, 1, 2], dtype=np.int64)
    probs = np.array(
        [
            [0.9, 0.1, 0.0],
            [0.2, 0.3, 0.5],
            [0.1, 0.8, 0.1],
        ],
        dtype=np.float64,
    )

    trace_df = build_probabilistic_trace_frame(
        run_id="run-123",
        common_meta=_common_meta(),
        uq_method="uncalibrated",
        corruption_type="clean",
        severity=0,
        y_true=y_true,
        probs=probs,
    )

    assert list(trace_df.columns) == list(TRACE_REQUIRED_COLUMNS)
    assert len(trace_df) == 3
    assert trace_df["run_id"].nunique() == 1
    assert trace_df["run_id"].iloc[0] == "run-123"
    assert trace_df["sample_idx"].tolist() == [0, 1, 2]
    assert trace_df["y_true"].tolist() == [0, 1, 2]
    assert trace_df["y_pred"].tolist() == [0, 2, 1]
    assert np.allclose(trace_df["confidence"].to_numpy(dtype=float), np.array([0.9, 0.5, 0.8]))
    assert trace_df["correct"].tolist() == [1, 0, 0]
    assert trace_df["is_error"].tolist() == [0, 1, 1]
