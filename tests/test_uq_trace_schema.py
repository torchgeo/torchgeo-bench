import numpy as np

from torchgeo_bench.uq.traces import (
    DISTANCE_TRACE_COLUMNS,
    TRACE_REQUIRED_COLUMNS,
    build_distance_trace_frame,
    build_probabilistic_trace_frame,
    resolve_trace_partition_path,
    scan_traces,
    write_trace_block_atomic,
)


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
        trace_block_key="block-123",
        run_id="run-123",
        common_meta=_common_meta(),
        uq_method="uncalibrated",
        corruption_type="clean",
        severity=0,
        config_hash="cfg-123",
        git_sha="deadbeef",
        created_at_utc="2026-05-13T00:00:00Z",
        y_true=y_true,
        probs=probs,
        sample_ids=np.array(["s0", "s1", "s2"], dtype=object),
    )

    assert list(trace_df.columns) == list(TRACE_REQUIRED_COLUMNS)
    assert len(trace_df) == 3
    assert trace_df["trace_block_key"].iloc[0] == "block-123"
    assert trace_df["run_id"].nunique() == 1
    assert trace_df["run_id"].iloc[0] == "run-123"
    assert trace_df["sample_id"].tolist() == ["s0", "s1", "s2"]
    assert trace_df["sample_idx"].tolist() == [0, 1, 2]
    assert trace_df["y_true"].tolist() == [0, 1, 2]
    assert trace_df["y_pred"].tolist() == [0, 2, 1]
    assert np.allclose(trace_df["confidence"].to_numpy(dtype=float), np.array([0.9, 0.5, 0.8]))
    assert trace_df["correct"].tolist() == [1, 0, 0]
    assert trace_df["is_error"].tolist() == [0, 1, 1]


def _probs() -> np.ndarray:
    return np.array(
        [
            [0.9, 0.1, 0.0],
            [0.2, 0.3, 0.5],
            [0.1, 0.8, 0.1],
        ],
        dtype=np.float64,
    )


def _distance_frame(**overrides):
    kwargs: dict[str, object] = {
        "trace_block_key": "block-dist",
        "run_id": "run-123",
        "common_meta": _common_meta(),
        "uq_method": "maha@backbone.layer4",
        "corruption_type": "poisson_gaussian",
        "severity": 3,
        "config_hash": "cfg-123",
        "git_sha": "deadbeef",
        "created_at_utc": "2026-08-11T00:00:00Z",
        "y_true": np.array([0, 1, 2], dtype=np.int64),
        "probs": _probs(),
        "distance_score": np.array([1.5, 9.0, 4.25], dtype=np.float64),
        "layer_name": "backbone.layer4",
        "score_kind": "maha",
        "sample_ids": np.array(["s0", "s1", "s2"], dtype=object),
    }
    kwargs.update(overrides)
    return build_distance_trace_frame(**kwargs)  # type: ignore[arg-type]


def test_build_distance_trace_frame_schema_and_values():
    trace_df = _distance_frame()

    assert list(trace_df.columns) == list(DISTANCE_TRACE_COLUMNS)
    # Additive extension: every shared column is still present and in order.
    assert list(trace_df.columns)[: len(TRACE_REQUIRED_COLUMNS)] == list(TRACE_REQUIRED_COLUMNS)
    assert trace_df["layer_name"].unique().tolist() == ["backbone.layer4"]
    assert trace_df["score_kind"].unique().tolist() == ["maha"]
    assert np.allclose(trace_df["distance_score"].to_numpy(dtype=float), [1.5, 9.0, 4.25])
    # Probe columns are carried over unchanged: the score reorders deferral, not predictions.
    assert trace_df["y_pred"].tolist() == [0, 2, 1]
    assert np.allclose(trace_df["confidence"].to_numpy(dtype=float), [0.9, 0.5, 0.8])
    assert trace_df["uq_method"].unique().tolist() == ["maha@backbone.layer4"]


def test_build_distance_trace_frame_roundtrips_through_parquet(tmp_path):
    trace_df = _distance_frame()
    path = tmp_path / "block.parquet"
    write_trace_block_atomic(trace_path=path, trace_df=trace_df, compression="zstd")

    import pandas as pd

    reloaded = pd.read_parquet(path)
    assert list(reloaded.columns) == list(DISTANCE_TRACE_COLUMNS)
    assert np.allclose(reloaded["distance_score"].to_numpy(dtype=float), [1.5, 9.0, 4.25])


def test_distance_and_probabilistic_fragments_scan_together(tmp_path):
    """A distance and a probabilistic fragment must union in one scan_traces call.

    The analysis joins distance rows against the uncalibrated rows they share
    samples with, so the two schemas have to coexist under one trace root.
    """
    root = tmp_path / "uq_traces"
    common = _common_meta()

    prob_df = build_probabilistic_trace_frame(
        trace_block_key="block-prob",
        run_id="run-123",
        common_meta=common,
        uq_method="uncalibrated",
        corruption_type="poisson_gaussian",
        severity=3,
        config_hash="cfg-123",
        git_sha="deadbeef",
        created_at_utc="2026-08-11T00:00:00Z",
        y_true=np.array([0, 1, 2], dtype=np.int64),
        probs=_probs(),
        sample_ids=np.array(["s0", "s1", "s2"], dtype=object),
    )
    dist_df = _distance_frame()

    for uq_method, frame, block_key in (
        ("uncalibrated", prob_df, "block-prob"),
        ("maha@backbone.layer4", dist_df, "block-dist"),
    ):
        path = resolve_trace_partition_path(
            trace_dataset_root=str(root),
            trace_block_key=block_key,
            dataset="m-eurosat",
            backbone="resnet50",
            uq_method=uq_method,
            corruption_type="poisson_gaussian",
            severity=3,
        )
        write_trace_block_atomic(trace_path=path, trace_df=frame, compression="zstd")

    shared = scan_traces(str(root), columns=["sample_id", "uq_method", "confidence", "correct"])
    assert len(shared) == 6
    assert set(shared["uq_method"]) == {"uncalibrated", "maha@backbone.layer4"}

    # The '@' and '.' in the scorer key survive as a partition directory name.
    assert (root / "dataset=m-eurosat" / "backbone=resnet50" / "uq_method=maha@backbone.layer4").is_dir()

    only_dist = scan_traces(
        str(root),
        filters={"uq_method": "maha@backbone.layer4"},
        columns=["sample_id", "distance_score", "severity"],
    )
    assert len(only_dist) == 3
    assert np.allclose(only_dist["distance_score"].to_numpy(dtype=float), [1.5, 9.0, 4.25])
