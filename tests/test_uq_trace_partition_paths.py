from torchgeo_bench.uq.traces import resolve_trace_partition_path


def test_resolve_trace_partition_path_template():
    path = resolve_trace_partition_path(
        trace_dataset_root="results/uq_traces",
        trace_block_key="abc123",
        dataset="m-eurosat",
        backbone="resnet50",
        uq_method="uncalibrated",
        corruption_type="cloud",
        severity=3,
    )

    expected = (
        "results/uq_traces/dataset=m-eurosat/backbone=resnet50/uq_method=uncalibrated/"
        "corruption_type=cloud/severity=3/trace_block_key=abc123.parquet"
    )
    assert path.as_posix() == expected
