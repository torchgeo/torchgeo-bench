from torchgeo_bench.uq.traces import resolve_trace_partition_path


def test_resolve_trace_partition_path_template():
    path = resolve_trace_partition_path(
        trace_root="results/uq_traces",
        run_id="RID-1",
        dataset="m-eurosat",
        backbone="resnet50",
        uq_method="uncalibrated",
        corruption_type="cloud",
        severity=3,
        trace_format="parquet",
    )

    expected = (
        "results/uq_traces/run_id=RID-1/"
        "dataset=m-eurosat/backbone=resnet50/uq_method=uncalibrated/"
        "corruption_type=cloud/severity=3/part-000.parquet"
    )
    assert path.as_posix() == expected
