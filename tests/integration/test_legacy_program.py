"""Exercise the legacy module interface against isolated, real on-disk inputs."""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.support.cli import cli_output, run_cli
from tests.support.data import write_classification_files

pytestmark = pytest.mark.integration


@pytest.fixture
def classification_files(tmp_path: Path) -> Path:
    return write_classification_files(tmp_path, "m-eurosat", (2, 7))


def classification_arguments(output: Path) -> list[str]:
    """Use a small real feature extractor and the real KNN/linear implementations."""
    return [
        "run",
        "model=rcf",
        "model.features=8",
        "dataset.names=[m-eurosat]",
        "dataset.image_size=16",
        "dataset.batch_size=8",
        "dataset.num_workers=0",
        "device=cpu",
        "eval.bootstrap=5",
        "eval.c_range=[1,2,2]",
        f"output={output}",
    ]


@pytest.mark.parametrize("temperature_scaling", [False, True])
def test_classification_program_handles_noncontiguous_labels(
    tmp_path: Path, classification_files: Path, *, temperature_scaling: bool
) -> None:
    output = tmp_path / "classification.csv"
    arguments = classification_arguments(output)
    if temperature_scaling:
        arguments.extend(["eval.merge_val=false", "eval.calibration.temp_scale=true"])
    completed = run_cli(*arguments, cwd=tmp_path)
    assert completed.returncode == 0, cli_output(completed)
    assert output.is_file(), cli_output(completed)

    rows = pd.read_csv(output).set_index("method")
    assert set(rows.index) == {"knn5", "linear"}
    assert (rows["dataset"] == "m-eurosat").all()
    assert (rows["metric_name"] == "accuracy").all()
    assert (rows["metric_value"] == 1.0).all()
    assert rows.loc["knn5", "ece"] == 0
    assert rows.loc["linear", "ece"] < 0.2
    if temperature_scaling:
        assert np.isfinite(rows.loc["linear", "temperature"])
        assert rows.loc["linear", "temperature"] > 0
        assert rows.loc["linear", "ece_ts"] < 0.2


def test_program_profiles_features_and_resumes_without_input_files(
    tmp_path: Path, classification_files: Path
) -> None:
    output = tmp_path / "measurements.csv"
    arguments = [
        *classification_arguments(output),
        "eval.intrinsic_dim.enabled=true",
        "eval.intrinsic_dim.estimators=[]",
        "eval.profile.enabled=true",
        "eval.profile.n_warmup=0",
        "eval.profile.n_measure=1",
    ]
    completed = run_cli(*arguments, cwd=tmp_path)
    assert completed.returncode == 0, cli_output(completed)
    assert output.is_file(), cli_output(completed)
    rows = pd.read_csv(output)
    assert set(rows["method"]) == {"knn5", "linear", "profile", "intrinsic_dim"}
    assert np.isfinite(rows["metric_value"]).all()
    assert rows.loc[rows["metric_name"] == "gflops", "metric_value"].item() > 0
    assert (rows["n_train"] == 24).all()
    assert (rows["n_val"] == 8).all()
    assert (rows["n_test"] == 8).all()

    before = output.read_bytes()
    shutil.rmtree(classification_files)
    resumed = run_cli(*arguments, "resume=true", cwd=tmp_path)
    assert resumed.returncode == 0, cli_output(resumed)
    assert output.read_bytes() == before


def test_program_reinitializes_for_multispectral_and_multilabel_datasets(tmp_path: Path) -> None:
    write_classification_files(tmp_path, "m-eurosat", (2, 7), all_bands=True)
    labels = tuple([int(index == label) for index in range(43)] for label in (0, 1))
    write_classification_files(tmp_path, "m-bigearthnet", labels, all_bands=True)
    output = tmp_path / "multiple.csv"
    completed = run_cli(
        *classification_arguments(output),
        "dataset.names=[m-eurosat,m-bigearthnet]",
        "dataset.bands=all",
        cwd=tmp_path,
    )
    assert completed.returncode == 0, cli_output(completed)
    assert output.is_file(), cli_output(completed)
    rows = pd.read_csv(output)
    assert len(rows) == 4
    for dataset, metric in (("m-eurosat", "accuracy"), ("m-bigearthnet", "micro_mAP")):
        selected = rows[rows["dataset"] == dataset]
        assert set(selected["method"]) == {"knn5", "linear"}
        assert (selected["metric_name"] == metric).all()
        assert (selected["metric_value"] > 0.99).all()
    assert (rows["bands"] == "all").all()
    assert np.isfinite(rows["metric_value"]).all()


@pytest.mark.parametrize(
    "dataset_names",
    [
        ("m-forestnet",),
        ("m-forestnet", "m-eurosat"),
        ("m-eurosat", "m-forestnet"),
    ],
)
def test_program_fails_when_any_requested_data_is_unavailable(
    tmp_path: Path, classification_files: Path, dataset_names: tuple[str, ...]
) -> None:
    output = tmp_path / "missing.csv"
    completed = run_cli(
        *classification_arguments(output),
        f"dataset.names=[{','.join(dataset_names)}]",
        cwd=tmp_path,
    )
    assert completed.returncode != 0, cli_output(completed)
    assert "Required files for 'm-forestnet' are missing" in completed.stderr
    assert "torchgeo-bench download geobench_v1 --datasets m-forestnet" in completed.stderr
    if dataset_names[0] == "m-eurosat":
        rows = pd.read_csv(output)
        assert set(rows["method"]) == {"knn5", "linear"}
        assert (rows["dataset"] == "m-eurosat").all()
    else:
        assert not output.exists()


def test_flops_program_writes_both_band_configurations_and_resumes(tmp_path: Path) -> None:
    output = tmp_path / "flops.csv"
    arguments = [
        "flops",
        "model=rcf",
        "model.features=8",
        "device=cpu",
        "image_size=16",
        "timing_batch_size=2",
        "n_warmup=0",
        "n_measure=1",
        "seg_band_configs=[]",
        f"output={output}",
    ]
    completed = run_cli(*arguments, cwd=tmp_path)
    assert completed.returncode == 0, cli_output(completed)
    assert output.is_file(), cli_output(completed)
    rows = pd.read_csv(output).set_index("band_config")
    assert set(rows.index) == {"rgb", "s2"}
    assert (rows["task"] == "classification").all()
    assert (rows["gflops_total"] > 0).all()
    assert rows.loc["s2", "gflops_backbone"] > rows.loc["rgb", "gflops_backbone"]
    assert (rows["throughput_samples_per_sec"] > 0).all()
    before = output.read_bytes()
    resumed = run_cli(*arguments, "resume=true", cwd=tmp_path)
    assert resumed.returncode == 0, cli_output(resumed)
    assert output.read_bytes() == before
