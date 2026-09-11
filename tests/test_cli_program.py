"""Run the installed program against tiny on-disk inputs without mocking its internals."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import pytest
import yaml

from torchgeo_bench.config_schema import RunConfig
from torchgeo_bench.datasets import get_bench_dataset_class
from torchgeo_bench.flops_config import FlopsConfig


def run_cli(
    *arguments: str, cwd: Path, timeout: int = 120, offline: bool = True
) -> subprocess.CompletedProcess[str]:
    """Invoke the same entry point used by the console command."""
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "PYTHONPATH": str(Path(__file__).parents[1] / "src"),
    }
    if offline:
        env["HF_HUB_OFFLINE"] = "1"
    return subprocess.run(
        [sys.executable, "-m", "torchgeo_bench", *arguments],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def write_classification_files(
    root: Path,
    dataset_name: str,
    labels: tuple[int | list[int], ...],
    *,
    all_bands: bool = False,
) -> Path:
    """Write small separable samples in the reader's actual on-disk format."""
    directory = root / "data" / "classification_v1.0" / dataset_name
    directory.mkdir(parents=True)
    bench = get_bench_dataset_class(dataset_name)()
    specs = bench.select_band_specs(None if all_bands else tuple(bench.rgb_bands))
    bands = [band.source_name for band in specs]
    partition: dict[str, list[str]] = {}
    for split, per_class in (("train", 12), ("valid", 4), ("test", 4)):
        partition[split] = []
        for class_index, label in enumerate(labels):
            for index in range(per_class):
                sample_id = f"{split}-{class_index}-{index}"
                partition[split].append(sample_id)
                with h5py.File(directory / f"{sample_id}.hdf5", "w") as sample:
                    for band in specs:
                        pixels = np.full(
                            (16, 16),
                            band.mean + band.std * (0.5 + 2.5 * class_index + 0.001 * index),
                            dtype=np.float32,
                        )
                        sample.create_dataset(band.source_name, data=pixels)
                    sample.attrs["metadata_json"] = json.dumps(
                        {"label": label, "bands_order": bands}
                    )
    (directory / "default_partition.json").write_text(json.dumps(partition))
    return directory


@pytest.fixture
def classification_files(tmp_path: Path) -> Path:
    return write_classification_files(tmp_path, "m-eurosat", (2, 7))


def classification_arguments(
    output: Path,
    *,
    dataset_names: tuple[str, ...] = ("m-eurosat",),
    extra_settings: dict[str, Any] | None = None,
) -> list[str]:
    """Use a small real feature extractor and the real KNN/linear implementations."""
    config = RunConfig.model_validate(
        {
            "model": {"name": "rcf", "kwargs": {"features": 8}},
            "datasets": list(dataset_names),
            "input": {"image_size": 16},
            "runtime": {"batch_size": 8, "workers": 0, "device": "cpu"},
            "classification": {
                "bootstrap_samples": 5,
                "linear": {"c_log10_start": 1.0, "c_log10_stop": 2.0, "c_count": 2},
            },
            **(extra_settings or {}),
        }
    )
    path = output.with_suffix(".yaml")
    path.write_text(yaml.safe_dump(config.model_dump_yaml()))
    return [
        "run",
        "--config",
        str(path),
        "--output",
        str(output),
    ]


@pytest.mark.parametrize("temperature_scaling", [False, True])
def test_classification_program_handles_noncontiguous_labels(
    tmp_path: Path, classification_files: Path, *, temperature_scaling: bool
) -> None:
    output = tmp_path / "classification=1.csv"
    arguments = classification_arguments(output)
    if temperature_scaling:
        arguments.extend(["--no-refit-train-val", "--temp-scale"])
    completed = run_cli(*arguments, cwd=tmp_path)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert output.is_file(), completed.stdout + completed.stderr

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
    else:
        assert rows["temperature"].isna().all()
    assert not (tmp_path / "results" / "profiles").exists()
    assert not (tmp_path / "results" / "intrinsic_dim").exists()


def test_program_profiles_features_and_resumes_without_input_files(
    tmp_path: Path, classification_files: Path
) -> None:
    output = tmp_path / "measurements.csv"
    arguments = classification_arguments(
        output,
        extra_settings={
            "intrinsic_dim": {"enabled": True, "estimators": []},
            "profile": {"enabled": True, "n_warmup": 0, "n_measure": 1},
        },
    )
    completed = run_cli(*arguments, cwd=tmp_path)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert output.is_file(), completed.stdout + completed.stderr
    rows = pd.read_csv(output)
    assert set(rows["method"]) == {"knn5", "linear", "profile", "intrinsic_dim"}
    assert np.isfinite(rows["metric_value"]).all()
    assert rows.loc[rows["metric_name"] == "gflops", "metric_value"].item() > 0
    assert (rows["n_train"] == 24).all()
    assert (rows["n_val"] == 8).all()
    assert (rows["n_test"] == 8).all()

    before = output.read_bytes()
    shutil.rmtree(classification_files)
    resumed = run_cli(*arguments, "--resume", cwd=tmp_path)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert output.read_bytes() == before


def test_program_reinitializes_for_multispectral_and_multilabel_datasets(tmp_path: Path) -> None:
    write_classification_files(tmp_path, "m-eurosat", (2, 7), all_bands=True)
    labels = tuple([int(index == label) for index in range(43)] for label in (0, 1))
    write_classification_files(tmp_path, "m-bigearthnet", labels, all_bands=True)
    output = tmp_path / "multiple.csv"
    completed = run_cli(
        *classification_arguments(output),
        "--dataset",
        "m-eurosat",
        "--dataset",
        "m-bigearthnet",
        "--bands",
        "all",
        cwd=tmp_path,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert output.is_file(), completed.stdout + completed.stderr
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
        *classification_arguments(output, dataset_names=dataset_names),
        cwd=tmp_path,
    )
    assert completed.returncode != 0
    assert "Required files for 'm-forestnet' are missing" in completed.stderr
    assert "torchgeo-bench download geobench_v1 --datasets m-forestnet" in completed.stderr
    if dataset_names[0] == "m-eurosat":
        rows = pd.read_csv(output)
        assert set(rows["method"]) == {"knn5", "linear"}
        assert (rows["dataset"] == "m-eurosat").all()
    else:
        assert not output.exists()


def test_flops_program_writes_both_band_configurations_and_resumes(tmp_path: Path) -> None:
    output = tmp_path / "flops=1.csv"
    config = FlopsConfig.model_validate(
        {
            "model": {"name": "rcf", "kwargs": {"features": 8}},
            "runtime": {"device": "cpu"},
            "input": {"image_size": 16},
            "timing": {"batch_size": 2, "n_warmup": 0, "n_measure": 1},
            "segmentation": {"band_configs": []},
            "output": {"resume": False},
        }
    )
    path = tmp_path / "flops.yaml"
    path.write_text(yaml.safe_dump(config.model_dump_yaml()))
    arguments = [
        "flops",
        "--config",
        str(path),
        "--output",
        str(output),
    ]
    completed = run_cli(*arguments, cwd=tmp_path)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert output.is_file(), completed.stdout + completed.stderr
    rows = pd.read_csv(output).set_index("band_config")
    assert set(rows.index) == {"rgb", "s2"}
    assert (rows["task"] == "classification").all()
    assert rows.loc["rgb", "n_channels"] == 3
    assert rows.loc["s2", "n_channels"] == 12
    assert (rows["gflops_total"] > 0).all()
    assert rows.loc["s2", "gflops_backbone"] > rows.loc["rgb", "gflops_backbone"]
    assert (rows["throughput_samples_per_sec"] > 0).all()
    before = output.read_bytes()
    resumed = run_cli(*arguments, "--resume", cwd=tmp_path)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert output.read_bytes() == before
