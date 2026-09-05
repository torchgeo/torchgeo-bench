"""CPU-capable program smoke tests using real data under the invocation's data/ directory.

Run with ``pytest -m slow tests/test_integration.py`` after downloading the datasets.
Missing datasets skip individually; present but incompatible or incomplete data must fail.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from torchgeo_bench.datasets import get_bench_dataset_class

from .test_cli_program import run_cli

pytestmark = pytest.mark.slow


def require_dataset_data(name: str) -> None:
    """Skip only when the requested dataset has not been supplied."""
    if name.startswith("m-"):
        paths = [
            Path("data/classification_v1.0") / name,
            Path("data/classification_v1.0_wds") / name,
        ]
    elif name in ("eurosat", "eurosat-spatial"):
        paths = [Path("data/eurosat")]
    else:
        paths = [Path("data/geobenchv2") / name]
    if not any(path.exists() for path in paths):
        pytest.skip(f"{name} data not supplied; expected one of {paths}")


@pytest.mark.parametrize(
    ("dataset", "bands", "partition"),
    [
        ("m-eurosat", "rgb", "0.01x_train"),
        ("so2sat", "rgb", "default"),
        ("eurosat", "all", "default"),
    ],
    ids=["v1-m-eurosat", "v2-so2sat", "eurosat"],
)
def test_real_classification_program(
    tmp_path: Path, dataset: str, bands: str, partition: str
) -> None:
    require_dataset_data(dataset)
    output = tmp_path / "classification.csv"
    arguments = [
        "run",
        "model=rcf",
        "model.features=32",
        f"dataset.names=[{dataset}]",
        f"dataset.bands={bands}",
        f"dataset.partition={partition}",
        "dataset.image_size=32",
        "dataset.batch_size=64",
        "dataset.num_workers=0",
        "device=cpu",
        "eval.bootstrap=5",
        "eval.c_range=[-2,2,3]",
        f"output={output}",
    ]
    result = run_cli(*arguments, cwd=Path.cwd(), timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    assert output.is_file(), result.stdout + result.stderr
    rows = pd.read_csv(output)
    assert set(rows["method"]) == {"knn5", "linear"}
    assert (rows["dataset"] == dataset).all()
    assert (rows["bands"] == bands).all()
    assert (rows["num_classes"] == get_bench_dataset_class(dataset).num_classes).all()
    assert (rows["feature_dim"] == 32).all()
    assert (rows[["n_train", "n_val", "n_test"]] > 0).all().all()
    assert np.isfinite(rows[["metric_value", "ci_lower", "ci_upper"]]).all().all()
    assert rows["metric_value"].between(0, 1).all()
    assert (rows["ci_lower"] <= rows["ci_upper"]).all()

    before = output.read_bytes()
    resumed = run_cli(*arguments, "resume=true", cwd=Path.cwd(), timeout=120)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert output.read_bytes() == before


@pytest.mark.parametrize("cached", [True, False], ids=["cached", "uncached"])
def test_real_segmentation_program(tmp_path: Path, cached: bool) -> None:
    require_dataset_data("caffe")
    output = tmp_path / "segmentation.csv"
    arguments = [
        "run",
        "model=timm/resnet18",
        "model.pretrained=false",
        "model.seed=0",
        "dataset.names=[caffe]",
        "dataset.image_size=32",
        "dataset.batch_size=32",
        "dataset.num_workers=0",
        "device=cpu",
        "eval.bootstrap=5",
        "eval.segmentation.head_type=linear",
        "eval.segmentation.epochs=1",
        "eval.segmentation.batch_size=16",
        f"eval.segmentation.cache_features={str(cached).lower()}",
        f"output={output}",
    ]
    result = run_cli(*arguments, cwd=Path.cwd(), timeout=600)
    assert result.returncode == 0, result.stdout + result.stderr
    assert output.is_file(), result.stdout + result.stderr
    rows = pd.read_csv(output)
    assert len(rows) == 1
    row = rows.iloc[0]
    assert row["dataset"] == "caffe"
    assert row["method"] == "seg-linear"
    assert row["metric_name"] == "mIoU"
    assert 0 <= row["metric_value"] <= 1
    assert np.isfinite(row[["metric_value", "ci_lower", "ci_upper"]].astype(float)).all()
    assert row["ci_lower"] <= row["ci_upper"]
    assert row["best_batch_size"] == (16 if cached else 32)

    before = output.read_bytes()
    resumed = run_cli(*arguments, "resume=true", cwd=Path.cwd(), timeout=120)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    assert output.read_bytes() == before
