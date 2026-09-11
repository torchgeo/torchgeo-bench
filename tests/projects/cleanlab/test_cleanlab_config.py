"""Behavioral coverage for analysis using the library configuration API."""

from pathlib import Path

import pandas as pd
import pytest
import torch

from projects.cleanlab import cleanlab_extract_probs
from torchgeo_bench.config import compose_config, instantiate
from torchgeo_bench.datasets import get_bench_dataset_class
from torchgeo_bench.main import resolve_model_config


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        ("rgb", ["red", "green", "blue"]),
        (["blue", "red", "green"], ["blue", "red", "green"]),
    ],
)
def test_extraction_preserves_requested_band_order(
    requested: str | list[str], expected: list[str]
) -> None:
    bench = get_bench_dataset_class("m-eurosat")()
    bands = cleanlab_extract_probs.band_specs(bench, requested)
    assert [band.name for band in bands] == expected


def test_extraction_discovers_and_builds_packaged_model_configs() -> None:
    model_name = cleanlab_extract_probs.build_name_to_config_map()["rcf"]
    bench = get_bench_dataset_class("m-eurosat")()
    bands = cleanlab_extract_probs.band_specs(bench, "rgb")
    cfg = compose_config([f"model={model_name}", "seed=17"])
    model_cfg = resolve_model_config(cfg.model, "m-eurosat")
    model = instantiate(model_cfg, bands=bands, normalization="identity")
    features = model(torch.zeros(1, 3, 16, 16))
    assert model_cfg.seed == 17
    assert features.shape == (1, cfg.model.features)
    assert torch.isfinite(features).all()


@pytest.mark.parametrize("directory", [False, True])
def test_cleanlab_selects_top_linear_result_from_file_or_directory(
    tmp_path: Path, *, directory: bool
) -> None:
    rows = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "name": name,
                "method": method,
                "normalization": normalization,
                "metric_value": score,
            }
            for dataset, name, method, normalization, score in [
                ("m-eurosat", "lower", "linear", "identity", 0.7),
                ("m-eurosat", "best", "linear", "bandspec_zscore", 0.9),
                ("m-eurosat", "legacy", "linear", "raw", 1.0),
                ("m-eurosat", "knn", "knn5", "identity", 1.0),
                ("m-pv4ger", "other-dataset", "linear", "identity", 1.0),
            ]
        ]
    )
    if directory:
        path = tmp_path / "models"
        path.mkdir()
        rows.iloc[:2].to_csv(path / "first.csv", index=False)
        rows.iloc[2:].to_csv(path / "second.csv", index=False)
    else:
        path = tmp_path / "results.csv"
        rows.to_csv(path, index=False)
    result = cleanlab_extract_probs.lookup_top1(path, "m-eurosat")
    assert result["name"] == "best"
    assert result["metric_value"] == 0.9


def test_cleanlab_reports_empty_result_directory(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="No linear-probe rows"):
        cleanlab_extract_probs.lookup_top1(tmp_path, "m-eurosat")


def test_extraction_defaults_use_repository_results(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["cleanlab_extract_probs.py", "--dataset", "m-eurosat"])
    args = cleanlab_extract_probs.parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    assert args.results == repo_root / "results" / "models"
    assert args.out == repo_root / "results" / "cleanlab" / "probs"
