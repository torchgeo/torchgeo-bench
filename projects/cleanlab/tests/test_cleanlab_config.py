"""Behavioral coverage for analysis using the library configuration API."""

from pathlib import Path

import cleanlab_extract_probs
import pandas as pd
import pytest
import torch

from torchgeo_bench.config_schema import ModelConfig, RunConfig, RuntimeConfig
from torchgeo_bench.datasets import get_bench_dataset_class
from torchgeo_bench.presets import build_model, resolve_run_config


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
    cfg, model_cfg = resolve_run_config(
        RunConfig(
            model=ModelConfig(name=model_name),
            datasets=["m-eurosat"],
            runtime=RuntimeConfig(seed=17),
        ),
        "m-eurosat",
    )
    model = build_model(model_cfg, bands=bands, normalization="identity")
    features = model(torch.zeros(1, 3, 16, 16))
    assert model_cfg.kwargs["seed"] == 17
    assert cfg.runtime.seed == 17
    assert features.shape == (1, model_cfg.kwargs["features"])
    assert torch.isfinite(features).all()


@pytest.mark.parametrize("directory", [False, True])
def test_cleanlab_selects_top_linear_result_from_file_or_directory(
    tmp_path: Path, *, directory: bool
) -> None:
    rows = pd.DataFrame(
        [
            {
                "dataset": "m-eurosat",
                "name": name,
                "method": method,
                "normalization": normalization,
                "metric_value": score,
            }
            for name, method, normalization, score in [
                ("lower", "linear", "identity", 0.7),
                ("best", "linear", "bandspec_zscore", 0.9),
                ("legacy", "linear", "raw", 1.0),
                ("knn", "knn5", "identity", 1.0),
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


def test_extraction_loads_strict_custom_model_config(tmp_path: Path) -> None:
    path = tmp_path / "run.yaml"
    path.write_text(
        "model:\n"
        "  name: custom-rcf\n"
        "  target: torchgeo_bench.models.RCFBench\n"
        "  kwargs:\n"
        "    features: 8\n"
        "    seed: 17\n"
        "datasets: [m-eurosat]\n"
        "input:\n"
        "  image_size: null\n"
        "  time_steps: 2\n"
    )
    cfg = cleanlab_extract_probs.source_config(path, "custom-rcf", "m-eurosat")
    effective, preset = resolve_run_config(cfg, "m-eurosat")
    assert effective.input.image_size is None
    assert effective.input.time_steps == 2
    assert preset.kwargs == {"features": 8, "seed": 17}
    bands = cleanlab_extract_probs.band_specs(get_bench_dataset_class("m-eurosat")(), "rgb")
    model = build_model(preset, bands=bands, normalization="identity")
    assert model(torch.zeros(1, 3, 16, 16)).shape == (1, 8)
    path.write_text(path.read_text() + "unknown: true\n")
    with pytest.raises(ValueError, match="Extra inputs"):
        cleanlab_extract_probs.source_config(path, "custom-rcf", "m-eurosat")
