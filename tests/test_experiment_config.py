"""Behavioral coverage for experiment scripts using the library config API."""

import json

import pandas as pd
import pytest
import torch

from experiments.scripts import (
    audit_model_native,
    cleanlab_extract_probs,
    introspect_seg_layers,
    tune_dataloader,
)
from torchgeo_bench.config import compose_config
from torchgeo_bench.models._normalization import UnsupportedNormalizationError


@pytest.mark.parametrize("script", [audit_model_native, introspect_seg_layers])
def test_analysis_scripts_preserve_requested_rgb_order(script) -> None:
    bands = script.band_specs("m-eurosat", "rgb")
    assert [band.name for band in bands] == ["red", "green", "blue"]


def test_scripts_discover_and_build_packaged_model_configs() -> None:
    model_name = cleanlab_extract_probs.build_name_to_config_map()["rcf"]
    bands = audit_model_native.band_specs("m-eurosat", "rgb")
    model = tune_dataloader._build_model(model_name, bands)
    features = model(torch.zeros(1, 3, 16, 16))
    cfg = compose_config([f"model={model_name}"])
    assert features.shape == (1, cfg.model.features)
    assert torch.isfinite(features).all()


@pytest.mark.parametrize("directory", [False, True])
def test_cleanlab_selects_top_linear_result_from_file_or_directory(tmp_path, directory) -> None:
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


def test_cleanlab_reports_empty_result_directory(tmp_path) -> None:
    with pytest.raises(SystemExit, match="No linear-probe rows"):
        cleanlab_extract_probs.lookup_top1(tmp_path, "m-eurosat")


@pytest.mark.parametrize(
    "error",
    [
        UnsupportedNormalizationError("no pretraining pipeline"),
        ValueError("invalid model_native configuration"),
    ],
)
def test_native_audit_only_classifies_unsupported_normalization(
    tmp_path, monkeypatch, error
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "rcf.yaml").write_text("_target_: torchgeo_bench.models.RCFBench\nname: rcf\n")
    output = tmp_path / "audit.json"
    monkeypatch.setattr(audit_model_native, "CONF", model_dir)
    monkeypatch.setattr("sys.argv", ["audit_model_native.py", "--out", str(output)])

    def fail(config, **kwargs) -> None:
        raise error

    monkeypatch.setattr(audit_model_native, "instantiate", fail)
    if isinstance(error, UnsupportedNormalizationError):
        audit_model_native.main()
        assert json.loads(output.read_text())["rcf"]["model_native"] == "unsupported"
    else:
        with pytest.raises(ValueError, match="invalid model_native configuration"):
            audit_model_native.main()
        assert not output.exists()


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (RuntimeError("loader failed"), "loader failed"),
        (torch.cuda.OutOfMemoryError("device full"), "No dataloader configuration"),
    ],
)
def test_dataloader_tuning_does_not_report_failed_sweeps_as_success(
    monkeypatch, error, message
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "tune_dataloader.py",
            "--model",
            "rcf",
            "--device",
            "cpu",
            "--batch-sizes",
            "1",
            "--num-workers",
            "0",
        ],
    )
    monkeypatch.setattr(tune_dataloader, "_build_dataset", lambda *args: [])
    monkeypatch.setattr(tune_dataloader, "_build_model", lambda *args: torch.nn.Identity())

    def fail(*args) -> None:
        raise error

    monkeypatch.setattr(tune_dataloader, "_bench", fail)
    with pytest.raises(RuntimeError, match=message):
        tune_dataloader.main()
