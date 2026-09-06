"""Behavioral coverage for experiment scripts using the library config API."""

import json

import pytest
import torch

from experiments.scripts import (
    audit_model_native,
    introspect_seg_layers,
    tune_dataloader,
)
from torchgeo_bench.config import compose_config
from torchgeo_bench.models._normalization import UnsupportedNormalizationError


@pytest.mark.parametrize("script", [audit_model_native, introspect_seg_layers])
def test_analysis_scripts_preserve_requested_rgb_order(script) -> None:
    bands = script.band_specs("m-eurosat", "rgb")
    assert [band.name for band in bands] == ["red", "green", "blue"]


def test_tuner_builds_packaged_model_config() -> None:
    bands = audit_model_native.band_specs("m-eurosat", "rgb")
    model = tune_dataloader._build_model("rcf", bands)
    features = model(torch.zeros(1, 3, 16, 16))
    cfg = compose_config(["model=rcf"])
    assert features.shape == (1, cfg.model.features)
    assert torch.isfinite(features).all()


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
