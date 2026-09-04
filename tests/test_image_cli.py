# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Tests for the discoverable image CLI."""

import subprocess
import sys

import pytest

from torchgeo_bench.config_schema import validate_run_config
from torchgeo_bench.image_cli import _image_size, _model_names, _set, main


def test_dry_run_applies_explicit_flags_and_preserves_false_values(capsys) -> None:
    main(
        [
            "run",
            "--model",
            "rcf",
            "--dataset",
            "m-eurosat",
            "--dataset",
            "burn_scars",
            "--image-size",
            "none",
            "--no-resume",
            "--methods",
            "knn",
            "--dry-run",
        ]
    )
    output = capsys.readouterr().out
    assert "image_size: null" in output
    assert "resume: false" in output
    assert "- burn_scars" in output
    assert "methods:\n  - knn" in output


def test_config_values_are_overridden_by_explicit_flags(tmp_path, capsys) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "model: {name: rcf}\ndatasets: [m-eurosat]\nruntime: {seed: 8}\noutput: {resume: true}\n",
        encoding="utf-8",
    )
    main(["run", "--config", str(path), "--seed", "3", "--no-resume", "--dry-run"])
    output = capsys.readouterr().out
    assert "seed: 3" in output
    assert "resume: false" in output


def test_nested_flag_mapping_and_image_size_validation() -> None:
    mapping = {}
    _set(mapping, "classification.linear", "refit_train_val", False)
    assert mapping == {"classification": {"linear": {"refit_train_val": False}}}
    assert _image_size("none") is None
    assert _image_size("224") == 224
    with pytest.raises(Exception, match="positive"):
        _image_size("0")


def test_boolean_flags_override_config(tmp_path, capsys) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "model: {name: rcf}\ndatasets: [m-eurosat]\n"
        "classification:\n  linear:\n    refit_train_val: true\n",
        encoding="utf-8",
    )
    main(["run", "--config", str(path), "--no-refit-train-val", "--no-temp-scale", "--dry-run"])
    assert "refit_train_val: false" in capsys.readouterr().out


def test_missing_model_or_dataset_fails_before_execution() -> None:
    with pytest.raises(SystemExit, match="2"):
        main(["run", "--dataset", "m-eurosat", "--dry-run"])
    with pytest.raises(SystemExit, match="2"):
        main(["run", "--model", "rcf", "--dry-run"])


def test_non_dry_run_calls_legacy_adapter(monkeypatch) -> None:
    received = []
    monkeypatch.setattr("torchgeo_bench.legacy_run.run", received.append)
    main(["run", "--model", "rcf", "--dataset", "m-eurosat"])
    assert received[0].model.name == "rcf"


def test_legacy_adapter_composes_and_translates_schema(monkeypatch) -> None:
    received = []
    monkeypatch.setattr("torchgeo_bench.main.main", received.append)
    config = validate_run_config(
        {
            "model": {"name": "rcf"},
            "datasets": ["m-eurosat"],
            "runtime": {"device": "cpu", "batch_size": 2},
            "input": {"normalization": "none"},
            "classification": {"methods": ["knn"]},
        }
    )
    from torchgeo_bench.legacy_run import run

    run(config)
    assert received[0].device == "cpu"
    assert received[0].dataset.normalization == "identity"
    assert received[0].eval.skip_linear is True


def test_linear_only_is_rejected_by_legacy_adapter() -> None:
    with pytest.raises(SystemExit, match="2"):
        main(["run", "--model", "rcf", "--dataset", "m-eurosat", "--methods", "linear"])


def test_unknown_config_field_fails_before_execution(tmp_path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("model: {name: rcf}\ndatasets: [m-eurosat]\nrunntim: {}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="2"):
        main(["run", "--config", str(path), "--dry-run"])


def test_catalogs_are_lightweight() -> None:
    assert "timm/resnet50" in _model_names()
    assert len(_model_names()) > 1


def test_catalog_name_selection_and_unimplemented_commands(capsys) -> None:
    main(["models"])
    assert "timm/resnet50" in capsys.readouterr().out
    main(["datasets"])
    assert "m-eurosat" in capsys.readouterr().out
    main(["models", "timm/resnet50"])
    assert capsys.readouterr().out.strip() == "timm/resnet50"
    main(["datasets", "m-eurosat"])
    assert capsys.readouterr().out.strip() == "m-eurosat"
    with pytest.raises(SystemExit, match="2"):
        main(["profile"])
    with pytest.raises(SystemExit, match="unknown model"):
        main(["models", "unknown"])
    with pytest.raises(SystemExit, match="unknown dataset"):
        main(["datasets", "unknown"])


def test_help_and_catalog_subprocesses_do_not_import_ml() -> None:
    code = (
        "import sys; from torchgeo_bench.image_cli import main; "
        "main(sys.argv[1:]); "
        "print([n for n in ('torch','torchgeo','pandas','numpy') if n in sys.modules])"
    )
    for args in (["models"], ["datasets"]):
        result = subprocess.run(
            [sys.executable, "-c", code, *args], capture_output=True, text=True, check=True
        )
        assert result.stdout.rstrip().endswith("[]")


def test_config_help_is_available_without_selection(capsys) -> None:
    with pytest.raises(SystemExit) as error:
        main(["run", "--config-help"])
    assert error.value.code == 0
    assert "title: RunConfig" in capsys.readouterr().out
