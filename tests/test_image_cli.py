# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Tests for the discoverable image CLI."""

import argparse
import os
import runpy
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest
import yaml
from _pytest.capture import CaptureFixture
from _pytest.monkeypatch import MonkeyPatch
from omegaconf import DictConfig, OmegaConf, open_dict

from torchgeo_bench.commands._image import _set
from torchgeo_bench.config_schema import validate_run_config
from torchgeo_bench.image_cli import _image_size, _model_names, main


def test_dry_run_applies_explicit_flags_and_preserves_false_values(
    capsys: CaptureFixture[str],
) -> None:
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


def test_config_values_are_overridden_by_explicit_flags(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
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
    _set(mapping, "runtime", "workers", 0)
    assert mapping == {
        "classification": {"linear": {"refit_train_val": False}},
        "runtime": {"workers": 0},
    }
    assert _image_size("none") is None
    assert _image_size("224") == 224
    with pytest.raises(argparse.ArgumentTypeError, match="positive"):
        _image_size("0")


def test_boolean_flags_override_config(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "model: {name: rcf}\ndatasets: [m-eurosat]\n"
        "classification:\n  linear:\n    refit_train_val: true\n",
        encoding="utf-8",
    )
    main(
        [
            "run",
            "--config",
            str(path),
            "--no-refit-train-val",
            "--no-temp-scale",
            "--dry-run",
        ]
    )
    assert "refit_train_val: false" in capsys.readouterr().out


def test_missing_model_or_dataset_fails_before_execution() -> None:
    with pytest.raises(SystemExit, match="2"):
        main(["run", "--dataset", "m-eurosat", "--dry-run"])
    with pytest.raises(SystemExit, match="2"):
        main(["run", "--model", "rcf", "--dry-run"])


def test_non_dry_run_calls_legacy_adapter(monkeypatch: MonkeyPatch) -> None:
    received = []
    monkeypatch.setattr("torchgeo_bench.commands._image_runtime.run", received.append)
    main(["run", "--model", "rcf", "--dataset", "m-eurosat"])
    assert received[0].model.name == "rcf"


def test_legacy_adapter_composes_and_translates_schema(
    monkeypatch: MonkeyPatch,
) -> None:
    received = []

    def capture(config: object, *, strict: bool = False) -> None:
        assert strict is True
        received.append(config)

    monkeypatch.setattr("torchgeo_bench.commands._image_runtime.main", capture)
    config = validate_run_config(
        {
            "model": {"name": "rcf"},
            "datasets": ["m-eurosat"],
            "runtime": {"device": "cpu", "batch_size": 2},
            "input": {
                "normalization": "none",
                "image_size": 8,
                "time_steps": 1,
                "interpolation": "nearest",
            },
            "classification": {"methods": ["knn"]},
        }
    )
    from torchgeo_bench.commands._image_runtime import run

    run(config)
    assert received[0].device == "cpu"
    assert received[0].dataset.normalization == "identity"
    assert received[0].eval.skip_linear is True


def test_legacy_adapter_rejects_unavailable_cuda(monkeypatch: MonkeyPatch) -> None:
    import torch

    config = validate_run_config(
        {
            "model": {"name": "rcf"},
            "datasets": ["m-eurosat"],
            "runtime": {"device": "cuda:0"},
        }
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    from torchgeo_bench.commands._image_runtime import run

    with pytest.raises(RuntimeError, match="CUDA device"):
        run(config)


def test_legacy_adapter_preserves_model_and_segmentation_overrides(
    monkeypatch: MonkeyPatch,
) -> None:
    from torchgeo_bench import config as config_module
    from torchgeo_bench.commands import _image_runtime as legacy_run

    received = []

    def capture(config: object, *, strict: bool = False) -> None:
        assert strict is True
        received.append(config)

    monkeypatch.setattr("torchgeo_bench.commands._image_runtime.main", capture)
    original_compose = config_module.compose_config

    def compose_with_optional_sections(
        overrides: Sequence[str] = (),
        *,
        config_name: str = "config",
        default_model: str | None = "rcf",
    ) -> DictConfig:
        legacy = original_compose(overrides, config_name=config_name, default_model=default_model)
        with open_dict(legacy.model):
            legacy.model.eval = {}
            legacy.model.dataset_overrides = {
                "m-eurosat": {"image_size": 16, "interpolation": "bicubic"}
            }
        with open_dict(legacy.eval.segmentation):
            legacy.eval.segmentation.layers = ["layer1"]
        return legacy

    monkeypatch.setattr(config_module, "compose_config", compose_with_optional_sections)
    monkeypatch.setattr(legacy_run, "compose_config", compose_with_optional_sections)
    config = validate_run_config(
        {
            "model": {"name": "rcf"},
            "datasets": ["m-eurosat"],
            "runtime": {"device": "cpu"},
            "input": {"image_size": 8, "interpolation": "nearest"},
            "segmentation": {"layers": ["layer1"]},
        }
    )
    from torchgeo_bench.commands._image_runtime import run

    run(config)
    assert received[0].model.eval == {}
    assert received[0].eval.segmentation.layers == ["layer1"]


def test_legacy_adapter_preserves_preset_layers_when_schema_omits_them(
    monkeypatch: MonkeyPatch,
) -> None:
    received = []

    def capture(config: object, *, strict: bool = False) -> None:
        assert strict is True
        received.append(config)

    monkeypatch.setattr("torchgeo_bench.commands._image_runtime.main", capture)
    config = validate_run_config(
        {
            "model": {"name": "torchgeo/resnet50_s2rgb_satlas_si"},
            "datasets": ["burn_scars"],
            "runtime": {"device": "cpu"},
        }
    )
    from torchgeo_bench.commands._image_runtime import run

    run(config)
    assert received[0].model.eval.segmentation.layers == [
        "layer4",
        "layer3",
        "layer2",
        "layer1",
    ]


def test_legacy_adapter_explicit_empty_layers_clear_preset(
    monkeypatch: MonkeyPatch,
) -> None:
    received = []

    def capture(config: object, *, strict: bool = False) -> None:
        assert strict is True
        received.append(config)

    monkeypatch.setattr("torchgeo_bench.commands._image_runtime.main", capture)
    config = validate_run_config(
        {
            "model": {"name": "torchgeo/resnet50_s2rgb_satlas_si"},
            "datasets": ["burn_scars"],
            "runtime": {"device": "cpu"},
            "segmentation": {"layers": []},
        }
    )
    from torchgeo_bench.commands._image_runtime import run

    run(config)
    assert received[0].eval.segmentation.layers == []


def test_linear_only_is_rejected_by_legacy_adapter() -> None:
    with pytest.raises(SystemExit, match="2"):
        main(["run", "--model", "rcf", "--dataset", "m-eurosat", "--methods", "linear"])


def test_unknown_config_field_fails_before_execution(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("model: {name: rcf}\ndatasets: [m-eurosat]\nrunntim: {}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="2"):
        main(["run", "--config", str(path), "--dry-run"])


@pytest.mark.parametrize(
    "error_type",
    [FileNotFoundError, PermissionError, ValueError, yaml.YAMLError, RuntimeError, TypeError],
)
def test_runtime_failure_propagates_from_image_cli(
    monkeypatch: MonkeyPatch, error_type: type[Exception]
) -> None:
    def fail(_: object) -> None:
        raise error_type("benchmark failed")

    monkeypatch.setattr("torchgeo_bench.commands._image_runtime.run", fail)
    with pytest.raises(error_type, match="benchmark failed"):
        main(["run", "--model", "rcf", "--dataset", "m-eurosat", "--device", "cpu"])


@pytest.mark.parametrize(
    ("contents", "diagnostic"),
    [
        (None, "No such file"),
        ("model: [\n", "while parsing"),
        ("model: {name: rcf}\nmodel: {name: other}\n", "duplicate key"),
        ("[a, b]: value\n", "unhashable key"),
        ("model: !unknown rcf\n", "could not determine a constructor"),
        ("null\n", "top level"),
        ("runtime: null\n", "runtime"),
        ("runtime: []\n", "runtime"),
        ("runtime: {workers: -1}\n", "runtime.workers"),
    ],
)
def test_config_errors_exit_without_tracebacks(
    contents: str | None, diagnostic: str, tmp_path: Path
) -> None:
    path = tmp_path / "invalid.yaml"
    if contents is not None:
        path.write_text(contents, encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torchgeo_bench.image_cli",
            "run",
            "--config",
            str(path),
            "--model",
            "rcf",
            "--dataset",
            "m-eurosat",
            "--device",
            "cpu",
            "--dry-run",
        ],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 2
    assert result.stdout == ""
    assert "error:" in result.stderr
    assert str(path) in result.stderr
    assert diagnostic in result.stderr
    assert "Traceback" not in result.stderr


def test_config_directory_is_reported(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(["run", "--config", str(tmp_path), "--dry-run"])
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert str(tmp_path) in message
    assert "directory" in message


def test_unreadable_config_is_reported(
    tmp_path: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    path = tmp_path / "unreadable.yaml"

    def deny_read(_: Path, **kwargs: object) -> None:
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(Path, "open", deny_read)
    with pytest.raises(SystemExit) as error:
        main(["run", "--config", str(path), "--dry-run"])
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert str(path) in message
    assert "Permission denied" in message


def test_invalid_config_encoding_is_reported(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    path = tmp_path / "invalid-encoding.yaml"
    path.write_bytes(b"\xff")
    with pytest.raises(SystemExit) as error:
        main(["run", "--config", str(path), "--dry-run"])
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert str(path) in message
    assert "utf-8" in message


@pytest.mark.parametrize(
    ("section", "flags"),
    [
        ("model", ["--model", "rcf"]),
        ("runtime", ["--seed", "3"]),
        ("input", ["--image-size", "32"]),
        ("classification", ["--knn-k", "3"]),
        ("output", ["--no-resume"]),
        ("classification.linear", ["--no-refit-train-val"]),
        ("classification.calibration", ["--no-temp-scale"]),
    ],
)
@pytest.mark.parametrize("value", [None, [], [["seed", 8]], 3])
def test_invalid_sections_are_not_coerced_or_replaced_by_overrides(
    section: str, flags: list[str], value: object, tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    config = {"model": {"name": "rcf"}, "datasets": ["m-eurosat"]}
    if "." in section:
        parent, child = section.split(".")
        config[parent] = {child: value}
    else:
        config[section] = value
    path = tmp_path / "invalid-section.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        main(["run", "--config", str(path), *flags, "--dry-run"])
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert str(path) in message
    assert section in message
    assert "mapping" in message


def test_flags_complete_partial_config(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    path = tmp_path / "partial.yaml"
    path.write_text("runtime: {seed: 8}\n", encoding="utf-8")
    main(
        [
            "run",
            "--config",
            str(path),
            "--model",
            "rcf",
            "--dataset",
            "m-eurosat",
            "--device",
            "cpu",
            "--seed",
            "3",
            "--dry-run",
        ]
    )
    config = yaml.safe_load(capsys.readouterr().out)
    assert config["runtime"] == {"device": "cpu", "seed": 3}
    assert config["model"]["name"] == "rcf"
    assert config["datasets"] == ["m-eurosat"]


def test_nested_linear_override_preserves_sibling_values(
    tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        "model: {name: rcf}\ndatasets: [m-eurosat]\n"
        "classification:\n  knn_k: 7\n  linear:\n    refit_train_val: true\n",
        encoding="utf-8",
    )
    main(["run", "--config", str(path), "--no-refit-train-val", "--dry-run"])
    output = capsys.readouterr().out
    assert "knn_k: 7" in output
    assert "refit_train_val: false" in output


def test_catalogs_are_lightweight() -> None:
    assert "timm/resnet50" in _model_names()
    assert len(_model_names()) > 1


def test_catalog_name_selection_and_unimplemented_commands(
    capsys: CaptureFixture[str],
) -> None:
    main(["models"])
    assert "timm/resnet50" in capsys.readouterr().out
    main(["datasets"])
    assert "m-eurosat" in capsys.readouterr().out
    main(["models", "timm/resnet50"])
    assert "_target_:" in capsys.readouterr().out
    main(["datasets", "m-eurosat"])
    dataset_detail = capsys.readouterr().out
    assert "name: m-eurosat" in dataset_detail
    assert "task: classification" in dataset_detail
    with pytest.raises(SystemExit, match="2"):
        main(["profile"])
    with pytest.raises(SystemExit, match="unknown model"):
        main(["models", "unknown"])
    with pytest.raises(SystemExit, match="unknown dataset"):
        main(["datasets", "unknown"])


def test_unknown_catalog_entries_fail_before_execution() -> None:
    with pytest.raises(SystemExit, match="2"):
        main(["run", "--model", "unknown", "--dataset", "m-eurosat"])
    with pytest.raises(SystemExit, match="2"):
        main(["run", "--model", "rcf", "--dataset", "unknown"])


def test_help_and_catalog_subprocesses_do_not_import_ml() -> None:
    code = (
        "import sys; from torchgeo_bench.image_cli import main; "
        "main(sys.argv[1:]); "
        "print([n for n in ('torch','torchgeo','pandas','numpy') if n in sys.modules])"
    )
    for args in (["models"], ["datasets"]):
        result = subprocess.run(
            [sys.executable, "-c", code, *args],
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.rstrip().endswith("[]")


def test_config_help_is_available_without_selection(
    capsys: CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        main(["run", "--config-help"])
    assert error.value.code == 0
    assert "title: RunConfig" in capsys.readouterr().out


def test_main_rejects_unknown_parser_command(monkeypatch: MonkeyPatch) -> None:
    import argparse

    monkeypatch.setattr(
        "torchgeo_bench.image_cli._parser",
        lambda: argparse.Namespace(parse_args=lambda _: argparse.Namespace(command="other")),
    )
    with pytest.raises(SystemExit, match="not implemented"):
        main([])


def test_module_entrypoint(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "argv", ["torchgeo-bench", "models", "rcf"])
    runpy.run_module("torchgeo_bench.image_cli", run_name="__main__")


@pytest.mark.parametrize("value", ["null", "3", "[]"])
def test_invalid_model_section_has_a_field_error(
    value: str, tmp_path: Path, capsys: CaptureFixture[str]
) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(f"model: {value}\ndatasets: [m-eurosat]\n")
    with pytest.raises(SystemExit) as error:
        main(["run", "--config", str(path), "--dry-run"])
    assert error.value.code == 2
    assert "model" in capsys.readouterr().err


def test_dry_run_rejects_unsupported_method_and_band_selections() -> None:
    for flags in (["--methods", "linear"], ["--bands", "red,,blue"]):
        with pytest.raises(SystemExit) as error:
            main(["run", "--model", "rcf", "--dataset", "m-eurosat", "--dry-run", *flags])
        assert error.value.code == 2


def test_yaml_bands_require_a_list(tmp_path: Path) -> None:
    path = tmp_path / "bands.yaml"
    path.write_text("model: {name: rcf}\ndatasets: [m-eurosat]\ninput:\n  bands: red,green\n")
    with pytest.raises(SystemExit) as error:
        main(["run", "--config", str(path), "--dry-run"])
    assert error.value.code == 2


def test_public_download_and_profile_dispatch(monkeypatch: MonkeyPatch) -> None:
    from torchgeo_bench import commands

    received = []
    monkeypatch.setattr(commands, "download", received.append)
    monkeypatch.setattr(commands, "profile", received.append)
    main(["download", "m-eurosat", "burn_scars"])
    main(["profile", "--model", "rcf", "--dataset", "m-eurosat", "--batch-size", "4"])
    assert received[0].target == ["m-eurosat", "burn_scars"]
    assert received[1].model == "rcf"
    assert received[1].dataset == "m-eurosat"
    assert received[1].batch_size == 4


@pytest.mark.parametrize(("available", "expected"), [(False, "cpu"), (True, "cuda:0")])
def test_auto_device_resolves_before_legacy_execution(
    monkeypatch: MonkeyPatch, *, available: bool, expected: str
) -> None:
    from torchgeo_bench.commands import _image_runtime

    received = []
    monkeypatch.setattr(_image_runtime.torch.cuda, "is_available", lambda: available)
    monkeypatch.setattr(_image_runtime, "main", lambda cfg, **kwargs: received.append(cfg))
    config = validate_run_config(
        {
            "model": {"name": "rcf"},
            "datasets": ["m-eurosat"],
            "runtime": {"device": "auto"},
        }
    )
    _image_runtime.run(config)
    assert received[0].device == expected


@pytest.mark.parametrize(
    "model", ["torchgeo/scalemae_large_fmow", "torchgeo/resnet50_s2rgb_satlas_si"]
)
def test_dry_run_reload_preserves_preset_settings(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture[str], model: str
) -> None:
    from torchgeo_bench.commands import _image_runtime

    received = []
    monkeypatch.setattr(
        _image_runtime,
        "main",
        lambda cfg, **kwargs: received.append(OmegaConf.to_container(cfg, resolve=True)),
    )
    arguments = ["run", "--model", model, "--dataset", "m-eurosat", "--device", "cpu"]
    main(arguments)
    main([*arguments, "--dry-run"])
    restored = validate_run_config(yaml.safe_load(capsys.readouterr().out))
    _image_runtime.run(restored)
    assert received[0] == received[1]
    assert not restored.input.model_fields_set
    assert not restored.segmentation.model_fields_set


def test_image_size_override_reaches_model_construction(monkeypatch: MonkeyPatch) -> None:
    from torchgeo_bench.commands import _image_runtime
    from torchgeo_bench.main import resolve_model_config

    received = []
    monkeypatch.setattr(_image_runtime, "main", lambda cfg, **kwargs: received.append(cfg))
    main(
        [
            "run",
            "--model",
            "torchgeo/scalemae_large_fmow",
            "--dataset",
            "m-eurosat",
            "--device",
            "cpu",
            "--image-size",
            "32",
        ]
    )
    config = received[0]
    assert config.dataset.image_size == 32
    assert resolve_model_config(config.model, "m-eurosat").image_size == 32
