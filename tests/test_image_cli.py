# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Tests for the discoverable image CLI."""

from collections.abc import Sequence
from pathlib import Path

import pytest
import yaml
from omegaconf import DictConfig, OmegaConf, open_dict

from torchgeo_bench.config_schema import validate_run_config
from torchgeo_bench.image_cli import main


def test_dry_run_applies_explicit_flags_and_preserves_false_values(
    capsys: pytest.CaptureFixture[str],
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
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
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


@pytest.mark.parametrize("size", ["0", "-1", "not-an-integer"])
def test_invalid_image_size_is_rejected_at_cli(
    size: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as error:
        main(["run", "--model", "rcf", "--dataset", "m-eurosat", "--image-size", size])
    assert error.value.code == 2
    assert "--image-size" in capsys.readouterr().err


def test_boolean_flags_override_config(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
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


@pytest.mark.parametrize("selection", [["--dataset", "m-eurosat"], ["--model", "rcf"]])
def test_missing_model_or_dataset_fails_before_execution(selection: list[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(["run", *selection, "--dry-run"])
    assert error.value.code == 2


def test_non_dry_run_calls_legacy_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    received = []
    monkeypatch.setattr("torchgeo_bench.commands._image_runtime.run", received.append)
    main(["run", "--model", "rcf", "--dataset", "m-eurosat"])
    assert received[0].model.name == "rcf"


def test_legacy_adapter_composes_and_translates_schema(
    monkeypatch: pytest.MonkeyPatch,
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


def test_legacy_adapter_rejects_unavailable_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
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
    monkeypatch: pytest.MonkeyPatch,
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
    monkeypatch: pytest.MonkeyPatch,
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
    monkeypatch: pytest.MonkeyPatch,
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
    with pytest.raises(SystemExit) as error:
        main(["run", "--model", "rcf", "--dataset", "m-eurosat", "--methods", "linear"])
    assert error.value.code == 2


def test_unknown_config_field_fails_before_execution(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("model: {name: rcf}\ndatasets: [m-eurosat]\nrunntim: {}\n", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        main(["run", "--config", str(path), "--dry-run"])
    assert error.value.code == 2


def test_runtime_failure_propagates_from_image_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(_: object) -> None:
        raise FileNotFoundError("dataset missing")

    monkeypatch.setattr("torchgeo_bench.commands._image_runtime.run", fail)
    with pytest.raises(FileNotFoundError, match="dataset missing"):
        main(["run", "--model", "rcf", "--dataset", "m-eurosat"])


def test_nested_linear_override_preserves_sibling_values(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
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


def test_catalog_name_selection_and_invalid_requests(
    capsys: pytest.CaptureFixture[str],
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
    with pytest.raises(SystemExit) as error:
        main(["profile"])
    assert error.value.code == 2
    with pytest.raises(SystemExit, match="unknown model"):
        main(["models", "unknown"])
    with pytest.raises(SystemExit, match="unknown dataset"):
        main(["datasets", "unknown"])


@pytest.mark.parametrize(("model", "dataset"), [("unknown", "m-eurosat"), ("rcf", "unknown")])
def test_unknown_catalog_entries_fail_before_execution(model: str, dataset: str) -> None:
    with pytest.raises(SystemExit) as error:
        main(["run", "--model", model, "--dataset", dataset])
    assert error.value.code == 2


def test_config_help_is_available_without_selection(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as error:
        main(["run", "--config-help"])
    assert error.value.code == 0
    assert "title: RunConfig" in capsys.readouterr().out


@pytest.mark.parametrize("value", ["null", "3", "[]"])
def test_invalid_model_section_has_a_field_error(
    value: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(f"model: {value}\ndatasets: [m-eurosat]\n")
    with pytest.raises(SystemExit) as error:
        main(["run", "--config", str(path), "--dry-run"])
    assert error.value.code == 2
    assert "model" in capsys.readouterr().err


@pytest.mark.parametrize("flags", [["--methods", "linear"], ["--bands", "red,,blue"]])
def test_dry_run_rejects_unsupported_method_and_band_selections(flags: list[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(["run", "--model", "rcf", "--dataset", "m-eurosat", "--dry-run", *flags])
    assert error.value.code == 2


def test_yaml_bands_require_a_list(tmp_path: Path) -> None:
    path = tmp_path / "bands.yaml"
    path.write_text("model: {name: rcf}\ndatasets: [m-eurosat]\ninput:\n  bands: red,green\n")
    with pytest.raises(SystemExit) as error:
        main(["run", "--config", str(path), "--dry-run"])
    assert error.value.code == 2


def test_public_download_and_profile_dispatch(monkeypatch: pytest.MonkeyPatch) -> None:
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
    monkeypatch: pytest.MonkeyPatch, *, available: bool, expected: str
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
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], model: str
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


def test_image_size_override_reaches_model_construction(monkeypatch: pytest.MonkeyPatch) -> None:
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
