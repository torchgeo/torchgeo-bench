"""Behavioral coverage for experiment scripts using the library config API."""

import importlib
import json
from pathlib import Path
from types import ModuleType
from unittest import mock

import pytest
import torch
import yaml

from experiments.scripts import (
    audit_model_native,
    introspect_seg_layers,
    tune_dataloader,
)
from tests.support.runner import _synthetic_splits
from torchgeo_bench.config.presets import ModelPreset, load_model_preset, resolve_run_config
from torchgeo_bench.config.run import RunConfig
from torchgeo_bench.config.schema import InputConfig, ModelConfig
from torchgeo_bench.models._normalization import UnsupportedNormalizationError

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("script", [audit_model_native, introspect_seg_layers])
def test_analysis_scripts_preserve_requested_rgb_order(script: ModuleType) -> None:
    bands = script.band_specs("m-eurosat", "rgb")
    assert [band.name for band in bands] == ["red", "green", "blue"]


@pytest.mark.parametrize("script", [audit_model_native, introspect_seg_layers])
@pytest.mark.parametrize(("dataset", "names"), [("caffe", ["gray"]), ("kuro_siwo", ["vv", "vh"])])
def test_analysis_scripts_require_explicit_non_rgb_selection(
    script: ModuleType, dataset: str, names: list[str]
) -> None:
    assert [band.name for band in script.band_specs(dataset, "default")] == names
    with pytest.raises(ValueError, match="no genuine RGB"):
        script.band_specs(dataset, "rgb")
    with pytest.raises(ValueError, match="Unknown band selection"):
        script.band_specs(dataset, "typo")


def test_main_study_explicitly_selects_non_rgb_without_overriding_all_band_presets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(ROOT / "experiments"))
    script = importlib.import_module("experiments.run_main_experiments")
    monkeypatch.setattr(script, "MODELS", ["timm/resnet18", "torchgeo/dofa_base"])

    def load(selection: ModelConfig, **kwargs) -> ModelPreset:
        preset = load_model_preset(selection, **kwargs)
        if selection.name == "torchgeo/dofa_base":
            return preset.model_copy(update={"input": InputConfig(bands="all")})
        return preset

    monkeypatch.setattr("torchgeo_bench.config.presets.load_model_preset", load)
    jobs = script.build_jobs()
    selections = {}
    for job in jobs:
        for dataset in job.config.datasets:
            effective, _ = resolve_run_config(job.config, dataset)
            key = (job.config.model.name, dataset)
            assert key not in selections
            selections[key] = effective.input.bands
    for dataset in ("caffe", "kuro_siwo"):
        assert selections[("timm/resnet18", dataset)] == "default"
        assert selections[("torchgeo/dofa_base", dataset)] == "all"
    assert selections[("timm/resnet18", "m-eurosat")] == "rgb"
    assert selections[("torchgeo/dofa_base", "m-eurosat")] == "all"


def test_tuner_builds_packaged_model_config() -> None:
    bands = audit_model_native.band_specs("m-eurosat", "rgb")
    model = tune_dataloader._build_model("rcf", bands)
    features = model(torch.zeros(1, 3, 16, 16))
    preset = load_model_preset(ModelConfig(name="rcf"))
    assert features.shape == (1, preset.kwargs["features"])
    assert torch.isfinite(features).all()


@pytest.mark.parametrize(
    "error",
    [
        UnsupportedNormalizationError("no pretraining pipeline"),
        ValueError("invalid model_native configuration"),
    ],
)
def test_native_audit_only_classifies_unsupported_normalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "rcf.yaml").write_text("target: torchgeo_bench.models.RCFBench\nname: rcf\n")
    output = tmp_path / "audit.json"
    monkeypatch.setattr("torchgeo_bench.config.catalog.CONF_DIR", tmp_path)
    monkeypatch.setattr("sys.argv", ["audit_model_native.py", "--out", str(output)])

    def fail(config: object, **kwargs: object) -> None:
        raise error

    monkeypatch.setattr(audit_model_native, "build_model", fail)
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
    monkeypatch: pytest.MonkeyPatch, error: RuntimeError, message: str
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
    monkeypatch.setattr(
        tune_dataloader, "load_split", lambda *args, **kwargs: _synthetic_splits()[0]
    )
    monkeypatch.setattr(tune_dataloader, "_build_model", lambda *args: torch.nn.Identity())

    def fail(*args: object) -> None:
        raise error

    monkeypatch.setattr(tune_dataloader, "_bench", fail)
    with pytest.raises(RuntimeError, match=message):
        tune_dataloader.main()


def test_segmentation_introspection_handles_transformer_tokens() -> None:
    class _TokenBackbone(torch.nn.Module):
        num_prefix_tokens = 1

        def __init__(self) -> None:
            super().__init__()
            self.blocks = torch.nn.ModuleList([torch.nn.Identity()])

        def forward(self, images: torch.Tensor) -> torch.Tensor:
            tokens = images.flatten(2).transpose(1, 2)
            prefix = torch.zeros_like(tokens[:, :1])
            return self.blocks[0](torch.cat([prefix, tokens], dim=1))

    assert introspect_seg_layers.measure(_TokenBackbone(), size=8) == {"blocks.0": (8, 8)}


def test_tuner_preserves_dataset_overrides_and_empirical_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "rcf.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "rcf",
                "target": "torchgeo_bench.models.RCFBench",
                "seed_from_run": True,
                "kwargs": {"mode": "empirical", "features": 512},
                "dataset_overrides": {"m-eurosat": {"kwargs": {"features": 8}}},
            }
        )
    )
    monkeypatch.setattr("torchgeo_bench.config.catalog.CONF_DIR", tmp_path)
    bands = audit_model_native.band_specs("m-eurosat", "rgb")
    dataset = torch.utils.data.TensorDataset(torch.zeros(1, 3, 8, 8))

    def build(preset: ModelPreset, **kwargs: object) -> torch.nn.Module:
        assert preset.kwargs["features"] == 8
        assert preset.kwargs["seed"] == 0
        assert kwargs["dataset"] is dataset
        assert kwargs["bands"] == bands
        assert kwargs["normalization"] == "bandspec_zscore"
        return torch.nn.Identity()

    monkeypatch.setattr(tune_dataloader, "build_model", build)
    assert isinstance(
        tune_dataloader._build_model("rcf", bands, "m-eurosat", dataset), torch.nn.Identity
    )


@pytest.mark.parametrize(
    "filename",
    ["run_c_sweep_experiment", "run_effect_of_lbfgs_vs_adam"],
)
def test_inprocess_experiments_use_typed_constructors(
    monkeypatch: pytest.MonkeyPatch, filename: str
) -> None:
    monkeypatch.syspath_prepend(str(ROOT / "experiments"))
    script = importlib.import_module(f"experiments.{filename}")
    bands = audit_model_native.band_specs("m-eurosat", "rgb")
    for name, selection in script.MODEL_CONFIGS.items():
        _, preset = resolve_run_config(
            RunConfig(model=selection, datasets=["m-eurosat"]), "m-eurosat"
        )
        assert preset.kwargs["pretrained"] is True
        assert preset.kwargs["global_pool"] == "avg"
        assert preset.kwargs.get("auto_resize", False) == (
            filename == "run_effect_of_lbfgs_vs_adam" and name == "dinov3sat"
        )
    model = script.instantiate_model(
        ModelConfig(
            name="custom-rcf",
            target="torchgeo_bench.models.RCFBench",
            kwargs={"features": 8, "seed": 17},
        ),
        bands,
        "m-eurosat",
    )
    assert model(torch.zeros(1, 3, 16, 16)).shape == (1, 8)


@pytest.mark.parametrize("filename", ["run_c_sweep_experiment", "run_effect_of_lbfgs_vs_adam"])
def test_experiments_build_models_from_loaded_metadata(
    monkeypatch: pytest.MonkeyPatch, filename: str
) -> None:
    monkeypatch.syspath_prepend(str(ROOT / "experiments"))
    script = importlib.import_module(f"experiments.{filename}")
    splits = _synthetic_splits()
    monkeypatch.setattr(script, "MODEL_CONFIGS", {"rcf": ModelConfig(name="rcf")})

    def build(config: ModelConfig, bands: list, dataset_name: str) -> torch.nn.Module:
        assert dataset_name == "m-eurosat"
        assert all(a is b for a, b in zip(bands, splits[0].bands, strict=True))
        return torch.nn.Identity()

    monkeypatch.setattr(script, "instantiate_model", build)
    if filename == "run_c_sweep_experiment":
        execute = script.run_dataset_sweep
        arguments = ("m-eurosat", torch.device("cpu"), [])
    else:
        execute = script.run_dataset
        arguments = (
            "m-eurosat",
            [{"solver": "lbfgs", "C": 1.0, "lr": 1.0}],
            torch.device("cpu"),
            [],
        )
    with (
        mock.patch.object(script, "load_split", side_effect=splits) as load,
        mock.patch.object(script, "DataLoader", wraps=script.DataLoader) as loaders,
        mock.patch.object(
            script, "extract_features", side_effect=RuntimeError("reached extraction")
        ),
        pytest.raises(RuntimeError, match="reached extraction"),
    ):
        execute(*arguments)
    assert [call.args[1] for call in load.call_args_list] == ["train", "val", "test"]
    assert loaders.call_count == 3
    for index, call in enumerate(loaders.call_args_list):
        assert call.args[0] is splits[index].dataset
        assert call.kwargs == {
            "batch_size": 64,
            "num_workers": 8,
            "shuffle": index == 0,
            "pin_memory": torch.cuda.is_available(),
        }


def test_tuner_uses_train_only_metadata_and_keeps_batching_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train = _synthetic_splits()[0]
    monkeypatch.setattr(
        "sys.argv",
        [
            "tune_dataloader.py",
            "--model",
            "rcf",
            "--dataset",
            "m-eurosat",
            "--bands",
            "rgb",
            "--batch-sizes",
            "2",
            "--num-workers",
            "0",
            "--device",
            "cpu",
        ],
    )
    with (
        mock.patch.object(tune_dataloader, "load_split", return_value=train) as load,
        mock.patch.object(
            tune_dataloader, "_build_model", return_value=torch.nn.Identity()
        ) as build,
        mock.patch.object(tune_dataloader, "_bench", return_value=(1.0, 0.0, 1.0)) as measure,
    ):
        tune_dataloader.main()
    load.assert_called_once_with("m-eurosat", "train", bands="rgb")
    assert build.call_args.args[3] is train.dataset
    assert all(a is b for a, b in zip(build.call_args.args[1], train.bands, strict=True))
    loader = measure.call_args.args[1]
    assert loader.dataset is train.dataset
    assert loader.batch_size == 2
    assert loader.num_workers == 0
    assert not loader.pin_memory
    assert not loader.persistent_workers


def test_model_variants_preserve_constructor_options_and_preset_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(ROOT / "experiments"))
    script = importlib.import_module("experiments.run_cls_token_experiment")
    monkeypatch.setattr(script, "MODELS", ["timm/vit/vit_large_patch16_dinov3sat"])
    monkeypatch.setattr(script, "DATASETS", ["m-eurosat"])
    jobs = script.build_jobs()
    assert len(jobs) == 2
    for job, use_cls in zip(jobs, [False, True], strict=True):
        config, preset = resolve_run_config(job.config, "m-eurosat")
        assert preset.name.endswith("_cls" if use_cls else "_avg")
        assert preset.kwargs["use_cls_token"] is use_cls
        assert preset.kwargs["model_name"] == "vit_large_patch16_dinov3.sat493m"
        assert preset.kwargs["pretrained"] is True
        assert preset.kwargs["auto_resize"] is True
        assert config.segmentation.layers == ["blocks.23", "blocks.17", "blocks.11", "blocks.5"]
        assert RunConfig.model_validate(config.model_dump_yaml()) == config


def test_resize_experiment_keeps_all_normalization_policies_and_null_resize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.syspath_prepend(str(ROOT / "experiments"))
    script = importlib.import_module("experiments.run_resize_and_normalization_experiment")
    jobs = script.build_jobs()
    assert len(jobs) == 4 * (1 + 4 * 3)
    for normalization in script.NORMALIZATIONS:
        matching = [
            job.config
            for job in jobs
            if job.config.model.kwargs["input_normalization"] == normalization
        ]
        assert len(matching) == 13
        assert sum(config.input.image_size is None for config in matching) == 1
        for config in matching:
            assert config.model.target == "torchgeo_bench.models.TimmPatchBenchModel"
            assert config.model.name == f"resnet18_{normalization}"
            assert config.model.kwargs["model_name"] == "resnet18"
            assert config.segmentation.layers == ["layer4", "layer3", "layer2", "layer1"]
            assert config.classification.linear.refit_train_val is False
            assert config.runtime.verbose is False
            if config.input.image_size is None:
                assert config.input.interpolation == "bilinear"
