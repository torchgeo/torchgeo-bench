# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Tests for the standalone profile command."""

import argparse
import hashlib
import json
import sys
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.utils.data import Dataset

from torchgeo_bench import commands
from torchgeo_bench.commands import _profile_runtime
from torchgeo_bench.commands._profile import profile
from torchgeo_bench.main import resolve_model_config


class _Band:
    def __init__(self, name: str) -> None:
        self.name = name


class _Dataset:
    rgb_bands = (_Band("red"), _Band("green"), _Band("blue"))

    def select_band_specs(self, bands: tuple[_Band, ...] | None) -> tuple[_Band, ...]:
        return self.rgb_bands if bands is None else tuple(bands)


class _Loader:
    def __init__(self, batch: torch.Tensor | None = None) -> None:
        self.batch = torch.ones(4, 3, 8, 8) if batch is None else batch

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        yield {"image": self.batch}


class _ImageDataset(Dataset):
    def __init__(self, images: torch.Tensor) -> None:
        self.images = images

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"image": self.images[index]}


@pytest.fixture
def profile_args() -> argparse.Namespace:
    return argparse.Namespace(
        model="toy",
        dataset="toy-dataset",
        partition="default",
        device="cpu",
        bands="rgb",
        image_size=8,
        interpolation="bilinear",
        batch_size=4,
        warmup=0,
        measurements=1,
        precision="float32",
        count_flops=False,
        seed=0,
        normalization="bandspec_zscore",
    )


@pytest.mark.parametrize(
    (
        "model_data",
        "image_size",
        "interpolation",
        "expected_size",
        "expected_interpolation",
    ),
    [
        (
            {"dataset_overrides": {"toy": {"image_size": 96}}},
            None,
            None,
            96,
            "bilinear",
        ),
        ({}, None, None, 224, "bilinear"),
        (
            {"dataset_overrides": {"toy": {"image_size": 96}}},
            128,
            "nearest",
            128,
            "nearest",
        ),
    ],
)
def test_input_settings_precedence(
    model_data: dict[str, object],
    image_size: int | None,
    interpolation: str | None,
    expected_size: int,
    expected_interpolation: str,
) -> None:
    cfg = OmegaConf.create({"dataset": {"image_size": 224, "interpolation": "bilinear"}})
    model_cfg = resolve_model_config(OmegaConf.create(model_data), "toy")
    args = argparse.Namespace(
        image_size=image_size, interpolation=interpolation, normalization=None
    )

    effective = _profile_runtime._resolve_input_settings(args, cfg, model_cfg)

    assert effective[:2] == (expected_size, expected_interpolation)


def test_input_settings_preserve_model_normalization() -> None:
    cfg = OmegaConf.create({"dataset": {"image_size": 224, "interpolation": "bilinear"}})
    model_cfg = resolve_model_config(OmegaConf.create({"input_normalization": "imagenet"}), "toy")
    args = argparse.Namespace(image_size=None, interpolation=None, normalization=None)

    effective = _profile_runtime._resolve_input_settings(args, cfg, model_cfg)

    assert effective[2:] == ("bandspec_zscore", "imagenet")


def test_profile_emits_real_batch_metadata(
    profile_args: argparse.Namespace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(_profile_runtime, "get_bench_dataset_class", lambda _: _Dataset)
    monkeypatch.setattr(_profile_runtime, "list_model_configs", lambda: ["toy"])
    monkeypatch.setattr(_profile_runtime, "list_datasets", lambda: ["toy-dataset"])
    monkeypatch.setattr(_profile_runtime, "get_datasets", lambda **_: (None, _Loader(), None, None))
    monkeypatch.setattr(
        _profile_runtime, "compose_config", lambda _: OmegaConf.create({"model": {}})
    )
    monkeypatch.setattr(
        _profile_runtime,
        "resolve_model_config",
        lambda model, dataset: OmegaConf.create({}),
    )
    monkeypatch.setattr(_profile_runtime, "instantiate", lambda *_, **__: nn.Conv2d(3, 3, 1))
    _profile_runtime.run(profile_args)
    record = json.loads(capsys.readouterr().out)
    assert record["model"] == "toy"
    assert record["bands"] == ["red", "green", "blue"]
    assert record["input_shape"] == [4, 3, 8, 8]
    assert record["profile"]["batch_size"] == 4
    assert record["profile"]["flops"]["status"] == "disabled"
    assert record["scope"].startswith("encoder inference")


@pytest.mark.parametrize("image_size", [None, 32])
def test_profile_applies_image_size_to_model_and_loader(
    profile_args: argparse.Namespace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    image_size: int | None,
) -> None:
    profile_args.model = "torchgeo/scalemae_large_fmow"
    profile_args.dataset = "m-eurosat"
    profile_args.image_size = image_size
    profile_args.interpolation = None
    profile_args.normalization = "identity"
    expected_size = 64 if image_size is None else image_size
    inputs: dict[str, Any] = {}
    construction: dict[str, Any] = {}

    def load(**kwargs: Any) -> tuple:
        inputs.update(kwargs)
        size = kwargs["image_size"]
        batch = torch.ones(kwargs["batch_size"], 3, size, size)
        return _ImageDataset(batch), _Loader(batch), None, None

    def build(model_cfg: DictConfig, **kwargs: Any) -> nn.Module:
        construction["config"] = OmegaConf.to_container(model_cfg, resolve=True)
        construction["options"] = kwargs
        return nn.Conv2d(3, 2, 1)

    monkeypatch.setattr(_profile_runtime, "get_datasets", load)
    monkeypatch.setattr(_profile_runtime, "instantiate", build)

    _profile_runtime.run(profile_args)

    record = json.loads(capsys.readouterr().out)
    assert inputs["image_size"] == construction["config"]["image_size"] == expected_size
    assert record["image_size"] == record["model_config"]["image_size"] == expected_size
    assert record["input_shape"] == [4, 3, expected_size, expected_size]
    assert record["profile"]["input_shape"] == record["input_shape"]
    assert inputs["interpolation"] == record["interpolation"] == "area"
    assert inputs["partition_name"] == profile_args.partition
    assert construction["config"]["res"] == record["model_config"]["res"] == 3.5
    assert construction["config"]["pool"] == record["model_config"]["pool"] == "cls"
    assert construction["config"]["auto_resize"] is False
    assert construction["options"]["normalization"] == record["normalization"] == "identity"
    assert record["input_normalization"] == "identity"
    assert [band.name for band in construction["options"]["bands"]] == record["bands"]
    assert "dataset" not in construction["options"]
    assert "dataset_overrides" not in record["model_config"]
    assert (
        record["model_config_hash"]
        == hashlib.sha256(json.dumps(record["model_config"], sort_keys=True).encode()).hexdigest()
    )


@pytest.mark.parametrize("seed", [7, 19])
def test_profile_resolves_requested_seed_before_building_rcf(
    profile_args: argparse.Namespace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    seed: int,
) -> None:
    profile_args.model = "rcf"
    profile_args.dataset = "m-eurosat"
    profile_args.seed = seed
    models: list[nn.Module] = []
    instantiate = _profile_runtime.instantiate

    def build(model_cfg: DictConfig, **kwargs: Any) -> nn.Module:
        model = instantiate(model_cfg, **kwargs)
        models.append(model)
        return model

    monkeypatch.setattr(_profile_runtime, "get_datasets", lambda **_: (None, _Loader(), None, None))
    monkeypatch.setattr(_profile_runtime, "instantiate", build)

    _profile_runtime.run(profile_args)

    record = json.loads(capsys.readouterr().out)
    assert record["seed"] == record["model_config"]["seed"] == seed
    assert "image_size" not in record["model_config"]
    expected_filters = torch.randn(256, 3, 3, 3, generator=torch.Generator().manual_seed(seed))
    torch.testing.assert_close(models[0].rcf.weights, expected_filters, rtol=0, atol=0)


def test_profile_supplies_selected_training_dataset_to_empirical_rcf(
    profile_args: argparse.Namespace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    profile_args.model = "rcf"
    profile_args.dataset = "m-eurosat"
    profile_args.seed = 23
    batch = torch.arange(4 * 3 * 8 * 8, dtype=torch.float32).reshape(4, 3, 8, 8)
    train_dataset = _ImageDataset(batch)
    compose_config = _profile_runtime.compose_config
    instantiate = _profile_runtime.instantiate
    construction: dict[str, Any] = {}

    def empirical_config(overrides: list[str]) -> DictConfig:
        cfg = compose_config(overrides)
        cfg.model.mode = "empirical"
        cfg.model.features = 8
        return cfg

    def build(model_cfg: DictConfig, **kwargs: Any) -> nn.Module:
        construction.update(kwargs)
        return instantiate(model_cfg, **kwargs)

    monkeypatch.setattr(_profile_runtime, "compose_config", empirical_config)
    monkeypatch.setattr(
        _profile_runtime,
        "get_datasets",
        lambda **_: (train_dataset, _Loader(batch), None, None),
    )
    monkeypatch.setattr(_profile_runtime, "instantiate", build)

    _profile_runtime.run(profile_args)

    record = json.loads(capsys.readouterr().out)
    assert construction["dataset"] is train_dataset
    assert record["model_config"]["mode"] == "empirical"
    assert record["model_config"]["features"] == 8
    assert record["model_config"]["seed"] == 23
    assert record["profile"]["throughput_samples_per_sec"] > 0


def test_profile_dispatches_shared_lazy_runtime(
    profile_args: argparse.Namespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[argparse.Namespace] = []
    monkeypatch.setattr(commands, "_profile_runtime", SimpleNamespace(run=calls.append))

    profile(profile_args)

    assert calls == [profile_args]


def test_profile_dispatches_without_a_runtime_stub_export(
    profile_args: argparse.Namespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delattr(commands, "_profile_runtime")
    monkeypatch.delitem(sys.modules, "torchgeo_bench.commands._profile_runtime")
    profile_args.batch_size = 0

    with pytest.raises(SystemExit, match="batch-size must be positive"):
        profile(profile_args)
