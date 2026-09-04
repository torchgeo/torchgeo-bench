# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Tests for the standalone profile command."""

import argparse
import json

import torch
from torch import nn

from torchgeo_bench.commands import _profile_runtime


class _Band:
    def __init__(self, name: str) -> None:
        self.name = name


class _Dataset:
    rgb_bands = (_Band("red"), _Band("green"), _Band("blue"))

    def select_band_specs(self, bands):
        return self.rgb_bands if bands is None else tuple(bands)


class _Loader:
    def __iter__(self):
        yield {"image": torch.ones(4, 3, 8, 8)}


def test_profile_emits_real_batch_metadata(monkeypatch, capsys) -> None:
    monkeypatch.setattr(_profile_runtime, "get_bench_dataset_class", lambda _: _Dataset)
    monkeypatch.setattr(_profile_runtime, "get_datasets", lambda **_: (None, _Loader(), None, None))
    monkeypatch.setattr(_profile_runtime, "compose_config", lambda _: argparse.Namespace(model={}))
    monkeypatch.setattr(_profile_runtime, "instantiate", lambda *_, **__: nn.Conv2d(3, 3, 1))
    args = argparse.Namespace(
        model="toy",
        dataset="toy-dataset",
        device="cpu",
        bands="rgb",
        image_size=8,
        batch_size=4,
        warmup=0,
        measurements=1,
        precision="float32",
        count_flops=False,
    )

    _profile_runtime.run(args)
    record = json.loads(capsys.readouterr().out)
    assert record["model"] == "toy"
    assert record["bands"] == ["red", "green", "blue"]
    assert record["input_shape"] == [4, 3, 8, 8]
    assert record["profile"]["batch_size"] == 4
    assert record["profile"]["flops"]["status"] == "disabled"
    assert record["scope"].startswith("encoder inference")
