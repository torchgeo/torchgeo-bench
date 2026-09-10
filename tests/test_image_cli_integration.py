# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Offline end-to-end image benchmark smoke test."""

from pathlib import Path

import pandas as pd
import torch
import yaml
from _pytest.monkeypatch import MonkeyPatch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset

from torchgeo_bench.datasets import BandSpec
from torchgeo_bench.image_cli import main as cli_main


class FakeBench:
    """Minimal classification dataset metadata used by the real runner."""

    name = 'm-eurosat'
    task = 'classification'
    num_classes = 2
    multilabel = False
    rgb_bands = ['red', 'green', 'blue']
    bands = [BandSpec('fake', name, name, 0.5, 0.25, 0.0, 1.0) for name in rgb_bands]

    def select_band_specs(self, bands: tuple[str, ...] | None) -> list[BandSpec]:
        if bands is None:
            return self.bands
        return [self.bands[self.rgb_bands.index(name)] for name in bands]


def test_cli_runs_real_knn_and_resume(monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
    torch.manual_seed(0)
    images = torch.rand(12, 3, 8, 8)
    labels = torch.tensor([0, 1] * 6)
    dataset = TensorDataset(images, labels)

    calls = 0

    def fake_get_datasets(
        **kwargs: object,
    ) -> tuple[object, DataLoader, DataLoader, DataLoader]:
        nonlocal calls
        calls += 1
        assert kwargs['bands'] == ['red', 'green', 'blue']
        assert kwargs['image_size'] == 8
        assert kwargs['interpolation'] == 'nearest'

        class Samples(Dataset[dict[str, Tensor]]):
            def __len__(self) -> int:
                return len(dataset)

            def __getitem__(self, index: int) -> dict[str, Tensor]:
                image, label = dataset[index]
                return {'image': image, 'label': label}

        samples = Samples()

        def loader() -> DataLoader:
            return DataLoader(samples, batch_size=4, num_workers=0)

        return samples, loader(), loader(), loader()

    import torchgeo_bench.main as runner

    monkeypatch.setattr(runner, 'get_datasets', fake_get_datasets)
    monkeypatch.setattr(runner, 'get_bench_dataset_class', lambda _: FakeBench)
    output = tmp_path / 'results.csv'
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.safe_dump({'output': {'file': str(output)}}))
    args = [
        'run',
        '--config',
        str(config_path),
        '--model',
        'rcf',
        '--dataset',
        'm-eurosat',
        '--device',
        'cpu',
        '--image-size',
        '8',
        '--bands',
        'red,green,blue',
        '--interpolation',
        'nearest',
        '--batch-size',
        '4',
        '--workers',
        '0',
        '--methods',
        'knn',
        '--bootstrap-samples',
        '2',
        '--seed',
        '7',
    ]
    # RCF features still flow through the real instantiate, extraction, KNN,
    # result writer, and resume planner. Only data and registry access are fake.
    cli_main(args)
    rows = pd.read_csv(output)
    assert len(rows) == 1
    assert rows.loc[0, 'metric_value'] == rows.loc[0, 'metric_value']
    assert rows.loc[0, 'dataset'] == 'm-eurosat'
    assert rows.loc[0, 'normalization'] == 'bandspec_zscore'
    assert rows.loc[0, 'seed'] == 7
    assert calls == 1
    before = output.read_bytes()
    monkeypatch.setattr(
        runner, 'get_datasets', lambda **_: (_ for _ in ()).throw(AssertionError())
    )
    cli_main([*args, '--resume'])
    assert output.read_bytes() == before
