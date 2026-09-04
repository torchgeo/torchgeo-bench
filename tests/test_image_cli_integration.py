"""Offline end-to-end image benchmark smoke test."""

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from torchgeo_bench.cli import main as cli_main
from torchgeo_bench.datasets import BandSpec


class FakeBench:
    """Minimal classification dataset metadata used by the real runner."""

    name = 'm-eurosat'
    task = 'classification'
    num_classes = 2
    multilabel = False
    rgb_bands = ['red', 'green', 'blue']
    bands = [
        BandSpec('fake', name, name, 0.5, 0.25, 0.0, 1.0)
        for name in rgb_bands
    ]

    def select_band_specs(self, bands: tuple[str, ...] | None) -> list[BandSpec]:
        if bands is None:
            return self.bands
        return [self.bands[self.rgb_bands.index(name)] for name in bands]


def test_cli_runs_real_knn_and_resume(monkeypatch, tmp_path: Path) -> None:
    torch.manual_seed(0)
    images = torch.rand(12, 3, 8, 8)
    labels = torch.tensor([0, 1] * 6)
    dataset = TensorDataset(images, labels)

    def fake_get_datasets(**kwargs: object) -> tuple[object, DataLoader, DataLoader, DataLoader]:
        class Samples:
            def __len__(self) -> int:
                return len(dataset)

            def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
                image, label = dataset[index]
                return {'image': image, 'label': label}

        samples = Samples()
        loader = lambda: DataLoader(samples, batch_size=4, num_workers=0)
        return samples, loader(), loader(), loader()

    import torchgeo_bench.main as runner

    monkeypatch.setattr(runner, 'get_datasets', fake_get_datasets)
    monkeypatch.setattr(runner, 'get_bench_dataset_class', lambda _: FakeBench)
    output = tmp_path / 'results.csv'
    args = [
        'run', '-m', 'rcf', '-d', 'm-eurosat', '--device', 'cpu', '--image-size', '8',
        '--batch-size', '4', '--bootstrap', '2', '--output', str(output),
        'dataset.num_workers=0', 'eval.skip_linear=true',
    ]
    # RCF features still flow through the real instantiate, extraction, KNN,
    # result writer, and resume planner. Only data and registry access are fake.
    cli_main(args)
    rows = pd.read_csv(output)
    assert len(rows) == 1
    assert rows.loc[0, 'metric_value'] == rows.loc[0, 'metric_value']
    assert rows.loc[0, 'dataset'] == 'm-eurosat'
    assert rows.loc[0, 'normalization'] == 'bandspec_zscore'
    before = output.read_bytes()
    cli_main([*args, '--resume'])
    assert output.read_bytes() == before
