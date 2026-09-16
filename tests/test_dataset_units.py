"""Complete source paths enforce raw canonical images and targets."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torchgeo.datasets
from torchgeo.datasets import EuroSAT

from tests.support.data import geotiff_bytes
from torchgeo_bench.datasets import get_dataset_spec, load_split


@pytest.mark.parametrize("dataset_name", ["eurosat", "eurosat-spatial"])
@pytest.mark.parametrize("split", ["train", "val", "test"])
@pytest.mark.parametrize("bands", [None, ("blue", "red", "green")])
def test_eurosat_explicit_source_and_canonical_sample(
    monkeypatch: pytest.MonkeyPatch,
    dataset_name: str,
    split: str,
    bands: tuple[str, ...] | None,
) -> None:
    spec = get_dataset_spec(dataset_name)
    selected = spec.select_band_specs(bands)
    source_image = torch.arange(len(selected), dtype=torch.int16)[:, None, None].expand(-1, 2, 2)
    upstream = MagicMock()
    upstream.all_band_names = EuroSAT.all_band_names
    upstream.return_value.__getitem__.side_effect = lambda index: upstream.call_args.kwargs[
        "transforms"
    ]({"image": source_image, "label": 7, "sample_id": "original"})
    monkeypatch.setattr(f"torchgeo.datasets.{spec.source.upstream_class}", upstream)
    loaded = load_split(spec, split, bands=bands, image_size=4)
    options = upstream.call_args.kwargs
    assert options["root"] == "data/eurosat"
    assert options["split"] == split
    assert options["bands"] == tuple(b.source_name for b in selected)
    assert options["download"] is False
    sample = loaded.dataset[0]
    torch.testing.assert_close(
        sample["image"], source_image.float().repeat_interleave(2, -2).repeat_interleave(2, -1)
    )
    assert sample["label"].dtype == torch.long
    assert sample["label"].shape == ()
    assert sample["label"].item() == 7
    assert sample["sample_id"] == "original"
    assert all(a is b for a, b in zip(loaded.bands, selected, strict=True))


@pytest.mark.parametrize("dataset_name", ["eurosat", "eurosat-spatial", "resisc45"])
@pytest.mark.parametrize("label", [1.25, [1], float("nan"), float("inf")])
def test_torchgeo_rejects_malformed_labels(
    monkeypatch: pytest.MonkeyPatch,
    dataset_name: str,
    label: object,
) -> None:
    spec = get_dataset_spec(dataset_name)
    upstream = MagicMock()
    upstream.all_band_names = EuroSAT.all_band_names
    upstream.return_value.__getitem__.side_effect = lambda index: upstream.call_args.kwargs[
        "transforms"
    ]({"image": torch.zeros(3, 4, 4), "label": label})
    monkeypatch.setattr(f"torchgeo.datasets.{spec.source.upstream_class}", upstream)
    loaded = load_split(spec, "train")
    with pytest.raises(ValueError, match=r"integer|scalar"):
        loaded.dataset[0]


@pytest.mark.parametrize("dataset_name", ["eurosat", "eurosat-spatial"])
@pytest.mark.parametrize("bands", ["all", ("nir", "red", "nir", "blue")])
def test_installed_torchgeo_train_only_source_keeps_channel_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dataset_name: str,
    bands: str | tuple[str, ...],
) -> None:
    spec = get_dataset_spec(dataset_name)
    cls = getattr(torchgeo.datasets, spec.source.upstream_class)
    root = tmp_path / "data/eurosat"
    directory = root / cls.base_dir / "AnnualCrop"
    directory.mkdir(parents=True)
    image = np.broadcast_to(
        (1000 + 512 * np.arange(13, dtype=np.uint16))[:, None, None], (13, 2, 3)
    )
    (directory / "AnnualCrop_train.tif").write_bytes(geotiff_bytes(image))
    (root / cls.split_filenames["train"]).write_text("AnnualCrop_train.jpg\n")
    monkeypatch.chdir(tmp_path)
    with patch.object(cls, "_load_image", side_effect=AssertionError("sample probe")):
        loaded = load_split(spec, "train", bands=bands, image_size=4)
    assert len(loaded.dataset) == 1
    sample = loaded.dataset[0]
    values = [1000 + 512 * cls.all_band_names.index(b.source_name) for b in loaded.bands]
    torch.testing.assert_close(
        sample["image"], torch.tensor(values).float()[:, None, None].expand(-1, 4, 4)
    )
    assert sample["label"].shape == ()
    assert sample["label"].dtype == torch.int64
    assert sample["label"].item() == 0
    assert all(b is spec.bands[spec.bands.index(b)] for b in loaded.bands)
    with pytest.raises(FileNotFoundError, match="split 'val'"):
        load_split(spec, "val")
