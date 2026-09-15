"""Unit tests for dataset classes that don't require real data on disk."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from torchgeo_bench.datasets import ResolvedInput, V2Source, get_dataset_spec, load_split
from torchgeo_bench.datasets.geobench_v2 import canonicalize_sample
from torchgeo_bench.datasets.torchgeo import load_torchgeo_split


def _canonicalize(name: str, sample: dict) -> dict:
    source = get_dataset_spec(name).source
    assert isinstance(source, V2Source)
    return canonicalize_sample(sample, source=source)


class TestFOTWCanonicalize:
    def test_existing_canonical_image_is_preserved(self) -> None:
        image = torch.zeros(3, 8, 8)
        image_b = torch.ones_like(image)
        sample = {"image": image, "image_b": image_b, "label": 0}
        result = _canonicalize("fotw", sample)
        assert result["image"] is image
        assert result["label"] == 0

    @pytest.mark.parametrize("has_image_a", [False, True])
    def test_image_b_becomes_image(self, *, has_image_a: bool) -> None:
        img_b = torch.ones(3, 8, 8)
        sample = {"image_b": img_b, "label": 1}
        if has_image_a:
            sample["image_a"] = torch.zeros_like(img_b)
        result = _canonicalize("fotw", sample)
        assert result["image"] is img_b
        assert result["label"] == 1
        assert "image_a" not in result
        assert "image_b" not in result


@pytest.mark.parametrize("dataset_name", ["spacenet2", "spacenet7"])
class TestSpaceNetCanonicalize:
    def test_mask_offset_and_reserved_zero(self, dataset_name: str) -> None:
        mask = torch.tensor([[0, 1, 2], [2, 1, 0]])
        image = torch.zeros(3, 2, 3)
        result = _canonicalize(dataset_name, {"image": image, "mask": mask})
        torch.testing.assert_close(result["mask"], torch.tensor([[0, 0, 1], [1, 0, 0]]))
        assert result["image"] is image
        assert int(result["mask"].max()) < get_dataset_spec(dataset_name).num_classes

    def test_inference_sample_without_mask(self, dataset_name: str) -> None:
        image = torch.zeros(3, 4, 4)
        result = _canonicalize(dataset_name, {"image": image})
        assert set(result) == {"image"}
        assert result["image"] is image


@pytest.mark.parametrize("dataset_name", ["eurosat", "eurosat-spatial"])
def test_eurosat_rejects_unknown_split(dataset_name: str) -> None:
    with pytest.raises(ValueError, match="Unknown split"):
        load_split(dataset_name, "invalid")


@pytest.mark.parametrize("dataset_name", ["eurosat", "eurosat-spatial"])
@pytest.mark.parametrize("split", ["train", "val", "test"])
@pytest.mark.parametrize(
    ("bands", "expected_codes"),
    [
        (("blue", "red", "green"), ("B02", "B04", "B03")),
        (
            None,
            (
                "B01",
                "B02",
                "B03",
                "B04",
                "B05",
                "B06",
                "B07",
                "B08",
                "B09",
                "B10",
                "B11",
                "B12",
                "B8A",
            ),
        ),
    ],
)
def test_eurosat_forwards_bands_split_and_transform(
    monkeypatch: pytest.MonkeyPatch,
    dataset_name: str,
    split: str,
    bands: tuple[str, ...] | None,
    expected_codes: tuple[str, ...],
) -> None:
    upstream = MagicMock()
    transform = torch.nn.Identity()
    bench = get_dataset_spec(dataset_name)
    monkeypatch.setattr(f"torchgeo.datasets.{bench.source.upstream_class}", upstream)
    result = load_torchgeo_split(
        bench,
        split,
        inputs=ResolvedInput(tuple(bench.select_band_specs(bands)), bands or "all"),
        transform=transform,
    )
    upstream.assert_called_once_with(
        root=str(Path("data/eurosat")),
        split=split,
        bands=expected_codes,
        transforms=transform,
        download=False,
    )
    assert result is upstream.return_value
