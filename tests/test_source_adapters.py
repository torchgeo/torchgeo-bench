"""Canonical contracts over complete source loading paths, without source I/O."""

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import geobench_v2.datasets as upstream
import pytest
import torch
import torch.nn.functional as F

from torchgeo_bench.datasets import (
    DatasetCapabilities,
    DatasetSpec,
    TorchGeoSource,
    V2Source,
    get_dataset_spec,
    load_split,
)


def _source(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    sample: dict,
    order: dict[str, list[str]] | list[str] | None = None,
) -> MagicMock:
    source = get_dataset_spec(name).source
    backend = MagicMock()
    backend.dataset_band_config = getattr(upstream, source.upstream_class).dataset_band_config

    def construct(**kwargs: object) -> MagicMock:
        inner = MagicMock()
        inner.band_order = order if order is not None else kwargs["band_order"]
        inner.__getitem__.side_effect = lambda index: deepcopy(sample)
        inner.__len__.return_value = 2
        return inner

    backend.side_effect = construct
    monkeypatch.setattr(f"geobench_v2.datasets.{source.upstream_class}", backend)
    return backend


@pytest.mark.parametrize("size", [None, 3, 7])
def test_grayscale_raw_dtype_targets_and_auxiliary_metadata(
    monkeypatch: pytest.MonkeyPatch,
    size: int | None,
) -> None:
    image = torch.tensor([[[-8.0, 400.5, 255.0], [2048.0, 1024.5, 0.0]]], dtype=torch.float64)
    mask = torch.tensor([[[-1, 255, 4], [3, 2, 1]]], dtype=torch.int16)
    auxiliary = torch.tensor([[[91.0]]])
    _source(
        monkeypatch,
        "caffe",
        {
            "image": image,
            "mask": mask,
            "sample_id": "tile-123",
            "dates": [1, 5],
            "image_preview": auxiliary,
        },
    )
    loaded = load_split("caffe", "val", image_size=size)
    sample = loaded.dataset[0]
    expected_image = image.float()
    expected_mask = mask[0].long()
    if size is not None:
        expected_image = F.interpolate(
            expected_image[None], (size, size), mode="bilinear", align_corners=False
        )[0]
        expected_mask = F.interpolate(mask.float()[None], (size, size), mode="nearest")[0, 0].long()
    torch.testing.assert_close(sample["image"], expected_image)
    torch.testing.assert_close(sample["mask"], expected_mask)
    torch.testing.assert_close(sample["image_preview"], auxiliary)
    assert sample["sample_id"] == "tile-123"
    assert sample["dates"] == [1, 5]
    assert loaded.bands[0] is get_dataset_spec("caffe").bands[0]
    assert sample["image"].dtype == torch.float32
    assert sample["mask"].dtype == torch.long


@pytest.mark.parametrize("name", ["spacenet2", "spacenet7"])
def test_spacenet_offset_precedes_nearest_mask_resize(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    image = torch.full((3, 2, 3), 1200, dtype=torch.int16)
    sample = {
        "image_worldview" if name == "spacenet2" else "image": image,
        "mask": torch.tensor([[[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]]]),
    }
    _source(monkeypatch, name, sample)
    result = load_split(name, "train", image_size=6).dataset[0]
    expected = torch.tensor([[0, 0, 1], [1, 0, 0]]).repeat_interleave(3, 0).repeat_interleave(2, 1)
    torch.testing.assert_close(result["mask"], expected)
    assert result["image"].dtype == torch.float32


def test_fotw_never_falls_back_to_earlier_or_generic_image(monkeypatch: pytest.MonkeyPatch) -> None:
    image_b = torch.full((3, 2, 2), 5000, dtype=torch.int32)
    _source(
        monkeypatch,
        "fotw",
        {
            "image_a": image_b - 100,
            "image_b": image_b,
            "image": image_b + 100,
            "mask": torch.ones(2, 2),
        },
    )
    sample = load_split("fotw", "train").dataset[0]
    assert set(sample) == {"image", "mask"}
    torch.testing.assert_close(sample["image"], image_b.float())


@pytest.mark.parametrize(
    ("name", "bands", "sample", "missing"),
    [
        ("fotw", "rgb", {"image_a": torch.ones(3, 2, 2)}, "image_b"),
        ("kuro_siwo", ("dem", "vv"), {"image_post": torch.ones(1, 2, 2)}, "image_dem"),
        ("kuro_siwo", ("dem", "vv"), {"image_dem": torch.ones(1, 2, 2)}, "image_post"),
        ("pastis", ("b04", "vv_asc"), {"image_s2": torch.ones(1, 2, 2)}, "image_s1_asc"),
    ],
)
def test_missing_required_components_fail_explicitly(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    bands: str | tuple[str, ...],
    sample: dict,
    missing: str,
) -> None:
    _source(monkeypatch, name, {**sample, "mask": torch.zeros(2, 2)})
    loaded = load_split(name, "train", bands=bands, image_size=4)
    with pytest.raises(ValueError, match=missing):
        loaded.dataset[0]


@pytest.mark.parametrize("value", [0.5, 1.0000000001, float("nan"), float("inf"), complex(1, 1)])
@pytest.mark.parametrize("name", ["caffe", "spacenet7", "forestnet"])
def test_nonintegral_targets_are_rejected_before_resize_or_label_offset(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: object,
) -> None:
    spec = get_dataset_spec(name)
    image = torch.ones(len(spec.rgb_bands), 2, 2)
    target = [[0, value], [0, 0]] if spec.task == "segmentation" else value
    _source(monkeypatch, name, {"image": image, spec.target_key: target})
    loaded = load_split(name, "train", image_size=1)
    with pytest.raises(ValueError, match="integer"):
        loaded.dataset[0]


@pytest.mark.parametrize("mask", [torch.zeros(2, 2, 2), torch.zeros(1, 1, 2, 2), torch.tensor(1)])
def test_mask_only_supports_hw_or_singleton_channel(
    monkeypatch: pytest.MonkeyPatch,
    mask: torch.Tensor,
) -> None:
    _source(monkeypatch, "caffe", {"image": torch.zeros(1, 2, 2), "mask": mask})
    with pytest.raises(ValueError, match="H,W mask"):
        load_split("caffe", "train").dataset[0]


def test_nearest_masks_do_not_round_large_integer_ids(monkeypatch: pytest.MonkeyPatch) -> None:
    mask = torch.tensor([[2**53 + 1, -1], [255, 0]])
    _source(monkeypatch, "caffe", {"image": torch.zeros(1, 2, 2), "mask": mask})
    sample = load_split("caffe", "train", image_size=4).dataset[0]
    torch.testing.assert_close(sample["mask"], mask.repeat_interleave(2, 0).repeat_interleave(2, 1))


@pytest.mark.parametrize("name", ["benv2", "treesatai"])
@pytest.mark.parametrize("malformed", [False, True])
def test_multilabel_vectors_follow_actual_metadata(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    *,
    malformed: bool,
) -> None:
    spec = get_dataset_spec(name)
    assert spec.multilabel
    label = torch.arange(spec.num_classes, dtype=torch.int8) % 2
    if malformed:
        label = label[:-1]
    images = {"image_s2" if name == "benv2" else "image_aerial": torch.ones(3, 2, 2)}
    _source(monkeypatch, name, {**images, "label": label})
    loaded = load_split(name, "train")
    if malformed:
        with pytest.raises(ValueError, match=f"vector of length {spec.num_classes}"):
            loaded.dataset[0]
    else:
        torch.testing.assert_close(loaded.dataset[0]["label"], label.float())


@pytest.mark.parametrize(
    ("name", "shape", "steps", "message"),
    [
        ("caffe", (1, 1, 2, 2), None, "CHW"),
        ("pastis", (3, 2, 2), 3, "TCHW"),
        ("pastis", (2, 3, 2, 2), 3, "3 time steps"),
        ("pastis", (1, 3, 2, 2), 1, "CHW"),
    ],
)
def test_exact_resolved_temporal_layout_is_enforced(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    shape: tuple[int, ...],
    steps: int | None,
    message: str,
) -> None:
    key = "image_s2" if name == "pastis" else "image"
    _source(monkeypatch, name, {key: torch.zeros(shape), "mask": torch.zeros(2, 2)})
    with pytest.raises(ValueError, match=message):
        load_split(name, "train", time_steps=steps).dataset[0]


def test_actual_backend_within_sensor_band_order_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    _source(
        monkeypatch,
        "benv2",
        {
            "image_s1": torch.tensor([7.0, 3.0])[:, None, None],
            "image_s2": torch.tensor([2000.0, 1000.0])[:, None, None],
            "label": torch.zeros(19),
        },
        order={"s1": ["VH", "VV"], "s2": ["B03", "B04"]},
    )
    loaded = load_split("benv2", "train", bands=("b04", "vv", "b03", "vh"))
    torch.testing.assert_close(
        loaded.dataset[0]["image"][:, 0, 0], torch.tensor([1000.0, 3.0, 2000.0, 7.0])
    )
    assert tuple(b.name for b in loaded.bands) == ("b04", "vv", "b03", "vh")


@pytest.mark.parametrize(
    "spec",
    [
        replace(get_dataset_spec("caffe"), source=None),
        replace(get_dataset_spec("caffe"), source=V2Source("UnknownReader")),
        replace(get_dataset_spec("caffe"), source=V2Source("GeoBenchCaFFe", time_step=("pre",))),
        replace(
            get_dataset_spec("caffe"), source=V2Source("GeoBenchCaFFe", band_order_strategy="wrong")
        ),
        replace(
            get_dataset_spec("caffe"), capabilities=DatasetCapabilities(supports_partitions=True)
        ),
        replace(get_dataset_spec("caffe"), capabilities=DatasetCapabilities(multi_temporal=True)),
        replace(get_dataset_spec("pastis"), capabilities=DatasetCapabilities()),
        replace(get_dataset_spec("caffe"), capabilities=DatasetCapabilities(multi_temporal=1)),
        replace(
            get_dataset_spec("eurosat"), source=TorchGeoSource("UnknownReader", "data/eurosat")
        ),
        replace(
            get_dataset_spec("kuro_siwo"),
            source=replace(get_dataset_spec("kuro_siwo").source, time_step=("pre_1",)),
        ),
    ],
)
def test_invalid_source_policies_fail_before_construction_or_io(spec: DatasetSpec) -> None:
    with (
        patch("torchgeo_bench.datasets.loading._load_source", side_effect=AssertionError("source")),
        patch.object(Path, "exists", side_effect=AssertionError("I/O")),
        pytest.raises((TypeError, ValueError)),
    ):
        load_split(spec, "train")


@pytest.mark.parametrize("name", ["caffe", "pastis", "benv2"])
def test_v2_loading_has_no_sample_probes_or_unrelated_splits(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    backend = _source(monkeypatch, name, {})
    loaded = load_split(name, "train")
    backend.assert_called_once()
    assert backend.call_args.kwargs["split"] == "train"
    assert backend.call_args.kwargs["transforms"] is None
    assert backend.call_args.kwargs["download"] is False
    assert len(loaded.dataset) == 2
    loaded.dataset._inner.__getitem__.assert_not_called()


@pytest.mark.parametrize("interpolation", ["area", "bicubic", "bilinear", "nearest"])
def test_sensor_alignment_uses_the_original_direct_output_grid(
    monkeypatch: pytest.MonkeyPatch,
    interpolation: str,
) -> None:
    aerial = (torch.arange(49).reshape(1, 7, 7).float() % 3) * 100
    sentinel = (torch.arange(9).reshape(1, 3, 3).float() % 2) * 1000
    _source(
        monkeypatch,
        "treesatai",
        {
            "image_aerial": aerial,
            "image_s2": sentinel,
            "label": torch.zeros(get_dataset_spec("treesatai").num_classes),
        },
    )
    loaded = load_split(
        "treesatai", "train", bands=("b04", "red"), image_size=5, interpolation=interpolation
    )
    expected = torch.cat(
        [
            F.interpolate(
                image[None],
                size=(5, 5),
                mode=interpolation,
                align_corners=False if interpolation in ("bilinear", "bicubic") else None,
            )[0]
            for image in (sentinel, aerial)
        ]
    )
    torch.testing.assert_close(loaded.dataset[0]["image"], expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="image_size"):
        load_split("treesatai", "train", bands=("b04", "red")).dataset[0]


@pytest.mark.parametrize("name", ["kuro_siwo", "benv2"])
def test_sources_without_grid_alignment_do_not_gain_it(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    sample = (
        {
            "image_post": torch.zeros(1, 2, 2),
            "image_dem": torch.zeros(1, 4, 4),
            "mask": torch.zeros(2, 2),
        }
        if name == "kuro_siwo"
        else {
            "image_s2": torch.zeros(1, 2, 2),
            "image_s1": torch.zeros(1, 4, 4),
            "label": torch.zeros(19),
        }
    )
    bands = ("vv", "dem") if name == "kuro_siwo" else ("b04", "vv")
    _source(monkeypatch, name, sample)
    with pytest.raises(ValueError, match="must already match"):
        load_split(name, "train", bands=bands, image_size=8).dataset[0]


@pytest.mark.parametrize("name", ["caffe", "pastis", "eurosat", "resisc45"])
def test_unknown_source_band_fails_before_constructor_or_io(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
) -> None:
    import torchgeo.datasets as torchgeo

    original = get_dataset_spec(name)
    spec = replace(
        original,
        bands=(replace(original.bands[0], source_name="invalid"),),
        rgb_bands=(original.bands[0].name,),
    )
    module = upstream if isinstance(spec.source, V2Source) else torchgeo
    cls = getattr(module, spec.source.upstream_class)
    constructor = MagicMock()
    if isinstance(spec.source, V2Source):
        constructor.dataset_band_config = cls.dataset_band_config
    elif name == "eurosat":
        constructor.all_band_names = cls.all_band_names
    monkeypatch.setattr(module, spec.source.upstream_class, constructor)
    with (
        patch.object(Path, "exists", side_effect=AssertionError("I/O")),
        pytest.raises(ValueError, match="source bands"),
    ):
        load_split(spec, "train")
    constructor.assert_not_called()
