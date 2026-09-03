"""Tests for the per-sample compute-cost (GFLOPs) pipeline.

The fast tests build tiny synthetic modules so they run on CPU without
downloading anything.  Tests that need a real pretrained backbone are marked
``slow`` (``-m slow`` to include), matching the repo-wide convention.
"""

import math

import pytest
import torch
from torch import nn

from torchgeo_bench.datasets import get_bench_dataset_class
from torchgeo_bench.flops_pipeline import (
    _MODALITY_FOR_BAND_CONFIG,
    _is_band_incompatibility,
    _load_completed,
    _n_tokens,
    _probe_gflops,
    _seg_head_gflops,
)
from torchgeo_bench.model_profile import _count_gflops
from torchgeo_bench.segmentation_task import build_seg_probe_and_solver

CPU = torch.device("cpu")


class _TinyConvNet(nn.Module):
    """Healthy model — never trips module_tracker's grad_fn assertion."""

    def __init__(self, in_ch: int = 3, width: int = 8) -> None:
        super().__init__()
        self.stem = nn.Conv2d(in_ch, width, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(width, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.stem(x)).flatten(1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Determinism and scaling
# ---------------------------------------------------------------------------


def test_gflops_independent_of_batch_size():
    """``_count_gflops`` slices ``sample[:1]``, so GFLOPs is per-sample
    regardless of the batch handed in."""
    model = _TinyConvNet().eval()
    one = _count_gflops(model, torch.randn(1, 3, 32, 32))
    many = _count_gflops(model, torch.randn(16, 3, 32, 32))
    assert one == many


def test_gflops_scales_with_resolution():
    """Doubling each spatial dim quadruples conv FLOPs."""
    model = _TinyConvNet().eval()
    small = _count_gflops(model, torch.randn(1, 3, 32, 32))
    large = _count_gflops(model, torch.randn(1, 3, 64, 64))
    assert large == pytest.approx(4 * small, rel=0.01)


def test_channel_count_changes_only_the_stem():
    """C=3 vs C=12 differ only in stem cost, so the delta is exactly the extra
    stem MACs — far smaller than the total."""
    x3, x12 = torch.randn(1, 3, 32, 32), torch.randn(1, 12, 32, 32)
    g3 = _count_gflops(_TinyConvNet(in_ch=3).eval(), x3)
    g12 = _count_gflops(_TinyConvNet(in_ch=12).eval(), x12)
    assert g12 > g3
    # stem is the only channel-dependent layer: 12/3 = 4x its cost
    assert g12 == pytest.approx(4 * g3, rel=0.05)


# ---------------------------------------------------------------------------
# Probe capacity confound
# ---------------------------------------------------------------------------


class _WidthModel(nn.Module):
    """Emits a fixed-width embedding, standing in for a pooled backbone."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.proj = nn.Linear(3, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.mean(dim=(-2, -1)))


def test_probe_scales_with_feature_dim_not_classes():
    """A ``pool='both'`` model hands the probe a 2x wider vector; the linear
    probe's cost and params scale with it."""
    narrow = _WidthModel(1024).eval()
    wide = _WidthModel(2048).eval()

    g_n, p_n, d_n = _probe_gflops(narrow, 3, 32, CPU, "linear", 10)
    g_w, p_w, d_w = _probe_gflops(wide, 3, 32, CPU, "linear", 10)

    assert (d_n, d_w) == (1024, 2048)
    assert g_w == pytest.approx(2 * g_n, rel=0.01)
    assert p_w == pytest.approx(2 * p_n, rel=0.01)


def test_mlp_probe_scales_as_feature_dim_squared():
    """``linear.py`` builds ``Linear(D, D, bias=False)`` — the confound is D^2,
    over the *feature* dim, not the class count."""
    narrow = _WidthModel(512).eval()
    wide = _WidthModel(1024).eval()

    g_n, _, _ = _probe_gflops(narrow, 3, 32, CPU, "mlp", 10)
    g_w, _, _ = _probe_gflops(wide, 3, 32, CPU, "mlp", 10)

    # The D x D projection dominates the D x n_classes classifier, so doubling
    # D roughly quadruples the probe.
    assert g_w / g_n == pytest.approx(4.0, rel=0.05)


# ---------------------------------------------------------------------------
# n_tokens
# ---------------------------------------------------------------------------


class _PatchModel(nn.Module):
    def __init__(self, patch: int) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 8, kernel_size=patch, stride=patch)
        self.patch_embed.patch_size = (patch, patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.patch_embed(x).flatten(2).transpose(1, 2)


def test_n_tokens_tracks_image_size_over_patch_squared():
    model = _PatchModel(16)
    assert _n_tokens(model, 224) == (224 // 16) ** 2 == 196
    assert _n_tokens(model, 112) == (112 // 16) ** 2 == 49


def test_n_tokens_is_none_for_cnns():
    assert _n_tokens(_TinyConvNet(), 224) is None


# ---------------------------------------------------------------------------
# Segmentation head
# ---------------------------------------------------------------------------


class _TapModel(nn.Module):
    """Backbone exposing named spatial layers at a chosen tap resolution."""

    def __init__(self, in_ch: int = 12, tap_stride: int = 16) -> None:
        super().__init__()
        # SegmentationProbe._dry_run_channels reads `num_channels` off the
        # backbone to size its probe tensor (defaulting to 3).
        self.num_channels = in_ch
        # Four isotropic taps — DPTHead requires exactly 4 feature layers.
        self.layer1 = nn.Conv2d(in_ch, 32, kernel_size=tap_stride, stride=tap_stride)
        self.layer2 = nn.Conv2d(32, 32, kernel_size=1)
        self.layer3 = nn.Conv2d(32, 32, kernel_size=1)
        self.layer4 = nn.Conv2d(32, 32, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return x.mean(dim=(-2, -1))


_TAP_LAYERS = ["layer4", "layer3", "layer2", "layer1"]


def _seg_cfg(layers: list[str], head_type: str):
    from omegaconf import OmegaConf

    return OmegaConf.create(
        {
            "segmentation": {
                "layers": layers,
                "head_type": head_type,
                "lr_scheduler": "cosine",
                "criterion": {
                    "_target_": "torch.nn.CrossEntropyLoss",
                    "ignore_index": 255,
                },
            }
        }
    )


@pytest.mark.parametrize("head_type", ["fpn", "dpt"])
def test_seg_head_gflops_is_positive_and_deterministic(head_type):
    if head_type == "dpt":
        pytest.importorskip("transformers")
    model = _TapModel().eval()
    probe, _ = build_seg_probe_and_solver(model, 4, _seg_cfg(_TAP_LAYERS, head_type), CPU, 1e-3)
    probe.eval()
    first = _seg_head_gflops(probe, 12, 224, CPU)
    second = _seg_head_gflops(probe, 12, 224, CPU)
    assert first > 0
    assert first == second


def test_finer_taps_make_a_more_expensive_head():
    """Head cost is driven by finest tap resolution — a backbone property —
    not by class count.  A stride-8 tap is 4x the pixels of a stride-16 one.
    """
    coarse = _TapModel(tap_stride=16).eval()
    fine = _TapModel(tap_stride=8).eval()

    probe_c, _ = build_seg_probe_and_solver(coarse, 4, _seg_cfg(_TAP_LAYERS, "fpn"), CPU, 1e-3)
    probe_f, _ = build_seg_probe_and_solver(fine, 4, _seg_cfg(_TAP_LAYERS, "fpn"), CPU, 1e-3)
    g_coarse = _seg_head_gflops(probe_c.eval(), 12, 224, CPU)
    g_fine = _seg_head_gflops(probe_f.eval(), 12, 224, CPU)
    assert g_fine > g_coarse


def test_num_classes_barely_moves_head_cost():
    """Sub-2% for 2 -> 15 classes, which is why num_classes is not an axis."""
    model = _TapModel().eval()
    probe2, _ = build_seg_probe_and_solver(model, 2, _seg_cfg(_TAP_LAYERS, "fpn"), CPU, 1e-3)
    probe15, _ = build_seg_probe_and_solver(model, 15, _seg_cfg(_TAP_LAYERS, "fpn"), CPU, 1e-3)
    g2 = _seg_head_gflops(probe2.eval(), 12, 224, CPU)
    g15 = _seg_head_gflops(probe15.eval(), 12, 224, CPU)
    assert abs(g15 - g2) / g2 < 0.02


# ---------------------------------------------------------------------------
# Band configs and the TerraMind modality agreement
# ---------------------------------------------------------------------------


def test_band_configs_come_from_cloudsen12_class_attributes():
    """Band specs are read off class attributes — no instantiation of data,
    no download."""
    bench = get_bench_dataset_class("cloudsen12")()
    rgb = bench.select_band_specs(bench.rgb_bands)
    s2 = bench.select_band_specs(None)

    assert len(rgb) == 3
    assert len(s2) == 12
    # cloudsen12's 12 bands are the canonical S2 set: all sensor "s2", unlike
    # so2sat's 10 S2 + 2 SAR.
    assert {b.sensor for b in s2} == {"s2"}
    assert [b.name for b in rgb] == ["b04", "b03", "b02"]


def test_terramind_modality_map_matches_shipped_configs():
    """Assert no cell can pair an RGB modality with a 12-channel tensor.

    This is the one failure mode that yields a plausible-looking wrong number
    instead of an exception, so it is pinned against the actual config files.
    """
    from torchgeo_bench.config import compose_config

    for config_name, band_config in [
        ("terratorch/terramind_v1_base", "s2"),
        ("terratorch/terramind_v1_base_rgb", "rgb"),
        ("terratorch/terramind_v1_large", "s2"),
        ("terratorch/terramind_v1_large_rgb", "rgb"),
    ]:
        cfg = compose_config([f"model={config_name}"])
        assert str(cfg.model.modality) == _MODALITY_FOR_BAND_CONFIG[band_config]


# ---------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------


def test_load_completed_missing_file(tmp_path):
    assert _load_completed(str(tmp_path / "nope.csv")) == frozenset()


def test_load_completed_reads_cell_keys(tmp_path):
    import pandas as pd

    path = tmp_path / "compute_cost.csv"
    pd.DataFrame(
        [
            {"name": "resnet50", "band_config": "rgb", "task": "classification", "head_type": None},
            {"name": "resnet50", "band_config": "s2", "task": "segmentation", "head_type": "fpn"},
        ]
    ).to_csv(path, index=False)

    completed = _load_completed(str(path))
    assert ("resnet50", "rgb", "classification", "") in completed
    assert ("resnet50", "s2", "segmentation", "fpn") in completed
    assert ("resnet50", "s2", "segmentation", "dpt") not in completed


def test_load_completed_tolerates_malformed_csv(tmp_path):
    """A partial/garbled CSV must not abort the sweep."""
    path = tmp_path / "broken.csv"
    path.write_text("this,is not\na valid;;csv\n")
    assert _load_completed(str(path)) == frozenset()


# ---------------------------------------------------------------------------
# Band-incompatibility detection
# ---------------------------------------------------------------------------


def _wrap(exc: BaseException, depth: int = 2) -> BaseException:
    """Nest *exc* in `depth` layers of wrapper exceptions."""
    out: BaseException = exc
    for i in range(depth):
        try:
            raise RuntimeError(f"instantiation failed (layer {i})") from out
        except RuntimeError as wrapper:
            out = wrapper
    return out


def test_band_incompatibility_detected_through_wrapping():
    """A wrapped band ValueError is still found by walking the chain."""
    inner = ValueError("Missing required model band 'nir'. Available canonical bands: ...")
    assert _is_band_incompatibility(_wrap(inner)) is inner


def test_band_incompatibility_matches_empty_intersection():
    inner = ValueError("select_src_bands: none of the target bands ['blue'] are present.")
    assert _is_band_incompatibility(_wrap(inner)) is inner


def test_interpolation_key_error_is_not_a_band_incompatibility():
    """omegaconf's InterpolationKeyError subclasses ValueError, so an
    unresolved ``${seed}`` must not be classified as a band incompatibility."""
    from omegaconf.errors import InterpolationKeyError

    assert issubclass(InterpolationKeyError, ValueError)  # why the narrowing is needed
    exc = InterpolationKeyError("Interpolation key 'seed' not found")
    assert _is_band_incompatibility(_wrap(exc)) is None


@pytest.mark.parametrize(
    "exc",
    [
        ValueError("Could not find weights for 'tt_clay_v1_5'"),
        OSError("[Errno 122] Disk quota exceeded"),
        RuntimeError("CUDA error: out of memory"),
        ValueError("invalid literal for int() with base 10: 'x'"),
    ],
)
def test_real_failures_are_not_swallowed(exc):
    """A missing checkpoint or a full disk is a failure, not a skip."""
    assert _is_band_incompatibility(_wrap(exc)) is None


def test_band_incompatibility_survives_a_cyclic_cause_chain():
    """``__cause__ or __context__`` can cycle when an exception is raised while
    handling itself; the walk must terminate rather than spin."""
    a = ValueError("Missing required model band 'swir1'")
    b = RuntimeError("wrapper")
    a.__context__ = b
    b.__context__ = a
    assert _is_band_incompatibility(b) is a


def test_band_incompatibility_terminates_on_unmatched_cycle():
    a = RuntimeError("one")
    b = RuntimeError("two")
    a.__context__ = b
    b.__context__ = a
    assert _is_band_incompatibility(a) is None


def test_band_markers_match_what_band_mapping_actually_raises():
    """The markers are matched by message, so they are pinned against the real
    raise sites.  A reword there without a matching update here would silently
    widen the sweep's blind spot back out."""
    from torchgeo_bench.datasets.base import BandSpec
    from torchgeo_bench.models._band_mapping import map_to_model_bands, select_src_bands

    rgb = [
        BandSpec(sensor="s2", name=n, source_name=n.upper(), mean=0.0, std=1.0, min=0.0, max=1.0)
        for n in ("red", "green", "blue")
    ]

    with pytest.raises(ValueError) as missing:
        map_to_model_bands(torch.zeros(1, 3, 4, 4), rgb, ["blue", "green", "red", "nir"])
    assert _is_band_incompatibility(missing.value) is missing.value

    with pytest.raises(ValueError) as empty:
        select_src_bands(rgb, ["swir1", "swir2"])
    assert _is_band_incompatibility(empty.value) is empty.value


def test_channel_count_disagreement_is_a_bug_not_a_band_skip():
    """map_to_model_bands' own caller assertion — a tensor whose channel count
    disagrees with its BandSpecs — means the *pipeline* is wrong, not the model.
    It must propagate rather than be logged as a routine band skip."""
    from torchgeo_bench.datasets.base import BandSpec
    from torchgeo_bench.models._band_mapping import map_to_model_bands

    rgb = [
        BandSpec(sensor="s2", name=n, source_name=n.upper(), mean=0.0, std=1.0, min=0.0, max=1.0)
        for n in ("red", "green", "blue")
    ]
    with pytest.raises(ValueError) as mismatch:
        map_to_model_bands(torch.zeros(1, 7, 4, 4), rgb, ["red", "green", "blue"])
    assert "channels but" in str(mismatch.value)
    assert _is_band_incompatibility(mismatch.value) is None


def test_terramind_modality_mismatch_is_a_band_incompatibility():
    """_build_model raises this itself for an S2L2A/rgb pairing."""
    exc = ValueError(
        "TerraMind modality 'S2L2A' does not match band config 'rgb' "
        "(3 channels, expects 'RGB'). Measuring this pair would map through "
        "the wrong band table."
    )
    assert _is_band_incompatibility(exc) is exc


def test_flops_config_resolves_every_shipped_model_config():
    """conf/model/rcf.yaml carries ``seed: ${seed}``, which raises
    InterpolationKeyError unless flops_config defines a top-level ``seed``,
    so the sweep's own config is checked here instead of in the job."""
    from omegaconf import OmegaConf

    from torchgeo_bench.config import compose_config

    cfg = compose_config(["model=rcf"], config_name="flops_config", default_model=None)
    resolved = OmegaConf.to_container(cfg, resolve=True)  # raises if unresolvable
    assert resolved["model"]["seed"] == resolved["seed"] == 0


# ---------------------------------------------------------------------------
# Real backbones (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_panopticon_yields_finite_gflops_without_patching_torch():
    """Panopticon is measurable through the public FLOP counter."""
    from torchgeo_bench.config import compose_config, instantiate

    bench = get_bench_dataset_class("cloudsen12")()
    cfg = compose_config(["model=torchgeo/panopticon"])
    model = instantiate(
        cfg.model,
        bands=bench.select_band_specs(None),
        normalization="bandspec_zscore",
    ).eval()

    gflops = _count_gflops(model, torch.randn(1, 12, 224, 224))

    assert math.isfinite(gflops)
    assert gflops > 0


@pytest.mark.slow
def test_vit_gflops_ordering_and_tokens():
    """Sanity ordering: ViT-L > ViT-B, and n_tokens tracks (size/patch)^2."""
    from torchgeo_bench.config import compose_config, instantiate

    bench = get_bench_dataset_class("cloudsen12")()
    rgb = bench.select_band_specs(bench.rgb_bands)

    def build(name):
        cfg = compose_config([f"model={name}"])
        return instantiate(cfg.model, bands=rgb, normalization="bandspec_zscore").eval()

    base = build("timm/vit/vit_base_patch16_224")
    large = build("timm/vit/vit_large_patch16_224")
    x = torch.randn(1, 3, 224, 224)

    assert _count_gflops(large, x) > _count_gflops(base, x)
    assert _n_tokens(base, 224) == 196
