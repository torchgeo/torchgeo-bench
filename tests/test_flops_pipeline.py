"""Tests for the per-sample compute-cost (GFLOPs) pipeline.

Fast tests run on CPU without downloads; use ``-m slow`` for real pretrained backbones.
"""

import math

import pytest
import torch
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationKeyError
from torch import nn

from torchgeo_bench.bands import BandCompatibilityError
from torchgeo_bench.config import compose_config
from torchgeo_bench.datasets import get_bench_dataset_class
from torchgeo_bench.flops_pipeline import (
    _MODALITY_FOR_BAND_CONFIG,
    _build_model,
    _load_completed,
    _n_tokens,
    _probe_gflops,
    _seg_head_gflops,
    main,
)
from torchgeo_bench.model_profile import _count_gflops
from torchgeo_bench.segmentation_task import build_seg_probe_and_solver

CPU = torch.device("cpu")


class _TinyConvNet(nn.Module):
    """Small backbone with known convolution and linear costs."""

    def __init__(self, in_ch: int = 3, width: int = 8) -> None:
        super().__init__()
        self.stem = nn.Conv2d(in_ch, width, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(width, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.stem(x)).flatten(1)
        return self.fc(x)


class _GradBreakingNet(nn.Module):
    """Reproduce Panopticon's detached channel-fusion features."""

    def __init__(self, in_ch: int = 3) -> None:
        super().__init__()
        self.conv3d = nn.Conv3d(1, 4, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        broken = x.mean(dim=1, keepdim=True).detach()
        y = self.conv3d(broken.unsqueeze(1)).squeeze(2)
        return self.head(y.mean(dim=(-2, -1)))


def test_counting_does_not_patch_autograd():
    import torch.autograd.graph as autograd_graph
    import torch.utils.module_tracker as module_tracker

    before = autograd_graph.register_multi_grad_hook
    before_mt = module_tracker.register_multi_grad_hook
    model = _TinyConvNet().eval()

    def check_hooks(module, inputs):
        assert autograd_graph.register_multi_grad_hook is before
        assert module_tracker.register_multi_grad_hook is before_mt
        assert not torch.is_grad_enabled()

    with model.register_forward_pre_hook(check_hooks):
        assert _count_gflops(model, torch.randn(2, 3, 32, 32)) > 0
    assert autograd_graph.register_multi_grad_hook is before
    assert module_tracker.register_multi_grad_hook is before_mt


def test_grad_breaking_model_is_measurable():
    model = _GradBreakingNet().eval()
    gflops = _count_gflops(model, torch.randn(2, 3, 32, 32))
    assert gflops == pytest.approx((2 * 4 * 32 * 32 * 9 + 2 * 4 * 2) / 1e9)


def test_gflops_independent_of_batch_size():
    model = _TinyConvNet().eval()
    one = _count_gflops(model, torch.randn(1, 3, 32, 32))
    many = _count_gflops(model, torch.randn(16, 3, 32, 32))
    assert one == many


def test_gflops_scales_with_resolution():
    """Doubling width and height quadruples convolution cost; the fixed-size classifier changes the total slightly."""
    model = _TinyConvNet().eval()
    small = _count_gflops(model, torch.randn(1, 3, 32, 32))
    large = _count_gflops(model, torch.randn(1, 3, 64, 64))
    assert large == pytest.approx(4 * small, rel=0.01)


def test_channel_count_changes_only_the_stem():
    """The first convolution dominates, so 3 to 12 input channels nearly quadruples total cost; the classifier is unchanged."""
    x3, x12 = torch.randn(1, 3, 32, 32), torch.randn(1, 12, 32, 32)
    g3 = _count_gflops(_TinyConvNet(in_ch=3).eval(), x3)
    g12 = _count_gflops(_TinyConvNet(in_ch=12).eval(), x12)
    assert g12 > g3
    assert g12 == pytest.approx(4 * g3, rel=0.05)


class _WidthModel(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.proj = nn.Linear(3, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x.mean(dim=(-2, -1)))


def test_probe_scales_with_feature_dim_not_classes():
    """Doubling embedding width, as ``pool='both'`` does, should double linear-probe cost."""
    narrow = _WidthModel(1024).eval()
    wide = _WidthModel(2048).eval()

    g_n, p_n, d_n = _probe_gflops(narrow, 3, 32, CPU, "linear", 10)
    g_w, p_w, d_w = _probe_gflops(wide, 3, 32, CPU, "linear", 10)

    assert (d_n, d_w) == (1024, 2048)
    assert g_w == pytest.approx(2 * g_n, rel=0.01)
    assert p_w == pytest.approx(2 * p_n, rel=0.01)


def test_mlp_probe_scales_as_feature_dim_squared():
    """The D-by-D projection dominates the class-output layer, so doubling width almost quadruples cost."""
    narrow = _WidthModel(512).eval()
    wide = _WidthModel(1024).eval()

    g_n, _, _ = _probe_gflops(narrow, 3, 32, CPU, "mlp", 10)
    g_w, _, _ = _probe_gflops(wide, 3, 32, CPU, "mlp", 10)

    assert g_w / g_n == pytest.approx(4.0, rel=0.05)


def test_unknown_probe_head_is_rejected():
    with pytest.raises(ValueError, match="Unknown probe head"):
        _probe_gflops(_WidthModel(8).eval(), 3, 32, CPU, "typo", 10)


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


class _TapModel(nn.Module):
    def __init__(self, in_ch: int = 12, tap_stride: int = 16) -> None:
        super().__init__()
        # The channel probe uses num_channels to size its input.
        self.num_channels = in_ch
        # DPT requires exactly four feature layers.
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
    """Halving feature stride gives the head four times as many pixels at the same class count."""
    coarse = _TapModel(tap_stride=16).eval()
    fine = _TapModel(tap_stride=8).eval()

    probe_c, _ = build_seg_probe_and_solver(coarse, 4, _seg_cfg(_TAP_LAYERS, "fpn"), CPU, 1e-3)
    probe_f, _ = build_seg_probe_and_solver(fine, 4, _seg_cfg(_TAP_LAYERS, "fpn"), CPU, 1e-3)
    g_coarse = _seg_head_gflops(probe_c.eval(), 12, 224, CPU)
    g_fine = _seg_head_gflops(probe_f.eval(), 12, 224, CPU)
    assert g_fine > g_coarse


def test_num_classes_barely_moves_head_cost():
    """Class count is not varied in the sweep because it barely changes head cost."""
    model = _TapModel().eval()
    probe2, _ = build_seg_probe_and_solver(model, 2, _seg_cfg(_TAP_LAYERS, "fpn"), CPU, 1e-3)
    probe15, _ = build_seg_probe_and_solver(model, 15, _seg_cfg(_TAP_LAYERS, "fpn"), CPU, 1e-3)
    g2 = _seg_head_gflops(probe2.eval(), 12, 224, CPU)
    g15 = _seg_head_gflops(probe15.eval(), 12, 224, CPU)
    assert abs(g15 - g2) / g2 < 0.02


def test_band_configs_come_from_cloudsen12_class_attributes():
    """Read representative RGB/S2 band metadata without loading data."""
    bench = get_bench_dataset_class("cloudsen12")()
    rgb = bench.select_band_specs(bench.rgb_bands)
    s2 = bench.select_band_specs(None)

    assert len(rgb) == 3
    assert len(s2) == 12
    # CloudSen12 supplies 12 optical S2 bands; So2Sat mixes S2 and SAR.
    assert {b.sensor for b in s2} == {"s2"}
    assert [b.name for b in rgb] == ["b04", "b03", "b02"]


def test_terramind_modality_map_matches_shipped_configs():
    """An RGB modality paired with 12 channels can produce plausible but wrong FLOP counts."""
    from torchgeo_bench.config import compose_config

    for config_name, band_config in [
        ("terratorch/terramind_v1_base", "s2"),
        ("terratorch/terramind_v1_base_rgb", "rgb"),
        ("terratorch/terramind_v1_large", "s2"),
        ("terratorch/terramind_v1_large_rgb", "rgb"),
    ]:
        cfg = compose_config([f"model={config_name}"])
        assert str(cfg.model.modality) == _MODALITY_FOR_BAND_CONFIG[band_config]


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


def test_load_completed_rejects_invalid_schema(tmp_path):
    path = tmp_path / "broken.csv"
    path.write_text("this,is not\na valid;;csv\n")
    with pytest.raises(KeyError, match="name"):
        _load_completed(str(path))


def test_build_model_skips_explicit_band_incompatibility(monkeypatch, caplog):
    def incompatible(config, **kwargs):
        raise BandCompatibilityError("unsupported selection")

    monkeypatch.setattr("torchgeo_bench.flops_pipeline.instantiate", incompatible)
    cfg = OmegaConf.create({"_target_": "example.Model", "name": "example"})
    assert _build_model(cfg, [], "identity", "rgb") is None
    assert "unsupported selection" in caplog.text


@pytest.mark.parametrize(
    "error",
    [
        InterpolationKeyError("Interpolation key 'seed' not found"),
        ValueError("input channels configuration is invalid"),
        ValueError("Could not find weights for 'tt_clay_v1_5'"),
        OSError("[Errno 122] Disk quota exceeded"),
        RuntimeError("CUDA error: out of memory"),
    ],
)
def test_build_model_propagates_real_failures(error, monkeypatch):
    def broken(config, **kwargs):
        raise error

    monkeypatch.setattr("torchgeo_bench.flops_pipeline.instantiate", broken)
    cfg = OmegaConf.create({"_target_": "example.Model", "name": "example"})
    with pytest.raises(type(error)) as exc:
        _build_model(cfg, [], "identity", "rgb")
    assert exc.value is error


def test_missing_bands_raise_typed_incompatibility():
    from torchgeo_bench.datasets.base import BandSpec
    from torchgeo_bench.models._band_mapping import map_to_model_bands, select_src_bands

    rgb = [
        BandSpec(sensor="s2", name=n, source_name=n.upper(), mean=0.0, std=1.0, min=0.0, max=1.0)
        for n in ("red", "green", "blue")
    ]

    with pytest.raises(BandCompatibilityError):
        map_to_model_bands(torch.zeros(1, 3, 4, 4), rgb, ["blue", "green", "red", "nir"])

    with pytest.raises(BandCompatibilityError):
        select_src_bands(rgb, ["swir1", "swir2"])


def test_channel_count_disagreement_is_a_bug_not_a_band_skip():
    """A tensor/BandSpec channel mismatch is a pipeline error, not an unsupported model-band combination."""
    from torchgeo_bench.datasets.base import BandSpec
    from torchgeo_bench.models._band_mapping import map_to_model_bands

    rgb = [
        BandSpec(sensor="s2", name=n, source_name=n.upper(), mean=0.0, std=1.0, min=0.0, max=1.0)
        for n in ("red", "green", "blue")
    ]
    with pytest.raises(ValueError) as mismatch:
        map_to_model_bands(torch.zeros(1, 7, 4, 4), rgb, ["red", "green", "blue"])
    assert "channels but" in str(mismatch.value)
    assert not isinstance(mismatch.value, BandCompatibilityError)


def test_terramind_modality_mismatch_is_rejected():
    cfg = OmegaConf.create(
        {"_target_": "example.TerraMind", "name": "example", "modality": "S2L2A"}
    )
    with pytest.raises(ValueError, match="does not match band config"):
        _build_model(cfg, [], "identity", "rgb")


@pytest.fixture
def flops_config(tmp_path):
    cfg = compose_config(["model=rcf"], config_name="flops_config")
    cfg.output = str(tmp_path / "flops.csv")
    cfg.device = "cpu"
    cfg.image_size = 32
    cfg.band_configs = ["rgb"]
    cfg.timing_batch_size = 2
    cfg.n_warmup = 0
    cfg.n_measure = 1
    return cfg


def test_completed_profile_survives_invalid_segmentation_head(flops_config, monkeypatch):
    import pandas as pd

    monkeypatch.setattr(
        "torchgeo_bench.flops_pipeline.instantiate", lambda *args, **kwargs: _TinyConvNet()
    )
    flops_config.eval.segmentation.layers = ["stem"]
    flops_config.seg_head_types = ["invalid"]
    with pytest.raises(ValueError, match="head"):
        main(flops_config)

    rows = pd.read_csv(flops_config.output)
    assert list(rows["task"]) == ["classification"]
    assert rows.iloc[0]["gflops_total"] > 0
    assert not rows.iloc[0]["lenient_grad_hooks"]


@pytest.mark.parametrize(
    "error",
    [
        BandCompatibilityError("unsupported selection"),
        ValueError("input channels implementation is broken"),
    ],
)
def test_forward_pass_only_skips_explicit_band_errors(flops_config, monkeypatch, caplog, error):
    class BrokenForward(nn.Module):
        def forward(self, images):
            raise error

    monkeypatch.setattr(
        "torchgeo_bench.flops_pipeline.instantiate", lambda *args, **kwargs: BrokenForward()
    )
    if isinstance(error, BandCompatibilityError):
        main(flops_config)
        assert "unsupported selection" in caplog.text
    else:
        with pytest.raises(ValueError, match="implementation is broken"):
            main(flops_config)


def test_flops_config_resolves_every_shipped_model_config():
    """rcf.yaml uses ``seed: ${seed}``, so flops_config must define the referenced top-level seed."""
    from omegaconf import OmegaConf

    from torchgeo_bench.config import compose_config

    cfg = compose_config(["model=rcf"], config_name="flops_config", default_model=None)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert resolved["model"]["seed"] == resolved["seed"] == 0


@pytest.mark.slow
def test_panopticon_yields_finite_gflops():
    """Count the real Panopticon backbone without replacing PyTorch hooks."""
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
    """ViT-L costs more than ViT-B, with patch tokens following (image_size / patch_size)^2."""
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
