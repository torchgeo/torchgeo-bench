"""Tests for the per-sample compute-cost (GFLOPs) pipeline.

The fast tests build tiny synthetic modules so they run on CPU without
downloading anything.  Tests that need a real pretrained backbone are marked
``slow`` (``-m slow`` to include), matching the repo-wide convention.
"""

import math
from pathlib import Path

import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn

from torchgeo_bench import flops_pipeline
from torchgeo_bench.datasets import get_bench_dataset_class
from torchgeo_bench.flops_pipeline import (
    _MODALITY_FOR_BAND_CONFIG,
    _is_band_incompatibility,
    _load_completed,
    _n_tokens,
    _probe_gflops,
    _seg_head_gflops,
)
from torchgeo_bench.model_profile import _count_gflops, lenient_grad_hooks
from torchgeo_bench.segmentation_task import build_seg_probe_and_solver

CPU = torch.device("cpu")


# ---------------------------------------------------------------------------
# lenient_grad_hooks: the no-op guarantee
# ---------------------------------------------------------------------------


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


class _GradBreakingNet(nn.Module):
    """Reproduces Panopticon's failure shape.

    A ``conv3d`` fed a tensor introduced *inside* the forward has no
    ``grad_fn``, so ``module_tracker``'s forward-pre-hook raises before any
    counting happens.  This is the structural pattern, not the model.
    """

    def __init__(self, in_ch: int = 3) -> None:
        super().__init__()
        # (B, 1, 1, H, W) -> (B, 4, 1, H, W), squeezed back to (B, 4, H, W).
        self.conv3d = nn.Conv3d(1, 4, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Detach severs the autograd chain the way Panopticon's chn-fusion does.
        broken = x.mean(dim=1, keepdim=True).detach()
        y = self.conv3d(broken.unsqueeze(1)).squeeze(2)
        return self.head(y.mean(dim=(-2, -1)))


def test_lenient_grad_hooks_is_bit_identical_for_healthy_models():
    """The load-bearing check: the workaround must change nothing for models
    that already work.  Bit-identical, not merely close."""
    model = _TinyConvNet().eval()
    x = torch.randn(2, 3, 32, 32)

    plain = _count_gflops(model, x)
    with lenient_grad_hooks():
        lenient = _count_gflops(model, x)

    assert plain == lenient, f"{plain!r} != {lenient!r}"
    assert plain > 0


def test_lenient_grad_hooks_restores_the_original_hook():
    """The patch must not leak outside the context manager."""
    import torch.autograd.graph as autograd_graph
    import torch.utils.module_tracker as module_tracker

    before = autograd_graph.register_multi_grad_hook
    before_mt = module_tracker.register_multi_grad_hook
    with lenient_grad_hooks():
        assert autograd_graph.register_multi_grad_hook is not before
    assert autograd_graph.register_multi_grad_hook is before
    assert module_tracker.register_multi_grad_hook is before_mt


def test_lenient_grad_hooks_restores_on_exception():
    import torch.autograd.graph as autograd_graph

    before = autograd_graph.register_multi_grad_hook
    with pytest.raises(RuntimeError), lenient_grad_hooks():
        raise RuntimeError("boom")
    assert autograd_graph.register_multi_grad_hook is before


def test_grad_breaking_model_is_measurable():
    """A model with Panopticon's grad-breaking shape yields finite GFLOPs.

    ``_count_gflops`` runs the whole tier ladder inside ``lenient_grad_hooks``,
    so this must not raise ``NotImplementedError``.
    """
    model = _GradBreakingNet().eval()
    gflops = _count_gflops(model, torch.randn(2, 3, 32, 32))
    assert math.isfinite(gflops)
    assert gflops > 0


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

    with pytest.raises(ValueError, match="Missing required model band 'nir'") as missing:
        map_to_model_bands(torch.zeros(1, 3, 4, 4), rgb, ["blue", "green", "red", "nir"])
    assert _is_band_incompatibility(missing.value) is missing.value

    with pytest.raises(ValueError, match="none of the target bands") as empty:
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
    with pytest.raises(ValueError, match="images has 7 channels but") as mismatch:
        map_to_model_bands(torch.zeros(1, 7, 4, 4), rgb, ["red", "green", "blue"])
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


type FlopsRun = tuple[DictConfig, list[dict[str, object]], list[str]]


@pytest.fixture
def flops_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FlopsRun:
    cfg = OmegaConf.create(
        {
            "output": str(tmp_path / "compute_cost.csv"),
            "device": "cpu",
            "image_size": 8,
            "normalization": "bandspec_zscore",
            "band_source": "cloudsen12",
            "band_configs": ["rgb", "s2"],
            "seg_band_configs": ["rgb", "s2"],
            "seg_head_types": ["fpn", "dpt"],
            "probe_head": "linear",
            "probe_num_classes": 7,
            "seg_num_classes": 3,
            "timing_batch_size": 8,
            "n_warmup": 1,
            "n_measure": 2,
            "resume": False,
            "model": {
                "_target_": "test.TinyModel",
                "name": "tiny",
                "eval": {"segmentation": {"layers": ["stem"]}},
            },
            "eval": {"segmentation": {"layers": [], "head_type": "fpn"}},
        }
    )
    rows: list[dict[str, object]] = []
    events: list[str] = []

    def build_model(_cfg: DictConfig, bands: list, *_args: object) -> nn.Module:
        events.append(f"model:{len(bands)}")
        return _WidthModel(5)

    def measure_backbone(_model: nn.Module, channels: int, *_args: object) -> tuple[dict, int]:
        events.append(f"backbone:{channels}")
        return {
            "gflops": 2.0,
            "params_m": 0.1,
            "throughput_samples_per_sec": 16.0,
            "latency_ms_per_batch_p50": 250.0,
            "peak_gpu_mem_gb": None,
            "reserved_gpu_mem_gb": None,
        }, 4

    def build_head(
        _model: nn.Module, _classes: int, head_cfg: DictConfig, *_args: object
    ) -> tuple[nn.Module, None]:
        events.append(f"head:{head_cfg.segmentation.head_type}")
        assert head_cfg.segmentation.layers == ["stem"]
        probe = nn.Module()
        probe.head = nn.Linear(5, 3)
        probe.channels_list = [2, 3]
        return probe, None

    monkeypatch.setattr(flops_pipeline, "_build_model", build_model)
    monkeypatch.setattr(flops_pipeline, "_measure_backbone", measure_backbone)
    monkeypatch.setattr(flops_pipeline, "_probe_gflops", lambda *_args: (0.5, 0.01, 5))
    monkeypatch.setattr(flops_pipeline, "build_seg_probe_and_solver", build_head)
    monkeypatch.setattr(flops_pipeline, "_seg_head_gflops", lambda *_args: 0.25)
    monkeypatch.setattr(
        flops_pipeline, "append_rows_atomic", lambda _path, values: rows.extend(values)
    )
    monkeypatch.setattr(flops_pipeline, "_now", lambda: "2026-09-04T00:00:00+00:00")
    return cfg, rows, events


def written_keys(rows: list[dict[str, object]]) -> list[tuple[object, object, object]]:
    return [(row["band_config"], row["task"], row["head_type"]) for row in rows]


def test_main_writes_ordered_measurement_rows(flops_run: FlopsRun) -> None:
    cfg, rows, _events = flops_run
    flops_pipeline.main(cfg)

    assert written_keys(rows) == [
        (band, task, head)
        for band in ("rgb", "s2")
        for task, head in (("classification", ""), ("segmentation", "fpn"), ("segmentation", "dpt"))
    ]
    assert [row["n_channels"] for row in rows] == [3, 3, 3, 12, 12, 12]
    classification, segmentation = rows[:2]
    assert classification["gflops_backbone"] == 2.0
    assert classification["gflops_probe"] == 0.5
    assert classification["gflops_total"] == 2.5
    assert classification["params_probe_m"] == 0.01
    assert classification["timing_batch_size"] == 4
    assert classification["throughput_samples_per_sec"] == 16.0
    assert classification["num_classes"] == 7
    assert segmentation["gflops_head"] == 0.25
    assert segmentation["gflops_backbone"] is None
    assert segmentation["gflops_total"] is None
    assert segmentation["feature_dim"] == 5
    assert segmentation["num_classes"] == 3
    assert segmentation["params_head_m"] == pytest.approx(18 / 1e6)
    assert segmentation["timing_batch_size"] is None


@pytest.mark.parametrize("all_complete", [False, True])
def test_main_skips_completed_cells(flops_run: FlopsRun, all_complete: bool) -> None:
    cfg, rows, events = flops_run
    cfg.resume = True
    completed = [("rgb", "classification", ""), ("s2", "segmentation", "fpn")]
    if all_complete:
        completed = [
            (band, task, head)
            for band in ("rgb", "s2")
            for task, head in (
                ("classification", ""),
                ("segmentation", "fpn"),
                ("segmentation", "dpt"),
            )
        ]
    Path(cfg.output).write_text(
        "name,band_config,task,head_type\n"
        + "".join(f"tiny,{band},{task},{head}\n" for band, task, head in completed)
    )
    flops_pipeline.main(cfg)

    if all_complete:
        assert rows == []
        assert events == ["model:3", "model:12"]
    else:
        assert written_keys(rows) == [
            ("rgb", "segmentation", "fpn"),
            ("rgb", "segmentation", "dpt"),
            ("s2", "classification", ""),
            ("s2", "segmentation", "dpt"),
        ]
        assert "backbone:3" not in events


def test_main_skips_unavailable_model(flops_run: FlopsRun, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg, rows, events = flops_run
    build = flops_pipeline._build_model
    monkeypatch.setattr(
        flops_pipeline,
        "_build_model",
        lambda model_cfg, bands, *args: None if len(bands) == 3 else build(model_cfg, bands, *args),
    )
    flops_pipeline.main(cfg)

    assert written_keys(rows) == [
        ("s2", "classification", ""),
        ("s2", "segmentation", "fpn"),
        ("s2", "segmentation", "dpt"),
    ]
    assert "backbone:3" not in events


@pytest.mark.parametrize("stage", ["_measure_backbone", "_probe_gflops"])
@pytest.mark.parametrize("band_mismatch", [False, True])
def test_main_classification_failure_handling(
    flops_run: FlopsRun, monkeypatch: pytest.MonkeyPatch, stage: str, band_mismatch: bool
) -> None:
    cfg, rows, events = flops_run
    measure = getattr(flops_pipeline, stage)

    def fail_rgb(model: nn.Module, channels: int, *args: object) -> object:
        if channels == 3:
            if band_mismatch:
                raise ValueError("Missing required model band 'nir'")
            raise RuntimeError("bad forward")
        return measure(model, channels, *args)

    monkeypatch.setattr(flops_pipeline, stage, fail_rgb)
    if not band_mismatch:
        with pytest.raises(RuntimeError, match="bad forward"):
            flops_pipeline.main(cfg)
        assert rows == []
        assert "model:12" not in events
        return

    flops_pipeline.main(cfg)
    assert written_keys(rows) == [
        ("s2", "classification", ""),
        ("s2", "segmentation", "fpn"),
        ("s2", "segmentation", "dpt"),
    ]
    assert events.count("head:fpn") == 1


def test_main_segmentation_failure_keeps_other_heads(
    flops_run: FlopsRun, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, rows, _events = flops_run
    cfg.band_configs = ["rgb"]
    build = flops_pipeline.build_seg_probe_and_solver

    def fail_fpn(
        model: nn.Module, classes: int, head_cfg: DictConfig, *args: object
    ) -> tuple[nn.Module, None]:
        if head_cfg.segmentation.head_type == "fpn":
            raise ValueError("unsupported head")
        return build(model, classes, head_cfg, *args)

    monkeypatch.setattr(flops_pipeline, "build_seg_probe_and_solver", fail_fpn)
    flops_pipeline.main(cfg)
    assert written_keys(rows) == [("rgb", "classification", ""), ("rgb", "segmentation", "dpt")]


@pytest.mark.parametrize("empty_layers", [False, True])
def test_main_skips_excluded_segmentation(flops_run: FlopsRun, empty_layers: bool) -> None:
    cfg, rows, events = flops_run
    if empty_layers:
        cfg.model.eval.segmentation.layers = []
    else:
        cfg.seg_band_configs = []
    flops_pipeline.main(cfg)

    assert written_keys(rows) == [("rgb", "classification", ""), ("s2", "classification", "")]
    assert "head:fpn" not in events


@pytest.mark.parametrize("layers", [None, 1])
def test_main_rejects_malformed_layers_before_measurement(
    flops_run: FlopsRun, layers: int | None
) -> None:
    cfg, rows, events = flops_run
    cfg.model.eval.segmentation.layers = layers
    with pytest.raises(TypeError):
        flops_pipeline.main(cfg)
    assert rows == []
    assert events == []


# ---------------------------------------------------------------------------
# Real backbones (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_panopticon_yields_finite_gflops():
    """Panopticon yields finite GFLOPs through the lenient_grad_hooks path."""
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
