import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from torchgeo_bench.results import bootstrap_miou
from torchgeo_bench.segmentation_probe import (
    CachedFeaturesDataset,
    GPUTensorCache,
    SegmentationProbe,
    _resolve_num_prefix_tokens,
)
from torchgeo_bench.segmentation_task import SegmentationSolver, build_seg_probe_and_solver

NUM_CLASSES = 5


def test_bootstrap_miou_uses_per_image_resampling() -> None:
    """Bootstrap intervals reflect variation across held-out images."""
    confusions = torch.tensor([[[0, 4], [0, 0]], [[0, 0], [0, 4]]])

    lower, upper = bootstrap_miou(confusions, n_boot=200, seed=7)

    assert 0.0 <= lower < upper <= 1.0


class MockBackbone(nn.Module):
    """A simple CNN to simulate a backbone with intermediate layers."""

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=2), nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2), nn.ReLU()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class WrappedBackbone(nn.Module):
    """Backbone whose layers are nested under a 'backbone' attribute, as in BenchModel wrappers."""

    def __init__(self):
        super().__init__()
        self.backbone = MockBackbone()

    def forward(self, x):
        return self.backbone(x)


class ViTBackbone(nn.Module):
    """Backbone that emits (B, L, C) tokens from an intermediate layer, mimicking a ViT patch encoder."""

    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 16, kernel_size=16, stride=16)
        self.blocks = nn.Identity()

    def forward(self, x):
        x = self.patch_embed(x)  # (B, 16, H/16, W/16)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C) = (B, L, C)
        x = self.blocks(x)
        return x


class TwoChannelBackbone(nn.Module):
    """Backbone with a BenchModel-like num_channels attribute."""

    num_channels = 2

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 8, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x).mean(dim=(-2, -1))


@pytest.fixture
def mock_backbone():
    return MockBackbone()


@pytest.fixture
def dummy_data():
    images = torch.randn(2, 3, 64, 64)
    masks = torch.randint(0, NUM_CLASSES, (2, 64, 64))
    return {"image": images, "mask": masks}


def make_probe(backbone, layers, head_type="linear", freeze=True, hidden_dim=None):
    return SegmentationProbe(
        backbone=backbone,
        layer_names=layers,
        num_classes=NUM_CLASSES,
        freeze_backbone=freeze,
        head_type=head_type,
        hidden_dim=hidden_dim,
    )


def make_loader(images, masks, as_dict=False, mask_4d=False):
    if mask_4d:
        masks = masks.unsqueeze(1)
    if as_dict:

        class DictDataset(torch.utils.data.Dataset):
            def __len__(self):
                return len(images)

            def __getitem__(self, idx):
                return {"image": images[idx], "mask": masks[idx]}

        return DataLoader(DictDataset(), batch_size=2)
    return DataLoader(TensorDataset(images, masks), batch_size=2)


def test_probe_unknown_head_type(mock_backbone):
    """Test that an invalid head_type raises a ValueError."""
    with pytest.raises(ValueError, match="Unknown head_type"):
        SegmentationProbe(
            backbone=mock_backbone, layer_names=["layer1"], num_classes=2, head_type="invalid_type"
        )


def test_probe_rejects_missing_or_duplicate_layers(mock_backbone):
    """Layer typos must fail before a benchmark starts extracting features."""
    with pytest.raises(ValueError, match="not found"):
        SegmentationProbe(
            backbone=mock_backbone,
            layer_names=["not_a_layer"],
            num_classes=NUM_CLASSES,
        )
    with pytest.raises(ValueError, match="must be unique"):
        SegmentationProbe(
            backbone=mock_backbone,
            layer_names=["layer1", "layer1"],
            num_classes=NUM_CLASSES,
        )


def test_build_seg_probe_requires_spatial_layers(mock_backbone):
    """Segmentation evaluation refuses the global-output fallback."""
    eval_cfg = OmegaConf.create(
        {
            "segmentation": {
                "layers": [],
                "head_type": "fpn",
                "criterion": {"_target_": "torch.nn.CrossEntropyLoss", "ignore_index": 255},
                "lr_scheduler": "none",
            }
        }
    )
    with pytest.raises(ValueError, match="requires eval.segmentation.layers"):
        build_seg_probe_and_solver(
            mock_backbone,
            num_classes=NUM_CLASSES,
            eval_cfg=eval_cfg,
            device=torch.device("cpu"),
            lr=1e-3,
        )


def test_build_seg_solver_uses_criterion_ignore_index(mock_backbone):
    """Metrics inherit the loss ignore_index when no separate override is set."""
    eval_cfg = OmegaConf.create(
        {
            "segmentation": {
                "layers": ["layer1"],
                "head_type": "linear",
                "criterion": {"_target_": "torch.nn.CrossEntropyLoss", "ignore_index": 7},
                "lr_scheduler": "none",
            }
        }
    )
    _, solver = build_seg_probe_and_solver(
        mock_backbone,
        num_classes=NUM_CLASSES,
        eval_cfg=eval_cfg,
        device=torch.device("cpu"),
        lr=1e-3,
    )
    assert solver.ignore_index == 7


def test_probe_dry_run_exception_handling():
    """Test that dry_run_channels catches exceptions from the backbone."""

    class BrokenBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            # A real `layer1` so the probe gets past its layer-name check and
            # this test exercises the forward crash it is named for.
            self.layer1 = nn.Conv2d(3, 4, 1)

        def forward(self, x):
            del x
            raise RuntimeError("Backbone crash")

    backbone = BrokenBackbone()

    with pytest.raises(RuntimeError):
        SegmentationProbe(backbone, ["layer1"], 2)


def test_segmentation_probe_initialization(mock_backbone, dummy_data):
    """Freezes backbone params and registers one hook per requested layer."""
    images = dummy_data["image"]
    num_classes = 5
    layer_names = ["layer1", "layer2"]

    probe = SegmentationProbe(
        backbone=mock_backbone,
        layer_names=layer_names,
        num_classes=num_classes,
        freeze_backbone=True,
        head_type="linear",
    )

    logits = probe(images)
    assert logits.shape == (2, num_classes, 64, 64)

    for param in probe.backbone.parameters():
        assert param.requires_grad is False

    for param in probe.head.parameters():
        assert param.requires_grad is True


def test_segmentation_probe_conv_block_head(mock_backbone, dummy_data):
    """Test the MLP head configuration."""
    data = dummy_data
    num_classes = 5

    probe = SegmentationProbe(
        backbone=mock_backbone,
        layer_names=["layer2"],
        num_classes=num_classes,
        head_type="conv_block",
        hidden_dim=16,
    )

    logits = probe(data["image"])
    assert logits.shape == (2, num_classes, 64, 64)
    # conv_block head is a ConvBlockHead with projectors + a final Conv2d
    from torchgeo_bench.models.segmentation_heads import ConvBlockHead

    assert isinstance(probe.head, ConvBlockHead)
    assert hasattr(probe.head, "projectors")
    assert isinstance(probe.head.head, nn.Conv2d)


def test_solver_fit_and_evaluate(mock_backbone, dummy_data):
    """Test the training loop and evaluation metric."""
    data = dummy_data
    dataset = TensorDataset(data["image"], data["mask"])
    loader = DataLoader(dataset, batch_size=2)

    probe = SegmentationProbe(
        backbone=mock_backbone, layer_names=["layer1", "layer2"], num_classes=NUM_CLASSES
    )

    solver = SegmentationSolver(model=probe, num_classes=NUM_CLASSES, lr=1e-3, device="cpu")

    solver.fit(loader, epochs=1, verbose=True)

    metrics = solver.evaluate(loader)

    assert isinstance(metrics, dict)
    assert set(metrics.keys()) == {"mIoU", "fw_IoU", "precision", "recall", "f1"}
    assert 0.0 <= metrics["mIoU"] <= 1.0


def test_probe_fpn_head(mock_backbone, dummy_data):
    """FPN head forward pass produces correct output shape and has expected attributes."""
    from torchgeo_bench.models.segmentation_heads import FPNHead

    probe = make_probe(mock_backbone, ["layer2", "layer1"], head_type="fpn", hidden_dim=16)

    assert isinstance(probe.head, FPNHead)
    assert hasattr(probe.head, "laterals")
    assert hasattr(probe.head, "fpn_convs")
    assert hasattr(probe.head, "fpn_head")

    logits = probe(dummy_data["image"])
    assert logits.shape == (2, NUM_CLASSES, 64, 64)


def test_probe_backbone_prefix_stripping(dummy_data):
    """Layer names prefixed with 'backbone.' are resolved in wrapped models."""
    backbone = WrappedBackbone()
    # The inner layers are at backbone.layer1 / backbone.layer2 inside the wrapper,
    # but SegmentationProbe should strip the leading 'backbone.' prefix so that
    # specifying ["layer1"] still works.
    probe = make_probe(backbone, ["layer1", "layer2"])
    logits = probe(dummy_data["image"])
    assert logits.shape == (2, NUM_CLASSES, 64, 64)


def test_probe_linear_multi_layer_weighted(mock_backbone, dummy_data):
    """Multi-layer linear probe uses scale_weights and returns correct shape."""
    probe = make_probe(mock_backbone, ["layer1", "layer2"], head_type="linear")
    assert hasattr(probe.head, "scale_weights")
    logits = probe(dummy_data["image"])
    assert logits.shape == (2, NUM_CLASSES, 64, 64)


def test_probe_conv_block_multi_layer(mock_backbone, dummy_data):
    """conv_block with two layers at different resolutions triggers interpolation alignment."""
    probe = make_probe(mock_backbone, ["layer1", "layer2"], head_type="conv_block", hidden_dim=16)
    logits = probe(dummy_data["image"])
    assert logits.shape == (2, NUM_CLASSES, 64, 64)


def test_probe_unfrozen_backbone(mock_backbone, dummy_data):
    """With freeze_backbone=False the backbone runs in train mode and grads flow."""
    probe = make_probe(mock_backbone, ["layer1"], freeze=False)
    for param in probe.backbone.parameters():
        assert param.requires_grad is True
    logits = probe(dummy_data["image"])
    assert logits.shape == (2, NUM_CLASSES, 64, 64)


# Probe: ViT-style (B, L, C) token features via _process_feature


def test_probe_vit_token_features():
    """ViT backbone emitting (B, L, C) tokens is reshaped to (B, C, H, H)."""
    backbone = ViTBackbone()
    # 'blocks' is an Identity that passes through (B, L, C); hook it directly
    probe = make_probe(backbone, ["blocks"], head_type="linear")
    images = torch.randn(2, 3, 64, 64)
    logits = probe(images)
    assert logits.shape == (2, NUM_CLASSES, 64, 64)


def test_resolve_num_prefix_tokens_walks_module_tree():
    """The resolver finds timm's num_prefix_tokens however deeply it is nested."""

    class _Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_prefix_tokens = 5

    class _Outer(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = _Inner()

    assert _resolve_num_prefix_tokens(_Outer()) == 5
    # Non-timm token models declare nothing and must not be guessed at.
    assert _resolve_num_prefix_tokens(nn.Linear(4, 4)) is None


def test_process_feature_drops_dinov3_register_tokens():
    """DINOv3's 1 CLS + 4 registers are dropped, not reshaped into a fake grid.

    (B, 261, 1024) has L=261 (neither s^2 nor s^2+1); without prefix-token
    stripping the (B, C, L) branch treats 1024 as a perfect square and
    produces a 32x32 grid of 261 "channels" with no error.
    """

    class _DinoV3Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_prefix_tokens = 5
            self.blocks = nn.Identity()

        def forward(self, x):
            b = x.shape[0]
            return self.blocks(torch.randn(b, 261, 1024))

    probe = make_probe(_DinoV3Backbone(), ["blocks"], head_type="linear")
    assert probe.channels_list == [1024]
    assert probe.feature_hw_list == [(16, 16)]


def test_process_feature_plain_vit_cls_token_unchanged():
    """A single CLS token (L=197) still resolves to a 14x14 grid."""

    class _ViTBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_prefix_tokens = 1
            self.blocks = nn.Identity()

        def forward(self, x):
            b = x.shape[0]
            return self.blocks(torch.randn(b, 197, 768))

    probe = make_probe(_ViTBackbone(), ["blocks"], head_type="linear")
    assert probe.channels_list == [768]
    assert probe.feature_hw_list == [(14, 14)]


def test_process_feature_rejects_unreshapeable_tokens():
    """A declared-prefix model with a non-square token count raises, not guesses.

    The (B, C, L) fallback is gated on the model *not* declaring
    num_prefix_tokens: a model that declares it is token-first by construction,
    so a token count that survives prefix-stripping without becoming square is a
    genuine mismatch.  Ungated, 1024 is a perfect square and this silently
    reshapes into a 32x32 grid of 261 "channels".
    """

    class _OddBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_prefix_tokens = 2  # 261 - 2 = 259, not a square
            self.blocks = nn.Identity()

        def forward(self, x):
            b = x.shape[0]
            return self.blocks(torch.randn(b, 261, 1024))

    with pytest.raises(ValueError, match="Could not reshape 3D feature map"):
        make_probe(_OddBackbone(), ["blocks"], head_type="linear")


def test_probe_dry_run_uses_backbone_num_channels():
    """Dry-run channel inference supports non-RGB benchmark models."""
    backbone = TwoChannelBackbone()
    probe = make_probe(backbone, [], head_type="linear")
    images = torch.randn(2, 2, 64, 64)
    logits = probe(images)
    assert logits.shape == (2, NUM_CLASSES, 64, 64)


def test_solver_no_lr_scheduler(mock_backbone, dummy_data):
    """lr_scheduler='none' runs without a scheduler and completes training."""
    images, masks = dummy_data["image"], dummy_data["mask"]
    loader = make_loader(images, masks)
    probe = make_probe(mock_backbone, ["layer1"])
    solver = SegmentationSolver(
        model=probe, num_classes=NUM_CLASSES, lr=1e-3, device="cpu", lr_scheduler="none"
    )
    result = solver.fit(loader, epochs=1, verbose=False)
    assert result is None  # no val_loader → returns None


def test_solver_dict_batches(mock_backbone, dummy_data):
    """fit and evaluate both handle dict-format batches {"image": ..., "mask": ...}."""
    images, masks = dummy_data["image"], dummy_data["mask"]
    loader = make_loader(images, masks, as_dict=True)
    probe = make_probe(mock_backbone, ["layer1"])
    solver = SegmentationSolver(model=probe, num_classes=NUM_CLASSES, lr=1e-3, device="cpu")
    solver.fit(loader, epochs=1, verbose=False)
    metrics = solver.evaluate(loader)
    assert 0.0 <= metrics["mIoU"] <= 1.0


def test_solver_4d_masks(mock_backbone, dummy_data):
    """fit and evaluate both squeeze (B, 1, H, W) masks to (B, H, W)."""
    images, masks = dummy_data["image"], dummy_data["mask"]
    loader = make_loader(images, masks, mask_4d=True)
    probe = make_probe(mock_backbone, ["layer1"])
    solver = SegmentationSolver(model=probe, num_classes=NUM_CLASSES, lr=1e-3, device="cpu")
    solver.fit(loader, epochs=1, verbose=False)
    metrics = solver.evaluate(loader)
    assert 0.0 <= metrics["mIoU"] <= 1.0


def test_solver_fit_with_val_loader(mock_backbone, dummy_data):
    """fit returns the final epoch val mIoU when a val_loader is provided."""
    images, masks = dummy_data["image"], dummy_data["mask"]
    train_loader = make_loader(images, masks)
    val_loader = make_loader(images, masks)
    probe = make_probe(mock_backbone, ["layer1"])
    solver = SegmentationSolver(model=probe, num_classes=NUM_CLASSES, lr=1e-3, device="cpu")
    val_miou = solver.fit(train_loader, val_loader=val_loader, epochs=1, verbose=False)
    assert isinstance(val_miou, float)
    assert 0.0 <= val_miou <= 1.0
    assert solver.val_history == [val_miou]


# Probe: DPT head


class MockBackbone4Layer(nn.Module):
    """CNN backbone with 4 strided layers to provide multi-scale features for DPT."""

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3, 8, kernel_size=3, padding=1, stride=1), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(8, 16, kernel_size=3, padding=1, stride=2), nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2), nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2), nn.ReLU()
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def test_probe_dpt_head_forward():
    """DPT head with 4 coarse-to-fine layers produces correct output shape."""
    pytest.importorskip("transformers")
    from torchgeo_bench.models.segmentation_heads import DPTHead

    backbone = MockBackbone4Layer()
    # Coarse-to-fine order (same convention as FPN)
    probe = make_probe(
        backbone,
        layers=["layer4", "layer3", "layer2", "layer1"],
        head_type="dpt",
        hidden_dim=16,
    )

    assert isinstance(probe.head, DPTHead)
    assert hasattr(probe.head, "convs")
    assert hasattr(probe.head, "ref")
    assert hasattr(probe.head, "out_conv")
    assert len(probe.head.convs) == 4
    assert len(probe.head.ref) == 4

    images = torch.randn(2, 3, 64, 64)
    logits = probe(images)
    assert logits.shape == (2, NUM_CLASSES, 64, 64)


def test_probe_dpt_wrong_num_layers():
    """DPT head raises ValueError when not exactly 4 layers are specified."""
    backbone = MockBackbone()  # only has layer1, layer2
    with pytest.raises(ValueError, match="DPTHead requires exactly 4 feature layers"):
        make_probe(backbone, layers=["layer1", "layer2"], head_type="dpt", hidden_dim=16)


def test_dpt_fusion_layer_shim_matches_reference():
    """The SimpleNamespace shim still satisfies the reference fusion layer's config API.

    ``DPTFeatureFusionLayer`` is a private ``transformers`` API with no stability
    guarantee. This asserts the four structural properties we depend on, so a
    breaking upstream bump fails loudly here instead of silently changing the
    decoder's arithmetic.
    """
    pytest.importorskip("transformers")
    from transformers.models.dpt.modeling_dpt import DPTPreActResidualLayer

    from torchgeo_bench.models.segmentation_heads import _dpt_fusion_layer

    layer = _dpt_fusion_layer(16)

    # 1. Post-fusion 1x1 projection is present (absent from the old implementation).
    assert isinstance(layer.projection, nn.Conv2d)
    assert layer.projection.kernel_size == (1, 1)

    # 2. Both residual units are pre-activation (relu -> conv -> relu -> conv).
    assert isinstance(layer.residual_layer1, DPTPreActResidualLayer)
    assert isinstance(layer.residual_layer2, DPTPreActResidualLayer)

    # 3. Each fusion layer upsamples 2x internally.
    out = layer(torch.randn(1, 16, 7, 7))
    assert out.shape == (1, 16, 14, 14)

    # 4. A mismatched skip is resized to the primary input, not the reverse.
    out = layer(torch.randn(1, 16, 7, 7), torch.randn(1, 16, 3, 3))
    assert out.shape == (1, 16, 14, 14)


def test_dpt_head_upsamples_purely_through_fusion_cascade():
    """Four fusion layers take a 14x14 ViT token grid to 224x224 exactly.

    The faithful cascade supplies all upsampling itself, so the head no longer
    needs the pre-projection 2x and terminal 4x interpolations. With input_h/w
    already at 224 the trailing resize is a no-op, which is what makes this a
    regression test on the schedule rather than on the final ``F.interpolate``.
    """
    pytest.importorskip("transformers")
    from torchgeo_bench.models.segmentation_heads import DPTHead

    head = DPTHead([32, 32, 32, 32], num_classes=NUM_CLASSES, hidden_dim=16)
    features = [torch.randn(1, 32, 14, 14) for _ in range(4)]

    # Intercept the cascade output before out_conv / the final resize.
    projected = [conv(norm(f)) for norm, conv, f in zip(head.input_norms, head.convs, features)]
    fused = head.ref[0](projected[0])
    for layer, feat in zip(head.ref[1:], projected[1:]):
        fused = layer(fused, feat)
    assert fused.shape[-2:] == (224, 224)

    logits = head(features, 224, 224)
    assert logits.shape == (1, NUM_CLASSES, 224, 224)


# Probe: PatchLinear head


def test_patch_linear_head_output_shape():
    """PatchLinearHead upsamples a 4x4 token grid back to 64x64."""
    from torchgeo_bench.models.segmentation_heads import PatchLinearHead

    head = PatchLinearHead([16], num_classes=5)
    logits = head([torch.randn(2, 16, 4, 4)], 64, 64)

    assert logits.shape == (2, 5, 64, 64)
    assert torch.isfinite(logits).all()


def test_patch_linear_head_small_patch():
    """PatchLinearHead infers smaller patch sizes from denser token grids."""
    from torchgeo_bench.models.segmentation_heads import PatchLinearHead

    head = PatchLinearHead([8], num_classes=3)
    logits = head([torch.randn(2, 8, 16, 16)], 64, 64)

    assert logits.shape == (2, 3, 64, 64)
    assert torch.isfinite(logits).all()


def test_patch_linear_head_non_exact_size():
    """PatchLinearHead resizes to the requested image size when pixel shuffle is not exact."""
    from torchgeo_bench.models.segmentation_heads import PatchLinearHead

    head = PatchLinearHead([8], num_classes=3)
    logits = head([torch.randn(2, 8, 16, 16)], 65, 65)

    assert logits.shape == (2, 3, 65, 65)
    assert torch.isfinite(logits).all()


def test_patch_linear_head_ignores_extra_channels():
    """PatchLinearHead uses only the first feature map when extra layers are passed."""
    from torchgeo_bench.models.segmentation_heads import PatchLinearHead

    head = PatchLinearHead([8, 16], num_classes=3)
    logits = head([torch.randn(2, 8, 16, 16), torch.randn(2, 16, 8, 8)], 64, 64)

    assert logits.shape == (2, 3, 64, 64)
    assert torch.isfinite(logits).all()


def test_probe_patch_linear_head_vit():
    """Patch-linear probe works end-to-end with ViT token features."""
    from torchgeo_bench.models.segmentation_heads import PatchLinearHead

    probe = make_probe(ViTBackbone(), ["blocks"], head_type="patch_linear")
    images = torch.randn(2, 3, 64, 64)
    logits = probe(images)

    assert isinstance(probe.head, PatchLinearHead)
    assert logits.shape == (2, NUM_CLASSES, 64, 64)


def test_probe_patch_linear_cached_features():
    """fit_cached trains a patch-linear probe on ViT feature caches."""
    images = torch.randn(2, 3, 64, 64)
    masks = torch.randint(0, NUM_CLASSES, (2, 64, 64))
    loader = make_loader(images, masks)
    probe = make_probe(ViTBackbone(), ["blocks"], head_type="patch_linear")
    solver = SegmentationSolver(model=probe, num_classes=NUM_CLASSES, lr=1e-3, device="cpu")

    train_cache = probe.extract_segmentation_features(loader, cache_dtype=torch.float32)
    val_miou = solver.fit_cached(train_cache, val_cache=train_cache, batch_size=2, epochs=1)

    assert isinstance(val_miou, float)
    assert 0.0 <= val_miou <= 1.0


# Feature caching: extract_segmentation_features + CachedFeaturesDataset


def test_extract_segmentation_features_returns_cached_dataset(mock_backbone, dummy_data):
    """extract_segmentation_features produces a CachedFeaturesDataset with correct length and dtypes."""
    images, masks = dummy_data["image"], dummy_data["mask"]
    loader = make_loader(images, masks)
    probe = make_probe(mock_backbone, ["layer1", "layer2"])

    cache = probe.extract_segmentation_features(loader, cache_dtype=torch.float16)

    assert isinstance(cache, CachedFeaturesDataset)
    assert len(cache) == len(images)
    feats, mask = cache[0]
    assert len(feats) == 2  # two hooked layers
    assert feats[0].dtype == torch.float16
    assert mask.dtype == torch.int64


def test_solver_fit_cached(mock_backbone, dummy_data):
    """fit_cached trains the head on cached features and evaluate_cached returns a valid mIoU."""
    images, masks = dummy_data["image"], dummy_data["mask"]
    loader = make_loader(images, masks)
    probe = make_probe(mock_backbone, ["layer1", "layer2"])
    solver = SegmentationSolver(model=probe, num_classes=NUM_CLASSES, lr=1e-3, device="cpu")

    train_cache = probe.extract_segmentation_features(loader, cache_dtype=torch.float32)
    val_cache = probe.extract_segmentation_features(loader, cache_dtype=torch.float32)

    val_miou = solver.fit_cached(
        train_cache, val_cache=val_cache, batch_size=2, epochs=1, verbose=False
    )
    assert isinstance(val_miou, float)
    assert solver.val_history == [val_miou]
    assert 0.0 <= val_miou <= 1.0

    metrics = solver.evaluate_cached(val_cache, batch_size=2)
    assert isinstance(metrics, dict)
    assert 0.0 <= metrics["mIoU"] <= 1.0


def test_extract_segmentation_features_dict_batches(mock_backbone, dummy_data):
    """extract_segmentation_features handles dict-format batches {"image": ..., "mask": ...}."""
    images, masks = dummy_data["image"], dummy_data["mask"]
    loader = make_loader(images, masks, as_dict=True)
    probe = make_probe(mock_backbone, ["layer1"])
    cache = probe.extract_segmentation_features(loader, cache_dtype=torch.float32)
    assert len(cache) == len(images)


def test_extract_segmentation_features_restores_backbone_mode(mock_backbone, dummy_data):
    """Feature extraction preserves the caller's backbone train/eval state."""
    images, masks = dummy_data["image"], dummy_data["mask"]
    loader = make_loader(images, masks)
    probe = make_probe(mock_backbone, ["layer1"])
    probe.backbone.train()

    probe.extract_segmentation_features(loader, cache_dtype=torch.float32)

    assert probe.backbone.training

# ---------------------------------------------------------------------------
# GPUTensorCache
# ---------------------------------------------------------------------------


def _make_cpu_cache(mock_backbone, dummy_data):
    """Helper: extract a CachedFeaturesDataset on CPU."""
    images, masks = dummy_data["image"], dummy_data["mask"]
    loader = make_loader(images, masks)
    probe = make_probe(mock_backbone, ["layer1", "layer2"])
    return probe.extract_segmentation_features(loader, cache_dtype=torch.float16)


def test_gpu_tensor_cache_from_cached_cpu(mock_backbone, dummy_data):
    """GPUTensorCache.from_cached builds correct tensors on CPU device."""
    cache = _make_cpu_cache(mock_backbone, dummy_data)
    gpu_cache = GPUTensorCache.from_cached(cache, device="cpu")

    assert len(gpu_cache) == len(cache)
    assert len(gpu_cache.layer_tensors) == 2  # two hooked layers
    assert gpu_cache.layer_tensors[0].dtype == torch.float32  # CPU path uses float32
    assert gpu_cache.masks.dtype == torch.long
    # Spatial dims should match the mask dims in the original cache
    assert gpu_cache.masks.shape == (len(cache), 64, 64)


def test_gpu_tensor_cache_shuffled_batches(mock_backbone, dummy_data):
    """shuffled_batches yields all samples exactly once with correct shapes."""
    cache = _make_cpu_cache(mock_backbone, dummy_data)
    gpu_cache = GPUTensorCache.from_cached(cache, device="cpu")

    all_masks = []
    for feats, masks in gpu_cache.shuffled_batches(batch_size=1):
        assert len(feats) == 2
        assert feats[0].shape[0] == masks.shape[0]  # batch dim matches
        all_masks.append(masks)

    total = sum(m.shape[0] for m in all_masks)
    assert total == len(cache)


def test_gpu_tensor_cache_ordered_batches(mock_backbone, dummy_data):
    """ordered_batches yields samples in order with correct total count."""
    cache = _make_cpu_cache(mock_backbone, dummy_data)
    gpu_cache = GPUTensorCache.from_cached(cache, device="cpu")

    total = 0
    for _feats, masks in gpu_cache.ordered_batches(batch_size=1):
        total += masks.shape[0]
    assert total == len(cache)


def test_solver_fit_cached_uses_gpu_cache_path(mock_backbone, dummy_data):
    """fit_cached falls back gracefully to DataLoader path on CPU (no CUDA available in CI)."""
    images, masks = dummy_data["image"], dummy_data["mask"]
    loader = make_loader(images, masks)
    probe = make_probe(mock_backbone, ["layer1", "layer2"])
    solver = SegmentationSolver(model=probe, num_classes=NUM_CLASSES, lr=1e-3, device="cpu")

    train_cache = probe.extract_segmentation_features(loader, cache_dtype=torch.float32)
    val_cache = probe.extract_segmentation_features(loader, cache_dtype=torch.float32)

    # On CPU, use_amp=False so GPUTensorCache path is skipped; DataLoader fallback runs.
    val_miou = solver.fit_cached(
        train_cache, val_cache=val_cache, batch_size=2, epochs=1, verbose=False
    )
    assert isinstance(val_miou, float)
    assert 0.0 <= val_miou <= 1.0


def test_solver_fit_cached_builds_missing_validation_gpu_cache(mock_backbone, dummy_data):
    """A supplied training GPU cache must not disable validation caching."""
    images, masks = dummy_data["image"], dummy_data["mask"]
    loader = make_loader(images, masks)
    probe = make_probe(mock_backbone, ["layer1", "layer2"])
    solver = SegmentationSolver(model=probe, num_classes=NUM_CLASSES, lr=1e-3, device="cpu")
    train_cache = probe.extract_segmentation_features(loader, cache_dtype=torch.float32)
    val_cache = probe.extract_segmentation_features(loader, cache_dtype=torch.float32)
    gpu_train = GPUTensorCache.from_cached(train_cache, device="cpu")

    val_miou = solver.fit_cached(
        train_cache,
        val_cache=val_cache,
        batch_size=2,
        epochs=1,
        verbose=False,
        gpu_train=gpu_train,
    )

    assert isinstance(val_miou, float)
