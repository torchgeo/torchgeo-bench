from unittest import mock

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.functional.classification import (
    multiclass_confusion_matrix,
    multiclass_f1_score,
    multiclass_jaccard_index,
    multiclass_precision,
    multiclass_recall,
)

from torchgeo_bench.config_schema import SegmentationConfig
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
    """Match BenchModel wrappers, which nest layers under ``backbone``."""

    def __init__(self):
        super().__init__()
        self.backbone = MockBackbone()

    def forward(self, x):
        return self.backbone(x)


class ViTBackbone(nn.Module):
    """Expose token-first ``(B, L, C)`` features to the hooks."""

    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 16, kernel_size=16, stride=16)
        self.blocks = nn.Identity()

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.blocks(x)
        return x


class TwoChannelBackbone(nn.Module):
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


def make_probe(backbone, layers, head_type="linear", *, freeze=True, hidden_dim=None):
    return SegmentationProbe(
        backbone=backbone,
        layer_names=layers,
        num_classes=NUM_CLASSES,
        freeze_backbone=freeze,
        head_type=head_type,
        hidden_dim=hidden_dim,
    )


def make_loader(images, masks, *, as_dict=False, mask_4d=False):
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
    with pytest.raises(ValueError, match="Unknown head_type"):
        SegmentationProbe(
            backbone=mock_backbone, layer_names=["layer1"], num_classes=2, head_type="invalid_type"
        )


def test_probe_rejects_missing_or_duplicate_layers(mock_backbone):
    """Reject invalid hook layers before feature extraction."""
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
    """A pooled image vector is not a spatial segmentation feature map."""
    config = SegmentationConfig(layers=[], head="fpn", scheduler="none")
    with pytest.raises(ValueError, match=r"requires segmentation\.layers"):
        build_seg_probe_and_solver(
            mock_backbone,
            num_classes=NUM_CLASSES,
            config=config,
            device=torch.device("cpu"),
        )


def test_build_seg_solver_uses_criterion_ignore_index(mock_backbone):
    """Metrics inherit the loss ignore_index when no separate override is set."""
    config = SegmentationConfig(layers=["layer1"], head="linear", ignore_index=7, scheduler="none")
    _, solver = build_seg_probe_and_solver(
        mock_backbone,
        num_classes=NUM_CLASSES,
        config=config,
        device=torch.device("cpu"),
    )
    assert solver.ignore_index == 7


def test_probe_dry_run_exception_handling():
    """Backbone failures during shape inference must remain visible."""

    class BrokenBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            # A valid hook layer lets construction reach the failing forward pass.
            self.layer1 = nn.Conv2d(3, 4, 1)

        def forward(self, x):
            del x
            raise RuntimeError("Backbone crash")

    backbone = BrokenBackbone()

    with pytest.raises(RuntimeError):
        SegmentationProbe(backbone, ["layer1"], 2)


def test_segmentation_probe_initialization(mock_backbone, dummy_data):
    """Freeze backbone parameters while leaving the probe head trainable."""
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
    from torchgeo_bench.models.segmentation_heads import ConvBlockHead

    assert isinstance(probe.head, ConvBlockHead)
    assert hasattr(probe.head, "projectors")
    assert isinstance(probe.head.head, nn.Conv2d)


def test_solver_fit_and_evaluate(mock_backbone, dummy_data):
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


@pytest.mark.parametrize("collect_confusions", [False, True])
@pytest.mark.parametrize(
    ("as_dict", "mask_4d", "ignore_index"), [(False, False, 255), (True, True, -1)]
)
def test_solver_evaluation_parity(
    *,
    collect_confusions: bool,
    as_dict: bool,
    mask_4d: bool,
    ignore_index: int,
) -> None:
    """Both paths reset metrics and preserve optional outputs in sample order."""
    predictions = torch.tensor(
        [
            [[0, 1, 2], [2, 1, 0]],
            [[2, 2, 1], [0, 0, 1]],
            [[1, 0, 1], [2, 2, 0]],
            [[0, 2, 2], [1, 1, 0]],
            [[2, 1, 0], [0, 2, 1]],
        ]
    )
    masks = torch.tensor(
        [
            [[0, 1, 2], [1, 255, 0]],
            [[2, 0, 1], [0, 1, 255]],
            [[255, 0, 1], [2, 0, 0]],
            [[2, 2, 1], [1, 0, 0]],
            [[255, 255, 255], [255, 255, 255]],
        ]
    )
    masks[masks == 255] = ignore_index
    images = nn.functional.one_hot(predictions, num_classes=3).permute(0, 3, 1, 2).float()
    backbone = nn.Sequential(nn.Conv2d(3, 3, 1, bias=False))
    probe = SegmentationProbe(backbone, ["0"], num_classes=3)
    for module in probe.modules():
        if isinstance(module, nn.Conv2d):
            nn.init.dirac_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    solver = SegmentationSolver(probe, num_classes=3, device="cpu", ignore_index=ignore_index)
    loader = make_loader(images, masks, as_dict=as_dict, mask_4d=mask_4d)
    cache = probe.extract_segmentation_features(loader, cache_dtype=torch.float32)

    solver.evaluate(make_loader(images, predictions))
    probe.train()
    raw_result = solver.evaluate(loader, collect_confusions=collect_confusions)
    assert not probe.training
    assert not probe.backbone.training

    with mock.patch.object(
        backbone,
        "forward",
        side_effect=AssertionError("Cached evaluation must bypass the backbone"),
    ):
        solver.evaluate_cached(
            CachedFeaturesDataset(cache.layer_tensors, predictions), batch_size=3
        )
        probe.train()
        cached_result = solver.evaluate_cached(
            cache, batch_size=3, collect_confusions=collect_confusions
        )
    assert not probe.training
    assert not probe.backbone.training

    metric_args = {"num_classes": 3, "ignore_index": ignore_index}
    expected_metrics = {
        "mIoU": multiclass_jaccard_index(predictions, masks, average="macro", **metric_args).item(),
        "fw_IoU": multiclass_jaccard_index(
            predictions, masks, average="weighted", **metric_args
        ).item(),
        "precision": multiclass_precision(
            predictions, masks, average="macro", **metric_args
        ).item(),
        "recall": multiclass_recall(predictions, masks, average="macro", **metric_args).item(),
        "f1": multiclass_f1_score(predictions, masks, average="macro", **metric_args).item(),
    }
    if collect_confusions:
        expected_confusions = torch.stack(
            [
                multiclass_confusion_matrix(pred, mask, **metric_args)
                for pred, mask in zip(predictions, masks, strict=True)
            ]
        )
    for result in (raw_result, cached_result):
        if collect_confusions:
            assert isinstance(result, tuple)
            metrics, confusions = result
            torch.testing.assert_close(confusions, expected_confusions)
        else:
            metrics = result
        assert isinstance(metrics, dict)
        assert metrics == pytest.approx(expected_metrics)


def test_probe_fpn_head(mock_backbone, dummy_data):
    from torchgeo_bench.models.segmentation_heads import FPNHead

    probe = make_probe(mock_backbone, ["layer2", "layer1"], head_type="fpn", hidden_dim=16)

    assert isinstance(probe.head, FPNHead)
    assert hasattr(probe.head, "laterals")
    assert hasattr(probe.head, "fpn_convs")
    assert hasattr(probe.head, "fpn_head")

    logits = probe(dummy_data["image"])
    assert logits.shape == (2, NUM_CLASSES, 64, 64)


def test_probe_backbone_prefix_stripping(dummy_data):
    """Unprefixed layer names must resolve inside a BenchModel-style wrapper."""
    backbone = WrappedBackbone()
    probe = make_probe(backbone, ["layer1", "layer2"])
    logits = probe(dummy_data["image"])
    assert logits.shape == (2, NUM_CLASSES, 64, 64)


def test_probe_linear_multi_layer_weighted(mock_backbone, dummy_data):
    probe = make_probe(mock_backbone, ["layer1", "layer2"], head_type="linear")
    assert hasattr(probe.head, "scale_weights")
    logits = probe(dummy_data["image"])
    assert logits.shape == (2, NUM_CLASSES, 64, 64)


def test_probe_conv_block_multi_layer(mock_backbone, dummy_data):
    """Feature maps at different resolutions must align before concatenation."""
    probe = make_probe(mock_backbone, ["layer1", "layer2"], head_type="conv_block", hidden_dim=16)
    logits = probe(dummy_data["image"])
    assert logits.shape == (2, NUM_CLASSES, 64, 64)


def test_probe_unfrozen_backbone(mock_backbone, dummy_data):
    probe = make_probe(mock_backbone, ["layer1"], freeze=False)
    for param in probe.backbone.parameters():
        assert param.requires_grad is True
    logits = probe(dummy_data["image"])
    assert logits.shape == (2, NUM_CLASSES, 64, 64)


def test_probe_vit_token_features():
    """Turn ViT tokens into spatial feature maps before applying the segmentation head."""
    backbone = ViTBackbone()
    probe = make_probe(backbone, ["blocks"], head_type="linear")
    images = torch.randn(2, 3, 64, 64)
    logits = probe(images)
    assert logits.shape == (2, NUM_CLASSES, 64, 64)


def test_resolve_num_prefix_tokens_walks_module_tree():
    """Read prefix-token counts through wrappers rather than assuming a bare timm model."""

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
    """DINOv3's 261 tokens contain 256 patches, one CLS token, and four registers.

    Mistaking feature width 1024 for the token count would create a 32x32 grid with 261 channels.
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
    """Prefix metadata fixes the token axis; reject non-square token counts."""

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


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize(
    ("lr_scheduler", "expected_lrs"),
    [("none", [1e-3, 1e-3, 1e-3]), ("cosine", [1e-3, 0.0005005, 1e-6])],
)
def test_solver_training_schedule_and_frozen_backbone(
    lr_scheduler: str, expected_lrs: list[float], *, cached: bool
) -> None:
    """Both training paths step the scheduler per epoch and only update the head."""
    rng = torch.Generator().manual_seed(3)
    images = torch.randn(2, 3, 8, 8, generator=rng)
    masks = torch.randint(0, NUM_CLASSES, (2, 8, 8), generator=rng)
    masks[0, :2, :2] = 255
    loader = make_loader(images, masks)
    backbone = nn.Sequential(nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4))
    probe = make_probe(backbone, ["1"])
    solver = SegmentationSolver(
        model=probe, num_classes=NUM_CLASSES, lr=1e-3, device="cpu", lr_scheduler=lr_scheduler
    )
    backbone_before = {name: value.clone() for name, value in backbone.state_dict().items()}
    head_before = [param.detach().clone() for param in probe.head.parameters()]
    lrs = []

    def record_lr(_module: nn.Module, _inputs: tuple[object, ...]) -> None:
        lrs.append(solver.optimizer.param_groups[0]["lr"])

    hook = probe.head.register_forward_pre_hook(record_lr)
    if cached:
        cache = probe.extract_segmentation_features(loader, cache_dtype=torch.float32)
        result = solver.fit_cached(cache, batch_size=2, epochs=2, verbose=False)
    else:
        result = solver.fit(loader, epochs=2, verbose=False)
    hook.remove()

    assert result is None
    assert solver.val_history == []
    assert [*lrs, solver.optimizer.param_groups[0]["lr"]] == pytest.approx(expected_lrs)
    assert not backbone.training
    for name, value in backbone.state_dict().items():
        torch.testing.assert_close(value, backbone_before[name], rtol=0, atol=0)
    assert all(param.grad is None for param in backbone.parameters())
    assert any(
        not torch.equal(before, after)
        for before, after in zip(head_before, probe.head.parameters(), strict=True)
    )


def test_solver_dict_batches(mock_backbone, dummy_data):
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


class MockBackbone4Layer(nn.Module):
    """Four feature scales, as required by DPT."""

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
    backbone = MockBackbone()
    with pytest.raises(ValueError, match="DPTHead requires exactly 4 feature layers"):
        make_probe(backbone, layers=["layer1", "layer2"], head_type="dpt", hidden_dim=16)


def test_dpt_fusion_layer_shim_matches_reference():
    """These checks protect decoder behavior when transformers changes its private fusion API."""
    pytest.importorskip("transformers")
    from transformers.models.dpt.modeling_dpt import DPTPreActResidualLayer

    from torchgeo_bench.models.segmentation_heads import _dpt_fusion_layer

    layer = _dpt_fusion_layer(16)

    assert isinstance(layer.projection, nn.Conv2d)
    assert layer.projection.kernel_size == (1, 1)

    # Residual blocks must use pre-activation: ReLU, conv, ReLU, conv.
    assert isinstance(layer.residual_layer1, DPTPreActResidualLayer)
    assert isinstance(layer.residual_layer2, DPTPreActResidualLayer)

    # Each fusion stage doubles the spatial size.
    out = layer(torch.randn(1, 16, 7, 7))
    assert out.shape == (1, 16, 14, 14)

    # Resize the skip input to the main input, not the reverse.
    out = layer(torch.randn(1, 16, 7, 7), torch.randn(1, 16, 3, 3))
    assert out.shape == (1, 16, 14, 14)


def test_dpt_head_upsamples_purely_through_fusion_cascade():
    """Four fusion stages must reach 224x224 from 14x14 without relying on the final resize."""
    pytest.importorskip("transformers")
    from torchgeo_bench.models.segmentation_heads import DPTHead

    head = DPTHead([32, 32, 32, 32], num_classes=NUM_CLASSES, hidden_dim=16)
    features = [torch.randn(1, 32, 14, 14) for _ in range(4)]

    projected = [
        conv(norm(f)) for norm, conv, f in zip(head.input_norms, head.convs, features, strict=True)
    ]
    fused = head.ref[0](projected[0])
    for layer, feat in zip(head.ref[1:], projected[1:], strict=True):
        fused = layer(fused, feat)
    assert fused.shape[-2:] == (224, 224)

    logits = head(features, 224, 224)
    assert logits.shape == (1, NUM_CLASSES, 224, 224)


def test_patch_linear_head_output_shape():
    from torchgeo_bench.models.segmentation_heads import PatchLinearHead

    head = PatchLinearHead([16], num_classes=5)
    logits = head([torch.randn(2, 16, 4, 4)], 64, 64)

    assert logits.shape == (2, 5, 64, 64)
    assert torch.isfinite(logits).all()


def test_patch_linear_head_small_patch():
    """Infer patch size from the token grid rather than assuming 16-pixel patches."""
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
    from torchgeo_bench.models.segmentation_heads import PatchLinearHead

    probe = make_probe(ViTBackbone(), ["blocks"], head_type="patch_linear")
    images = torch.randn(2, 3, 64, 64)
    logits = probe(images)

    assert isinstance(probe.head, PatchLinearHead)
    assert logits.shape == (2, NUM_CLASSES, 64, 64)


def test_probe_patch_linear_cached_features():
    images = torch.randn(2, 3, 64, 64)
    masks = torch.randint(0, NUM_CLASSES, (2, 64, 64))
    loader = make_loader(images, masks)
    probe = make_probe(ViTBackbone(), ["blocks"], head_type="patch_linear")
    solver = SegmentationSolver(model=probe, num_classes=NUM_CLASSES, lr=1e-3, device="cpu")

    train_cache = probe.extract_segmentation_features(loader, cache_dtype=torch.float32)
    val_miou = solver.fit_cached(train_cache, val_cache=train_cache, batch_size=2, epochs=1)

    assert isinstance(val_miou, float)
    assert 0.0 <= val_miou <= 1.0


def test_extract_segmentation_features_returns_cached_dataset(mock_backbone, dummy_data):
    images, masks = dummy_data["image"], dummy_data["mask"]
    loader = make_loader(images, masks)
    probe = make_probe(mock_backbone, ["layer1", "layer2"])

    cache = probe.extract_segmentation_features(loader, cache_dtype=torch.float16)

    assert isinstance(cache, CachedFeaturesDataset)
    assert len(cache) == len(images)
    feats, mask = cache[0]
    assert len(feats) == 2
    assert feats[0].dtype == torch.float16
    assert mask.dtype == torch.int64


def test_solver_fit_cached(mock_backbone, dummy_data):
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


def _make_cpu_cache(mock_backbone, dummy_data):
    images, masks = dummy_data["image"], dummy_data["mask"]
    loader = make_loader(images, masks)
    probe = make_probe(mock_backbone, ["layer1", "layer2"])
    return probe.extract_segmentation_features(loader, cache_dtype=torch.float16)


def test_gpu_tensor_cache_from_cached_cpu(mock_backbone, dummy_data):
    """The device-cache path must also work without CUDA."""
    cache = _make_cpu_cache(mock_backbone, dummy_data)
    gpu_cache = GPUTensorCache.from_cached(cache, device="cpu")

    assert len(gpu_cache) == len(cache)
    assert len(gpu_cache.layer_tensors) == 2
    assert gpu_cache.layer_tensors[0].dtype == torch.float32  # CPU path uses float32
    assert gpu_cache.masks.dtype == torch.long
    assert gpu_cache.masks.shape == (len(cache), 64, 64)


def test_gpu_tensor_cache_shuffled_batches(mock_backbone, dummy_data):
    cache = _make_cpu_cache(mock_backbone, dummy_data)
    gpu_cache = GPUTensorCache.from_cached(cache, device="cpu")

    all_masks = []
    for feats, masks in gpu_cache.shuffled_batches(batch_size=1):
        assert len(feats) == 2
        assert feats[0].shape[0] == masks.shape[0]
        all_masks.append(masks)

    total = sum(m.shape[0] for m in all_masks)
    assert total == len(cache)


def test_gpu_tensor_cache_ordered_batches(mock_backbone, dummy_data):
    cache = _make_cpu_cache(mock_backbone, dummy_data)
    gpu_cache = GPUTensorCache.from_cached(cache, device="cpu")

    batches = list(gpu_cache.ordered_batches(batch_size=1))
    torch.testing.assert_close(torch.cat([masks for _, masks in batches]), cache.masks)
    for layer, expected in enumerate(cache.layer_tensors):
        actual = torch.cat([features[layer] for features, _ in batches])
        torch.testing.assert_close(actual, expected.float())


def test_solver_fit_cached_reuses_device_caches(mock_backbone, dummy_data):
    """Pre-built caches avoid both backbone calls and repeated device transfers."""
    images, masks = dummy_data["image"], dummy_data["mask"]
    loader = make_loader(images, masks)
    probe = make_probe(mock_backbone, ["layer1", "layer2"])
    solver = SegmentationSolver(model=probe, num_classes=NUM_CLASSES, lr=1e-3, device="cpu")

    train_cache = probe.extract_segmentation_features(loader, cache_dtype=torch.float32)
    gpu_train = GPUTensorCache.from_cached(train_cache, device="cpu")
    gpu_val = GPUTensorCache.from_cached(train_cache, device="cpu")

    with (
        mock.patch.object(
            mock_backbone,
            "forward",
            side_effect=AssertionError("Cached training must bypass the backbone"),
        ),
        mock.patch.object(
            GPUTensorCache,
            "from_cached",
            side_effect=AssertionError("Pre-built caches must not be transferred again"),
        ),
    ):
        val_miou = solver.fit_cached(
            gpu_train,
            val_cache=gpu_val,
            batch_size=2,
            epochs=1,
            verbose=False,
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
        gpu_train,
        val_cache=val_cache,
        batch_size=2,
        epochs=1,
        verbose=False,
    )

    assert isinstance(val_miou, float)


def test_probe_pools_time_series_features(mock_backbone, dummy_data):
    """PASTIS needs every date encoded before temporal pooling to retain seasonal information."""
    probe = SegmentationProbe(mock_backbone, ["layer1", "layer2"], NUM_CLASSES, head_type="fpn")
    images = dummy_data["image"]
    single = probe(images)
    series = probe(images.unsqueeze(1).repeat(1, 3, 1, 1, 1))
    assert series.shape == single.shape
    # Repeating one date must pool back to that date's logits.
    assert torch.allclose(series, single, atol=1e-4)


def test_probe_temporal_pool_max_differs_from_mean(mock_backbone, dummy_data):
    images = dummy_data["image"]
    series = torch.stack([images, images * 0.5], dim=1)
    mean_probe = SegmentationProbe(
        mock_backbone, ["layer1"], NUM_CLASSES, head_type="fpn", temporal_pool="mean"
    )
    max_probe = SegmentationProbe(
        mock_backbone, ["layer1"], NUM_CLASSES, head_type="fpn", temporal_pool="max"
    )
    max_probe.head.load_state_dict(mean_probe.head.state_dict())
    assert not torch.allclose(mean_probe(series), max_probe(series), atol=1e-5)


def test_probe_rejects_unknown_temporal_pool(mock_backbone):
    with pytest.raises(ValueError, match="temporal_pool"):
        SegmentationProbe(mock_backbone, ["layer1"], NUM_CLASSES, temporal_pool="median")
