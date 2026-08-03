import math

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from torchgeo_bench.main import _build_seg_probe_and_solver
from torchgeo_bench.segmentation_probe import (
    CachedFeaturesDataset,
    GPUTensorCache,
    SegmentationProbe,
    _estimate_cache_bytes,
)
from torchgeo_bench.segmentation_task import SegmentationSolver

NUM_CLASSES = 5


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


class BatchNormBackbone(nn.Module):
    """Backbone carrying BatchNorm, so train/eval mode is observable in behavior."""

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layer2(self.layer1(x))


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


class ViTRegisterBackbone(nn.Module):
    """ViT emitting CLS + 4 register tokens ahead of the patch tokens, like DINOv3."""

    num_prefix_tokens = 5

    def __init__(self):
        super().__init__()
        # Embed dim is deliberately NOT a perfect square: a square C lets the
        # (B, C, L) fallback branch "succeed" on a (B, L, C) tensor and silently
        # mis-grid the tokens, which would mask the bug this fixture exists for.
        self.patch_embed = nn.Conv2d(3, 24, kernel_size=16, stride=16)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 24))
        self.reg_token = nn.Parameter(torch.zeros(1, 4, 24))
        self.blocks = nn.Identity()

    def forward(self, x):
        x = self.patch_embed(x)  # (B, 16, H/16, W/16)
        B = x.shape[0]
        x = x.flatten(2).transpose(1, 2)  # (B, L, C)
        prefix = torch.cat([self.cls_token, self.reg_token], dim=1).expand(B, -1, -1)
        x = torch.cat([prefix, x], dim=1)  # (B, 5 + L, C)
        return self.blocks(x)


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
    # Batch=2, Channels=3, H=64, W=64
    images = torch.randn(2, 3, 64, 64)
    # Batch=2, H=64, W=64 (values 0-4)
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
        _build_seg_probe_and_solver(
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
    _, solver = _build_seg_probe_and_solver(
        mock_backbone,
        num_classes=NUM_CLASSES,
        eval_cfg=eval_cfg,
        device=torch.device("cpu"),
        lr=1e-3,
    )
    assert solver.ignore_index == 7


def test_build_seg_solver_rejects_ignore_index_mismatch(mock_backbone):
    """Loss and metric ignore indices must not silently diverge."""
    eval_cfg = OmegaConf.create(
        {
            "segmentation": {
                "layers": ["layer1"],
                "head_type": "linear",
                "ignore_index": 255,
                "criterion": {"_target_": "torch.nn.CrossEntropyLoss", "ignore_index": 7},
                "lr_scheduler": "none",
            }
        }
    )
    with pytest.raises(ValueError, match="ignore_index mismatch"):
        _build_seg_probe_and_solver(
            mock_backbone,
            num_classes=NUM_CLASSES,
            eval_cfg=eval_cfg,
            device=torch.device("cpu"),
            lr=1e-3,
        )


def test_probe_dry_run_exception_handling():
    """Test that dry_run_channels catches exceptions from the backbone."""

    class BrokenBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy_layer = nn.Linear(2, 2)

        def forward(self, x):
            del x
            raise RuntimeError("Backbone crash")

    backbone = BrokenBackbone()

    with pytest.raises(RuntimeError):
        SegmentationProbe(backbone, ["layer1"], 2)


def test_segmentation_probe_initialization(mock_backbone, dummy_data):
    """Test if the probe initializes correctly and freezes the backbone."""
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
    assert set(metrics.keys()) == {
        "mIoU",
        "fw_IoU",
        "per_class_IoU",
        "precision",
        "recall",
        "f1",
    }
    assert 0.0 <= metrics["mIoU"] <= 1.0


# ---------------------------------------------------------------------------
# Training stability: gradient clipping + non-finite guard
# ---------------------------------------------------------------------------


def _stability_solver(mock_backbone, **kwargs):
    probe = SegmentationProbe(
        backbone=mock_backbone, layer_names=["layer1", "layer2"], num_classes=NUM_CLASSES
    )
    return SegmentationSolver(model=probe, num_classes=NUM_CLASSES, lr=1e-3, device="cpu", **kwargs)


def test_train_on_batch_clips_gradients(mock_backbone, dummy_data, monkeypatch):
    """Gradients are clipped to max_grad_norm before the optimizer steps."""
    solver = _stability_solver(mock_backbone, max_grad_norm=1.0)

    observed = {}
    real_clip = torch.nn.utils.clip_grad_norm_

    def recording_clip(params, max_norm, **kw):
        params = list(params)
        total = real_clip(params, max_norm, **kw)
        observed["max_norm"] = max_norm
        # Norm *after* clipping is what actually reaches the optimizer.
        observed["after"] = torch.nn.utils.get_total_norm(
            [p.grad for p in params if p.grad is not None]
        ).item()
        return total

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", recording_clip)

    # A large-loss batch: without clipping this produces a big gradient norm.
    batch = (dummy_data["image"], dummy_data["mask"])
    solver.model.train()
    solver._train_on_batch(batch)

    assert observed["max_norm"] == 1.0
    assert observed["after"] <= 1.0 + 1e-4


def test_clipping_happens_after_unscale(mock_backbone, dummy_data, monkeypatch):
    """Clipping must land between scaler.unscale_ and scaler.step.

    Clipping still-scaled gradients would threshold against the GradScaler's
    current scale factor — an effectively random magnitude that changes whenever
    the scaler backs off. CPU tests run with AMP disabled, so the ordering is
    asserted against a recording stand-in scaler rather than requiring a GPU.
    """
    solver = _stability_solver(mock_backbone, max_grad_norm=1.0)

    order = []

    class RecordingScaler:
        def scale(self, loss):
            order.append("scale")
            return loss

        def unscale_(self, optimizer):
            order.append("unscale_")

        def step(self, optimizer):
            order.append("step")
            optimizer.step()

        def update(self, *args):
            order.append("update")

    solver.scaler = RecordingScaler()

    real_clip = torch.nn.utils.clip_grad_norm_
    monkeypatch.setattr(
        torch.nn.utils,
        "clip_grad_norm_",
        lambda p, max_norm, **k: (order.append("clip"), real_clip(p, max_norm, **k))[1],
    )

    solver.model.train()
    solver._train_on_batch((dummy_data["image"], dummy_data["mask"]))

    assert order == ["scale", "unscale_", "clip", "step", "update"]


def test_train_on_batch_clipping_can_be_disabled(mock_backbone, dummy_data, monkeypatch):
    """max_grad_norm=None skips clipping entirely."""
    solver = _stability_solver(mock_backbone, max_grad_norm=None)

    called = []
    monkeypatch.setattr(
        torch.nn.utils,
        "clip_grad_norm_",
        lambda *a, **k: called.append(True),
    )

    solver.model.train()
    solver._train_on_batch((dummy_data["image"], dummy_data["mask"]))

    assert called == []


def test_train_on_batch_raises_on_non_finite_loss(mock_backbone, dummy_data):
    """A NaN loss raises NonFiniteLossError instead of poisoning the optimizer."""
    from torchgeo_bench.segmentation_task import NonFiniteLossError

    solver = _stability_solver(mock_backbone)

    class NanCriterion(nn.Module):
        def forward(self, logits, masks):
            return logits.sum() * float("nan")

    solver.criterion = NanCriterion()
    solver.model.train()

    with pytest.raises(NonFiniteLossError, match="Non-finite training loss"):
        solver._train_on_batch((dummy_data["image"], dummy_data["mask"]))


def test_non_finite_loss_raised_before_optimizer_is_touched(mock_backbone, dummy_data):
    """The guard fires before any state reaches AdamW.

    This is the whole point of catching at the first occurrence: once a
    non-finite value lands in exp_avg/exp_avg_sq, every later step is poisoned.
    """
    from torchgeo_bench.segmentation_task import NonFiniteLossError

    solver = _stability_solver(mock_backbone)

    class NanCriterion(nn.Module):
        def forward(self, logits, masks):
            return logits.sum() * float("nan")

    solver.criterion = NanCriterion()
    solver.model.train()

    with pytest.raises(NonFiniteLossError):
        solver._train_on_batch((dummy_data["image"], dummy_data["mask"]))

    # AdamW populates per-parameter state lazily on its first step; an empty
    # state dict proves no step ran on the non-finite loss.
    assert all(not st for st in solver.optimizer.state.values())


def test_fit_cached_is_not_clipped(mock_backbone, dummy_data, monkeypatch):
    """fit_cached is deliberately untouched: clipping it would invalidate the LP sweep."""
    solver = _stability_solver(mock_backbone, max_grad_norm=1.0)

    called = []
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", lambda *a, **k: called.append(True))

    features = [torch.randn(4, c, 8, 8) for c in solver.model.channels_list]
    masks = torch.randint(0, NUM_CLASSES, (4, 8, 8))
    cached = CachedFeaturesDataset(features, masks)
    solver.fit_cached(cached, epochs=1, batch_size=2, verbose=False)

    assert called == []


# ---------------------------------------------------------------------------
# Probe: FPN head
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Probe: backbone.* layer name stripping
# ---------------------------------------------------------------------------


def test_probe_backbone_prefix_stripping(dummy_data):
    """Layer names prefixed with 'backbone.' are correctly resolved in wrapped models."""
    backbone = WrappedBackbone()
    # The inner layers are at backbone.layer1 / backbone.layer2 inside the wrapper,
    # but SegmentationProbe should strip the leading 'backbone.' prefix so that
    # specifying ["layer1"] still works.
    probe = make_probe(backbone, ["layer1", "layer2"])
    logits = probe(dummy_data["image"])
    assert logits.shape == (2, NUM_CLASSES, 64, 64)


# ---------------------------------------------------------------------------
# Probe: single-layer linear
# ---------------------------------------------------------------------------


def test_probe_linear_single_layer(mock_backbone, dummy_data):
    """Single-layer linear probe returns logits without scale_weights."""
    probe = make_probe(mock_backbone, ["layer1"], head_type="linear")
    assert not hasattr(probe, "scale_weights")
    logits = probe(dummy_data["image"])
    assert logits.shape == (2, NUM_CLASSES, 64, 64)


# ---------------------------------------------------------------------------
# Probe: multi-layer linear
# ---------------------------------------------------------------------------


def test_probe_linear_multi_layer_weighted(mock_backbone, dummy_data):
    """Multi-layer linear probe uses scale_weights and returns correct shape."""
    probe = make_probe(mock_backbone, ["layer1", "layer2"], head_type="linear")
    assert hasattr(probe.head, "scale_weights")
    logits = probe(dummy_data["image"])
    assert logits.shape == (2, NUM_CLASSES, 64, 64)


# ---------------------------------------------------------------------------
# Probe: conv_block with multiple layers
# ---------------------------------------------------------------------------


def test_probe_conv_block_multi_layer(mock_backbone, dummy_data):
    """conv_block with two layers at different resolutions triggers interpolation alignment."""
    probe = make_probe(mock_backbone, ["layer1", "layer2"], head_type="conv_block", hidden_dim=16)
    logits = probe(dummy_data["image"])
    assert logits.shape == (2, NUM_CLASSES, 64, 64)


# ---------------------------------------------------------------------------
# Probe: unfrozen backbone forward path
# ---------------------------------------------------------------------------


def test_probe_unfrozen_backbone(mock_backbone, dummy_data):
    """With freeze_backbone=False the backbone runs in train mode and grads flow."""
    probe = make_probe(mock_backbone, ["layer1"], freeze=False)
    for param in probe.backbone.parameters():
        assert param.requires_grad is True
    logits = probe(dummy_data["image"])
    assert logits.shape == (2, NUM_CLASSES, 64, 64)


# ---------------------------------------------------------------------------
# Probe: ViT-style (B, L, C) token features via _process_feature
# ---------------------------------------------------------------------------


def test_probe_vit_token_features():
    """ViT backbone emitting (B, L, C) tokens is correctly reshaped to (B, C, H, H)."""
    backbone = ViTBackbone()
    # 'blocks' is an Identity that passes through (B, L, C); hook it directly
    probe = make_probe(backbone, ["blocks"], head_type="linear")
    images = torch.randn(2, 3, 64, 64)
    logits = probe(images)
    assert logits.shape == (2, NUM_CLASSES, 64, 64)


def test_probe_vit_register_token_features():
    """A DINOv3-style backbone with CLS + register tokens grids on num_prefix_tokens.

    Assuming a single CLS token leaves L=s^2+5 matching neither the square nor
    the square+1 branch, so these models previously failed to grid at all.
    """
    backbone = ViTRegisterBackbone()
    probe = make_probe(backbone, ["blocks"], head_type="linear")
    images = torch.randn(2, 3, 64, 64)
    logits = probe(images)
    assert logits.shape == (2, NUM_CLASSES, 64, 64)
    # feature_hw_list comes from the 224px dry run: 224/16 = 14, so the 5 prefix
    # tokens must have been dropped off a 201-token stream to leave a 14x14 grid.
    assert probe.feature_hw_list == [(14, 14)]


def test_probe_dry_run_uses_backbone_num_channels():
    """Dry-run channel inference supports non-RGB benchmark models."""
    backbone = TwoChannelBackbone()
    probe = make_probe(backbone, [], head_type="linear")
    images = torch.randn(2, 2, 64, 64)
    logits = probe(images)
    assert logits.shape == (2, NUM_CLASSES, 64, 64)


# ---------------------------------------------------------------------------
# Solver: no LR scheduler path
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Solver: dict-format batches in fit and evaluate
# ---------------------------------------------------------------------------


def test_solver_dict_batches(mock_backbone, dummy_data):
    """fit and evaluate both handle dict-format batches {"image": ..., "mask": ...}."""
    images, masks = dummy_data["image"], dummy_data["mask"]
    loader = make_loader(images, masks, as_dict=True)
    probe = make_probe(mock_backbone, ["layer1"])
    solver = SegmentationSolver(model=probe, num_classes=NUM_CLASSES, lr=1e-3, device="cpu")
    solver.fit(loader, epochs=1, verbose=False)
    metrics = solver.evaluate(loader)
    assert 0.0 <= metrics["mIoU"] <= 1.0


# ---------------------------------------------------------------------------
# Solver: 4D mask squeezing in fit and evaluate
# ---------------------------------------------------------------------------


def test_solver_4d_masks(mock_backbone, dummy_data):
    """fit and evaluate both squeeze (B, 1, H, W) masks to (B, H, W)."""
    images, masks = dummy_data["image"], dummy_data["mask"]
    loader = make_loader(images, masks, mask_4d=True)
    probe = make_probe(mock_backbone, ["layer1"])
    solver = SegmentationSolver(model=probe, num_classes=NUM_CLASSES, lr=1e-3, device="cpu")
    solver.fit(loader, epochs=1, verbose=False)
    metrics = solver.evaluate(loader)
    assert 0.0 <= metrics["mIoU"] <= 1.0


# ---------------------------------------------------------------------------
# Solver: val_loader passed to fit returns mIoU
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Probe: DPT head
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Probe: PatchLinear head
# ---------------------------------------------------------------------------


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


def test_patch_linear_head_size_robust_across_calls():
    """The decode factor is fixed on the first call and reused at a different native size.

    Regression for the auto-resize ViT path: a fixed-224 ViT is dry-run at the
    224 model frame (grid 14 -> P=16), then scores 650px tiles (native frame).
    The head must not re-derive P from 650/14 and must still emit native-frame
    logits — the trailing interpolate absorbs the size difference.
    """
    from torchgeo_bench.models.segmentation_heads import PatchLinearHead

    head = PatchLinearHead([8], num_classes=3)
    # First (dry-run) call at the model frame fixes P = 224/14 = 16.
    _ = head([torch.randn(1, 8, 14, 14)], 224, 224)
    assert head.patch_size == 16

    # A subsequent call at a larger native frame must not raise and must reuse P.
    logits = head([torch.randn(2, 8, 14, 14)], 650, 650)
    assert head.patch_size == 16  # unchanged
    assert logits.shape == (2, 3, 650, 650)
    assert torch.isfinite(logits).all()


def test_patch_linear_head_explicit_decode_patch_size():
    """An explicit decode_patch_size pins P regardless of the token grid / input."""
    from torchgeo_bench.models.segmentation_heads import PatchLinearHead

    head = PatchLinearHead([8], num_classes=3, decode_patch_size=16)
    logits = head([torch.randn(2, 8, 14, 14)], 650, 650)
    assert head.patch_size == 16
    assert logits.shape == (2, 3, 650, 650)


def test_patch_linear_head_ignores_extra_channels():
    """PatchLinearHead uses only the first feature map when extra layers are passed."""
    from torchgeo_bench.models.segmentation_heads import PatchLinearHead

    head = PatchLinearHead([8, 16], num_classes=3)
    logits = head([torch.randn(2, 8, 16, 16), torch.randn(2, 16, 8, 8)], 64, 64)

    assert logits.shape == (2, 3, 64, 64)
    assert torch.isfinite(logits).all()


def test_patch_linear_head_has_expected_attributes():
    """PatchLinearHead exposes the expected norm and projection layers."""
    from torchgeo_bench.models.segmentation_heads import ChannelLayerNorm, PatchLinearHead

    head = PatchLinearHead([16], num_classes=5)
    head([torch.randn(2, 16, 4, 4)], 64, 64)

    assert isinstance(head.norm, ChannelLayerNorm)
    assert isinstance(head.conv, nn.Conv2d)
    assert head.conv.out_channels == 5 * 16 * 16


def test_probe_patch_linear_head_vit():
    """Patch-linear probe works end-to-end with ViT token features."""
    from torchgeo_bench.models.segmentation_heads import PatchLinearHead

    probe = make_probe(ViTBackbone(), ["blocks"], head_type="patch_linear")
    images = torch.randn(2, 3, 64, 64)
    logits = probe(images)

    assert isinstance(probe.head, PatchLinearHead)
    assert logits.shape == (2, NUM_CLASSES, 64, 64)


def test_probe_patch_linear_head_attributes():
    """Patch-linear probe head exposes the expected normalization and conv layers."""
    from torchgeo_bench.models.segmentation_heads import ChannelLayerNorm, PatchLinearHead

    probe = make_probe(ViTBackbone(), ["blocks"], head_type="patch_linear")

    assert isinstance(probe.head, PatchLinearHead)
    assert isinstance(probe.head.norm, ChannelLayerNorm)
    assert isinstance(probe.head.conv, nn.Conv2d)


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


# ---------------------------------------------------------------------------
# Feature caching: extract_segmentation_features + CachedFeaturesDataset
# ---------------------------------------------------------------------------


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


def test_cached_features_dataset_indexing(mock_backbone, dummy_data):
    """CachedFeaturesDataset returns correct per-sample features and masks."""
    images, masks = dummy_data["image"], dummy_data["mask"]
    loader = make_loader(images, masks)
    probe = make_probe(mock_backbone, ["layer1", "layer2"])
    cache = probe.extract_segmentation_features(loader, cache_dtype=torch.float32)

    feats, mask = cache[0]
    assert len(feats) == 2  # two layers
    assert mask.shape == (64, 64)


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


# ---------------------------------------------------------------------------
# GPUTensorCache
# ---------------------------------------------------------------------------


def _make_cpu_cache(mock_backbone, dummy_data):
    """Helper: extract a CachedFeaturesDataset on CPU."""
    images, masks = dummy_data["image"], dummy_data["mask"]
    loader = make_loader(images, masks)
    probe = make_probe(mock_backbone, ["layer1", "layer2"])
    return probe.extract_segmentation_features(loader, cache_dtype=torch.float16)


def test_estimate_cache_bytes(mock_backbone, dummy_data):
    """_estimate_cache_bytes returns a positive integer for a non-empty cache."""
    cache = _make_cpu_cache(mock_backbone, dummy_data)
    size = _estimate_cache_bytes(cache)
    assert size > 0
    assert isinstance(size, int)


def test_estimate_cache_bytes_empty():
    """_estimate_cache_bytes returns 0 for an empty cache."""
    empty = CachedFeaturesDataset([], [])
    assert _estimate_cache_bytes(empty) == 0


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


# ---------------------------------------------------------------------------
# Solver: differential LR param groups + fixed step budget
# ---------------------------------------------------------------------------


def test_solver_differential_lr_param_groups(mock_backbone, dummy_data):
    """Passing backbone_lr and head_lr yields two param groups with those LRs."""
    del dummy_data
    probe = make_probe(mock_backbone, ["layer1", "layer2"], freeze=False)
    solver = SegmentationSolver(
        model=probe,
        num_classes=NUM_CLASSES,
        backbone_lr=1e-5,
        head_lr=1e-3,
        device="cpu",
    )

    groups = solver.optimizer.param_groups
    assert len(groups) == 2
    lrs = {g["lr"] for g in groups}
    assert lrs == {1e-5, 1e-3}

    backbone_params = {id(p) for p in probe.backbone.parameters()}
    backbone_group = next(g for g in groups if g["lr"] == 1e-5)
    assert backbone_group["params"], "backbone group must be non-empty"
    assert all(id(p) in backbone_params for p in backbone_group["params"])


def test_solver_fixed_step_budget(mock_backbone, dummy_data):
    """fit(max_steps=N) runs exactly N optimizer steps regardless of epoch count."""
    images, masks = dummy_data["image"], dummy_data["mask"]
    loader = make_loader(images, masks)  # 2 samples, batch_size=2 -> 1 batch/epoch
    probe = make_probe(mock_backbone, ["layer1", "layer2"])
    solver = SegmentationSolver(model=probe, num_classes=NUM_CLASSES, lr=1e-3, device="cpu")

    step_count = 0
    real_step = solver.optimizer.step

    def counting_step(*args, **kwargs):
        nonlocal step_count
        step_count += 1
        return real_step(*args, **kwargs)

    solver.optimizer.step = counting_step
    solver.fit(loader, max_steps=3, verbose=False)

    assert step_count == 3


def test_fixed_budget_uninstrumented_records_no_history(mock_backbone, dummy_data):
    """Without eval_every the fixed-budget path stays silent: no history, no val passes."""
    images, masks = dummy_data["image"], dummy_data["mask"]
    loader = make_loader(images, masks)
    probe = make_probe(mock_backbone, ["layer1", "layer2"])
    solver = SegmentationSolver(model=probe, num_classes=NUM_CLASSES, lr=1e-3, device="cpu")

    eval_calls = 0
    real_evaluate = solver.evaluate

    def counting_evaluate(*args, **kwargs):
        nonlocal eval_calls
        eval_calls += 1
        return real_evaluate(*args, **kwargs)

    solver.evaluate = counting_evaluate
    # A val_loader alone must not trigger evaluation — eval_every gates it.
    result = solver.fit(loader, val_loader=loader, max_steps=3, verbose=False)

    assert eval_calls == 0
    assert solver.history == []
    assert solver.val_history == []
    assert result is None


def test_fixed_budget_history_records_curve(mock_backbone, dummy_data):
    """eval_every populates history with (step, train_loss, val_mIoU, per_class_IoU)."""
    images, masks = dummy_data["image"], dummy_data["mask"]
    loader = make_loader(images, masks)  # 1 batch/epoch
    probe = make_probe(mock_backbone, ["layer1", "layer2"])
    solver = SegmentationSolver(model=probe, num_classes=NUM_CLASSES, lr=1e-3, device="cpu")

    last = solver.fit(loader, val_loader=loader, max_steps=6, eval_every=2, verbose=False)

    # Evaluations at steps 2, 4, 6 — the final step coincides with an eval_every
    # multiple, so it must not be recorded twice.
    assert [h["step"] for h in solver.history] == [2, 4, 6]
    assert all(len(h["per_class_IoU"]) == NUM_CLASSES for h in solver.history)
    assert all(math.isfinite(h["train_loss"]) for h in solver.history)
    assert solver.val_history == [h["val_mIoU"] for h in solver.history]
    assert last == solver.history[-1]["val_mIoU"]


def test_fixed_budget_always_evaluates_final_step(mock_backbone, dummy_data):
    """The final step is always recorded, even when it is not an eval_every multiple."""
    images, masks = dummy_data["image"], dummy_data["mask"]
    loader = make_loader(images, masks)
    probe = make_probe(mock_backbone, ["layer1", "layer2"])
    solver = SegmentationSolver(model=probe, num_classes=NUM_CLASSES, lr=1e-3, device="cpu")

    solver.fit(loader, val_loader=loader, max_steps=5, eval_every=2, verbose=False)

    assert [h["step"] for h in solver.history] == [2, 4, 5]


def test_fixed_budget_step_callback_prunes(mock_backbone, dummy_data):
    """A step_callback returning True stops training at that evaluation."""
    images, masks = dummy_data["image"], dummy_data["mask"]
    loader = make_loader(images, masks)
    probe = make_probe(mock_backbone, ["layer1", "layer2"])
    solver = SegmentationSolver(model=probe, num_classes=NUM_CLASSES, lr=1e-3, device="cpu")

    step_count = 0
    real_step = solver.optimizer.step

    def counting_step(*args, **kwargs):
        nonlocal step_count
        step_count += 1
        return real_step(*args, **kwargs)

    solver.optimizer.step = counting_step
    seen: list[int] = []

    def callback(step, val_miou):
        del val_miou
        seen.append(step)
        return step >= 4

    solver.fit(
        loader,
        val_loader=loader,
        max_steps=100,
        eval_every=2,
        step_callback=callback,
        verbose=False,
    )

    assert seen == [2, 4]
    assert step_count == 4


def test_fixed_budget_restores_train_mode_after_eval(dummy_data):
    """A mid-pass eval must not leave the model in eval mode for the rest of the pass.

    ``evaluate()`` calls ``model.eval()`` and does not restore; the fixed-budget
    loop only calls ``train()`` once per loader pass, so without an explicit
    restore the remainder of that pass would train with eval-mode
    BatchNorm/dropout — silently wrong, and invisible in the loss.
    """
    images = torch.randn(8, 3, 64, 64)
    masks = torch.randint(0, NUM_CLASSES, (8, 64, 64))
    loader = make_loader(images, masks)  # 4 batches/epoch
    probe = make_probe(BatchNormBackbone(), ["layer1", "layer2"], freeze=False)
    solver = SegmentationSolver(model=probe, num_classes=NUM_CLASSES, lr=1e-3, device="cpu")

    modes: list[bool] = []
    real_train_on_batch = solver._train_on_batch

    def recording_train_on_batch(batch):
        modes.append(solver.model.training)
        return real_train_on_batch(batch)

    solver._train_on_batch = recording_train_on_batch
    # eval_every=1 evaluates after every step, so every step after the first
    # follows an ``evaluate()`` call within the same loader pass.
    solver.fit(loader, val_loader=loader, max_steps=4, eval_every=1, verbose=False)

    assert all(modes), "model must be in train mode for every training step"


def test_fixed_budget_eval_keeps_frozen_backbone_in_eval_mode(dummy_data):
    """Restoring train mode must not un-freeze a frozen backbone's BatchNorm."""
    images = torch.randn(8, 3, 64, 64)
    masks = torch.randint(0, NUM_CLASSES, (8, 64, 64))
    loader = make_loader(images, masks)
    probe = make_probe(BatchNormBackbone(), ["layer1", "layer2"], freeze=True)
    solver = SegmentationSolver(model=probe, num_classes=NUM_CLASSES, lr=1e-3, device="cpu")

    backbone_modes: list[bool] = []
    real_train_on_batch = solver._train_on_batch

    def recording_train_on_batch(batch):
        backbone_modes.append(solver.model.backbone.training)
        return real_train_on_batch(batch)

    solver._train_on_batch = recording_train_on_batch
    solver.fit(loader, val_loader=loader, max_steps=4, eval_every=1, verbose=False)

    assert not any(backbone_modes), "frozen backbone must stay in eval mode"


def test_evaluate_reports_per_class_iou(mock_backbone, dummy_data):
    """evaluate() returns per-class IoU alongside the scalar aggregates."""
    images, masks = dummy_data["image"], dummy_data["mask"]
    loader = make_loader(images, masks)
    probe = make_probe(mock_backbone, ["layer1", "layer2"])
    solver = SegmentationSolver(model=probe, num_classes=NUM_CLASSES, lr=1e-3, device="cpu")

    metrics = solver.evaluate(loader)

    per_class = metrics["per_class_IoU"]
    assert isinstance(per_class, list)
    assert len(per_class) == NUM_CLASSES
    # macro mIoU is the mean over classes, so the two must agree.
    assert metrics["mIoU"] == pytest.approx(sum(per_class) / NUM_CLASSES, abs=1e-6)


def test_solver_single_lr_backward_compatible(mock_backbone, dummy_data):
    """A single lr keeps the default single param-group optimizer."""
    del dummy_data
    probe = make_probe(mock_backbone, ["layer1", "layer2"], freeze=False)
    solver = SegmentationSolver(model=probe, num_classes=NUM_CLASSES, lr=1e-3, device="cpu")
    assert len(solver.optimizer.param_groups) == 1
    assert solver.optimizer.param_groups[0]["lr"] == 1e-3
