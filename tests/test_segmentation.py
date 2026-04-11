import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from torchgeo_bench.segmentation_probe import SegmentationProbe
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

    for param in probe.heads.parameters():
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
    # conv_block head uses projectors + a final Conv2d head
    assert hasattr(probe, "projectors")
    assert hasattr(probe, "head")
    assert isinstance(probe.head, nn.Conv2d)


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

    miou = solver.evaluate(loader)

    assert isinstance(miou, float)
    assert 0.0 <= miou <= 1.0


# ---------------------------------------------------------------------------
# Probe: FPN head
# ---------------------------------------------------------------------------


def test_probe_fpn_head(mock_backbone, dummy_data):
    """FPN head forward pass produces correct output shape and has expected attributes."""
    probe = make_probe(mock_backbone, ["layer2", "layer1"], head_type="fpn", hidden_dim=16)

    assert hasattr(probe, "laterals")
    assert hasattr(probe, "fpn_convs")
    assert hasattr(probe, "fpn_head")

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
    assert hasattr(probe, "scale_weights")
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
    miou = solver.evaluate(loader)
    assert 0.0 <= miou <= 1.0


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
    miou = solver.evaluate(loader)
    assert 0.0 <= miou <= 1.0


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
