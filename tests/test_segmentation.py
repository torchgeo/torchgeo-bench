import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from torchgeo_bench.segmentation_probe import SegmentationProbe
from torchgeo_bench.segmentation_task import SegmentationSolver


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


@pytest.fixture
def mock_backbone():
    return MockBackbone()


@pytest.fixture
def dummy_data():
    # Batch=2, Channels=3, H=64, W=64
    images = torch.randn(2, 3, 64, 64)
    # Batch=2, H=64, W=64 (values 0-4)
    masks = torch.randint(0, 5, (2, 64, 64))
    return {"image": images, "mask": masks}


def test_probe_unknown_head_type(mock_backbone):
    """Test that invalid head_type does not create head modules."""
    probe = SegmentationProbe(
        backbone=mock_backbone, layer_names=["layer1"], num_classes=2, head_type="invalid_type"
    )
    # Unknown head_type skips both branches, so neither 'heads' nor 'head' is created
    assert not hasattr(probe, "heads")
    assert not hasattr(probe, "head")


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

    num_classes = 5
    probe = SegmentationProbe(
        backbone=mock_backbone, layer_names=["layer1", "layer2"], num_classes=num_classes
    )

    solver = SegmentationSolver(model=probe, num_classes=num_classes, lr=1e-3, device="cpu")

    solver.fit(loader, epochs=1, verbose=True)

    miou = solver.evaluate(loader)

    assert isinstance(miou, float)
    assert 0.0 <= miou <= 1.0
