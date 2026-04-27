import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

from torchgeo_bench.sanity_checks import run_overfit_check
from torchgeo_bench.segmentation_probe import SegmentationProbe

NUM_CLASSES = 3
IMG_SIZE = 64
BATCH_SIZE = 4


class SimpleBackbone(nn.Module):
    """Simple CNN backbone with hookable intermediate layers."""

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=2), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2), nn.ReLU())

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class ZeroBackbone(nn.Module):
    """Backbone that returns all-zero features — simulates a degenerate encoder."""

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=2), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2), nn.ReLU())

    def forward(self, x):
        # Still call layers so hooks fire, but zero out the outputs
        h = self.layer1(x)
        h = self.layer2(h)
        return torch.zeros_like(h)


def _make_loader(backbone_class, n_samples=8):
    """Create a DataLoader of synthetic (image, mask) pairs.

    Masks are spatially constant per image (one class fills the whole image),
    which makes them easily memorizable by a small head.
    """
    images = torch.randn(n_samples, 3, IMG_SIZE, IMG_SIZE)
    # Assign a constant class per image — trivially overfittable
    class_ids = torch.arange(n_samples) % NUM_CLASSES
    masks = class_ids[:, None, None].expand(n_samples, IMG_SIZE, IMG_SIZE).clone()
    dataset = TensorDataset(images, masks)

    def collate(batch):
        imgs = torch.stack([b[0] for b in batch])
        msks = torch.stack([b[1] for b in batch])
        return {"image": imgs, "mask": msks}

    return DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate)


def _make_probe(backbone, head_type="linear"):
    return SegmentationProbe(
        backbone=backbone,
        layer_names=["layer1", "layer2"],
        num_classes=NUM_CLASSES,
        head_type=head_type,
        freeze_backbone=True,
    )


@pytest.fixture
def check_cfg():
    return OmegaConf.create({
        "overfit_n_batches": 2,
        "overfit_steps": 300,
        "overfit_threshold": 0.5,
        "overfit_lr": 1e-2,
    })


def test_overfit_check_passes_with_functional_encoder(check_cfg):
    backbone = SimpleBackbone()
    probe = _make_probe(backbone, head_type="linear")
    loader = _make_loader(SimpleBackbone)

    result = run_overfit_check(
        probe=probe,
        train_loader=loader,
        num_classes=NUM_CLASSES,
        device=torch.device("cpu"),
        check_cfg=check_cfg,
    )

    assert result["n_batches"] == 2
    assert result["steps"] == 300
    assert result["threshold"] == 0.9
    assert isinstance(result["achieved_miou"], float)
    assert result["passed"], (
        f"Expected overfit check to pass for a functional backbone, "
        f"got mIoU={result['achieved_miou']:.3f}"
    )


def test_overfit_check_fails_with_zero_backbone(check_cfg):
    backbone = ZeroBackbone()
    probe = _make_probe(backbone, head_type="linear")
    loader = _make_loader(ZeroBackbone)

    result = run_overfit_check(
        probe=probe,
        train_loader=loader,
        num_classes=NUM_CLASSES,
        device=torch.device("cpu"),
        check_cfg=check_cfg,
    )

    assert not result["passed"], (
        f"Expected overfit check to fail for a zero-output backbone, "
        f"got mIoU={result['achieved_miou']:.3f}"
    )


def test_overfit_check_result_keys(check_cfg):
    backbone = SimpleBackbone()
    probe = _make_probe(backbone)
    loader = _make_loader(SimpleBackbone)

    result = run_overfit_check(
        probe=probe,
        train_loader=loader,
        num_classes=NUM_CLASSES,
        device=torch.device("cpu"),
        check_cfg=check_cfg,
    )

    assert set(result.keys()) == {"passed", "achieved_miou", "threshold", "n_batches", "steps"}


def test_overfit_check_fpn_head(check_cfg):
    backbone = SimpleBackbone()
    probe = _make_probe(backbone, head_type="fpn")
    loader = _make_loader(SimpleBackbone)

    result = run_overfit_check(
        probe=probe,
        train_loader=loader,
        num_classes=NUM_CLASSES,
        device=torch.device("cpu"),
        check_cfg=check_cfg,
    )

    assert isinstance(result["achieved_miou"], float)
    assert result["passed"]


def test_overfit_check_empty_loader(check_cfg):
    backbone = SimpleBackbone()
    probe = _make_probe(backbone)

    # Empty DataLoader
    dataset = TensorDataset(torch.empty(0, 3, IMG_SIZE, IMG_SIZE), torch.empty(0, IMG_SIZE, IMG_SIZE, dtype=torch.long))
    loader = DataLoader(dataset, batch_size=4)

    result = run_overfit_check(
        probe=probe,
        train_loader=loader,
        num_classes=NUM_CLASSES,
        device=torch.device("cpu"),
        check_cfg=check_cfg,
    )

    assert not result["passed"]
    assert result["n_batches"] == 0
