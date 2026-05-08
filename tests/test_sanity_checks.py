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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REQUIRED_KEYS = {
    "passed",
    "achieved_miou",
    "threshold",
    "n_batches",
    "steps",
    "batch_size",
    "unique_labels",
    "feature_norm",
    "feature_std",
    "initial_loss",
    "loss_delta",
}


class SimpleBackbone(nn.Module):
    """Simple CNN backbone with hookable intermediate layers.

    Images are constructed so channel 0 encodes the class label — these
    fixed conv weights read that signal out, giving discriminative features
    even without any training.
    """

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=2), nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2), nn.ReLU()
        )
        # Fix layer1 weights so channel 0 → feature channels, making features
        # class-discriminative even with no gradient updates on the backbone.
        with torch.no_grad():
            self.layer1[0].weight.zero_()
            self.layer1[0].bias.zero_()
            for i in range(min(NUM_CLASSES, 16)):
                self.layer1[0].weight[i, 0] = 1.0

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class ZeroBackbone(nn.Module):
    """Backbone whose hooked layers emit all-zero tensors — simulates degenerate features."""

    def __init__(self):
        super().__init__()
        # Use identity-shaped layers whose output will be zeroed before hooks fire.
        self.layer1 = _ZeroLayer(3, 16)
        self.layer2 = _ZeroLayer(16, 32)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x


class _ZeroLayer(nn.Module):
    """A strided layer that always emits zeros (hooks capture this zero output)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.out_ch = out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _c, h, w = x.shape
        return torch.zeros(b, self.out_ch, h // 2, w // 2, device=x.device, dtype=x.dtype)


def _make_loader(backbone_class, n_samples=8):
    """Create a DataLoader of synthetic (image, mask) pairs.

    Each image encodes its class label in channel 0 so the fixed backbone
    weights can read out discriminative features. Masks are spatially constant
    per image (one class fills the whole image).
    """
    class_ids = torch.arange(n_samples) % NUM_CLASSES
    images = torch.randn(n_samples, 3, IMG_SIZE, IMG_SIZE)
    images[:, 0] = class_ids.float()[:, None, None].expand(n_samples, IMG_SIZE, IMG_SIZE)
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
    return OmegaConf.create(
        {
            "overfit_n_batches": 2,
            "overfit_steps": 300,
            "overfit_threshold": 0.5,
            "overfit_lr": 1e-2,
        }
    )


def test_overfit_check_passes_with_functional_encoder(check_cfg):
    backbone = SimpleBackbone()
    probe = _make_probe(backbone, head_type="linear")
    loader = _make_loader(SimpleBackbone)

    result = run_overfit_check(
        probe=probe,
        train_loader=loader,
        num_classes=NUM_CLASSES,
        device=DEVICE,
        check_cfg=check_cfg,
    )

    assert result["n_batches"] == 2
    assert result["steps"] == 300
    assert result["threshold"] == 0.5
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
        device=DEVICE,
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
        device=DEVICE,
        check_cfg=check_cfg,
    )

    assert REQUIRED_KEYS.issubset(result.keys())


def test_overfit_check_fpn_head(check_cfg):
    backbone = SimpleBackbone()
    probe = _make_probe(backbone, head_type="fpn")
    loader = _make_loader(SimpleBackbone)

    result = run_overfit_check(
        probe=probe,
        train_loader=loader,
        num_classes=NUM_CLASSES,
        device=DEVICE,
        check_cfg=check_cfg,
    )

    assert isinstance(result["achieved_miou"], float)
    assert result["passed"]


def test_overfit_check_empty_loader(check_cfg):
    backbone = SimpleBackbone()
    probe = _make_probe(backbone)

    # Empty DataLoader
    dataset = TensorDataset(
        torch.empty(0, 3, IMG_SIZE, IMG_SIZE), torch.empty(0, IMG_SIZE, IMG_SIZE, dtype=torch.long)
    )
    loader = DataLoader(dataset, batch_size=4)

    result = run_overfit_check(
        probe=probe,
        train_loader=loader,
        num_classes=NUM_CLASSES,
        device=DEVICE,
        check_cfg=check_cfg,
    )

    assert not result["passed"]
    assert result["n_batches"] == 0
