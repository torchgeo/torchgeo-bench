import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from torchgeo_bench.segmentation_probe import (
    CachedFeaturesDataset,
    SegmentationProbe,
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


# ---------------------------------------------------------------------------
# Feature caching: extract_all_features + CachedFeaturesDataset
# ---------------------------------------------------------------------------


def test_extract_all_features_returns_cached_dataset(mock_backbone, dummy_data):
    """extract_all_features produces a CachedFeaturesDataset with correct length and dtypes."""
    images, masks = dummy_data["image"], dummy_data["mask"]
    loader = make_loader(images, masks)
    probe = make_probe(mock_backbone, ["layer1", "layer2"])

    cache = probe.extract_all_features(loader, cache_dtype=torch.float16)

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
    cache = probe.extract_all_features(loader, cache_dtype=torch.float32)

    feats, mask = cache[0]
    assert len(feats) == 2  # two layers
    assert mask.shape == (64, 64)


def test_solver_fit_cached(mock_backbone, dummy_data):
    """fit_cached trains the head on cached features and evaluate_cached returns a valid mIoU."""
    images, masks = dummy_data["image"], dummy_data["mask"]
    loader = make_loader(images, masks)
    probe = make_probe(mock_backbone, ["layer1", "layer2"])
    solver = SegmentationSolver(model=probe, num_classes=NUM_CLASSES, lr=1e-3, device="cpu")

    train_cache = probe.extract_all_features(loader, cache_dtype=torch.float32)
    val_cache = probe.extract_all_features(loader, cache_dtype=torch.float32)

    val_miou = solver.fit_cached(
        train_cache, val_cache=val_cache, batch_size=2, epochs=1, verbose=False
    )
    assert isinstance(val_miou, float)
    assert 0.0 <= val_miou <= 1.0

    metrics = solver.evaluate_cached(val_cache, batch_size=2)
    assert isinstance(metrics, dict)
    assert 0.0 <= metrics["mIoU"] <= 1.0


def test_extract_all_features_dict_batches(mock_backbone, dummy_data):
    """extract_all_features handles dict-format batches {"image": ..., "mask": ...}."""
    images, masks = dummy_data["image"], dummy_data["mask"]
    loader = make_loader(images, masks, as_dict=True)
    probe = make_probe(mock_backbone, ["layer1"])
    cache = probe.extract_all_features(loader, cache_dtype=torch.float32)
    assert len(cache) == len(images)


# ---------------------------------------------------------------------------
# GPUTensorCache
# ---------------------------------------------------------------------------


from torchgeo_bench.segmentation_probe import GPUTensorCache, _estimate_cache_bytes


def _make_cpu_cache(mock_backbone, dummy_data):
    """Helper: extract a CachedFeaturesDataset on CPU."""
    images, masks = dummy_data["image"], dummy_data["mask"]
    loader = make_loader(images, masks)
    probe = make_probe(mock_backbone, ["layer1", "layer2"])
    return probe.extract_all_features(loader, cache_dtype=torch.float16)


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
    for feats, masks in gpu_cache.ordered_batches(batch_size=1):
        total += masks.shape[0]
    assert total == len(cache)


def test_solver_fit_cached_uses_gpu_cache_path(mock_backbone, dummy_data):
    """fit_cached falls back gracefully to DataLoader path on CPU (no CUDA available in CI)."""
    images, masks = dummy_data["image"], dummy_data["mask"]
    loader = make_loader(images, masks)
    probe = make_probe(mock_backbone, ["layer1", "layer2"])
    solver = SegmentationSolver(model=probe, num_classes=NUM_CLASSES, lr=1e-3, device="cpu")

    train_cache = probe.extract_all_features(loader, cache_dtype=torch.float32)
    val_cache = probe.extract_all_features(loader, cache_dtype=torch.float32)

    # On CPU, use_amp=False so GPUTensorCache path is skipped; DataLoader fallback runs.
    val_miou = solver.fit_cached(
        train_cache, val_cache=val_cache, batch_size=2, epochs=1, verbose=False
    )
    assert isinstance(val_miou, float)
    assert 0.0 <= val_miou <= 1.0
