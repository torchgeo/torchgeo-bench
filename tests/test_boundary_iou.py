"""Contracts for the opt-in Boundary IoU segmentation metric."""

import math

import pytest
import torch
import torch.nn.functional as F
from torchmetrics.functional.classification import multiclass_jaccard_index

from torchgeo_bench.boundary_iou import (
    DEFAULT_DILATION_RATIO,
    MulticlassBoundaryIoU,
    boundary_region,
    resolve_boundary_width,
)

NUM_CLASSES = 2


def square(size: int, top: int, left: int, side: int) -> torch.Tensor:
    mask = torch.zeros(1, size, size, dtype=torch.long)
    mask[0, top : top + side, left : left + side] = 1
    return mask


def score(target: torch.Tensor, preds: torch.Tensor, **kwargs: object) -> float:
    metric = MulticlassBoundaryIoU(num_classes=NUM_CLASSES, **kwargs)
    metric.update(preds, target)
    return metric.compute_mean()


@pytest.mark.parametrize(
    ("height", "width", "ratio", "expected"),
    [(100, 100, 0.02, round(0.02 * math.sqrt(2) * 100)), (8, 8, 0.001, 1)],
    ids=["scales-with-diagonal", "floors-at-one-pixel"],
)
def test_resolve_boundary_width(height: int, width: int, ratio: float, expected: int) -> None:
    assert resolve_boundary_width(height, width, ratio) == expected


def test_boundary_region_keeps_the_rim_and_counts_the_image_edge() -> None:
    binary = torch.zeros(1, 9, 9, dtype=torch.bool)
    binary[0, 2:8, 2:8] = True

    band = boundary_region(binary, width=1)

    assert band.sum() == binary.sum() - 4**2
    assert not band[0, 4, 4]
    assert boundary_region(torch.ones(1, 5, 5, dtype=torch.bool), width=1)[0, 0, 0]


def test_separable_erosion_matches_a_single_two_dimensional_pass() -> None:
    torch.manual_seed(0)
    binary = torch.rand(3, 32, 32) > 0.4
    width = 3
    padded = F.pad(binary.unsqueeze(1).float(), (width,) * 4, value=0.0)
    direct = -F.max_pool2d(-padded, kernel_size=2 * width + 1, stride=1)

    assert torch.equal(boundary_region(binary, width), binary & (direct.squeeze(1) < 0.5))


def test_perfect_and_disjoint_predictions_bound_the_score() -> None:
    mask = square(48, 12, 12, 20)
    assert score(mask, mask) == pytest.approx(1.0)

    metric = MulticlassBoundaryIoU(num_classes=NUM_CLASSES)
    metric.update(square(64, 44, 44, 12), square(64, 2, 2, 12))
    assert metric.compute()[1].item() == pytest.approx(0.0)


def test_a_shifted_mask_costs_more_than_ordinary_iou() -> None:
    """The reason the metric exists: contour error must not be forgiven."""
    target, preds = square(128, 32, 32, 64), square(128, 32, 40, 64)
    boundary = score(target, preds)
    ordinary = multiclass_jaccard_index(
        preds, target, num_classes=NUM_CLASSES, average="macro"
    ).item()

    assert boundary < ordinary
    assert ordinary > 0.7
    assert boundary < 0.5


def test_ignored_pixels_cannot_change_the_score() -> None:
    """Contour membership depends on neighbours, so ignored labels have to be fenced off."""
    target, preds = square(48, 12, 12, 24), square(48, 12, 14, 24)
    ignored = target.clone()
    ignored[0, 40:, :] = 255
    corrupted = preds.clone()
    corrupted[0, 40:, :] = 1

    masked = score(ignored, preds, ignore_index=255)

    assert score(ignored, corrupted, ignore_index=255) == pytest.approx(masked)
    assert score(target, preds, ignore_index=255) != pytest.approx(masked)


def test_absent_classes_are_nan_and_leave_the_mean_alone() -> None:
    target, preds = square(48, 12, 12, 24), square(48, 12, 14, 24)
    two = MulticlassBoundaryIoU(num_classes=2)
    two.update(preds, target)
    three = MulticlassBoundaryIoU(num_classes=3)
    three.update(preds, target)

    assert math.isnan(three.compute()[2].item())
    assert three.compute_mean() == pytest.approx(two.compute_mean())


def test_totals_pool_over_the_dataset_and_reset_clears_them() -> None:
    first = (square(48, 4, 4, 16), square(48, 4, 8, 16))
    second = (square(48, 24, 24, 16), square(48, 24, 24, 16))

    metric = MulticlassBoundaryIoU(num_classes=NUM_CLASSES)
    assert metric.boundary_width is None
    for target, preds in (first, second):
        metric.update(preds, target)
    assert metric.boundary_width == resolve_boundary_width(48, 48, DEFAULT_DILATION_RATIO)

    pooled = MulticlassBoundaryIoU(num_classes=NUM_CLASSES)
    pooled.update(torch.cat([first[1], second[1]]), torch.cat([first[0], second[0]]))

    assert metric.compute_mean() == pytest.approx(pooled.compute_mean())
    assert metric.compute_mean() != pytest.approx((score(*first) + score(*second)) / 2)

    metric.reset()
    assert metric.union.sum() == 0


def test_logits_and_labels_give_the_same_score() -> None:
    target, labels = square(48, 12, 12, 20), square(48, 12, 14, 20)
    logits = F.one_hot(labels, num_classes=NUM_CLASSES).permute(0, 3, 1, 2).float()

    assert score(target, logits) == pytest.approx(score(target, labels))


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: MulticlassBoundaryIoU(num_classes=0), "num_classes"),
        (lambda: MulticlassBoundaryIoU(num_classes=2, dilation_ratio=1.5), "dilation_ratio"),
        (lambda: boundary_region(torch.zeros(1, 4, 4, dtype=torch.bool), width=0), "at least 1"),
        (
            lambda: MulticlassBoundaryIoU(num_classes=2).update(
                torch.zeros(1, 8, 8, dtype=torch.long), torch.zeros(1, 4, 4, dtype=torch.long)
            ),
            "same pixels",
        ),
        (
            lambda: MulticlassBoundaryIoU(num_classes=2).update(
                torch.zeros(8, 8, dtype=torch.long), torch.zeros(8, 8, dtype=torch.long)
            ),
            r"\(N, H, W\)",
        ),
    ],
    ids=["num-classes", "dilation-ratio", "zero-width", "shape-mismatch", "unbatched"],
)
def test_invalid_input_is_rejected(call, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        call()
