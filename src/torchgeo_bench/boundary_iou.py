"""Boundary IoU, an opt-in segmentation metric that scores contour accuracy.

Ordinary IoU saturates once a prediction covers most of a region, so a mask whose
interior is right and whose edge is wrong still scores well. Boundary IoU
(Cheng et al., CVPR 2021, https://arxiv.org/abs/2103.16562) restricts both masks
to a band along their own contours before taking the intersection over union, so
the same displacement costs far more.

The band width follows the reference implementation: a fraction of the image
diagonal, rounded to whole pixels and never below one.
"""

import math

import torch
import torch.nn.functional as F
from torchmetrics import Metric

# The dilation ratio used throughout Cheng et al., CVPR 2021.
DEFAULT_DILATION_RATIO = 0.02


def resolve_boundary_width(height: int, width: int, dilation_ratio: float) -> int:
    """Return the band width in pixels for one image size.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        dilation_ratio: Band width as a fraction of the image diagonal.

    Returns:
        The width in whole pixels, at least one.
    """
    diagonal = math.sqrt(height**2 + width**2)
    return max(1, round(dilation_ratio * diagonal))


def _square_pool(binary: torch.Tensor, width: int, *, erode: bool) -> torch.Tensor:
    """Erode or dilate by a square of side ``2 * width + 1``, treating outside as background."""
    sign = -1.0 if erode else 1.0
    padded = sign * F.pad(binary.unsqueeze(1).float(), (width,) * 4, value=0.0)
    # A square structuring element is separable, so two 1-D passes agree exactly with one
    # 2-D pass at O(k) per pixel instead of O(k^2).
    kernel = 2 * width + 1
    pooled = F.max_pool2d(padded, kernel_size=(kernel, 1), stride=1)
    pooled = F.max_pool2d(pooled, kernel_size=(1, kernel), stride=1)
    return (sign * pooled).squeeze(1) > 0.5


def boundary_region(binary: torch.Tensor, width: int) -> torch.Tensor:
    """Return the pixels of ``binary`` lying within ``width`` of its outer edge.

    Erode the mask and keep what the erosion removed. Pixels beyond the image count as
    background, so a region running off the edge has a contour there, which matches the
    reference implementation.

    Args:
        binary: Boolean tensor shaped ``(N, H, W)``.
        width: Band width in pixels.

    Returns:
        A boolean tensor shaped like ``binary``.
    """
    if width < 1:
        raise ValueError(f"width must be at least 1, got {width}")
    return binary & ~_square_pool(binary, width, erode=True)


class MulticlassBoundaryIoU(Metric):
    """Accumulate per-class Boundary IoU over batches of dense predictions.

    Whether a pixel sits on a contour depends on its neighbours out to ``width``, so an
    ignored label can move a contour that lies up to ``width`` away from it. Scoring a
    pixel that close to an ignored one would make the result depend on labels nobody
    supplied, so every pixel within ``width`` of an ignored pixel is left out of both the
    intersection and the union. That is what makes changing an ignored label unable to
    change the score.

    A class absent from both the prediction and the target has an empty union. Its
    per-class value is NaN and it is left out of the mean rather than counted as a zero,
    so adding an unused class cannot lower the score.
    """

    is_differentiable: bool | None = False
    higher_is_better: bool | None = True
    full_state_update: bool | None = False
    # add_state registers these through setattr, which a type checker cannot follow.
    intersection: torch.Tensor
    union: torch.Tensor

    def __init__(
        self,
        num_classes: int,
        ignore_index: int | None = None,
        dilation_ratio: float = DEFAULT_DILATION_RATIO,
        **kwargs: object,
    ) -> None:
        """Configure the metric.

        Args:
            num_classes: Number of segmentation classes.
            ignore_index: Label value excluded from the score, or None.
            dilation_ratio: Band width as a fraction of the image diagonal.
            **kwargs: Forwarded to :class:`torchmetrics.Metric`.
        """
        super().__init__(**kwargs)
        if num_classes < 1:
            raise ValueError(f"num_classes must be positive, got {num_classes}")
        if not 0.0 < dilation_ratio <= 1.0:
            raise ValueError(f"dilation_ratio must be in (0, 1], got {dilation_ratio}")
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.dilation_ratio = dilation_ratio
        # Resolved from each batch's image size; reported alongside the score so a
        # published number carries the tolerance it was measured at.
        self.boundary_width: int | None = None
        self.add_state(
            "intersection",
            default=torch.zeros(num_classes, dtype=torch.long),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "union", default=torch.zeros(num_classes, dtype=torch.long), dist_reduce_fx="sum"
        )

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Accumulate one batch.

        Args:
            preds: Logits shaped ``(N, C, H, W)`` or labels shaped ``(N, H, W)``.
            target: Labels shaped ``(N, H, W)``.
        """
        if preds.ndim == target.ndim + 1:
            preds = preds.argmax(dim=1)
        if preds.shape != target.shape:
            raise ValueError(
                f"preds and target must describe the same pixels, got {tuple(preds.shape)} "
                f"and {tuple(target.shape)}"
            )
        if target.ndim != 3:
            raise ValueError(f"expected labels shaped (N, H, W), got {tuple(target.shape)}")

        width = resolve_boundary_width(target.shape[-2], target.shape[-1], self.dilation_ratio)
        self.boundary_width = width

        scored = None
        if self.ignore_index is not None:
            ignored = target == self.ignore_index
            scored = ~_square_pool(ignored, width, erode=False)

        # A class missing from both maps has empty bands, so it can only add zero to each total.
        present = torch.unique(torch.cat([preds.reshape(-1), target.reshape(-1)]))
        for value in present.tolist():
            if not 0 <= value < self.num_classes:
                continue
            target_band = boundary_region(target == value, width)
            pred_band = boundary_region(preds == value, width)
            if scored is not None:
                target_band &= scored
                pred_band &= scored
            self.intersection[value] += (target_band & pred_band).sum()
            self.union[value] += (target_band | pred_band).sum()

    def compute(self) -> torch.Tensor:
        """Return per-class Boundary IoU, NaN where a class has an empty union."""
        scores = torch.full(
            (self.num_classes,), float("nan"), dtype=torch.float64, device=self.union.device
        )
        present = self.union > 0
        scores[present] = self.intersection[present].double() / self.union[present].double()
        return scores

    def compute_mean(self) -> float:
        """Return the mean over the classes that appear, or NaN if none do."""
        return torch.nanmean(self.compute()).item()
