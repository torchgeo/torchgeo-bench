"""Segmentation losses for severely foreground-imbalanced datasets.

Plain :class:`torch.nn.CrossEntropyLoss` collapses toward the majority class on
the SpaceNet building-footprint datasets (spacenet7 is 93.5% background / 6.5%
building): the model predicts building on 0.26% of pixels while ~99.7% of tiles
actually contain one. Every published SpaceNet recipe avoids this with a
*composite* loss pairing pixelwise CE with a region loss, not with class
weighting — region losses are scale-invariant to the foreground fraction, so
they carry the imbalance correction implicitly:

- the SpaceNet 7 official baseline (``solaris`` ``sn7_baseline_train.yml``) uses
  ``bcewithlogits: 10`` + ``jaccard: 2.5``,
- the SpaceNet 7 paper's VGG16 model uses ``L = J + 4*BCE``, i.e. the same 4:1
  ratio, and its EfficientNet-B5 model uses ``Focal + Dice``,
- the 2nd/3rd/4th place solutions use ``dice+focal``, ``dice+bce+focal`` and
  ``dice+bce``.

:class:`ComboLoss` therefore takes the ``{name: weight}`` mapping verbatim, so
``ComboLoss({"bce": 10.0, "jaccard": 2.5})`` *is* the published baseline recipe.
:class:`WeightedCrossEntropyLoss` is provided as the inverse-frequency control
arm; no published SpaceNet solution used class weighting, so it exists to test
that preference rather than to encode it.

Three constraints from the training loop shape every implementation here; see
the module-level notes on each.

**AMP.** ``SegmentationSolver`` calls the criterion *inside*
``torch.autocast`` (``segmentation_task.py`` ``_train_on_batch`` and the cached
path). Region losses reduce over N*H*W ~ 262k pixels per tile, and an fp16
accumulator saturates at 65504 — a full-background batch overflows to ``inf``
in the union term alone. Every reduction in this module therefore runs in fp32,
forced by :func:`_to_float32_probs`, regardless of the surrounding autocast
state.

**Non-finite losses are fatal.** ``_train_on_batch`` raises
:class:`~torchgeo_bench.segmentation_task.NonFiniteLossError` on a non-finite
loss, because one NaN gradient permanently poisons AdamW's ``exp_avg`` /
``exp_avg_sq``. A batch whose tiles are entirely background (or entirely
``ignore_index``) drives every region-loss denominator to zero, so all of them
are epsilon-guarded and every mean is taken over a *non-empty* class set --
never a bare ``.mean()`` over a possibly-empty selection, which returns NaN.

**``ignore_index`` is introspected.** ``main.py``'s
``_resolve_segmentation_ignore_index`` reads ``criterion.ignore_index`` and
raises on a mismatch with ``eval.segmentation.ignore_index``. Every loss here
exposes that attribute with the same meaning as ``nn.CrossEntropyLoss``.
"""

import torch
import torch.nn.functional as F
from torch import nn

__all__ = [
    "ComboLoss",
    "DiceLoss",
    "FocalLoss",
    "SafeCrossEntropyLoss",
    "SoftJaccardLoss",
    "WeightedCrossEntropyLoss",
    "build_loss",
    "inverse_frequency_weights",
]

# Guards every region-loss ratio. Large enough to dominate fp32 rounding on a
# 262k-pixel reduction, small enough not to shift a well-populated class's score.
_EPS = 1e-6


def _to_float32_probs(logits: torch.Tensor) -> torch.Tensor:
    """Softmax probabilities in fp32, safe to reduce over a whole batch.

    The cast is unconditional rather than ``if is_autocast_enabled()``: the
    criterion may also be called outside autocast, and softmaxing an fp16 tensor
    to fp32 costs one kernel while a silent overflow costs a training run.

    Args:
        logits: Raw model outputs ``(B, C, H, W)``, any floating dtype.

    Returns:
        ``(B, C, H, W)`` fp32 probabilities along the class axis.
    """
    return logits.float().softmax(dim=1)


def _valid_mask_and_targets(
    targets: torch.Tensor, ignore_index: int, num_classes: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split targets into a validity mask and a safely-clamped label map.

    ``ignore_index`` (255) is out of range for :func:`~torch.nn.functional.one_hot`,
    which would raise, so ignored positions are rewritten to class 0 and then
    zeroed out through ``valid``. The clamp also absorbs any stray out-of-range
    label rather than letting it index out of bounds.

    Args:
        targets: Integer labels ``(B, H, W)``.
        ignore_index: Label value excluded from the loss.
        num_classes: Number of classes ``C``.

    Returns:
        ``(valid, safe_targets)`` — a bool mask ``(B, H, W)`` and labels
        ``(B, H, W)`` guaranteed to lie in ``[0, C-1]``.
    """
    valid = targets != ignore_index
    safe = torch.where(valid, targets, torch.zeros_like(targets))
    return valid, safe.clamp_(0, num_classes - 1)


def _region_terms(
    logits: torch.Tensor, targets: torch.Tensor, ignore_index: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-class intersection and cardinalities for the region losses.

    Shared by :class:`SoftJaccardLoss` and :class:`DiceLoss`, which differ only
    in how they combine these three sums. All reductions are fp32 (see the
    module docstring on AMP) and are taken over the batch *and* both spatial
    axes, giving one scalar per class -- a batch-level region loss, matching the
    ``solaris`` reference implementation.

    Args:
        logits: Raw model outputs ``(B, C, H, W)``.
        targets: Integer labels ``(B, H, W)``.
        ignore_index: Label value excluded from every sum.

    Returns:
        ``(intersection, prob_sum, target_sum)``, each ``(C,)`` in fp32.
    """
    num_classes = logits.shape[1]
    probs = _to_float32_probs(logits)
    valid, safe_targets = _valid_mask_and_targets(targets, ignore_index, num_classes)

    # (B, H, W) -> (B, C, H, W), with ignored pixels contributing nothing to
    # either operand so they cannot enter intersection or union.
    onehot = F.one_hot(safe_targets, num_classes).permute(0, 3, 1, 2).float()
    keep = valid.unsqueeze(1).float()
    probs = probs * keep
    onehot = onehot * keep

    dims = (0, 2, 3)
    return (probs * onehot).sum(dims), probs.sum(dims), onehot.sum(dims)


def _mean_over_present(per_class: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
    """Mean of ``per_class`` over present classes, NaN-free when none are present.

    A ``per_class[present].mean()`` on an all-ignored batch reduces over an empty
    tensor and returns NaN, which ``_train_on_batch`` escalates to
    :class:`NonFiniteLossError` and which kills the run. Falling back to a zero
    loss is correct here: a batch with no supervised pixel carries no region
    signal, so it should contribute no gradient rather than abort training.

    Args:
        per_class: Per-class loss values ``(C,)``.
        present: Bool mask ``(C,)`` marking classes to average over.

    Returns:
        Scalar mean, or ``0.0`` when ``present`` selects nothing.
    """
    count = present.sum()
    if count == 0:
        return per_class.sum() * 0.0  # keeps the graph connected; contributes no gradient
    return (per_class * present.float()).sum() / count


class SoftJaccardLoss(nn.Module):
    """Soft Jaccard (IoU) loss, averaged over the classes present in the batch.

    ``1 - (sum p*y + eps) / (sum p + sum y - sum p*y + eps)`` per class. This is
    the ``jaccard`` term of the SpaceNet 7 baseline's ``bce:10 + jaccard:2.5``
    and the ``J`` of the paper's ``L = J + 4*BCE``.

    Absent classes are excluded from the mean rather than scored: a class with no
    labelled pixel has intersection 0 and would otherwise contribute a constant
    loss of 1.0, swamping the classes that carry signal on background-dominated
    tiles.
    """

    def __init__(self, ignore_index: int = 255) -> None:
        """Initialize the loss.

        Args:
            ignore_index: Label value excluded from the loss (default 255).
        """
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the loss.

        Args:
            logits: Raw model outputs ``(B, C, H, W)``.
            targets: Integer labels ``(B, H, W)``.

        Returns:
            Scalar loss in ``[0, 1]``.
        """
        inter, prob_sum, target_sum = _region_terms(logits, targets, self.ignore_index)
        union = prob_sum + target_sum - inter
        per_class = 1.0 - (inter + _EPS) / (union + _EPS)
        return _mean_over_present(per_class, target_sum > 0)


class DiceLoss(nn.Module):
    """Soft Dice loss, averaged over the classes present in the batch.

    ``1 - (2*sum p*y + eps) / (sum p + sum y + eps)`` per class. Differs from
    :class:`SoftJaccardLoss` only in the denominator; it is the ``dice`` term of
    the 2nd/3rd/4th place SpaceNet 7 solutions and of the paper's
    ``Focal + Dice`` model. Absent classes are excluded for the same reason.
    """

    def __init__(self, ignore_index: int = 255) -> None:
        """Initialize the loss.

        Args:
            ignore_index: Label value excluded from the loss (default 255).
        """
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the loss.

        Args:
            logits: Raw model outputs ``(B, C, H, W)``.
            targets: Integer labels ``(B, H, W)``.

        Returns:
            Scalar loss in ``[0, 1]``.
        """
        inter, prob_sum, target_sum = _region_terms(logits, targets, self.ignore_index)
        per_class = 1.0 - (2.0 * inter + _EPS) / (prob_sum + target_sum + _EPS)
        return _mean_over_present(per_class, target_sum > 0)


class FocalLoss(nn.Module):
    """Multiclass focal loss, ``mean over valid pixels of (1-p_t)^gamma * CE``.

    Down-weights the easy, confidently-correct background pixels that dominate
    the gradient on 93/7 data. Used by the SpaceNet 7 paper's EfficientNet-B5
    model and by the 2nd/3rd place solutions.

    Computed from ``log_softmax`` in fp32 rather than from ``probs.log()``: the
    latter is what produces ``-inf`` when a saturated fp16 probability
    underflows to exactly zero.
    """

    def __init__(self, gamma: float = 2.0, ignore_index: int = 255) -> None:
        """Initialize the loss.

        Args:
            gamma: Focusing exponent; 0 recovers plain cross-entropy.
            ignore_index: Label value excluded from the loss (default 255).
        """
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the loss.

        Args:
            logits: Raw model outputs ``(B, C, H, W)``.
            targets: Integer labels ``(B, H, W)``.

        Returns:
            Scalar loss, ``0.0`` if the batch has no valid pixel.
        """
        ce = F.cross_entropy(
            logits.float(),
            targets,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        # p_t = exp(-CE) is exact here and avoids a second gather.
        loss = (1.0 - torch.exp(-ce)).pow(self.gamma) * ce

        valid = targets != self.ignore_index
        count = valid.sum()
        if count == 0:
            return loss.sum() * 0.0
        return (loss * valid.float()).sum() / count


class SafeCrossEntropyLoss(nn.Module):
    """Cross-entropy that returns 0.0 instead of NaN when every pixel is ignored.

    ``nn.CrossEntropyLoss(ignore_index=...)`` with ``reduction="mean"`` divides by
    the number of *valid* pixels, so a tile whose label is entirely
    ``ignore_index`` yields ``0/0 = NaN``. That NaN is not hypothetical here:
    ``_train_on_batch`` escalates it to :class:`NonFiniteLossError` and aborts the
    run, and it is reachable on any dataset with fully-masked tiles.

    Used as the ``"bce"`` term inside :class:`ComboLoss` so that a composite
    cannot be poisoned by a term the region losses already handle correctly. The
    zero is the same convention :func:`_mean_over_present` uses: an unsupervised
    batch contributes no gradient rather than killing training.
    """

    def __init__(self, weight: torch.Tensor | None = None, ignore_index: int = 255) -> None:
        """Initialize the loss.

        Args:
            weight: Optional per-class weights ``(C,)``.
            ignore_index: Label value excluded from the loss (default 255).
        """
        super().__init__()
        self.ignore_index = ignore_index
        if weight is None:
            self.weight: torch.Tensor | None = None
        else:
            self.register_buffer("weight", torch.as_tensor(weight, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the loss.

        Args:
            logits: Raw model outputs ``(B, C, H, W)``.
            targets: Integer labels ``(B, H, W)``.

        Returns:
            Scalar loss, ``0.0`` if the batch has no valid pixel.
        """
        logits = logits.float()
        if not (targets != self.ignore_index).any():
            return logits.sum() * 0.0
        return F.cross_entropy(logits, targets, weight=self.weight, ignore_index=self.ignore_index)


class WeightedCrossEntropyLoss(SafeCrossEntropyLoss):
    """Cross-entropy with fixed per-class weights.

    The inverse-frequency control arm. No published SpaceNet solution used class
    weighting -- the composites above are the attested remedy -- so this exists
    to *test* that preference rather than to encode it. Note that up-weighting a
    rare class trades calibration for recall, which matters because Cleanlab
    consumes the softmax directly while AER only sees the argmax.

    Inherits the registered ``weight`` buffer, so ``.to(device)`` moves the
    weights with the module -- a CPU weight tensor against CUDA logits is a
    runtime error -- and inherits the all-ignored guard.
    """

    def __init__(self, weight: torch.Tensor, ignore_index: int = 255) -> None:
        """Initialize the loss.

        Args:
            weight: Per-class weights ``(C,)``, e.g. from
                :func:`inverse_frequency_weights`.
            ignore_index: Label value excluded from the loss (default 255).
        """
        super().__init__(weight=weight, ignore_index=ignore_index)


class ComboLoss(nn.Module):
    """Weighted sum of named loss terms, e.g. ``{"bce": 10.0, "jaccard": 2.5}``.

    The mapping mirrors the ``solaris`` YAML loss spec, so a published recipe
    transfers verbatim -- ``ComboLoss({"bce": 10.0, "jaccard": 2.5})`` is the
    SpaceNet 7 official baseline, and ``ComboLoss({"bce": 4.0, "jaccard": 1.0})``
    is the paper's ``L = J + 4*BCE``.

    ``"bce"`` names multiclass cross-entropy, not binary BCE: the probe emits
    ``C``-channel logits and softmax CE is its two-class equivalent. The name is
    kept for fidelity to the published configs.

    Terms are summed *unnormalized*, as in the reference configs -- with
    ``bce:10 + jaccard:2.5`` the loss starts near 25, which is expected and not a
    divergence.
    """

    def __init__(self, weights: dict[str, float], ignore_index: int = 255) -> None:
        """Initialize the loss.

        Args:
            weights: Map of term name to weight. Recognised names: ``"bce"`` /
                ``"ce"``, ``"jaccard"`` / ``"iou"``, ``"dice"``, ``"focal"``.
            ignore_index: Label value excluded from every term (default 255).

        Raises:
            ValueError: If ``weights`` is empty or names an unknown term.
        """
        super().__init__()
        if not weights:
            raise ValueError("ComboLoss requires at least one weighted term.")

        self.ignore_index = ignore_index
        self.weights = dict(weights)

        terms: dict[str, nn.Module] = {}
        for name in self.weights:
            key = name.lower()
            if key in {"bce", "ce", "cross_entropy"}:
                terms[name] = SafeCrossEntropyLoss(ignore_index=ignore_index)
            elif key in {"jaccard", "iou", "soft_jaccard"}:
                terms[name] = SoftJaccardLoss(ignore_index=ignore_index)
            elif key == "dice":
                terms[name] = DiceLoss(ignore_index=ignore_index)
            elif key == "focal":
                terms[name] = FocalLoss(ignore_index=ignore_index)
            else:
                raise ValueError(
                    f"Unknown loss term {name!r}. "
                    "Expected one of: bce/ce, jaccard/iou, dice, focal."
                )
        self.terms = nn.ModuleDict(terms)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the weighted sum of all terms.

        Args:
            logits: Raw model outputs ``(B, C, H, W)``.
            targets: Integer labels ``(B, H, W)``.

        Returns:
            Scalar loss.
        """
        total = None
        for name, term in self.terms.items():
            # nn.CrossEntropyLoss is autocast-safe but returns fp16 under AMP;
            # promote so a large weight cannot overflow the accumulator.
            value = term(logits, targets).float() * self.weights[name]
            total = value if total is None else total + value
        return total


def inverse_frequency_weights(
    class_counts: torch.Tensor, *, max_ratio: float | None = None
) -> torch.Tensor:
    """Inverse-frequency class weights, normalized so the smallest weight is 1.

    Only for :class:`WeightedCrossEntropyLoss` (the control arm). ``max_ratio``
    caps the spread: spacenet7's raw inverse-frequency ratio is 1:14.4, and
    weighting that hard trades away the calibration Cleanlab depends on, so the
    ablation compares a capped (~5x) against an uncapped arm.

    Classes with zero count get weight 1.0 rather than infinity.

    Args:
        class_counts: Per-class pixel counts ``(C,)``.
        max_ratio: Optional cap on max/min weight. ``None`` leaves it uncapped.

    Returns:
        ``(C,)`` fp32 weights with ``min == 1.0``.
    """
    counts = torch.as_tensor(class_counts, dtype=torch.float64)
    weights = torch.where(counts > 0, counts.sum() / counts.clamp(min=1.0), torch.ones_like(counts))
    weights = weights / weights.min()
    if max_ratio is not None:
        weights = weights.clamp(max=float(max_ratio))
    return weights.float()


def build_loss(name: str, *, ignore_index: int = 255, **kwargs) -> nn.Module:
    """Build a loss by short name, for the ablation's ``--loss`` flag.

    Args:
        name: One of ``ce``, ``bce_jaccard_4to1``, ``sn7_baseline``, ``dice_bce``,
            ``dice_focal``, ``jaccard``, ``dice``, ``focal``, ``weighted_ce``.
        ignore_index: Label value excluded from the loss (default 255).
        **kwargs: Extra arguments; ``weighted_ce`` requires ``weight``.

    Returns:
        The configured loss module.

    Raises:
        ValueError: If ``name`` is unknown.
    """
    key = name.lower()
    if key == "ce":
        # SafeCrossEntropyLoss, not nn.CrossEntropyLoss: identical on every batch
        # with at least one valid pixel, but 0.0 rather than NaN on a fully
        # ignored one. The ablation's baseline arm must not be the only one that
        # can die on a masked tile, or a crash would read as a loss-function result.
        return SafeCrossEntropyLoss(ignore_index=ignore_index)
    if key in {"bce_jaccard_4to1", "sn7_paper"}:
        # SpaceNet 7 paper, VGG16: L = J + 4*BCE.
        return ComboLoss({"bce": 4.0, "jaccard": 1.0}, ignore_index=ignore_index)
    if key in {"sn7_baseline", "bce10_jaccard25"}:
        # solaris sn7_baseline_train.yml, verbatim.
        return ComboLoss({"bce": 10.0, "jaccard": 2.5}, ignore_index=ignore_index)
    if key == "dice_bce":
        return ComboLoss({"dice": 1.0, "bce": 1.0}, ignore_index=ignore_index)
    if key == "dice_focal":
        return ComboLoss({"dice": 1.0, "focal": 2.0}, ignore_index=ignore_index)
    if key in {"jaccard", "iou"}:
        return SoftJaccardLoss(ignore_index=ignore_index)
    if key == "dice":
        return DiceLoss(ignore_index=ignore_index)
    if key == "focal":
        return FocalLoss(ignore_index=ignore_index, **kwargs)
    if key == "weighted_ce":
        return WeightedCrossEntropyLoss(ignore_index=ignore_index, **kwargs)
    raise ValueError(f"Unknown loss {name!r}.")
