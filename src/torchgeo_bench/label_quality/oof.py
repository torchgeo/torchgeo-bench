"""Member-level out-of-fold (OOF) prediction substrate.

Both label-noise scorers (Cleanlab, AER) share the same predictions: for each
of ``M`` reseeded K-fold members we train on ``K-1`` folds and infer the
held-out fold, so every sample is scored by models that *never* saw it. Dihedral
test-time augmentation (the full D4 group) is folded in *here* — applied once at
inference and averaged back into the native label frame — so the downstream
scorers consume clean predictions and never re-run TTA.

The result exposes two views over the same tensor:

- ``mean_softmax`` — ``(N, C, H, W)`` per-pixel softmax averaged over members,
  the substrate Cleanlab ranks.
- ``member_stack`` — ``(M, N, C, H, W)`` per-member predictions, the substrate
  AER measures disagreement over.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .predictors import _unpack, build_member

logger = logging.getLogger(__name__)


def _rot(k: int):
    """A ``(forward, inverse)`` pair rotating by ``k`` quarter-turns."""
    return (
        lambda x, k=k: torch.rot90(x, k, (-2, -1)),
        lambda x, k=k: torch.rot90(x, -k, (-2, -1)),
    )


# The dihedral group D4 as ``(forward, inverse)`` transform pairs on ``(B,C,H,W)``
# tensors. Each inverse is exact (integer pixel permutations, no interpolation),
# so a TTA round-trip is lossless.
_DIHEDRAL_OPS: list[tuple[Callable, Callable]] = [
    (lambda x: x, lambda x: x),  # identity
    _rot(1),  # rot90
    _rot(2),  # rot180
    _rot(3),  # rot270
    (lambda x: x.flip(-1), lambda x: x.flip(-1)),  # horizontal flip
    (lambda x: x.flip(-2), lambda x: x.flip(-2)),  # vertical flip
    (lambda x: x.transpose(-2, -1), lambda x: x.transpose(-2, -1)),  # main-diagonal transpose
    (  # anti-diagonal transpose: rot90 then hflip, undone in reverse order
        lambda x: torch.rot90(x, 1, (-2, -1)).flip(-1),
        lambda x: torch.rot90(x.flip(-1), -1, (-2, -1)),
    ),
]


@dataclass
class OOFResult:
    """Out-of-fold predictions in the two views the scorers consume."""

    member_stack: torch.Tensor  # (M, N, C, H, W) per-member softmax
    fold_ids: np.ndarray  # (M, N) held-out fold assignment per member

    @property
    def mean_softmax(self) -> torch.Tensor:
        """``(N, C, H, W)`` softmax averaged over the member axis."""
        return self.member_stack.mean(dim=0)


def run_oof(
    dataset,
    members: list[dict],
    folds: Callable[[int], np.ndarray],
    *,
    batch_size: int = 8,
    tta: bool = True,
    member_factory: Callable[[dict], object] = build_member,
) -> OOFResult:
    """Run the shared OOF substrate over ``M`` reseeded K-fold members.

    Args:
        dataset: An indexable seg dataset yielding ``{"image", "mask"}`` (or a
            ``(image, mask)`` tuple) at a fixed native label frame ``(H, W)``.
        members: One spec dict per member (passed to ``member_factory``). Each is
            reseeded via its ``seed`` so the members use distinct fold partitions.
            ``num_classes`` sets the prediction channel count.
        folds: ``folds(seed) -> fold_ids`` mapping a member seed to a per-sample
            fold assignment array of shape ``(N,)`` covering ``range(k)``.
        batch_size: Loader batch size for both training and held-out inference.
        tta: Apply dihedral (D4) test-time augmentation at inference.
        member_factory: Builds a member from a spec; the member must expose
            ``fit(train_loader)``, ``probe`` (native-frame logits module) and
            ``device``. Injectable for testing.

    Returns:
        An :class:`OOFResult` with ``member_stack`` ``(M, N, C, H, W)`` and its
        member-mean view ``mean_softmax``.
    """
    n = len(dataset)
    num_classes = members[0]["num_classes"]
    h, w = _native_frame(dataset)

    m = len(members)
    member_stack = torch.zeros(m, n, num_classes, h, w)
    fold_ids = np.empty((m, n), dtype=np.int64)

    for member_idx, spec in enumerate(members):
        seed = spec.get("seed", member_idx)
        assignment = np.asarray(folds(seed))
        fold_ids[member_idx] = assignment
        k = int(assignment.max()) + 1

        for fold_idx in range(k):
            hold_idx = np.where(assignment == fold_idx)[0]
            if hold_idx.size == 0:
                continue
            train_idx = np.where(assignment != fold_idx)[0]

            predictor = member_factory(spec)
            predictor.fit(_subset_loader(dataset, train_idx, batch_size, shuffle=True))
            preds = _predict_holdout(
                predictor,
                _subset_loader(dataset, hold_idx, batch_size, shuffle=False),
                num_classes=num_classes,
                tta=tta,
            )
            member_stack[member_idx, torch.as_tensor(hold_idx)] = preds

    return OOFResult(member_stack=member_stack, fold_ids=fold_ids)


def _native_frame(dataset) -> tuple[int, int]:
    """Native label ``(H, W)`` read from the first sample's mask."""
    sample = dataset[0]
    mask = sample["mask"] if isinstance(sample, dict) else sample[1]
    if mask.ndim == 3:
        mask = mask.squeeze(0)
    return int(mask.shape[-2]), int(mask.shape[-1])


def _subset_loader(dataset, indices, batch_size: int, shuffle: bool) -> DataLoader:
    """Batched loader over a global-index subset (indices recoverable via ``.dataset``)."""
    return DataLoader(Subset(dataset, list(indices)), batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def _predict_holdout(predictor, loader: DataLoader, *, num_classes: int, tta: bool) -> torch.Tensor:
    """Softmax predictions for the held-out fold at the native frame, with dihedral TTA."""
    probe = predictor.probe
    probe.eval()
    ops = _DIHEDRAL_OPS if tta else _DIHEDRAL_OPS[:1]

    chunks: list[torch.Tensor] = []
    for batch in loader:
        images, masks = _unpack(batch)
        target_hw = (masks.shape[-2], masks.shape[-1])
        images = images.to(predictor.device)

        acc = torch.zeros(images.shape[0], num_classes, *target_hw)
        for fwd, inv in ops:
            t_img = fwd(images)
            t_hw = fwd(torch.zeros(1, 1, *target_hw)).shape[-2:]
            logits = probe(t_img).float()
            if logits.shape[-2:] != t_hw:
                logits = F.interpolate(logits, size=t_hw, mode="bilinear", align_corners=False)
            acc = acc + inv(logits.softmax(dim=1)).cpu()
        chunks.append(acc / len(ops))

    return torch.cat(chunks, dim=0)
