"""Member-level out-of-fold (OOF) prediction substrate.

Both label-noise scorers (Cleanlab, AER) share the same predictions: for each
of ``M`` reseeded K-fold members we train on ``K-1`` folds and infer the
held-out fold, so every sample is scored by models that *never* saw it. Dihedral
test-time augmentation (the full D4 group) is folded in *here* — applied once at
inference and averaged back into the native label frame — so the downstream
scorers consume clean predictions and never re-run TTA.

The result exposes exactly the two reductions the scorers consume, never the
full ``(M, N, C, H, W)`` float stack (which is ~276 GB for a 13-class dataset
and OOM-kills the host — see ``docs/plans/label_quality_flair2_oom_fix.md``):

- ``mean_softmax`` — ``(N, C, H, W)`` per-pixel softmax averaged over members,
  the substrate Cleanlab ranks. Built as a running sum ÷ M.
- ``member_preds`` — ``(M, N, H, W)`` ``uint8`` per-member hard predictions
  (argmax over the class axis), the substrate AER measures disagreement over.
"""

import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass, field

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
    """Out-of-fold predictions in the two reductions the scorers consume.

    The full ``(M, N, C, H, W)`` float softmax stack is deliberately never
    materialized (it OOM-kills the host on high-class-count datasets). Instead
    we carry a member-summed softmax and per-member argmax; ``member_stack`` is
    kept only as a guard that refuses to re-materialize the full tensor.
    """

    mean_softmax: torch.Tensor  # (N, C, H, W) softmax averaged over members
    member_preds: torch.Tensor  # (M, N, H, W) uint8 per-member argmax
    fold_ids: np.ndarray  # (M, N) held-out fold assignment per member
    # Per-(member, fold) training curves, empty unless ``eval_every`` was set.
    # Each entry: {"member_idx", "fold_idx", "n_train", "n_holdout", "history"}.
    curves: list[dict] = field(default_factory=list)

    @property
    def member_stack(self) -> torch.Tensor:
        """Refuse to re-materialize the full ``(M, N, C, H, W)`` float stack.

        Kept as an explicit tripwire: nothing needs the full stack (Cleanlab
        reads ``mean_softmax``, AER reads ``member_preds``), and rebuilding it
        is exactly the allocation that OOM-killed flair2.
        """
        raise AttributeError(
            "OOFResult no longer materializes member_stack (M,N,C,H,W) — it OOM-kills "
            "the host on high-class-count datasets. Use mean_softmax (Cleanlab) or "
            "member_preds (AER) instead."
        )


def run_oof(
    dataset,
    members: list[dict],
    folds: Callable[[int], np.ndarray],
    *,
    batch_size: int = 8,
    num_workers: int = 0,
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
        num_workers: DataLoader worker processes for both loaders. These datasets
            are decode-bound (512-650px tiles), so 0 leaves the GPU starved.
        tta: Apply dihedral (D4) test-time augmentation at inference.
        member_factory: Builds a member from a spec; the member must expose
            ``fit(train_loader)``, ``probe`` (native-frame logits module) and
            ``device``. Injectable for testing.

    Returns:
        An :class:`OOFResult` with ``mean_softmax`` ``(N, C, H, W)`` (the
        member-summed softmax ÷ M) and ``member_preds`` ``(M, N, H, W)`` uint8.

    Rather than allocating the full ``(M, N, C, H, W)`` float stack (which
    OOM-kills the host on high-class-count datasets), each fold's hold-out
    softmax is folded into two much smaller accumulators as it lands: a running
    ``softmax_sum`` for Cleanlab's ``mean_softmax`` and a per-member ``uint8``
    argmax for AER.
    """
    n = len(dataset)
    num_classes = members[0]["num_classes"]
    h, w = _native_frame(dataset)

    m = len(members)
    softmax_sum = torch.zeros(n, num_classes, h, w)  # (N, C, H, W) running sum over members
    member_preds = torch.zeros(m, n, h, w, dtype=torch.uint8)  # (M, N, H, W) per-member argmax
    fold_ids = np.empty((m, n), dtype=np.int64)
    curves: list[dict] = []

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
            hold_loader = _subset_loader(
                dataset, hold_idx, batch_size, shuffle=False, num_workers=num_workers
            )
            # The held-out fold doubles as the curve's eval set when the member
            # was built with ``eval_every``. ``fit(train_loader)`` stays the
            # member contract: the val loader is passed only to members that
            # accept it, so custom/injected members need no change.
            train_loader = _subset_loader(
                dataset, train_idx, batch_size, shuffle=True, num_workers=num_workers
            )
            if _accepts_val_loader(predictor.fit):
                predictor.fit(train_loader, hold_loader)
            else:
                predictor.fit(train_loader)
            history = list(getattr(predictor, "history", []) or [])
            if history:
                curves.append(
                    {
                        "member_idx": member_idx,
                        "fold_idx": fold_idx,
                        "seed": int(seed),
                        "n_train": int(train_idx.size),
                        "n_holdout": int(hold_idx.size),
                        "history": history,
                    }
                )
            preds = _predict_holdout(
                predictor,
                hold_loader,
                num_classes=num_classes,
                tta=tta,
            )
            hold = torch.as_tensor(hold_idx)
            # Fold this member's hold-out softmax into the two reductions the
            # scorers consume; the full (M,N,C,H,W) stack is never held.
            softmax_sum[hold] += preds
            member_preds[member_idx, hold] = preds.argmax(dim=1).to(torch.uint8)

    return OOFResult(
        mean_softmax=softmax_sum / m,
        member_preds=member_preds,
        fold_ids=fold_ids,
        curves=curves,
    )


def _accepts_val_loader(fit) -> bool:
    """Whether a member's ``fit`` takes a second (validation loader) argument.

    Keeps ``fit(train_loader)`` as the member contract ``run_oof`` documents:
    only members that opt into curve recording get the held-out loader, so
    injected test doubles and any custom member keep working unchanged.
    """
    try:
        params = inspect.signature(fit).parameters
    except (TypeError, ValueError):
        return False
    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params.values()):
        return True
    positional = [
        p
        for p in params.values()
        if p.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    return len(positional) >= 2


def _native_frame(dataset) -> tuple[int, int]:
    """Native label ``(H, W)`` read from the first sample's mask."""
    sample = dataset[0]
    mask = sample["mask"] if isinstance(sample, dict) else sample[1]
    if mask.ndim == 3:
        mask = mask.squeeze(0)
    return int(mask.shape[-2]), int(mask.shape[-1])


def _subset_loader(
    dataset, indices, batch_size: int, shuffle: bool, num_workers: int = 0
) -> DataLoader:
    """Batched loader over a global-index subset (indices recoverable via ``.dataset``).

    ``num_workers > 0`` parallelizes the per-sample decode, which is the
    bottleneck on these datasets: a 512-650px tile read dominates the training
    step, so a serial loader leaves the GPU idle. ``persistent_workers`` keeps
    them alive across the many loader passes a fixed-step budget makes, and
    ``pin_memory`` overlaps the host->device copy.
    """
    return DataLoader(
        Subset(dataset, list(indices)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=num_workers > 0 and torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


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
