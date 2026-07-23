"""Orchestration for the segmentation label-quality pipeline.

Wires the whole diagnostic behind ``mode=label_quality``: assign leakage-safe
folds, run the member-level OOF substrate (with dihedral TTA), score every
train mask with both Cleanlab and AER, and persist tidy rows + per-sample
artifacts. Resume is anchored at the fold-model checkpoint
``(dataset, member_idx, fold_idx, seed)``: on a rerun, existing checkpoints are
loaded instead of retrained, scores are recomputed from them, and rows already
in the CSV are not re-appended.
"""

import logging
import os

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from ..datasets.loading import get_bench_dataset_class
from . import store
from .aer_score import score as aer_score
from .cleanlab_score import score as cleanlab_score
from .folds import assign_folds
from .oof import run_oof
from .predictors import build_member

logger = logging.getLogger(__name__)


def run_label_quality(cfg: DictConfig) -> None:
    """Run the label-quality diagnostic for every configured dataset."""
    lq = cfg.label_quality
    output = str(lq.output)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    root = os.path.dirname(output) or "."
    resume = bool(cfg.get("resume", False))

    for name in _dataset_names(cfg):
        logger.info("Label-quality: dataset=%s", name)
        _run_one_dataset(cfg, name, output=output, root=root, resume=resume)

    logger.info("Label-quality complete. Results appended to %s", output)


def _run_one_dataset(cfg: DictConfig, name: str, *, output: str, root: str, resume: bool) -> None:
    lq = cfg.label_quality
    bench = _load_bench_dataset(cfg, name)
    num_classes = int(bench.num_classes)
    ignore_index = int(lq.get("ignore_index", 255))
    k = int(lq.k)
    n_members = int(lq.n_members)
    base_seed = int(cfg.seed)
    seeds = [base_seed + i for i in range(n_members)]

    fold_ids, tier = assign_folds(
        bench, "train", k, cell_deg=float(lq.grid_cell_deg), seed=base_seed
    )

    train_ds = bench.get_dataset("train", metadata=["lat", "lon"])
    labels = _stack_masks(train_ds)

    specs = build_member_specs(cfg, num_classes=num_classes, device=str(cfg.device), seeds=seeds)

    def folds(seed: int) -> np.ndarray:
        # Relabel fold ids per member: keeps groups intact, diversifies partitions.
        return np.random.default_rng(seed).permutation(k)[fold_ids]

    factory = _make_checkpointing_factory(root, name, k=k, resume=resume)
    oof = run_oof(
        train_ds,
        specs,
        folds,
        member_factory=factory,
        tta=bool(lq.get("tta", True)),
        batch_size=int(lq.get("batch_size", 8)),
    )

    present = list(range(num_classes))
    methods = list(lq.get("methods", ["cleanlab", "aer"]))

    # AER also yields the OOF macro-mIoU used for the low-capacity annotation.
    aer_img, aer_pix = aer_score(labels, oof.member_stack, present, ignore_index=ignore_index)
    oof_miou = float(1.0 - aer_img.mean())
    low_capacity = store.is_low_capacity(
        oof_miou, threshold=float(lq.get("low_capacity_threshold", store.LOW_CAPACITY_THRESHOLD))
    )
    logger.info("Label-quality[%s]: tier=%s OOF-mIoU=%.3f low_capacity=%s", name, tier, oof_miou, low_capacity)

    common = {
        "member_set": f"M{n_members}",
        "grouping_tier": tier,
        "folds": fold_ids,
        "native_ids": None,
        "k": k,
        "n_members": n_members,
        "seed": base_seed,
        "bands": str(cfg.dataset.get("bands", "rgb")),
        "partition": str(cfg.dataset.get("partition", "default")),
        "low_capacity": low_capacity,
    }

    if "cleanlab" in methods and not _completed(output, name, "cleanlab", resume):
        cl_img, cl_pix, cl_issue = cleanlab_score(
            labels,
            oof.mean_softmax,
            ignore_index=ignore_index,
            soft_min_temp=float(lq.get("cleanlab_soft_min_temp", 0.1)),
        )
        store.write_results_csv(
            output,
            dataset=name,
            method="cleanlab",
            image_scores=cl_img,
            n_flagged_pixels=_flagged_counts(cl_issue),
            lower_is_suspect=True,
            **common,
        )

    if "aer" in methods and not _completed(output, name, "aer", resume):
        store.write_results_csv(
            output,
            dataset=name,
            method="aer",
            image_scores=aer_img,
            n_flagged_pixels=_flagged_counts(aer_pix > 0.5),
            lower_is_suspect=False,
            **common,
        )


def _dataset_names(cfg: DictConfig) -> list[str]:
    """Configured dataset identifiers as a list."""
    names = cfg.dataset.names
    return [names] if isinstance(names, str) else list(names)


def _load_bench_dataset(cfg: DictConfig, name: str):
    """Instantiate the registered :class:`BenchDataset` for ``name``."""
    del cfg
    return get_bench_dataset_class(name)()


def build_member_specs(cfg: DictConfig, *, num_classes: int, device: str, seeds: list[int]) -> list[dict]:
    """Build one member spec per seed from the model + segmentation config.

    Each spec's backbone is a zero-arg factory so every member gets a freshly
    initialised backbone (full fine-tuning from the same pretrained weights).
    """
    lq = cfg.label_quality
    seg = cfg.eval.segmentation
    specs = []
    for seed in seeds:
        specs.append(
            {
                "backbone": lambda: instantiate(cfg.model, _convert_="object"),
                "layers": list(seg.layers),
                "num_classes": num_classes,
                "head_type": str(seg.get("head_type", "fpn")),
                "device": device,
                "seed": int(seed),
                "max_steps": int(lq.max_steps),
                "epoch_cap": int(lq.epoch_cap),
                "backbone_lr": lq.get("backbone_lr"),
                "head_lr": lq.get("head_lr"),
                "weight_decay": float(lq.get("weight_decay", 0.0)),
            }
        )
    return specs


def _make_checkpointing_factory(root: str, dataset: str, *, k: int, resume: bool):
    """Member factory that keys fold checkpoints by call order ``(member_idx, fold_idx)``.

    ``run_oof`` invokes the factory once per ``(member, fold)`` in order, so a
    running counter recovers the resume key deterministically.
    """
    counter = {"i": 0}

    def factory(spec: dict) -> "_CheckpointingMember":
        member_idx, fold_idx = divmod(counter["i"], k)
        counter["i"] += 1
        predictor = build_member(spec)
        return _CheckpointingMember(
            predictor, root, dataset, member_idx, fold_idx, int(spec.get("seed", 0)), resume
        )

    return factory


class _CheckpointingMember:
    """Wraps a :class:`Predictor`, reusing a saved fold checkpoint instead of retraining."""

    def __init__(self, predictor, root, dataset, member_idx, fold_idx, seed, resume):
        self._predictor = predictor
        self.probe = predictor.probe
        self.device = predictor.device
        self._key = {
            "root": root,
            "dataset": dataset,
            "member_idx": member_idx,
            "fold_idx": fold_idx,
            "seed": seed,
        }
        self._resume = resume

    def fit(self, train_loader):
        root = self._key["root"]
        ckpt = {k: v for k, v in self._key.items() if k != "root"}
        if self._resume and store.checkpoint_exists(root, **ckpt):
            state = torch.load(store.checkpoint_path(root, **ckpt), weights_only=True)
            self.probe.load_state_dict(state)
            return self
        self._predictor.fit(train_loader)
        store.save_checkpoint(self.probe.state_dict(), root, **ckpt)
        return self

    def predict_proba(self, loader):
        return self._predictor.predict_proba(loader)


def _stack_masks(dataset) -> torch.Tensor:
    """Stack the train split's masks into a single ``(N, H, W)`` long tensor."""
    masks = []
    for i in range(len(dataset)):
        sample = dataset[i]
        mask = sample["mask"] if isinstance(sample, dict) else sample[1]
        mask = torch.as_tensor(mask)
        if mask.ndim == 3:
            mask = mask.squeeze(0)
        masks.append(mask.long())
    return torch.stack(masks)


def _flagged_counts(mask) -> np.ndarray:
    """Per-sample count of flagged pixels from a boolean ``(N, H, W)`` mask."""
    mask = np.asarray(mask)
    return mask.reshape(mask.shape[0], -1).sum(axis=1)


def _completed(output: str, dataset: str, method: str, resume: bool) -> bool:
    """Whether ``(dataset, method)`` rows already exist (resume skip guard)."""
    if not resume or not os.path.exists(output):
        return False
    import pandas as pd

    df = pd.read_csv(output)
    if "dataset" not in df.columns or "method" not in df.columns:
        return False
    return bool(((df["dataset"] == dataset) & (df["method"] == method)).any())
