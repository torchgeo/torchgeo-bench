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
from omegaconf import DictConfig, OmegaConf

from ..datasets.loading import get_bench_dataset_class
from . import degeneracy, store
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

    # Per-model overrides (from the hyperparameter search) win over the globals.
    # The model-side key is `finetune_batch_size`: `eval.segmentation.batch_size`
    # already means the frozen cached-feature probe batch size (64), which would
    # OOM an unfrozen fine-tune.
    seg = _seg_cfg(cfg)
    batch_size = int(_hparam(seg, lq, "batch_size", 8, seg_key="finetune_batch_size"))

    fold_ids, tier = assign_folds(
        bench, "train", k, cell_deg=float(lq.grid_cell_deg), seed=base_seed
    )

    # Load only the configured bands (RGB by default) so the OOF loader's
    # channel count matches the backbone the members are built for.
    band_names = _resolve_band_names(cfg, bench)
    train_ds = bench.get_dataset("train", bands=band_names, metadata=["lat", "lon"])
    # Parallelize the disk-bound mask read; defaults to the OOF loader worker
    # count, capped so a small dataset does not oversubscribe.
    mask_workers = int(lq.get("mask_num_workers", lq.get("num_workers", 8)))
    labels, native_ids = _stack_masks(train_ds, num_workers=mask_workers)

    model_slug = store.sanitize_slug(str(cfg.model.name))

    specs = build_member_specs(
        cfg, bench, num_classes=num_classes, device=str(cfg.device), seeds=seeds
    )

    def folds(seed: int) -> np.ndarray:
        # Relabel fold ids per member: keeps groups intact, diversifies partitions.
        return np.random.default_rng(seed).permutation(k)[fold_ids]

    factory = _make_checkpointing_factory(root, name, model_slug=model_slug, k=k, resume=resume)
    oof = run_oof(
        train_ds,
        specs,
        folds,
        member_factory=factory,
        tta=bool(lq.get("tta", True)),
        batch_size=batch_size,
        num_workers=int(lq.get("num_workers", 8)),
    )

    if oof.curves:
        curves_path = store.save_training_curves(
            root, name, model_slug, oof.curves,
            backbone_lr=_hparam(seg, lq, "backbone_lr", None),
            head_lr=_hparam(seg, lq, "head_lr", None),
            max_steps=_hparam(seg, lq, "max_steps", None),
            batch_size=batch_size,
            eval_every=int(lq.get("eval_every") or 0),
            k=k,
            n_members=n_members,
            grouping_tier=tier,
        )
        logger.info("Label-quality[%s]: wrote %d member curves to %s", name, len(oof.curves), curves_path)

    present = list(range(num_classes))
    methods = list(lq.get("methods", ["cleanlab", "aer"]))

    # AER also yields the OOF macro-mIoU used for the low-capacity annotation.
    # It consumes the per-member argmax directly (never the full float stack).
    aer_img, aer_pix = aer_score(labels, oof.member_preds, present, ignore_index=ignore_index)
    oof_miou = float(1.0 - aer_img.mean())
    low_capacity = store.is_low_capacity(
        oof_miou, threshold=float(lq.get("low_capacity_threshold", store.LOW_CAPACITY_THRESHOLD))
    )
    # The scalar mIoU gate above cannot see a majority-class collapse (a
    # background-only predictor still averages ~0.5). These GT-relative cell
    # metrics can; both reuse tensors already in memory, with no re-inference.
    cell = degeneracy.cell_metrics(
        labels, oof.member_preds, num_classes=num_classes, ignore_index=ignore_index
    )
    cov_thr = float(lq.get("degenerate_coverage_threshold", degeneracy.DEGENERATE_COVERAGE_THRESHOLD))
    iqr_thr = float(lq.get("min_score_iqr", degeneracy.MIN_SCORE_IQR))
    logger.info(
        "Label-quality[%s]: tier=%s OOF-mIoU=%.3f low_capacity=%s "
        "min_class_coverage=%.3f oof_per_class_iou_min=%.3f pooled_macro_iou=%.3f",
        name, tier, oof_miou, low_capacity,
        cell["min_class_coverage"], cell["oof_per_class_iou_min"], cell["macro_iou"],
    )

    # Persist the per-class detail the two CSV minima are taken over. `oof_miou`
    # above is a per-image proxy that averages ~0.5 for a background-only
    # predictor, so it cannot answer "did the rare class die, or is nothing
    # fitting?" -- `macro_iou` here is the pooled figure that can.
    diagnostics_path = store.save_per_class_diagnostics(
        root, name, model_slug, cell,
        num_classes=num_classes,
        ignore_index=ignore_index,
        oof_miou=oof_miou,
        low_capacity=low_capacity,
        k=k,
        n_members=n_members,
        bands=str(cfg.dataset.get("bands", "rgb")),
    )
    logger.info("Label-quality[%s]: wrote per-class diagnostics to %s", name, diagnostics_path)

    def _degeneracy_kwargs(image_scores):
        """Per-method degeneracy fields: IQR of this method's own score distribution."""
        iqr = degeneracy.score_iqr(image_scores)
        return {
            "score_iqr": iqr,
            "degenerate": degeneracy.is_degenerate(
                cell["min_class_coverage"], iqr,
                coverage_threshold=cov_thr, iqr_threshold=iqr_thr,
            ),
        }

    common = {
        "model": model_slug,
        "member_set": f"M{n_members}",
        "grouping_tier": tier,
        "folds": fold_ids,
        "native_ids": native_ids,
        "k": k,
        "n_members": n_members,
        "seed": base_seed,
        "bands": str(cfg.dataset.get("bands", "rgb")),
        "partition": str(cfg.dataset.get("partition", "default")),
        "low_capacity": low_capacity,
        # Cell-level: identical on both methods' rows. `score_iqr` / `degenerate`
        # are per method and are added at each write site instead.
        "min_class_coverage": cell["min_class_coverage"],
        "oof_per_class_iou_min": cell["oof_per_class_iou_min"],
    }

    # Predicted mask (argmax of the OOF mean softmax) is shared by both methods'
    # gallery artifacts; the per-method pixel score map is what differs.
    pred_masks = np.asarray(oof.mean_softmax.argmax(dim=1).cpu())
    n_suspect = int(lq.get("npz_top_n", 50))
    n_control = int(lq.get("npz_control_n", 20))

    bands_str = str(common["bands"])

    if "cleanlab" in methods and not _completed(
        output, name, model_slug, "cleanlab", bands_str, resume
    ):
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
            **_degeneracy_kwargs(cl_img),
            **common,
        )
        _persist_gallery_npz(
            root, name, model_slug, "cleanlab",
            image_scores=np.asarray(cl_img), pixel_scores=np.asarray(cl_pix),
            pred_masks=pred_masks, native_ids=native_ids, base_seed=base_seed,
            lower_is_suspect=True, n_suspect=n_suspect, n_control=n_control,
        )

    if "aer" in methods and not _completed(
        output, name, model_slug, "aer", bands_str, resume
    ):
        store.write_results_csv(
            output,
            dataset=name,
            method="aer",
            image_scores=aer_img,
            n_flagged_pixels=_flagged_counts(aer_pix > 0.5),
            lower_is_suspect=False,
            **_degeneracy_kwargs(aer_img),
            **common,
        )
        _persist_gallery_npz(
            root, name, model_slug, "aer",
            image_scores=np.asarray(aer_img), pixel_scores=np.asarray(aer_pix),
            pred_masks=pred_masks, native_ids=native_ids, base_seed=base_seed,
            lower_is_suspect=False, n_suspect=n_suspect, n_control=n_control,
        )


def _dataset_names(cfg: DictConfig) -> list[str]:
    """Configured dataset identifiers as a list."""
    names = cfg.dataset.names
    return [names] if isinstance(names, str) else list(names)


def _load_bench_dataset(cfg: DictConfig, name: str):
    """Instantiate the registered :class:`BenchDataset` for ``name``."""
    del cfg
    return get_bench_dataset_class(name)()


def _resolve_band_names(cfg: DictConfig, bench) -> tuple[str, ...] | None:
    """Canonical band names for ``cfg.dataset.bands`` (``None`` -> all bands).

    Single source of truth shared by the OOF loader (``get_dataset(bands=...)``)
    and the backbone factory (``select_band_specs``) so their channel counts
    always agree. Segmentation defaults to RGB.
    """
    bands = cfg.dataset.get("bands", "rgb")
    if bands == "rgb":
        return tuple(bench.rgb_bands)
    if bands in ("all", None):
        return None
    return tuple(bands)


def build_member_specs(
    cfg: DictConfig, bench, *, num_classes: int, device: str, seeds: list[int]
) -> list[dict]:
    """Build one member spec per seed from the model + segmentation config.

    Each spec's backbone is a zero-arg factory so every member gets a freshly
    initialised backbone (full fine-tuning from the same pretrained weights).
    """
    lq = cfg.label_quality
    seg = _seg_cfg(cfg)

    # Resolve the BandSpec list + normalization the backbone needs, mirroring
    # the image pipeline: models like TimmPatchBenchModel require `bands` at
    # instantiation, which the label_quality dispatch otherwise never supplies.
    # Must match the bands the OOF loader is built with (see _resolve_band_names).
    bands_list = bench.select_band_specs(_resolve_band_names(cfg, bench))
    normalization = str(cfg.dataset.get("normalization", "bandspec_zscore"))

    def make_backbone():
        # `bands`/`normalization` are passed post-hoc so Hydra never OmegaConf-ifies
        # the BandSpec list; `_convert_="object"` keeps the rest as plain Python.
        return instantiate(
            cfg.model, bands=bands_list, normalization=normalization, _convert_="object"
        )

    specs = []
    for seed in seeds:
        specs.append(
            {
                "backbone": make_backbone,
                "layers": list(seg.layers),
                "num_classes": num_classes,
                "head_type": str(seg.get("head_type", "fpn")),
                "device": device,
                "seed": int(seed),
                "max_steps": int(_hparam(seg, lq, "max_steps")),
                "epoch_cap": int(_hparam(seg, lq, "epoch_cap")),
                "backbone_lr": _hparam(seg, lq, "backbone_lr"),
                "head_lr": _hparam(seg, lq, "head_lr"),
                "weight_decay": float(_hparam(seg, lq, "weight_decay", 0.0)),
                # None -> no per-member curve (today's behavior); an int records
                # one every N steps on the held-out fold (D11).
                "eval_every": lq.get("eval_every"),
            }
        )
    return specs


def _seg_cfg(cfg: DictConfig):
    """The segmentation eval config with model-specific overrides merged in.

    Merges ``cfg.model.eval`` over ``cfg.eval`` exactly like the image pipeline
    does; the label_quality dispatch returns before that merge, so without this
    timm/ViT backbones would see empty ``layers``.
    """
    eval_cfg = cfg.eval
    if "eval" in cfg.model and cfg.model.eval is not None:
        eval_cfg = OmegaConf.merge(eval_cfg, cfg.model.eval)
    return eval_cfg.segmentation


# Sentinel distinguishing "no default given" from a legitimate ``None`` default.
_UNSET = object()


def _hparam(seg, lq, key: str, default=_UNSET, *, seg_key: str | None = None):
    """Resolve a training hyperparameter, per-model override winning over the global.

    The hyperparameter search (``docs/plans/segmentation_hparam_search.md``)
    produces *per-model* ``backbone_lr`` / ``head_lr`` / ``batch_size``, but
    these were historically read from ``cfg.label_quality`` only — one value for
    every model. Reading ``cfg.model.eval.segmentation`` first (the merged
    ``eval_cfg``, the same merge that already carries ``layers`` / ``head_type``)
    lets an adopted per-model config override the global default, while a model
    without an override still gets the ``label_quality`` value.

    Args:
        seg: Merged ``eval.segmentation`` config (per-model side), or ``None``.
        lq: The ``cfg.label_quality`` config (global side).
        key: Key to read on the ``label_quality`` side.
        default: Fallback when neither side defines the key. Omit to require it.
        seg_key: Key to read on the model side when it differs from ``key``.
            Needed for ``batch_size``, whose model-side name is
            ``finetune_batch_size`` because ``eval.segmentation.batch_size``
            already means the frozen probe batch size.
    """
    lookup = seg_key or key
    if seg is not None and lookup in seg and seg.get(lookup) is not None:
        return seg.get(lookup)
    if default is _UNSET:
        return lq.get(key) if key in lq else getattr(lq, key)
    return lq.get(key, default)


def _make_checkpointing_factory(root: str, dataset: str, *, model_slug: str, k: int, resume: bool):
    """Member factory that keys fold checkpoints by call order ``(member_idx, fold_idx)``.

    ``run_oof`` invokes the factory once per ``(member, fold)`` in order, so a
    running counter recovers the resume key deterministically. ``model_slug``
    isolates each backbone's checkpoints so a resume never cross-loads another
    model's weights (FM-1).
    """
    counter = {"i": 0}

    def factory(spec: dict) -> "_CheckpointingMember":
        member_idx, fold_idx = divmod(counter["i"], k)
        counter["i"] += 1
        predictor = build_member(spec)
        return _CheckpointingMember(
            predictor, root, dataset, model_slug, member_idx, fold_idx,
            int(spec.get("seed", 0)), resume,
        )

    return factory


class _CheckpointingMember:
    """Wraps a :class:`Predictor`, reusing a saved fold checkpoint instead of retraining."""

    def __init__(self, predictor, root, dataset, model_slug, member_idx, fold_idx, seed, resume):
        self._predictor = predictor
        self.probe = predictor.probe
        self.device = predictor.device
        self._key = {
            "root": root,
            "dataset": dataset,
            "model_slug": model_slug,
            "member_idx": member_idx,
            "fold_idx": fold_idx,
            "seed": seed,
        }
        self._resume = resume

    @property
    def history(self):
        """The wrapped member's training curve; empty when a checkpoint was reused."""
        return getattr(self._predictor, "history", [])

    def fit(self, train_loader, val_loader=None):
        root = self._key["root"]
        ckpt = {k: v for k, v in self._key.items() if k != "root"}
        if self._resume and store.checkpoint_exists(root, **ckpt):
            state = torch.load(store.checkpoint_path(root, **ckpt), weights_only=True)
            self.probe.load_state_dict(state)
            # No training ran, so there is no curve for this fold on a resume.
            return self
        self._predictor.fit(train_loader, val_loader)
        store.save_checkpoint(self.probe.state_dict(), root, **ckpt)
        return self

    def predict_proba(self, loader):
        return self._predictor.predict_proba(loader)


def _extract_mask(sample) -> torch.Tensor:
    """Pull the ``(H, W)`` long mask out of one dataset sample."""
    mask = sample["mask"] if isinstance(sample, dict) else sample[1]
    mask = torch.as_tensor(mask)
    if mask.ndim == 3:
        mask = mask.squeeze(0)
    return mask.long()


def _stack_masks(dataset, *, num_workers: int = 0) -> tuple[torch.Tensor, list[str] | None]:
    """Stack the train split's masks and collect per-sample native ids.

    Returns ``(labels, native_ids)`` where ``labels`` is a single ``(N, H, W)``
    long tensor and ``native_ids`` is a list of stable per-sample ids (from the
    dataset, see :func:`_dataset_native_ids`) or ``None`` when the dataset
    exposes no stable id — the gallery then falls back to the positional
    ``image_id``.

    With ``num_workers > 0`` the per-sample reads run in a DataLoader so the
    disk-bound decode is parallelized (a serial pass over a few-thousand-sample
    split is minutes; workers cut it near-linearly). Order is preserved
    (``shuffle=False``), so the returned labels still align with ``native_ids``
    and the positional ``image_id``.
    """
    native_ids = _dataset_native_ids(dataset)

    if num_workers and len(dataset) > 0:
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda batch: _extract_mask(batch[0]),
        )
        masks = list(loader)
    else:
        masks = [_extract_mask(dataset[i]) for i in range(len(dataset))]

    return torch.stack(masks), native_ids


def _dataset_native_ids(dataset) -> list[str] | None:
    """Best-effort stable per-sample ids for the train split.

    Tries, in order: a ``tortilla:id`` column on the upstream V2 ``data_df``
    (the swept caffe/cloudsen12/flair2/fotw/spacenet* datasets), then a
    ``sample_ids`` attribute (V1 loaders). Returns ``None`` with a warning when
    neither is present, so the pipeline still runs on positional ids.
    """
    # Unwrap the framework's GeoBenchv2 adapter (``_inner``) down to the
    # upstream loader that carries the tortilla ``data_df``.
    inner = dataset
    for _ in range(3):
        if hasattr(inner, "data_df"):
            break
        if hasattr(inner, "_inner"):
            inner = inner._inner
        else:
            break

    df = getattr(inner, "data_df", None)
    if df is not None and "tortilla:id" in getattr(df, "columns", []):
        return [str(x) for x in df["tortilla:id"].tolist()]

    sample_ids = getattr(inner, "sample_ids", None) or getattr(dataset, "sample_ids", None)
    if sample_ids is not None:
        return [str(x) for x in sample_ids]

    logger.warning(
        "No stable native id found for this dataset; falling back to positional image_id."
    )
    return None


def _persist_gallery_npz(
    root: str,
    dataset: str,
    model_slug: str,
    method: str,
    *,
    image_scores: np.ndarray,
    pixel_scores: np.ndarray,
    pred_masks: np.ndarray,
    native_ids: list[str] | None,
    base_seed: int,
    lower_is_suspect: bool,
    n_suspect: int,
    n_control: int,
) -> None:
    """Persist per-sample npz for a bounded audit set: top-N suspect + clean control.

    Everything needed is already in memory at scoring time (no re-inference).
    The suspect set is the ``n_suspect`` most-suspect images by ``image_scores``
    (direction per ``lower_is_suspect``); the control set is a deterministic
    random sample of ``n_control`` images drawn from the remainder, so the
    gallery can show a suspect-vs-clean contrast strip.
    """
    n = len(image_scores)
    order = np.argsort(image_scores if lower_is_suspect else -image_scores, kind="stable")
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(1, n + 1)

    suspect = order[: min(n_suspect, n)]
    remainder = order[min(n_suspect, n):]
    rng = np.random.default_rng(base_seed)
    control = (
        rng.choice(remainder, size=min(n_control, len(remainder)), replace=False)
        if len(remainder)
        else np.array([], dtype=int)
    )

    for idx in np.concatenate([suspect, control]).astype(int):
        native = "" if native_ids is None else native_ids[idx]
        store.save_pixel_artifact(
            root,
            dataset,
            model_slug,
            method,
            store.image_id(idx),
            np.asarray(pixel_scores[idx]),
            np.asarray(pred_masks[idx]),
            native_id=native,
            image_score=float(image_scores[idx]),
            rank=int(ranks[idx]),
        )


def _flagged_counts(mask) -> np.ndarray:
    """Per-sample count of flagged pixels from a boolean ``(N, H, W)`` mask."""
    mask = np.asarray(mask)
    return mask.reshape(mask.shape[0], -1).sum(axis=1)


def _completed(
    output: str, dataset: str, model: str, method: str, bands: str, resume: bool
) -> bool:
    """Whether ``(dataset, model, method, bands)`` rows already exist (resume skip guard).

    The key must include ``model`` and ``bands``: a sweep shards several
    backbones across nodes that all append to one CSV, so a ``(dataset,
    method)`` key would let whichever backbone writes first suppress every
    other backbone's rows for that dataset -- discarding a finished OOF run.
    """
    if not resume or not os.path.exists(output):
        return False
    import pandas as pd

    df = pd.read_csv(output)
    required = {"dataset", "model", "method", "bands"}
    if not required.issubset(df.columns):
        return False
    hit = (
        (df["dataset"] == dataset)
        & (df["model"] == model)
        & (df["method"] == method)
        & (df["bands"].astype(str) == str(bands))
    )
    return bool(hit.any())
