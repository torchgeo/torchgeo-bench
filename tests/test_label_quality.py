"""Tests for the segmentation label-quality pipeline (Cleanlab + AER)."""

from unittest.mock import MagicMock, patch

import numpy as np
import torch

from torchgeo_bench.datasets.burn_scars import BurnScars


class _CaptureUpstream:
    """Minimal stand-in for a ``geobench_v2.datasets.GeoBench<X>`` class."""

    def __init__(self, root, split, transforms=None, band_order=None, **kwargs):
        del root, split, transforms
        self.band_order = band_order
        self.kwargs = kwargs

    def __len__(self):
        return 4

    def __getitem__(self, idx):
        del idx
        return {"image": torch.randn(6, 8, 8), "mask": torch.zeros(8, 8, dtype=torch.long)}


# --- Slice 1: metadata threading -------------------------------------------


def test_metadata_kwarg_forwarded_to_upstream():
    """``metadata=[...]`` must reach the upstream loader constructor."""
    ds = BurnScars()
    mocked = MagicMock(side_effect=_CaptureUpstream)
    with patch("geobench_v2.datasets.GeoBenchBurnScars", mocked):
        ds.get_dataset(split="train", metadata=["lat", "lon"])
    assert mocked.call_args.kwargs.get("metadata") == ["lat", "lon"]


def test_metadata_kwarg_defaults_none():
    """Omitting ``metadata`` leaves the upstream call unchanged (no key)."""
    ds = BurnScars()
    mocked = MagicMock(side_effect=_CaptureUpstream)
    with patch("geobench_v2.datasets.GeoBenchBurnScars", mocked):
        ds.get_dataset(split="train")
    assert "metadata" not in mocked.call_args.kwargs


# --- Slice 3: tiered leakage-safe fold assignment --------------------------


class _FoldFakeDataset:
    """Minimal ``BenchDataset`` stand-in: ``get_dataset`` returns sample dicts."""

    def __init__(self, samples):
        self._samples = samples

    def get_dataset(self, split, *, metadata=None, **kwargs):
        del split, metadata, kwargs
        return self._samples


def _sinusoid_patch(freq: int) -> torch.Tensor:
    """A deterministic single-frequency patch with a distinct perceptual hash."""
    x = torch.linspace(0.0, 1.0, 16)
    grid = torch.outer(torch.sin((freq + 1) * 3.0 * x), torch.cos((freq + 1) * 2.0 * x))
    return grid.unsqueeze(0).repeat(3, 1, 1)


def test_spatial_block_groupkfold_keeps_cells_together():
    """lat/lon-bearing samples group by grid cell; tier is ``latlon_block``."""
    from torchgeo_bench.label_quality.folds import assign_folds

    rng = np.random.default_rng(0)
    lon = rng.uniform(-180, 180, size=400)
    lat = np.degrees(np.arcsin(rng.uniform(-1, 1, size=400)))
    samples = [
        {"image": torch.zeros(3, 4, 4), "lat": float(lat[i]), "lon": float(lon[i])}
        for i in range(400)
    ]
    fold_ids, tier = assign_folds(_FoldFakeDataset(samples), "train", k=5, cell_deg=10.0)

    assert tier == "latlon_block"
    assert fold_ids.shape == (400,)
    cell = np.floor(lat / 10.0).astype(int) * 100003 + np.floor(lon / 10.0).astype(int)
    for c in np.unique(cell):
        assert len(np.unique(fold_ids[cell == c])) == 1


def test_phash_groups_near_duplicates():
    """Byte-identical patches share a fold; tier is ``phash``."""
    from torchgeo_bench.label_quality.folds import assign_folds

    dup = _sinusoid_patch(99)
    images = [_sinusoid_patch(i) for i in range(6)] + [dup, dup.clone()]
    samples = [{"image": img} for img in images]  # no lat/lon, no native_id
    fold_ids, tier = assign_folds(_FoldFakeDataset(samples), "train", k=3)

    assert tier == "phash"
    assert fold_ids[6] == fold_ids[7]  # the duplicate pair stays together


def test_fold_cascade_falls_back_to_random():
    """No coords, no ID, no duplicates -> random tier ``leakage_uncontrolled``."""
    from torchgeo_bench.label_quality.folds import assign_folds

    samples = [{"image": _sinusoid_patch(i)} for i in range(9)]
    fold_ids, tier = assign_folds(_FoldFakeDataset(samples), "train", k=3)

    assert tier == "leakage_uncontrolled"
    assert set(np.unique(fold_ids)) == set(range(3))  # all folds populated


def test_assign_folds_shape_and_range():
    """Every tier yields ``fold_ids.shape == (N,)`` with values in ``range(k)``."""
    from torchgeo_bench.label_quality.folds import assign_folds

    rng = np.random.default_rng(1)
    lon = rng.uniform(-180, 180, size=60)
    lat = np.degrees(np.arcsin(rng.uniform(-1, 1, size=60)))
    latlon_ds = _FoldFakeDataset(
        [
            {"image": torch.zeros(3, 4, 4), "lat": float(lat[i]), "lon": float(lon[i])}
            for i in range(60)
        ]
    )
    dup = _sinusoid_patch(42)
    phash_ds = _FoldFakeDataset(
        [{"image": img} for img in [_sinusoid_patch(i) for i in range(6)] + [dup, dup.clone()]]
    )
    random_ds = _FoldFakeDataset([{"image": _sinusoid_patch(i)} for i in range(9)])

    for ds, n in ((latlon_ds, 60), (phash_ds, 8), (random_ds, 9)):
        fold_ids, _ = assign_folds(ds, "train", k=3, cell_deg=10.0)
        assert fold_ids.shape == (n,)
        assert set(np.unique(fold_ids)).issubset(set(range(3)))


# --- Slice 4: uniform member predictor -------------------------------------


class _MemberBackbone(torch.nn.Module):
    """Two stride-2 conv layers exposing spatial ``layer1``/``layer2`` features."""

    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, padding=1, stride=2), torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(16, 32, 3, padding=1, stride=2), torch.nn.ReLU()
        )

    def forward(self, x):
        return self.layer2(self.layer1(x))


def _dict_loader(images, masks, batch_size=2):
    class _DS(torch.utils.data.Dataset):
        def __len__(self):
            return len(images)

        def __getitem__(self, idx):
            return {"image": images[idx], "mask": masks[idx]}

    return torch.utils.data.DataLoader(_DS(), batch_size=batch_size)


def _member_spec(**overrides):
    spec = {
        "backbone": _MemberBackbone(),
        "layers": ["layer1", "layer2"],
        "num_classes": 5,
        "head_type": "fpn",
        "device": "cpu",
        "seed": 0,
    }
    spec.update(overrides)
    return spec


def test_build_member_unfrozen_probe():
    """The wrapped SegmentationProbe fine-tunes: unfrozen with gradients."""
    from torchgeo_bench.label_quality.predictors import build_member

    member = build_member(_member_spec())
    assert member.probe.freeze_backbone is False
    assert all(p.requires_grad for p in member.probe.backbone.parameters())


def test_member_predict_proba_shape_and_normalization():
    """predict_proba yields (N,C,H,W) probabilities at the native label frame."""
    from torchgeo_bench.label_quality.predictors import build_member

    images = torch.randn(4, 3, 64, 64)
    masks = torch.randint(0, 5, (4, 64, 64))
    loader = _dict_loader(images, masks)
    member = build_member(_member_spec(max_steps=2))

    member.fit(loader)
    probs = member.predict_proba(loader)

    assert probs.shape == (4, 5, 64, 64)
    assert torch.isfinite(probs).all()
    assert (probs >= 0).all()
    assert torch.allclose(probs.sum(dim=1), torch.ones(4, 64, 64), atol=1e-4)


def test_member_predict_proba_upsamples_to_label_frame():
    """Logits are upsampled to the native mask frame; labels are never resampled."""
    from torchgeo_bench.label_quality.predictors import build_member

    images = torch.randn(2, 3, 64, 64)  # forward resolution
    masks = torch.randint(0, 5, (2, 128, 128))  # native label frame (larger)
    loader = _dict_loader(images, masks)
    member = build_member(_member_spec())

    raw = member.probe(images)  # probe emits at the (smaller) image frame
    assert raw.shape[-2:] == (64, 64)

    probs = member.predict_proba(loader)
    assert probs.shape == (2, 5, 128, 128)
    assert masks.shape == (2, 128, 128)  # labels untouched
    assert torch.allclose(probs.sum(dim=1), torch.ones(2, 128, 128), atol=1e-4)


# --- Slice 5: member-level OOF substrate with dihedral TTA ------------------


class _SegDS(torch.utils.data.Dataset):
    """Synthetic seg dataset yielding ``{"image", "mask"}`` dicts."""

    def __init__(self, images, masks):
        self.images = images
        self.masks = masks

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {"image": self.images[idx], "mask": self.masks[idx]}


class _ZeroProbe(torch.nn.Module):
    """Spatial-identity probe emitting ``num_classes`` all-zero logit channels."""

    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        b, _, h, w = x.shape
        return torch.zeros(b, self.num_classes, h, w)


class _RecordingMember:
    """Fake member recording the train-fold index set it was fitted on."""

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.trained = None
        self.probe = _ZeroProbe(num_classes)
        self.device = "cpu"

    def fit(self, loader):
        self.trained = set(loader.dataset.indices)
        return self


class _IdentityProbe(torch.nn.Module):
    """Returns its input unchanged (logits == input channels)."""

    def forward(self, x):
        return x


class _IdentityMember:
    """Fake member whose forward is the identity (isolates the TTA machinery)."""

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.probe = _IdentityProbe()
        self.device = "cpu"

    def fit(self, loader):
        del loader
        return self


def test_run_oof_every_sample_held_out_once():
    """Each sample is predicted by a member trained only on the other folds."""
    from torchgeo_bench.label_quality.oof import run_oof

    n, c, h, w, k = 12, 3, 8, 8, 3
    ds = _SegDS(torch.randn(n, c, h, w), torch.randint(0, c, (n, h, w)))
    fold_ids = np.array([i % k for i in range(n)])

    created = []

    def factory(spec):
        member = _RecordingMember(spec["num_classes"])
        created.append(member)
        return member

    oof = run_oof(
        ds,
        [{"num_classes": c, "seed": 0}],
        lambda seed: fold_ids,
        member_factory=factory,
        tta=False,
        batch_size=4,
    )

    assert len(created) == k  # one predictor per held-out fold, single member
    for fold_idx, member in enumerate(created):
        holdout = set(np.where(fold_ids == fold_idx)[0])
        assert member.trained == set(range(n)) - holdout  # trained on the complement
        assert member.trained.isdisjoint(holdout)
    np.testing.assert_array_equal(oof.fold_ids[0], fold_ids)


def test_oof_views_shapes():
    """``member_stack`` is (M,N,C,H,W); ``mean_softmax`` is its member-axis mean."""
    from torchgeo_bench.label_quality.oof import run_oof

    n, c, h, w, k = 8, 4, 6, 6, 2
    ds = _SegDS(torch.randn(n, 3, h, w), torch.randint(0, c, (n, h, w)))
    fold_ids = np.array([i % k for i in range(n)])
    members = [{"num_classes": c, "seed": 0}, {"num_classes": c, "seed": 1}]

    oof = run_oof(
        ds,
        members,
        lambda seed: fold_ids,
        member_factory=lambda spec: _RecordingMember(spec["num_classes"]),
        tta=False,
        batch_size=4,
    )

    assert oof.member_stack.shape == (2, n, c, h, w)
    assert oof.mean_softmax.shape == (n, c, h, w)
    assert torch.allclose(oof.mean_softmax, oof.member_stack.mean(0))


def test_dihedral_tta_invertible_identity():
    """Every dihedral op has an exact inverse and TTA recovers an asymmetric pattern."""
    from torchgeo_bench.label_quality.oof import _DIHEDRAL_OPS, run_oof

    # Each op's inverse is exact (integer pixel permutations, no interpolation).
    probe_frame = torch.randn(2, 4, 8, 8)
    for fwd, inv in _DIHEDRAL_OPS:
        assert torch.allclose(inv(fwd(probe_frame)), probe_frame, atol=1e-6)

    # An asymmetric one-hot pattern: TTA averaging must map every op back and
    # reproduce the pattern's argmax exactly (a wrong inverse would blur it).
    c, h = 4, 8
    logits = torch.zeros(c, h, h)
    for i in range(h):
        for j in range(h):
            logits[(i * 3 + j) % c, i, j] = 6.0
    pattern = logits.argmax(0)

    n, k = 4, 2
    ds = _SegDS(logits.unsqueeze(0).repeat(n, 1, 1, 1), pattern.unsqueeze(0).repeat(n, 1, 1))
    oof = run_oof(
        ds,
        [{"num_classes": c, "seed": 0}],
        lambda seed: np.array([i % k for i in range(n)]),
        member_factory=lambda spec: _IdentityMember(spec["num_classes"]),
        tta=True,
        batch_size=2,
    )

    for i in range(n):
        assert torch.equal(oof.mean_softmax[i].argmax(0), pattern)


def test_members_reseeded_distinctly():
    """The M members use distinct fold partitions (reseeded per member)."""
    from torchgeo_bench.label_quality.oof import run_oof

    n, c, k = 10, 3, 2

    def folds(seed):
        return np.random.default_rng(seed).permutation(n) % k

    ds = _SegDS(torch.randn(n, 3, 6, 6), torch.randint(0, c, (n, 6, 6)))
    oof = run_oof(
        ds,
        [{"num_classes": c, "seed": 0}, {"num_classes": c, "seed": 1}],
        folds,
        member_factory=lambda spec: _RecordingMember(spec["num_classes"]),
        tta=False,
        batch_size=4,
    )

    assert oof.fold_ids.shape == (2, n)
    assert not np.array_equal(oof.fold_ids[0], oof.fold_ids[1])


# --- Slice 6: Cleanlab per-pixel scoring with soft-min aggregation ---------


def _confident_probs(labels, num_classes, conf=0.9):
    """One-hot-ish softmax probs (N,C,H,W) confidently agreeing with ``labels``."""
    n, h, w = labels.shape
    probs = torch.full((n, num_classes, h, w), (1.0 - conf) / (num_classes - 1))
    probs.scatter_(1, labels.clamp(0, num_classes - 1).unsqueeze(1), conf)
    return probs


def test_cleanlab_score_outputs_shapes():
    """Shapes/dtypes: image (N,), pixel maps (N,H,W), boolean issue masks (N,H,W)."""
    from torchgeo_bench.label_quality.cleanlab_score import score

    n, c, h, w = 3, 4, 8, 8
    probs = torch.randn(n, c, h, w).softmax(dim=1)
    labels = torch.randint(0, c, (n, h, w))

    image_scores, pixel_score_maps, issue_masks = score(labels, probs)

    assert image_scores.shape == (n,)
    assert (image_scores >= 0).all() and (image_scores <= 1).all()
    assert pixel_score_maps.shape == (n, h, w)
    assert issue_masks.shape == (n, h, w)
    assert issue_masks.dtype == bool


def test_cleanlab_flags_planted_mislabel():
    """A confidently-disagreeing region ranks more suspect and is flagged."""
    from torchgeo_bench.label_quality.cleanlab_score import score

    n, c, h, w = 2, 3, 12, 12
    labels = torch.zeros(n, h, w, dtype=torch.long)  # all class 0
    clean = _confident_probs(labels, c)

    planted = clean.clone()
    planted[0, :, 1:5, 1:5] = 0.02
    planted[0, 1, 1:5, 1:5] = 0.96  # confidently predict class 1 where label is 0

    clean_img, _, _ = score(labels, clean)
    bad_img, _, bad_issues = score(labels, planted)

    assert bad_img[0] < clean_img[0]  # planted image more suspect
    assert bad_issues[0, 1:5, 1:5].any()  # the region is flagged


def test_cleanlab_ignores_ignore_index():
    """Pixels at ``ignore_index`` are excluded: the score is invariant to their content."""
    from torchgeo_bench.label_quality.cleanlab_score import score

    n, c, h, w = 2, 3, 10, 10
    labels = torch.randint(0, c, (n, h, w))
    labels[:, 2:6, 2:6] = 255  # ignored block

    probs_a = _confident_probs(labels, c)
    probs_b = probs_a.clone()
    probs_b[:, :, 2:6, 2:6] = torch.randn(n, c, 4, 4).softmax(dim=1)  # wildly different content

    img_a, _, _ = score(labels, probs_a)
    img_b, _, _ = score(labels, probs_b)

    assert np.allclose(img_a, img_b)


def test_cleanlab_soft_min_temp_effect():
    """Low soft-min temperature weights the worst pixels more strongly than the mean."""
    from torchgeo_bench.label_quality.cleanlab_score import score

    n, c, h, w = 1, 3, 12, 12
    labels = torch.zeros(n, h, w, dtype=torch.long)
    probs = _confident_probs(labels, c, conf=0.95)
    probs[0, :, 0, 0] = 0.02
    probs[0, 1, 0, 0] = 0.96  # one confidently-wrong pixel (very low pixel score)

    img_low, pixel_maps, _ = score(labels, probs, soft_min_temp=0.01)
    img_high, _, _ = score(labels, probs, soft_min_temp=10.0)

    assert img_low[0] < pixel_maps[0].mean()  # soft-min emphasizes the worst pixel
    assert img_low[0] <= img_high[0]  # colder temperature ranks more suspect


# --- Slice 7: AER macro-IoU disagreement scoring ---------------------------


def _onehot_stack(preds, num_classes):
    """Build a one-hot ``(M,N,C,H,W)`` member stack from argmax preds ``(M,N,H,W)``."""
    m, n, h, w = preds.shape
    stack = torch.zeros(m, n, num_classes, h, w)
    stack.scatter_(2, preds.clamp(0, num_classes - 1).unsqueeze(2), 1.0)
    return stack


def test_aer_score_outputs_shapes():
    """Shapes: image scores (N,) in [0,1]; pixel disagreement maps (N,H,W)."""
    from torchgeo_bench.label_quality.aer_score import score

    m, n, c, h, w = 3, 4, 5, 8, 8
    member_stack = torch.randn(m, n, c, h, w).softmax(dim=2)
    labels = torch.randint(0, c, (n, h, w))

    image_scores, pixel_maps = score(labels, member_stack, list(range(c)))

    assert image_scores.shape == (n,)
    assert (image_scores >= 0).all() and (image_scores <= 1).all()
    assert pixel_maps.shape == (n, h, w)


def test_aer_perfect_agreement_zero_score():
    """Members matching the label exactly yield score ~0."""
    from torchgeo_bench.label_quality.aer_score import score

    m, n, c, h, w = 3, 4, 5, 8, 8
    labels = torch.randint(0, c, (n, h, w))
    preds = labels.unsqueeze(0).repeat(m, 1, 1, 1)  # every member == label
    member_stack = _onehot_stack(preds, c)

    image_scores, _ = score(labels, member_stack, list(range(c)))

    assert np.allclose(image_scores, 0.0, atol=1e-6)


def test_aer_macro_excludes_absent_and_ignore():
    """Macro-IoU is over present classes only; absent classes and ignore are excluded."""
    from torchgeo_bench.label_quality.aer_score import score

    m, c, h, w = 2, 5, 10, 10
    labels = torch.zeros(1, h, w, dtype=torch.long)  # background = class 0
    labels[0, 0:2, 0:2] = 1  # a small rare-class-1 region (4 px)
    labels[0, 5:8, 5:8] = 255  # ignored block

    preds = torch.zeros(m, 1, h, w, dtype=torch.long)  # every member misses class 1
    member_stack = _onehot_stack(preds, c)

    img_sub, _ = score(labels, member_stack, [0, 1])
    # The rare-class failure is NOT masked by the dominant background (macro weights it).
    assert img_sub[0] > 0.4

    # Absent classes {2,3,4} in present_classes do not change the score.
    img_full, _ = score(labels, member_stack, list(range(c)))
    assert np.allclose(img_sub, img_full)

    # Predictions inside the ignore block do not enter the score.
    member_stack2 = member_stack.clone()
    garbage = torch.randint(0, c, (m, 1, 3, 3))
    member_stack2[:, :, :, 5:8, 5:8] = _onehot_stack(garbage, c)[:, :, :, :, :]
    img_ignore, _ = score(labels, member_stack2, [0, 1])
    assert np.allclose(img_sub, img_ignore)


def test_aer_spurious_foreground_flags_empty_label():
    """An all-background label with predicted foreground yields high 1-IoU."""
    from torchgeo_bench.label_quality.aer_score import score

    m, c, h, w = 2, 3, 10, 10
    labels = torch.zeros(1, h, w, dtype=torch.long)  # empty / all background

    preds = torch.zeros(m, 1, h, w, dtype=torch.long)
    preds[:, 0, 2:8, 2:8] = 1  # spurious 6x6 foreground -> IoU(bg) = 64/100
    member_stack = _onehot_stack(preds, c)

    image_scores, pixel_maps = score(labels, member_stack, list(range(c)))

    assert image_scores[0] > 0.3  # 1 - 0.64 = 0.36
    assert pixel_maps[0, 2:8, 2:8].mean() > 0.5  # the spurious region disagrees


# --- Slice 8: storage, resume-checkpoint keys, inspection overlays ---------


_CSV_SCHEMA = {
    "dataset",
    "method",
    "member_set",
    "image_id",
    "image_score",
    "rank",
    "grouping_tier",
    "fold",
    "n_flagged_pixels",
    "k",
    "n_members",
    "seed",
    "bands",
    "partition",
    "low_capacity",
    "native_id",
}


def test_write_results_csv_schema(tmp_path):
    """The tidy CSV has the exact schema and zero-padded image ids."""
    import pandas as pd

    from torchgeo_bench.label_quality.store import write_results_csv

    n = 50
    path = tmp_path / "label_quality_results.csv"
    write_results_csv(
        str(path),
        dataset="flair2",
        method="cleanlab",
        member_set="M5",
        image_scores=np.linspace(0.0, 1.0, n),
        n_flagged_pixels=np.arange(n),
        grouping_tier="latlon_block",
        folds=np.arange(n) % 5,
        native_ids=[f"scene{i}" for i in range(n)],
        k=5,
        n_members=5,
        seed=0,
        bands="rgb",
        partition="default",
        low_capacity=False,
    )

    df = pd.read_csv(path, dtype={"image_id": str})
    assert set(df.columns) == _CSV_SCHEMA
    assert len(df) == n
    assert "00042" in df["image_id"].values  # zero-padded
    assert set(df["rank"]) == set(range(1, n + 1))  # dense 1..N ranking


def test_npz_roundtrip_pixel_artifacts(tmp_path):
    """Per-sample pixel artifacts round-trip under the documented path layout."""
    import os

    from torchgeo_bench.label_quality.store import load_pixel_artifact, save_pixel_artifact

    pixel_scores = np.random.rand(8, 8).astype(np.float32)
    pred_mask = np.random.randint(0, 5, (8, 8)).astype(np.int64)

    path = save_pixel_artifact(str(tmp_path), "flair2", "aer", "00042", pixel_scores, pred_mask)
    assert path.endswith(os.path.join("label_quality", "flair2", "aer", "00042.npz"))

    loaded_scores, loaded_mask = load_pixel_artifact(str(tmp_path), "flair2", "aer", "00042")
    np.testing.assert_array_equal(loaded_scores, pixel_scores)
    np.testing.assert_array_equal(loaded_mask, pred_mask)


def test_checkpoint_key_and_exists(tmp_path):
    """The checkpoint path encodes (dataset, member_idx, fold_idx, seed) and exists after write."""
    from torchgeo_bench.label_quality.store import (
        checkpoint_exists,
        save_checkpoint,
    )

    key = {"dataset": "flair2", "member_idx": 1, "fold_idx": 2, "seed": 7}
    assert checkpoint_exists(str(tmp_path), **key) is False

    path = save_checkpoint({"w": torch.zeros(3)}, str(tmp_path), **key)
    for token in ("flair2", "member1", "fold2", "seed7"):
        assert token in path
    assert checkpoint_exists(str(tmp_path), **key) is True


def test_low_capacity_flag_threshold(tmp_path):
    """Sub-threshold OOF-mIoU sets low_capacity=True and annotates rows without dropping any."""
    import pandas as pd

    from torchgeo_bench.label_quality.store import is_low_capacity, write_results_csv

    assert is_low_capacity(0.1, threshold=0.3) is True
    assert is_low_capacity(0.5, threshold=0.3) is False

    n = 10
    path = tmp_path / "lc.csv"
    write_results_csv(
        str(path),
        dataset="d",
        method="cleanlab",
        member_set="M3",
        image_scores=np.zeros(n),
        n_flagged_pixels=np.zeros(n, dtype=int),
        grouping_tier="phash",
        folds=np.zeros(n, dtype=int),
        native_ids=None,
        k=3,
        n_members=3,
        seed=1,
        bands="rgb",
        partition="default",
        low_capacity=is_low_capacity(0.1, threshold=0.3),
    )

    df = pd.read_csv(path)
    assert df["low_capacity"].all()  # every row annotated
    assert len(df) == n  # nothing dropped


# --- Slice 9: orchestration, resume, mode=label_quality dispatch -----------


class _SynthSegBench:
    """Minimal segmentation ``BenchDataset`` stand-in for orchestration tests."""

    task = "segmentation"
    num_classes = 4

    def __init__(self, ds):
        self._ds = ds

    def get_dataset(self, split, *, metadata=None, **kwargs):
        del split, metadata, kwargs
        return self._ds


def _synthetic_seg_dataset(n=12, h=8, w=8, c=4, seed=0):
    torch.manual_seed(seed)
    images = torch.randn(n, 3, h, w)
    masks = torch.randint(0, c, (n, h, w))
    return _SegDS(images, masks)


def _tiny_specs(cfg, *, num_classes, device, seeds):
    """Member specs backed by the tiny conv backbone (no registry / network)."""
    return [
        {
            "backbone": _MemberBackbone,  # zero-arg factory -> fresh backbone per member
            "layers": ["layer1", "layer2"],
            "num_classes": num_classes,
            "head_type": "fpn",
            "device": device,
            "seed": s,
            "max_steps": 2,
            "epoch_cap": 1,
        }
        for s in seeds
    ]


def _lq_cfg(tmp_path, **overrides):
    from omegaconf import OmegaConf

    lq = {
        "output": str(tmp_path / "lq.csv"),
        "k": 3,
        "n_members": 2,
        "grid_cell_deg": 10.0,
        "methods": ["cleanlab", "aer"],
        "max_steps": 2,
        "epoch_cap": 1,
        "tta": False,
        "cleanlab_soft_min_temp": 0.1,
        "batch_size": 4,
        "ignore_index": 255,
    }
    lq.update(overrides)
    return OmegaConf.create(
        {
            "seed": 0,
            "device": "cpu",
            "resume": False,
            "mode": "label_quality",
            "model": {"_target_": "unused", "name": "mock"},
            "dataset": {"names": ["synthetic"], "bands": "rgb", "partition": "default"},
            "eval": {"segmentation": {"layers": ["layer1", "layer2"], "head_type": "fpn"}},
            "label_quality": lq,
        }
    )


def test_run_label_quality_end_to_end(tmp_path, monkeypatch):
    """A full CPU run writes both cleanlab and aer rows, one per sample per method."""
    import pandas as pd

    from torchgeo_bench.label_quality import run as lq_run

    ds = _synthetic_seg_dataset()
    monkeypatch.setattr(lq_run, "_load_bench_dataset", lambda cfg, name: _SynthSegBench(ds))
    monkeypatch.setattr(lq_run, "build_member_specs", _tiny_specs)

    cfg = _lq_cfg(tmp_path)
    lq_run.run_label_quality(cfg)

    df = pd.read_csv(cfg.label_quality.output)
    assert set(df["method"]) == {"cleanlab", "aer"}
    assert len(df) == 2 * len(ds)  # one row per sample per method
    for method in ("cleanlab", "aer"):
        assert (df["method"] == method).sum() == len(ds)
    assert df["grouping_tier"].notna().all()
    assert df["grouping_tier"].iloc[0] != ""


def test_run_label_quality_resume_skips_training(tmp_path, monkeypatch):
    """Resume reuses fold checkpoints: Predictor.fit is never called and rows are stable."""
    import pandas as pd

    from torchgeo_bench.label_quality import run as lq_run
    from torchgeo_bench.label_quality.predictors import Predictor

    ds = _synthetic_seg_dataset()
    monkeypatch.setattr(lq_run, "_load_bench_dataset", lambda cfg, name: _SynthSegBench(ds))
    monkeypatch.setattr(lq_run, "build_member_specs", _tiny_specs)

    cfg = _lq_cfg(tmp_path)
    lq_run.run_label_quality(cfg)
    n_first = len(pd.read_csv(cfg.label_quality.output))

    def _boom(self, *a, **k):
        raise AssertionError("Predictor.fit must not be called on resume")

    monkeypatch.setattr(Predictor, "fit", _boom)
    cfg.resume = True
    lq_run.run_label_quality(cfg)  # must not raise (checkpoints reused)

    assert len(pd.read_csv(cfg.label_quality.output)) == n_first  # nothing re-appended
