"""Tests for the segmentation label-quality pipeline (Cleanlab + AER)."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

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


class _DataFrameStub:
    """Minimal ``data_df`` stand-in exposing ``columns`` + column access."""

    def __init__(self, cols: dict):
        self._cols = cols
        self.columns = list(cols)

    def __getitem__(self, key):
        class _Col:
            def __init__(s, vals):
                s._vals = vals

            def tolist(s):
                return list(s._vals)

        return _Col(self._cols[key])


class _DfLoader:
    """A loader carrying a ``data_df`` (like the V2 upstream), with N samples."""

    def __init__(self, df, n):
        self.data_df = df
        self._n = n

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        raise AssertionError("fast fold path must not materialize samples")


class _DfBench:
    """BenchDataset stand-in whose ``get_dataset`` returns a ``data_df`` loader."""

    def __init__(self, df, n):
        self._df, self._n = df, n

    def get_dataset(self, split, *, metadata=None, **kwargs):
        del split, metadata, kwargs
        return _DfLoader(self._df, self._n)


def test_assign_folds_fast_path_uses_data_df_without_image_io():
    """latlon_block folds come from data_df lat/lon columns; no sample is read.

    Regression for the mask-loading speedup: the fast path must (a) never call
    __getitem__ (which raises here), and (b) produce byte-identical folds to the
    slow, sample-materializing path on the same coordinates.
    """
    from torchgeo_bench.label_quality.folds import assign_folds

    rng = np.random.default_rng(0)
    lon = rng.uniform(-180, 180, size=200)
    lat = np.degrees(np.arcsin(rng.uniform(-1, 1, size=200)))

    df = _DataFrameStub({"lat": lat, "lon": lon})
    fast_ids, fast_tier = assign_folds(_DfBench(df, 200), "train", k=5, cell_deg=10.0)

    samples = [
        {"image": torch.zeros(3, 4, 4), "lat": float(lat[i]), "lon": float(lon[i])}
        for i in range(200)
    ]
    slow_ids, slow_tier = assign_folds(_FoldFakeDataset(samples), "train", k=5, cell_deg=10.0)

    assert fast_tier == slow_tier == "latlon_block"
    np.testing.assert_array_equal(fast_ids, slow_ids)  # identical folds, no image I/O


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
    """``mean_softmax`` is (N,C,H,W); ``member_preds`` is (M,N,H,W) uint8 argmax."""
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

    assert oof.mean_softmax.shape == (n, c, h, w)
    assert oof.member_preds.shape == (2, n, h, w)
    assert oof.member_preds.dtype == torch.uint8
    assert int(oof.member_preds.max()) < c  # every stored argmax is a valid class id


def test_oof_mean_softmax_matches_full_stack_mean():
    """``mean_softmax`` (running sum ÷ M) equals the old ``stack.mean(0)`` path.

    Rebuilds the discarded full ``(M,N,C,H,W)`` stack from the per-member
    hold-out softmax and asserts the accumulator-based mean is numerically
    identical (up to float summation order).
    """
    from torchgeo_bench.label_quality.oof import run_oof

    n, c, h, w, k = 6, 3, 5, 5, 2
    imgs = torch.randn(n, c, h, w)
    ds = _SegDS(imgs, torch.randint(0, c, (n, h, w)))
    fold_ids = np.array([i % k for i in range(n)])
    members = [{"num_classes": c, "seed": 0}, {"num_classes": c, "seed": 1}]

    oof = run_oof(
        ds,
        members,
        lambda seed: fold_ids,
        member_factory=lambda spec: _IdentityMember(spec["num_classes"]),
        tta=False,
        batch_size=3,
    )

    # Identity member with tta=False -> each member's softmax is softmax(image).
    per_member = imgs.softmax(dim=1)  # (N, C, H, W)
    stack = per_member.unsqueeze(0).repeat(2, 1, 1, 1, 1)  # (M, N, C, H, W)
    assert torch.allclose(oof.mean_softmax, stack.mean(dim=0), atol=1e-6)


def test_oof_member_stack_refuses_rematerialization():
    """The old ``member_stack`` view is a tripwire, not a re-allocation."""
    from torchgeo_bench.label_quality.oof import run_oof

    n, c, k = 4, 3, 2
    ds = _SegDS(torch.randn(n, 3, 5, 5), torch.randint(0, c, (n, 5, 5)))
    oof = run_oof(
        ds,
        [{"num_classes": c, "seed": 0}],
        lambda seed: np.array([i % k for i in range(n)]),
        member_factory=lambda spec: _RecordingMember(spec["num_classes"]),
        tta=False,
        batch_size=2,
    )
    with pytest.raises(AttributeError, match="member_preds"):
        _ = oof.member_stack


def test_oof_never_allocates_full_float_stack():
    """Regression for FM-5: no ``(M,N,C,H,W)`` float tensor is ever allocated.

    The 276 GB flair2 OOM was exactly that allocation. We wrap ``torch.zeros``
    and assert nothing 5-D with the member axis is created during ``run_oof``;
    the only large buffers are ``(N,C,H,W)`` softmax_sum and ``(M,N,H,W)`` uint8.
    """
    from torchgeo_bench.label_quality import oof as oof_mod

    m, n, c, h, w, k = 3, 6, 4, 5, 5, 2
    ds = _SegDS(torch.randn(n, 3, h, w), torch.randint(0, c, (n, h, w)))
    members = [{"num_classes": c, "seed": s} for s in range(m)]

    real_zeros = torch.zeros
    five_d_float = []

    def spy_zeros(*args, **kwargs):
        t = real_zeros(*args, **kwargs)
        if t.ndim == 5 and t.dtype.is_floating_point:
            five_d_float.append(tuple(t.shape))
        return t

    with patch.object(oof_mod.torch, "zeros", spy_zeros):
        oof_mod.run_oof(
            ds,
            members,
            lambda seed: np.array([i % k for i in range(n)]),
            member_factory=lambda spec: _RecordingMember(spec["num_classes"]),
            tta=False,
            batch_size=3,
        )

    assert five_d_float == [], f"full float stack allocated: {five_d_float}"


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
    "model",
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
    "degenerate",
    "min_class_coverage",
    "oof_per_class_iou_min",
    "score_iqr",
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
        model="resnet50",
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
    """Per-sample pixel artifacts round-trip under the model-keyed path layout."""
    import os

    from torchgeo_bench.label_quality.store import load_pixel_artifact, save_pixel_artifact

    pixel_scores = np.random.rand(8, 8).astype(np.float32)
    pred_mask = np.random.randint(0, 5, (8, 8)).astype(np.int64)

    path = save_pixel_artifact(
        str(tmp_path), "flair2", "resnet50", "aer", "00042", pixel_scores, pred_mask,
        native_id="scene42", image_score=0.7, rank=3,
    )
    assert path.endswith(
        os.path.join("label_quality", "flair2", "resnet50", "aer", "00042.npz")
    )

    loaded = load_pixel_artifact(str(tmp_path), "flair2", "resnet50", "aer", "00042")
    np.testing.assert_array_equal(loaded["pixel_scores"], pixel_scores)
    np.testing.assert_array_equal(loaded["pred_mask"], pred_mask)
    assert str(loaded["native_id"]) == "scene42"
    assert int(loaded["rank"]) == 3
    assert float(loaded["image_score"]) == 0.7


def test_checkpoint_key_and_exists(tmp_path):
    """The checkpoint path encodes (dataset, model_slug, member_idx, fold_idx, seed)."""
    from torchgeo_bench.label_quality.store import (
        checkpoint_exists,
        save_checkpoint,
    )

    key = {"dataset": "flair2", "model_slug": "resnet50", "member_idx": 1, "fold_idx": 2, "seed": 7}
    assert checkpoint_exists(str(tmp_path), **key) is False

    path = save_checkpoint({"w": torch.zeros(3)}, str(tmp_path), **key)
    for token in ("flair2", "resnet50", "member1", "fold2", "seed7"):
        assert token in path
    assert checkpoint_exists(str(tmp_path), **key) is True


def test_checkpoint_model_slug_isolates_paths(tmp_path):
    """FM-1 regression: two models on the same (dataset,member,fold,seed) never collide.

    Distinct ``model_slug`` values must resolve to distinct checkpoint paths, so
    a resume with ``resume=true`` loads each backbone's own weights and never
    cross-loads the other model's checkpoint.
    """
    from torchgeo_bench.label_quality.store import (
        checkpoint_exists,
        checkpoint_path,
        save_checkpoint,
    )

    key = {"dataset": "cloudsen12", "member_idx": 0, "fold_idx": 0, "seed": 0}
    p_resnet = checkpoint_path(str(tmp_path), model_slug="resnet50", **key)
    p_vit = checkpoint_path(str(tmp_path), model_slug="vit_base_patch16_224", **key)
    assert p_resnet != p_vit

    # Writing the resnet checkpoint must not make the ViT one "exist" on resume.
    save_checkpoint({"w": torch.zeros(3)}, str(tmp_path), model_slug="resnet50", **key)
    assert checkpoint_exists(str(tmp_path), model_slug="resnet50", **key) is True
    assert checkpoint_exists(str(tmp_path), model_slug="vit_base_patch16_224", **key) is False


def test_sanitize_slug_path_safe():
    """A model name with a ``/`` slug becomes a single path-safe segment."""
    from torchgeo_bench.label_quality.store import sanitize_slug

    assert sanitize_slug("timm/resnet50") == "timm-resnet50"
    assert "/" not in sanitize_slug("terratorch/terramind_v1_base")


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
        model="resnet50",
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
    rgb_bands = ["red", "green", "blue"]

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


def _tiny_specs(cfg, bench=None, *, num_classes, device, seeds):
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
    assert set(df["model"]) == {"mock"}  # FM-2: model column populated from cfg.model.name
    # Degeneracy columns come from the run wiring, not just the store layer.
    for column in ("degenerate", "min_class_coverage", "oof_per_class_iou_min", "score_iqr"):
        assert column in df.columns, f"{column} missing from run output"
    assert df["min_class_coverage"].nunique() == 1  # cell-level: same on both methods


# --- Slice 10: per-model hyperparameter overrides (Phase 5) ---------------


class _HparamBench:
    """Bench stand-in exposing just what ``build_member_specs`` reads."""

    num_classes = 4
    rgb_bands = ["red", "green", "blue"]

    def select_band_specs(self, names):
        del names
        return []


def _build_specs(tmp_path, monkeypatch, *, model_seg=None, **lq_overrides):
    """Run ``build_member_specs`` with an optional ``cfg.model.eval.segmentation``."""
    from torchgeo_bench.label_quality import run as lq_run

    cfg = _lq_cfg(tmp_path, **lq_overrides)
    cfg.label_quality.backbone_lr = lq_overrides.get("backbone_lr", 1e-5)
    cfg.label_quality.head_lr = lq_overrides.get("head_lr", 1e-3)
    if model_seg is not None:
        cfg.model.eval = {"segmentation": model_seg}

    # The backbone factory is never called here — only the spec dict is inspected.
    monkeypatch.setattr(lq_run, "instantiate", lambda *a, **k: None)
    return lq_run.build_member_specs(
        cfg, _HparamBench(), num_classes=4, device="cpu", seeds=[0]
    )


def test_member_specs_default_to_global_label_quality_hparams(tmp_path, monkeypatch):
    """Without a per-model override the label_quality globals still apply."""
    spec = _build_specs(tmp_path, monkeypatch)[0]

    assert spec["backbone_lr"] == 1e-5
    assert spec["head_lr"] == 1e-3
    assert spec["max_steps"] == 2  # from _lq_cfg


def test_member_specs_per_model_override_wins(tmp_path, monkeypatch):
    """A per-model eval.segmentation LR overrides the global label_quality value."""
    spec = _build_specs(
        tmp_path,
        monkeypatch,
        model_seg={
            "layers": ["layer1", "layer2"],
            "head_type": "fpn",
            "backbone_lr": 1e-4,
            "head_lr": 3e-3,
            "max_steps": 8000,
        },
    )[0]

    assert spec["backbone_lr"] == 1e-4
    assert spec["head_lr"] == 3e-3
    assert spec["max_steps"] == 8000


def test_member_specs_null_override_falls_back_to_global(tmp_path, monkeypatch):
    """An explicit null override (the config default) must not shadow the global.

    ``conf/config.yaml`` ships these keys as ``null`` so they are discoverable;
    a present-but-null key has to read as "no override".
    """
    spec = _build_specs(
        tmp_path,
        monkeypatch,
        model_seg={
            "layers": ["layer1", "layer2"],
            "head_type": "fpn",
            "backbone_lr": None,
            "head_lr": None,
        },
    )[0]

    assert spec["backbone_lr"] == 1e-5
    assert spec["head_lr"] == 1e-3


def test_finetune_batch_size_does_not_read_frozen_probe_batch_size(tmp_path, monkeypatch):
    """``eval.segmentation.batch_size`` is the frozen probe's, and must be ignored.

    That key already means the cached-feature probe batch size (64 by default).
    Resolving the unfrozen fine-tune's loader batch size from it would ship 64
    into a full fine-tune and OOM; the override key is ``finetune_batch_size``.
    """
    from torchgeo_bench.label_quality.run import _hparam

    lq = OmegaConf.create({"batch_size": 8})
    seg = OmegaConf.create({"batch_size": 64})  # frozen probe value
    assert _hparam(seg, lq, "batch_size", 8, seg_key="finetune_batch_size") == 8

    seg = OmegaConf.create({"batch_size": 64, "finetune_batch_size": 16})
    assert _hparam(seg, lq, "batch_size", 8, seg_key="finetune_batch_size") == 16


def test_oof_loaders_get_configured_num_workers(monkeypatch):
    """``num_workers`` reaches BOTH the train and held-out loaders.

    These datasets decode 512-650px tiles per sample, so a serial loader leaves
    the GPU idle; the workers are what make the fixed step budget affordable.
    """
    from torchgeo_bench.label_quality import oof as lq_oof

    seen: list[int] = []
    real_loader = lq_oof._subset_loader

    def spy(dataset, indices, batch_size, shuffle, num_workers=0):
        seen.append(num_workers)
        # Actually build with 0 workers so the test stays in-process.
        return real_loader(dataset, indices, batch_size, shuffle, 0)

    monkeypatch.setattr(lq_oof, "_subset_loader", spy)

    ds = _synthetic_seg_dataset(n=8)
    specs = _tiny_specs(None, None, num_classes=4, device="cpu", seeds=[0])
    lq_oof.run_oof(
        ds, specs, lambda seed: np.arange(len(ds)) % 2,
        batch_size=4, num_workers=6, tta=False,
    )

    assert seen, "loaders must have been built"
    assert set(seen) == {6}, f"every loader must get num_workers=6, got {sorted(set(seen))}"


def test_oof_loader_defaults_to_serial(monkeypatch):
    """Omitting ``num_workers`` keeps the historical serial loader."""
    from torchgeo_bench.label_quality import oof as lq_oof

    loader = lq_oof._subset_loader(_synthetic_seg_dataset(n=4), [0, 1], 2, shuffle=False)
    assert loader.num_workers == 0
    assert loader.persistent_workers is False


def test_oof_records_no_curves_without_eval_every(tmp_path, monkeypatch):
    """Default launch behavior is unchanged: no evaluation, no curves, no file."""
    import os

    from torchgeo_bench.label_quality import run as lq_run

    ds = _synthetic_seg_dataset()
    monkeypatch.setattr(lq_run, "_load_bench_dataset", lambda cfg, name: _SynthSegBench(ds))
    monkeypatch.setattr(lq_run, "build_member_specs", _tiny_specs)

    cfg = _lq_cfg(tmp_path)
    lq_run.run_label_quality(cfg)

    assert not os.path.exists(
        tmp_path / "label_quality" / "synthetic" / "mock" / "training_curves.json"
    )


def test_oof_records_member_curves_with_eval_every(tmp_path, monkeypatch):
    """``eval_every`` yields one curve per (member, fold) with the D8 fields."""
    import json
    import os

    from torchgeo_bench.label_quality import run as lq_run

    ds = _synthetic_seg_dataset()
    monkeypatch.setattr(lq_run, "_load_bench_dataset", lambda cfg, name: _SynthSegBench(ds))

    def _specs_with_eval(cfg, bench=None, *, num_classes, device, seeds):
        specs = _tiny_specs(cfg, bench, num_classes=num_classes, device=device, seeds=seeds)
        for s in specs:
            s["eval_every"] = 1
        return specs

    monkeypatch.setattr(lq_run, "build_member_specs", _specs_with_eval)

    cfg = _lq_cfg(tmp_path)
    cfg.label_quality.eval_every = 1
    lq_run.run_label_quality(cfg)

    path = tmp_path / "label_quality" / "synthetic" / "mock" / "training_curves.json"
    assert os.path.exists(path)
    payload = json.loads(path.read_text())

    # k=2 folds x n_members=2 -> 4 trainings, each with a recorded curve.
    assert len(payload["curves"]) == cfg.label_quality.k * cfg.label_quality.n_members
    for curve in payload["curves"]:
        assert curve["n_train"] > 0 and curve["n_holdout"] > 0
        assert curve["history"], "each training must record at least one point"
        point = curve["history"][-1]
        # D8: train loss + held-out mIoU + per-class IoU.
        assert {"step", "train_loss", "val_mIoU", "per_class_IoU"} <= set(point)
        assert len(point["per_class_IoU"]) == _SynthSegBench.num_classes
    # The hparams the curves were produced under travel with them.
    assert payload["eval_every"] == 1
    assert payload["k"] == cfg.label_quality.k


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


def test_resume_skip_is_per_model(tmp_path, monkeypatch):
    """A second backbone still writes its rows when the first already wrote that dataset.

    Regression: the resume guard keyed on ``(dataset, method)`` alone, so in a
    sharded sweep whose nodes share one CSV, the backbone that finished first
    suppressed every other backbone's rows -- silently discarding a completed
    OOF run.
    """
    import pandas as pd

    from torchgeo_bench.label_quality import run as lq_run

    ds = _synthetic_seg_dataset()
    monkeypatch.setattr(lq_run, "_load_bench_dataset", lambda cfg, name: _SynthSegBench(ds))
    monkeypatch.setattr(lq_run, "build_member_specs", _tiny_specs)

    out = str(tmp_path / "shared.csv")

    cfg_a = _lq_cfg(tmp_path, output=out)
    cfg_a.model.name = "resnet50"
    lq_run.run_label_quality(cfg_a)

    # Same dataset + methods, different backbone, resume on -- as in a sweep.
    cfg_b = _lq_cfg(tmp_path, output=out)
    cfg_b.model.name = "convnext_base"
    cfg_b.resume = True
    lq_run.run_label_quality(cfg_b)

    df = pd.read_csv(out)
    assert set(df["model"]) == {"resnet50", "convnext_base"}
    for model in ("resnet50", "convnext_base"):
        methods = set(df[df["model"] == model]["method"])
        assert methods == {"cleanlab", "aer"}, f"{model} missing methods: {methods}"

    # And the same backbone twice is still a no-op (real resume still skips).
    n_before = len(df)
    lq_run.run_label_quality(cfg_b)
    assert len(pd.read_csv(out)) == n_before


# --- Slice 11: degeneracy detection ----------------------------------------


def _collapsed_cell(n=40, h=32, w=32, m=5, fg_rate=0.05, seed=0):
    """Rare-foreground labels (spacenet7-like) with every member predicting background."""
    rng = np.random.default_rng(seed)
    labels = torch.from_numpy((rng.random((n, h, w)) < fg_rate).astype(np.int64))
    preds = torch.zeros(m, n, h, w, dtype=torch.uint8)
    return labels, preds


def test_min_class_coverage_flags_collapsed_cell():
    """A majority-class collapse is flagged -- and provably clears the old mIoU gate."""
    from torchgeo_bench.label_quality.aer_score import score
    from torchgeo_bench.label_quality.degeneracy import cell_metrics, is_degenerate
    from torchgeo_bench.label_quality.store import LOW_CAPACITY_THRESHOLD

    labels, preds = _collapsed_cell()

    metrics = cell_metrics(labels, preds, num_classes=2)
    assert metrics["min_class_coverage"] == pytest.approx(0.0, abs=1e-9)
    assert is_degenerate(metrics["min_class_coverage"], score_iqr=0.5)

    # Why this gate exists: the pre-existing scalar guard passes this same cell.
    # AER's macro-IoU is per image over classes present in that image's label, so
    # a background-only predictor scores ~1 on background and ~0 on foreground.
    image_scores, _ = score(labels, preds, [0, 1])
    assert 1.0 - image_scores.mean() > LOW_CAPACITY_THRESHOLD


def test_min_class_coverage_healthy_cell_not_flagged():
    """Class imbalance alone does not trip the gate: coverage is GT-relative."""
    from torchgeo_bench.label_quality.degeneracy import cell_metrics, is_degenerate

    rng = np.random.default_rng(1)
    n, h, w, m = 40, 32, 32, 5
    # ~5% foreground, i.e. the same imbalance a raw majority-fraction test would
    # false-positive on; members mostly agree with the label.
    labels_np = (rng.random((n, h, w)) < 0.05).astype(np.int64)
    preds_np = np.stack(
        [np.where(rng.random((n, h, w)) < 0.9, labels_np, 1 - labels_np) for _ in range(m)]
    )
    labels = torch.from_numpy(labels_np)
    preds = torch.from_numpy(preds_np.astype(np.uint8))

    metrics = cell_metrics(labels, preds, num_classes=2)
    assert metrics["min_class_coverage"] > 0.5
    assert not is_degenerate(metrics["min_class_coverage"], score_iqr=0.5)


def test_score_iqr_flags_near_constant_scores():
    """The IQR arm fires alone: healthy coverage, but a ranking with no spread."""
    from torchgeo_bench.label_quality.degeneracy import is_degenerate, score_iqr

    rng = np.random.default_rng(2)
    flat = 0.5 + rng.normal(0.0, 1e-4, 4000)
    spread = rng.uniform(0.0, 1.0, 4000)

    assert score_iqr(flat) < 0.01
    assert score_iqr(spread) > 0.1
    # Coverage is healthy in both cases, so only the IQR arm can differ.
    assert is_degenerate(0.9, score_iqr(flat))
    assert not is_degenerate(0.9, score_iqr(spread))


def test_degeneracy_respects_ignore_index():
    """A class living only under ``ignore_index`` never enters the present set."""
    from torchgeo_bench.label_quality.degeneracy import class_mass_coverage

    labels = torch.zeros(3, 10, 10, dtype=torch.long)
    labels[:, 0, 0] = 1  # class 1 ...
    labels[:, 0:3, 0:3] = 255  # ... is then fully covered by the ignore block
    preds = torch.zeros(2, 3, 10, 10, dtype=torch.uint8)  # background only

    min_coverage, coverage = class_mass_coverage(labels, preds, num_classes=2)

    # Only class 0 is present, and the members predict it perfectly -> healthy.
    assert coverage.shape[1] == 1
    assert min_coverage == pytest.approx(1.0)


def test_global_per_class_iou_min_below_macro_miou():
    """Global per-class IoU exposes the collapse that AER's per-image macro hides."""
    from torchgeo_bench.label_quality.aer_score import score
    from torchgeo_bench.label_quality.degeneracy import global_per_class_iou

    labels, preds = _collapsed_cell()

    min_iou, per_class = global_per_class_iou(labels, preds, num_classes=2)
    image_scores, _ = score(labels, preds, [0, 1])

    assert min_iou == pytest.approx(0.0, abs=1e-9)  # foreground IoU is zero
    assert per_class.max() > 0.9  # ... masked by a near-perfect background IoU
    assert 1.0 - image_scores.mean() > 0.3  # which is why the macro mean clears 0.3


def test_degeneracy_columns_written_per_method(tmp_path):
    """Cell columns are constant across methods; ``score_iqr``/``degenerate`` are not."""
    import pandas as pd

    from torchgeo_bench.label_quality.store import write_results_csv

    n = 20
    path = tmp_path / "results.csv"
    shared = {
        "dataset": "spacenet7",
        "model": "dinov3sat",
        "member_set": "M5",
        "n_flagged_pixels": np.arange(n),
        "grouping_tier": "latlon_block",
        "folds": np.arange(n) % 5,
        "native_ids": None,
        "k": 5,
        "n_members": 5,
        "seed": 0,
        "bands": "rgb",
        "partition": "default",
        "low_capacity": False,
        "min_class_coverage": 0.0,  # cell-level: same on both methods
        "oof_per_class_iou_min": 0.0,
    }
    write_results_csv(
        str(path), method="cleanlab", image_scores=np.linspace(0.0, 1.0, n),
        score_iqr=0.4, degenerate=False, **shared,
    )
    write_results_csv(
        str(path), method="aer", image_scores=np.full(n, 0.5),
        score_iqr=0.0, degenerate=True, **shared,
    )

    df = pd.read_csv(str(path))
    for column in ("min_class_coverage", "oof_per_class_iou_min"):
        assert df[column].nunique() == 1, f"{column} must be cell-level"
    assert set(df[df.method == "cleanlab"]["degenerate"]) == {False}
    assert set(df[df.method == "aer"]["degenerate"]) == {True}
    assert df[df.method == "aer"]["score_iqr"].iloc[0] == pytest.approx(0.0)
