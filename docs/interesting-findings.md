# Interesting findings

Running notes on things we've noticed during torchgeo-bench sweeps that
might be worth a paragraph in a paper or a future ablation. One section
per finding; nothing on this page is a benchmark result — those live in
`results/` and the explorer.

## Prithvi / Clay CLS-token embeddings collapse on visually redundant patches (2026-05-16)

While running intrinsic-dimension probes on `eurosat-spatial`, TwoNN returned
`nan` for 12 of 34 (model, bands) combos. All 12 were Prithvi-EO (v1/v2 100,
v2 300, v2 300_tl, v2 600) or Clay-v1.5; the other backbones (Terramind,
CROMA, DOFA, ScaleMAE, the timm baselines) were fine.

Diagnostic dump from a representative failing run:

```
[intrinsic-dim] TwoNN nan on X(10000, 768) —
  d1[min=0 median=4.6e-1 zeros=348]
  d2[min=0 median=5.2e-1 zeros=156]
  mu[min=0 max=2.9e+37 zeros=156]
  X[norm=7.8–9.2 std=0.31]
```

348 of 10,000 rows had a nearest-neighbour distance of exactly zero in
fp32. The rows weren't bit-equal — `np.unique(axis=0)` left them in —
only the *squared distance* `‖a‖² + ‖b‖² − 2·a·b` underflowed.

Why these specific backbones:

- Both Prithvi (MAE-style pretraining) and Clay (self-supervised MAE
  variant) end the feature extractor with `LayerNorm + [CLS]` pooling.
  LayerNorm forces every output to zero mean / unit variance per-sample,
  collapsing the 768-D space onto a hypersphere — embeddings vary in
  direction, not magnitude (norms 7.8–9.2 across 10k samples).
- EuroSAT has lots of visually redundant patches (forest, water,
  cropland at the same Sentinel-2 capture time). A frozen MAE encoder
  pretrained on Earth observation has been *taught* to abstract these
  into the same compact cluster.
- fp32 precision wall: at `‖a‖² ≈ 7.7`, the squared-distance dot product
  loses ~7 digits; embeddings with true L2 distance below ~1e-3 round to
  squared-distance zero.
- Terramind doesn't show the failure because its multi-modal generative
  pretraining rewards distinguishing instances cross-modally rather than
  abstracting them.

The mathematical degeneracy aside, this is a substantive observation
about Prithvi/Clay's feature geometry on satellite imagery: a non-trivial
fraction of EuroSAT patches are *literally indistinguishable* under their
CLS-token representation, in fp32 and likely in semantic content too.

Code workaround in `src/torchgeo_bench/intrinsic_dim.py`: filter rows
where `d1 == 0` before TwoNN. Fixes all 12 nan cases on rerun.

## Open ablations / TODOs

### CLS-token vs mean-pool ablation
The finding above is specifically about CLS-token pooling. Different
pretraining objectives may make CLS more or less informative:

- **MAE-style (Prithvi, Clay):** trained to reconstruct masked patches.
  The CLS token isn't directly supervised; it picks up whatever signal
  the encoder happens to route through it. Mean-pooling the patch
  tokens might preserve more information.
- **DINOv3-style:** explicit CLS supervision (self-distillation
  objective); CLS is the canonical feature.
- **CLIP-style:** pooled patch features are the canonical output.

Today the wrappers in `src/torchgeo_bench/models/` mostly hard-code
their pooling choice. Worth adding a `pooling: cls|mean|both` config
knob and re-running the GeoFM sweep with each — could meaningfully
change Prithvi / Clay numbers and the resulting ID estimates.
