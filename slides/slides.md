---
theme: default
layout: cover
title: torchgeo-bench — Sprint Recap
highlighter: shiki
lineNumbers: true
fonts:
  sans: Inter
  serif: Source Serif 4
  mono: Fira Code
transition: fade
controls: false
progress: false
---

# torchgeo-bench
## Two-Week Sprint Recap

<div class="rule"></div>

<span style="font-family:'Inter',sans-serif; font-size:0.9em; color:var(--ft-muted)">
~50 PRs · GPU evaluation · OlmoEarth v1.1 · calibration · efficiency profiling · leaderboards
</span>

<br>

<span style="font-family:'Inter',sans-serif; font-size:0.8em; color:var(--ft-muted)">June 2026</span>

---

# What shipped

<div class="rule"></div>

<div style="display:grid; grid-template-columns:1fr 1fr; gap:1rem; font-family:'Inter',sans-serif; font-size:0.78em; margin-top:0.3rem;">
<div>

<p><span class="tag">GPU</span> <strong>GPU KNN via faissknn</strong><br>
<span class="muted">FAISS-backed KNN with optional CUDA path. Auto-fallback to CPU.</span></p>

<p><span class="tag tag-claret">CLI</span> <strong>Typer + Rich</strong><br>
<span class="muted">Replaced argparse + tqdm. Beautiful progress bars, rich tracebacks.</span></p>

<p><span class="tag tag-oxford">PROFILE</span> <strong>Efficiency Profiling</strong><br>
<span class="muted">Throughput, GFLOPs, peak GPU mem, latency, params. Pareto front: accuracy vs throughput.</span></p>

<p><span class="tag tag-wheat">MODELS</span> <strong>OlmoEarth v1 + v1.1</strong><br>
<span class="muted">nano→large, plus v1.1 linear-embed family. DINOv3-SAT ViT-L web-pretrained.</span></p>

<p><span class="tag tag-oxford">CALIB</span> <strong>Calibration Metrics</strong><br>
<span class="muted">ECE · RMS-CE · MCE + temperature scaling on every probe.</span></p>

</div>
<div>

<p><span class="tag tag-claret">DATA</span> <strong>EuroSAT Spatial Split</strong><br>
<span class="muted">Geographically disjoint train/test. Harder than random split.</span></p>

<p><span class="tag tag-oxford">QUALITY</span> <strong>Cleanlab Label Audit</strong><br>
<span class="muted">Label-quality scores across all GeoBench V1+V2 datasets.</span></p>

<p><span class="tag">POOL</span> <strong>CLS + Mean Pool Ablations</strong><br>
<span class="muted">TerraTorch/ScaleMAE/Clay — pool=cls|mean|both sweep.</span></p>

<p><span class="tag tag-wheat">FIX</span> <strong>Silent-bug Sweep</strong><br>
<span class="muted">Removed try/except covers; fixed minmax_zscore, fp16 overflow, label gaps.</span></p>

<p><span class="tag tag-wheat">SEG</span> <strong>Patch-Linear Head</strong><br>
<span class="muted">ViT-native segmentation decoder. Segmentation probing now first-class.</span></p>

<p><span class="tag tag-wheat">MULTI</span> <strong>SAR + Landsat Modalities</strong><br>
<span class="muted">OlmoEarth mixed-sensor support, auto input-resolution.</span></p>

</div>
</div>

---

# GPU KNN — faissknn
<span class="tag">PR #53</span> <span class="tag">PR #55</span> <span class="tag">PR #89</span> <span class="tag">PR #94</span> <span class="tag">PR #101</span>

<div class="rule"></div>

<div style="display:grid; grid-template-columns:1fr 1fr; gap:1.5rem; align-items:start;">
<div>

CPU path:

```python
clf = KNNClassifier(n_neighbors=5, device="cpu")
clf.fit(x_train, y_train)
preds = clf.predict(x_test)
```

GPU path (opt-in via `pip install -e ".[cuda]"`):

```python
clf = KNNClassifier(
    n_neighbors=5,
    device="cuda:0",
    metric="cosine",  # l2 | ip | cosine
)
clf.fit(x_train, y_train)
preds  = clf.predict(x_test)
proba  = clf.predict_proba(x_test)
```


</div>
<div style="font-family:'Inter',sans-serif; font-size:0.8em;">

**Key fixes:**

- `n_classes = max(y)+1` not `len(unique(y))` — avoids `IndexError` on partitions with missing class labels
- `use_fp16=False` in evaluation — raw sensor DN values (~10 000) overflow fp16 L2 distances → random KNN
- `faiss-cpu` is now the **default**; `faiss-cuda-cu128` moved to `[cuda]` extra — fixes install on macOS + non-manylinux Linux (`#120`)

<br>

**Config knobs** (`#94`, `#99`):

```yaml
eval:
  knn_k: 5          # neighbours
  knn_device: null  # null → inherit cfg.device
                    # "cpu" forces faiss-cpu KNN
```

</div>
</div>

---

# Typer + Rich CLI
<span class="tag tag-claret">PR #88</span>

<div class="rule"></div>

<div style="font-family:'Inter',sans-serif; font-size:0.88em; margin-top:0.5rem;">

Replaced `argparse` + `tqdm` with **Typer** + **Rich** across the whole CLI.

</div>

<div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:1.5rem; margin-top:1.2rem; font-family:'Inter',sans-serif; font-size:0.85em;">
<div>

**Typed commands**

`torchgeo-bench run`, `torchgeo-bench download` — auto-generated `--help`, type-checked args, Hydra passthrough for config overrides.

</div>
<div>

**Rich progress bars**

Live extraction progress with ETA and throughput instead of plain tqdm spinners.

</div>
<div>

**Rich tables + tracebacks**

Dataloader tuning results rendered as formatted tables. Full color tracebacks with local variable context on errors.

</div>
</div>

---

# Efficiency Profiling
<span class="tag tag-oxford">PR #60</span> <span class="tag tag-oxford">PR #67</span>

<div class="rule"></div>

<div style="display:grid; grid-template-columns:1fr 1fr; gap:1.5rem;">
<div>

Metrics recorded per model run:

```python
{
  # GPU
  "throughput_samples_per_sec": 1420.3,
  "latency_ms_per_batch_p50":   18.0,
  "peak_gpu_mem_gb":            3.2,
  "reserved_gpu_mem_gb":        4.1,
  "gflops":                     61.6,
  "params_m":                   307.4,
}
```

</div>
<div>

**GFLOPs via FlopCounterMode:**

```python
with torch.profiler.FlopCounterMode() as fc:
    model(sample)
gflops = fc.get_total_flops() / 1e9
```

**Throughput:**

```python
# warm-up, then timed forward passes
sps = n_samples / elapsed_s
```

Explorer shows Pareto front: accuracy vs throughput.

</div>
</div>

---

# OlmoEarth Integration
<span class="tag tag-wheat">PR #84</span> <span class="tag tag-wheat">PR #85</span> <span class="tag tag-wheat">PR #93</span>

<div class="rule"></div>

<div style="display:grid; grid-template-columns:1fr 1fr; gap:1.5rem;">
<div>

Config (nano → large):

```yaml
# conf/model/olmoearth_v1_base.yaml
_target_: torchgeo_bench.models.OlmoEarthBenchModel
name: olmoearth_v1_base
variant: base
normalization: identity  # model handles its own
```

Auto-rescale to S2 DN:

```python
class OlmoEarthBenchModel(BenchModel):
    expected_input_unit = InputUnit.S2_DN

    def _forward_patch_features(
        self, images: Tensor, **_
    ) -> Tensor:
        # model_native normalizer already
        # rescaled inputs to S2 DN range
        return self.backbone(images)
```

</div>
<div style="font-family:'Inter',sans-serif; font-size:0.82em;">

**m-eurosat Linear Accuracy:**

| | Model | Score |
|--|-------|-------|
| <span class="gold">1</span> | OlmoEarth Large | **0.976** |
| <span class="silver">2</span> | OlmoEarth Base | **0.975** |
| 3 | DOFA Large | 0.973 |
| 4 | OlmoEarth v1.1 Base | 0.970 |
| 5 | DINOv3-SAT ViT-L | 0.969 |

<br>

**Dominates S2 datasets** — EuroSAT, BigEarthNet, So2Sat. Nano/Tiny competitive at a fraction of the size.

`normalization=identity` bypasses z-score — OlmoEarth handles its own preprocessing.

</div>
</div>

---

# OlmoEarth v1.1
<span class="tag tag-wheat">PR #99</span>

<div class="rule"></div>

<div style="display:grid; grid-template-columns:0.95fr 1.05fr; gap:1.5rem; align-items:start;">
<div style="font-family:'Inter',sans-serif; font-size:0.82em;">

**What changed (v1 → v1.1):**

- **Linear** patch embedding (vs. convolutional)
- Single bandset per modality
- Updated masking + loss functions

<p style="margin-top:0.6rem;"><span class="tag tag-claret">≈ 3× fewer MACs</span> at comparable accuracy.</p>

```yaml
# conf/model/olmoearth_v1_1_base.yaml
name: olmoearth_v1_1_base
version: v1.1     # selects weight family
```

<span class="muted">30/30 sweep tasks · 0 failures · Nano/Tiny/Base.</span>

</div>
<div style="font-family:'Inter',sans-serif; font-size:0.78em;">

**Linear probe — v1 vs v1.1**

| Dataset | Metric | v1 | v1.1 |
|---------|:------:|:--:|:----:|
| m-eurosat (Base) | Acc | .975 | .970 |
| m-so2sat (Base) | Acc | .720 | **.728** |
| m-so2sat (Tiny) | Acc | .656 | **.693** |
| m-bigearthnet (Tiny) | mAP | .691 | **.717** |
| benv2 (Tiny) | mAP | .817 | **.826** |
| treesatai (Base) | mAP | .647 | .645 |

<p style="margin-top:0.4rem;"><strong>Smaller variants gain most</strong> — Tiny/Nano jump on So2Sat + BigEarthNet (KNN So2Sat Tiny .506 → <strong>.567</strong>). Slight EuroSAT dip. Net: same accuracy at a third of the compute.</p>

</div>
</div>

---

# CLS Token + Pool Ablations
<span class="tag">PR #73</span> <span class="tag">PR #74</span> <span class="tag">PR #75</span> <span class="tag">PR #76</span>

<div class="rule"></div>

<div style="display:grid; grid-template-columns:1fr 1fr; gap:2rem; align-items:start;">
<div style="font-family:'Inter',sans-serif; font-size:0.85em;">

Added `pool=cls|mean|both` sweep across TerraTorch/ScaleMAE/Clay wrappers.

**Finding:** token choice matters dataset-to-dataset.

- **CLS** wins on ForestNet, Brick Kiln — tasks with strong global structure
- **Patch-mean** wins on EuroSAT, BigEarthNet — spatially distributed labels
- **Terramind** has no CLS token — `_cls` configs dropped (`#76`)

</div>
<div style="font-family:'Inter',sans-serif; font-size:0.82em;">

**m-brick-kiln Linear Accuracy:**

| | Model | Score |
|--|-------|-------|
| <span class="gold">1</span> | DINOv3-SAT ViT-L | **0.976** |
| <span class="silver">2</span> | Clay v1.5 Base+CLS | **0.975** |
| 3 | DOFA Base | 0.974 |
| 4 | DOFA Large | 0.974 |
| 5 | Clay v1.5 Base | 0.972 |

</div>
</div>

---

# Patch-Linear Head
<span class="tag tag-wheat">PR #124</span>

<div class="rule"></div>

<div style="display:grid; grid-template-columns:1fr 1fr; gap:1.5rem; align-items:start;">
<div>

New segmentation decoder wired into `SegmentationProbe`:

```python
probe = SegmentationProbe(
    model=backbone,
    head_type="patch_linear",  # new
)
```

**`PatchLinearHead`** — lightweight ViT decoder. Projects each patch token directly to pixel logits:

```
ChannelLayerNorm
→ Conv2d(D, C × P², 1)
→ pixel_shuffle(P)
→ bilinear resize (if needed)
```

No skip connections, no upsampling pyramid — purely linear.

</div>
<div style="font-family:'Inter',sans-serif; font-size:0.82em;">

**Why it matters:**

<p><span class="tag">SIMPLE</span> No skip connections, no upsampling pyramid. Purely linear — interpretable and fast.</p>

<p><span class="tag tag-claret">ViT-NATIVE</span> Designed for ViTs where spatial tokens are the primary representation. Complements the existing <code>LinearHead</code> (classification) decoder.</p>

<p><span class="tag tag-oxford">FLEXIBLE</span> Handles arbitrary output sizes via bilinear resize fallback — works across datasets regardless of patch/image size mismatch.</p>

<br>

Previously only `head_type="linear"` (classification) was available. Segmentation probing is now a first-class evaluation path.

</div>
</div>

---

# EuroSAT Spatial Split + Cleanlab
<span class="tag tag-claret">PR #50</span> <span class="tag tag-claret">PR #52</span>

<div class="rule"></div>

<div style="display:grid; grid-template-columns:1fr 1fr; gap:1.5rem;">
<div>

**Geographically disjoint train/test:**

```python
# Standard: random split
# → models exploit spatial autocorrelation

# Spatial: tiles from different regions
# → true generalization test
ds = EuroSATSpatial(
    root=DATA_ROOT,
    split="test",
    download=True,
)
```

OlmoEarth Large KNN: **0.959** vs 0.956 random split<br>
OlmoEarth Base Linear: **0.978** vs 0.975 random split

</div>
<div>

**Cleanlab label audit:**

```python
from cleanlab.filter import find_label_issues

issues = find_label_issues(
    labels=y_train,
    pred_probs=pred_proba,
    return_indices_ranked_by=
        "self_confidence",
)
# → flags likely mislabeled samples
```

Applied across all GeoBench V1+V2. Results in `results/cleanlab/`.

Surfaces annotation noise in BigEarthNet, ForestNet, TreeSatAI — useful for reweighting or curriculum training.

</div>
</div>

---

# Leaderboard — EuroSAT
<span class="tag tag-oxford">Results</span>

<div class="rule"></div>

<div style="display:grid; grid-template-columns:1fr 1fr; gap:2rem;">
<div>

**m-eurosat** (Accuracy)

| | Model | KNN | Lin |
|--|-------|:---:|:---:|
| <span class="gold">1</span> | OlmoEarth Large | .956 | .976 |
| <span class="silver">2</span> | OlmoEarth Base | .946 | .975 |
| <span class="bronze">3</span> | Panopticon | .948 | .968 |
| 4 | DOFA Large | .936 | .973 |
| 5 | OlmoEarth v1.1 Base | .930 | .970 |

</div>
<div>

**eurosat-spatial** (Accuracy)

| | Model | KNN | Lin |
|--|-------|:---:|:---:|
| <span class="gold">1</span> | OlmoEarth Large | .959 | .977 |
| <span class="silver">2</span> | OlmoEarth Base | .941 | .978 |
| <span class="bronze">3</span> | Panopticon | .930 | .962 |
| 4 | OlmoEarth Tiny | .928 | .963 |
| 5 | ResNet50-RGB MoCo | .933 | .958 |

</div>
</div>

---

# Leaderboard — BigEarthNet
<span class="tag tag-oxford">Results</span>

<div class="rule"></div>

<div style="display:grid; grid-template-columns:1fr 1fr; gap:2rem;">
<div>

**m-bigearthnet** (micro-mAP)

| | Model | KNN | Lin |
|--|-------|:---:|:---:|
| <span class="gold">1</span> | OlmoEarth v1.1 Base | .664 | .771 |
| <span class="silver">2</span> | OlmoEarth Large | .664 | .764 |
| <span class="bronze">3</span> | OlmoEarth Base | .658 | .769 |
| 4 | Panopticon | .652 | .735 |
| 5 | Terramind Large | .628 | .750 |

</div>
<div>

**benv2** (micro-mAP)

| | Model | KNN | Lin |
|--|-------|:---:|:---:|
| <span class="gold">1</span> | OlmoEarth Base | .735 | .853 |
| <span class="silver">2</span> | OlmoEarth v1.1 Base | .734 | .852 |
| <span class="bronze">3</span> | OlmoEarth Large | .728 | .850 |
| 4 | Terramind Large | .712 | .846 |
| 5 | OlmoEarth v1.1 Tiny | .725 | .826 |

</div>
</div>

---

# Leaderboard — ForestNet + So2Sat
<span class="tag tag-oxford">Results</span>

<div class="rule"></div>

<div style="display:grid; grid-template-columns:1fr 1fr; gap:2rem;">
<div>

**m-forestnet** (Accuracy)

| | Model | KNN | Lin |
|--|-------|:---:|:---:|
| <span class="gold">1</span> | DINOv3-SAT ViT-L | .425 | .582 |
| <span class="silver">2</span> | Panopticon | .427 | .550 |
| <span class="bronze">3</span> | ScaleMAE Large+CLS | .403 | .569 |
| 4 | Clay v1.5 Base+CLS | .414 | .556 |
| 5 | Clay v1.5 Base | .395 | .551 |

<span class="muted" style="font-size:0.8em;">Hard dataset — all models below .60</span>

</div>
<div>

**m-so2sat** (Accuracy)

| | Model | KNN | Lin |
|--|-------|:---:|:---:|
| <span class="gold">1</span> | OlmoEarth v1.1 Base | .606 | .728 |
| <span class="silver">2</span> | OlmoEarth Base | .577 | .720 |
| <span class="bronze">3</span> | OlmoEarth v1.1 Tiny | .567 | .693 |
| 4 | OlmoEarth Large | .562 | .689 |
| 5 | Panopticon | .532 | .693 |

<span class="muted" style="font-size:0.8em;">Linear leader: Terramind Base .739 — biggest KNN/Lin gap in the set</span>

</div>
</div>

---

# Leaderboard — Brick Kiln + PV4GER + TreeSatAI
<span class="tag tag-oxford">Results</span>

<div class="rule"></div>

<div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:1.2rem; font-size:0.88em;">
<div>

**m-brick-kiln** (Acc)

| | Model | KNN | Lin |
|--|-------|:---:|:---:|
| <span class="gold">1</span> | DOFA Large | .969 | .974 |
| <span class="silver">2</span> | DOFA Base | .965 | .974 |
| <span class="bronze">3</span> | Clay Base | .964 | .972 |
| 4 | Clay Base+CLS | .960 | .975 |
| 5 | OlmoEarth Base | .945 | .970 |

</div>
<div>

**m-pv4ger** (Acc)

| | Model | KNN | Lin |
|--|-------|:---:|:---:|
| <span class="gold">1</span> | DINOv3 ViT-L | .964 | .974 |
| <span class="silver">2</span> | DOFA Large | .967 | .969 |
| <span class="bronze">3</span> | DOFA Base | .965 | .966 |
| 4 | ScaleMAE+CLS | .956 | .970 |
| 5 | Panopticon | .958 | .968 |

</div>
<div>

**treesatai** (mAP)

| | Model | KNN | Lin |
|--|-------|:---:|:---:|
| <span class="gold">1</span> | DINOv3-SAT ViT-L | .477 | .682 |
| <span class="silver">2</span> | Clay Base+CLS | .475 | .671 |
| <span class="bronze">3</span> | DINOv3 ViT-L | .473 | .664 |
| 4 | DOFA Large | .474 | .647 |
| 5 | OlmoEarth Base | .469 | .647 |

</div>
</div>

---

# Efficiency — Throughput vs Accuracy
<span class="tag tag-oxford">PR #60–#80</span>

<div class="rule"></div>

<div style="display:grid; grid-template-columns:1fr 1fr; gap:2rem;">
<div>

**GPU throughput (img/s) — m-eurosat**

| Model | img/s | GFLOPs | Acc* |
|-------|------:|-------:|----:|
| ResNet-50 MoCo | 3 361 | 9 | .75 |
| DOFA Base | 1 807 | 37 | .76 |
| OlmoEarth Nano | 789 | 2 | .72 |
| OlmoEarth Tiny | 780 | 9 | .75 |
| DOFA Large | 594 | 125 | .78 |
| Terramind Large | 456 | 123 | .79 |
| DINOv3-SAT ViT-L | 352 | 166 | .78 |
| OlmoEarth Large | 151 | 381 | .76 |

<span class="muted" style="font-size:0.75em;">*Acc = mean linear accuracy across 11 datasets · throughput &amp; GFLOPs on m-eurosat</span>

</div>
<div style="font-family:'Inter',sans-serif; font-size:0.82em;">

**Pareto surprises:**

<p><span class="tag tag-claret">BEST VALUE</span> <strong>ResNet-50 MoCo</strong> — 3 361 img/s, 9 GFLOPs, 24M params, <strong>.75 mean acc</strong>. Within a few points of the ViT-scale FMs (.76–.79) at 5–10× the throughput.</p>

<p><span class="tag">EFFICIENT</span> <strong>OlmoEarth Nano</strong> — 3.6M params, 0.6 GB peak VRAM, 1.6 GFLOPs, <strong>.72 mean acc</strong> — within a few points of models 30× its size.</p>

<p><span class="tag">EFFICIENT</span> <strong>OlmoEarth Tiny</strong> (14M, .75) lands ~3 pts behind DOFA Large (337M) and DINOv3-SAT (308M) at higher throughput.</p>

**Worst value:** Prithvi 100M — ~1 750 img/s, 35 GFLOPs but <strong>.73 mean acc</strong>, trailing DOFA Base (.76) at the same speed tier.

</div>
</div>

---

# Intrinsic Dimensionality Analysis
<span class="tag tag-oxford">method: intrinsic_dim</span>

<div class="rule"></div>

<div style="display:grid; grid-template-columns:1fr 1fr; gap:2rem; align-items:start;">
<div style="font-size:0.78em;">

**id_TwoNN avg · norm. acc = rank-normalized within each dataset**

| Model | id_TwoNN | norm. acc |
|-------|:--------:|:---------:|
| EarthLoc ResNet-50 | **44** | 17% |
| Prithvi v2 100M CLS | 18 | 31% |
| DOFA Large | 17 | 85% |
| DINOv3 ViT-L | 17 | 78% |
| DINOv3-SAT ViT-L | 16 | 82% |
| DOFA Base | 15 | 71% |
| OlmoEarth Large | 14 | 72% |

</div>
<div style="font-family:'Inter',sans-serif; font-size:0.82em;">

**Pattern:** mid-id models (15–17: DOFA, DINOv3) pair high intrinsic dim with strong accuracy (78–85%) — but high id alone doesn't guarantee it.

<p style="margin-top:0.6rem;"><span class="tag tag-claret">TASK GAP</span> <strong>EarthLoc ResNet-50</strong> — id = 44 but only 17% norm. acc. Geo-localization training, not classification — high dimensionality without discriminative structure.</p>

<p><span class="tag">OUTLIER</span> <strong>Prithvi v2 100M (CLS)</strong> — 2nd-highest id (18) yet 31% norm. acc. Like EarthLoc, dimensionality ≠ separability.</p>

<p><span class="tag tag-wheat">FLAT</span> <strong>Prithvi (patch-mean) cluster</strong> — id ≈ 7, geometrically flat despite 768–1280-d features.</p>

</div>
</div>

---

# Calibration Metrics
<span class="tag tag-oxford">PR #105</span>

<div class="rule"></div>

<div style="display:grid; grid-template-columns:1fr 1fr; gap:1.5rem; align-items:start;">
<div>

ECE / RMS-CE / MCE on every classification probe, plus a temperature-scaling baseline (Guo et al., 2017):

```yaml
eval:
  calibration:
    n_bins_knn: null   # null → knn_k + 1
    n_bins_linear: 15
    temp_scale: true   # +ece_ts / mce_ts
                       #  + temperature
```

```python
# fit T on val logits, NLL via LBFGS
T = fit_temperature(val_logits, y_val)
# T > 1 overconfident · T < 1 under
```

<span class="muted">KNN probs quantize to k+1 levels → binned at knn_k+1. TS applies to Linear only.</span>

</div>
<div style="font-family:'Inter',sans-serif; font-size:0.82em;">

**Linear probe — eurosat-spatial**

| Model | ECE | ECE·TS | T |
|-------|:---:|:------:|:-:|
| Swin Satlas-B | **.026** | .047 | 0.83 |
| OlmoEarth v1.1 Tiny | .031 | .035 | 0.98 |
| OlmoEarth v1.1 Base | .036 | .047 | 0.90 |
| OlmoEarth v1.1 Nano | .051 | **.021** | 1.19 |
| ResNet-50 SeCo | .075 | **.041** | 1.18 |

<p style="margin-top:0.4rem;"><span class="tag tag-claret">FINDING</span> Most probes are well-calibrated (ECE ≈ .03–.05). Temperature scaling is <strong>situational</strong> — it helps the overconfident models (<strong>T &gt; 1</strong>: Nano, ResNet) but <strong>raises</strong> ECE on the already-underconfident ones (T &lt; 1). Not a free win.</p>

</div>
</div>

---
layout: cover
---

# Key Findings

<div class="rule"></div>

<div style="font-family:'Inter',sans-serif; font-size:0.9em; display:grid; grid-template-columns:1fr 1fr; gap:1.5rem; margin-top:0.5rem;">
<div>

🌍 **S2 multispectral** (EuroSAT, BigEarthNet, So2Sat)<br>
→ <span class="accent"><strong>OlmoEarth</strong></span> dominates KNN + Linear

🌲 **RGB / aerial** (ForestNet, PV4GER, Brick Kiln)<br>
→ <span class="accent"><strong>DINOv3-SAT</strong></span> or <span class="accent"><strong>Panopticon</strong></span> lead

🌿 **Multi-label** (BigEarthNet, BENv2, TreeSatAI)<br>
→ OlmoEarth top KNN; Terramind competitive linear

</div>
<div>

⚡ Terramind **Base** beats Terramind **Large** on So2Sat linear (0.739 vs 0.712)

⚡ CLS token **collapses** on EuroSAT for Prithvi/Clay — patch-mean wins

⚡ **ImageStats** (raw pixel stats) still hits .95 on m-brick-kiln — low semantic difficulty, though GeoFMs now edge it out

⚡ **SatlasPretrain Swin** (RGB-only) peaks at m-pv4ger #9 / m-brick-kiln #16 — RGB backbones trail multispectral FMs on S2

⚡ **OlmoEarth v1.1** matches v1 at **≈ 3× fewer MACs**; Tiny/Nano gain most on So2Sat + BigEarthNet

⚡ Linear probes are **well-calibrated out of the box** (ECE ≈ .03–.05) — temperature scaling is situational (helps overconfident models, hurts underconfident ones)

</div>
</div>

<br>

<div style="font-family:'Source Serif 4',serif; font-size:1em; color:var(--ft-teal); font-style:italic;">
"OlmoEarth leads on S2 · DINOv3-SAT leads aerial · Terramind surprises on So2Sat"
</div>
