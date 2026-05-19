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
~40 PRs · GPU evaluation · new models · efficiency profiling · leaderboard results
</span>

<br>

<span style="font-family:'Inter',sans-serif; font-size:0.8em; color:var(--ft-muted)">May 2026</span>

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
<span class="muted">Throughput, GFLOPs, peak GPU mem, energy (Wh/1k), $/inference.</span></p>

<p><span class="tag tag-wheat">MODELS</span> <strong>OlmoEarth + DINOv3-SAT</strong><br>
<span class="muted">OlmoEarth nano/tiny/base/large. DINOv3-sat ViT-L web-pretrained.</span></p>

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

</div>
</div>

---

# GPU KNN — faissknn
<span class="tag">PR #53</span> <span class="tag">PR #55</span> <span class="tag">PR #89</span>

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
- faiss-cpu + faiss-cuda namespace conflict resolved on GPU SLURM nodes

<br>

**Auto-fallback:**

```python
# faissknn not installed →
# silently falls back to CPU faiss path
```

</div>
</div>

---

# Typer + Rich CLI
<span class="tag tag-claret">PR #88</span>

<div class="rule"></div>

<div style="display:grid; grid-template-columns:1fr 1fr; gap:1.5rem;">
<div>

**Before** (argparse + tqdm):

```python
parser = argparse.ArgumentParser()
parser.add_argument("--model", ...)
args = parser.parse_args()

for batch in tqdm(dataloader):
    ...
```

**After** (Typer + Rich):

```python
app = typer.Typer(no_args_is_help=True)

@app.command(context_settings={
    "allow_extra_args": True,
    "ignore_unknown_options": True,
})
def run(ctx: typer.Context) -> None:
    """Run benchmarks; extra args → Hydra."""
    sys.argv = [sys.argv[0], *ctx.args]
    hydra_main()

@app.command()
def download(target: str,
             output_dir: Path = Path("data"),
             ) -> None:
    """Download benchmark datasets."""
    ...
```

</div>
<div>

**Rich progress bars:**

```python
from rich.progress import track

for batch in track(dataloader,
                   description="Extracting"):
    features = model(batch["image"])
```

**Rich logging:**

```python
from rich.logging import RichHandler

logging.basicConfig(handlers=[
    RichHandler(rich_tracebacks=True)
])
```

**Rich tables (tune_dataloader.py):**

```python
from rich.table import Table
table = Table(header_style="bold cyan")
table.add_column("bs",  justify="right")
table.add_column("sps", justify="right")
table.add_row("256", "1420.3")
console.print(table)
```

</div>
</div>

---

# Efficiency Profiling
<span class="tag tag-oxford">PR #60</span> <span class="tag tag-oxford">PR #62</span> <span class="tag tag-oxford">PR #63</span> <span class="tag tag-oxford">PR #67</span>

<div class="rule"></div>

<div style="display:grid; grid-template-columns:1fr 1fr; gap:1.5rem;">
<div>

Metrics recorded per model run:

```python
{
  # GPU
  "throughput_samples_per_sec": 1420.3,
  "peak_gpu_mem_gb":            3.2,
  "gpu_power_w_avg":            182.0,
  "energy_wh_per_1k_samples":   0.036,
  "gflops":                     61.6,
  "params_m":                   307.4,
  # CPU
  "throughput_samples_per_sec_cpu": 42.1,
  # Cost
  "cost_usd_per_1M_samples":    0.12,
  "gco2_per_1M_samples":        18.4,
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

**Energy via pynvml:**

```python
import pynvml
pynvml.nvmlInit()
h = pynvml.nvmlDeviceGetHandleByIndex(0)
mw  = pynvml.nvmlDeviceGetPowerUsage(h)
wh  = (mw / 1000) * (elapsed_s / 3600)
```

**Cost extrapolation:**

```python
# A100 spot ~$1.50/hr on Lambda
cost = (1_000_000 / throughput) / 3600 * 1.50
```

Explorer shows Pareto front: accuracy vs cost / CO₂.

</div>
</div>

---

# OlmoEarth Integration
<span class="tag tag-wheat">PR #84</span> <span class="tag tag-wheat">PR #85</span>

<div class="rule"></div>

<div style="display:grid; grid-template-columns:1fr 1fr; gap:1.5rem;">
<div>

Config (nano → large):

```yaml
# conf/model/olmoearth_v1_base.yaml
_target_: torchgeo_bench.models.OlmoEarthBench
name: olmoearth_v1_base
variant: base
normalization: identity  # model handles its own
```

Auto-rescale to S2 DN:

```python
class OlmoEarthBench(BenchModel):
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
| 4 | DINOv3-SAT ViT-L | 0.969 |
| 5 | Panopticon | 0.968 |

<br>

**Dominates S2 datasets** — EuroSAT, BigEarthNet, So2Sat. Nano/Tiny competitive at a fraction of the size.

`normalization=identity` bypasses z-score — OlmoEarth handles its own preprocessing.

</div>
</div>

---

# CLS Token + Pool Ablations
<span class="tag">PR #73</span> <span class="tag">PR #74</span> <span class="tag">PR #75</span> <span class="tag">PR #76</span>

<div class="rule"></div>

<div style="display:grid; grid-template-columns:1fr 1fr; gap:1.5rem;">
<div>

`pool` kwarg across TerraTorch wrappers:

```python
class TerramindBench(BenchModel):
    def _forward_patch_features(
        self, images: Tensor, **_
    ) -> Tensor:
        out = self.model.encode(images)
        if self.pool == "cls":
            return out[:, 0]
        elif self.pool == "mean":
            return out[:, 1:].mean(1)
        else:  # "both"
            return torch.cat([
                out[:, 0],
                out[:, 1:].mean(1),
            ], dim=1)
```

Config variants:

```yaml
name: tt_clay_v1_5_base_cls   # pool: cls
name: tt_clay_v1_5_base       # pool: mean (default)
```

</div>
<div style="font-family:'Inter',sans-serif; font-size:0.82em;">

**Finding:** CLS helps on ForestNet / Brick Kiln; patch-mean wins on EuroSAT / BigEarthNet.

**Terramind** has no CLS token — `_cls` configs dropped (`#76`).

<br>

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
ds = EuroSATSpatialBench(
    root=DATA_ROOT,
    partition="default",
    bands=["B02","B03","B04","B08"],
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
| <span class="bronze">3</span> | Panopticon | .948 | .962 |
| 4 | OlmoEarth Tiny | .926 | .953 |
| 5 | DOFA Large | .911 | .967 |

</div>
<div>

**eurosat-spatial** (Accuracy)

| | Model | KNN | Lin |
|--|-------|:---:|:---:|
| <span class="gold">1</span> | OlmoEarth Large | .959 | .977 |
| <span class="silver">2</span> | OlmoEarth Base | .941 | .978 |
| <span class="bronze">3</span> | Panopticon | .930 | .962 |
| 4 | OlmoEarth Tiny | .928 | .963 |
| 5 | DOFA Large | .924 | .965 |

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
| <span class="gold">1</span> | OlmoEarth Large | .664 | .764 |
| <span class="silver">2</span> | OlmoEarth Base | .658 | .769 |
| <span class="bronze">3</span> | Panopticon | .652 | .735 |
| 4 | Terramind Base | .615 | .740 |
| 5 | OlmoEarth Tiny | .607 | .691 |

</div>
<div>

**benv2** (micro-mAP)

| | Model | KNN | Lin |
|--|-------|:---:|:---:|
| <span class="gold">1</span> | OlmoEarth Base | .735 | .853 |
| <span class="silver">2</span> | OlmoEarth Large | .728 | .850 |
| <span class="bronze">3</span> | Terramind Large | .712 | .846 |
| 4 | Terramind Base | .710 | .839 |
| 5 | Panopticon | .713 | .824 |

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
| <span class="gold">1</span> | ScaleMAE Large+CLS | .403 | .569 |
| <span class="silver">2</span> | DINOv3-SAT ViT-L | .386 | .573 |
| <span class="bronze">3</span> | DOFA Large | .369 | .574 |
| 4 | ScaleMAE Large | .390 | .544 |
| 5 | ResNet50-RGB MoCo | .389 | .537 |

<span class="muted" style="font-size:0.8em;">Hard dataset — all models below .60</span>

</div>
<div>

**m-so2sat** (Accuracy)

| | Model | KNN | Lin |
|--|-------|:---:|:---:|
| <span class="gold">1</span> | OlmoEarth Base | .576 | .712 |
| <span class="silver">2</span> | OlmoEarth Large | .560 | .678 |
| <span class="bronze">3</span> | Clay v1.5 Base+CLS | .448 | .694 |
| 4 | OlmoEarth Tiny | .506 | .635 |
| 5 | Terramind Base | .385 | .739 |

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
| <span class="gold">1</span> | Clay Base+CLS | .952 | .974 |
| <span class="silver">2</span> | OlmoEarth Base | .945 | .970 |
| <span class="bronze">3</span> | **ImageStats** | .949 | .953 |
| 4 | OlmoEarth Large | .929 | .968 |
| 5 | Terramind Base | .916 | .969 |

</div>
<div>

**m-pv4ger** (Acc)

| | Model | KNN | Lin |
|--|-------|:---:|:---:|
| <span class="gold">1</span> | DINOv3 ViT-L | .964 | .974 |
| <span class="silver">2</span> | DOFA Large | .967 | .969 |
| <span class="bronze">3</span> | DOFA Base | .965 | .966 |
| 4 | ScaleMAE+CLS | .956 | .970 |
| 5 | ResNet50-RGB MoCo | .954 | .961 |

</div>
<div>

**treesatai** (mAP)

| | Model | KNN | Lin |
|--|-------|:---:|:---:|
| <span class="gold">1</span> | DOFA Large | .474 | .647 |
| <span class="silver">2</span> | OlmoEarth Base | .469 | .647 |
| <span class="bronze">3</span> | **RCF** | .459 | .644 |
| 4 | Terramind Base | .457 | .645 |
| 5 | DOFA Base | .456 | .644 |

</div>
</div>

---

# Efficiency — Throughput vs Accuracy
<span class="tag tag-oxford">PR #60–#80</span>

<div class="rule"></div>

<div style="display:grid; grid-template-columns:1fr 1fr; gap:2rem;">
<div>

**GPU throughput (img/s) — m-eurosat**

| Model | img/s | GFLOPs | Acc |
|-------|------:|-------:|----:|
| ResNet-50 MoCo | 3 193 | 8 | .76 |
| DOFA Base | 1 747 | 36 | .71 |
| Terramind Base | 1 680 | 36 | .68 |
| OlmoEarth Tiny | 780 | 9 | .73 |
| OlmoEarth Nano | 789 | 2 | .67 |
| DOFA Large | 581 | 124 | .73 |
| DINOv3-SAT ViT-L | 346 | 165 | .74 |
| OlmoEarth Large | 151 | 381 | .74 |

</div>
<div style="font-family:'Inter',sans-serif; font-size:0.82em;">

**Pareto surprises:**

<p><span class="tag tag-claret">WINNER</span> <strong>ResNet-50 MoCo</strong> — 3 193 img/s, 8 GFLOPs, 23M params, yet <strong>.76 avg acc</strong>. Matches or beats every ViT-scale model while running 5–10× faster.</p>

<p><span class="tag">EFFICIENT</span> <strong>OlmoEarth Nano</strong> — 3.6M params, 0.6 GB peak VRAM, 1.6 GFLOPs. Matches Panopticon accuracy at 5× lower cost.</p>

<p><span class="tag">EFFICIENT</span> <strong>OlmoEarth Tiny</strong> (14M) ties DOFA Large (337M) and DINOv3-SAT (304M) at 2× the throughput.</p>

**Worst value:** Prithvi v1/v2 — 1 700 img/s, 35 GFLOPs, only .56–.60 acc. Same throughput tier as DOFA Base but far lower accuracy.

</div>
</div>

---

# Intrinsic Dimensionality Analysis
<span class="tag tag-oxford">method: intrinsic_dim</span>

<div class="rule"></div>

<div style="display:grid; grid-template-columns:1fr 1fr; gap:2rem;">
<div>

**id_TwoNN (avg across datasets)**

| Model | id_TwoNN | Feat dim | Avg acc |
|-------|:--------:|:--------:|:-------:|
| EarthLoc ResNet-50 | **44** | 4 096 | .59 |
| OlmoEarth Large | 22 | 1 024 | **.74** |
| DOFA Large | 18 | 1 024 | .73 |
| DINOv3-SAT ViT-L | 16 | 1 024 | .74 |
| ResNet-50 MoCo | 15 | 2 048 | .76 |
| OlmoEarth Nano | 14 | **128** | .67 |
| Panopticon | 13 | 768 | .66 |
| Terramind Base | 10 | 768 | .68 |
| CROMA Base | 8 | 768 | .59 |
| Prithvi v1/v2 | 7–8 | 768–1280 | .56–.60 |
| imagestats | 6 | 12 | .57 |

</div>
<div style="font-family:'Inter',sans-serif; font-size:0.82em;">

**Pattern:** High intrinsic dim ↔ high accuracy. Top models (OlmoEarth, DOFA, DINOv3-SAT, ResNet-50 MoCo) all ID > 14.

<br>

**Outliers:**

<p><span class="tag tag-claret">TASK GAP</span> <strong>EarthLoc ResNet-50</strong> — id = 44 (highest), but only .59 acc. Trained for geo-localization, not classification. High dimensionality without discriminative structure.</p>

<p><span class="tag">SURPRISE</span> <strong>OlmoEarth Nano</strong> — 128-d output yet id_TwoNN = 14, matching DINOv3-SAT ViT-L (1024-d). Packs equivalent intrinsic complexity into 8× fewer dimensions.</p>

<p><span class="tag tag-wheat">FLAT</span> <strong>Prithvi cluster</strong> — id ≈ 7, just above imagestats (6) and RCF (5). Geometrically flat despite 768–1280-d features — explains consistently low accuracy.</p>

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

⚡ **ImageStats** (raw pixel stats) ranks #3 on m-brick-kiln — dataset may have low semantic difficulty

⚡ **RCF** (random conv. features) ranks #3 on treesatai — signals weak label discrimination

</div>
</div>

<br>

<div style="font-family:'Source Serif 4',serif; font-size:1em; color:var(--ft-teal); font-style:italic;">
"OlmoEarth leads on S2 · DINOv3-SAT leads aerial · Terramind surprises on So2Sat"
</div>
