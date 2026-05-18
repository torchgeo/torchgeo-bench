---
theme: default
title: torchgeo-bench — What's New
highlighter: shiki
lineNumbers: true
fonts:
  sans: Inter
  serif: Source Serif 4
  mono: Fira Code
transition: fade
---

<style>
:root {
  --ft-pink:    #fff1e5;
  --ft-paper:   #fff1e5;
  --ft-rule:    #b3a9a0;
  --ft-text:    #262a33;
  --ft-muted:   #66605c;
  --ft-teal:    #0d7680;
  --ft-claret:  #990f3d;
  --ft-oxford:  #0f5499;
  --ft-wheat:   #b89b5e;
}

.slidev-layout {
  background: var(--ft-paper);
  color: var(--ft-text);
  font-family: "Source Serif 4", Georgia, serif;
}

h1, h2, h3 {
  font-family: "Source Serif 4", Georgia, serif;
  font-weight: 700;
  color: var(--ft-text);
}

code, .slidev-code {
  font-family: "Fira Code", monospace !important;
}

.accent { color: var(--ft-teal); }
.muted  { color: var(--ft-muted); }
.claret { color: var(--ft-claret); }
.wheat  { color: var(--ft-wheat); }

.rule {
  border-top: 2px solid var(--ft-claret);
  margin: 0.5rem 0 1rem;
}

table {
  font-family: "Inter", sans-serif;
  font-size: 0.82em;
  border-collapse: collapse;
  width: 100%;
}
th {
  background: var(--ft-text);
  color: var(--ft-paper);
  padding: 0.3rem 0.6rem;
  text-align: left;
}
td {
  padding: 0.25rem 0.6rem;
  border-bottom: 1px solid var(--ft-rule-soft, #d9cfc6);
}
tr:nth-child(even) td { background: rgba(0,0,0,0.03); }
.gold   { color: #b8860b; font-weight: 700; }
.silver { color: #708090; font-weight: 700; }
.bronze { color: #8b5e3c; font-weight: 700; }

.tag {
  display: inline-block;
  background: var(--ft-teal);
  color: var(--ft-paper);
  font-family: "Inter", sans-serif;
  font-size: 0.65em;
  font-weight: 600;
  padding: 0.15em 0.5em;
  border-radius: 3px;
  vertical-align: middle;
  letter-spacing: 0.04em;
}
.tag-claret { background: var(--ft-claret); }
.tag-oxford { background: var(--ft-oxford); }
.tag-wheat  { background: var(--ft-wheat); color: var(--ft-text); }
</style>

---
layout: cover
---

# torchgeo-bench
## Two-Week Sprint Recap

<div class="rule"></div>

<span class="muted" style="font-family:'Inter',sans-serif; font-size:0.9em;">
~40 PRs · GPU evaluation · new models · efficiency profiling · leaderboard results
</span>

<br>

<span class="muted" style="font-family:'Inter',sans-serif; font-size:0.8em;">May 2026</span>

---

# What shipped

<div class="rule"></div>

<div style="display:grid; grid-template-columns:1fr 1fr; gap:1.2rem; font-family:'Inter',sans-serif; font-size:0.85em; margin-top:0.5rem;">

<div>
<span class="tag">GPU</span> <strong>GPU KNN via faissknn</strong>
<p class="muted" style="margin:0.2rem 0 0.8rem;">FAISS-backed KNN with optional CUDA path. Automatic fallback to CPU.</p>

<span class="tag tag-claret">CLI</span> <strong>Typer + Rich</strong>
<p class="muted" style="margin:0.2rem 0 0.8rem;">Replaced argparse + tqdm. Beautiful progress bars, rich tracebacks.</p>

<span class="tag tag-oxford">PROFILE</span> <strong>Efficiency Profiling</strong>
<p class="muted" style="margin:0.2rem 0 0.8rem;">Throughput, GFLOPs, peak GPU mem, energy (Wh/1k samples), $/inference.</p>

<span class="tag tag-wheat">MODELS</span> <strong>OlmoEarth + DINOv3-SAT</strong>
<p class="muted" style="margin:0.2rem 0 0.8rem;">OlmoEarth nano/tiny/base/large. DINOv3-sat ViT-L web-pretrained.</p>
</div>

<div>
<span class="tag tag-claret">DATA</span> <strong>EuroSAT Spatial Split</strong>
<p class="muted" style="margin:0.2rem 0 0.8rem;">Geographically disjoint train/test. Harder than random split.</p>

<span class="tag tag-oxford">QUALITY</span> <strong>Cleanlab Audit</strong>
<p class="muted" style="margin:0.2rem 0 0.8rem;">Label-quality scores across all GeoBench V1+V2 datasets.</p>

<span class="tag">POOL</span> <strong>CLS + Mean Pool Ablations</strong>
<p class="muted" style="margin:0.2rem 0 0.8rem;">TerraTorch/ScaleMAE/Clay — pool=cls|mean|both sweep.</p>

<span class="tag tag-wheat">FIX</span> <strong>Silent-bug Sweep</strong>
<p class="muted" style="margin:0.2rem 0 0.8rem;">Removed try/except covers; fixed minmax_zscore, fp16 overflow, label gaps.</p>
</div>

</div>

---

# GPU KNN — faissknn
<span class="tag">PR #53 · #55 · #89</span>

<div class="rule"></div>

<div style="display:grid; grid-template-columns:1fr 1fr; gap:1.5rem; align-items:start;">

<div>

CPU path (always available):

```python
# faiss-cpu IndexFlatL2
clf = KNNClassifier(n_neighbors=5, device="cpu")
clf.fit(x_train, y_train)
preds = clf.predict(x_test)
```

GPU path (opt-in):

```python
# pip install -e ".[cuda]"
clf = KNNClassifier(
    n_neighbors=5,
    device="cuda:0",
    metric="cosine",   # l2 | ip | cosine
)
clf.fit(x_train, y_train)
preds = clf.predict(x_test)    # numpy out
proba = clf.predict_proba(x_test)
```

Multi-label — auto-detected from `y` shape:

```python
# y shape (n, n_classes) → multilabel mode
clf.fit(x_train, Y_multilabel)
scores = clf.predict_proba(x_test)  # (n, C)
```

</div>

<div style="font-family:'Inter',sans-serif; font-size:0.82em;">

**Key fixes landed:**

- `n_classes = max(y)+1` instead of `len(unique(y))` — avoids `IndexError` when a small partition is missing class labels
- `use_fp16=False` in evaluation — raw sensor DN values (~10 000) overflow fp16 L2 distances → random KNN
- faiss-cpu + faiss-cuda namespace conflict resolved: GPU SLURM jobs swap packages before running tests

<br>

**Auto-fallback:**

```python
# If faissknn not installed → silently falls
# back to CPU faiss path (no crash)
```

</div>
</div>

---

# Typer + Rich CLI
<span class="tag tag-claret">PR #88</span>

<div class="rule"></div>

<div style="display:grid; grid-template-columns:1fr 1fr; gap:1.5rem;">

<div>

**Before** (argparse):

```python
parser = argparse.ArgumentParser()
parser.add_argument("--model", ...)
parser.add_argument("--dataset", ...)
args = parser.parse_args()
```

**After** (Typer):

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
def download(
    target: str,
    output_dir: Path = Path("data"),
) -> None:
    """Download benchmark datasets."""
    ...
```

</div>

<div>

**Rich progress** (replacing tqdm):

```python
from rich.progress import track

for batch in track(dataloader,
                   description="Extracting"):
    features = model(batch["image"])

for ds in track(dataset_names,
                description="Datasets"):
    run_benchmark(ds)
```

**Rich logging:**

```python
from rich.logging import RichHandler

logging.basicConfig(
    handlers=[RichHandler(rich_tracebacks=True)]
)
```

**Rich tables** (tune_dataloader.py):

```python
from rich.table import Table
table = Table(header_style="bold cyan")
table.add_column("bs", justify="right")
table.add_column("sps")
table.add_row("256", "1420.3")
console.print(table)
```

</div>
</div>

---

# Efficiency Profiling
<span class="tag tag-oxford">PR #60 · #62 · #63 · #67 · #80</span>

<div class="rule"></div>

<div style="display:grid; grid-template-columns:1fr 1fr; gap:1.5rem;">

<div>

Metrics recorded per model:

```python
{
  # GPU throughput
  "throughput_samples_per_sec": 1420.3,
  "latency_ms_per_batch_p50":   18.1,
  "peak_gpu_mem_gb":             3.2,
  "reserved_gpu_mem_gb":         4.0,
  "sm_utilization_avg":          87.4,
  "gpu_power_w_avg":             182.0,

  # CPU throughput (for deployment cost)
  "throughput_samples_per_sec_cpu": 42.1,
  "latency_ms_per_batch_p50_cpu":   238.0,

  # Compute cost
  "params_m":    307.4,
  "gflops":      61.6,

  # Energy
  "energy_wh_per_1k_samples": 0.036,

  # Derived
  "cost_usd_per_1M_samples":  0.12,
  "gco2_per_1M_samples":      18.4,
}
```

</div>

<div style="font-family:'Inter',sans-serif; font-size:0.82em;">

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
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
energy_wh = (power_mw / 1000) * (elapsed_s / 3600)
```

**Cost extrapolation:**

```python
# A100 spot ~$1.50/hr on Lambda
cost_per_1M = (1_000_000 / throughput) \
              / 3600 * 1.50
```

Explorer shows Pareto front: accuracy vs cost/CO₂.

</div>
</div>

---

# OlmoEarth Integration
<span class="tag tag-wheat">PR #84 · #85</span>

<div class="rule"></div>

<div style="display:grid; grid-template-columns:1fr 1fr; gap:1.5rem;">

<div>

Configs added (nano → large):

```yaml
# conf/model/olmoearth_v1_base.yaml
_target_: torchgeo_bench.models.OlmoEarthBench
name: olmoearth_v1_base
variant: base
normalization: identity   # model does own norm
```

Auto-rescale to S2 DN:

```python
class OlmoEarthBench(BenchModel):
    expected_input_unit = InputUnit.S2_DN

    def _forward_patch_features(
        self, images: Tensor, **_
    ) -> Tensor:
        # images already rescaled to S2 DN
        # by the model_native normalizer
        return self.backbone(images)
```

</div>

<div style="font-family:'Inter',sans-serif; font-size:0.82em;">

**Results — m-eurosat Linear Accuracy:**

| Model | Acc |
|-------|-----|
| <span class="gold">OlmoEarth Large</span> | **0.976** |
| <span class="silver">OlmoEarth Base</span> | **0.975** |
| DOFA Large | 0.973 |
| DINOv3-SAT ViT-L | 0.969 |
| Panopticon | 0.968 |

**Dominates S2 datasets** (EuroSAT, BigEarthNet, So2Sat). Nano/Tiny competitive despite much smaller size.

<br>

**Key**: `normalization=identity` bypasses z-score — OlmoEarth handles its own preprocessing internally.

</div>
</div>

---

# CLS Token Ablations + Pool Modes
<span class="tag">PR #73 · #74 · #75 · #76</span>

<div class="rule"></div>

<div style="display:grid; grid-template-columns:1fr 1fr; gap:1.5rem;">

<div>

`pool` kwarg across TerraTorch wrappers:

```python
class TerramindBench(BenchModel):
    def _forward_patch_features(
        self,
        images: Tensor,
        bboxes: Tensor | None = None,
    ) -> Tensor:
        out = self.model.encode(images)
        if self.pool == "cls":
            return out[:, 0]        # CLS token
        elif self.pool == "mean":
            return out[:, 1:].mean(1)  # patch mean
        else:  # "both"
            return torch.cat([
                out[:, 0],
                out[:, 1:].mean(1)
            ], dim=1)
```

Config variants:

```yaml
# _cls suffix → pool: cls
name: tt_clay_v1_5_base_cls
pool: cls

# default → pool: mean
name: tt_clay_v1_5_base
pool: mean
```

</div>

<div style="font-family:'Inter',sans-serif; font-size:0.82em;">

**Finding:** CLS token helps on ForestNet / Brick Kiln; patch-mean wins on EuroSAT / BigEarthNet.

**Terramind** has no CLS token — those configs dropped (`#76`).

<br>

**Clay v1.5 Base+CLS — m-brick-kiln Linear:**

| Model | Acc |
|-------|-----|
| DINOv3-SAT ViT-L | 0.976 |
| <span class="gold">Clay Base+CLS</span> | **0.975** |
| DOFA Base | 0.974 |
| DOFA Large | 0.974 |
| Clay Base | 0.972 |

</div>
</div>

---

# EuroSAT Spatial Split + Cleanlab
<span class="tag tag-claret">PR #50 · #52</span>

<div class="rule"></div>

<div style="display:grid; grid-template-columns:1fr 1fr; gap:1.5rem;">

<div>

**EuroSAT Spatial** — geographically disjoint split:

```python
# Standard split: random train/test
# → models can memorize spatial autocorrelation

# Spatial split: tiles from different
# geographic regions in train vs test
# → true generalization test

ds = EuroSATSpatialBench(
    root=DATA_ROOT,
    partition="default",  # spatial disjoint
    bands=["B02","B03","B04","B08"],
)
```

Harder than `m-eurosat`:
- OlmoEarth Large KNN: **0.959** (vs 0.956 random)
- OlmoEarth Base Linear: **0.978** (vs 0.975 random)

</div>

<div>

**Cleanlab label audit:**

```python
from cleanlab.filter import find_label_issues

issues = find_label_issues(
    labels=y_train,
    pred_probs=pred_proba,
    return_indices_ranked_by="self_confidence",
)
# Flags likely mislabeled samples
```

Applied across all GeoBench V1+V2 datasets. Results saved to `results/cleanlab/`.

<br>

<span class="muted" style="font-family:'Inter',sans-serif; font-size:0.82em;">
Surfaces annotation noise in BigEarthNet, ForestNet, and TreeSatAI — useful for reweighting or curriculum training experiments.
</span>

</div>
</div>

---

# Leaderboard — EuroSAT + BigEarthNet
<span class="tag tag-oxford">Results</span>

<div class="rule"></div>

<div style="display:grid; grid-template-columns:1fr 1fr; gap:1.5rem;">

<div>

**m-eurosat** (Accuracy)

| | Model | KNN | Lin |
|--|-------|-----|-----|
| <span class="gold">1</span> | OlmoEarth Large | .956 | .976 |
| <span class="silver">2</span> | OlmoEarth Base | .946 | .975 |
| <span class="bronze">3</span> | Panopticon | .948 | .968 |
| 4 | DOFA Large | .936 | .973 |
| 5 | DINOv3-SAT ViT-L | — | .969 |

**eurosat-spatial** (Accuracy)

| | Model | KNN | Lin |
|--|-------|-----|-----|
| <span class="gold">1</span> | OlmoEarth Large | .959 | .977 |
| <span class="silver">2</span> | OlmoEarth Base | .941 | .978 |
| <span class="bronze">3</span> | ResNet50-S2 MoCo | .933 | — |
| 4 | DINOv3 ViT-L | .932 | — |
| 5 | Panopticon | .930 | .962 |

</div>

<div>

**m-bigearthnet** (micro-mAP)

| | Model | KNN | Lin |
|--|-------|-----|-----|
| <span class="gold">1</span> | OlmoEarth Large | .664 | .764 |
| <span class="silver">2</span> | OlmoEarth Base | .658 | .769 |
| <span class="bronze">3</span> | Panopticon | .652 | — |
| 4 | Terramind Large | .628 | .750 |
| 5 | DINOv3-SAT ViT-L | .625 | .741 |

**benv2** (micro-mAP)

| | Model | KNN | Lin |
|--|-------|-----|-----|
| <span class="gold">1</span> | OlmoEarth Base | .735 | .853 |
| <span class="silver">2</span> | OlmoEarth Large | .728 | .850 |
| <span class="bronze">3</span> | OlmoEarth Tiny | .716 | — |
| 4 | Panopticon | .716 | — |
| 5 | Terramind Large | .712 | .846 |

</div>
</div>

---

# Leaderboard — ForestNet + So2Sat
<span class="tag tag-oxford">Results</span>

<div class="rule"></div>

<div style="display:grid; grid-template-columns:1fr 1fr; gap:1.5rem;">

<div>

**m-forestnet** (Accuracy)

| | Model | KNN | Lin |
|--|-------|-----|-----|
| <span class="gold">1</span> | Panopticon | .427 | — |
| <span class="silver">2</span> | DINOv3-SAT ViT-L | .425 | .582 |
| <span class="bronze">3</span> | Clay v1.5 Base+CLS | .414 | .556 |
| 4 | ScaleMAE Large+CLS | .403 | .569 |
| 5 | Clay v1.5 Base | .395 | — |

**m-so2sat** (Accuracy)

| | Model | KNN | Lin |
|--|-------|-----|-----|
| <span class="gold">1</span> | OlmoEarth Base | .576 | .712 |
| <span class="silver">2</span> | OlmoEarth Large | .560 | — |
| <span class="bronze">3</span> | Panopticon | .532 | — |
| 4 | OlmoEarth Tiny | .506 | — |
| 5 | DINOv3 ViT-L | .490 | — |

<span class="muted" style="font-size:0.8em; font-family:'Inter',sans-serif;">Linear: Terramind Base .739 leads So2Sat</span>

</div>

<div>

**m-brick-kiln** (Accuracy)

| | Model | KNN | Lin |
|--|-------|-----|-----|
| <span class="gold">1</span> | DOFA Large | .969 | .974 |
| <span class="silver">2</span> | DOFA Base | .965 | .974 |
| <span class="bronze">3</span> | Clay v1.5 Base | .964 | .972 |
| 4 | Clay v1.5 Base+CLS | .960 | .975 |
| 5 | DINOv3-SAT ViT-L | — | .976 |

**m-pv4ger** (Accuracy)

| | Model | KNN | Lin |
|--|-------|-----|-----|
| <span class="gold">1</span> | DOFA Large | .967 | .969 |
| <span class="silver">2</span> | DOFA Base | .965 | — |
| <span class="bronze">3</span> | DINOv3 ViT-L | .964 | .974 |
| 4 | Panopticon | .958 | .968 |
| 5 | ScaleMAE Large+CLS | .956 | .970 |

</div>
</div>

---

# Leaderboard — TreeSatAI + Key Findings
<span class="tag tag-oxford">Results</span>

<div class="rule"></div>

<div style="display:grid; grid-template-columns:1fr 1fr; gap:1.5rem;">

<div>

**treesatai** (micro-mAP)

| | Model | KNN | Lin |
|--|-------|-----|-----|
| <span class="gold">1</span> | DINOv3-SAT ViT-L | .477 | .682 |
| <span class="silver">2</span> | Clay v1.5 Base+CLS | .475 | .671 |
| <span class="bronze">3</span> | DOFA Large | .474 | — |
| 4 | DINOv3 ViT-L | .473 | .664 |
| 5 | OlmoEarth Nano | .470 | — |

<span class="muted" style="font-size:0.8em; font-family:'Inter',sans-serif;">
Linear: Clay v1.5 Base .673 — CLS barely helps here
</span>

</div>

<div style="font-family:'Inter',sans-serif; font-size:0.84em;">

**Overall winners by task type:**

🌍 **S2 multispectral** (EuroSAT, BigEarthNet, So2Sat)
→ <strong class="accent">OlmoEarth</strong> dominates KNN + Linear

🌲 **RGB / aerial** (ForestNet, PV4GER, Brick Kiln)
→ <strong class="accent">DINOv3-SAT</strong> or <strong class="accent">Panopticon</strong> lead

🌿 **Multi-label** (BigEarthNet, BENv2, TreeSatAI)
→ OlmoEarth top KNN; Terramind competitive on Linear

⚡ **Surprising:** Terramind Base beats Terramind Large on So2Sat Linear (0.739 vs 0.712)

⚡ **Surprising:** CLS token collapse on EuroSAT for Prithvi/Clay — patch-mean wins

</div>
</div>

---
layout: cover
---

# Summary

<div class="rule"></div>

<div style="font-family:'Inter',sans-serif; font-size:0.9em; display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-top:0.5rem;">

<div>
✅ GPU KNN via faissknn (FAISS-backed, CUDA opt-in)<br>
✅ Typer + Rich CLI (argparse gone)<br>
✅ Full efficiency profile (throughput, energy, cost)<br>
✅ OlmoEarth nano/tiny/base/large<br>
✅ DINOv3-SAT ViT-L
</div>

<div>
✅ EuroSAT spatial split (geographically disjoint)<br>
✅ Cleanlab label audit (all GeoBench datasets)<br>
✅ CLS vs mean pool ablations<br>
✅ Silent-bug sweep (fp16 overflow, label gaps, zscore)<br>
✅ ~40 PRs merged
</div>

</div>

<br>

<div style="font-family:'Source Serif 4',serif; font-size:1.1em; color:var(--ft-teal); font-style:italic; margin-top:1rem;">
"OlmoEarth leads on S2 · DINOv3-SAT leads aerial · Terramind surprises on So2Sat"
</div>
