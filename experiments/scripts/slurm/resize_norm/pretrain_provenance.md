# Pretraining input-size and normalization provenance

Paper-sourced facts for models whose `model_native` normalization was
undefined or whose resize behavior needed grounding. Collected 2026-08-18 by
research agents from the cited papers/repos; used to (a) backfill
`pretrain_mean/std` + `expected_input_unit` where defined, and (b) document
`target_size` choices. Say-UNKNOWN entries stay unknown — no guessed stats.

## DINOv3 (arXiv:2508.10104; facebookresearch/dinov3; timm)

| model | pretrain size | patch | normalization | variable size |
|---|---|---|---|---|
| vit_large_patch16_dinov3.lvd1689m | 256 global / 112 local; high-res adapt up to 768 | 16 | uint8/255 then ImageNet mean (0.485,0.456,0.406) std (0.229,0.224,0.225) | yes (RoPE, no abs pos-embed); H,W must be multiples of 16 |
| vit_large_patch16_dinov3.sat493m | 256; high-res fine-tune at 512 (max 512); Maxar RGB 0.6 m | 16 | uint8/255 then mean (0.430,0.411,0.296) std (0.213,0.156,0.143); no official guidance for non-8-bit sensors | yes, same constraint |
| convnext_large.dinov3_lvd1689m | distilled at 256/112 regime (high-res phase unconfirmed); evals at 256 and 512 | conv (stride 32) | uint8/255 then ImageNet stats (same web transform) | yes (fully conv); safe sizes are multiples of 32; timm cfg's 224 is a default, not the pretrain res |

Notes: timm defaults `global_pool='avg'`; upstream DINOv3 uses CLS pooling.
HF config says `fixed_input_size: false`; timm master `_dinov3_cfg` says
`True` but the builder passes `dynamic_img_size=True`, so variable size works.

Sources: paper §3.2/§5.1/§5.2/§7.2/§8.1; dinov3 README transforms
(README.md:255-285); dinov3_vit7b16_pretrain.yaml:134-135;
dinov3_vitl16_lvd1689m_distilled.yaml:205-206; MODEL_CARD.md:105; timm
eva.py `_dinov3_cfg` and HF config.json per weight.

## SSL4EO-S12 MoCo / SeCo / GASSL (agents, 2026-08-18)

| model | pretrain size | normalization | notes |
|---|---|---|---|
| SSL4EO MoCo rn18/rn50 S2 (arXiv:2211.07044) | 224 (RandomResizedCrop from 264px patches) | DN/10000*255 -> uint8 -> ToTensor (/255); no mean/std standardization; code does NOT clip >10000 (wraps mod 256) despite README | RGB checkpoints: which of the two 8-bit builds was used is UNKNOWN (no RGB pretrain script committed) |
| SeCo rn18/rn50 S2 RGB (ICCV 2021) | 224 (RandomResizedCrop from 264px) | L2A /10000 -> uint8; per-band quantile stretch min={B2:3,B3:2,B4:0} max={B2:88,B3:103,B4:129} -> /255 -> ImageNet mean/std | |
| GASSL rn50 fMoW RGB (NeurIPS 2021) | 224 (temporal path: Resize 448 -> RRC 224) | JPEG /255 -> ImageNet mean/std | |

Cross-check: torchgeo's weights transforms reproduce these chains (resnet.py
L85-173, L325-380) except Resize256+CenterCrop224 instead of the training
random crop; torchgeo assumes the /10000 path for SSL4EO RGB weights.

Sources: zhu-xlab/SSL4EO-S12 pretrain_moco_v2_s2c.py, ssl4eo_dataset.py
(get_array), srun_train_moco_rn50_s2c.sh; ServiceNow/seasonal-contrast
seco_dataset.py, seco_downloader.py; sustainlab-group/geography-aware-ssl
main_moco_geo+tp.py, fmow_dataloader.py.

## DOFA / Panopticon (agents, 2026-08-18)

| model | pretrain size | patch | normalization | variable size |
|---|---|---|---|---|
| DOFA ViT-B/L (arXiv:2403.15356) | 224 | 16 | per-sensor per-band z-score on BYTE-scale (0-255) values; S2 is 9-band (B04,B03,B02,B05,B06,B07,B08,B11,B12); S2 RGB byte stats: B04 114.11/77.84, B03 114.82/69.97, B02 126.64/67.42. DOFA's own GeoBench eval uses the DATASET's stats (paper 4.1.2) | NO: plain pos_embed add, exactly 196 tokens; flexibility is spectral, not spatial |
| Panopticon ViT-B/14 (arXiv:2503.10845) | 224 global / 98 local | 14 (pos_embed grid 518/14=37) | per-dataset per-band standard normal (each pretraining dataset standardized by its own train-split stats); GeoBench eval uses GeoBench normalization_stats() | reference model yes (pos-embed interp); torchgeo wrapper fixes size at init (224) |

Decision for the sweep: DOFA model_native stays undefined — pretraining used
byte-scale stats behind an unreconstructible DN->byte prep, and the paper's
own downstream protocol is dataset stats. Panopticon's native convention IS
dataset-stats z-score, so its bandspec_zscore cells are its native protocol
(no separate model_native cell exists in principle).

torchgeo gotcha (both): weight enums ship no normalization transform (DOFA =
center-crop only; Panopticon = Identity).

Sources: DOFA paper 4.1.2/4.5, zhu-xlab/DOFA ofall_dataset.py + waves.json +
dofa_v1.py; Panopticon stage1.yaml, fmow.py, satlas.py,
vision_transformer.py; torchgeo dofa.py / panopticon.py.

## Scale-MAE / SatlasPretrain (agents, 2026-08-18)

| model | pretrain size | patch/window | normalization | variable size |
|---|---|---|---|---|
| Scale-MAE ViT-L (ICCV 2023) | 224 encoder (448 recon target); fMoW base_resolution 2.5 m; reference GSD 1 m | 16 | /255 then ImageNet mean/std (fMoW stats commented out in repo: "imagenet numbers work a bit better") | YES: sin-cos pos embed generated on the fly from grid+GSD; sizes must be multiples of 16. torchgeo default res=1.0 mismatches pretraining 2.5 |
| SatlasPretrain Swin-v2-B/T, RN50/152 (ICCV 2023) | 512x512 for ALL variants | Swin patch 4, window 8; RN strides 4-32 | uint8/255 only, NO mean/std; S2 non-TCI bands /8160 clip [0,1]; Landsat (x-4000)/16320 | YES (log-CPB windows, internal padding); prefer multiples of 32. Trained/evaled at 512 only |

Key: torchgeo's Satlas transforms CenterCrop(256) — HALF the 512 pretraining
chip — and omit the clip step for S1/Landsat/non-TCI. "Checkpoint resolution"
is therefore ill-defined in released wrappers, supporting the native-vs-224
axis over a "pretrain size" axis.

## Backfill decisions (final)

- DOFA: model_native stays undefined — byte-scale pretraining stats sit
  behind an unreconstructible DN->byte prep; DOFA's own GeoBench protocol is
  dataset stats.
- Panopticon: pretraining normalization IS per-dataset z-score, so the
  bandspec_zscore cells are its native protocol; no separate cell exists.
- convnext_large_dinov3: released rows are all-bands multispectral; ImageNet
  RGB stats don't apply to 12/13-channel input. Undefined is correct.
- RCF: random features, no pretrain stats by construction.
- No wrapper config changes or supplemental reruns required.

## CROMA / Clay / EarthLoc (agents, 2026-08-18)

| model | pretrain size | patch | normalization | variable size |
|---|---|---|---|---|
| CROMA base/large (NeurIPS 2023) | 120x120 (random 60-180px crops of 264px SSL4EO chips, resized to 120) | 8 (225 patches) | SatMAE procedure: per-band +/-2sigma min-max with fMoW-Sentinel constants -> uint8 -> /255. README example instead uses BATCH stats (deviation!); torchgeo ships Identity | yes in principle (2D-ALiBi, %8, square) but resolution fixed at construction; paper shows 120->504 works |
| Clay v1 base (the bench checkpoint; v1.5 released large only) | 224x224 (v1); v1.5 large = 256 | 8 | per-band z-score on raw DN, constants in configs/metadata.yaml (S2 L2A mean 1105..1835, std 1809..1379); S1 to dB first | YES in original code (sincos regenerated per forward, %8, square, GSD-aware) — the fixed 256 grid is the terratorch port's limitation |
| EarthLoc (CVPR 2024) | 320x320 | CNN (MixVPR over fixed 20x20 grid) | jpeg /255 -> ImageNet stats (radiometry baked into rendered S2 basemap) | NO: MixVPR mixer fixed to 20x20 tokens |

Correction to earlier note: our bench runs Clay v1 BASE (224 pretrain), and
the "fixed 32x32 pos_embed @ 256" is the terratorch port, not the original.
"Fixed-grid" in the sweep = property of the released evaluation-stack
wrapper, not always the upstream model. CROMA normalization has three
mutually inconsistent published versions (paper/SatMAE constants, README
batch stats, torchgeo Identity) — strong material for the preprocessing-
sensitivity argument.

Sources: arXiv:2311.00566 §5/§5.4/appendix + antofuller/CROMA use_croma.py,
README; Clay-foundation/model configs/config.yaml (v1.0 tag: size 224),
configs/metadata.yaml, claymodel/datamodule.py L37-72, model.py L61-88;
arXiv:2403.06758 §5.1 + gmberton/EarthLoc parser.py, apl_model.py.

## TerraMind / Prithvi / OlmoEarth (agents, 2026-08-18)

| model | pretrain size | patch | normalization | variable size |
|---|---|---|---|---|
| TerraMind v1 base/large | 224 (TerraMesh tiles 264) | 16 (196 tokens) | per-band z-score on raw S2 L2A DN; 12 bands no B10; stats in terratorch terramind_register.py (v1_pretraining_mean/std); HF ships no config.json | yes: bicubic pos-embed interp; square, /16 |
| Prithvi-EO v1 100M | 224, 3 timesteps | [1,16,16] | z-score on raw HLS DN; v1 stats full-precision in HF config.json | yes: v1 recomputes sincos exactly |
| Prithvi-EO v2 300M/600M | 224 (tiles 256, random crop), 4 timesteps | 300M [1,16,16]; 600M [1,14,14] | z-score, v2 integer stats (differ from v1) | yes: bicubic interp |
| OlmoEarth v1 nano-large | stored 256@10m; model sees random 1-12 tokens/side at patch 1-8 (up to 96px), token_budget 2250 | FlexiViT patch 1-8 (base 8); downstream default 4 | NOT z-score: per-band min-max over mean+/-2std -> [0,1] (= (x-mean)/(4 std)+0.5), no clip; stats in computed.json | most flexible: trained across sizes; no learned abs pos-embed |

Cross-model: z-scoring OlmoEarth input would be a silent ~4x scale error
(its wrapper correctly applies its own normalizer). TerraMind/Prithvi v2
"pretrain size" (224) differs from their dataset tile sizes (264/256).
Prithvi 600M patch 14 breaks the universal 16.

Sources: arXiv:2504.11171; IBM/terratorch terramind_register.py L199-355,
modality_info.py; HF ibm-nasa-geospatial Prithvi config.json (v1/v2);
arXiv:2412.02732; arXiv:2511.13655 §2.1/§3.1; allenai
olmoearth_pretrain_minimal normalize.py + computed.json.

Addenda (TerraMind/Prithvi/OlmoEarth agent, second pass): Prithvi config
"bands": ["B02".."B07"] are NOT S2 band ids (Blue/Green/Red/NarrowNIR/
SWIR1/SWIR2); OlmoEarth band order is non-monotonic in wavelength — both
invite silent band-ordering bugs. Upstream olmoearth_pretrain dataset.py
silently falls back from computed stats to a hardcoded 0/10000 min-max on
bare except — do not vendor that path.

Addendum: OlmoEarth pretraining crops max out at a 12x12 token grid (96 px);
feeding 224 px at patch 4 gives a 56x56 grid — the 224 cells for OlmoEarth
are extrapolation beyond its pretraining regime, while its native cells are
in-distribution. Relevant when interpreting OlmoEarth's resize contrast.
