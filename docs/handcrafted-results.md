# Handcrafted baseline results

The sweep covers all 13 image-classification protocols: ten single-label and three multilabel. Every model/dataset pair has both KNN-5 and linear-probe results, for 104 completed metric rows.

All runs use all available bands, identity input normalization, the benchmark's 224px bilinear resize, seed 0, the default partitions and 40-point C grid, and 200 bootstrap resamples. C is selected on validation; the final linear model is fit on train plus validation. Each model/dataset pair runs in its own seeded subprocess.

The levels are cumulative. Level 1 adds percentiles and available sensor-local NDVI, NDWI, NDBI, and NBR maps to per-channel statistics. Level 2 adds multiscale gradient, coherence, and orientation summaries. Level 3 adds local binary patterns, Harris corners, quadrant summaries, and weighted region shape. There are no learned extractor weights.

## Linear probes

Scores are percentages. Multilabel rows use micro-average precision, not accuracy.

| Dataset | Metric | ImageStats | Level 1 | Level 2 | Level 3 |
|---|---|---:|---:|---:|---:|
| benv2 | micro-AP | 67.57 | 69.45 | 71.17 | 71.32 |
| eurosat | accuracy | 89.07 | 91.04 | 92.48 | 91.26 |
| eurosat-spatial | accuracy | 86.83 | 88.15 | 89.81 | 89.93 |
| forestnet | accuracy | 43.00 | 46.02 | 51.96 | 48.74 |
| m-bigearthnet | micro-AP | 47.18 | 49.31 | 51.47 | 51.70 |
| m-brick-kiln | accuracy | 95.20 | 95.10 | 95.10 | 94.99 |
| m-eurosat | accuracy | 86.20 | 88.40 | 90.50 | 88.00 |
| m-forestnet | accuracy | 42.40 | 44.51 | 51.66 | 47.63 |
| m-pv4ger | accuracy | 87.49 | 88.69 | 91.19 | 92.49 |
| m-so2sat | accuracy | 25.76 | 32.56 | 33.87 | 44.22 |
| resisc45 | accuracy | 35.75 | 42.25 | 53.97 | 53.70 |
| so2sat | accuracy | 45.13 | 55.38 | 56.29 | 56.39 |
| treesatai | micro-AP | 53.96 | 47.37 | 47.50 | 49.52 |

Each handcrafted level has a higher linear-probe point estimate than ImageStats on 11 of 13 protocols. The exceptions are TreeSatAI and m-brick-kiln. More complexity is not always better: level 2 beats level 3 on EuroSAT, both ForestNet versions, m-eurosat, and RESISC45.

These are comparisons of point estimates, not paired significance tests or claims that a test-selected level is universally best. Raw, unscaled feature coordinates also affect KNN distances and linear-probe regularization; these runs do not add a separate fitted feature scaler.

## Outputs and reproduction

Full KNN and linear metrics, confidence intervals, feature dimensions, selected C values, and config hashes are stored in:

- `results/models/imagestats_handcrafted_control.csv`
- `results/models/handcrafted_level1.csv`
- `results/models/handcrafted_level2.csv`
- `results/models/handcrafted_level3.csv`

Feature names, actual spectral-band mappings, skipped indices, runtime settings, and the combined summary are saved locally under `outputs/handcrafted/`. The feature widths vary by dataset: RGB has 27/63/111 outputs, full Sentinel-2 has 153/357/629, and mixed-sensor TreeSatAI has 225/525/925.

From a prepared checkout:

```bash
.venv/bin/python -m experiments.run_handcrafted --output-dir results/handcrafted-rerun
```

The measured sweep used CUDA 0 for ImageStats and levels 1/3, and CUDA 1 for level 2. Resume on those existing result files with the same device assignments, or choose a new output directory for a new run. The host used PyTorch 2.11 with the CUDA 12.8 runtime overlay because its driver cannot run the lockfile's CUDA 13 build.

The sweep exposed an existing FAISS/Torch stream-context bug on nondefault GPUs. The KNN wrapper now enters the requested CUDA device context; large single-label and multilabel tests match CPU results. All level-2 rows from the affected initial run were excluded and that level was rerun completely. The initial control was also rerun to match the per-dataset process boundary used by the handcrafted levels.
