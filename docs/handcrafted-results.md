# Handcrafted baseline results

The sweep covers all 13 image-classification protocols: ten single-label and three multilabel. Every model/dataset pair has both KNN-5 and linear-probe results, for 104 completed metric rows.

All runs use all available bands, identity input normalization, the benchmark's 224px bilinear resize, seed 0, the default partitions and 40-point C grid, and 200 bootstrap resamples. C is selected on validation; the final linear model is fit on train plus validation. Each model/dataset pair runs in its own seeded subprocess.

The levels are cumulative. Level 1 adds percentiles and available sensor-local NDVI, NDWI, NDBI, and NBR maps to per-channel statistics. Level 2 adds multiscale gradient, coherence, and orientation summaries. Level 3 adds local binary patterns, Harris corners, quadrant summaries, and weighted region shape. There are no learned extractor weights.

## Linear probes

Scores are percentages. Multilabel rows use micro-average precision, not accuracy.

| Dataset | Metric | ImageStats | Level 1 | Level 2 | Level 3 |
|---|---|---:|---:|---:|---:|
| benv2 | micro-AP | 68.84 | 69.47 | 71.49 | 71.25 |
| eurosat | accuracy | 88.41 | 90.93 | 93.04 | 93.24 |
| eurosat-spatial | accuracy | 86.17 | 87.91 | 90.04 | 89.89 |
| forestnet | accuracy | 41.79 | 45.42 | 51.56 | 47.83 |
| m-bigearthnet | micro-AP | 46.95 | 49.54 | 51.51 | 51.34 |
| m-brick-kiln | accuracy | 94.99 | 95.10 | 95.10 | 94.79 |
| m-eurosat | accuracy | 85.20 | 88.20 | 90.00 | 87.80 |
| m-forestnet | accuracy | 42.20 | 45.02 | 51.56 | 47.03 |
| m-pv4ger | accuracy | 87.29 | 88.89 | 89.59 | 92.19 |
| m-so2sat | accuracy | 24.75 | 32.96 | 36.11 | 47.77 |
| resisc45 | accuracy | 35.87 | 42.68 | 53.35 | 52.83 |
| so2sat | accuracy | 45.33 | 53.65 | 55.68 | 56.69 |
| treesatai | micro-AP | 51.87 | 46.84 | 47.24 | 44.47 |

Levels 1 and 2 have higher linear-probe point estimates than ImageStats on 12 of 13 protocols; TreeSatAI is the exception. Level 3 improves 11 of 13, with TreeSatAI and m-brick-kiln as the exceptions. More complexity is not always better: level 2 beats level 3 on both ForestNet versions, m-eurosat, and RESISC45, among others.

These are comparisons of point estimates, not paired significance tests or claims that a test-selected level is universally best. Raw, unscaled feature coordinates also affect KNN distances and linear-probe regularization; these runs do not add a separate fitted feature scaler.

## Outputs and reproduction

Full KNN and linear metrics, confidence intervals, feature dimensions, selected C values, and config hashes are stored in:

- `results/models/imagestats_handcrafted_control.csv`
- `results/models/handcrafted_level1.csv`
- `results/models/handcrafted_level2.csv`
- `results/models/handcrafted_level3.csv`

The submitted rows were measured from source commit `74051ef8deac1b77f32b1d60f266bd427ffeb001`, based on upstream `d5168ad71447b06ff2519e8aa8a0dc67c40724e2` with the CUDA-context fix in [torchgeo/torchgeo-bench#342](https://github.com/torchgeo/torchgeo-bench/pull/342). Every model/dataset pair was rerun after updating to that upstream revision and refreshing V1 data from its pinned JSON-metadata mirror. Earlier measurements are not mixed into these CSVs.

Feature names, actual spectral-band mappings, skipped indices, runtime settings, and the combined summary from this run are saved locally under `outputs/handcrafted-pr/run/`. The runner regenerates these artifacts; they are not needed as inputs. The feature widths vary by dataset: RGB has 27/63/111 outputs, full Sentinel-2 has 153/357/629, and mixed-sensor TreeSatAI has 225/525/925.

The host ran Microsoft Azure Linux 3.0, Python 3.13.13, PyTorch 2.11.0 with CUDA 12.8, torchvision 0.26.0, TorchGeo 0.10.0, and faissknn 0.4.1. Both visible GPUs were H100 NVL MIG 3g.47gb instances. The CUDA 12.8 overlay in `requirements-cu128.txt` was installed after `uv sync` because the host driver cannot run the lockfile's CUDA 13 build.

From a prepared checkout, repeat the measured device assignments into fresh output files:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
.venv/bin/python -m experiments.run_handcrafted --levels 0 1 --devices 0 \
  --output-dir results/handcrafted-rerun --run-dir outputs/handcrafted-rerun
.venv/bin/python -m experiments.run_handcrafted --levels 2 3 --devices 1 \
  --output-dir results/handcrafted-rerun --run-dir outputs/handcrafted-rerun
.venv/bin/python -m experiments.run_handcrafted --report-only \
  --output-dir results/handcrafted-rerun --run-dir outputs/handcrafted-rerun
```

The two benchmark commands can run concurrently. ImageStats and level 1 use CUDA 0; levels 2 and 3 use CUDA 1. Config hashes include the device, so use the same assignments when resuming the submitted rows. On a single-GPU host, use `--devices 0` for both commands and a new output directory.

The KNN wrapper enters the requested CUDA device context for every submitted run. The ImageStats control is contributed separately in [torchgeo/torchgeo-bench#349](https://github.com/torchgeo/torchgeo-bench/pull/349); the model contribution adds one results file for each of its three presets.
