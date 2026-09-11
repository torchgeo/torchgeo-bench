# torchgeo-bench

[![CI](https://github.com/torchgeo/torchgeo-bench/actions/workflows/ci.yaml/badge.svg)](https://github.com/torchgeo/torchgeo-bench/actions/workflows/ci.yaml)
[![PyPI version](https://img.shields.io/pypi/v/torchgeo-bench.svg)](https://pypi.org/project/torchgeo-bench/)
[![Python 3.12+](https://img.shields.io/pypi/pyversions/torchgeo-bench.svg)](https://pypi.org/project/torchgeo-bench/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

A lightweight benchmarking framework for evaluating **frozen** geospatial
foundation models on GeoBench V1/V2 and location encoders on CoordBench. Plug
in a backbone or coordinate encoder and run consistent downstream probes through
OmegaConf configs.

- **Frozen-backbone evaluation** — KNN-5, L-BFGS logistic regression, and
  linear / conv / FPN / DPT segmentation probes.
- **GeoBench V1 + V2 built in** — classification and segmentation, RGB or
  full multispectral / multi-modal stacks.
- **CoordBench built in** — coordinate-only regression and classification with
  random, spatial-block, and official holdouts.
- **Config-driven** — sweep models, datasets, partitions, image sizes, and
  bands without code changes.
- **Resumable** — `--resume` skips already-computed `(dataset, method, model, …)` rows. Atomic CSV appends are safe across parallel jobs.
- **Bring your own model** — copy
  [`contrib_template.py`](src/torchgeo_bench/models/contrib_template.py),
  implement `_forward_patch_features`, and add a one-file model config.
  See the [Stage 1 guide](https://torchgeo.org/torchgeo-bench/user/eval_own_model.html)
  for a full walkthrough, or the
  [Stage 2 guide](https://torchgeo.org/torchgeo-bench/user/contribute_model.html)
  to contribute the model back upstream.

## Installation

```bash
pip install torchgeo-bench
```

For development:

```bash
git clone https://github.com/torchgeo/torchgeo-bench
cd torchgeo-bench
uv sync --extra dev
```

On a host limited to CUDA 12.8, install the matching runtime after syncing:

```bash
uv pip install --python .venv/bin/python -r requirements-cu128.txt
```

Use `.venv/bin/python` and `.venv/bin/torchgeo-bench` directly with this overlay; another `uv sync` restores the locked default PyTorch build.

Requires Python 3.12+ and runs on **Linux**, **macOS**, and **Windows**. On
Linux x86_64 the install automatically includes GPU-accelerated FAISS KNN
(CUDA 12, driver R525+, also works on GPU-less machines); other platforms get
CPU FAISS.

## Download a dataset

The runner expects datasets under `./data/`. To grab GeoBench V1:

```bash
torchgeo-bench download geobench_v1
```

To download only the datasets needed for a run:

```bash
torchgeo-bench download geobench_v1 --datasets m-eurosat
```

V1 uses the pickle-free [JSON-metadata mirror](https://huggingface.co/datasets/calebrob6/geobenchv1-webdataset) under `data/classification_v1.0_wds/`, with a pinned revision and archive SHA-256 verification. Re-download existing pickle-based V1 caches; the readers no longer unpickle metadata.

V2 (classification + segmentation) and torchgeo's EuroSAT downloader work the
same way (`torchgeo-bench download geobench_v2`, `torchgeo-bench download eurosat`). See the [documentation](https://torchgeo.org/torchgeo-bench/user/datasets.html)
for all options.

## Run a basic experiment

```bash
# Inspect available presets and datasets without loading ML dependencies
torchgeo-bench models
torchgeo-bench datasets

# A single dataset with a pretrained ImageNet ResNet-50
torchgeo-bench run --model timm/resnet50 --dataset m-eurosat
```

The default device is `cuda:0`. On a machine without a working CUDA GPU (or if
a GPU run crashes — see [troubleshooting](https://torchgeo.org/torchgeo-bench/user/troubleshooting.html)),
fall back to CPU:

```bash
torchgeo-bench run --model rcf --dataset m-eurosat --device cpu
```

Results are appended to `results/models/<model name>.csv`, which **ship pre-populated with reference results**. To start from a clean slate, set `output.file: results/my_run.csv` in a YAML file passed with `--config`. Re-run with `--resume` to skip completed rows. Each evaluation is saved as soon as it finishes, so a later failure does not discard completed metrics.

See `examples/image-run.yaml` for a complete config and `run --config-help` for its schema. The standalone `profile` command writes JSON to stdout; redirect it to a separate file when needed.

The previous interface remains available through `python -m torchgeo_bench.cli` for existing `key=value` scripts, `flops`, and coordinate workflows. Without an explicit `output=`, its profile and intrinsic-dimension rows retain their separate per-model files.

## Handcrafted classification baseline

The handcrafted model adds deterministic spectral and spatial measurements to the ImageStats baseline. `model.level=1`, `2`, and `3` select cumulative feature sets; the feature width depends on the input bands and available spectral indices.

Download the classification datasets and run the level sweep from this checkout:

```bash
.venv/bin/torchgeo-bench download geobench_v1 \
  --datasets m-eurosat,m-forestnet,m-so2sat,m-pv4ger,m-brick-kiln,m-bigearthnet
.venv/bin/torchgeo-bench download geobench_v2 \
  --datasets benv2,treesatai,so2sat,forestnet
.venv/bin/torchgeo-bench download eurosat
.venv/bin/torchgeo-bench download resisc45
.venv/bin/python -m experiments.run_handcrafted
```

The sweep includes ImageStats as level 0 and all 13 registered classification protocols, including multilabel datasets and EuroSAT's spatial split. It uses all bands with identity input normalization, otherwise retaining the normal 224px resize, KNN-5, validation-selected C, train-plus-validation final refit, and 200 bootstrap draws.

Results go to separate `results/models/handcrafted_level*.csv` files and `imagestats_handcrafted_control.csv`. Feature lists and completion status are saved under `outputs/handcrafted/`. Resume skips matching completed rows; a missing linear or KNN result is still reported as a failure.

The [completed sweep](docs/handcrafted-results.md) contains all 104 result rows. Each handcrafted level improves the linear-probe point estimate on 11 of 13 protocols; the largest level is not always the best.

Use `--levels 1 2`, `--datasets eurosat resisc45`, or `--dry-run` for a smaller run. The extractor also works through the normal CLI:

```bash
.venv/bin/torchgeo-bench run model=handcrafted model.level=2 \
  dataset.names=[eurosat] dataset.bands=all dataset.normalization=identity
```

## CoordBench — location encoders

`mode=coord` swaps the image pipeline for a **coordinate-only** track: point
`(lon, lat)` in, a downstream label out. Benchmarks are streamed directly from
the unified [`taylor-geospatial/coordbench`](https://huggingface.co/datasets/taylor-geospatial/coordbench)
HuggingFace dataset (PDFM, SatCLIP, SustainBench, CDC PLACES, MOSAIKS/USAVars,
the DeepMind/AlphaEarth suite, and more — no local download). A frozen encoder
is probed with **KNN** and a **ridge linear** head under **random** or
**spatial-block** cross-validation (regression → R², classification → accuracy).

```bash
# MIND on the full suite, using random and spatial cross-validation
python -m torchgeo_bench.cli run mode=coord model=mind coord.split=both

# One dataset family with the sine/cosine baseline and a linear probe
python -m torchgeo_bench.cli run mode=coord model=sincos coord.names=pdfm coord.methods=[linear]
```

`model=mind` and `model=mind-small` ([MIND](https://huggingface.co/isaaccorley/MIND),
distilled from AlphaEarth/Climplicit/GeoCLIP/SINR) and `model=sincos` work with
the base install. The other pretrained encoders (SatCLIP / GeoCLIP / Climplicit /
SINR, via `rshf`) need the `coordbench` extra:
`pip install "torchgeo-bench[coordbench]"`,
then `model=climplicit` (etc.). Results land in
`results/coordbench_results.csv`. Add your own encoder by subclassing
`LocationEncoder` (implement `_encode`) and pointing a `model` config's
`_target_` at it. See the
[CoordBench guide](https://torchgeo.org/torchgeo-bench/user/coordbench.html)
and the runnable
[`FourierLocationEncoder` example](https://github.com/torchgeo/torchgeo-bench/blob/main/examples/coordbench_location_encoder.py).

<!-- skip-on-docs-landing-start -->

## Learn more

- **[Documentation](https://torchgeo.org/torchgeo-bench/)** — full
  configuration reference, available models, dataset tables, multi-band
  experiments, evaluation methodology, output schema, dev / release
  workflow, and troubleshooting.
- **[AGENTS.md](https://github.com/torchgeo/torchgeo-bench/blob/main/AGENTS.md)**
  — contributor guide and house style.
- **[Cleanlab analysis](projects/cleanlab/README.md)** — standalone dataset auditing, probability extraction, and review galleries, with separate dependencies.

<!-- skip-on-docs-landing-end -->

## Citation

If you use this framework, please cite it (once the `torchgeo-bench` paper is
available):

```bibtex
@misc{torchgeobench,
  title  = {torchgeo-bench: A lightweight benchmarking framework for geospatial foundation models},
  author = {torchgeo-bench Contributors},
  year   = {TBD},
  note   = {Software},
  url    = {https://github.com/torchgeo/torchgeo-bench}
}
```

## License

[MIT](LICENSE).
