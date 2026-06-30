# torchgeo-bench

[![CI](https://github.com/torchgeo/torchgeo-bench/actions/workflows/ci.yaml/badge.svg)](https://github.com/torchgeo/torchgeo-bench/actions/workflows/ci.yaml)
[![PyPI version](https://img.shields.io/pypi/v/torchgeo-bench.svg)](https://pypi.org/project/torchgeo-bench/)
[![Python 3.12+](https://img.shields.io/pypi/pyversions/torchgeo-bench.svg)](https://pypi.org/project/torchgeo-bench/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

A lightweight benchmarking framework for evaluating **frozen** geospatial
foundation models on the GeoBench V1 and V2 suites. Plug in any backbone, get
KNN-5 / linear-probe accuracy on classification datasets and mIoU on
segmentation datasets, with bootstrapped 95% confidence intervals — all
configured through Hydra.

- **Frozen-backbone evaluation** — KNN-5, L-BFGS logistic regression, and
  linear / conv / FPN / DPT segmentation probes.
- **GeoBench V1 + V2 built in** — classification and segmentation, RGB or
  full multispectral / multi-modal stacks.
- **Hydra-driven** — sweep models, datasets, partitions, image sizes, and
  bands without code changes.
- **Resumable** — `resume=true` skips already-computed `(dataset, method,
  model, …)` rows. Atomic CSV appends are safe across parallel jobs.
- **Bring your own model** — copy
  [`contrib_template.py`](src/torchgeo_bench/models/contrib_template.py),
  implement `_forward_patch_features`, and add a one-file Hydra config.
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

For GPU-accelerated KNN (Linux + CUDA 12 + glibc ≥ 2.28):

```bash
pip install 'torchgeo-bench[cuda]'
```

Requires Python 3.12+. The default (CPU) install runs on **Linux**, **macOS**,
and **Windows**; GPU-accelerated KNN (the `[cuda]` extra) is Linux-only.

## Download a dataset

The runner expects datasets under `./data/`. To grab GeoBench V1:

```bash
torchgeo-bench download geobench_v1
```

V2 (classification + segmentation) and torchgeo's EuroSAT downloader work the
same way (`torchgeo-bench download geobench_v2`, `torchgeo-bench download
eurosat`). See the [documentation](https://torchgeo.org/torchgeo-bench/user/datasets.html)
for all options.

## Run a basic experiment

```bash
# Default: random convolutional features (RCF) on every available dataset
torchgeo-bench run

# A single dataset with a pretrained ImageNet ResNet-50
torchgeo-bench run model=timm/resnet50 dataset.names=[m-eurosat]
```

The default device is `cuda:0`. On a machine without a working CUDA GPU (or if
a GPU run crashes — see [troubleshooting](https://torchgeo.org/torchgeo-bench/user/troubleshooting.html)),
fall back to CPU:

```bash
torchgeo-bench run dataset.names=[m-eurosat] device=cpu
```

Results are appended to `results/all_results.csv`, which **ships pre-populated
with reference results** — to start from a clean slate, write to your own file
with `output=results/my_run.csv`. Re-run with `resume=true` to skip
already-completed rows.

<!-- skip-on-docs-landing-start -->
## Learn more

- **[Documentation](https://torchgeo.org/torchgeo-bench/)** — full
  configuration reference, available models, dataset tables, multi-band
  experiments, evaluation methodology, output schema, dev / release
  workflow, and troubleshooting.
- **[AGENTS.md](https://github.com/torchgeo/torchgeo-bench/blob/main/AGENTS.md)**
  — contributor guide and house style.
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
