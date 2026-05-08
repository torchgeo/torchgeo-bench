# torchgeo-bench

[![CI](https://github.com/torchgeo/torchgeo-bench/actions/workflows/ci.yaml/badge.svg)](https://github.com/torchgeo/torchgeo-bench/actions/workflows/ci.yaml)
[![PyPI version](https://img.shields.io/pypi/v/torchgeo-bench.svg)](https://pypi.org/project/torchgeo-bench/)
[![Python 3.12+](https://img.shields.io/pypi/pyversions/torchgeo-bench.svg)](https://pypi.org/project/torchgeo-bench/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

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
- **Bring your own model** — implement `BenchModel._forward_patch_features`
  and add a one-file Hydra config.

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

Requires Python 3.12+.

## Download a dataset

The runner expects datasets under `./data/`. To grab GeoBench V1:

```bash
torchgeo-bench download geobench_v1
```

V2 (classification + segmentation) and torchgeo's EuroSAT downloader work the
same way (`torchgeo-bench download geobench_v2`, `torchgeo-bench download
eurosat`). See the [documentation](https://torchgeo-bench.readthedocs.io/en/latest/user/datasets.html)
for all options.

## Run a basic experiment

```bash
# Default: random convolutional features (RCF) on every available dataset
torchgeo-bench run

# A single dataset with a pretrained ImageNet ResNet-50
torchgeo-bench run model=timm/resnet50 dataset.names=[m-eurosat]
```

Results are appended to `results/all_results.csv`. Re-run with `resume=true`
to skip already-completed rows.

## Learn more

- **[Documentation](https://torchgeo-bench.readthedocs.io/)** — full
  configuration reference, available models, dataset tables, multi-band
  experiments, output schema, dev / release workflow, and
  troubleshooting.
- **[METHODOLOGY.md](https://github.com/torchgeo/torchgeo-bench/blob/main/METHODOLOGY.md)**
  — formal description of the KNN, linear-probe, and segmentation-probe
  protocols.
- **[AGENTS.md](https://github.com/torchgeo/torchgeo-bench/blob/main/AGENTS.md)**
  — contributor guide and house style.

## Citation

If you use this framework, please cite the GeoBench paper:

```bibtex
@article{lacoste2023geobench,
  title   = {GEO-Bench: Toward Foundation Models for Earth Monitoring},
  author  = {Lacoste, Alexandre and Lehmann, Nils and others},
  journal = {NeurIPS Datasets and Benchmarks Track},
  year    = {2023}
}
```

## License

[MIT](LICENSE).
