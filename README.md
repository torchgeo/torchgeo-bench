# torchgeo-bench

[![CI](https://github.com/torchgeo/torchgeo-bench/actions/workflows/ci.yaml/badge.svg)](https://github.com/torchgeo/torchgeo-bench/actions/workflows/ci.yaml)
[![PyPI version](https://img.shields.io/pypi/v/torchgeo-bench.svg)](https://pypi.org/project/torchgeo-bench/)
[![Python 3.12+](https://img.shields.io/pypi/pyversions/torchgeo-bench.svg)](https://pypi.org/project/torchgeo-bench/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%20License%202.0-blue)](LICENSE)

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

For GPU-accelerated KNN (Linux + CUDA 12 + glibc ≥ 2.28):

```bash
pip install 'torchgeo-bench[cuda]'
```

Requires Python 3.12+.

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

Results are appended to `results/all_results.csv`. Re-run with `resume=true`
to skip already-completed rows.

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

If you use this framework, please cite the underlying GeoBench papers
(and, once available, the `torchgeo-bench` paper itself):

```bibtex
@misc{torchgeobench,
  title  = {torchgeo-bench: A lightweight benchmarking framework for geospatial foundation models},
  author = {torchgeo-bench Contributors},
  year   = {TBD},
  note   = {Software},
  url    = {https://github.com/torchgeo/torchgeo-bench}
}

@misc{lacoste2023geobench,
  title         = {GEO-Bench: Toward Foundation Models for Earth Monitoring},
  author        = {Alexandre Lacoste and Nils Lehmann and Pau Rodriguez and Evan David Sherwin and
                   Hannah Kerner and Bj{\"o}rn L{\"u}tjens and Jeremy Andrew Irvin and David Dao and
                   Hamed Alemohammad and Alexandre Drouin and Mehmet Gunturkun and Gabriel Huang and
                   David Vazquez and Dava Newman and Yoshua Bengio and Stefano Ermon and Xiao Xiang Zhu},
  year          = {2023},
  eprint        = {2306.03831},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2306.03831},
  doi           = {10.48550/arXiv.2306.03831}
}

@misc{simumba2025geobench2,
  title         = {{GEO-Bench-2}: From Performance to Capability, Rethinking Evaluation in Geospatial AI},
  author        = {Naomi Simumba and Nils Lehmann and Paolo Fraccaro and Hamed Alemohammad and
                   Geeth De Mel and Salman Khan and Manil Maskey and Nicolas Longepe and
                   Xiao Xiang Zhu and Hannah Kerner and Juan Bernabe-Moreno and Alexandre Lacoste},
  year          = {2025},
  eprint        = {2511.15658},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2511.15658},
  doi           = {10.48550/arXiv.2511.15658}
}
```

## License

[MIT](LICENSE).
