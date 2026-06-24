## Add model: <!-- model name here -->

Use this template for PRs that add a new model and contribute its benchmark
rows to `results/all_results.csv`.

Docs:

- [Evaluate your own model](https://torchgeo.org/torchgeo-bench/user/eval_own_model.html)
- [Contribute a model](https://torchgeo.org/torchgeo-bench/user/contribute_model.html)
- [Datasets](https://torchgeo.org/torchgeo-bench/user/datasets.html)
- [Results format](https://torchgeo.org/torchgeo-bench/user/results-format.html)

### 1. Model summary

| Field | Value |
|-------|-------|
| **Model name** | <!-- e.g. `new_model` as it appears in the results CSV --> |
| **Class** | <!-- e.g. `torchgeo_bench.models.NewModel` --> |
| **Hydra config** | <!-- e.g. `src/torchgeo_bench/conf/model/new_model.yaml` --> |
| **Pretraining data** | <!-- e.g. Sentinel-2 global, fMoW, ImageNet, etc. --> |
| **Sensor coverage** | <!-- e.g. S2 RGB, S2 all-bands, NAIP RGB, multi-sensor --> |
| **Weights URL** | <!-- Hugging Face Hub repo, release asset, or equivalent public URL --> |
| **Paper / project page** | <!-- URL if available, otherwise N/A --> |
| **Required extra** | <!-- e.g. `newmodel`, or N/A if no new extra is needed --> |

### 2. Add the model

Follow the Stage 2 guide: [Contribute a model](https://torchgeo.org/torchgeo-bench/user/contribute_model.html).

- [ ] Class inherits `BenchModel` and implements `_forward_patch_features(images) -> (B, K)`.
- [ ] Class is exported from `src/torchgeo_bench/models/__init__.py` and listed in `__all__`.
- [ ] Hydra config exists at `src/torchgeo_bench/conf/model/<name>.yaml` with the correct `_target_`.
- [ ] Model weights are publicly accessible without authentication.
- [ ] Optional dependencies are declared under `[project.optional-dependencies]` in `pyproject.toml`, if needed.
- [ ] Tests cover all added code in `tests/test_<model>.py`.
- [ ] Fast tests use random tensors and no network I/O.
- [ ] Weight-download tests are marked `@pytest.mark.slow`.

### 3. Run the model on every dataset

Set up the environment and download the benchmark data:

```bash
conda activate torchgeo-bench
uv sync --extra dev
torchgeo-bench download geobench_v1
torchgeo-bench download geobench_v2
torchgeo-bench download eurosat
```

Run `torchgeo-bench` with only this model selected. Use `resume=true` so an
interrupted run can continue without duplicating completed rows.

```bash
torchgeo-bench run model=<model_config_name> \
  dataset.names=all \
  output=results/all_results.csv \
  resume=true
```

If a dataset cannot run because the model does not support that sensor or
modality, list it here with the exact error or reason. Otherwise write `None`.

| Skipped dataset | Reason |
|-----------------|--------|
| <!-- e.g. kuro_siwo --> | <!-- e.g. SAR + DEM unsupported by this RGB-only model --> |

### 4. Commit results

The goal is for maintainers to review and merge the model code and its
benchmark rows together.

- [ ] New rows are committed to `results/all_results.csv`.
- [ ] Added result rows are only for this model and the command above.
- [ ] No existing `results/all_results.csv` rows were reordered, edited, or removed.
- [ ] Result rows match the documented [CSV schema](https://torchgeo.org/torchgeo-bench/user/results-format.html).
- [ ] Added row count: <!-- e.g. 42 rows -->

### 5. Reproduction details

Maintainers should be able to check out this PR and rerun the exact benchmark.

```bash
git checkout <this-branch>
conda activate torchgeo-bench
uv sync --extra dev
torchgeo-bench run model=<model_config_name> dataset.names=all output=results/all_results.csv resume=true
```

Hardware and software used for submitted results:

| Field | Value |
|-------|-------|
| **Commit SHA** | <!-- `git rev-parse HEAD` --> |
| **OS** | <!-- e.g. Ubuntu 24.04 --> |
| **CPU/GPU** | <!-- e.g. A100 80GB, M3 Max, CPU-only --> |
| **Python** | <!-- `python --version` --> |
| **torchgeo-bench command** | <!-- paste the exact command used --> |

### 6. Local checks

- [ ] `pytest --no-cov tests/test_<model>.py` passes locally.
- [ ] `pytest --no-cov -m slow tests/test_<model>.py` passes locally, if pretrained weights were added.
- [ ] `pytest --no-cov` passes locally.
- [ ] `ruff check . && ruff format --check .` passes locally.
