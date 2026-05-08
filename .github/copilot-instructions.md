# Copilot instructions for torchgeo-bench

`torchgeo-bench` is a Python 3.12+ benchmarking framework that evaluates frozen
geospatial foundation models on GeoBench V1 (classification) and V2
(classification + segmentation) datasets via KNN-5, linear probing, and
segmentation probes (mIoU), with bootstrapped 95% CIs.

For deeper context see [`AGENTS.md`](../AGENTS.md) (style + dataset list) and
[`METHODOLOGY.md`](../METHODOLOGY.md) (evaluation protocol details).

## Source layout

The Python package lives at **`src/torchgeo_bench/`**. Important pieces:

- `cli.py` / `__main__.py` / `main.py` — `torchgeo-bench` console entry point.
  `cli.py` calls Hydra's `main()` directly in-process (no subprocess); it also
  hosts the `torchgeo-bench download {geobench_v1|geobench_v2|eurosat}`
  subcommand.
- `download.py` — fetches GeoBench V1 / V2 from Hugging Face via
  `snapshot_download`, plus a torchgeo-backed `download_eurosat` helper.
- `conf/` — **Hydra configs are packaged inside the source tree**
  (`src/torchgeo_bench/conf/{config.yaml, model/}`). Add new model configs
  here. There is no `conf/dataset/` directory — every dataset's metadata
  (bands, normalization stats, num_classes, splits) lives in its Python
  wrapper class.
- `models/interface.py` — `BenchModel(nn.Module, ABC)`. Subclasses **must
  implement `forward_patch_features(images, bboxes=None) -> (B, K)`**;
  `forward()` aliases it.
- `models/{bench_models,timm,torchgeo_models,olmoearth}.py` — concrete model
  wrappers registered via Hydra `_target_:` strings.
- `datasets/` — per-dataset wrappers plus three family base classes:
  `_V1Dataset` (in `geobench_v1.py`), `_V2Dataset` (in `geobench_v2.py`), and
  the standalone `EuroSAT` (in `eurosat.py`). Each per-dataset file just
  declares metadata (name, num_classes, bands, rgb_bands, split_sizes) and
  inherits the family's `get_dataset` boilerplate. `datasets/__init__.py`
  exposes `get_datasets`, `get_bench_dataset_class`, and `list_datasets`.
- `geobench_v1.py` — `GeoBenchv1` HDF5 reader (no `geobench` dependency); takes
  exact source band names like `"04 - Red"` (the wrapper translates from short
  canonical names).
- `geobench_v2.py` — `GeoBenchv2` adapter that dispatches to
  `geobench_v2.datasets.GeoBench<X>` upstream classes; wrappers opt into
  multi-modality via `band_order_strategy = "by_sensor"`.
- `linear.py` — custom L-BFGS `LogisticRegression` matching scikit-learn's
  objective scaling (1/n CE + 1/(2nC)·‖W‖²); used for the linear probe sweep.
- `knn.py` — FAISS-CPU KNN classifier (no GPU branch).
- `segmentation_probe.py` / `segmentation_task.py` — hook-based dense feature
  probe + training loop (linear or `conv_block` head).
- `utils.py` — `extract_features` handles dict outputs (`norm`/`global_pool`/
  `head.global_pool` keys) and 3-D ViT outputs (mean-pools tokens).

## Build, test, lint

**Always activate the `torchgeo-bench` conda environment before running any
commands** (or use `conda run -n torchgeo-bench …`). The `Makefile` targets
(`make install/tests/lint/format`) wrap these commands and assume that env.

```bash
conda activate torchgeo-bench                                       # do this first
conda run -n torchgeo-bench uv sync --extra dev                     # install deps + dev tools
conda run -n torchgeo-bench torchgeo-bench run model=timm/resnet50 dataset.names=[m-eurosat]
conda run -n torchgeo-bench pytest                                  # full suite (skips `slow` by default)
conda run -n torchgeo-bench pytest tests/test_geobench_dataset.py -v  # one file
conda run -n torchgeo-bench pytest tests/test_geobench_dataset.py::TestClass::test_method -v
conda run -n torchgeo-bench pytest -k "m-eurosat" -v                # by keyword
conda run -n torchgeo-bench pytest -m slow                          # include integration tests (load real data)
conda run -n torchgeo-bench pytest --no-cov                         # faster iteration (skip coverage)
conda run -n torchgeo-bench ruff check . --fix                      # lint + autofix
conda run -n torchgeo-bench ruff format .                           # format
```

If the env is already activated you can drop the `conda run -n torchgeo-bench`
prefix and call the tools directly (`pytest`, `ruff …`, `torchgeo-bench run …`).

`pyproject.toml` configures pytest with `--cov=torchgeo_bench` and
`-m "not slow"` by default; the `slow` marker is for integration tests that
load real datasets.

Tests skip gracefully if data is missing — they look under `data/` from CWD:
- V1 → `data/classification_v1.0/`
- V2 → `data/geobenchv2/<dataset>/`
- EuroSAT → `data/eurosat/`

Download with `torchgeo-bench download {geobench_v1|geobench_v2|eurosat}`.

## Architecture (the parts you can't see from one file)

1. **Hydra-driven entry point.** `torchgeo-bench run …` mutates `sys.argv` and
   calls the `@hydra.main`-decorated function in-process (no subprocess
   re-launch). Hydra resolves `src/torchgeo_bench/conf/config.yaml`. The
   default model is `rcf`. Override anything from the CLI: `model=timm/resnet50`,
   `dataset.names=[m-eurosat]`, `eval.bootstrap=100`, `device=cuda:1`,
   `resume=true`.
2. **Per-dataset model reinitialization.** Models are instantiated once per
   dataset because `num_channels` varies (RGB vs multispectral). The Hydra
   `model:` config is a partial; `main.py` injects the right `num_channels`
   when calling `instantiate(...)`.
3. **Classification path** (KNN-5 + linear probe): extract train/val/test
   embeddings once → KNN-5 with FAISS + bootstrap CIs → L-BFGS logistic
   regression sweep over `c_range` (log-spaced), pick best on val, refit on
   train+val if `eval.merge_val=true`, evaluate on test with bootstrap CIs.
4. **Segmentation path** (`seg-linear` / `seg-conv_block`): `SegmentationProbe`
   registers forward hooks on configured backbone layers, reshapes features
   (2-D/3-D ViT/4-D), upsamples bilinearly, applies BN+1×1 conv head (or a
   conv block + concat for `conv_block`), trains via `SegmentationSolver`
   (AdamW, CrossEntropy with `ignore_index=255`), evaluates with
   `MulticlassJaccardIndex`. Method labels in CSV: `seg-linear` /
   `seg-conv_block` (matches the resume key).
5. **Datasets are pure-Python wrappers.** Each dataset has
   `src/torchgeo_bench/datasets/<safe_name>.py` (a subclass of `_V1Dataset`,
   `_V2Dataset`, or `BenchDataset` directly for `eurosat`). `safe_name` is the
   dataset name with hyphens → underscores. Class attributes carry every piece
   of metadata: `name`, `task`, `num_classes`, `multilabel`, `bands` (list of
   `BandSpec`), `rgb_bands`, `split_sizes`. Dispatch happens via
   `_REGISTRY` in `datasets/loading.py`. `dataset.names=all` expands via
   `list_datasets()`.
6. **Dataset taxonomy.** Authoritative facts:
   - V1 (`m-` prefix by convention): `m-eurosat` (10), `m-forestnet` (12),
     `m-so2sat` (17, sentinel-2 + SAR, 18 bands), `m-pv4ger` (2, aerial RGB),
     `m-brick-kiln` (2), and `m-bigearthnet` (43, `multilabel=True`).
   - V2 classification: `benv2` (19, S2+SAR), `treesatai` (13, aerial+S2+SAR,
     19 bands), `so2sat` (17, S2+SAR), and `forestnet` (12, S2 6-band).
   - V2 segmentation: `burn_scars`, `caffe` (4, aerial grayscale),
     `cloudsen12`, `dynamic_earthnet`, `flair2`, `fotw`, `kuro_siwo`
     (SAR vv/vh + DEM, no RGB), `pastis` (S2+SAR, 16 bands), `spacenet2`
     (WorldView 8-band + pan), `spacenet7`.
   - torchgeo template: `eurosat` (uses `torchgeo.datasets.EuroSAT`).
   - V1 vs V2 is decided by which family base class the wrapper inherits from
     (`_V1Dataset` vs `_V2Dataset`). `forestnet` exists in both
     (`m-forestnet` is V1, `forestnet` is V2) and they have different sensors
     / band counts.
7. **`num_channels = len(bands)`**, so a model wrapper that hard-codes
   channel counts will break on multi-sensor datasets like `treesatai` (19),
   `pastis` (16), or `m-so2sat` (18). When `dataset.bands=rgb`, the runner
   picks `rgb_bands` by short name — those names differ across V1/V2
   (`red,green,blue` vs `b04,b03,b02` vs `gray` for caffe, `vv,vh` for
   kuro_siwo).
8. **Multilabel path.** When a dataset's wrapper sets `multilabel = True`
   (only `m-bigearthnet` today), `main.py` switches from `accuracy` to
   `micro_mAP` and takes a multilabel KNN/linear branch.
9. **Multi-modality V2 path.** V2 wrappers whose upstream class expects
   `band_order` as a `dict[modality, list[str]]` set
   `band_order_strategy = "by_sensor"`. The `_V2Dataset` base groups
   `BandSpec` entries by sensor and passes `return_stacked_image=True` so
   the upstream class concatenates per-modality tensors into a single
   `image` key.
10. **Sample canonicalization.** A handful of V2 wrappers
    (`KuroSiwo`, `FieldsOfTheWorld`) override `canonicalize_sample()` to
    remap upstream sample keys (collapsing temporal dims, picking
    `image_b` over `image_a`, etc.). The default is a no-op.
11. **Resume + atomic writes.** Results append to a single CSV with `fcntl`
    advisory locking so parallel jobs are safe. With `resume=true`, the
    script reads the CSV and skips any
    `(dataset, method, model, name, normalization, image_size, interpolation, partition)`
    tuple already present.

## Conventions specific to this codebase

- **Python 3.12+ only.** Use modern type hints (`list[str]`, `dict[str, Any]`,
  `X | None`). Do NOT import `List`/`Dict`/`Optional`/`Union` from `typing`.
  Do NOT add `from __future__ import annotations` — use `Self`, quoted
  annotations, or explicit imports for forward references.
- **Datasets always live at `data/<canonical>`.** No env vars, no config
  overrides. V1 → `data/classification_v1.0/<name>`, V2 →
  `data/geobenchv2/<name>`, EuroSAT → `data/eurosat/`. Each family base
  class hard-codes its `data_root()`.
- **No defensive imports for hard deps.** Every package in
  `[project.dependencies]` is guaranteed to be installed. Do **not** wrap
  imports in `try` / `except ImportError` "just in case" — that pattern
  hides real failures behind a fake fallback path. Just import them:

  ```python
  # ❌ BAD
  try:
      from torchgeo.datasets.errors import DatasetNotFoundError
  except ImportError:  # pragma: no cover - older torchgeo versions
      DatasetNotFoundError = FileNotFoundError

  # ✅ GOOD
  from torchgeo.datasets.errors import DatasetNotFoundError
  ```

  Same rule for bare `except Exception:` blocks that swallow errors to
  "keep going" — catch the *specific* exception you expect (e.g.
  `FileNotFoundError`, `DatasetNotFoundError`, `pandas.errors.ParserError`)
  and let unexpected failures crash so they're visible.
- **Logging, not `print()`.** Use `logger = logging.getLogger(__name__)`. The
  benchmark script writes structured logs; `print` interleaves badly.
- **Google-style docstrings.** Pydocstyle is enforced (ruff rule `D`,
  convention `google`). `D104`/`D105`/`D107` are ignored. Tests and `scripts/`
  are exempt from pydocstyle (`per-file-ignores`); tests are also exempt from
  `ARG` (pytest fixture params look syntactically unused).
- **Ruff rules enabled:** `ARG, B, C4, D, E, F, I, SIM, UP, W`. Ignored:
  `B008, B905, D104/105/107, E501`. Line length 100, but `E501` is off because
  `ruff format` handles wrapping.
- **Pre-commit runs `uv lock`** as a local hook — touching `pyproject.toml`
  without re-locking will fail CI. Run `uv lock` (or `pre-commit run -a`)
  after dep changes.
- **Don't add a new model by editing `main.py`.** Implement `BenchModel`
  somewhere importable, then add a `conf/model/<name>.yaml` with
  `_target_: dotted.path.to.YourModel` and any kwargs. `num_channels` is a
  placeholder — the runner overrides it per dataset.
- **Adding a dataset:** add a loader at
  `src/torchgeo_bench/datasets/<safe_name>.py` that subclasses `_V1Dataset`,
  `_V2Dataset`, or `BenchDataset`. Declare the class attributes (`name`,
  `task`, `num_classes`, `multilabel`, `bands`, `rgb_bands`, `split_sizes`).
  For multi-modality V2 datasets also set `band_order_strategy = "by_sensor"`.
  Wire it through `datasets/__init__.py` and add an entry to
  `_REGISTRY` in `datasets/loading.py`. For new V2 datasets that need
  downloads, also add the name to `DEFAULT_V2_DATASETS` in `download.py` and
  add an entry to `_V2_REGISTRY` in `geobench_v2.py`.
- **No `from geobench import …`.** That dependency was removed; use
  `GeoBenchv1` (V1) or the V2 loaders.
- **Don't write refactor-only docs.** Per `AGENTS.md`, internal refactors
  should not produce new markdown files.
