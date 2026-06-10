## Add model: <!-- model name here -->

### Model summary

| Field | Value |
|-------|-------|
| **Name** | <!-- e.g. `NewModel` --> |
| **Pretraining data** | <!-- e.g. Sentinel-2 global, fMoW, etc. --> |
| **Sensor coverage** | <!-- e.g. S2 RGB, S2 all-bands, NAIP, multi-sensor --> |
| **Weight hosting URL** | <!-- HuggingFace Hub repo or equivalent --> |
| **Paper / project page** | <!-- URL if available, otherwise N/A --> |

### Checklist

- [ ] Class inherits `BenchModel` and implements `_forward_patch_features(images) -> (B, K)`
- [ ] Class is exported from `src/torchgeo_bench/models/__init__.py` and listed in `__all__`
- [ ] Hydra config exists at `src/torchgeo_bench/conf/model/<name>.yaml` with correct `_target_`
- [ ] Model weights are publicly accessible without authentication (HuggingFace Hub preferred)
- [ ] Optional dependencies declared as an extra in `pyproject.toml` (`[project.optional-dependencies]`)
- [ ] Full test coverage of all added code in `tests/test_<model>.py`
- [ ] Fast tests use random tensors (no network I/O); weight-download tests are marked `@pytest.mark.slow`
- [ ] `pytest --no-cov tests/test_<model>.py` passes locally
- [ ] `pytest --no-cov` (full suite) passes locally
- [ ] Benchmark run on all applicable datasets; results written to `results/contributed/<model_name>.csv`
- [ ] Skipped datasets (sensor coverage mismatch) documented in this PR description with reason
- [ ] `ruff check . && ruff format --check .` passes
