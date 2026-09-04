# R08 - Choose a narrow and reproducible profiling command

Status: implementation started in the profile worktree; GPU oracle validation remains pending.
Decision issue: https://github.com/torchgeo/torchgeo-bench/issues/314

Measure one explicit inference configuration and expose unsupported FLOP counting without global framework patches.

## Decision

Should profiling focus on inference measurements or retain full backbone/head/probe decomposition initially?

| Option | Choice | Tradeoff |
| --- | --- | --- |
| A | Narrow inference profiler (recommended) | Parameters, timing, supported memory measurements, and optional FLOPs for one configuration. |
| B | Detailed decomposition from the start | Include backbone, segmentation heads, and probes; more interfaces and validation required. |

Discuss the two scopes in the linked issue with a concrete rationale or amendment. The recommendation remains provisional until maintainers record the decision.

## Proposed contract

Measure one model, input configuration, device, precision, and batch size per invocation. A dataset can provide real metadata and a representative batch. Synthetic inputs must identify units, shape, and generated values; shape alone does not ensure equivalence for data-dependent models.

Report parameters, warmed-up latency, throughput, and supported device-memory measurements. Define included operations. The initial default measures preprocessing plus encoder forward on a batch already on the device, excluding data loading and host transfer. Use eval mode and the recorded inference/precision policy.

Synchronize accelerators around timed work and distinguish synchronized per-batch latency from sustained throughput. Record warmup/measurement counts, actual batch size, image/time/channel shape, hardware, software, precision, scope, and config identity. Distinguish allocated and reserved memory.

Make FLOPs optional. Record counter identity, multiply-add convention, and known operator coverage. Unsupported counting is null with a reason, never zero or an invented total. Check supported small models against analytical counts. Unexpected execution failures must still propagate.

Avoid global PyTorch gradient-hook patches, execution-mode retry ladders, and exception-message classifiers in the common profiler. If a model requires substantial special machinery, revisit whether its FLOP count belongs in scope.

Keep batch size fixed; OOM fails the requested measurement. CPU profiling is a separate invocation. Do not promise hard timeouts through elapsed-time checks after a forward pass. Do not construct a segmentation training solver to measure an encoder.

Store profile results separately with independent completion and identity. Successful timing may coexist with a known unavailable FLOP count if per-metric status is explicit. An unexpected failure cannot produce a successfully completed profile record.

## Acceptance criteria for implementation

- [x] CPU fixture tests verify timing boundaries, fixed batch size, and JSON metadata.
- [x] Profiling does not change model parameters or leave global hooks; unsupported counting has explicit status.
- [x] Invalid iteration counts and unsupported devices fail without changing requested settings.
- [ ] GPU fixture tests and real hardware oracle checks remain pending.
- [ ] Resume distinguishes hardware, input, precision, batch size, and counter changes without invalidating image metrics.

## Review boundary

Standalone inference profiling and its results. Option A defers head decomposition, automatic batch search, energy, hard process timeouts, and profiling sweeps.

## Initial implementation

The standalone `profile` command measures one real batch from one selected
dataset. It uses `profile_inference` directly, keeps the requested batch size
fixed, and emits one JSON record containing model, dataset, bands, shape,
hardware, software, precision, scope, timing, memory, and parameter metadata.
FLOP counting is opt-in and carries the counter status and coverage metadata;
disabled or partial counts are never represented as zero or as a complete
total. The legacy `flops` command remains separate for compatibility.

## Existing work and references

Compute-definition discussion: [#121](https://github.com/torchgeo/torchgeo-bench/issues/121). Existing CLI separation: [#306](https://github.com/torchgeo/torchgeo-bench/pull/306). Preserve historical profile files.

Source observations are pinned to `9c8e4af`; refresh before implementation.

- [Global hook patch](https://github.com/torchgeo/torchgeo-bench/blob/9c8e4afab46675d7279c88828dfcbf0ca99b3a07/src/torchgeo_bench/model_profile.py#L33)
- [CPU time budget](https://github.com/torchgeo/torchgeo-bench/blob/9c8e4afab46675d7279c88828dfcbf0ca99b3a07/src/torchgeo_bench/model_profile.py#L237)
- [Current orchestration](https://github.com/torchgeo/torchgeo-bench/blob/9c8e4afab46675d7279c88828dfcbf0ca99b3a07/src/torchgeo_bench/flops_pipeline.py#L1)
