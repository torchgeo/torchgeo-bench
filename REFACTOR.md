# Refactor proposal

The draft PRs below implement individual design choices for review.
The first target is the core image benchmark; coordinate and analysis workflows follow later.
The issues each describe the current code and two alternatives, so people can discuss the choices before we merge them.
Discussion will be collected later; there is no vote tally or automatic decision rule.

Keep the evaluation code readable, make commands quick to discover, and preserve the scientific behavior we already trust.
The current implementation remains the reference while model and dataset families move over individually.

## Constraints from the design discussion

- Keep configuration files; use PyYAML with a real schema.
- Make options discoverable through ordinary CLI help and model details.
- Keep help and catalogs responsive without loading ML frameworks.
- Prefer top-level implementation imports, with a narrow lazy boundary.
- Prefer straightforward constructors and tolerate duplication before refactoring.
- Reconsider features that require disproportionate code or framework workarounds.
- Make datasets independently downloadable; rehost when necessary.
- Keep each implementation PR focused on one adopted decision or model/dataset.
- Keep the current Python 3.12+ baseline for this proposal.

## Drafts for review

Each PR contains a focused implementation and its design notes under `design/refactor/`.
The implementations demonstrate one proposed option; they do not settle the discussion.
All issues and PRs carry the `refactor` label.

| Decision | Discussion | Draft PR | Current scope |
| --- | --- | --- | --- |
| CLI | [#307](https://github.com/torchgeo/torchgeo-bench/issues/307) | [#315](https://github.com/torchgeo/torchgeo-bench/pull/315) | Explicit image commands and a compatibility adapter to the existing runner |
| YAML schema | [#308](https://github.com/torchgeo/torchgeo-bench/issues/308) | [#316](https://github.com/torchgeo/torchgeo-bench/pull/316) | PyYAML loading and strict Pydantic image-run settings |
| Lazy imports | [#309](https://github.com/torchgeo/torchgeo-bench/issues/309) | [#317](https://github.com/torchgeo/torchgeo-bench/pull/317) | A lazy command namespace with ordinary top-level imports in implementations |
| Code style | [#310](https://github.com/torchgeo/torchgeo-bench/issues/310) | [#318](https://github.com/torchgeo/torchgeo-bench/pull/318) | Ruff formatting, annotations, docstrings, complexity checks, and ty for rewritten modules |
| Model construction | [#311](https://github.com/torchgeo/torchgeo-bench/issues/311) | [#319](https://github.com/torchgeo/torchgeo-bench/pull/319) | Explicit timm and RCF constructors; other families retain their current path |
| Normalization | [#312](https://github.com/torchgeo/torchgeo-bench/issues/312) | [#320](https://github.com/torchgeo/torchgeo-bench/pull/320) | Explicit units, band order, statistics, nodata, and a timm embedding comparison |
| Downloads | [#313](https://github.com/torchgeo/torchgeo-bench/issues/313) | [#321](https://github.com/torchgeo/torchgeo-bench/pull/321) | Named dataset downloads using existing backends; no implicit download while loading |
| Profiling | [#314](https://github.com/torchgeo/torchgeo-bench/issues/314) | [#322](https://github.com/torchgeo/torchgeo-bench/pull/322) | Fixed real-batch encoder timing, memory, and optional FLOP counting |

The command PRs are stacked to keep shared code out of each diff:
style → lazy imports → downloads → profiling → schema → image CLI.
Model construction and normalization each build directly on the style PR.
This overview is separate in [#323](https://github.com/torchgeo/torchgeo-bench/pull/323).

The tests include synthetic image evaluation through RCF, KNN, CSV writing, and resume, plus seeded comparisons with existing model wrappers.
Pinned real datasets/checkpoints, physical GPU measurements, and migration of every model's normalization remain follow-up work.
The download draft uses existing mirrors; immutable asset manifests and source-to-mirror equivalence still need verification.
FLOP counts cover registered PyTorch operators and are not certified complete for every model.

## Existing work

As inspected on 2026-09-04, Caleb Robinson has two open implementation PRs:

- [#305: validate model preset schema](https://github.com/torchgeo/torchgeo-bench/pull/305).
- [#306: replace OmegaConf with explicit commands](https://github.com/torchgeo/torchgeo-bench/pull/306),
  stacked on #305 and covering typed dataclasses, PyYAML, and command separation.

These remain separate implementation candidates.
The drafts above leave their branches unchanged. CLI discovery is also discussed in
[#255](https://github.com/torchgeo/torchgeo-bench/issues/255). Normalization discussions
[#259](https://github.com/torchgeo/torchgeo-bench/issues/259) and [#260](https://github.com/torchgeo/torchgeo-bench/issues/260)
remain relevant; opening new design issues does not close those discussions.

Source observations use
[`9c8e4af`](https://github.com/torchgeo/torchgeo-bench/tree/9c8e4afab46675d7279c88828dfcbf0ca99b3a07).
Use this revision for comparison, and record any later change to the reference.

## Oracle and scientific acceptance

Pin the reference source revision, dependency lock, checkpoint identity,
dataset revision/split manifest, configuration, device, and precision. Preserve
a runnable reference environment alongside the rewrite. The current implementation
is a comparison oracle, not proof that every historical behavior is correct.

Capture representative sample IDs, decoded inputs, preprocessed tensors,
embeddings, probe predictions/scores, selected hyperparameters, and final metrics.
Compare each stage with tolerances chosen and documented before interpreting
differences. Do not tune tolerances simply to make a changed result pass.

Cover single-label and multilabel classification, segmentation, RGB,
multispectral, mixed modalities, and temporal inputs. Start with deterministic
CPU fixtures; add pinned real-data/checkpoint smoke cases and representative
accelerator comparisons. Do not require cross-device bitwise equality.

Independently test invariants that an implementation oracle could get wrong:

- Training, validation, and test split membership stays fixed and disjoint.
- Learned preprocessing uses training data only.
- Validation selects hyperparameters; test data never selects the model.
- Train/validation refitting and calibration preserve the declared holdouts.
- Backbones stay frozen and in evaluation mode during probe training.
- Band order, units, image/mask resizing, nodata, and temporal pooling match metadata.
- KNN distance/voting, linear objective/solver, multilabel metrics, ignore labels,
  and class aggregation retain the specified protocol.
- Segmentation uncertainty resamples held-out image-level confusion matrices.
- Non-finite or failed experiments cannot appear as completed valid results.

An intentional scientific correction needs a separate explanation, test, and
protocol version. Preserve old results as historical evidence rather than
rewriting them to appear compatible with the new protocol.

## Results, resume, and interruption

Keep simple, versioned result records and atomic writes with locking where
multiple processes share a destination. Record the resolved experiment and
required artifacts with each completed unit of work. Completion means all
required outputs exist and passed validation, not that some CSV rows exist.

Fingerprint the effective model/checkpoint, dataset and split revision, band and
preprocessing contract, probe settings, seed, and result-affecting execution
settings such as precision and feature-cache dtype. Version the scientific
protocol. Record exact source and dependency provenance. Classify identity fields
explicitly; do not hash arbitrary dictionaries or silently ignore new settings.

Additive profiling/analysis has separate completion and identity. Its switches
cannot force already completed image metrics to rerun. Profiling identity also
includes hardware, batch size, measurement scope, and counter settings.

Do not reinterpret old hashes or mutate historical CSVs. Any result-schema or
resume-identity change needs an explicit migration/invalidation policy and tests.
Test partial writes, interruptions, reruns, and concurrent writers before adoption.

## Implementation sequence

1. Record the configuration, CLI, import, and style decisions. Establish the
   reference fixtures and comparison harness; inspect #305/#306 for reusable work.
2. Deliver one complete classification path from dataset download/loading
   through one straightforward model, KNN/linear probing, and saved results.
3. Add segmentation with explicit spatial features and matched uncertainty.
4. Migrate remaining model and dataset families in separate PRs, including
   multispectral, mixed-modality, and temporal edge cases.
5. Add the adopted narrow profiler independently of accuracy evaluation.
6. Add coordinate and analysis workflows after core image parity. Keep coordinate
   encoders and label-fitted spatial priors separate if both are introduced.

Normalization ownership and adapter contracts must be settled before migrating
unusual models. Dataset storage and preprocessing changes require source-to-new
input parity in addition to final score comparisons.

## Review and validation gates

Every implementation PR identifies its proposed choice, scope, behavior changes, and evidence. Use one logical change per PR; stack only when a real
dependency requires it. A model migration and a new normalization policy should
not be bundled to make a failing comparison disappear.

Use the adopted style/type gate and appropriate offline tests, then representative
end-to-end oracle comparisons. Keep mechanical formatting separate. Document CLI
and public behavior changes. Validate documentation, examples, and diff scope alongside the code.

Help timing is a product requirement. On the inspected local environment, five
fresh-process measurements gave about 42 ms median for help. Use a reference
machine target below 100 ms and subprocess import checks, not a brittle wall-time
threshold on every shared CI runner.

## Deferred work

The first release does not need a sweep scheduler, generic factory/plugin engine,
a universal storage format, automatic OOM recovery, global framework monkeypatches,
or exhaustive profiler decomposition. Revisit each when there is a concrete use
case and a small, reviewable implementation. This proposal does not remove any
current feature from the existing release.
