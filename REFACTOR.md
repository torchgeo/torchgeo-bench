# Refactor proposal

Status: proposed, open for discussion and voting. These specifications do not
change runtime behavior or approve any option. The first implementation target
is the core image benchmark; coordinate and analysis workflows follow later.

The objective is a readable evaluation harness with discoverable, fast commands,
validated YAML configuration, explicit model construction, individually
downloadable datasets, and auditable preprocessing. Preserve trusted scientific
behavior while removing unnecessary machinery.

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

Library choices, formatting defaults, normalization ownership, and exact command
names below are recommendations for voting, not previously accepted decisions.

## Decision index

Each issue contains the alternatives and voting instructions. Each draft PR
adds one independent document under `design/refactor/` and targets `main`.
The overview PR adds only this file. Merging a proposal document does not mark
its implementation complete; the issues remain open until the adopted behavior
and acceptance criteria have been delivered.

| ID | Decision | Vote / discussion | Draft specification | Status |
| --- | --- | --- | --- | --- |
| R01 | Choose the refactor CLI command surface | [#307](https://github.com/torchgeo/torchgeo-bench/issues/307) | [#315](https://github.com/torchgeo/torchgeo-bench/pull/315) | Proposed |
| R02 | Choose a validated YAML configuration schema | [#308](https://github.com/torchgeo/torchgeo-bench/issues/308) | [#316](https://github.com/torchgeo/torchgeo-bench/pull/316) | Proposed |
| R03 | Choose lazy command imports while keeping top-level imports | [#309](https://github.com/torchgeo/torchgeo-bench/issues/309) | [#317](https://github.com/torchgeo/torchgeo-bench/pull/317) | Proposed |
| R04 | Choose a human-readable coding style and enforcement gate | [#310](https://github.com/torchgeo/torchgeo-bench/issues/310) | [#318](https://github.com/torchgeo/torchgeo-bench/pull/318) | Proposed |
| R05 | Choose explicit model constructors and feature contracts | [#311](https://github.com/torchgeo/torchgeo-bench/issues/311) | [#319](https://github.com/torchgeo/torchgeo-bench/pull/319) | Proposed |
| R06 | Choose normalization ownership and explicit input units | [#312](https://github.com/torchgeo/torchgeo-bench/issues/312) | [#320](https://github.com/torchgeo/torchgeo-bench/pull/320) | Proposed |
| R07 | Choose individual dataset downloads and canonical storage | [#313](https://github.com/torchgeo/torchgeo-bench/issues/313) | [#321](https://github.com/torchgeo/torchgeo-bench/pull/321) | Proposed |
| R08 | Choose a narrow and reproducible profiling command | [#314](https://github.com/torchgeo/torchgeo-bench/issues/314) | [#322](https://github.com/torchgeo/torchgeo-bench/pull/322) | Proposed |

Comment `Vote: A` or `Vote: B` with a short rationale in the relevant issue.
Propose an amendment if neither option fits. A reaction expresses interest,
not a vote for a particular option. Maintainers record the outcome and update
the document; no deadline or automatic majority rule is assumed.

## Existing work

As inspected on 2026-09-04, Caleb Robinson has two open implementation PRs:

- [#305: validate model preset schema](https://github.com/torchgeo/torchgeo-bench/pull/305).
- [#306: replace OmegaConf with explicit commands](https://github.com/torchgeo/torchgeo-bench/pull/306),
  stacked on #305 and covering typed dataclasses, PyYAML, and command separation.

Treat these as implementation candidates and reuse accepted work. The decision
PRs do not alter, supersede, or approve them. CLI discovery is also discussed in
[#255](https://github.com/torchgeo/torchgeo-bench/issues/255). Normalization discussions
[#259](https://github.com/torchgeo/torchgeo-bench/issues/259) and [#260](https://github.com/torchgeo/torchgeo-bench/issues/260)
remain relevant; opening new design issues does not close those discussions.

Source observations use
[`9c8e4af`](https://github.com/torchgeo/torchgeo-bench/tree/9c8e4afab46675d7279c88828dfcbf0ca99b3a07).
Refresh source, open PRs, and reference artifacts before starting implementation.

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

Every implementation PR names its adopted decision, scope, intentional behavior
changes, and evidence. Use one logical change per PR; stack only when a real
dependency requires it. A model migration and a new normalization policy should
not be bundled to make a failing comparison disappear.

Use the adopted style/type gate and appropriate offline tests, then representative
end-to-end oracle comparisons. Keep mechanical formatting separate. Document CLI
and public behavior changes. For proposal-only PRs, validate Markdown, links,
embedded examples, and diff scope; runtime acceptance remains future work.

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
