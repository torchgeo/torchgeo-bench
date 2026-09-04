# R05 - Choose explicit model constructors and feature contracts

Status: proposed; no option has been accepted. This file changes no runtime behavior.
Decision issue: https://github.com/torchgeo/torchgeo-bench/issues/311

Keep wrappers understandable in one file and accept duplication before introducing abstractions.

## Decision

Should model selection use explicit constructor dispatch or a small registry of loader callables?

| Option | Choice | Tradeoff |
| --- | --- | --- |
| A | Explicit constructors (recommended) | Named branches call concrete constructors; some duplication with visible argument flow. |
| B | Small callable registry | Less dispatch branching; explicit loaders with no decorators or discovery framework. |

Vote in the linked issue. Comment `Vote: A` or `Vote: B` with a short rationale. If neither fits, propose a concrete amendment. Recommendations are proposals, not recorded votes or maintainer approval. Maintainers will summarize the outcome in the issue. Reactions indicate interest, not a choice between options.

## Proposed contract

Use ordinary `nn.Module` adapters with explicit, annotated constructors. The selected adapter owns checkpoint loading, supported bands, channel mapping, and upstream quirks. A small dispatcher calls concrete constructors with named arguments after validation. Avoid signature introspection, arbitrary `_target_` imports, discarded kwargs, and registration side effects.

Each image adapter exposes pooled embeddings `(B, D)`. Segmentation-capable adapters also expose ordered spatial feature maps with documented channels and strides. State the frozen-backbone/eval contract explicitly. Reject segmentation for a model with no spatial-feature contract before head training.

Prefer direct spatial-feature methods. When upstream models require hooks, keep selection, token reshaping, and cleanup in the adapter. The runner cannot inspect private `_features`, guess layer names, or branch on model class names. Temporal adapters state whether they encode sequences jointly or dates independently; pooling is recorded in the protocol.

Help metadata lives in a lightweight catalog. Model-specific schema fields map explicitly to constructor arguments. Validate checkpoint identity and input compatibility. Preserve random/untrained baselines as named choices rather than fallback paths.

The normalization decision assigns preprocessing one owner. Adapters declare native requirements or expose a native operator; they cannot silently apply a second policy. Embedding pooling/L2 normalization remains separate from image preprocessing.

Adding a model should require one understandable adapter, catalog/schema entries, and focused tests. Allow duplicated construction code. Extract helpers only when concrete wrappers demonstrate identical behavior and simpler call sites.

## Acceptance criteria for implementation

- [ ] A representative CNN and transformer match reference embeddings, pooling, band order, and spatial features.
- [ ] Unsupported bands and missing spatial features produce explicit errors rather than silent adaptation.
- [ ] The backbone stays frozen and in eval mode while probe parameters train.
- [ ] Adding a model does not require editing evaluation loops or profiling logic.
- [ ] Optional dependencies load only for the selected model.

## Review boundary

Dispatch, adapter interfaces, capabilities, and a representative migration. Port other model families separately; do not rewrite all adapters in one PR.

This draft contains only this decision document. Implementation must pass
the oracle and migration requirements described in the refactor overview;
adding this document does not satisfy the criteria above.

## Existing work and references

Existing discussions: [#219](https://github.com/torchgeo/torchgeo-bench/issues/219), [#258](https://github.com/torchgeo/torchgeo-bench/issues/258), and [#16](https://github.com/torchgeo/torchgeo-bench/issues/16). Preserve accepted fixes.

Source observations are pinned to `9c8e4af`; refresh before implementation.

- [Current interface](https://github.com/torchgeo/torchgeo-bench/blob/9c8e4afab46675d7279c88828dfcbf0ca99b3a07/src/torchgeo_bench/models/interface.py#L1)
- [Current segmentation probe](https://github.com/torchgeo/torchgeo-bench/blob/9c8e4afab46675d7279c88828dfcbf0ca99b3a07/src/torchgeo_bench/segmentation_probe.py#L1)
