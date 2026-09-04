# R07 - Choose individual dataset downloads and canonical storage

Status: proposed; implementation draft exists on `docs/refactor-downloads-decision`; no option has been accepted.
Decision issue: https://github.com/torchgeo/torchgeo-bench/issues/313

Make download NAME install exactly the versioned representation read by run --dataset NAME.

## Decision

Should each dataset have one canonical downloadable representation or multiple first-class layouts?

| Option | Choice | Tradeoff |
| --- | --- | --- |
| A | One representation per dataset (recommended) | Simple download/load contract; mirror or repackage unsuitable upstream collections. |
| B | Multiple supported representations | Retain upstream and mirror layouts explicitly; more compatibility and parity tests. |

Vote in the linked issue. Comment `Vote: A` or `Vote: B` with a short rationale. If neither fits, propose a concrete amendment. Recommendations are proposals, not recorded votes or maintainer approval. Maintainers will summarize the outcome in the issue. Reactions indicate interest, not a choice between options.

## Proposed contract

`download m-eurosat burn_scars` downloads those datasets individually. `run --dataset NAME` reads the representation installed by that command. Keep the current `./data/<canonical-subdir>/` convention; this decision does not introduce environment variables or data-root overrides.

Choose a canonical format per dataset from actual loader requirements. Do not mandate one universal storage format for classification, segmentation, multimodal, and temporal data. Prefer individually downloadable upstream assets; mirror or repackage when necessary.

Each dataset declares source/revision, independently fetchable assets, checksums, split manifests, sample IDs, band/unit metadata, and local layout. Revisions identify immutable content. Use the selected backend's checksums and resumable cache rather than adding a second completion-marker system. A directory existing is not proof of a complete dataset.

Repeated downloads verify/reuse complete assets without fetching the entire benchmark. Loading is offline when assets exist. Missing data errors name the exact download command; evaluation should not unexpectedly download a collection.

Before adopting a mirror, compare counts, IDs, labels, masks, splits, band order, and numerical values with the source. Preserve citations and redistribution terms in the manifest. A format change cannot silently change the scientific dataset or statistics.

Provide an explicit migration for existing data. Reuse or convert verified local assets where practical; do not delete a collection to force migration. Isolate any old-format compatibility adapter and document retirement criteria. This proposal does not upload or repackage data.

## Acceptance criteria for implementation

- [ ] A local fixture server tests individual/multiple downloads without touching unrelated assets.
- [ ] Download followed by all split loaders works in a clean directory using the same manifest/layout.
- [ ] Interrupted, corrupted, and repeated downloads preserve valid assets and cannot appear complete incorrectly.
- [ ] A source-to-mirror report verifies samples, splits, metadata, and pixel/label equivalence before adoption.

## Review boundary

Download/load contracts, manifests, completion semantics, and one dataset migration. Mirror publication and other dataset migrations get separate implementation PRs.

The implementation draft adds named downloads, validates all names before
dispatch, and prevents V1/V2 evaluation from implicitly downloading missing
data. It keeps the legacy full V1 collection path. Backend and loader parity
requirements above remain open; this draft does not claim mirror equivalence
or a complete interrupted-download fixture.

## Existing work and references

Current V1 full downloads use the legacy collection while individual downloads use a sharded mirror. V2 already has per-dataset repositories; preserve working individual-download behavior.

Source observations are pinned to `9c8e4af`; refresh before implementation.

- [Current downloads](https://github.com/torchgeo/torchgeo-bench/blob/9c8e4afab46675d7279c88828dfcbf0ca99b3a07/src/torchgeo_bench/download.py#L1)
- [V1 loader](https://github.com/torchgeo/torchgeo-bench/blob/9c8e4afab46675d7279c88828dfcbf0ca99b3a07/src/torchgeo_bench/datasets/geobench_v1.py#L1)
