# R01 - Choose the refactor CLI command surface

Status: proposed; no option has been accepted. This file changes no runtime behavior.
Decision issue: https://github.com/torchgeo/torchgeo-bench/issues/307

Use discoverable commands with separate image, coordinate, and profiling workflows.

## Decision

Should the rewrite use explicit argparse commands or generate the CLI from the configuration schema?

| Option | Choice | Tradeoff |
| --- | --- | --- |
| A | Explicit argparse commands (recommended) | Visible command surface and ordinary flags; small handwritten mappings. |
| B | Schema-generated CLI | Automatic flag coverage; additional parser dependency and coupling to schema layout. |

Vote in the linked issue. Comment `Vote: A` or `Vote: B` with a short rationale. If neither fits, propose a concrete amendment. Recommendations are proposals, not recorded votes or maintainer approval. Maintainers will summarize the outcome in the issue. Reactions indicate interest, not a choice between options.

## Proposed contract

The proposed commands are `run`, `coordbench`, `profile`, `download`, `models [NAME]`, and `datasets [NAME]`. Implement image evaluation, catalogs, and individual downloads first. Add a narrow profiler separately; coordinate and analysis workflows follow the image migration.

```console
torchgeo-bench models
torchgeo-bench datasets m-eurosat
torchgeo-bench download m-eurosat burn_scars
torchgeo-bench run --model timm/resnet50 --dataset m-eurosat
torchgeo-bench run --config experiments/resnet50.yaml --device cuda:1
torchgeo-bench run --config experiments/resnet50.yaml --dry-run
torchgeo-bench profile --model timm/resnet50 --dataset m-eurosat
```

One run selects one model. Repeated `--dataset` flags select datasets through a simple serial loop. Require an explicit model and dataset selection in a file or flags; a bare command prints usage. Keep larger model/seed sweeps in ordinary Python, shell, or SLURM scripts initially.

Use named flags for common run, probe, input, and runtime settings. Group help by responsibility and show choices and defaults. `models NAME` describes every model-specific config field without loading weights. Do not introduce a replacement `--set` grammar.

Precedence is schema defaults, then one YAML file, then explicitly supplied flags. An omitted boolean flag cannot overwrite a file value. Provide paired boolean flags where needed. A flag-provided list replaces the file list. Document explicit null/unset behavior for settings such as image size.

`--dry-run` validates configuration and metadata compatibility and prints resolved settings without downloading data, loading weights, or probing a GPU. Tensor-dependent validation happens during execution. Exit 0 on success, 2 on invalid input, and nonzero on execution failure. Machine-readable output goes to stdout; diagnostics go to stderr.

## Acceptance criteria for implementation

- [ ] Help, invalid-option handling, and catalogs work without ML imports or network access.
- [ ] Tests cover precedence, false values, list replacement, missing selections, and incompatible task options.
- [ ] Every config field is discoverable through command help or model details, with flag/file mapping documented.
- [ ] Legacy command migration examples preserve resolved scientific settings and identify intentional incompatibilities.

## Review boundary

Parser, routing, help, catalogs, and migration examples. Schema-library choice, lazy-import implementation, and numerical evaluation are separate decisions.

This draft contains only this decision document. Implementation must pass
the oracle and migration requirements described in the refactor overview;
adding this document does not satisfy the criteria above.

## Existing work and references

Existing implementation: [#306](https://github.com/torchgeo/torchgeo-bench/pull/306). Existing request: [#255](https://github.com/torchgeo/torchgeo-bench/issues/255). This proposal neither replaces nor approves that implementation.

Source observations are pinned to `9c8e4af`; refresh before implementation.

- [Current parser](https://github.com/torchgeo/torchgeo-bench/blob/9c8e4afab46675d7279c88828dfcbf0ca99b3a07/src/torchgeo_bench/cli.py#L1)
