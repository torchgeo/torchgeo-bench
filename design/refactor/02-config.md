# R02 - Choose a validated YAML configuration schema

Status: proposed; no option has been accepted. This file changes no runtime behavior.
Decision issue: https://github.com/torchgeo/torchgeo-bench/issues/308

Load YAML with PyYAML and validate explicit command-specific settings before execution.

## Decision

Should command configuration use Pydantic models or standard-library dataclasses with explicit validation?

| Option | Choice | Tradeoff |
| --- | --- | --- |
| A | Pydantic models (recommended) | Nested validation, useful field errors, constraints, and schema export with little loader code. |
| B | Dataclasses and explicit validation | Smaller dependency surface; maintain nested conversion and validation ourselves. |

Vote in the linked issue. Comment `Vote: A` or `Vote: B` with a short rationale. If neither fits, propose a concrete amendment. Recommendations are proposals, not recorded votes or maintainer approval. Maintainers will summarize the outcome in the issue. Reactions indicate interest, not a choice between options.

## Proposed contract

Keep YAML files. Use PyYAML's SafeLoader with a small duplicate-key rejection extension. Apply explicitly supplied CLI overrides, then validate. Reject unknown fields at every level. Errors identify the file, field, supplied value, and expected constraint.

Prefer Pydantic for configuration and ordinary dataclasses for internal records such as band metadata. Use strict types deliberately: `Literal` for YAML string choices, lists for YAML sequences. Do not silently coerce strings such as `"false"` or `"64"`.

Define separate `RunConfig`, `ProfileConfig`, and later `CoordBenchConfig`. Image runs have explicit classification and segmentation sections. Reject settings for a task absent from the selected datasets; mixed-task runs validate each applicable section and resolve one experiment per dataset. Model options have named schema fields, not an unvalidated kwargs dictionary.

Illustrative classification config (proposed, not executable today):

```yaml
schema_version: 1
model:
  name: timm/resnet50
datasets: [m-eurosat]
input:
  bands: rgb
  image_size: 224
  interpolation: bilinear
  normalization: dataset
classification:
  methods: [knn, linear]
  knn:
    neighbors: 5
  linear:
    c_log10_start: -6
    c_log10_stop: 4
    c_count: 40
    refit_train_val: true
  bootstrap_samples: 200
runtime:
  device: cuda:0
  batch_size: 64
  workers: 4
  seed: 0
output:
  directory: results
  resume: true
```

Defaults live in the schema. Model presets supply documented model settings but cannot silently change the evaluation protocol. Remove `_target_`, arbitrary key additions, environment substitution, and interpolation from the new format. Save the resolved configuration with each run. Old-config translation is a separate migration boundary.

If dataclasses win, use explicit constructors and short validators for each section. Do not build a generic reflection-based configuration engine. Preserve accepted work in the existing schema/CLI PRs whichever option is chosen.

## Acceptance criteria for implementation

- [ ] Wrong nested fields/types, duplicate keys, empty selections, and invalid ranges fail before weights/data load.
- [ ] Every published preset/example validates; resolved configs serialize and reload without semantic changes.
- [ ] Cross-field validation covers held-out calibration, task/probe compatibility, and band selections.
- [ ] Configuration handling stays outside the help path and does not import ML frameworks.

## Review boundary

Schema definitions, YAML parsing, explicit overrides, and config migration. Model construction and result identity are distinct concerns.

This draft contains only this decision document. Implementation must pass
the oracle and migration requirements described in the refactor overview;
adding this document does not satisfy the criteria above.

## Existing work and references

Existing implementations: [#305](https://github.com/torchgeo/torchgeo-bench/pull/305) validates model presets; [#306](https://github.com/torchgeo/torchgeo-bench/pull/306) introduces dataclass settings and PyYAML. Review and reuse accepted work.

Source observations are pinned to `9c8e4af`; refresh before implementation.

- [Current configuration](https://github.com/torchgeo/torchgeo-bench/blob/9c8e4afab46675d7279c88828dfcbf0ca99b3a07/src/torchgeo_bench/config.py#L1)
- [Pydantic models](https://docs.pydantic.dev/latest/concepts/models/)
- [PyYAML](https://pyyaml.org/wiki/PyYAMLDocumentation)
