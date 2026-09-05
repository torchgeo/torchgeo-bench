# R04 - Choose a human-readable coding style and enforcement gate

Status: option A is implemented in the draft; discussion remains open.
Decision issue: https://github.com/torchgeo/torchgeo-bench/issues/310

Adopt explicit conventions and narrow mechanical checks while reviewing code for readability.

## Decision

Should rewritten modules use a TorchGeo-derived formatting profile or preserve the current 100-column layout?

| Option | Choice | Tradeoff |
| --- | --- | --- |
| A | TorchGeo-derived profile (recommended) | 88 columns, single quotes, collapsed trailing commas, full annotations, Google docstrings. |
| B | Preserve current formatting profile | Keep 100 columns and current quotes/wrapping; still add typing, complexity, and suppression checks. |

Discuss the two options in the linked issue. We will collect feedback before choosing.

## Proposed contract

Use explicit functions and concrete names. Keep construction and data flow visible. Duplication is acceptable until a helper improves real call sites. Comments explain constraints or rationale; docstrings specify shapes, units, and behavior.

Keep comments to one sentence explaining a non-obvious constraint or reason.
Do not narrate the code or add background paragraphs above a function.
Docstrings default to one sentence; add only needed shapes, units, argument constraints, or exceptions.
Types and obvious parameter names do not need repeating in prose.
Put extended rationale and examples in documentation.

The gate rejects comment blocks longer than one line, except the standard copyright/license header.
Docstrings may contain at most 80 words and 10 nonblank lines, including Google-style fields.
Ruff's `W505` also caps docstring and standalone-comment lines at 88 characters; wrapping alone cannot evade the block limits.
These are ceilings, not targets, and the checks do not replace review for useful content.

Option A formatting:

```toml
[tool.ruff]
target-version = "py312"
line-length = 88

[tool.ruff.format]
quote-style = "single"
skip-magic-trailing-comma = true
docstring-code-format = true

[tool.ruff.lint.isort]
split-on-trailing-comma = false

[tool.ruff.lint.pydocstyle]
convention = "google"
```

Pin Ruff and explicitly select rules. Add `ANN`, `D`, `I`, `N`, `UP`, `B`, and `PT` alongside existing correctness checks. Add `C901`, `PLR0912`, and `PLR0915`; initially allow complexity 10, 12 branches, and 50 statements. Allow reviewed, specific exceptions for clear parser/catalog tables instead of artificial splitting.

Use `BLE001`, `PGH004`, `RUF100`, and `TID251` for broad catches, blanket/stale suppressions, and banned imports such as future annotations. Broad-catch lint has exceptions and still needs review. Fully annotate functions and tests; keep `Any` at genuine external boundaries. Use relative package imports and a sorted tuple for `__all__`.

Use project copyright headers and one-line module docstrings. Check copyright-rule support against the pinned Ruff version before enabling it. Google docstring entries use capitalized descriptions and periods. Document public constructors; avoid repetitive private-helper/test prose.

Retain project logging and Conventional Commit policies. Permit the selected schema library and dataclasses. Do not copy TorchGeo's Path-name restrictions or domain architecture merely for consistency. Prefer files below about 500 lines, with reviewed exceptions. Do not enable duplicate-code lint.

Use Ruff, ty, pytest/coverage, and pre-commit. Mirror source modules in tests with deterministic, offline fixtures. Require 100% coverage of touched rewritten modules using meaningful assertions, alongside separate oracle and invariant checks. Import Linter is optional if boundary drift becomes recurrent.

## Acceptance criteria for implementation

- [ ] One documented local gate matches non-mutating CI checks.
- [ ] Rewritten functions/tests are annotated and type-check without broad ignores.
- [ ] Formatting-only changes are isolated from scientific/API changes; no wholesale legacy-tree reformat in a behavior PR.
- [ ] Coverage identifies missing branches, and oracle/invariant checks independently validate behavior.

## Review boundary

Conventions and enforcement for rewritten modules. Repository-wide formatting, dependency upgrades, and unrelated CI repairs stay separate.

The proposal adds a separate Ruff profile and a non-mutating style/type gate for rewritten modules. Run `uv run --extra dev python scripts/check_refactor_style.py`, or `uv run --extra dev pre-commit run --all-files` for the repository checks. CI uses the same pre-commit hook. Legacy files retain their current formatting; the explicit file list grows as modules are migrated.

## Existing work and references

Existing infrastructure issues: [#264](https://github.com/torchgeo/torchgeo-bench/issues/264), [#266](https://github.com/torchgeo/torchgeo-bench/issues/266). This proposal does not close them.

Source observations are pinned to `9c8e4af`; refresh before implementation.

- [Current Ruff configuration](https://github.com/torchgeo/torchgeo-bench/blob/9c8e4afab46675d7279c88828dfcbf0ca99b3a07/pyproject.toml#L129)
- [Ruff configuration](https://docs.astral.sh/ruff/configuration/)
- [Ruff complexity](https://docs.astral.sh/ruff/rules/complex-structure/)
- [ty](https://docs.astral.sh/ty/)
