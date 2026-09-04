# R03 - Choose lazy command imports while keeping top-level imports

Status: proposed; no option has been accepted. This file changes no runtime behavior.
Decision issue: https://github.com/torchgeo/torchgeo-bench/issues/309

Keep help fast and implementation imports at module scope by loading only the selected command.

## Decision

Should command handlers use lazy-loader with typed stubs or a small function-local import boundary?

| Option | Choice | Tradeoff |
| --- | --- | --- |
| A | lazy-loader command exports (recommended) | Top-level implementation imports and typed exports; adds a dependency and packaged .pyi files. |
| B | Local imports in dispatch only | No dependency or stubs; a few function-local imports remain. |

Vote in the linked issue. Comment `Vote: A` or `Vote: B` with a short rationale. If neither fits, propose a concrete amendment. Recommendations are proposals, not recorded votes or maintainer approval. Maintainers will summarize the outcome in the issue. Reactions indicate interest, not a choice between options.

## Proposed contract

Keep the package initializer, parser, command namespace, and catalog metadata cheap to import. Load the selected command's numerical implementation after parsing and lightweight configuration validation.

For option A, use Scientific Python's `lazy-loader` at command-package boundaries with `attach_stub`:

```python
# commands/__init__.py
import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)
```

```python
# commands/__init__.pyi
from ._profile import main as profile
from ._run import main as run
```

```python
# cli.py (dispatch excerpt)
import argparse

from . import commands


def dispatch(args: argparse.Namespace) -> None:
    if args.command == 'run':
        commands.run(args)
    elif args.command == 'profile':
        commands.profile(args)
```

Accessing `commands.run` in argparse defaults or a module-level dispatch table resolves it too early. Resolve handlers only after parsing. Keep validation ahead of numerical imports; catalogs read lightweight metadata rather than import every adapter.

Ship `.pyi` files in both wheel and sdist. Keep the command namespace limited to the exports described by its stub. Test a built wheel: an editable checkout can hide omitted files.

Implementation modules can use ordinary top-level imports for hard dependencies. Optional model packages load only when their adapter is selected. `torch.nn.Module` in a class definition counts as first use, so laziness belongs around the adapter module rather than individual Torch attributes.

Retain Python 3.12+ for this decision. Python 3.15 native `lazy import` is a future alternative. Do not globally patch imports or enable interpreter-wide lazy-import behavior.

## Acceptance criteria for implementation

- [ ] Subprocess tests prove help, catalogs, and invalid arguments do not import torch, torchgeo, timm, or pandas.
- [ ] Each command resolves the correct implementation and reports genuine import failures.
- [ ] Built-wheel tests verify stubs, lazy exports, type-checker visibility, and operation without unrelated extras.
- [ ] Record fresh-process help timing on a reference machine; target median below 100 ms without a fragile universal CI threshold.

## Review boundary

Import boundaries, lightweight metadata, stubs, and startup tests. Command naming and model construction remain separate choices.

This draft contains only this decision document. Implementation must pass
the oracle and migration requirements described in the refactor overview;
adding this document does not satisfy the criteria above.

## Existing work and references

Coordinate with [#306](https://github.com/torchgeo/torchgeo-bench/pull/306), which already addresses explicit command routing.

Source observations are pinned to `9c8e4af`; refresh before implementation.

- [Current package exports](https://github.com/torchgeo/torchgeo-bench/blob/9c8e4afab46675d7279c88828dfcbf0ca99b3a07/src/torchgeo_bench/__init__.py#L1)
- [Scientific Python lazy loading](https://scientific-python.org/specs/spec-0001/)
- [Python lazy imports](https://docs.python.org/3.15/reference/simple_stmts.html#lazy-imports)
