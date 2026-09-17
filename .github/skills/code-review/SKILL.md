---
name: code-review
description: Review torchgeo-bench pull requests for correctness and unnecessary complexity, tests, comments, private helpers, duplicated behavior, and inflated prose. Use during automated Copilot PR reviews and requested code reviews.
---

# Code review

Review the changed code and its callers before suggesting a fix. Read the repository's `AGENTS.md` and relevant configuration. Apply the stricter preferences below to newly added code even where older examples allow explanatory comments or private helpers. Review the implementation, not whether its author used an LLM.

## Tests that earn their place

- A test should catch a plausible regression or verify a supported behavior. Identify the failure it would detect. Flag tests that merely repeat an assignment, check that a mock returns its configured value, compare a result with itself, or recompute the expected value with the same function being tested.
- Test observable behavior through the existing interface. Flag source-text searches, assertions about helper names or call order, and mocks that replace the behavior under test. Keep checks on ordering or calls when those are themselves required behavior, such as preprocessing exactly once or avoiding weight downloads during catalog discovery.
- Extend an existing test or parameterization when it covers the same behavior. Flag duplicated fixtures, repeated cases with no distinct failure mode, and large mock setups where a small real input would exercise the code.
- Do not request tests just because a file changed or coverage could increase. Documentation and instruction edits do not need tests that lock in their wording. Preserve useful regression tests and boundary cases for supported inputs, error handling, split isolation, band order, normalization, and result/resume identity.

## No explanatory code comments

- Flag added inline or block comments, including section banners, numbered steps, commented-out code, and narration of the next statement. Ask for clearer names or simpler code; put necessary design rationale in documentation or the PR description. Do not propose adding comments as the fix.
- Keep required license notices and functional directives such as shebangs, justified lint/type suppressions, and the repository's mandatory `# allow-except: <reason>` annotations. Concise Google-style API docstrings remain required where Ruff enforces them; remove filler and implementation walkthroughs from docstrings.

## Reuse and straightforward code

- Search the repository for equivalent behavior before judging a new function, including helpers with different names and inline implementations. When behavior already exists, cite its file and symbol and suggest reusing or narrowly extending it, updating callers, and removing the superseded copy. Do not combine distinct task or backend semantics just because their signatures look alike.
- Flag new project-owned private functions or methods with leading underscores. Prefer extending the existing function, inlining a trivial single-use helper, or a plainly named function with a clear responsibility. Preserve required interface hooks such as `BenchModel._forward_patch_features`, external overrides, and Python special methods. Do not request renaming untouched internals or breaking an existing interface to satisfy a naming preference.
- Flag one-line forwarding wrappers, classes that only wrap a function, factories for a single implementation, speculative plugin systems, and new configuration switches without a current caller. Prefer an explicit function or constructor and the smallest change that handles the actual requirement.
- Flag fragmented control flow, redundant intermediate containers, repeated conversions, and state that can be derived from existing values. Show the simpler equivalent. Fewer lines alone are not an improvement; do not replace readable code with dense comprehensions, nested ternaries, or helpers controlled by many flags.
- Keep shared facts in one authoritative place. Flag copied catalogs, defaults, band metadata, and preprocessing decisions that can drift between consumers. Preserve configuration precedence, explicit `false`/`null`/`[]`, lazy optional imports, and stored result compatibility when suggesting consolidation.

## Defensive code without a supported failure mode

- Flag fallback imports for hard dependencies, broad exception catches, silent defaults after failures, and retries without a specific recoverable condition. Optional dependencies can remain lazy; unexpected failures should propagate with their context.
- Flag repeated validation after the input has already been validated, `getattr`/`hasattr` on guaranteed fields, impossible `None` checks, and compatibility branches for unsupported versions. Check the declared types, callers, supported versions, and dependency requirements before claiming a branch is unnecessary.
- Preserve validation at external boundaries and handling for real failures such as missing data, invalid user configuration, or recoverable CUDA OOM. Do not trade correct error reporting or a supported behavior for shorter code.

## Plain prose

Apply these rules to added documentation, docstrings, error messages, and PR text. Preserve facts, numbers, citations, qualifications, and exact API names.

- Replace inflated wording with the concrete action: `leverage`/`utilize` → `use`, `delve` → `study`, `showcase`/`underscore` → `show`, `elucidate` → `explain`, `ameliorate` → `improve`, and `ships with` → `includes`.
- Flag filler such as `seamlessly`, `powerful`, `game-changing`, `pivotal`, `comprehensive`, `robust`, and `unlock` when no specific property is stated. Cut `Notably`, `Importantly`, `It is worth noting`, stacked hedges, and closing sentences that repeat the paragraph.
- Flag dramatic colon reveals, repeated em-dash asides, “not just X, but Y” pivots, invented `X-first` labels, and marketing claims that replace an explanation. State what the code does and why the reader needs to know.
- Prefer `setup` to a vague `protocol`, `format` to a vague `schema`, and a specific verb to catch-all `supports`. These are contextual prose preferences, not token bans: keep technical uses such as Pydantic schemas, `typing.Protocol`, robust statistics, published titles, and quoted text. Do not rename APIs to satisfy a word list.

## Useful review findings

Report concrete violations introduced by the PR, including these style rules, with a changed line, the applicable rule or consequence, and the smallest useful correction. For duplication, name the existing implementation; for a weak test, explain why it cannot catch the claimed failure. Group repeated instances of the same issue. Keep correctness findings ahead of style findings and label style findings accordingly. Skip generic praise, summaries of the diff, speculative hardening, unrelated cleanup, and comments that only say something “looks like AI.” Do not invent findings when the change already meets these rules.
