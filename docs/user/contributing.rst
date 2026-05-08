Contributing
============

We welcome contributions!  This page summarises the local development
workflow.  See `AGENTS.md <https://github.com/torchgeo/torchgeo-bench/blob/main/AGENTS.md>`__
for the longer-form guide aimed at AI coding agents (which captures the
same conventions).

Environment
-----------

.. code-block:: console

   $ git clone https://github.com/torchgeo/torchgeo-bench.git
   $ cd torchgeo-bench
   $ uv sync --extra dev

The conda environment ``torchgeo-bench`` is also supported; activate it
before running any tooling.

Linting and formatting
----------------------

We use `ruff <https://docs.astral.sh/ruff/>`_ for both linting and
formatting:

.. code-block:: console

   $ uv run ruff check .
   $ uv run ruff format .

The project's ruff configuration enables ``E``, ``W``, ``F``, ``I``,
``B``, ``C4``, ``UP``, ``ARG``, ``SIM``, and ``D`` (pydocstyle, Google
convention) checks.  Line length is 100.

Tests
-----

.. code-block:: console

   $ uv run pytest                                  # all tests (skipping slow)
   $ uv run pytest -m slow                          # only slow integration tests
   $ uv run pytest tests/test_intrinsic_dim.py -v   # one file
   $ uv run pytest --no-cov                         # disable coverage for speed

Tests that depend on real GeoBench data look up ``GEOBENCH_ROOT`` and
``GEOBENCH_V2_ROOT`` environment variables and skip cleanly if the data
isn't present.

Code style
----------

* Python 3.12+ throughout. Use modern type hints (``list[str]``,
  ``X | None``) — do **not** import from ``typing.List`` / ``Optional`` /
  ``Union``.
* Avoid ``from __future__ import annotations``; prefer ``Self``, quoted
  annotations, or explicit imports for forward references.
* Google-style docstrings (configured via ``ruff.lint.pydocstyle.convention``).
* Use the ``logging`` module — no bare ``print`` calls.

Documentation
-------------

This very site is built with Sphinx.  To build it locally:

.. code-block:: console

   $ uv sync --extra docs
   $ cd docs && uv run make html
   $ open _build/html/index.html
