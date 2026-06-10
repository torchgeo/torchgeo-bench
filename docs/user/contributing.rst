Contributing
============

We welcome contributions!  This page summarises the local development
workflow.  See `AGENTS.md <https://github.com/torchgeo/torchgeo-bench/blob/main/AGENTS.md>`__
for the longer-form guide aimed at AI coding agents (which captures the
same conventions).

Environment
-----------

The repo's canonical workflow uses both `conda <https://docs.conda.io>`_
and `uv <https://docs.astral.sh/uv/>`_:

.. code-block:: console

   $ git clone https://github.com/torchgeo/torchgeo-bench.git
   $ cd torchgeo-bench
   $ conda env update -n torchgeo-bench -f environment.yml
   $ conda activate torchgeo-bench
   $ uv sync --extra dev

If you'd rather skip conda, ``uv sync --extra dev`` alone is enough on
any Python 3.12+ install.

Makefile shortcuts
------------------

The top-level :file:`Makefile` provides convenient wrappers around the
above:

=================== ===================================================
Target              What it does
=================== ===================================================
``make install``    Create / update the conda env and install ``[dev]``.
``make sync``       Alias for ``install``.
``make tests``      ``pytest`` (skips ``slow`` integration tests).
``make lint``       ``pre-commit run --all-files``.
``make format``     ``ruff format`` then ``ruff check --fix --select I``.
``make docs``       Build HTML documentation into ``docs/_build/html``.
``make docs-clean`` Remove the ``docs/_build`` directory.
``make clean``      Removes ``htmlcov``, ``.coverage``, ``.pytest_cache``.
=================== ===================================================

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
   $ uv run pytest -k "m-eurosat" -v                # by keyword
   $ uv run pytest --no-cov                         # disable coverage for speed

Tests skip gracefully when ``data/`` is missing — they look up the
canonical subdirs documented in :doc:`datasets`.

Code style
----------

* Python 3.12+ throughout. Use modern type hints (``list[str]``,
  ``X | None``) — do **not** import from ``typing.List`` / ``Optional`` /
  ``Union``.
* Avoid ``from __future__ import annotations``; prefer ``Self``, quoted
  annotations, or explicit imports for forward references.
* Google-style docstrings (configured via ``ruff.lint.pydocstyle.convention``).
* Use the ``logging`` module — no bare ``print`` calls.
* No defensive ``try/except ImportError`` for hard dependencies — every
  package in ``[project.dependencies]`` is guaranteed to be installed.

Documentation
-------------

This very site is built with Sphinx.  The quickest way to build it locally
is via the Makefile shortcut:

.. code-block:: console

   $ make docs
   $ open docs/_build/html/index.html

This assumes ``sphinx-build`` is on your ``PATH`` (install with
``uv sync --extra docs``).  To rebuild from scratch:

.. code-block:: console

   $ make docs-clean && make docs

Releasing to PyPI
-----------------

1. Configure a `PyPI Trusted Publisher
   <https://docs.pypi.org/trusted-publishers/>`_ for this repository
   with environment name ``pypi``.
2. Tag and push:

   .. code-block:: console

      $ git tag v0.2.0
      $ git push origin v0.2.0

The ``Publish to PyPI`` workflow (:file:`.github/workflows/release.yml`)
builds and uploads the release automatically.
