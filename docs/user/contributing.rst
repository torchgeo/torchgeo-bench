Contributing
============

We welcome contributions!  This page summarises the local development
workflow.  See `AGENTS.md <https://github.com/torchgeo/torchgeo-bench/blob/main/AGENTS.md>`__
for the longer-form guide aimed at AI coding agents (which captures the
same conventions).

Environment
-----------

The canonical workflow uses `uv <https://docs.astral.sh/uv/>`_, which
resolves dependencies from :file:`pyproject.toml` and manages the virtual
environment for you:

.. code-block:: console

   $ git clone https://github.com/torchgeo/torchgeo-bench.git
   $ cd torchgeo-bench
   $ uv sync --extra dev

If you prefer `conda <https://docs.conda.io>`_, create an environment with a
compatible Python first and run ``uv sync`` inside it:

.. code-block:: console

   $ conda create -n torchgeo-bench 'python>=3.12,<3.13'
   $ conda activate torchgeo-bench
   $ uv sync --extra dev

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

PyPI publishing uses the ``pypi`` GitHub environment and a
`Trusted Publisher <https://docs.pypi.org/trusted-publishers/>`_. Prepare every
release through a pull request:

1. Update :file:`docs/user/changelog.rst`, ``project.version`` in
   :file:`pyproject.toml`, the fallback ``__version__`` in
   :file:`src/torchgeo_bench/__init__.py`, and :file:`uv.lock`.
2. Run the release gate from a clean checkout:

   .. code-block:: console

      $ uv run pre-commit run --all-files
      $ uv run pytest
      $ uv run --extra docs sphinx-build -W --keep-going -b html docs docs/_build/html
      $ uv build

3. Merge the release PR only after CI and the docs preview pass. Pull the merge
   commit onto ``main`` and confirm that the worktree is clean.
4. Create and push a tag that exactly matches ``project.version``:

   .. code-block:: console

      $ git tag vX.Y.Z
      $ git push origin vX.Y.Z

5. Create the matching GitHub release with curated notes (or generated notes
   reviewed against the changelog):

   .. code-block:: console

      $ gh release create vX.Y.Z --verify-tag --title vX.Y.Z --generate-notes

6. Watch the ``Publish to PyPI`` workflow, then verify the project page and a
   fresh install:

   .. code-block:: console

      $ gh run list --workflow release.yml
      $ uvx --from "torchgeo-bench==X.Y.Z" torchgeo-bench --help

The ``Publish to PyPI`` workflow (:file:`.github/workflows/release.yml`)
builds the source distribution and wheel in an unprivileged job, then uploads
them from a separate OIDC-authenticated job. The tag is the publication trigger;
do not move or reuse a published version tag.
