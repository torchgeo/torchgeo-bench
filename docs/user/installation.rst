Installation
============

``torchgeo-bench`` targets Python 3.12 or newer.  We recommend installing
inside a fresh virtual environment using `uv`_, which is the project's
canonical workflow.

.. _uv: https://docs.astral.sh/uv/

.. note::

   The default (CPU) install runs on **Linux**, **macOS**, and **Windows**.
   GPU-accelerated KNN (the ``[cuda]`` extra) is Linux-only.

uv (recommended)
----------------

Clone the repository and let ``uv`` create the environment and install all
development dependencies:

.. code-block:: console

   $ git clone https://github.com/torchgeo/torchgeo-bench.git
   $ cd torchgeo-bench
   $ uv sync --extra dev

The default install pulls in ``faissknn[cpu]`` (CPU FAISS), which works on
all three platforms.  For GPU-accelerated KNN (Linux + CUDA 12 + glibc ≥ 2.28),
which swaps in ``faissknn[cuda]``:

.. code-block:: console

   $ pip install 'torchgeo-bench[cuda]'

The ``torchgeo-bench`` console script is then available via ``uv run``:

.. code-block:: console

   $ uv run torchgeo-bench --help

pip
---

If you prefer ``pip`` (e.g. inside a conda environment), install the
project in editable mode:

.. code-block:: console

   $ git clone https://github.com/torchgeo/torchgeo-bench.git
   $ cd torchgeo-bench
   $ pip install -e ".[dev]"

Optional extras
---------------

A number of dependencies are gated behind PEP 621 extras so that lean
benchmarking installations do not pull in heavyweight libraries by
default.  Combine extras with comma-separated lists:

.. code-block:: console

   $ uv sync --extra docs --extra viz
   $ # or
   $ pip install -e ".[docs,viz]"

.. note::

   ``uv sync`` installs *exactly* the extras you pass and removes anything not
   listed, so extras do **not** accumulate across calls.  Request them together
   in one command (``uv sync --extra docs --extra viz``) rather than running
   ``uv sync --extra docs`` and then ``uv sync --extra viz`` (the second drops
   the first).

================  ==============================================================
Extra             Pulls in
================  ==============================================================
``cleanlab``      ``cleanlab``, ``imagehash``, ``matplotlib``, ``pillow`` (label-noise audit)
``cuda``          ``faissknn[cuda]`` → ``faiss-cuda-cu128`` for GPU KNN (Linux only; shares the ``faiss`` namespace with the default ``faissknn[cpu]`` — install in a fresh env)
``dev``           ruff, pytest, pytest-cov, pytest-xdist, pre-commit, mdformat, pyproject-fmt
``docs``          sphinx, pydata-sphinx-theme, myst-parser, sphinx-copybutton, sphinx-design
``id``            ``torchid`` for intrinsic-dimension metrics (Python ≥ 3.13 only)
``olmoearth``     ``olmoearth-pretrain-minimal`` for the OlmoEarth backbone
``sam3``          ``transformers`` for the SAM 3 encoder
``terratorch``    ``terratorch`` for TerraTorch backbones
``viz``           ``matplotlib``, ``pillow`` for segmentation visualisations
``all``           every extra above **except** ``cuda`` (FAISS conflict) and ``olmoearth``
================  ==============================================================

Datasets are not installed by these extras — see :doc:`datasets` for how to
download GeoBench V1 / V2 with the bundled ``torchgeo-bench download`` command.
