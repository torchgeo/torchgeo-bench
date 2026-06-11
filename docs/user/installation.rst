Installation
============

``torchgeo-bench`` targets Python 3.12 or newer.  We recommend installing
inside a fresh virtual environment using `uv`_, which is the project's
canonical workflow.

.. _uv: https://docs.astral.sh/uv/

.. note::

   ``torchgeo-bench`` is supported on **Linux** and **macOS**.  Windows is
   not supported; on Windows, install inside `WSL2
   <https://learn.microsoft.com/windows/wsl/>`_.

uv (recommended)
----------------

Clone the repository and let ``uv`` create the environment and install all
development dependencies:

.. code-block:: console

   $ git clone https://github.com/torchgeo/torchgeo-bench.git
   $ cd torchgeo-bench
   $ uv sync --extra dev

For GPU-accelerated KNN (Linux + CUDA 12 + glibc ≥ 2.28):

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

================  ==============================================================
Extra             Pulls in
================  ==============================================================
``dev``           ruff, pytest, pre-commit, mdformat, pyproject-fmt
``docs``          sphinx, pydata-sphinx-theme, myst-parser, sphinx-copybutton
``id``            ``torchid`` for intrinsic-dimension metrics (Python ≥ 3.13)
``olmoearth``     ``olmoearth-pretrain-minimal`` for the OlmoEarth backbone
``sam3``          ``transformers`` for the SAM 3 encoder
``viz``           ``matplotlib``, ``pillow`` for segmentation visualisations
``all``           every extra above
================  ==============================================================

Datasets are not installed by these extras — see :doc:`datasets` for how to
download GeoBench V1 / V2 with the bundled ``torchgeo-bench download`` command.
