Command-line interface
======================

.. module:: torchgeo_bench.cli

The ``torchgeo-bench`` console script exposes six subcommands:

``torchgeo-bench run [flags]``
    Runs the benchmark pipeline. Common settings are plain flags (``--model``,
    ``--datasets``, ``--device``, ``--resume``, ...; see ``run --help`` for
    the full list). Anything uncommon goes in a YAML file passed via
    ``--config PATH``, merged under the built-in defaults but under explicit
    CLI flags (flags always win); there is no ``key=value`` override syntax.
    Output is quiet by default; pass ``-v``/``--verbose`` for INFO-level
    progress logs. ``--list-models``/``--list-datasets`` list valid
    ``--model``/``--datasets`` values, ``--model-help <name>`` prints that
    model preset's YAML (its available ``model.*`` settings) without running
    anything, and ``--print-config`` prints the fully merged settings and
    exits.

``torchgeo-bench profile [flags]``
    A thin alias over ``run`` (same flags) that additionally enables the
    compute-profile pass (throughput/latency/params), written to
    ``results/profiles/``.

``torchgeo-bench intrinsic-dim [flags]``
    A thin alias over ``run`` (same flags) that additionally enables the
    intrinsic-dimension pass, written to ``results/intrinsic_dim/``.

``torchgeo-bench coord [flags]``
    Runs the CoordBench location-encoder track (``--model``, ``--names``,
    ``--methods``, ``--split``, ``--folds``, ...; see ``coord --help``).

``torchgeo-bench flops [flags]``
    Measures per-sample backbone, head, and probe compute (``--model``,
    ``--device``, ``--output``).

``torchgeo-bench download {geobench_v1|geobench_v2|eurosat|resisc45}``
    Downloads benchmark datasets into ``./data/`` (or a custom location with
    ``--output-dir``). For GeoBench V1 and V2, individual datasets can be
    selected with ``--datasets a,b,c``.

Every subcommand above except ``download`` accepts ``--config PATH`` (a YAML
file of uncommon settings) and ``--print-config``.

Config composition
------------------

.. currentmodule:: torchgeo_bench.config

.. autofunction:: compose_config
.. autofunction:: list_model_configs

Configuration settings
----------------------

.. currentmodule:: torchgeo_bench.settings

.. autoclass:: RunSettings
.. autoclass:: FlopsSettings
.. autoclass:: EvalSettings

Benchmark entry point
---------------------

The benchmark loop lives in :mod:`torchgeo_bench.main`; the CLI composes
the config with :func:`torchgeo_bench.config.compose_config` and calls it:

.. currentmodule:: torchgeo_bench.main
.. autofunction:: main

Download helpers
----------------

.. currentmodule:: torchgeo_bench.download
.. autofunction:: download_geobench_v1
.. autofunction:: download_geobench_v2
.. autofunction:: download_eurosat
