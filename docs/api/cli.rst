Command-line interface
======================

.. module:: torchgeo_bench.cli

The ``torchgeo-bench`` console script exposes three subcommands:

``torchgeo-bench run [flags] [key=value ...]``
    Runs the benchmark pipeline. Common settings have flags (``--model``,
    ``--datasets``, ``--device``, ``--resume``, ...; see ``run --help``), and
    any value in :file:`src/torchgeo_bench/conf/config.yaml` (or any model
    preset under :file:`conf/model/`) can be overridden with ``key=value``
    pairs, e.g. ``model=timm/resnet50 dataset.names=[m-eurosat]``.

``torchgeo-bench download {geobench_v1|geobench_v2|eurosat}``
    Downloads benchmark datasets into ``./data/`` (or a custom location with
    ``--output-dir``). For GeoBench V1 and V2, individual datasets can be
    selected with ``--datasets a,b,c``.

``torchgeo-bench flops [flags] [key=value ...]``
    Measures per-sample backbone, head, and probe compute. Overrides are
    composed from :file:`src/torchgeo_bench/conf/flops_config.yaml`.

Config composition
------------------

.. currentmodule:: torchgeo_bench.config

.. autofunction:: compose_config
.. autofunction:: list_model_configs

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
