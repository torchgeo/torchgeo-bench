Command-line interface
======================

.. module:: torchgeo_bench.cli

The ``torchgeo-bench`` console script exposes two subcommands:

``torchgeo-bench run [hydra overrides...]``
    Runs the benchmark pipeline. All extra arguments are forwarded to
    Hydra, so any value in :file:`src/torchgeo_bench/conf/config.yaml` (or
    any model preset under :file:`conf/model/`) can be overridden directly
    on the command line, e.g. ``model=timm/resnet50 dataset.names=[m-eurosat]``.

``torchgeo-bench download {geobench_v1|geobench_v2|eurosat}``
    Downloads benchmark datasets into ``./data/`` (or a custom location with
    ``--output-dir``). For GeoBench V2, individual datasets can be selected
    with ``--datasets a,b,c``.

Hydra entry point
-----------------

The actual benchmark loop lives in :mod:`torchgeo_bench.main` and is
decorated with ``@hydra.main``:

.. currentmodule:: torchgeo_bench.main
.. autofunction:: main

Download helpers
----------------

.. currentmodule:: torchgeo_bench.download
.. autofunction:: download_geobench_v1
.. autofunction:: download_geobench_v2
.. autofunction:: download_eurosat
