CoordBench
==========

The coordinate-only pipeline loads point-label benchmarks, embeds coordinates
with a frozen :class:`~torchgeo_bench.coordbench.LocationEncoder`, and evaluates
lightweight downstream probes. See :doc:`/user/coordbench` for runnable examples.

Typed configuration
-------------------

The public runner accepts ``CoordConfig``, not image settings:

.. code-block:: python

   from torchgeo_bench.coordbench import run_coordbench
   from torchgeo_bench.coordbench.config import load_coord_config

   config = load_coord_config("docs/examples/coord-run.yaml")
   run_coordbench(config)

Loading validates YAML without fetching tables or weights. Calling the runner
executes the benchmark and appends results to ``config.output.file``.

.. currentmodule:: torchgeo_bench.coordbench.config

.. autoclass:: CoordConfig
   :members: model_dump_yaml
   :no-show-inheritance:

.. autofunction:: load_coord_config
.. autofunction:: resolve_coord_preset

Encoders
--------

.. currentmodule:: torchgeo_bench.coordbench

.. autoclass:: LocationEncoder
.. autoclass:: SinCosLocationEncoder
.. autoclass:: MINDLocationEncoder
.. autoclass:: ClimplicitLocationEncoder
.. autoclass:: GeoCLIPLocationEncoder
.. autoclass:: SatCLIPLocationEncoder
.. autoclass:: SINRLocationEncoder
.. autoclass:: XYZLocationEncoder
.. autoclass:: NeRFLocationEncoder
.. autoclass:: SphericalHarmonicLocationEncoder

Spatial priors
--------------

These CPU estimators fit task labels and belong to the separate ``coord-prior`` track, not the frozen-encoder catalog. Coordinates and distance bandwidths are in longitude/latitude degrees.

.. code-block:: python

   from torchgeo_bench.coordbench import load_coord_prior_config, run_coordbench_priors

   config = load_coord_prior_config("docs/examples/coord-prior.yaml")
   run_coordbench_priors(config)

.. autoclass:: CoordPriorConfig
   :members: model_dump_yaml
   :no-show-inheritance:

.. autofunction:: load_coord_prior_config

.. autoclass:: SpatialPrior
.. autoclass:: UniformPrior
.. autoclass:: ClassFrequencyPrior
.. autoclass:: GridPrior
.. autoclass:: NearestNeighborPrior
.. autoclass:: KDEPrior

``EmpiricalPrior`` and ``FrequencyBaseline`` alias ``ClassFrequencyPrior``; ``DistancePrior`` and ``NearestNeighborBaseline`` alias ``NearestNeighborPrior``. ``UniformBaseline``, ``GridBaseline``, and ``KDEBaseline`` are also available.

Benchmarks
----------

.. autoclass:: CoordBenchmark
.. autofunction:: list_families
.. autofunction:: list_benchmarks
.. autofunction:: load_benchmarks

Probes and runner
-----------------

.. autofunction:: linear_probe_score
.. autofunction:: knn_probe_score
.. autofunction:: spatial_fold_ids
.. autoclass:: CoordResult
.. autofunction:: run_coordbench
.. autoclass:: CoordPriorResult
.. autofunction:: run_coordbench_priors
