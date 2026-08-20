CoordBench
==========

The coordinate-only pipeline loads point-label benchmarks, embeds coordinates
with a frozen :class:`~torchgeo_bench.coordbench.LocationEncoder`, and evaluates
lightweight downstream probes. See :doc:`/user/coordbench` for runnable examples.

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
