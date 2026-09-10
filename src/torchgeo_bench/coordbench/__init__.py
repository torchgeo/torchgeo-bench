"""CoordBench: coordinate-only location-encoder evaluation for torchgeo-bench.

Loads the unified ``taylor-geospatial/coordbench`` benchmark suite (point
``(lon, lat)`` -> label) and probes a frozen coordinate encoder with KNN and a
ridge linear head under random or spatial-block cross-validation.

Public API
----------
.. autoclass:: LocationEncoder
.. autoclass:: SinCosLocationEncoder
.. autoclass:: CoordBenchmark
.. autofunction:: load_benchmarks
.. autofunction:: run_coordbench
"""

from torchgeo_bench.coordbench.datasets import (
    CoordBenchmark,
    list_benchmarks,
    list_families,
    load_benchmarks,
)
from torchgeo_bench.coordbench.models import (
    ClimplicitLocationEncoder,
    GeoCLIPLocationEncoder,
    LocationEncoder,
    MINDLocationEncoder,
    SatCLIPLocationEncoder,
    SinCosLocationEncoder,
    SINRLocationEncoder,
)
from torchgeo_bench.coordbench.probe import (
    knn_probe_score,
    linear_probe_score,
    spatial_fold_ids,
)
from torchgeo_bench.coordbench.run import CoordResult, run_coordbench

__all__ = [
    "ClimplicitLocationEncoder",
    "CoordBenchmark",
    "CoordResult",
    "GeoCLIPLocationEncoder",
    "LocationEncoder",
    "MINDLocationEncoder",
    "SINRLocationEncoder",
    "SatCLIPLocationEncoder",
    "SinCosLocationEncoder",
    "knn_probe_score",
    "linear_probe_score",
    "list_benchmarks",
    "list_families",
    "load_benchmarks",
    "run_coordbench",
    "spatial_fold_ids",
]
