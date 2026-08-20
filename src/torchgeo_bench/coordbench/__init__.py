"""CoordBench: coordinate-only evaluation for torchgeo-bench.

Loads the unified ``taylor-geospatial/coordbench`` benchmark suite (point
``(lon, lat)`` -> label) and probes a frozen coordinate encoder with KNN and a
ridge linear head under random or spatial-block cross-validation. Label-informed
spatial priors are available through a separate runner.

Public API
----------
.. autoclass:: LocationEncoder
.. autoclass:: SinCosLocationEncoder
.. autoclass:: CoordBenchmark
.. autofunction:: load_benchmarks
.. autofunction:: run_coordbench
.. autofunction:: run_coordbench_priors
"""

from torchgeo_bench.coordbench.baselines import (
    ClassFrequencyPrior,
    GridPrior,
    KDEPrior,
    NearestNeighborPrior,
    SpatialPrior,
    UniformPrior,
)
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
    NeRFLocationEncoder,
    SatCLIPLocationEncoder,
    SinCosLocationEncoder,
    SINRLocationEncoder,
    SphericalHarmonicLocationEncoder,
    XYZLocationEncoder,
)
from torchgeo_bench.coordbench.prior_run import CoordPriorResult, run_coordbench_priors
from torchgeo_bench.coordbench.probe import (
    knn_probe_score,
    linear_probe_score,
    spatial_fold_ids,
)
from torchgeo_bench.coordbench.run import CoordResult, run_coordbench

__all__ = [
    "CoordBenchmark",
    "CoordResult",
    "CoordPriorResult",
    "LocationEncoder",
    "SinCosLocationEncoder",
    "MINDLocationEncoder",
    "ClimplicitLocationEncoder",
    "SINRLocationEncoder",
    "GeoCLIPLocationEncoder",
    "SatCLIPLocationEncoder",
    "XYZLocationEncoder",
    "SpatialPrior",
    "UniformPrior",
    "ClassFrequencyPrior",
    "GridPrior",
    "NearestNeighborPrior",
    "KDEPrior",
    "NeRFLocationEncoder",
    "SphericalHarmonicLocationEncoder",
    "list_benchmarks",
    "list_families",
    "load_benchmarks",
    "knn_probe_score",
    "linear_probe_score",
    "spatial_fold_ids",
    "run_coordbench",
    "run_coordbench_priors",
]
