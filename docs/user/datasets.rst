Datasets
========

``torchgeo-bench`` supports two generations of GeoBench datasets — V1 and
V2 — plus a small wrapper around torchgeo's standalone EuroSAT dataset
for sanity checks.  All datasets share the
:class:`~torchgeo_bench.datasets.BenchDataset` interface and are
auto-registered on import so they can be selected by their CLI name.

Filesystem layout
-----------------

By default datasets are loaded from ``./data/`` relative to the current
working directory:

.. code-block:: text

   data/
   ├── geobench-1.0/        # V1 HDF5 distributions
   ├── geobench-v2/         # V2 datasets (one subdir per dataset)
   └── EuroSAT/             # torchgeo's EuroSAT mirror

Override via environment variables when your data lives elsewhere:

.. code-block:: console

   $ export GEOBENCH_ROOT=/data/shared/geobench-1.0
   $ export GEOBENCH_V2_ROOT=/data/shared/geobench-v2

Downloading
-----------

The :doc:`/api/cli` provides a one-shot downloader for each family:

.. code-block:: console

   $ torchgeo-bench download                              # GeoBench V1 (full bundle)
   $ torchgeo-bench download --version v2                 # all V2 datasets
   $ torchgeo-bench download --version v2 --datasets benv2,treesatai

GeoBench V1 — classification
----------------------------

V1 datasets use the ``m-`` prefix on the command line.

============================  ====================================================
CLI name                      Class
============================  ====================================================
``m-bigearthnet``             :class:`~torchgeo_bench.datasets.MBigEarthNet`
``m-brick-kiln``              :class:`~torchgeo_bench.datasets.MBrickKiln`
``m-eurosat``                 :class:`~torchgeo_bench.datasets.MEurosat`
``m-forestnet``               :class:`~torchgeo_bench.datasets.MForestnet`
``m-pv4ger``                  :class:`~torchgeo_bench.datasets.MPv4ger`
``m-so2sat``                  :class:`~torchgeo_bench.datasets.MSo2Sat`
============================  ====================================================

GeoBench V2 — classification
----------------------------

================== ====================================================
CLI name           Class
================== ====================================================
``benv2``          :class:`~torchgeo_bench.datasets.BENV2`
``forestnet``      :class:`~torchgeo_bench.datasets.Forestnet`
``so2sat``         :class:`~torchgeo_bench.datasets.So2Sat`
``treesatai``      :class:`~torchgeo_bench.datasets.TreeSatAI`
================== ====================================================

GeoBench V2 — segmentation
--------------------------

==================== ===============================================================
CLI name             Class
==================== ===============================================================
``burn_scars``       :class:`~torchgeo_bench.datasets.BurnScars`
``caffe``            :class:`~torchgeo_bench.datasets.CaFFe`
``cloudsen12``       :class:`~torchgeo_bench.datasets.CloudSEN12`
``dynamic_earthnet`` :class:`~torchgeo_bench.datasets.DynamicEarthNet`
``flair2``           :class:`~torchgeo_bench.datasets.FLAIR2`
``fotw``             :class:`~torchgeo_bench.datasets.FieldsOfTheWorld`
``kuro_siwo``        :class:`~torchgeo_bench.datasets.KuroSiwo`
``pastis``           :class:`~torchgeo_bench.datasets.PASTIS`
``spacenet2``        :class:`~torchgeo_bench.datasets.SpaceNet2`
``spacenet7``        :class:`~torchgeo_bench.datasets.SpaceNet7`
==================== ===============================================================

Other
-----

============  ========================================================
CLI name      Class
============  ========================================================
``eurosat``   :class:`~torchgeo_bench.datasets.EuroSAT`  (torchgeo wrapper)
============  ========================================================

Selecting datasets
------------------

Pass a single dataset, a comma-separated list, or ``all`` to evaluate every
registered dataset:

.. code-block:: console

   $ torchgeo-bench run dataset.names=[m-eurosat]
   $ torchgeo-bench run dataset.names=[burn_scars,pastis,flair2]
   $ torchgeo-bench run dataset.names=all

Bands selection
---------------

Each dataset declares an ordered list of :class:`~torchgeo_bench.datasets.BandSpec`
objects.  Three modes are supported:

* ``dataset.bands=rgb`` *(default)* — only the bands listed in
  :attr:`~torchgeo_bench.datasets.BenchDataset.rgb_bands`.
* ``dataset.bands=all`` — every band the dataset exposes.
* ``dataset.bands=[red,green,blue,nir]`` — an explicit subset.
