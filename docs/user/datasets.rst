Datasets
========

``torchgeo-bench`` supports two generations of GeoBench datasets — V1 and
V2 — plus a small wrapper around torchgeo's standalone EuroSAT dataset
for sanity checks.  All datasets share the
:class:`~torchgeo_bench.datasets.BenchDataset` interface and are
auto-registered on import so they can be selected by their CLI name.

Filesystem layout
-----------------

All data lives under ``./data/`` relative to the current working
directory.  Paths are **fixed** — the runner does not honour environment
variables like ``GEOBENCH_ROOT``; if you keep data elsewhere, symlink
``data/`` to the real location.

.. list-table::
   :header-rows: 1
   :widths: 15 35 50

   * - Family
     - Default destination
     - Source
   * - ``geobench_v1``
     - ``data/classification_v1.0/``
     - Hugging Face ``recursix/geo-bench-1.0``
   * - ``geobench_v2``
     - ``data/geobenchv2/<name>/``
     - Hugging Face ``aialliance/<name>``
   * - ``eurosat``
     - ``data/eurosat/``
     - torchgeo's ``EuroSAT`` downloader

Downloading
-----------

The bundled :doc:`/api/cli` provides one subcommand per family:

.. code-block:: console

   $ torchgeo-bench download geobench_v1                              # full V1 bundle
   $ torchgeo-bench download geobench_v2                              # default V2 set
   $ torchgeo-bench download geobench_v2 --datasets benv2,burn_scars  # V2 subset
   $ torchgeo-bench download eurosat                                  # torchgeo EuroSAT
   $ torchgeo-bench download geobench_v2 --output-dir /scratch/data   # custom root

The default V2 download set is: ``benv2``, ``burn_scars``, ``caffe``,
``cloudsen12``, ``dynamic_earthnet``, ``flair2``, ``forestnet``,
``fotw``, ``kuro_siwo``, ``pastis``, ``so2sat``, ``spacenet2``,
``spacenet7``, ``treesatai``.

GeoBench V1 — classification
----------------------------

V1 datasets use the ``m-`` prefix on the command line.

.. list-table::
   :header-rows: 1
   :widths: 18 8 8 12 22 32

   * - CLI name
     - #cls
     - bands
     - multilabel
     - sensor
     - Class
   * - ``m-bigearthnet``
     - 43
     - 12
     - **yes**
     - Sentinel-2
     - :class:`~torchgeo_bench.datasets.MBigEarthNet`
   * - ``m-brick-kiln``
     - 2
     - 13
     - no
     - Sentinel-2
     - :class:`~torchgeo_bench.datasets.MBrickKiln`
   * - ``m-eurosat``
     - 10
     - 13
     - no
     - Sentinel-2
     - :class:`~torchgeo_bench.datasets.MEurosat`
   * - ``m-forestnet``
     - 12
     - 6
     - no
     - Landsat
     - :class:`~torchgeo_bench.datasets.MForestnet`
   * - ``m-pv4ger``
     - 2
     - 3
     - no
     - Aerial RGB
     - :class:`~torchgeo_bench.datasets.MPv4ger`
   * - ``m-so2sat``
     - 17
     - 18
     - no
     - Sentinel-1 + S2
     - :class:`~torchgeo_bench.datasets.MSo2Sat`

Multi-label datasets (``m-bigearthnet``) report the ``micro_mAP`` metric
instead of accuracy.

GeoBench V2 — classification
----------------------------

================== ====== ===== ====================================== =====================================
CLI name           #cls   bands sensor                                 Class
================== ====== ===== ====================================== =====================================
``benv2``          19     14    Sentinel-1 + Sentinel-2 (multi-modal)  :class:`~torchgeo_bench.datasets.BENV2`
``forestnet``      12     6     Sentinel-2                             :class:`~torchgeo_bench.datasets.Forestnet`
``so2sat``         17     12    Sentinel-1 + Sentinel-2                :class:`~torchgeo_bench.datasets.So2Sat`
``treesatai``      13     19    Aerial + S2 + S1 (multi-modal)         :class:`~torchgeo_bench.datasets.TreeSatAI`
================== ====== ===== ====================================== =====================================

.. note::

   ``m-forestnet`` and ``forestnet`` are *different* datasets.  V1 uses
   Landsat with 6 bands; V2 uses Sentinel-2 with the same number of
   bands but a different sensor and split.

GeoBench V2 — segmentation
--------------------------

==================== ====== ===== ============================================ ==============================================================
CLI name             #cls   bands notes                                        Class
==================== ====== ===== ============================================ ==============================================================
``burn_scars``       3      6                                                  :class:`~torchgeo_bench.datasets.BurnScars`
``caffe``            4      1     aerial grayscale                             :class:`~torchgeo_bench.datasets.CaFFe`
``cloudsen12``       4      12                                                 :class:`~torchgeo_bench.datasets.CloudSEN12`
``dynamic_earthnet`` 7      16                                                 :class:`~torchgeo_bench.datasets.DynamicEarthNet`
``flair2``           13     5     aerial + Sentinel-2                          :class:`~torchgeo_bench.datasets.FLAIR2`
``fotw``             4      4     Fields of the World                          :class:`~torchgeo_bench.datasets.FieldsOfTheWorld`
``kuro_siwo``        4      3     SAR ``vv`` / ``vh`` + DEM (no RGB triplet)   :class:`~torchgeo_bench.datasets.KuroSiwo`
``pastis``           20     16    Sentinel-2 + Sentinel-1 (multi-modal)        :class:`~torchgeo_bench.datasets.PASTIS`
``spacenet2``        3      9     WorldView 8-band + pan                       :class:`~torchgeo_bench.datasets.SpaceNet2`
``spacenet7``        3      3                                                  :class:`~torchgeo_bench.datasets.SpaceNet7`
==================== ====== ===== ============================================ ==============================================================

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

The runner derives ``num_channels`` from the loaded tensor and constructs
the matching ``list[BandSpec]`` so the model wrapper can size its input
layer and per-channel normalization correctly.  The selected ``bands``
value is recorded in the results CSV so multiple runs writing to the same
file (and ``resume=true``) keep RGB and multispectral results
distinguishable.

.. code-block:: console

   $ # All 13 Sentinel-2 bands on EuroSAT with a pretrained timm ResNet-18
   $ torchgeo-bench run model=timm/resnet18 dataset.names=[m-eurosat] dataset.bands=all

Multi-modality (V2)
-------------------

Several V2 datasets are multi-sensor (e.g. ``treesatai`` = aerial + S2 +
S1, ``pastis`` = S2 + S1, ``kuro_siwo`` = SAR + DEM).  Their wrappers
set ``band_order_strategy = "by_sensor"`` and the V2 base class groups
``BandSpec`` entries by sensor before passing them to the upstream
``geobench_v2`` loader.  End users do not need to do anything special —
set ``dataset.bands=all`` (or an explicit subset) and the right
per-modality tensors are concatenated into a single ``image`` key.

Model compatibility
^^^^^^^^^^^^^^^^^^^

* :doc:`timm <models>` wrappers rebuild the input conv for any
  ``num_channels``.
* :class:`~torchgeo_bench.models.RCFBench` and
  :class:`~torchgeo_bench.models.ImageStatsBench` are band-agnostic.
* The torchgeo RGB-only wrappers hold fixed-channel pretrained weights
  and don't currently adapt to non-RGB inputs — see
  `#16 <https://github.com/torchgeo/torchgeo-bench/issues/16>`__.
* :class:`~torchgeo_bench.models.TorchGeoDOFABench` accepts variable
  channels via wavelength tokens but the current wrapper hard-codes
  Sentinel-2 RGB wavelengths — see
  `#15 <https://github.com/torchgeo/torchgeo-bench/issues/15>`__.

Data partitions (V1 only)
-------------------------

V1 datasets honour the ``dataset.partition`` argument (which selects one
of the partition JSON files distributed with each dataset).  V2 datasets
ignore it.

.. code-block:: console

   $ # Train on 1% of the V1 training split, write to a separate CSV
   $ torchgeo-bench run dataset.partition=0.01x_train output=results/1pct.csv

Common partition values: ``default``, ``0.01x_train``, ``0.02x_train``,
``0.05x_train``, ``0.10x_train``, ``0.20x_train``, ``0.50x_train``,
``1.00x_train``.  The exact set available depends on which partition
JSON files ship with the dataset.
