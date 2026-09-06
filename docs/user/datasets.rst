Datasets
========

``torchgeo-bench`` supports two generations of GeoBench datasets — V1 and
V2 — plus wrappers around torchgeo's standalone EuroSAT and NWPU-RESISC45
datasets.  All datasets share the
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
   * - ``resisc45``
     - ``data/resisc45/``
     - torchgeo's ``RESISC45`` downloader

Downloading
-----------

The bundled :doc:`/api/cli` provides one subcommand per family:

.. code-block:: console

   $ torchgeo-bench download geobench_v1                              # full V1 bundle
   $ torchgeo-bench download geobench_v1 --datasets m-eurosat         # V1 subset
   $ torchgeo-bench download geobench_v2                              # default V2 set
   $ torchgeo-bench download geobench_v2 --datasets benv2,burn_scars  # V2 subset
   $ torchgeo-bench download eurosat                                  # torchgeo EuroSAT
   $ torchgeo-bench download resisc45                                 # torchgeo RESISC45
   $ torchgeo-bench download geobench_v2 --output-dir /scratch/data   # custom root

The default V2 download set is: ``benv2``, ``burn_scars``, ``caffe``,
``cloudsen12``, ``dynamic_earthnet``, ``flair2``, ``forestnet``,
``fotw``, ``kuro_siwo``, ``pastis``, ``so2sat``, ``spacenet2``,
``spacenet7``, ``treesatai``.

GeoBench V1 — classification
----------------------------

V1 datasets use the ``m-`` prefix on the command line.

The first time a V1 dataset is requested without a local copy under
``data/classification_v1.0`` or ``data/classification_v1.0_wds``, the
loader auto-downloads the requested dataset from the public mirror
``isaaccorley/geobenchv1-webdataset`` on the Hugging Face Hub. Both automatic and explicit V1 downloads use pinned revisions covered by the library's metadata checksums. Set
``GEOBENCH_V1_NO_HF_DOWNLOAD=1`` to disable the auto-download and force a
local-only path (``torchgeo-bench download geobench_v1`` still works for
the per-sample HDF5 layout).

V1 metadata format
^^^^^^^^^^^^^^^^^^

The published V1 datasets store metadata in HDF5 ``pickle`` attributes or ``<sample_id>.meta.pkl`` shard members. The readers accept these records only when their SHA-256 matches the expected checksum for that dataset and sample ID in the manifest bundled with ``torchgeo-bench``. The two pinned layouts contain identical metadata bytes and share these checksums. The readers unpickle the same in-memory bytes that were verified, without reopening the file. Unknown samples and checksum mismatches are errors; there is no opt-out or automatic approval of downloaded metadata.

The manifest comes from inspected, pinned reference releases, not from a checksum file accompanying the dataset. Its source revisions and archive hashes are recorded in ``torchgeo_bench/datasets/_v1_checksums/sources.json``. These checks authenticate the metadata records only, not the image arrays or partition files.

For custom data, V1 readers also accept data-only JSON metadata. HDF5 samples store it as a UTF-8 JSON string in the ``metadata_json`` attribute; tar shards pair ``<sample_id>.bands.npz`` with ``<sample_id>.meta.json``. JSON takes precedence when both formats are present. The JSON object must contain a numeric ``label`` (or a numeric list for multilabel data) and a non-empty ``bands_order`` list of source-band names.

.. code-block:: json

   {
     "label": 2,
     "bands_order": ["04 - Red", "03 - Green", "02 - Blue"],
     "04 - Red": {
       "transform": [10, 0, 500000, 0, -10, 5200000],
       "crs": "EPSG:32631"
     }
   }

Per-band ``transform`` and ``crs`` entries are optional. Geographic extraction accepts six affine coefficients ``[a, b, c, d, e, f]`` (or the nine-element affine matrix) and a CRS string; use ``null`` when georeferencing is unavailable. Python objects such as affine/CRS classes and NumPy label arrays must be exported as JSON numbers, lists, and strings by the data producer.

.. warning::

   Checksums do not make arbitrary pickle safe. A new release or differently serialized metadata needs independently reviewed reference checksums before it can be loaded as pickle. Do not generate approval hashes from an untrusted download; provide JSON metadata instead.

``experiments/scripts/repack_geobench_v1.py`` accepts JSON or checksum-approved HDF5 metadata and always writes JSON-based shards. Its ``--validate`` mode compares band arrays and normalized metadata using the same verification rules.

Supported V1 datasets
^^^^^^^^^^^^^^^^^^^^^

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

================== ====== ===== ============ ====================================== =====================================
CLI name           #cls   bands multilabel   sensor                                 Class
================== ====== ===== ============ ====================================== =====================================
``benv2``          19     14    **yes**      Sentinel-1 + Sentinel-2 (multi-modal)  :class:`~torchgeo_bench.datasets.BENV2`
``forestnet``      12     6     no           Sentinel-2                             :class:`~torchgeo_bench.datasets.Forestnet`
``so2sat``         17     12    no           Sentinel-1 + Sentinel-2                :class:`~torchgeo_bench.datasets.So2Sat`
``treesatai``      13     19    **yes**      Aerial + S2 + S1 (multi-modal)         :class:`~torchgeo_bench.datasets.TreeSatAI`
================== ====== ===== ============ ====================================== =====================================

V2 datasets are stored as a single ``.tortilla`` file each, hosted under
``aialliance/<name>`` on the Hugging Face Hub.  ``_V2Dataset.get_dataset``
passes ``download=True`` to the upstream class, so a missing tortilla is
fetched on first use — no separate ``torchgeo-bench download`` step
required for the sweep.

.. note::

   V1 and V2 share several CLI names but the underlying data are
   *different* datasets — different sensors, splits, label spaces, or
   normalisation conventions:

   * ``m-bigearthnet`` (V1) — original BigEarthNet, 43-class
     **multilabel** scene tags, S2 top-of-atmosphere DN.
     ``benv2`` (V2) — BigEarthNet v2.0 with 19-class multilabel and
     stacked S1 + S2.
   * ``m-so2sat`` (V1) — 18-band stack including LCZ context.
     ``so2sat`` (V2) — 10 S2 + 2 S1 bands stored at reflectance scale.
   * ``m-forestnet`` (V1) — Landsat-8 6-band uint8.
     ``forestnet`` (V2) — Sentinel-2 6-band uint8 with a different split
     than V1 despite the same channel count.

   The leaderboard prefixes panel titles with ``GeoBench V1`` /
   ``GeoBench V2`` so V1 and V2 results are never compared on the same
   axis.

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
``spacenet2``        2      9     WorldView 8-band + pan                       :class:`~torchgeo_bench.datasets.SpaceNet2`
``spacenet7``        2      3                                                  :class:`~torchgeo_bench.datasets.SpaceNet7`
==================== ====== ===== ============================================ ==============================================================

Other
-----

==================== ============================================================================
CLI name             Class
==================== ============================================================================
``eurosat``          :class:`~torchgeo_bench.datasets.EuroSAT`  (torchgeo wrapper, random split)
``eurosat-spatial``  :class:`~torchgeo_bench.datasets.EuroSATSpatial`  (longitude-based split)
``resisc45``         :class:`~torchgeo_bench.datasets.RESISC45`  (45-class aerial scenes, RGB)
==================== ============================================================================

``resisc45`` is 31,500 RGB scenes at 256x256 across 45 classes, on torchgeo's
published 60/20/20 split.  It is the most-divergent benchmark in the GFM
literature — the same Scale-MAE checkpoint is reported at 33.0 and 89.6
linear-probe accuracy by different papers under the same nominal protocol —
which is why it is worth running under a fixed harness.  The imagery carries
no geolocation, so it appears on the coverage map as an explicit gap rather
than being silently omitted.

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
