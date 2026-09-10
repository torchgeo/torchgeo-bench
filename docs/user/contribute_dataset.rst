Add a Dataset
=============

This page explains how to wire a new geospatial dataset into torchgeo-bench
so that any registered model can be evaluated on it automatically.

Prerequisites
-------------

Clone the repository and install the development dependencies:

.. code-block:: console

   $ git clone https://github.com/torchgeo/torchgeo-bench.git
   $ cd torchgeo-bench
   $ conda activate torchgeo-bench
   $ uv sync --extra dev

This is the same setup used in :doc:`eval_own_model`.  Download the dataset
files so you can test loading locally.  ``download`` takes a *family*, not an
individual dataset name; GeoBench families accept ``--datasets`` to narrow the
fetch:

.. code-block:: console

   $ torchgeo-bench download geobench_v2 --datasets burn_scars
   $ torchgeo-bench download resisc45

If you are adding a dataset that no existing family covers, you will wire up
its own download target below.

Implement BenchDataset
----------------------

Create a new module under :file:`src/torchgeo_bench/datasets/` and subclass
:class:`~torchgeo_bench.datasets.base.BenchDataset`:

.. code-block:: python

   from pathlib import Path
   from collections.abc import Callable

   from torchgeo_bench.datasets.base import BenchDataset, BandSpec

   class MyDataset(BenchDataset):
       name = "my_dataset"
       task = "classification"        # or "segmentation"
       num_classes = 10
       bands: list[BandSpec] = [...]
       rgb_bands = ["red", "green", "blue"]
       split_sizes = {"train": 5000, "val": 1000, "test": 2000}

       @classmethod
       def data_root(cls) -> Path:
           return Path("data/my_dataset")

       def get_dataset(
           self,
           split: str,
           *,
           partition: str = "default",
           bands: tuple[str, ...] | None = None,
           transform: Callable | None = None,
       ) -> torch.utils.data.Dataset:
           ...  # return a Dataset yielding {"image": tensor, "label": tensor} samples

Required class-level attributes:

* ``name`` — unique string identifier used by the dataset registry and CLI
* ``task`` — ``"classification"`` or ``"segmentation"``
* ``num_classes`` — integer label count
* ``bands`` — list of :class:`~torchgeo_bench.datasets.BandSpec` objects
  supplying per-channel sensor / wavelength / normalisation stats.  See
  `Compute the band statistics`_ — these must be measured, not copied.
* ``rgb_bands`` — short names of the bands used in RGB-only mode
* ``split_sizes`` — dict with ``train``, ``val``, and ``test`` keys
* ``multilabel`` — ``True`` for multi-hot labels (BigEarthNet, TreeSatAI).
  Selects micro-mAP over accuracy as the reported metric, so getting it wrong
  silently reports the wrong number.
* ``supports_partitions`` — ``True`` only for V1 GeoBench datasets, which ship
  partition JSON files.  When ``False``, ``get_datasets`` warns and ignores a
  non-default ``--partition``.

The ``get_dataset`` method takes ``split`` (``"train"``, ``"val"``, or
``"test"``) plus the keyword-only arguments ``partition``, ``bands`` (the
subset of bands requested by the model), and ``transform``.  It returns a
:class:`torch.utils.data.Dataset` whose ``__getitem__`` yields **dict**
samples — ``{"image": tensor, "label": tensor}`` for classification, with
``"mask"`` in place of ``"label"`` for segmentation.  Datasets always emit
raw float32 values; normalization is the model's job.

If you inherit from ``_V1Dataset`` or ``_V2Dataset``, both ``data_root`` and
``get_dataset`` are already implemented — your subclass is pure metadata.  See
:file:`src/torchgeo_bench/datasets/m_eurosat.py` for a minimal example.

.. note::

   **V1 vs V2 loader patterns.** V1 datasets (``m-`` prefix) read images
   directly from HDF5 files via
   :class:`~torchgeo_bench.datasets.geobench_v1._V1Dataset`.  V2 datasets use
   torchgeo dataset classes as the underlying loader and inherit from
   :class:`~torchgeo_bench.datasets.geobench_v2._V2Dataset`.  When adding a
   genuinely new dataset, prefer the V2 torchgeo pattern so the loader can
   participate in torchgeo's transform pipeline.

Band selection when the loader has no ``bands`` argument
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The runner passes ``get_dataset`` the band subset a model asked for, and then
checks that the loaded tensor's channel count matches the ``BandSpec`` list it
built.  How you honour that subset depends on the upstream loader:

* **The loader accepts bands.** Forward them and you are done —
  :class:`~torchgeo_bench.datasets.EuroSAT` passes ``source_name`` codes
  straight to :class:`torchgeo.datasets.EuroSAT`.
* **The loader does not.** Most torchgeo classification datasets are
  fixed-channel ``ImageFolder`` wrappers with no band argument at all, so the
  subset has to be applied in *your* wrapper, as a transform that indexes the
  channel axis.  :class:`~torchgeo_bench.datasets.RESISC45` is the worked
  example: see ``_make_band_select`` and ``_compose`` in
  :file:`src/torchgeo_bench/datasets/resisc45.py`.

Two details matter in the second case.  Compose your selection **before** the
``transform`` the caller handed you — that argument is the resize built by
:func:`~torchgeo_bench.datasets.get_datasets`, and it should only see channels
that survive selection.  And return ``None`` rather than an identity transform
when the selection is a no-op, so the common ``bands=rgb`` / ``bands=all``
paths add no per-sample work.

Compute the band statistics
---------------------------

Each :class:`~torchgeo_bench.datasets.BandSpec` carries ``mean``, ``std``,
``min``, and ``max``.  These are not decoration: they back ``bandspec_zscore``,
the benchmark's default normalisation strategy, and in raw units they are what
``detect_input_unit`` reads to decide whether a dataset is Sentinel-2 DN,
uint8, or reflectance under ``model_native``.  Wrong statistics mis-normalize
every model evaluated on your dataset, and nothing will fail loudly.

So measure them.  Two rules:

* **Train split only.**  Statistics that include val or test leak evaluation
  data into the normalisation every model sees.
* **Raw sensor units.**  Do not pre-scale to ``[0, 1]``; unit detection depends
  on the raw magnitudes.

Once your class is registered with placeholder statistics and loads, run:

.. code-block:: console

   $ python scripts/compute_band_statistics.py --dataset my_dataset

It accumulates in float64 over the train split and prints a paste-ready
``bands = [...]`` block.  Record in a comment that the numbers came from this
script, so the next person knows they are measured rather than copied from a
paper.

Register and configure
----------------------

**1. Register the class** by adding an entry to ``_REGISTRY_SPEC`` in
:file:`src/torchgeo_bench/datasets/loading.py`, mapping the dataset name to
its ``(submodule, class_name)``:

.. code-block:: python

   _REGISTRY_SPEC: dict[str, tuple[str, str]] = {
       ...,
       "my_dataset": ("my_dataset", "MyDataset"),
   }

This is the step that actually makes the dataset available — it backs
``get_bench_dataset_class``, :func:`~torchgeo_bench.datasets.list_datasets`,
and the CLI.  There is no per-dataset config file; datasets are selected by
name on the command line (``--datasets``). The registry is kept as
module/class-name strings rather than imported classes so that importing
``loading`` stays cheap; ``get_bench_dataset_class`` imports only the one
module it needs.

**2. Export the class** from :file:`src/torchgeo_bench/datasets/__init__.py`
by adding an ``__all__`` entry and a matching ``_LAZY_CLASSES`` mapping:

.. code-block:: python

   __all__ = [
       ...,
       "MyDataset",
   ]

   _LAZY_CLASSES: dict[str, str] = {
       ...,
       "MyDataset": "my_dataset",
   }

Individual dataset classes load lazily through module ``__getattr__`` so that
``import torchgeo_bench.datasets`` — and CLI startup — stays fast.  Do not add
an eager ``from .my_dataset import MyDataset`` import; register the name in
``_LAZY_CLASSES`` instead.

Keep both lists alphabetically sorted; they are read by humans far more often
than by the loader.

**3. Wire up the download.** Which files you touch depends on the family:

* **GeoBench V2** — add the name → upstream class mapping to ``_V2_REGISTRY``
  in :file:`src/torchgeo_bench/datasets/geobench_v2.py`.
  ``DEFAULT_V2_DATASETS`` is derived from that registry automatically.
* **A torchgeo wrapper or anything else** — add a ``download_<name>`` function
  to :file:`src/torchgeo_bench/download.py`, then add your target to the
  ``choices=`` list of the ``download`` subparser and dispatch to it in
  ``_cmd_download``, both in :file:`src/torchgeo_bench/cli.py`.  Without the
  ``choices`` entry the CLI rejects your target name.  ``download_resisc45``
  is the reference implementation.

**4. Add the expected split sizes** to ``EXPECTED_SIZES`` in
:file:`tests/test_split_sizes.py`.  Those cases are marked
``@pytest.mark.slow`` and the default ``addopts`` deselect them, so verify
yours against the data on disk explicitly:

.. code-block:: console

   $ uv run pytest tests/test_split_sizes.py -m slow -k my_dataset

**5. Document it.** Three files, none optional:

* :file:`docs/user/datasets.rst` — a row in the relevant family table, plus
  the filesystem-layout table and the ``download`` command block if you added
  a new target.
* :file:`docs/api/datasets.rst` — an ``.. autoclass::`` entry, or your class
  gets no API page.
* :file:`docs/user/changelog.rst` — an entry under ``Unreleased``.

Run the smoke test
------------------

With the dataset on disk, run a quick benchmark to verify the dataset loads
and produces sensible results:

.. code-block:: console

   $ torchgeo-bench run -m timm/resnet50 -d my_dataset --skip-linear --bootstrap 10

The config default is ``device: cuda:0``.  On a machine without a GPU, pass
``--device cpu`` or the run fails inside feature extraction with a bare CUDA
driver error rather than anything that names the real problem.  ``-m
imagestats`` is a useful first pass either way: it is a 12-dimensional colour
baseline that runs in seconds, so a score comfortably above chance tells you
the plumbing is right before you spend time on a real backbone.

Write down the chance level for your dataset and compare against it.  A
45-class dataset where ``imagestats`` scores 0.34 against a 0.022 chance level
is loading correctly; one that scores 0.02 is not.

Add the geographic metadata
---------------------------

Every registered dataset carries a record in the committed geographic store
under :file:`docs/_static/_dataset_geography/`, which drives the spatial
coverage map.  Generate yours:

.. code-block:: console

   $ python experiments/scripts/extract_dataset_geography.py --dataset my_dataset

Commit the resulting :file:`docs/_static/_dataset_geography/my_dataset.json`
together with the regenerated :file:`index.json`.

:file:`tests/test_geography.py` fails when a registered dataset has no record,
so this step is not optional — but "no coordinates" is a perfectly valid
answer.  If the dataset genuinely carries no geolocation, add it to ``NO_GEO``
in :file:`src/torchgeo_bench/geography.py` with the reason you verified, and
the map will disclose the gap rather than quietly omit the dataset.

Coordinates are read from the raw files (V2 ``.tortilla`` metadata columns, V1
HDF5 affine transforms).  A dataset stored in some other layout needs a branch
in ``extract_geography``.

Once results look sensible, follow the PR workflow described in
:doc:`contribute_model` to open a pull request.

A worked example
----------------

`#234 <https://github.com/torchgeo/torchgeo-bench/pull/234>`__ adds NWPU-RESISC45
and touches every step on this page: a torchgeo wrapper whose loader takes no
``bands`` argument, measured band statistics, its own download target, a
``no_geo`` record with the check that justified it, and unit tests that run
without the data on disk.  Read
:file:`src/torchgeo_bench/datasets/resisc45.py` alongside this guide.
