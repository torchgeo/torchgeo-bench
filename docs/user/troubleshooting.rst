Troubleshooting
===============

``Dataset directory not found`` / files missing
-----------------------------------------------

Datasets must live under ``./data/`` from the directory where you run
``torchgeo-bench``.  The runner does **not** honour ``GEOBENCH_ROOT``
or ``GEOBENCH_V2_ROOT`` environment variables — paths are fixed:

* V1: :file:`data/classification_v1.0/<name>/`
* V2: :file:`data/geobenchv2/<name>/`
* EuroSAT: :file:`data/eurosat/`

Re-run ``torchgeo-bench download …`` to fetch missing data.  If your
data lives elsewhere, symlink ``data/`` to the real location.

``ModuleNotFoundError: geobench``
---------------------------------

The legacy ``geobench`` package is no longer a dependency.  V1 datasets
are read directly from HDF5 (the internal ``GeoBenchv1`` loader in
:file:`src/torchgeo_bench/datasets/geobench_v1.py`); V2 dispatches to
upstream ``geobench_v2.datasets.GeoBench<X>``.  Make sure your
environment matches the pinned ``geobenchv2`` version in
:file:`pyproject.toml`.

CUDA out of memory
------------------

.. code-block:: console

   $ torchgeo-bench run dataset.batch_size=32
   $ # or run on CPU
   $ torchgeo-bench run device=cpu

For segmentation, also try

.. code-block:: console

   $ torchgeo-bench run \
       eval.segmentation.cache_dtype=float32 \
       eval.segmentation.cache_features=false

if RAM (rather than GPU memory) is the bottleneck.

GPU run crashes immediately
---------------------------

The default config is ``device: cuda:0``, so the first documented run uses the
GPU.  ``uv sync`` installs the latest ``torch``, whose bundled CUDA and kernel
architectures may not match your GPU or driver.  Two distinct failures:

* ``RuntimeError: The NVIDIA driver on your system is too old`` — the installed
  ``torch`` was built against a newer CUDA than your driver supports.
* ``CUDA error: no kernel image is available for execution on the device``
  (``cudaErrorNoKernelImageForDevice``) — torch's CUDA 12.8 wheels dropped
  Volta (``sm_70``) kernels, so they fail on a V100 *even though* CUDA
  initialises.  Older GPUs need a **cu126** (or earlier) build, e.g.
  ``torch==2.7.1+cu126`` + ``torchvision==0.22.1+cu126`` from
  ``https://download.pytorch.org``.

Either way you can fall back to CPU (slower, but always works):

.. code-block:: console

   $ torchgeo-bench run dataset.names=[m-eurosat] device=cpu

CPU is fine for the small V1 splits, but large V2 datasets (e.g. ``benv2`` /
BigEarthNet) can take far longer — prefer a working GPU for those.

``KeyError: 's2'`` on a V2 dataset
----------------------------------

A known V2 issue: ``geobench_v2.rearrange_bands`` expects modality keys
(``'s2'``, ``'s1'``, …) that aren't present when a flat band list is
requested.  Workaround: use ``dataset.bands=all`` for affected V2
datasets.

``eurosat-spatial`` reports ``Dataset not found``
-------------------------------------------------

``torchgeo-bench download eurosat`` fetches EuroSAT plus the standard
``eurosat-{train,val,test}.txt`` splits, but the ``eurosat-spatial`` dataset
uses ``torchgeo.datasets.EuroSATSpatial``, which needs its own *spatial* split
files.  Those download automatically on the first CLI run that uses
``eurosat-spatial``; the plain ``download eurosat`` command does not provision
them, so its slow test skips until that first run.

V1 slow tests skip after the auto-download
------------------------------------------

The per-dataset auto-download (triggered by running a V1 dataset such as
``dataset.names=[m-eurosat]``) writes the webdataset layout under
``data/classification_v1.0_wds/``.  The V1 *slow* integration tests instead
read the legacy HDF5 layout under ``data/classification_v1.0/`` and skip if only
the ``_wds`` data is present.  Fetch the legacy bundle with
``torchgeo-bench download geobench_v1`` to run them.

Build / docs warnings
---------------------

If you build the docs locally without internet access, expect ~9
``WARNING: failed to reach any of the inventories`` messages from
``sphinx.ext.intersphinx``.  These are network reachability errors, not
real issues — the GitHub Pages build runner has network access and
resolves these inventories cleanly.