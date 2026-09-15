Add a Dataset
=============

Dataset definitions are lightweight, immutable records. Runtime readers, sample conversion, preprocessing, and caller-owned batching are separate.

Prerequisites
-------------

Install with ``uv sync --extra dev`` and use ``uv run`` for commands, or activate the conda environment and install with ``pip install -e ".[dev]"``. These are separate environments; do not use ``uv sync`` to install into conda.

Download explicitly before loading. Individual names may span families:

.. code-block:: console

   $ uv run torchgeo-bench download m-eurosat burn_scars resisc45
   $ uv run torchgeo-bench download geobench_v2 --datasets burn_scars

Define the scientific metadata
------------------------------

Create a module under :file:`src/torchgeo_bench/datasets/` containing a ``SPEC``. Use frozen nested records and tuples, not mutable class attributes or dictionaries. For example, a new V2 definition has this structure (replace the illustrative bands, counts, and upstream class with verified metadata):

.. code-block:: python

   from torchgeo_bench.bands import BandSpec
   from torchgeo_bench.datasets.spec import DatasetSpec, SplitSizes, V2Source

   SPEC = DatasetSpec(
       name="my_dataset",
       task="classification",
       num_classes=10,
       multilabel=False,
       bands=(
           BandSpec("aerial", "red", "R", mean=100, std=40, min=0, max=255),
           BandSpec("aerial", "green", "G", mean=100, std=40, min=0, max=255),
           BandSpec("aerial", "blue", "B", mean=100, std=40, min=0, max=255),
       ),
       rgb_bands=("red", "green", "blue"),
       split_sizes=SplitSizes(train=5000, val=1000, test=2000),
       source=V2Source("GeoBenchMyDataset"),
   )

The canonical ``name`` identifies a benchmark, not necessarily a unique archive. Use ``source.storage_name`` when another benchmark shares the same source files. For example, standard/spatial TorchGeo EuroSAT share imagery and band objects, but their upstream classes select different splits. Do not merge their scientific identities, or replace GeoBench V1 EuroSAT's separate statistics.

Declare the exact task, class count, and multilabel semantics from verified source metadata. ``multilabel=True`` selects micro-mAP rather than accuracy (``m-bigearthnet``, ``benv2``, and ``treesatai``). Do not infer a class vocabulary from a count or change labels while moving metadata. ``split_sizes`` describes the default partition; a requested subset may have different lengths.

``BandSpec`` stores source names, sensor tags, statistics, and available wavelengths in tensor order. ``rgb_bands`` currently preserves the benchmark's existing selector, including CaFFe grayscale and KuroSiwo SAR. Do not change those choices incidentally. Normalization remains model-owned.

Source policies and capabilities
-------------------------------

* ``V1Source`` selects verified JSON shards, with a retained custom JSON-metadata HDF5 fallback. Pickle metadata is not supported. ``DatasetCapabilities(supports_partitions=True)`` enables non-default partitions.
* ``V2Source`` contains the upstream class identifier, split-name policy, and typed options. Set ``band_order_strategy="by_sensor"`` when the upstream reader groups channels by modality. Declare an existing named sample adapter for nonstandard image or mask keys; do not import runtime adapter functions into the definition module.
* ``TorchGeoSource`` identifies the upstream reader, fixed root, optional shared storage name, and download-checksum policy.

``DatasetCapabilities(multi_temporal=True)`` enables explicit ``time_steps`` (currently PASTIS). Unsupported partitions, splits, temporal requests, and band selections must fail before source construction or filesystem access. Preserve the common V2 temporal path rather than adding a PASTIS-specific loader.

Readers receive the same ``ResolvedInput`` returned to callers and derive source requests from ``inputs.bands`` without resolving names again. The source must emit channels in that order, even when its backend stacks in a different sensor order. Preserve sensor alignment, acquisition selection, and categorical mask semantics.

Each reader loads only the requested split, with downloading disabled. Samples use ``image`` plus ``label`` for classification, or ``mask`` for segmentation. Images retain raw float32 values. Single-acquisition inputs are CHW; supported explicit multi-step inputs are TCHW. No sample probing is needed for metadata. The public ``LoadedSplit`` contains its authoritative ``spec`` and ``input``; callers construct DataLoaders and pass ``list(loaded.bands)`` to model construction.

For readers without a band argument, apply channel selection before resizing. The RESISC45 implementation in :file:`src/torchgeo_bench/datasets/torchgeo.py` demonstrates this, including avoiding a copy for identity selection.

Compute the band statistics
--------------------------

Measure ``mean``, ``std``, ``min``, and ``max`` over the **train split only**, in **raw sensor units**. Validation/test statistics would leak evaluation data. Statistics drive dataset normalization and model-native unit detection.

.. code-block:: console

   $ uv run python scripts/compute_band_statistics.py --dataset my_dataset

The script accumulates in float64. Copy its measured BandSpec values into the definition's tuple, and retain a short provenance comment. Do not infer missing physical calibration or borrow another dataset's statistics.

Register, download, and document
-------------------------------

Import the definition module in :file:`src/torchgeo_bench/datasets/catalog.py` and include its ``SPEC`` in ``_make_catalog``. The catalog derives IDs from the records; do not add another name/task/routing registry or export wrapper classes. ``list_datasets()``, ``get_dataset_spec(name)``, CLI details, config validation, FLOPs metadata, downloads, and geography all use this catalog.

Existing V2 download coverage is derived automatically, for both classification and segmentation. Shared TorchGeo downloads include the sibling split definitions once per storage identity. A new source family requires an explicit typed source record and runtime/download factory, not a generic kwargs bag in the spec. Definitions and full catalog lookups must never import Torch, TorchGeo, upstream readers, model weights, or samples.

Use the existing dataset catalog tests as the parity pattern. The committed :file:`tests/fixtures/dataset_metadata.json` freezes the 23 definitions before the spec migration; do not regenerate it from changed metadata to conceal a scientific change. Add fixture-backed tests for genuinely new definitions. Verify split counts against local data using the slow split-size tests.

Update :file:`docs/user/datasets.rst`, relevant API documentation, and :file:`docs/user/changelog.rst` for a new dataset.

Validate the loading contract
----------------------------

Add fast tests with tiny local files for requested-split isolation, input-option validation, channel values and metadata identity/order, raw dtype, targets, source-specific adapters, and visible missing-data failures. Keep real-data tests marked ``slow``. Exercise applicable offline integrations.

.. code-block:: console

   $ uv run torchgeo-bench datasets my_dataset
   $ uv run torchgeo-bench run --model imagestats --dataset my_dataset --device cpu \
       --methods knn --bootstrap-samples 10 --dry-run

For classification, remove ``--dry-run`` once local data is ready. For segmentation, use a spatial backbone with compatible feature layers and head instead of ``imagestats``; inspect masks and ignore labels on a small sample. See :doc:`segmentation-layers`. Do not treat above-chance accuracy as proof of correct loading; assert channels and labels directly.

Add geographic metadata
-----------------------

Every registered dataset needs a JSON record and index entry under :file:`docs/_static/_dataset_geography/`. If coordinates are unavailable, declare the verified explanation in ``GeographySpec(reason=...)``. If a re-split shares imagery, declare ``GeographySpec(alias_of=...)``. Geography routing derives from the source record; do not add a class check or a separate identity list.

.. code-block:: console

   $ uv run python experiments/scripts/extract_dataset_geography.py --dataset my_dataset

Commit the new record and regenerated index. Existing records must keep exact catalog coverage and reasons for absent coordinates; :file:`tests/test_geography.py` checks these requirements. V2 reads tortilla coordinate metadata; V1 reads JSON affine/CRS metadata from shards or custom HDF5. A different format needs an explicit extraction implementation.

See :file:`src/torchgeo_bench/datasets/resisc45.py` and its runtime factory for a TorchGeo example, and :doc:`contribute_model` for the PR workflow.
