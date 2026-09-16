torchgeo_bench.datasets
=======================

.. module:: torchgeo_bench.datasets

Every benchmark identity has an immutable :class:`DatasetSpec`. The definition owns its task, class count, multilabel flag, ordered bands and statistics, default and optional genuine RGB sets, default split sizes, source policies, capabilities, and geography aliases. Per-dataset modules contain ``SPEC`` records; the read-only catalog derives names and routing from those records, without a second identity registry.

Definitions and catalog lookups do not import Torch, TorchGeo, upstream GeoBench readers, or models, and never read samples. ``torchgeo-bench datasets NAME`` shows the complete definition as YAML.

Metadata API
------------

.. autofunction:: get_dataset_spec
.. autofunction:: get_dataset_task
.. autofunction:: list_datasets
.. autofunction:: list_v2_datasets
.. autoclass:: DatasetSpec
   :members: rgb_bands
   :undoc-members:

.. autodata:: torchgeo_bench.datasets.spec.Task
.. autodata:: torchgeo_bench.datasets.spec.Split
.. autodata:: torchgeo_bench.datasets.spec.DatasetSource

.. autoclass:: BandSpec
.. autoclass:: SplitSizes
.. autoclass:: DatasetCapabilities
.. autoclass:: GeographySpec
.. autoclass:: V1Source
.. autoclass:: V2Source
.. autoclass:: TorchGeoSource

``DatasetSpec.resolve_band_specs()`` translates ``rgb``, ``default``, ``all``, or ordered explicit names into the original frozen ``BandSpec`` objects. Band collections and selections are tuples. ``default_bands`` declares the reduced inputs independently of optional ``rgb_bands``. CaFFe defaults to gray and KuroSiwo to vv/vh; both have ``rgb_bands=None``. Omitted or explicit ``rgb`` fails for these datasets before data access. It never synthesizes RGB or falls back to ``default``.

``resolve_input(spec_or_name, bands="rgb", partition="default", time_steps=None)`` validates source capabilities and resolves ordered inputs without importing readers. Its immutable ``ResolvedInput`` exposes ``selection``, ``band_names``, ``bands``, ``partition``, ``time_steps``, ``layout``, a JSON-serializable ``description``, and its SHA-256 ``fingerprint``. The description includes protocol version 2, canonical dataset/target identity, source identity and adapter version, split/partition policy, every selected raw BandSpec field, and temporal/acquisition settings. Unselected band statistics, runtime datasets, samples, and installed dependency versions are not fingerprinted.

.. autofunction:: resolve_input

Loading API
-----------

``load_split(spec_or_name, split, ...)`` loads only the requested split and returns a ``LoadedSplit``. Its ``spec`` is the authoritative definition, ``dataset`` is the PyTorch Dataset, and ``bands`` exposes the original ordered ``BandSpec`` objects used by the source and model. ``input`` describes the selection and actual layout: CHW for ordinary or single-step inputs, TCHW for explicit multi-step PASTIS inputs. Convenience properties expose ``dataset_name``, ``task``, ``num_classes``, ``multilabel``, and ``target_key`` from the spec.

Callers own DataLoaders, including shuffle, workers, pinning, and seeds:

.. code-block:: python

   from torch.utils.data import DataLoader
   from torchgeo_bench.datasets import get_dataset_spec, load_split

   spec = get_dataset_spec("m-eurosat")
   train = load_split(spec, "train", bands="rgb", image_size=64)
   assert train.spec is spec
   loader = DataLoader(train.dataset, batch_size=32, shuffle=True, num_workers=0)
   # Supply list(train.bands) to build_model; do not select model bands again.

Preflight and loading can reuse the exact same resolved metadata:

.. code-block:: python

   from torchgeo_bench.datasets import load_split, resolve_input

   inputs = resolve_input("caffe", bands="default")
   train = load_split("caffe", "train", bands="default", inputs=inputs)
   assert train.input is inputs
   assert train.input.band_names == ("gray",)

When supplying ``inputs``, the dataset and selection/partition/temporal options must match; conflicting options fail instead of silently overriding preflight. The image runner reuses its training resolution before checking resume and when constructing the source and model.

Request validation/test explicitly when needed. The image runner applies a non-default partition only to training; library callers can request a supported partition on any split. Unsupported options fail before source construction. There are no automatic downloads or sample probes in ``load_split``. Runtime split objects are not configuration or resume-hash payloads.

.. autofunction:: load_split
.. autoclass:: LoadedSplit
.. autoclass:: ResolvedInput

Source identities
-----------------

``V1Source`` uses only JSON-metadata shards under ``data/classification_v1.0_wds/<name>/``. HDF5 and pickle caches are not supported or converted. Missing shards require an explicit ``torchgeo-bench download geobench_v1 --datasets <name>``; loading never downloads data or falls back to another format. V1 partitions translate ``val`` to the stored ``valid`` split.

``V2Source`` identifies an upstream class and explicitly declares modality grouping, validation-split naming, and any acquisition or label adapter. Runtime loading preserves upstream sensor alignment and restores the requested channel order, including multi-sensor temporal PASTIS inputs.

``TorchGeoSource`` identifies the upstream reader and storage root. ``m-eurosat``, ``eurosat``, and ``eurosat-spatial`` are distinct benchmarks. Standard and spatial TorchGeo EuroSAT share imagery and downloads, but use different upstream split definitions. GeoBench V1 retains its separate statistics and storage. See :doc:`../user/datasets` for the dataset tables.
