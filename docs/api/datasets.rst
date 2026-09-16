torchgeo_bench.datasets
=======================

.. module:: torchgeo_bench.datasets

Every benchmark identity has an immutable :class:`DatasetSpec`. The definition owns its task, class count, multilabel flag, ordered bands and statistics, RGB selector, default split sizes, source policies, capabilities, and geography aliases. Per-dataset modules contain ``SPEC`` records; the read-only catalog derives names and routing from those records, without a second identity registry.

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

``DatasetSpec.resolve_band_specs()`` translates ``rgb``, ``all``, or ordered explicit names into the original frozen ``BandSpec`` objects. Band collections and selections are tuples. RGB selection currently follows each dataset's existing metadata, including grayscale and SAR selections.

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

Request validation/test explicitly when needed. The image runner applies a non-default partition only to training; library callers can request a supported partition on any split. Unsupported options fail before source construction. There are no automatic downloads or sample probes in ``load_split``. Runtime split objects are not configuration or resume-hash payloads.

.. autofunction:: load_split
.. autoclass:: LoadedSplit
.. autoclass:: ResolvedInput

Source identities
-----------------

``V1Source`` uses only JSON-metadata shards under ``data/classification_v1.0_wds/<name>/``. HDF5 and pickle caches are not supported or converted. Missing shards require an explicit ``torchgeo-bench download geobench_v1 --datasets <name>``; loading never downloads data or falls back to another format. V1 partitions translate ``val`` to the stored ``valid`` split.

``V2Source`` identifies an upstream class and explicitly declares modality grouping, validation-split naming, and any acquisition or label adapter. Runtime loading preserves upstream sensor alignment and restores the requested channel order, including multi-sensor temporal PASTIS inputs.

``TorchGeoSource`` identifies the upstream reader and storage root. ``m-eurosat``, ``eurosat``, and ``eurosat-spatial`` are distinct benchmarks. Standard and spatial TorchGeo EuroSAT share imagery and downloads, but use different upstream split definitions. GeoBench V1 retains its separate statistics and storage. See :doc:`../user/datasets` for the dataset tables.
