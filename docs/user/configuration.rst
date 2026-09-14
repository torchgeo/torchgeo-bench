Configuration
=============

All commands use explicit flags and strict Pydantic-validated YAML.
``torchgeo-bench``, ``python -m torchgeo_bench``, and
``python -m torchgeo_bench.cli`` expose the same interface.

.. warning::

   The old ``key=value`` and ``+key=value`` overrides are rejected, including
   through the module entry points. Replace them with flags or ``--config``.
   There is no Hydra/OmegaConf composition, interpolation, or recursive
   ``_target_`` instantiation.

For example:

.. code-block:: console

   $ torchgeo-bench run --model timm/resnet50 --dataset m-eurosat \
       --methods knn --bootstrap-samples 100 --device cpu
   $ torchgeo-bench run --config examples/image-run.yaml --methods linear --dry-run
   $ torchgeo-bench run --config-help

``--dry-run`` validates selections and prints reusable YAML without loading
weights or dataset samples. ``run --config-help`` and ``flops --config-help``
print JSON schemas; each command's ``--help`` lists its flags.

Precedence and validation
-------------------------

Settings are resolved independently for each dataset, in this order from
lowest to highest priority:

1. Built-in schema defaults.
2. Model preset defaults.
3. The preset's defaults for that dataset.
4. Explicit values in the supplied YAML.
5. Explicit command-line flags.

Omitting a field is different from explicitly setting its default. For
example, omitting ``segmentation.layers`` keeps the preset's layers, while
``layers: []`` replaces them with an empty selection. ``image_size: null``
disables resizing, and ``cache_features: false`` disables feature caching.
Neither is silently replaced by a preset value. Mappings merge recursively;
lists and scalar values replace earlier values.

YAML must be a mapping with unique keys and known fields. Types are strict:
write ``false``, not ``"false"``, and numeric values without quotes.
Scientific notation such as ``1e-3`` is supported. Constructor options inside
``model.kwargs`` are passed to the selected constructor; they are not
additional benchmark configuration fields.

Image runs
----------

A minimal run file selects a model and one or more datasets:

.. code-block:: yaml

   model:
     name: timm/resnet50
   datasets: [m-eurosat]

Run it with ``torchgeo-bench run --config my-run.yaml``. A model selection is
always a mapping, not the old ``model: timm/resnet50`` scalar.
Use ``datasets: [all]`` to select the image dataset catalog. On the CLI,
``--dataset`` is repeatable:

.. code-block:: console

   $ torchgeo-bench run --model timm/resnet50 \
       --dataset m-eurosat --dataset m-pv4ger --device cpu

The complete field example is :file:`examples/image-run.yaml`. It explicitly
sets defaults for illustration; remove fields that should inherit preset
settings.

Input and runtime
^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   input:
     bands: rgb
     partition: default
     time_steps: null
     image_size: 224
     interpolation: bilinear
     normalization: dataset
   runtime:
     device: cuda:0
     batch_size: 64
     workers: 4
     seed: 0
     verbose: false

``bands`` accepts ``rgb``, ``all``, or an ordered YAML list of dataset band
names. The equivalent flag accepts comma-separated names, for example
``--bands red,green,blue``. Band names differ by dataset; see :doc:`datasets`.
``time_steps`` selects a positive temporal length where supported.
``image_size: null`` (``--image-size none``) retains native dimensions.
Interpolation is ``area``, ``bilinear``, ``bicubic``, or ``nearest``.

Normalization choices map to the model interface as follows:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - YAML / flag
     - Model strategy
     - Meaning
   * - ``dataset``
     - ``bandspec_zscore``
     - Per-band dataset statistics.
   * - ``model``
     - ``model_native``
     - Model-declared input units and pretraining statistics.
   * - ``minmax``
     - ``minmax``
     - Per-band scaling using dataset ``BandSpec.min`` and ``BandSpec.max``.
   * - ``minmax_zscore``
     - ``minmax_zscore``
     - Min-max scaling followed by z-scoring with the scaled dataset statistics.
   * - ``none``
     - ``identity``
     - No normalization.

``model`` requires a wrapper that declares its expected units; it is not an
automatic substitute for dataset normalization. Wrappers that own their
normalization retain their documented behavior.

Use ``--device cpu`` without CUDA, a specific device such as ``cuda:1``, or
``auto`` to choose an available CUDA device or CPU. The image default remains
``cuda:0``. Runtime flags are ``--device``, ``--batch-size``, ``--workers``,
``--seed``, and ``--verbose`` / ``--no-verbose``. Input flags are ``--bands``,
``--partition``, ``--time-steps``, ``--image-size``, ``--interpolation``, and
``--normalization``.

Classification
^^^^^^^^^^^^^^

.. code-block:: yaml

   classification:
     methods: [knn, linear]
     knn_k: 5
     knn_device: null
     bootstrap_samples: 200
     linear:
       c_log10_start: -6.0
       c_log10_stop: 4.0
       c_count: 40
       refit_train_val: true
     calibration:
       n_bins_knn: null
       n_bins_linear: 15
       temp_scale: false

Select ``--methods knn``, ``--methods linear``, or ``--methods knn linear``.
Linear-only runs do not require a KNN result. ``--knn-k``,
``--knn-device``, and ``--bootstrap-samples`` override the corresponding
fields. A null KNN device inherits the feature-extraction device; the backend
logs a CPU fallback if GPU KNN is unavailable.

The linear probe selects regularization on validation data from ``c_count``
log-spaced values between the two log10 endpoints. It then optionally refits
on train plus validation. ``--refit-train-val`` /
``--no-refit-train-val`` control that final refit. Temperature scaling
(``--temp-scale`` / ``--no-temp-scale``) requires linear probing and
``refit_train_val: false`` so validation logits stay held out.
``n_bins_knn: null`` uses ``knn_k + 1`` calibration bins.

Segmentation
^^^^^^^^^^^^

.. code-block:: yaml

   segmentation:
     head: fpn
     layers: []
     learning_rate: 1e-3
     epochs: 10
     batch_size: 64
     temporal_pool: mean
     scheduler: cosine
     ignore_index: 255
     cache_features: true
     cache_dtype: float16

These YAML settings apply to segmentation datasets, independently of
``classification.methods``. Supported heads are ``linear``, ``conv_block``,
``fpn``, ``dpt``, and ``patch_linear``; not every head works with every
backbone. Omit ``layers`` to inherit the model's selection. An explicit
``[]`` clears it and is rejected by image segmentation, which requires explicit
feature layers. See
:doc:`segmentation-layers` for verified layer paths.

``runtime.batch_size`` controls feature extraction;
``segmentation.batch_size`` controls probe training. ``temporal_pool`` is
``mean`` or ``max``, ``scheduler`` is ``cosine`` or ``none``, and the cache
dtype is ``float16`` or ``float32``. The loss is cross entropy with the
configured ``ignore_index``; there is no constructor-style criterion block.
Segmentation confidence intervals use ``classification.bootstrap_samples``.

Optional image measurements
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   profile:
     enabled: false
     n_warmup: 3
     n_measure: 20
     cpu_throughput:
       enabled: false
       batch_size: 8
       n_warmup: 1
       n_measure: 5
       time_budget_s: 300.0
   intrinsic_dim:
     enabled: false
     estimators: [TwoNN, MLE, lPCA]
     splits: [train]
     max_samples: 10000
     device: null

These are additive passes within an image run, not the standalone
``profile`` command. Intrinsic-dimension estimators require the ``id`` extra;
``estimators: []`` requests only the dependency-free feature-spectrum
diagnostics. ``splits`` accepts ``train``, ``val``, and ``test``.
``max_samples: null`` removes the sample cap.

Results and resume
^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

   output:
     directory: results/models
     file: null
     resume: false
     profile_directory: results/profiles
     intrinsic_dim_directory: results/intrinsic_dim

Normal image metrics append to ``output.directory/<model name>.csv``.
Optional profile and intrinsic-dimension rows use their own directories.
An explicit ``output.file`` combines all selected image-run measurements in
that CSV. ``--output`` sets the file, and ``--results-dir`` sets the normal
metrics directory. Neither changes standalone profile JSON or other commands'
outputs.

``--resume`` / ``--no-resume`` control skipping completed measurements.
Resume compares method/config keys and metric completeness; additive profile
or intrinsic-dimension selections do not invalidate otherwise equivalent
classification rows. Completed evaluations are appended atomically before
later evaluations run. See :doc:`results-format` for stored columns and keys.

Model presets and custom models
--------------------------------

``torchgeo-bench models`` lists packaged presets and
``torchgeo-bench models timm/resnet50`` displays one. Presets live under
:file:`src/torchgeo_bench/conf/model/`, including ``timm/`` and ``torchgeo/``
subdirectories. They are validated separately from run files:

.. code-block:: yaml

   name: my-model
   target: my_package.MyBenchModel
   track: image
   seed_from_run: false
   kwargs:
     feature_dim: 256
   input:
     image_size: 224
   classification:
     linear:
       c_count: 20
   segmentation:
     head: linear
   dataset_overrides:
     caffe:
       input:
         image_size: 128

Only ``kwargs`` are constructor options. ``name``, ``target``, ``track``,
``seed_from_run``, preprocessing/evaluation defaults, and dataset overrides
are preset metadata. ``seed_from_run: true`` explicitly injects the run seed
for constructors that support it, without interpolation.

For a one-off custom model, a run file can instead contain:

.. code-block:: yaml

   model:
     name: my-model
     target: my_package.MyBenchModel
     kwargs:
       feature_dim: 256
   datasets: [m-eurosat]

The target must be importable in the active Python environment. There is no
recursive instantiation inside ``kwargs``. Runtime ``BandSpec`` objects and
empirical-RCF datasets are passed explicitly to ``build_model``, never
serialized into YAML. See :doc:`/api/cli` for typed loading and resolution,
and :doc:`eval_own_model` for the model interface.

Standalone profiling
--------------------

``profile`` repeatedly measures one fixed real dataset batch, with the same
model/dataset input-default resolution as an image run. It writes one JSON
record to stdout, not to an image-results CSV:

.. code-block:: console

   $ torchgeo-bench profile --model rcf --dataset m-eurosat --device cpu \
       --batch-size 8 --warmup 1 --measurements 5 > profile.json
   $ torchgeo-bench profile --config examples/profile.yaml --dry-run

Its YAML uses singular ``dataset`` and top-level measurement settings:

.. code-block:: yaml

   model:
     name: rcf
   dataset: m-eurosat
   input:
     bands: rgb
   runtime:
     device: cpu
     batch_size: 8
     seed: 0
   warmup: 1
   measurements: 5
   precision: float32
   count_flops: false

It supports ``--model``, ``--dataset``, ``--device``, ``--batch-size``,
``--seed``, ``--bands``, ``--partition``, ``--image-size``,
``--interpolation``, and ``--normalization``. Timing options are
``--warmup`` and ``--measurements``. ``--precision`` is ``float32``,
``float16``, or ``bfloat16``. Reduced precision uses PyTorch autocast;
operation support depends on the model and device.
``--count-flops`` / ``--no-count-flops`` add or omit FLOP counting.
The JSON includes effective preprocessing, device, precision, timing, and
memory metadata. Local dataset samples must already be available.

Synthetic compute measurements
-------------------------------

``flops`` measures backbone and probe compute from synthetic tensors and
appends to ``results/compute_cost.csv`` by default. It reads dataset band
metadata but not samples. Model weights may still need downloading unless
the selected model is an offline baseline or already cached.

.. code-block:: console

   $ torchgeo-bench flops --config examples/flops.yaml --dry-run
   $ torchgeo-bench flops --model rcf --device cpu --band-configs rgb \
       --seg-heads --output results/my_compute_cost.csv

Its YAML schema is independent of image runs:

.. code-block:: yaml

   model:
     name: rcf
   runtime:
     device: cpu
     seed: 0
     verbose: true
   input:
     band_source: cloudsen12
     band_configs: [rgb, s2]
     image_size: 224
     normalization: dataset
   classification:
     head: linear
     num_classes: 10
   segmentation:
     heads: [fpn, dpt]
     band_configs: [rgb, s2]
     num_classes: 4
   timing:
     batch_size: 64
     n_warmup: 3
     n_measure: 20
   output:
     file: results/compute_cost.csv
     resume: true

``--band-source`` selects the dataset metadata. ``--band-configs rgb s2``
selects input stacks: ``s2`` means **all bands from the source**, which is
twelve Sentinel-2 bands for the default CloudSen12 source.
``--image-size`` must be positive; synthetic inputs cannot use native
dataset dimensions. ``--normalization`` has the same five choices as image
runs. ``--probe-head`` is ``linear`` or ``mlp``, and
``--probe-num-classes`` sets the classification output width.

Use ``--seg-heads`` with any of ``linear``, ``conv_block``, ``fpn``, ``dpt``,
and ``patch_linear``; supplying the flag without values disables
segmentation. ``--seg-band-configs`` likewise accepts an empty selection.
``--seg-num-classes``, ``--seg-layers``, and ``--temporal-pool`` configure
segmentation accounting. YAML may include a ``segmentation.probe`` block
with the image segmentation fields. Omitted probe layers inherit model
and dataset defaults; explicit ``[]`` clears them.

``--timing-batch-size``, ``--n-warmup``, and ``--n-measure`` control timing;
FLOPs are always counted for one sample. ``--output`` selects the CSV,
and ``--no-resume`` disables the default per-cell resume behavior.
``--model-target`` and ``--model-kwargs`` accept a custom constructor and a
YAML mapping of constructor options; reusable custom models can use
``model.target`` and ``model.kwargs`` in the file instead.

Coordinate evaluation
----------------------

``coord`` has its own ``model``, ``datasets``, ``evaluation``, ``runtime``,
and ``output`` sections. It does not accept image-run ``input`` or
``classification`` blocks:

.. code-block:: console

   $ torchgeo-bench coord --model sincos --dataset california_housing \
       --methods linear --folds 2 --device cpu
   $ torchgeo-bench coord --config examples/coord-run.yaml --dry-run

See :doc:`coordbench` for the full YAML, encoder targets, method selection,
random/spatial/official splits, and coordinate CSV resume semantics.
