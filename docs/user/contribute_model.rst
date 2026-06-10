Contribute a model (Stage 2)
============================

This guide covers everything a pull request needs to add a new frozen
pretrained model to ``torchgeo-bench``.  Before continuing, complete
:doc:`eval_own_model` (Stage 1) to verify your model produces sensible results
on the applicable datasets.

.. _contrib-prerequisites:

Prerequisites
-------------

The setup steps are the same as Stage 1:

.. code-block:: console

   $ git clone https://github.com/torchgeo/torchgeo-bench.git
   $ cd torchgeo-bench
   $ conda activate torchgeo-bench
   $ uv sync --extra dev

Then fork the repository on GitHub and create a feature branch:

.. code-block:: console

   $ git remote add fork https://github.com/<your-username>/torchgeo-bench.git
   $ git checkout -b add-<model-name>

For the model implementation and YAML config steps, follow
:doc:`eval_own_model` (Sections :ref:`eval-implement` and
:ref:`eval-hydra-config`).  The remainder of this page covers integration,
tests, and the PR submission.

.. _contrib-integrate:

Integrate into the package
--------------------------

Once your model class is working locally, move it into the package and export
it so the Hydra registry can resolve ``_target_``.

**1. Place the module** under :file:`src/torchgeo_bench/models/`:

.. code-block:: console

   $ mv my_geofm.py src/torchgeo_bench/models/my_geofm.py

**2. Export the class** from :file:`src/torchgeo_bench/models/__init__.py`:

.. code-block:: python

   # src/torchgeo_bench/models/__init__.py
   from .my_geofm import MyGeoFM

   __all__: list[str] = [
       ...,
       "MyGeoFM",
   ]

**3. Update the Hydra config** ``_target_`` to the package path:

.. code-block:: yaml

   # src/torchgeo_bench/conf/model/my_model.yaml
   _target_: torchgeo_bench.models.MyGeoFM
   name: my_model
   pretrained: true

**4. Declare optional dependencies** in :file:`pyproject.toml` if your model
requires packages beyond ``[project.dependencies]``:

.. code-block:: toml

   [project.optional-dependencies]
   mymodel = ["mypackage>=1.0"]

Install the extra locally to confirm it resolves:

.. code-block:: console

   $ uv sync --extra mymodel

.. note::

   If your model requires an optional extra, document it clearly in your
   model's ``__init__`` docstring and in the PR description.  The test suite
   must still import the class without the extra installed (guard weight
   loading with a ``try/except ImportError`` only around the optional import,
   not the class definition).

.. _contrib-weights:

Weights
-------

* Pretrained weights must be **publicly accessible without authentication**.
  `HuggingFace Hub <https://huggingface.co/models>`_ is the preferred host.
* The model must load after a fresh ``pip install 'torchgeo-bench[myextra]'``
  with no manual file placement.  Use
  `huggingface_hub.hf_hub_download <https://huggingface.co/docs/huggingface_hub/>`_
  or an equivalent auto-download call inside your ``__init__``.
* The weights URL must appear in the PR description so reviewers can verify
  provenance.

.. _contrib-tests:

Write tests
-----------

Create :file:`tests/test_<model>.py`.  Every added code path must be covered.

**Fast tests (run in CI) — no network I/O:**

.. code-block:: python

   import torch
   import pytest
   from torchgeo_bench.datasets.base import BandSpec
   from torchgeo_bench.models.my_geofm import MyGeoFM


   def _bands(n: int = 3) -> list[BandSpec]:
       return [
           BandSpec(sensor="s2", name=f"b{i}", source_name=f"B{i}",
                    mean=500.0, std=100.0, min=0.0, max=10000.0)
           for i in range(n)
       ]


   def test_my_geofm_output_shape():
       """Model returns (B, K) with random weights."""
       model = MyGeoFM(bands=_bands(), pretrained=False)
       x = torch.randn(2, 3, 64, 64)
       with torch.no_grad():
           out = model.forward_patch_features(x)
       assert out.ndim == 2
       assert out.shape[0] == 2


   def test_my_geofm_num_channels():
       """`num_channels` matches the input BandSpec list length."""
       model = MyGeoFM(bands=_bands(5), pretrained=False)
       assert model.num_channels == 5

**Weight-download tests (slow, run locally before PR) — mark with** ``@pytest.mark.slow``:

.. code-block:: python

   @pytest.mark.slow
   def test_my_geofm_pretrained_loads():
       """Pretrained weights download and load without error."""
       model = MyGeoFM(bands=_bands(), pretrained=True)
       x = torch.randn(1, 3, 64, 64)
       with torch.no_grad():
           out = model.forward_patch_features(x)
       assert out.ndim == 2

Run the fast tests before opening the PR:

.. code-block:: console

   $ pytest --no-cov tests/test_my_geofm.py

Slow tests must pass locally but are excluded from the default CI run
(``pytest`` without ``-m slow`` skips them automatically):

.. code-block:: console

   $ pytest --no-cov -m slow tests/test_my_geofm.py

.. _contrib-results:

Submit results
--------------

Run the full benchmark on all datasets applicable to your model's sensor
coverage and write the results to :file:`results/contributed/<model_name>.csv`:

.. code-block:: console

   $ torchgeo-bench run model=my_model \
       dataset.names=[m-eurosat,m-so2sat,m-bigearthnet,m-brick-kiln,m-forestnet,m-pv4ger] \
       eval.output=results/contributed/my_model.csv

For V2 datasets:

.. code-block:: console

   $ torchgeo-bench run model=my_model \
       dataset.names=[benv2,treesatai,so2sat,forestnet] \
       eval.output=results/contributed/my_model.csv resume=true

The CSV schema is identical to :file:`results/all_results.csv` — see
:doc:`results-format` for the full column reference.

In the PR description, document any datasets you skipped and the reason
(sensor mismatch, missing modality, etc.):

.. code-block:: text

   Skipped datasets:
   - m-forestnet: model trained on Sentinel-2 only; Landsat not supported
   - burn_scars: model does not support SAR inputs

.. _contrib-pr:

Open the PR
-----------

When all checklist items are satisfied, open a pull request against ``main``
using the **"Add model"** template:

:file:`.github/PULL_REQUEST_TEMPLATE/add_model.md`

All checklist items must be checked before requesting a review.  The
template prompts you for:

* A model summary table (name, architecture, pretraining data, sensor
  coverage, weights URL).
* Confirmation that each technical requirement is satisfied (class exported,
  config present, weights public, tests written and passing, results
  submitted, lint clean).

.. seealso::

   :doc:`eval_own_model` — Stage 1: benchmark your model locally without
   opening a PR.
