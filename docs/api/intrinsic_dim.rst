torchgeo_bench.intrinsic_dim
============================

.. module:: torchgeo_bench.intrinsic_dim

Wrapper around the `torchid <https://github.com/jacobpennington/torchid>`__
intrinsic-dimension estimators plus dependency-free centered feature-spectrum
diagnostics.  Used by
:func:`torchgeo_bench.main.evaluate_intrinsic_dim` to attach
``method="intrinsic_dim"`` rows to the standard results CSV alongside the
KNN and linear-probe metrics.

The ``torchid`` dependency is optional and gated behind the ``[id]`` extra
in :file:`pyproject.toml` (it requires Python ≥ 3.13).  When the dependency
is not installed an :class:`ImportError` is raised on first use.

Public API
----------

.. autoexception:: DegenerateSpectrumError

.. autofunction:: compute_intrinsic_dim

.. autofunction:: compute_feature_spectrum
