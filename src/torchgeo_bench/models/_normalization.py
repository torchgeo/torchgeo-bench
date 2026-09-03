"""Selectable input-normalisation strategies for benchmark models.

Each pretrained backbone was trained against a specific input pipeline, but
"the right" cross-dataset normalisation is empirical — what works for
m-eurosat (raw S2 DN) doesn't generalise to m-so2sat (already reflectance)
or m-pv4ger (uint8 NAIP).  Rather than hard-code one policy, expose a
strategy enum and let the sweep treat it as another axis.

Strategies:

* ``bandspec_zscore`` — per-channel ``(x - mean) / std`` from the dataset's
  :class:`BandSpec` stats.  Yields ~N(0, 1) regardless of source unit;
  good cross-dataset uniformity, ignores the model's training pipeline.
* ``model_native`` — bring inputs into the model's
  ``expected_input_unit`` (e.g. ``s2_dn`` -> ``/10000``), then apply any
  ``pretrain_mean`` / ``pretrain_std`` declared on the wrapper.  Faithful
  to the model's training pipeline; less robust to unit mismatches.
* ``minmax`` — scale each channel to ``[0, 1]`` using
  :attr:`BandSpec.min`/:attr:`BandSpec.max`.
* ``minmax_zscore`` — minmax to ``[0, 1]`` then per-channel z-score
  (against the post-minmax dataset stats).
* ``identity`` — no rescaling.  For models that handle raw inputs
  internally (e.g. OlmoEarth's ``Normalizer``).
"""

from collections.abc import Callable
from enum import StrEnum

import torch

from torchgeo_bench.datasets.base import BandSpec

from ._input_units import InputUnit, detect_input_unit, to_reflectance, to_s2_dn


class NormalizationStrategy(StrEnum):
    BANDSPEC_ZSCORE = "bandspec_zscore"
    MODEL_NATIVE = "model_native"
    MINMAX = "minmax"
    MINMAX_ZSCORE = "minmax_zscore"
    IDENTITY = "identity"


def _bandspec_mean_std(bands: list[BandSpec]) -> tuple[torch.Tensor, torch.Tensor]:
    n = len(bands)
    mean = torch.tensor([b.mean for b in bands], dtype=torch.float32).view(1, n, 1, 1)
    std = torch.tensor([b.std for b in bands], dtype=torch.float32).view(1, n, 1, 1).clamp_min(1e-8)
    return mean, std


def _bandspec_min_max(bands: list[BandSpec]) -> tuple[torch.Tensor, torch.Tensor]:
    n = len(bands)
    lo = torch.tensor([b.min for b in bands], dtype=torch.float32).view(1, n, 1, 1)
    hi = torch.tensor([b.max for b in bands], dtype=torch.float32).view(1, n, 1, 1)
    span = (hi - lo).clamp_min(1e-8)
    return lo, span


def build_normalizer(
    strategy: NormalizationStrategy | str,
    bands: list[BandSpec],
    *,
    expected_input_unit: InputUnit | None = None,
    pretrain_mean: list[float] | None = None,
    pretrain_std: list[float] | None = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build a callable that normalises ``(B, C, H, W)`` tensors per the chosen strategy."""
    strategy = NormalizationStrategy(strategy)

    if strategy is NormalizationStrategy.IDENTITY:
        return lambda x: x

    if strategy is NormalizationStrategy.BANDSPEC_ZSCORE:
        mean, std = _bandspec_mean_std(bands)

        def _f(x: torch.Tensor) -> torch.Tensor:
            return (x - mean.to(x.device, x.dtype)) / std.to(x.device, x.dtype)

        return _f

    if strategy is NormalizationStrategy.MINMAX:
        lo, span = _bandspec_min_max(bands)

        def _f(x: torch.Tensor) -> torch.Tensor:
            return (x - lo.to(x.device, x.dtype)) / span.to(x.device, x.dtype)

        return _f

    if strategy is NormalizationStrategy.MINMAX_ZSCORE:
        lo, span = _bandspec_min_max(bands)
        # Post-minmax mean_i = (raw_mean_i - min_i) / (max_i - min_i)
        # Post-minmax std_i  = raw_std_i  / (max_i - min_i)
        n = len(bands)
        pmean = torch.tensor(
            [(b.mean - b.min) / max(b.max - b.min, 1e-8) for b in bands],
            dtype=torch.float32,
        ).view(1, n, 1, 1)
        pstd = (
            torch.tensor(
                [b.std / max(b.max - b.min, 1e-8) for b in bands],
                dtype=torch.float32,
            )
            .view(1, n, 1, 1)
            .clamp_min(1e-8)
        )

        def _f(x: torch.Tensor) -> torch.Tensor:
            x = (x - lo.to(x.device, x.dtype)) / span.to(x.device, x.dtype)
            return (x - pmean.to(x.device, x.dtype)) / pstd.to(x.device, x.dtype)

        return _f

    if expected_input_unit is None:
        raise ValueError("model_native normalisation requires expected_input_unit")
    src = detect_input_unit(bands)
    if expected_input_unit == InputUnit.S2_DN:
        convert = lambda x: to_s2_dn(x, src)  # noqa: E731
    elif expected_input_unit == InputUnit.REFLECTANCE_0_1:
        convert = lambda x: to_reflectance(x, src)  # noqa: E731
    elif expected_input_unit == InputUnit.UINT8:
        # uint8-trained models want [0, 1] before ImageNet stats, so route through reflectance.
        convert = lambda x: to_reflectance(x, src)  # noqa: E731
    else:
        convert = lambda x: x  # noqa: E731

    if pretrain_mean is None:
        # Unit conversion alone is not a normalisation: it would feed raw DN
        # (0 - 10 000) straight into the backbone, which measurably collapses
        # the features — Prithvi scored an identical 0.264 on treesatai at
        # 86M, 304M and 631M parameters, below the imagestats baseline.
        #
        # Wrappers that install their own model_native normaliser (either
        # after super().__init__, e.g. TerraMind, or by overriding
        # normalize_inputs entirely, e.g. the torchgeo wrappers) must declare
        # that explicitly via `BenchModel.installs_own_model_native_normalizer`
        # and finalize their own normalizer through
        # `BenchModel.finalize_model_native_normalizer()` -- they never reach
        # this call in the first place.  Any wrapper that does reach here has
        # neither a usable pretrain_mean/pretrain_std nor its own normalizer,
        # so fail immediately instead of returning a normalizer that only
        # raises lazily on first forward.
        raise ValueError(
            "model_native normalisation is undefined for this model: it declares "
            f"expected_input_unit={expected_input_unit.value!r} but no pretrain_mean/"
            "pretrain_std, and it does not supply its own normaliser.  Converting "
            "units without standardising leaves raw sensor values.  Set "
            "pretrain_mean/pretrain_std on the wrapper, or evaluate it with "
            "dataset.normalization=bandspec_zscore."
        )

    n = len(pretrain_mean)
    pm = torch.tensor(pretrain_mean, dtype=torch.float32).view(1, n, 1, 1)
    ps = (
        torch.tensor(pretrain_std or [1.0] * n, dtype=torch.float32)
        .view(1, n, 1, 1)
        .clamp_min(1e-8)
    )

    def _f(x: torch.Tensor) -> torch.Tensor:
        x = convert(x)
        return (x - pm.to(x.device, x.dtype)) / ps.to(x.device, x.dtype)

    return _f
