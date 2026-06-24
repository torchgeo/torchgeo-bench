"""UniverSat (AnySat v2) wrapper for torchgeo-bench.

Wraps the UniverSat multimodal Earth-observation encoder
(https://github.com/gastruc/UniverSat) for the BenchModel interface.

UniverSat ingests a ``{modality: tensor}`` dict and embeds each channel by
its physical wavelength, so it accepts an arbitrary number of bands per
sensor.  The wrapper picks the UniverSat modality from ``BandSpec.sensor``,
passes the per-channel wavelengths straight from the ``BandSpec`` list, runs
``model.encode(...)``, and mean-pools the spatial tokens into a ``(B, 768)``
tile embedding.

The model code is fetched via ``torch.hub`` (no pip package); pretrained
weights load from the HuggingFace Hub (``g-astruc/UniverSat``).  Both are
cached after the first load, so offline compute nodes work once the cache is
warm.  Inputs are z-scored per channel by the framework
(``normalization="bandspec_zscore"``), matching UniverSat's own GeoBench
loader.
"""

import logging
from typing import ClassVar

import torch
import torch.nn.functional as F

from torchgeo_bench.datasets.base import BandSpec

from .interface import BenchModel

logger = logging.getLogger(__name__)

# Pinned UniverSat revision for reproducible torch.hub loads. Bump together
# with the cached weights when the upstream model changes.
UNIVERSAT_REPO = "gastruc/UniverSat"
UNIVERSAT_REF = "f6df2eec54955b0f7524cc95fe21a5e80c0239d9"

# BandSpec.sensor -> UniverSat modality name (must be a key the released
# checkpoint built a projector for; see DEFAULT_MODALITIES_DICT in the repo).
_SENSOR_TO_MODALITY: dict[str, str] = {
    "s2": "s2",
    "sentinel2": "s2",
    "sentinel-2": "s2",
    "s1": "s1",
    "sentinel1": "s1",
    "sar": "s1",
    "naip": "naip",
    "aerial": "aerial",
    "spot": "spot",
    "landsat": "l8",
}

# Physical resolution (m/px) and sub-patch factor per UniverSat modality,
# from the model's modality_registry.
_MODALITY_INPUT_RES: dict[str, float] = {
    "s2": 10.0,
    "s1": 10.0,
    "naip": 1.25,
    "aerial": 0.2,
    "spot": 1.0,
    "l8": 10.0,
    "l7": 30.0,
}
_MODALITY_SUBPATCH: dict[str, int] = {
    "s2": 1,
    "s1": 1,
    "naip": 10,
    "aerial": 10,
    "spot": 10,
    "l8": 1,
    "l7": 1,
}

# SAR channels are sensor codes (not wavelengths) in UniverSat's registry.
_SAR_CODES: list[str] = ["VV", "VH", "Ratio_VV_VH"]


class UniverSatBenchModel(BenchModel):
    """BenchModel wrapper for the UniverSat (AnySat v2) EO encoder.

    The modality is auto-detected from ``bands[0].sensor`` and per-channel
    wavelengths are read from the ``BandSpec`` list, so both RGB and
    full-multispectral band modes work without a fixed channel layout.

    Args:
        bands: Ordered ``BandSpec`` list describing the input channels. All
            bands must share one sensor.
        modality: Override the auto-detected UniverSat modality name (e.g.
            ``"s2"``). ``None`` (default) detects from ``BandSpec.sensor``.
        patch_size: Patch size in metres passed to ``encode`` (default 40).
        output_grid: Side ``G`` of the ``G×G`` token grid to request. ``None``
            (default) lets UniverSat infer the natural patch grid from the
            input size; tokens are mean-pooled either way.
        input_res: Override the modality's physical resolution (m/px). ``None``
            uses the registry value for the modality.
        normalize: If True, L2-normalize the output embeddings.
        repo: torch.hub source for the model code.
        repo_ref: Pinned git ref for the torch.hub load.
    """

    # Output embedding dimension of the released Base model.
    embed_dim: ClassVar[int] = 768

    def __init__(
        self,
        bands: list[BandSpec],
        *,
        modality: str | None = None,
        patch_size: float = 40.0,
        output_grid: int | None = None,
        input_res: float | None = None,
        normalize: bool = False,
        repo: str = UNIVERSAT_REPO,
        repo_ref: str | None = UNIVERSAT_REF,
        **_kwargs: object,
    ) -> None:
        super().__init__(bands=bands, **_kwargs)

        sensor = self.bands[0].sensor.lower()
        if not all(b.sensor.lower() == sensor for b in self.bands):
            sensors = sorted({b.sensor.lower() for b in self.bands})
            raise ValueError(f"UniverSat wrapper expects a single sensor per call; got {sensors}.")
        self.modality = modality or _SENSOR_TO_MODALITY.get(sensor)
        if self.modality is None:
            raise ValueError(
                f"No UniverSat modality mapping for sensor {sensor!r}. "
                f"Pass `modality=` explicitly. Known: {sorted(_SENSOR_TO_MODALITY)}."
            )

        if self.modality == "s1":
            self.wavelengths: list[float | str] = _SAR_CODES[: self.num_channels]
        else:
            wl = [b.wavelength_um for b in self.bands]
            if any(w is None for w in wl):
                missing = [b.name for b in self.bands if b.wavelength_um is None]
                raise ValueError(
                    f"UniverSat needs a wavelength per channel; BandSpecs {missing} "
                    f"have wavelength_um=None."
                )
            self.wavelengths = list(wl)

        self.patch_size = patch_size
        self.output_grid = output_grid
        self.input_res = input_res if input_res is not None else _MODALITY_INPUT_RES[self.modality]
        self.subpatch = _MODALITY_SUBPATCH.get(self.modality, 1)
        self.do_normalize = normalize

        source = f"{repo}:{repo_ref}" if repo_ref else repo
        self.model = torch.hub.load(source, "from_pretrained", trust_repo=True).eval()
        logger.info(
            "UniverSat loaded (modality=%s, %d channels, patch_size=%sm)",
            self.modality,
            self.num_channels,
            self.patch_size,
        )

    @torch.no_grad()
    def _forward_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        """Embed ``(B, C, H, W)`` into ``(B, 768)`` mean-pooled tile features."""
        x = {self.modality: images}
        tokens, _ = self.model.encode(
            x,
            patch_size=self.patch_size,
            output_grid=self.output_grid,
            wavelengths={self.modality: self.wavelengths},
            input_res={self.modality: self.input_res},
            subpatches={self.modality: self.subpatch},
        )
        embeddings = tokens.mean(dim=1)
        if self.do_normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
        return embeddings
