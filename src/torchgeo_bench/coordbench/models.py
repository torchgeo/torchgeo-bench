"""Coordinate-only encoders for the CoordBench location-encoder track.

A :class:`LocationEncoder` maps points ``(lon, lat[, year])`` to a fixed-length
feature vector, one row per point; the probes and cross-validation live downstream.
Add a model by subclassing :class:`LocationEncoder`, implementing :meth:`_encode`,
and pointing a Hydra ``model`` config's ``_target_`` at it.

The trivial :class:`SinCosLocationEncoder` and the pretrained
:class:`MINDLocationEncoder` ship in the base install. The other pretrained
reference encoders (SatCLIP / GeoCLIP / Climplicit / SINR) are thin wrappers
over the ``rshf`` package and require the ``coordbench`` extra
(``pip install -e ".[coordbench]"``). The dependency-free coordinate position
encoders below do not load external released weights.
"""

import logging
from abc import ABC, abstractmethod

import numpy as np
import torch

logger = logging.getLogger(__name__)


class LocationEncoder(ABC):
    """Frozen coordinate encoder: ``(lon, lat[, year]) -> (N, D)`` features.

    Args:
        device: Torch device string for the forward pass.
        batch_size: Points per forward chunk.
    """

    #: Human-readable identifier recorded in result rows.
    name: str = "location_encoder"

    def __init__(self, device: str = "cpu", batch_size: int = 8192) -> None:
        self.device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        self.batch_size = int(batch_size)

    @abstractmethod
    def _encode(self, lon: np.ndarray, lat: np.ndarray, year: np.ndarray | None) -> np.ndarray:
        """Encode a single chunk of points; return ``(len(lon), D)`` float32.

        Args:
            lon: Longitudes, shape ``(B,)``.
            lat: Latitudes, shape ``(B,)``.
            year: Optional per-point year, shape ``(B,)`` (``None`` if the
                encoder is time-invariant).
        """
        raise NotImplementedError

    def encode(
        self, lon: np.ndarray, lat: np.ndarray, year: np.ndarray | None = None
    ) -> np.ndarray:
        """Encode all points, batching internally.

        Args:
            lon: Longitudes, shape ``(N,)``.
            lat: Latitudes, shape ``(N,)``.
            year: Optional per-point year, shape ``(N,)``.

        Returns:
            Feature matrix of shape ``(N, D)``, dtype float32.
        """
        lon = np.asarray(lon, dtype=np.float64)
        lat = np.asarray(lat, dtype=np.float64)
        n = len(lon)
        out: list[np.ndarray] = []
        for start in range(0, n, self.batch_size):
            end = min(start + self.batch_size, n)
            yr = None if year is None else np.asarray(year)[start:end]
            out.append(np.asarray(self._encode(lon[start:end], lat[start:end], yr), np.float32))
        return np.concatenate(out, axis=0) if out else np.empty((0, 0), np.float32)


class SinCosLocationEncoder(LocationEncoder):
    """Dependency-free baseline: ``[sin(lat), cos(lat), sin(lon), cos(lon)]``."""

    name = "sincos"

    def _encode(self, lon: np.ndarray, lat: np.ndarray, _year: np.ndarray | None) -> np.ndarray:
        lat_r, lon_r = np.deg2rad(lat), np.deg2rad(lon)
        return np.stack(
            [np.sin(lat_r), np.cos(lat_r), np.sin(lon_r), np.cos(lon_r)], axis=1
        ).astype(np.float32)


def _unit_xyz(lon: np.ndarray, lat: np.ndarray) -> torch.Tensor:
    """Convert finite geographic coordinates in degrees to unit XYZ tensors."""
    if lon.ndim != 1 or lat.ndim != 1 or lon.shape != lat.shape:
        raise ValueError("lon and lat must be one-dimensional arrays with equal length")
    if not np.isfinite(lon).all() or not np.isfinite(lat).all():
        raise ValueError("lon and lat must contain only finite values")
    if ((lat < -90.0) | (lat > 90.0)).any():
        raise ValueError("lat must be in the inclusive range [-90, 90] degrees")

    lon_radians = torch.as_tensor(np.deg2rad(lon), dtype=torch.float32)
    lat_radians = torch.as_tensor(np.deg2rad(lat), dtype=torch.float32)
    cos_lat = torch.cos(lat_radians)
    return torch.stack(
        (
            cos_lat * torch.cos(lon_radians),
            cos_lat * torch.sin(lon_radians),
            torch.sin(lat_radians),
        ),
        dim=1,
    )


class XYZLocationEncoder(LocationEncoder):
    """Return the three-dimensional unit-sphere representation of each point."""

    name = "xyz"

    @torch.no_grad()
    def _encode(self, lon: np.ndarray, lat: np.ndarray, _year: np.ndarray | None) -> np.ndarray:
        return _unit_xyz(lon, lat).numpy()


class NeRFLocationEncoder(LocationEncoder):
    """Return deterministic NeRF-style Fourier features of spherical XYZ.

    For each XYZ coordinate, frequencies ``2**k * pi`` are applied for
    ``k = 0 .. num_frequencies - 1``. The encoder emits only the transformed
    features by default. Set ``include_xyz`` to add raw spherical coordinates
    as an ablation.

    Args:
        num_frequencies: Number of geometric frequency bands. Must be positive.
        include_xyz: Include the three untransformed XYZ coordinates.
    """

    name = "nerf"

    def __init__(
        self,
        num_frequencies: int = 10,
        include_xyz: bool = False,
        device: str = "cpu",
        batch_size: int = 8192,
    ) -> None:
        super().__init__(device=device, batch_size=batch_size)
        if num_frequencies < 1:
            raise ValueError("num_frequencies must be positive")
        self.num_frequencies = int(num_frequencies)
        self.include_xyz = bool(include_xyz)

    @torch.no_grad()
    def _encode(self, lon: np.ndarray, lat: np.ndarray, _year: np.ndarray | None) -> np.ndarray:
        xyz = _unit_xyz(lon, lat)
        frequencies = torch.pow(2.0, torch.arange(self.num_frequencies, dtype=torch.float32))
        angles = xyz[:, :, None] * frequencies[None, None, :] * torch.pi
        features = [torch.sin(angles).flatten(1), torch.cos(angles).flatten(1)]
        if self.include_xyz:
            features.insert(0, xyz)
        return torch.cat(features, dim=1).numpy()


class SphericalHarmonicLocationEncoder(LocationEncoder):
    """Return normalized real spherical-harmonic features through degree 3.

    The output has ``(degree + 1)**2`` columns, ordered by increasing degree
    and then the conventional real ``m`` order from ``-l`` to ``l``. Degree
    three is a deliberate upper bound: it provides a compact continuous
    spherical basis without pretending to reproduce a learned task head.

    Args:
        degree: Maximum harmonic degree, from 0 through 3.
    """

    name = "spherical-harmonics"

    def __init__(
        self,
        degree: int = 3,
        device: str = "cpu",
        batch_size: int = 8192,
    ) -> None:
        super().__init__(device=device, batch_size=batch_size)
        if degree < 0 or degree > 3:
            raise ValueError("degree must be between 0 and 3")
        self.degree = int(degree)

    @torch.no_grad()
    def _encode(self, lon: np.ndarray, lat: np.ndarray, _year: np.ndarray | None) -> np.ndarray:
        x, y, z = _unit_xyz(lon, lat).unbind(dim=1)
        one = torch.ones_like(x)
        features = [0.28209479177387814 * one]
        if self.degree >= 1:
            features.extend(
                (-0.4886025119029199 * y, 0.4886025119029199 * z, -0.4886025119029199 * x)
            )
        if self.degree >= 2:
            features.extend(
                (
                    1.0925484305920792 * x * y,
                    -1.0925484305920792 * y * z,
                    0.31539156525252005 * (3.0 * z.square() - 1.0),
                    -1.0925484305920792 * x * z,
                    0.5462742152960396 * (x.square() - y.square()),
                )
            )
        if self.degree >= 3:
            features.extend(
                (
                    -0.5900435899266435 * y * (3.0 * x.square() - y.square()),
                    2.890611442640554 * x * y * z,
                    -0.4570457994644658 * y * (5.0 * z.square() - 1.0),
                    0.3731763325901154 * z * (5.0 * z.square() - 3.0),
                    -0.4570457994644658 * x * (5.0 * z.square() - 1.0),
                    1.445305721320277 * z * (x.square() - y.square()),
                    -0.5900435899266435 * x * (x.square() - 3.0 * y.square()),
                )
            )
        return torch.stack(features, dim=1).numpy()


class MINDLocationEncoder(LocationEncoder):
    """MIND location encoder (distilled from AlphaEarth/Climplicit/GeoCLIP/SINR).

    Loads a released checkpoint from the HuggingFace Hub. Two configs ship:
    ``mind`` (the 64-d Matryoshka deploy prefix of the pooled trunk) and
    ``mind_small`` (the distilled student's 128-d head output). ``feature`` picks
    the trunk (``pooled``) or the projected head (``head``); ``dim`` truncates the
    Matryoshka embedding.

    Args:
        repo: HuggingFace repo id holding the weights.
        filename: Checkpoint file within the repo (``.safetensors`` or ``.pt``).
        dim: Embedding width to keep (Matryoshka prefix).
        feature: ``"pooled"`` (trunk) or ``"head"`` (projected output).
        default_year: Year supplied to year-conditioned checkpoints when a
            benchmark carries no per-point year.
    """

    name = "mind"

    def __init__(
        self,
        repo: str = "isaaccorley/MIND",
        filename: str = "mind.safetensors",
        dim: int = 64,
        feature: str = "pooled",
        default_year: int = 2021,
        device: str = "cpu",
        batch_size: int = 8192,
    ) -> None:
        super().__init__(device=device, batch_size=batch_size)
        from huggingface_hub import hf_hub_download

        from torchgeo_bench.coordbench.mind import load_mind

        self.model = load_mind(hf_hub_download(repo, filename), device=self.device)
        self.dim = int(dim)
        self.feature = feature
        self.default_year = default_year

    @torch.no_grad()
    def _encode(self, lon: np.ndarray, lat: np.ndarray, year: np.ndarray | None) -> np.ndarray:
        latlon = torch.stack([torch.as_tensor(lat), torch.as_tensor(lon)], dim=1).float()
        yr = None
        if self.model.use_year:
            y = self.default_year if year is None else year
            yr = torch.as_tensor(np.broadcast_to(y, (len(lat),)), dtype=torch.float32)
            yr = yr.to(self.device)
        emb = self.model(latlon.to(self.device), yr, return_features=(self.feature == "pooled"))
        return emb.float().cpu().numpy()[:, : self.dim]


class _RSHFEncoder(LocationEncoder):
    """Base for pretrained encoders loaded from the ``rshf`` package.

    Subclasses build ``self.model`` in :meth:`__init__` and set
    :attr:`coord_order` (``"lonlat"`` or ``"latlon"``). The default
    :meth:`_encode` stacks coordinates in that order and calls the model.
    """

    coord_order: str = "lonlat"
    dtype: torch.dtype = torch.float32

    @torch.no_grad()
    def _encode(self, lon: np.ndarray, lat: np.ndarray, _year: np.ndarray | None) -> np.ndarray:
        first, second = (lon, lat) if self.coord_order == "lonlat" else (lat, lon)
        x = torch.stack([torch.as_tensor(first), torch.as_tensor(second)], dim=1).to(
            self.device, self.dtype
        )
        return self.model(x).float().cpu().numpy()


class ClimplicitLocationEncoder(_RSHFEncoder):
    """Climplicit climate-specialist encoder (CHELSA, ReSIREN) -> 1024-d.

    Requires the ``coordbench`` extra (``rshf``).
    """

    name = "climplicit"
    coord_order = "lonlat"

    def __init__(
        self,
        repo: str = "Jobedo/climplicit",
        device: str = "cpu",
        batch_size: int = 8192,
    ) -> None:
        super().__init__(device=device, batch_size=batch_size)
        from rshf.climplicit import Climplicit

        self.model = (
            Climplicit.from_pretrained(repo, config={"return_chelsa": False}).to(self.device).eval()
        )


class SINRLocationEncoder(_RSHFEncoder):
    """SINR location encoder (Cole et al. 2023) -> 256-d features.

    Requires the ``coordbench`` extra (``rshf``).
    """

    name = "sinr"
    coord_order = "lonlat"

    def __init__(
        self,
        repo: str = "MVRL/sinr-location-encoder-1000-cls",
        device: str = "cpu",
        batch_size: int = 8192,
    ) -> None:
        super().__init__(device=device, batch_size=batch_size)
        import json
        from pathlib import Path

        from huggingface_hub import hf_hub_download
        from rshf.sinr import SINR, SINRConfig

        cfg = json.loads(Path(hf_hub_download(repo, "config.json")).read_text())
        conf = SINRConfig(
            num_inputs=cfg["num_inputs"],
            num_filts=cfg["num_filts"],
            depth=cfg["depth"],
            num_classes=cfg["num_classes"],
        )
        self.model = SINR.from_pretrained(repo, config=conf).to(self.device).eval()

    @torch.no_grad()
    def _encode(self, lon: np.ndarray, lat: np.ndarray, _year: np.ndarray | None) -> np.ndarray:
        from rshf.sinr import preprocess_locs

        x = torch.stack([torch.as_tensor(lon), torch.as_tensor(lat)], dim=1).float().to(self.device)
        return self.model(preprocess_locs(x), return_feats=True).float().cpu().numpy()


class GeoCLIPLocationEncoder(_RSHFEncoder):
    """GeoCLIP location encoder (Equal-Earth + RFF) -> 512-d. Input order (lat, lon).

    Requires the ``coordbench`` extra (``rshf``). Note the weights expect (lat, lon)
    despite the upstream docstring's "Lon/Lat" example.
    """

    name = "geoclip"
    coord_order = "latlon"

    def __init__(
        self,
        repo: str = "MVRL/geoclip-location-encoder",
        device: str = "cpu",
        batch_size: int = 8192,
    ) -> None:
        super().__init__(device=device, batch_size=batch_size)
        import json
        from pathlib import Path

        from huggingface_hub import hf_hub_download
        from rshf.geoclip import GeoCLIP, GeoCLIPConfig
        from safetensors.torch import load_file

        # Build from config + weights explicitly: PyTorchModelHubMixin.from_pretrained
        # no longer reconstructs the GeoCLIPConfig under recent huggingface_hub.
        cfg = json.loads(Path(hf_hub_download(repo, "config.json")).read_text())
        config = GeoCLIPConfig(
            sigma=cfg["sigma"],
            input_size=cfg["input_size"],
            encoded_size=cfg["encoded_size"],
            dim=cfg["dim"],
        )
        model = GeoCLIP(config)
        model.load_state_dict(load_file(hf_hub_download(repo, "model.safetensors")))
        self.model = model.to(self.device).eval()


class SatCLIPLocationEncoder(_RSHFEncoder):
    """SatCLIP location encoder (spherical harmonics + SirenNet). Input order (lon, lat).

    Runs in float64 (the released weights are double precision). Requires the
    ``coordbench`` extra (``rshf``).
    """

    name = "satclip"
    coord_order = "lonlat"
    dtype = torch.float64

    def __init__(
        self,
        repo: str = "MVRL/satclip-loc-enc-vit16-l40",
        device: str = "cpu",
        batch_size: int = 8192,
    ) -> None:
        super().__init__(device=device, batch_size=batch_size)
        from rshf.satclip import SatClip

        self.model = SatClip.from_pretrained(repo).double().to(self.device).eval()
