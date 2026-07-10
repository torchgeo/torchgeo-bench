import io
import logging
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
import torch
import torch.nn as nn

from torchgeo_bench.datasets.base import BandSpec

HF_UNAUTHENTICATED_WARNING = (
    "Warning: You are sending unauthenticated requests to the HF Hub. "
    "Please set a HF_TOKEN to enable higher rate limits and faster downloads."
)
HF_AUTH_FAILURE_WARNING = "The token from HF_TOKEN environment variable is invalid."
HF_HTTP_WARNING = "HTTP Error 401 thrown while requesting GET https://huggingface.co/timm/example"


def _rgb_bands() -> list[BandSpec]:
    return [
        BandSpec(
            sensor="s2",
            name=name,
            source_name=name.upper(),
            mean=1500.0,
            std=600.0,
            min=0.0,
            max=10000.0,
        )
        for name in ("red", "green", "blue")
    ]


@contextmanager
def _captured_hf_logging() -> Iterator[tuple[io.StringIO, io.StringIO]]:
    root_logger = logging.getLogger()
    hf_logger = logging.getLogger("huggingface_hub")

    old_root_handlers = root_logger.handlers[:]
    old_root_level = root_logger.level
    old_hf_handlers = hf_logger.handlers[:]
    old_hf_level = hf_logger.level
    old_hf_propagate = hf_logger.propagate

    root_stream = io.StringIO()
    hf_stream = io.StringIO()
    root_handler = logging.StreamHandler(root_stream)
    hf_handler = logging.StreamHandler(hf_stream)
    root_handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
    hf_handler.setFormatter(logging.Formatter("%(message)s"))

    root_logger.handlers = [root_handler]
    root_logger.setLevel(logging.WARNING)
    hf_logger.handlers = [hf_handler]
    hf_logger.setLevel(logging.WARNING)
    hf_logger.propagate = True

    try:
        yield root_stream, hf_stream
    finally:
        root_logger.handlers = old_root_handlers
        root_logger.setLevel(old_root_level)
        hf_logger.handlers = old_hf_handlers
        hf_logger.setLevel(old_hf_level)
        hf_logger.propagate = old_hf_propagate


class _TinyBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.default_cfg: dict[str, object] = {}

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return images.mean(dim=(-2, -1))


def test_hf_warning_matcher_is_specific():
    from torchgeo_bench.models.timm import _is_hf_hub_unauthenticated_warning

    assert _is_hf_hub_unauthenticated_warning(HF_UNAUTHENTICATED_WARNING)
    assert _is_hf_hub_unauthenticated_warning(HF_UNAUTHENTICATED_WARNING.removeprefix("Warning: "))
    assert not _is_hf_hub_unauthenticated_warning(HF_AUTH_FAILURE_WARNING)
    assert not _is_hf_hub_unauthenticated_warning(HF_HTTP_WARNING)


def test_pretrained_timm_suppresses_only_unauthenticated_hf_warning(
    monkeypatch: pytest.MonkeyPatch,
):
    import timm

    from torchgeo_bench.models.timm import TimmPatchBenchModel

    def _fake_create_model(*args, **kwargs):
        del args, kwargs
        logger = logging.getLogger("huggingface_hub.utils._http")
        logger.warning(HF_UNAUTHENTICATED_WARNING)
        logger.warning(HF_AUTH_FAILURE_WARNING)
        logger.warning(HF_HTTP_WARNING)
        return _TinyBackbone()

    monkeypatch.setattr(timm, "create_model", _fake_create_model)

    with _captured_hf_logging() as (root_stream, hf_stream):
        TimmPatchBenchModel(
            bands=_rgb_bands(),
            model_name="resnet18",
            pretrained=True,
        )

    root_output = root_stream.getvalue()
    hf_output = hf_stream.getvalue()
    combined_output = root_output + hf_output

    assert "unauthenticated requests to the HF Hub" not in combined_output
    assert HF_AUTH_FAILURE_WARNING in root_output
    assert HF_AUTH_FAILURE_WARNING in hf_output
    assert HF_HTTP_WARNING in root_output
    assert HF_HTTP_WARNING in hf_output
