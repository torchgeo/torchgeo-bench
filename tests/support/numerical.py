"""Scoped randomness for tests that construct or train PyTorch modules."""

from collections.abc import Iterator

import pytest
import torch


@pytest.fixture
def isolated_torch_rng() -> Iterator[None]:
    """Restore random state and matmul settings changed by model constructors."""
    precision = torch.get_float32_matmul_precision()
    with torch.random.fork_rng():
        torch.manual_seed(0)
        try:
            yield
        finally:
            torch.set_float32_matmul_precision(precision)
