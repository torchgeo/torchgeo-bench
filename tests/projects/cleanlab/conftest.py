"""Optional Cleanlab dependencies, isolated from the core test suite."""

from functools import partial
from types import ModuleType

import pytest


@pytest.fixture
def cleanlab_filter(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    module = pytest.importorskip(
        "cleanlab.filter",
        reason="Install projects/cleanlab/requirements.txt to test real Cleanlab reports",
        exc_type=ModuleNotFoundError,
    )
    monkeypatch.setattr(module, "find_label_issues", partial(module.find_label_issues, n_jobs=1))
    return module
