"""CPU↔CUDA parity for :class:`KNNClassifier`.

Ported from the former ``experiments/scripts/test_knn_gpu_smoke.py`` smoke
script. Skipped when CUDA is unavailable.
"""

import numpy as np
import pytest
import torch

from torchgeo_bench.knn import KNNClassifier

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA for cpu/cuda parity"
)

N_TRAIN, N_TEST, DIM, K, N_CLASSES = 2000, 500, 64, 5, 10


@pytest.fixture(scope="module")
def features() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x_train = rng.standard_normal((N_TRAIN, DIM)).astype(np.float32)
    x_test = rng.standard_normal((N_TEST, DIM)).astype(np.float32)
    return x_train, x_test


def test_singlelabel_parity(features: tuple[np.ndarray, np.ndarray]) -> None:
    x_train, x_test = features
    rng = np.random.default_rng(1)
    y_train = rng.integers(0, N_CLASSES, size=N_TRAIN).astype(np.int64)

    cpu = KNNClassifier(n_neighbors=K, device="cpu").fit(x_train, y_train)
    cu = KNNClassifier(n_neighbors=K, device="cuda").fit(x_train, y_train)

    assert (cpu.predict(x_test) == cu.predict(x_test)).all()

    pp_cpu = cpu.predict_proba(x_test)
    pp_cu = cu.predict_proba(x_test)
    assert pp_cpu.shape == pp_cu.shape
    np.testing.assert_allclose(pp_cpu, pp_cu, atol=1e-5)


def test_multilabel_parity(features: tuple[np.ndarray, np.ndarray]) -> None:
    x_train, x_test = features
    rng = np.random.default_rng(2)
    y_train = (rng.random((N_TRAIN, N_CLASSES)) > 0.7).astype(np.int64)

    cpu = KNNClassifier(n_neighbors=K, device="cpu").fit(x_train, y_train)
    cu = KNNClassifier(n_neighbors=K, device="cuda").fit(x_train, y_train)

    p_cpu = cpu.predict(x_test)
    p_cu = cu.predict(x_test)
    assert p_cpu.shape == p_cu.shape
    assert (p_cpu == p_cu).all()

    np.testing.assert_allclose(cpu.predict_proba(x_test), cu.predict_proba(x_test), atol=1e-5)
