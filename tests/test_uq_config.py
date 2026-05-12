import pytest
from hydra import compose, initialize_config_module
from hydra.errors import HydraException


def test_uq_config_composes():
    with initialize_config_module(config_module="torchgeo_bench.conf", version_base="1.3"):
        cfg = compose(config_name="uq_config", overrides=["model=timm/resnet50"])
    assert cfg.uq.cal_size == 400
    assert cfg.uq.n_ensemble == 5
    assert cfg.seed == 42


def test_uq_config_requires_model():
    with initialize_config_module(config_module="torchgeo_bench.conf", version_base="1.3"):
        with pytest.raises(HydraException):
            compose(config_name="uq_config")
