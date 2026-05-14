import pytest
from hydra import compose, initialize_config_module
from hydra.errors import HydraException


def test_uq_config_composes():
    with initialize_config_module(config_module="torchgeo_bench.conf", version_base="1.3"):
        cfg = compose(config_name="uq_config", overrides=["model=timm/resnet50"])
    assert cfg.uq.cal_size == 400
    assert cfg.uq.n_ensemble == 5
    assert cfg.uq.ece_bins == 15
    assert cfg.uq.ece_binning == "equal_width"
    assert cfg.seed == 42


def test_uq_config_requires_model():
    with initialize_config_module(
        config_module="torchgeo_bench.conf", version_base="1.3"
    ), pytest.raises(HydraException):
        compose(config_name="uq_config")


def test_cka_config_composes():
    with initialize_config_module(config_module="torchgeo_bench.conf", version_base="1.3"):
        cfg = compose(config_name="cka_config", overrides=["model=timm/resnet50"])
    assert cfg.cka.cal_size == 400
    assert cfg.cka.output == "results/cka_results.csv"
    assert cfg.seed == 42


def test_cka_config_requires_model():
    with initialize_config_module(
        config_module="torchgeo_bench.conf", version_base="1.3"
    ), pytest.raises(HydraException):
        compose(config_name="cka_config")


def test_cka_layers_all_have_four_paths():
    with initialize_config_module(config_module="torchgeo_bench.conf", version_base="1.3"):
        cfg = compose(config_name="cka_config", overrides=["model=timm/resnet50"])
    for paths in cfg.cka.layers.values():
        assert len(paths) == 4
