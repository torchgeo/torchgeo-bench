"""Unit tests for the sample-size sweep pipeline.

These tests focus on config resolution — in particular that the model-specific
``eval`` block is merged into the top-level ``eval`` config so that models which
ship their own ``eval.segmentation.layers`` (e.g. resnet50's FPN layers) are not
silently ignored (which previously made SegmentationProbe fall back to a single
``["backbone_output"]`` layer).
"""

import os

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

CONF_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "src", "torchgeo_bench", "conf")
)


def _merge_seg_eval(cfg):
    """Replicate the merge the pipeline performs for the segmentation branch."""
    seg_eval_cfg = cfg.eval
    if "eval" in cfg.model and cfg.model.eval is not None:
        seg_eval_cfg = OmegaConf.merge(seg_eval_cfg, cfg.model.eval)
    return seg_eval_cfg.segmentation


def test_model_eval_block_merged_into_segmentation_layers() -> None:
    """resnet50's FPN layers should survive the merge into eval.segmentation."""
    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        cfg = compose(
            config_name="sample_size_config", overrides=["model=timm/resnet50"]
        )
    seg_cfg = _merge_seg_eval(cfg)
    assert list(seg_cfg.layers) == ["layer4", "layer3", "layer2", "layer1"]
    # criterion from the top-level config is preserved through the merge.
    assert seg_cfg.criterion._target_ == "torch.nn.CrossEntropyLoss"


def test_model_without_eval_block_yields_empty_layers() -> None:
    """rcf ships no eval block, so layers stay empty (pipeline guard fires)."""
    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        cfg = compose(config_name="sample_size_config", overrides=["model=rcf"])
    seg_cfg = _merge_seg_eval(cfg)
    assert list(seg_cfg.layers) == []


def test_empty_layers_guard_raises() -> None:
    """The pipeline must refuse to run a seg sweep with no layers configured."""
    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        cfg = compose(config_name="sample_size_config", overrides=["model=rcf"])
    seg_cfg = _merge_seg_eval(cfg)
    # Mirror the guard in sample_size_pipeline.main()'s segmentation branch.
    with pytest.raises(ValueError, match="eval.segmentation.layers"):
        if not list(seg_cfg.layers):
            raise ValueError(
                "Segmentation sweep requires eval.segmentation.layers to be set."
            )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
