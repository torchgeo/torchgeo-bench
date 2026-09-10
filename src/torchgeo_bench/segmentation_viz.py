"""Visualization utilities for segmentation probe evaluation."""

import logging
import os
from dataclasses import dataclass

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class SegmentationSamples:
    """Images, ground truth, and predictions for the same samples."""

    images: torch.Tensor
    targets: torch.Tensor
    predictions: torch.Tensor


@dataclass
class SegmentationVizSpec:
    """Class labels and image channels used in segmentation plots."""

    num_classes: int
    rgb_indices: list[int]
    ignore_index: int = 255
    class_names: list[str] | None = None


# Fixed palette: tab20 (20 colours) concatenated with tab20b (20 more) for up to 40 classes.
# Index 0 is black (background / class 0). 255 (ignore) is rendered as white.
_TAB20_COLORS: list[tuple[int, int, int]] = [
    (31, 119, 180),
    (174, 199, 232),
    (255, 127, 14),
    (255, 187, 120),
    (44, 160, 44),
    (152, 223, 138),
    (214, 39, 40),
    (255, 152, 150),
    (148, 103, 189),
    (197, 176, 213),
    (140, 86, 75),
    (196, 156, 148),
    (227, 119, 194),
    (247, 182, 210),
    (127, 127, 127),
    (199, 199, 199),
    (188, 189, 34),
    (219, 219, 141),
    (23, 190, 207),
    (158, 218, 229),
]
_TAB20B_COLORS: list[tuple[int, int, int]] = [
    (57, 59, 121),
    (82, 84, 163),
    (107, 110, 207),
    (156, 158, 222),
    (99, 121, 57),
    (140, 162, 82),
    (181, 207, 107),
    (206, 219, 156),
    (140, 109, 49),
    (189, 158, 57),
    (231, 186, 82),
    (231, 203, 148),
    (132, 60, 57),
    (173, 73, 74),
    (214, 97, 107),
    (231, 150, 156),
    (123, 65, 115),
    (165, 81, 148),
    (206, 109, 189),
    (222, 158, 214),
]
_PALETTE: list[tuple[int, int, int]] = _TAB20_COLORS + _TAB20B_COLORS  # 40 entries


def _build_colormap(num_classes: int) -> np.ndarray:
    """Return (num_classes+1, 3) uint8 array; index num_classes → white for ignore."""
    colors = np.zeros((num_classes + 1, 3), dtype=np.uint8)
    for i in range(num_classes):
        colors[i] = _PALETTE[i % len(_PALETTE)]
    colors[num_classes] = (255, 255, 255)  # ignore index placeholder
    return colors


def colorize_mask(mask: np.ndarray, num_classes: int, ignore_index: int = 255) -> np.ndarray:
    """Map a (H, W) integer mask to an (H, W, 3) uint8 RGB image.

    Args:
        mask: Integer class map, shape (H, W).
        num_classes: Total number of classes (defines colormap size).
        ignore_index: Pixels with this value are rendered white.

    Returns:
        RGB image, shape (H, W, 3), uint8.
    """
    colormap = _build_colormap(num_classes)
    # Clamp ignore_index values to the last slot (white).
    idx = mask.copy()
    idx[idx == ignore_index] = num_classes
    idx = np.clip(idx, 0, num_classes)
    return colormap[idx]


def _denorm_image(img: np.ndarray) -> np.ndarray:
    """Stretch a (H, W, 3) float array to uint8 via per-channel min-max normalisation."""
    out = np.zeros_like(img, dtype=np.float32)
    for c in range(img.shape[2]):
        lo, hi = img[:, :, c].min(), img[:, :, c].max()
        if hi > lo:
            out[:, :, c] = (img[:, :, c] - lo) / (hi - lo)
        else:
            out[:, :, c] = 0.0
    return (out * 255).clip(0, 255).astype(np.uint8)


def render_error_map(gt: np.ndarray, pred: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    """Return an (H, W, 3) uint8 error map.

    Colour coding:
      - Green  : correct prediction
      - Red    : false negative (GT has class, pred is different)
      - Blue   : false positive (pred has class, GT is different)
      - White  : ignored pixel
    """
    h, w = gt.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    valid = gt != ignore_index
    correct = valid & (gt == pred)
    fn = valid & (gt != pred) & (pred == 0)  # model missed the class (predicted background)
    fp = valid & (gt != pred) & (gt == 0)  # model hallucinated (GT is background)
    other = valid & (gt != pred) & ~fn & ~fp  # class-to-class confusion

    out[correct] = (80, 200, 80)  # green
    out[fn] = (220, 50, 50)  # red
    out[fp] = (50, 100, 220)  # blue
    out[other] = (220, 160, 50)  # orange — class confusion
    out[~valid] = (255, 255, 255)  # white for ignore
    return out


def _make_header_row(
    col_width: int, num_cols: int, labels: list[str], height: int = 24
) -> np.ndarray:
    """Return a (height, num_cols*col_width, 3) uint8 header banner with centered column labels."""
    from PIL import Image, ImageDraw

    header_pil = Image.new("RGB", (num_cols * col_width, height), color=(40, 40, 40))
    draw = ImageDraw.Draw(header_pil)
    for i, label in enumerate(labels):
        x_center = i * col_width + col_width // 2
        bbox = draw.textbbox((0, 0), label)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = x_center - tw // 2
        y = height // 2 - th // 2
        draw.text((x, y), label, fill=(220, 220, 220))
    return np.asarray(header_pil)


def render_sample_grid(
    samples: SegmentationSamples,
    spec: SegmentationVizSpec,
    n_samples: int = 8,
) -> np.ndarray:
    """Render RGB images, ground truth, predictions, and errors in a sample grid."""
    images, gt_masks, pred_masks = samples.images, samples.targets, samples.predictions
    num_classes, rgb_indices, ignore_index = spec.num_classes, spec.rgb_indices, spec.ignore_index
    n = min(n_samples, len(images))
    # Deterministic sample selection: evenly spaced across the test set
    indices = np.linspace(0, len(images) - 1, n, dtype=int)

    panels: list[np.ndarray] = []

    for idx in indices:
        img = images[idx].cpu().numpy()  # (C, H, W)
        gt = gt_masks[idx].cpu().numpy()  # (H, W)
        pred = pred_masks[idx].cpu().numpy()  # (H, W)

        # RGB image: pick channels, transpose to (H, W, 3)
        ri = [min(c, img.shape[0] - 1) for c in rgb_indices]
        rgb = img[ri, :, :].transpose(1, 2, 0)  # (H, W, 3)
        rgb_u8 = _denorm_image(rgb)

        gt_u8 = colorize_mask(gt, num_classes, ignore_index)
        pred_u8 = colorize_mask(pred, num_classes, ignore_index)
        err_u8 = render_error_map(gt, pred, ignore_index)

        row = np.concatenate([rgb_u8, gt_u8, pred_u8, err_u8], axis=1)  # (H, 4*W, 3)
        panels.append(row)

    grid = np.concatenate(panels, axis=0)  # (n*H, 4*W, 3)

    # Prepend a header row with column labels
    col_width = images.shape[-1]  # W dimension
    header = _make_header_row(col_width, 4, ["Image", "Ground Truth", "Prediction", "Error Map"])
    return np.concatenate([header, grid], axis=0)


def render_confusion_matrix(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
    class_names: list[str] | None = None,
) -> np.ndarray:
    """Build a normalised confusion matrix as a (H, W, 3) uint8 heatmap image.

    Args:
        preds: (N, H, W) int64 predicted class maps.
        targets: (N, H, W) int64 ground truth class maps.
        num_classes: Number of classes.
        ignore_index: Label value to exclude.
        class_names: Optional list of class name strings for axis labels.

    Returns:
        (H_img, W_img, 3) uint8 heatmap rendered with matplotlib Blues colormap.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    # Flatten and mask out ignored pixels
    p = preds.reshape(-1).numpy()
    t = targets.reshape(-1).numpy()
    valid = t != ignore_index
    p, t = p[valid], t[valid]

    # Clamp predictions to valid range
    p = np.clip(p, 0, num_classes - 1)
    t = np.clip(t, 0, num_classes - 1)

    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (t, p), 1)

    # Row-normalise (true-class frequencies)
    row_sums = cm.sum(axis=1, keepdims=True).astype(np.float64)
    row_sums = np.where(row_sums == 0, 1, row_sums)
    cm_norm = cm.astype(np.float64) / row_sums

    fig_size = max(6, num_classes * 0.4)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Integer-only ticks; use class names when available
    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    if class_names and len(class_names) == num_classes:
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)

    # Annotate each cell with the percentage value
    font_size = max(5, min(9, 72 // num_classes))
    for row in range(num_classes):
        for col in range(num_classes):
            val = cm_norm[row, col]
            color = "white" if val > 0.5 else "black"
            ax.text(
                col, row, f"{val:.0%}", ha="center", va="center", fontsize=font_size, color=color
            )

    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Confusion matrix (row-normalised)")
    plt.tight_layout()

    # Render to numpy array (tostring_rgb removed in matplotlib ≥ 3.8; use buffer_rgba)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    w, h = canvas.get_width_height()
    arr = buf.reshape(h, w, 4)[:, :, :3].copy()  # drop alpha
    plt.close(fig)
    return arr


def save_segmentation_viz(
    dest: str,
    dataset_name: str,
    samples: SegmentationSamples,
    spec: SegmentationVizSpec,
    n_samples: int = 8,
) -> None:
    """Save sample and confusion-matrix PNGs in the model's output directory."""
    from PIL import Image

    os.makedirs(dest, exist_ok=True)
    grid = render_sample_grid(samples, spec, n_samples)
    grid_path = os.path.join(dest, f"{dataset_name}_samples.png")
    Image.fromarray(grid).save(grid_path)
    logger.info("Saved sample grid to %s", grid_path)

    cm_arr = render_confusion_matrix(
        samples.predictions, samples.targets, spec.num_classes, spec.ignore_index, spec.class_names
    )
    cm_path = os.path.join(dest, f"{dataset_name}_confusion.png")
    Image.fromarray(cm_arr).save(cm_path)
    logger.info("Saved confusion matrix to %s", cm_path)


def collect_viz_inputs(test_loader: "object") -> tuple[torch.Tensor, torch.Tensor]:
    """Gather test images and GT masks for visualization (cheap pass, no backbone)."""
    images, masks = [], []
    for batch in test_loader:  # type: ignore[attr-defined]
        if isinstance(batch, dict):
            images.append(batch["image"])
            mask = batch["mask"]
        else:
            images.append(batch[0])
            mask = batch[1]
        if mask.ndim == 4:
            mask = mask.squeeze(1)
        masks.append(mask.long())
    return torch.cat(images, dim=0), torch.cat(masks, dim=0)
