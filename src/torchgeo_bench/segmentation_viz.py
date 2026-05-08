"""Visualization utilities for segmentation probe evaluation."""

<<<<<<< HEAD
from __future__ import annotations

=======
>>>>>>> main
import logging
import os

import numpy as np
import torch

logger = logging.getLogger(__name__)

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
    try:
        from PIL import Image, ImageDraw
    except ImportError as e:
        raise ImportError(
            "Pillow is required for segmentation visualization. "
            "Install it with: pip install torchgeo-bench[viz]"
        ) from e

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
    images: torch.Tensor,
    gt_masks: torch.Tensor,
    pred_masks: torch.Tensor,
    num_classes: int,
    rgb_indices: list[int],
    n_samples: int = 8,
    ignore_index: int = 255,
) -> np.ndarray:
    """Build a visualization grid for up to n_samples test samples.

    Each row shows: [RGB image | GT mask | Pred mask | Error map]

    Args:
        images: (N, C, H, W) float tensor, normalized (will be min-max stretched).
        gt_masks: (N, H, W) int64 tensor.
        pred_masks: (N, H, W) int64 tensor.
        num_classes: Number of segmentation classes.
        rgb_indices: Channel indices [R, G, B] into the C dimension.
        n_samples: Maximum number of samples to visualise.
        ignore_index: Label value treated as ignore.

    Returns:
        (H_grid, W_grid, 3) uint8 numpy array.
    """
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
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for segmentation visualization. "
            "Install it with: pip install torchgeo-bench[viz]"
        ) from e

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
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    arr = buf.reshape(h, w, 4)[:, :, :3].copy()  # drop alpha
    plt.close(fig)
    return arr


def save_segmentation_viz(
    out_dir: str,
    model_name: str,
    dataset_name: str,
    images: torch.Tensor,
    gt_masks: torch.Tensor,
    pred_masks: torch.Tensor,
    num_classes: int,
    rgb_indices: list[int],
    ignore_index: int = 255,
    n_samples: int = 8,
    class_names: list[str] | None = None,
) -> None:
    """Save a sample grid PNG and a confusion matrix PNG.

    Files are written to ``{out_dir}/{model_name}/``:
      - ``{dataset_name}_samples.png``
      - ``{dataset_name}_confusion.png``

    Args:
        out_dir: Root visualization output directory.
        model_name: Model name (used as sub-directory).
        dataset_name: Dataset name (used in filenames).
        images: (N, C, H, W) float tensor.
        gt_masks: (N, H, W) int64 ground truth.
        pred_masks: (N, H, W) int64 predictions.
        num_classes: Number of segmentation classes.
        rgb_indices: Channel indices [R, G, B] for image rendering.
        ignore_index: Label value to exclude from metrics/confusion.
        n_samples: Number of sample rows in the grid image.
        class_names: Optional class name strings for confusion matrix axis labels.
    """
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError(
            "Pillow is required for segmentation visualization. "
            "Install it with: pip install torchgeo-bench[viz]"
        ) from e
    dest = os.path.join(out_dir, model_name)
    os.makedirs(dest, exist_ok=True)

    # Sample grid
    grid = render_sample_grid(
        images, gt_masks, pred_masks, num_classes, rgb_indices, n_samples, ignore_index
    )
    grid_path = os.path.join(dest, f"{dataset_name}_samples.png")
    Image.fromarray(grid).save(grid_path)
    logger.info(f"Saved sample grid → {grid_path}")

    # Confusion matrix
    cm_arr = render_confusion_matrix(pred_masks, gt_masks, num_classes, ignore_index, class_names)
    cm_path = os.path.join(dest, f"{dataset_name}_confusion.png")
    Image.fromarray(cm_arr).save(cm_path)
    logger.info(f"Saved confusion matrix → {cm_path}")
