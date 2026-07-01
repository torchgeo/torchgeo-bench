"""Visualization utilities for segmentation probe evaluation."""

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


def _entropy_to_rgb(entropy: np.ndarray) -> np.ndarray:
    """Map a (H, W) float32 normalized-entropy map to (H, W, 3) uint8 via plasma colormap."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for segmentation visualization. "
            "Install it with: pip install torchgeo-bench[viz]"
        ) from e

    cmap = plt.get_cmap("plasma")
    rgba = cmap(np.clip(entropy, 0.0, 1.0))  # (H, W, 4) float64
    return (rgba[:, :, :3] * 255).astype(np.uint8)


def render_uq_subsample_grid(
    images: "torch.Tensor",
    gt_masks: "torch.Tensor",
    fraction_results: list[dict],
    num_classes: int,
    rgb_indices: list[int],
    ignore_index: int = 255,
) -> np.ndarray:
    """Build a UQ subsample grid comparing predictions across training fractions.

    Layout (one row per image):
      [RGB image | GT mask] | [Pred_f1 / Entropy_f1] | [Pred_f2 / Entropy_f2] | ...

    The RGB image and GT mask appear once on the left; each fraction column
    stacks the predicted mask above the normalized-entropy heatmap (plasma).
    Column headers label the fraction percentage.

    Args:
        images: (N, C, H, W) float tensor of the selected test images.
        gt_masks: (N, H, W) int64 ground-truth masks.
        fraction_results: List of dicts, one per fraction, each with keys:
            ``fraction`` (float), ``preds`` (N, H, W int64 Tensor),
            ``entropy`` (N, H, W float32 Tensor, normalized to [0, 1]).
        num_classes: Number of segmentation classes.
        rgb_indices: Channel indices [R, G, B] into the C dimension.
        ignore_index: Label value rendered as white.

    Returns:
        (H_grid, W_grid, 3) uint8 numpy array.
    """
    n = len(images)
    h, w = images.shape[-2], images.shape[-1]
    sep = 2  # 2-pixel separator between pred and entropy sub-panels

    # Pre-render left block panels (image + GT) for each sample row
    left_panels: list[np.ndarray] = []
    for i in range(n):
        img = images[i].cpu().numpy()
        gt = gt_masks[i].cpu().numpy()
        ri = [min(c, img.shape[0] - 1) for c in rgb_indices]
        rgb_u8 = _denorm_image(img[ri, :, :].transpose(1, 2, 0))
        gt_u8 = colorize_mask(gt, num_classes, ignore_index)
        left_panels.append(np.concatenate([rgb_u8, gt_u8], axis=1))  # (H, 2W, 3)

    # Build fraction column headers
    left_header_w = 2 * w
    frac_labels = [f"{int(round(fr['fraction'] * 100))}% train" for fr in fraction_results]
    col_labels = ["Image", "GT"] + frac_labels

    # Pre-render per-fraction panels for each sample
    frac_panels: list[list[np.ndarray]] = []  # [fraction_idx][sample_idx]
    for fr in fraction_results:
        preds = fr["preds"]
        entropy = fr["entropy"]
        col: list[np.ndarray] = []
        for i in range(n):
            pred_u8 = colorize_mask(preds[i].cpu().numpy(), num_classes, ignore_index)
            ent_u8 = _entropy_to_rgb(entropy[i].cpu().numpy())
            sep_bar = np.full((sep, w, 3), 40, dtype=np.uint8)
            col.append(np.concatenate([pred_u8, sep_bar, ent_u8], axis=0))  # (2H+sep, W, 3)
        frac_panels.append(col)

    # Build rows: left block (H tall) + fraction columns (2H+sep tall) — pad left to match
    rows: list[np.ndarray] = []
    frac_col_h = 2 * h + sep
    for i in range(n):
        left = left_panels[i]  # (H, 2W, 3)
        # Pad left block to match fraction column height
        pad = np.full((frac_col_h - h, 2 * w, 3), 20, dtype=np.uint8)
        left_padded = np.concatenate([left, pad], axis=0)  # (2H+sep, 2W, 3)
        frac_cols = [fp[i] for fp in frac_panels]  # each (2H+sep, W, 3)
        row = np.concatenate([left_padded, *frac_cols], axis=1)
        rows.append(row)

    grid = np.concatenate(rows, axis=0)

    # Prepend column header row
    total_w = 2 * w + len(fraction_results) * w
    num_cols = 2 + len(fraction_results)
    col_widths = [w, w] + [w] * len(fraction_results)
    header = _make_variable_header_row(col_widths, col_labels)
    return np.concatenate([header, grid], axis=0)


def _make_variable_header_row(col_widths: list[int], labels: list[str], height: int = 24) -> np.ndarray:
    """Return a header banner with per-column widths."""
    try:
        from PIL import Image, ImageDraw
    except ImportError as e:
        raise ImportError(
            "Pillow is required for segmentation visualization. "
            "Install it with: pip install torchgeo-bench[viz]"
        ) from e

    total_w = sum(col_widths)
    header_pil = Image.new("RGB", (total_w, height), color=(40, 40, 40))
    draw = ImageDraw.Draw(header_pil)
    x_offset = 0
    for cw, label in zip(col_widths, labels):
        bbox = draw.textbbox((0, 0), label)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = x_offset + cw // 2 - tw // 2
        y = height // 2 - th // 2
        draw.text((x, y), label, fill=(220, 220, 220))
        x_offset += cw
    return np.asarray(header_pil)


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
