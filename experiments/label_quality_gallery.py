"""Suspect-audit overlay gallery for the segmentation label-quality sweep.

Phase B of ``docs/plans/label_quality_fixes_and_viz.md``. For each
``(model, dataset, method)`` the fixed sweep persists a bounded set of per-image
``.npz`` artifacts (top-N most-suspect plus a random clean control) under
``results/label_quality/<dataset>/<model>/<method>/<image_id>.npz`` — each
carrying the pixel score map, the predicted mask, and the join metadata
(``native_id``, ``image_id``, ``image_score``, ``rank``).

This script joins those artifacts back to the source imagery and renders a grid:
one row per suspect image (source RGB | GT mask | predicted mask | pixel-score
heatmap), plus a contrast strip of clean-control images, so "suspect vs clean"
reads at a glance. It is the runnable-after-sweep half of the workflow whose
scalar counterpart is ``experiments/label_quality_analysis.py``.

Usage (after a fixed sweep has written npz artifacts):
    python experiments/label_quality_gallery.py \
        --root results \
        --out experiments/label_quality_figures \
        --n-suspect 12 --n-control 6

    # one (model, dataset, method) only
    python experiments/label_quality_gallery.py \
        --model resnet50 --dataset spacenet2 --method cleanlab
"""

import argparse
import glob
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

# Reuse the pipeline's house-style mask palette + image stretch.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from torchgeo_bench.segmentation_viz import _denorm_image, colorize_mask  # noqa: E402

_TERRAMIND_RE = re.compile(r".*terramind.*", re.IGNORECASE)

# Class names for the mask legend, indexed by the label values that actually
# appear in our masks. These are NOT always upstream geobench_v2's `classes`
# tuple: two datasets are remapped by our wrappers, so copying upstream
# verbatim would mislabel every pixel.
#
#   spacenet2 / spacenet7 -- upstream ships ("background", "no-building",
#     "building") after adding 1 to the native {0, 1} labels.
#     `canonicalize_sample` subtracts that offset again, dropping the
#     never-used "background", so our label 0 is "no-building" and 1 is
#     "building" (see the wrappers' num_classes = 2).
#   fotw -- upstream lists 3 classes but our wrapper declares 4 and the
#     predictions really do take values {0, 1, 2, 3}; the 4th is the "other"
#     class named in the wrapper docstring.
#
# Verified against the label values present in the persisted npz pred_masks.
# A dataset missing here (or a mask value past the end of its tuple) falls
# back to a plain "class <i>" label, so the legend never silently lies.
_CLASS_NAMES: dict[str, tuple[str, ...]] = {
    "caffe": ("N/A", "rock", "glacier", "ocean/ice melange"),
    "cloudsen12": ("clear", "thick cloud", "thin cloud", "cloud shadow"),
    "fotw": ("background", "field", "field-boundary", "other"),
    "spacenet2": ("no-building", "building"),
    "spacenet7": ("no-building", "building"),
    "flair2": (
        "building", "previous surface", "impervious surface", "bare soil",
        "water", "coniferous", "deciduous", "brushwood", "vineyard",
        "herbaceous vegetation", "agricultural land", "plowed land", "other",
    ),
}


def class_names(dataset: str, num_classes: int) -> list[str]:
    """Legend labels for ``dataset``, padded/truncated to ``num_classes``."""
    named = _CLASS_NAMES.get(str(dataset), ())
    return [named[i] if i < len(named) else f"class {i}" for i in range(num_classes)]


def canonical_model(name: str) -> str:
    """Canonical model name: terramind variants → ``terramind`` (matches analysis)."""
    return "terramind" if _TERRAMIND_RE.match(str(name)) else str(name)


def _artifact_glob(root: str, dataset: str, model_slug: str, method: str) -> list[str]:
    """All npz artifacts for one (dataset, model_slug, method), sorted by rank."""
    pattern = os.path.join(root, "label_quality", dataset, model_slug, method, "*.npz")
    return sorted(glob.glob(pattern))


def _discover(root: str) -> list[tuple[str, str, str]]:
    """Every (dataset, model_slug, method) triple with persisted npz artifacts."""
    triples = set()
    for path in glob.glob(os.path.join(root, "label_quality", "*", "*", "*", "*.npz")):
        rel = os.path.relpath(path, os.path.join(root, "label_quality"))
        parts = rel.split(os.sep)
        if len(parts) == 4:  # dataset / model_slug / method / <id>.npz
            triples.add((parts[0], parts[1], parts[2]))
    return sorted(triples)


def _load_artifacts(paths: list[str]) -> list[dict]:
    """Load npz artifacts into dicts, sorted most-suspect first (rank ascending)."""
    out = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            out.append({key: data[key] for key in data.files})
    out.sort(key=lambda d: int(d.get("rank", np.asarray(1 << 30))))
    return out


class _SourceImages:
    """Lazily reload source RGB + GT mask from a dataset, joined by native id.

    Falls back to the positional ``image_id`` when the sweep recorded no
    ``native_id`` (dataset without a stable id). Returns ``(None, None)`` if the
    dataset cannot be built (e.g. data not present locally); the gallery then
    renders placeholder panels rather than crashing.
    """

    def __init__(self, dataset: str):
        self.dataset = dataset
        self._bench = None
        self._ds = None
        self._by_native: dict[str, int] = {}
        self._num_classes = 0
        self._rgb_indices: list[int] = []
        self._ok = self._try_build()

    def _try_build(self) -> bool:
        try:
            from torchgeo_bench.datasets.loading import get_bench_dataset_class

            self._bench = get_bench_dataset_class(self.dataset)()
            self._num_classes = int(self._bench.num_classes)
            self._rgb_indices = list(range(len(self._bench.rgb_bands)))
            self._ds = self._bench.get_dataset(
                "train", bands=tuple(self._bench.rgb_bands), metadata=["lat", "lon"]
            )
            self._build_native_index()
            return True
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"  [warn] could not load source dataset {self.dataset!r}: {exc}")
            return False

    def _build_native_index(self) -> None:
        # Mirror run._dataset_native_ids: the V2 tortilla id column joins on
        # native_id; without it we fall back to positional lookup.
        inner = self._ds
        for _ in range(3):
            if hasattr(inner, "data_df"):
                break
            inner = getattr(inner, "_inner", inner)
        df = getattr(inner, "data_df", None)
        if df is not None and "tortilla:id" in getattr(df, "columns", []):
            for pos, native in enumerate(df["tortilla:id"].tolist()):
                self._by_native[str(native)] = pos

    def get(self, native_id: str, image_id: str) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return (rgb_uint8, gt_mask) for an artifact, or (None, None) if unavailable."""
        if not self._ok:
            return None, None
        pos = self._by_native.get(str(native_id)) if native_id else None
        if pos is None:
            try:
                pos = int(image_id)  # positional fallback
            except (TypeError, ValueError):
                return None, None
        if not (0 <= pos < len(self._ds)):
            return None, None
        sample = self._ds[pos]
        image = np.asarray(sample["image"])  # (C, H, W)
        mask = np.asarray(sample["mask"])
        if mask.ndim == 3:
            mask = mask[0]
        rgb = image[self._rgb_indices] if image.shape[0] >= len(self._rgb_indices) else image[:3]
        rgb = _denorm_image(np.transpose(rgb, (1, 2, 0)))
        return rgb, mask

    @property
    def num_classes(self) -> int:
        return self._num_classes


def _panel_source(ax, rgb, title):
    if rgb is None:
        ax.text(0.5, 0.5, "source\nunavailable", ha="center", va="center", fontsize=8)
        ax.set_facecolor("#F1F3F5")
    else:
        ax.imshow(rgb)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def _panel_mask(ax, mask, num_classes, title):
    if mask is None:
        ax.text(0.5, 0.5, "n/a", ha="center", va="center", fontsize=8)
        ax.set_facecolor("#F1F3F5")
    else:
        ax.imshow(colorize_mask(np.asarray(mask).astype(int), max(num_classes, 1)))
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def _panel_heat(ax, pixel_scores, title):
    im = ax.imshow(np.asarray(pixel_scores), cmap="magma")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    return im


def render_gallery(
    root: str,
    model_slug: str,
    dataset: str,
    method: str,
    out_dir: str,
    *,
    n_suspect: int,
    n_control: int,
    degenerate: bool = False,
) -> str | None:
    """Render one (model, dataset, method) suspect-audit grid; return its path.

    ``degenerate`` stamps a banner: a "most suspect" gallery drawn from a
    collapsed predictor is showing rank noise, not label problems.
    """
    artifacts = _load_artifacts(_artifact_glob(root, dataset, model_slug, method))
    if not artifacts:
        return None

    # The npz set = top-N suspect (rank 1..N) + a random clean control. Split by
    # rank: the suspects are the lowest ranks, the control the rest.
    suspects = [a for a in artifacts if int(a.get("rank", 1 << 30)) <= n_suspect][:n_suspect]
    if not suspects:  # no rank metadata → just take the most-suspect head
        suspects = artifacts[:n_suspect]
    suspect_ids = {str(a.get("image_id", "")) for a in suspects}
    control = [a for a in artifacts if str(a.get("image_id", "")) not in suspect_ids][:n_control]

    sources = _SourceImages(dataset)
    ncls = sources.num_classes
    name_of = class_names(dataset, max(ncls, 1))

    control_ids = {str(a.get("image_id", "")) for a in control}
    rows = suspects + control
    n_rows = len(rows)
    present_labels: set[int] = set()
    fig, axes = plt.subplots(n_rows, 4, figsize=(12, 2.7 * max(n_rows, 1)), squeeze=False)

    for r, art in enumerate(rows):
        native = str(art.get("native_id", ""))
        image_id = str(art.get("image_id", ""))
        rank = int(art.get("rank", -1))
        score = float(art.get("image_score", float("nan")))
        rgb, gt = sources.get(native, image_id)
        is_control = image_id in control_ids
        tag = "CLEAN" if is_control else f"rank {rank}"
        label = f"{tag}  score={score:.3f}\n{native or image_id}"

        _panel_source(axes[r][0], rgb, label if r == 0 else (tag + f"  score={score:.3f}"))
        _panel_mask(axes[r][1], gt, ncls, "GT" if r == 0 else "")
        _panel_mask(axes[r][2], art["pred_mask"], ncls, "pred" if r == 0 else "")

        # Collect the labels actually drawn, so the legend lists only those
        # (flair2 has 13 classes but any single gallery shows far fewer).
        for m in (gt, art.get("pred_mask")):
            if m is not None:
                present_labels.update(int(v) for v in np.unique(np.asarray(m)))
        im = _panel_heat(axes[r][3], art["pixel_scores"], "pixel score" if r == 0 else "")
        fig.colorbar(im, ax=axes[r][3], fraction=0.046, pad=0.04)

        # A faint separator so the clean-control strip reads as a distinct block.
        if is_control:
            for ax in axes[r]:
                ax.set_facecolor("#F8F0FC")

    n_susp = len(suspects)
    title = (
        f"{model_slug} / {dataset} / {method} — "
        f"{n_susp} most suspect + {len(control)} clean control"
    )
    if degenerate:
        fig.suptitle(
            f"{title}\nDEGENERATE — ranking is noise (OOF members collapsed to the majority class)",
            fontsize=12, color="#C92A2A", fontweight="bold",
        )
    else:
        fig.suptitle(title, fontsize=12)

    # Class legend for the GT/pred panels. Swatch colors come from the same
    # colormap colorize_mask uses, so legend and masks cannot drift apart.
    # 255 (ignore) renders white and is labelled as such rather than as a class.
    handles = []
    ignore_seen = 255 in present_labels
    for i in sorted(v for v in present_labels if v != 255):
        if not 0 <= i < max(ncls, 1):
            continue
        swatch = colorize_mask(np.array([[i]]), max(ncls, 1))[0, 0] / 255.0
        handles.append(mpatches.Patch(facecolor=swatch, edgecolor="#666", label=f"{i}: {name_of[i]}"))
    if ignore_seen:
        handles.append(mpatches.Patch(facecolor="white", edgecolor="#666", label="255: ignore"))
    if handles:
        fig.legend(
            handles=handles,
            loc="lower center",
            ncol=min(len(handles), 5),
            frameon=False,
            fontsize=9,
            title="mask classes (GT / pred)",
            title_fontsize=9,
            bbox_to_anchor=(0.5, -0.004),
        )

    # Leave room at the bottom for the legend (more when it wraps to 2+ rows).
    legend_rows = (len(handles) + 4) // 5 if handles else 0
    bottom = min(0.06, 0.012 * legend_rows * (12 / max(n_rows, 1)))
    fig.tight_layout(rect=(0, bottom, 1, 0.98))

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{model_slug}_{dataset}_{method}_gallery.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def _degenerate_cells(csv_path: str) -> set[tuple[str, str, str]]:
    """``(model_slug, dataset, method)`` triples flagged degenerate in the results CSV.

    Keyed by the sanitized slug because that is what the npz directory layout
    (and therefore ``_discover``) uses, while the CSV stores the raw model name.
    A missing CSV or column yields an empty set: no banner, never a false one.
    """
    if not os.path.exists(csv_path):
        return set()
    import pandas as pd

    from torchgeo_bench.label_quality.store import sanitize_slug

    df = pd.read_csv(csv_path)
    if "degenerate" not in df.columns:
        return set()
    flagged = df[df["degenerate"].fillna(False).astype(bool)]
    return {
        (sanitize_slug(m), str(d), str(me))
        for m, d, me in zip(flagged["model"], flagged["dataset"], flagged["method"])
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--root", default="results", help="root holding label_quality/<dataset>/...")
    ap.add_argument("--out", default="experiments/label_quality_figures")
    ap.add_argument("--model", default=None, help="restrict to one model slug")
    ap.add_argument("--dataset", default=None, help="restrict to one dataset")
    ap.add_argument("--method", default=None, help="restrict to one method (cleanlab/aer)")
    ap.add_argument("--n-suspect", type=int, default=12, help="suspect rows per gallery")
    ap.add_argument("--n-control", type=int, default=6, help="clean-control rows per gallery")
    ap.add_argument(
        "--csv",
        default="results/label_quality_v3/label_quality_results.csv",
        help="results CSV supplying the degenerate flag (banner only; missing = no banner)",
    )
    args = ap.parse_args()

    degenerate_cells = _degenerate_cells(args.csv)

    triples = _discover(args.root)
    if args.model:
        triples = [t for t in triples if t[1] == args.model]
    if args.dataset:
        triples = [t for t in triples if t[0] == args.dataset]
    if args.method:
        triples = [t for t in triples if t[2] == args.method]

    if not triples:
        raise SystemExit(
            f"No label-quality npz artifacts under {args.root}/label_quality/. "
            "Run a fixed sweep first (it persists top-N suspect + control npz)."
        )

    for dataset, model_slug, method in triples:
        path = render_gallery(
            args.root, model_slug, dataset, method, args.out,
            n_suspect=args.n_suspect, n_control=args.n_control,
            degenerate=(model_slug, dataset, method) in degenerate_cells,
        )
        if path:
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
