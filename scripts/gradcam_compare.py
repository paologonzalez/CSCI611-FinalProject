#!/usr/bin/env python3
"""
scripts/gradcam_compare.py
==========================

Run Grad-CAM on the *same* image across all three best-trained
architectures and produce a 2-row grid figure:

  Row 0 (Early layer):  [ original | MobileNetV2 | EfficientNet-B0 | ResNet50 ]
  Row 1 (Late layer):   [           | MobileNetV2 | EfficientNet-B0 | ResNet50 ]

The original image spans both rows in column 0. Each heatmap panel has
a caption below it naming the exact layer used (e.g. "layer1 — residual
stage 1"). The top-row uses an early convolutional layer; the bottom row
uses the final convolutional layer (the same target used in the original
single-row figure).

Each per-arch panel is titled with the model's top-1 prediction and
softmax confidence, colored green/red for correct/wrong.

Why a separate script from `src/gradcam.py`?
    `src/gradcam.py` is a clean single-arch tool. The cross-architecture
    comparison is a project-level analysis (like `analyze_confusions.py`
    or `report.py`) so it lives under `scripts/` and just reuses
    `src/gradcam.py`'s public helpers.

Usage
-----
From the repo root, with the venv activated:

    python scripts/gradcam_compare.py --image ../dataset/Taco/<filename>.jpg

The script discovers each architecture's best checkpoint by reading
`outputs/<arch>_best_summary.json` for the best trial number and then
loading `outputs/checkpoints/<arch>/trial_<N>/<arch>_trial<N>_best.pt`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# Allow `from src.X import Y` when this script is run from the repo root.
# scripts/ is a sibling of src/, so we prepend the repo root to sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_prep import get_eval_transforms, load_manifest
from src.gradcam import (
    compute_gradcam,
    denormalize_for_display,
    overlay_heatmap_on_image,
)
from src.models import build_model


# ---------------------------------------------------------------------------
# Project-wide configuration
# ---------------------------------------------------------------------------

# Order chosen to read lightest -> heaviest in the figure, matching the
# size-vs-accuracy story we tell elsewhere in the report.
ARCHITECTURES: List[str] = ["mobilenet_v2", "efficientnet_b0", "resnet50"]

ARCH_DISPLAY_NAMES: Dict[str, str] = {
    "mobilenet_v2": "MobileNetV2",
    "efficientnet_b0": "EfficientNet-B0",
    "resnet50": "ResNet50",
}

# Early-layer target for each arch.  These are the first meaningful
# residual/MBConv stages -- shallow enough to show low-level texture
# responses, giving a visible contrast against the semantic late-layer maps.
EARLY_LAYERS: Dict[str, str] = {
    "resnet50":        "layer1",      # residual stage 1, 56×56 feature maps
    "efficientnet_b0": "features.2",  # MBConv stage 2
    "mobilenet_v2":    "features.4",  # inverted residual block 4
}

# Human-readable captions that appear under each heatmap panel.
EARLY_LAYER_CAPTIONS: Dict[str, str] = {
    "resnet50":        "layer1  (residual stage 1)",
    "efficientnet_b0": "features.2  (MBConv stage 2)",
    "mobilenet_v2":    "features.4  (inverted residual 4)",
}

LATE_LAYER_CAPTIONS: Dict[str, str] = {
    "resnet50":        "layer4  (residual stage 4)",
    "efficientnet_b0": "features.8  (final conv block)",
    "mobilenet_v2":    "features.18  (final expansion conv)",
}


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def find_best_checkpoint(arch: str, output_dir: Path) -> Path:
    """Resolve the best checkpoint path for a given arch.

    Each per-arch training script writes `<arch>_best_summary.json` at
    the end of its sweep, listing the winning trial. The actual weights
    live at the canonical per-trial checkpoint path that `train.py`
    constructed during training.
    """
    summary_path = output_dir / f"{arch}_best_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"No best summary at {summary_path}. "
            f"Has `python -m src.train_{arch}` been run?"
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    trial_n = summary["best_trial_number"]

    ckpt = (
        output_dir / "checkpoints" / arch
        / f"trial_{trial_n}"
        / f"{arch}_trial{trial_n}_best.pt"
    )
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Best summary points to trial {trial_n} but the checkpoint "
            f"is missing: {ckpt}"
        )
    return ckpt


# ---------------------------------------------------------------------------
# Per-architecture inference + Grad-CAM
# ---------------------------------------------------------------------------

def predict_and_attribute(
    arch: str,
    checkpoint_path: Path,
    image_tensor: torch.Tensor,
    num_classes: int,
    device: torch.device,
    top_k: int = 3,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, float]]]:
    """Load `arch`, run Grad-CAM on both the early and late target layers.

    Returns:
        (early_heatmap, late_heatmap, top_k_predictions)

    One `torch.no_grad()` forward extracts softmax probabilities; two
    separate Grad-CAM passes (one per layer) then attribute the top-1
    predicted class so every heatmap corresponds to the same decision.
    """
    built = build_model(arch, num_classes=num_classes, pretrained=False)
    state = torch.load(checkpoint_path, map_location=device)
    built.model.load_state_dict(state["model_state_dict"])
    built.model.to(device).eval()

    # Pass 1: probabilities for the top-k extraction.
    with torch.no_grad():
        logits = built.model(image_tensor)
        probs = F.softmax(logits, dim=1)[0]
        k = min(top_k, num_classes)
        top_probs, top_idx = probs.topk(k)
        top_predictions: List[Tuple[int, float]] = [
            (int(i.item()), float(p.item()))
            for p, i in zip(top_probs, top_idx)
        ]

    pred_idx = top_predictions[0][0]

    # Pass 2a: Grad-CAM on the early convolutional layer.
    early_heatmap, _ = compute_gradcam(
        model=built.model,
        image_tensor=image_tensor,
        target_layer_name=EARLY_LAYERS[arch],
        target_class=pred_idx,
    )

    # Pass 2b: Grad-CAM on the late (final) convolutional layer.
    late_heatmap, _ = compute_gradcam(
        model=built.model,
        image_tensor=image_tensor,
        target_layer_name=built.gradcam_target_layer,
        target_class=pred_idx,
    )

    return early_heatmap, late_heatmap, top_predictions


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------

def build_figure(
    image_tensor: torch.Tensor,
    per_arch: Dict[str, Tuple[np.ndarray, np.ndarray, List[Tuple[int, float]]]],
    class_names: List[str],
    true_label: Optional[str],
    output_path: Path,
) -> None:
    """Compose a 2-row grid figure:

        Row 0 (Early layer):  Original  | Arch-1  | Arch-2  | Arch-3
        Row 1 (Late  layer):  (blank)   | Arch-1  | Arch-2  | Arch-3

    The original image spans both rows in column 0. Each heatmap panel
    has a caption below it (via xlabel) naming the exact target layer.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    display_image = denormalize_for_display(image_tensor[0])

    present_archs = [a for a in ARCHITECTURES if a in per_arch]
    n_arch = len(present_archs)
    n_cols = 1 + n_arch

    fig = plt.figure(figsize=(4 * n_cols, 10), dpi=150)
    gs = gridspec.GridSpec(
        2, n_cols, figure=fig,
        hspace=0.55,   # vertical gap between rows
        wspace=0.15,
    )

    # Column 0: original image, spans both rows.
    ax_orig = fig.add_subplot(gs[:, 0])
    ax_orig.imshow(display_image)
    orig_title = "Input"
    if true_label is not None:
        orig_title += f"\nTrue: {true_label}"
    ax_orig.set_title(orig_title, fontsize=11)
    ax_orig.axis("off")

    row_labels = ["Early layer", "Late layer"]
    row_axes: List = []  # first arch axis per row, for the row-label annotation

    for row, (layer_key, caption_dict) in enumerate(
        [("early", EARLY_LAYER_CAPTIONS), ("late", LATE_LAYER_CAPTIONS)]
    ):
        for col, arch in enumerate(present_archs):
            ax = fig.add_subplot(gs[row, col + 1])
            if col == 0:
                row_axes.append(ax)

            early_hm, late_hm, top_predictions = per_arch[arch]
            heatmap = early_hm if layer_key == "early" else late_hm
            overlay = overlay_heatmap_on_image(display_image, heatmap)
            ax.imshow(overlay)

            # Title: arch name + ranked predictions (top row only, to avoid
            # cluttering the late-row which shares the same predictions).
            if row == 0:
                title_lines = [ARCH_DISPLAY_NAMES[arch]]
                for rank, (idx, prob) in enumerate(top_predictions, start=1):
                    title_lines.append(
                        f"{rank}. {class_names[idx]:<14} {prob * 100:5.1f}%"
                    )
                title = "\n".join(title_lines)
                pred_name = class_names[top_predictions[0][0]]
                if true_label is not None:
                    color = "green" if pred_name == true_label else "red"
                    ax.set_title(title, fontsize=9, color=color,
                                 fontfamily="monospace")
                else:
                    ax.set_title(title, fontsize=9, fontfamily="monospace")

            # Caption below the image naming the exact layer.
            ax.set_xlabel(caption_dict[arch], fontsize=8, labelpad=4)
            ax.xaxis.set_label_position("bottom")
            ax.tick_params(bottom=False, left=False,
                           labelbottom=False, labelleft=False)
            for spine in ax.spines.values():
                spine.set_visible(False)

    # Row annotations: "Early layer" / "Late layer" rotated on the y-axis of
    # the first arch column in each row.  We kept references in row_axes.
    for ax_first, label in zip(row_axes, row_labels):
        ax_first.annotate(
            label,
            xy=(0, 0.5), xycoords="axes fraction",
            xytext=(-48, 0), textcoords="offset points",
            va="center", ha="right",
            fontsize=10, fontweight="bold",
            rotation=90,
        )

    fig.suptitle(
        "Grad-CAM: early vs. late convolutional layer across architectures",
        fontsize=13, y=1.01,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _infer_true_label(image_path: Path, class_names: List[str]) -> Optional[str]:
    """Infer the true label from the image's parent directory.

    Any image pulled straight from `dataset/<Class>/<file>` will have a
    parent directory whose name is exactly the class label, so this is a
    cheap and reliable inference. Returns None for arbitrary paths.
    """
    parent = image_path.parent.name
    return parent if parent in class_names else None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run Grad-CAM on the same image across the three best trained "
            "architectures and write a side-by-side comparison figure."
        ),
    )
    p.add_argument("--image", required=True,
                   help="Path to the image to explain.")
    p.add_argument("--manifest", default="configs/data_split.yaml",
                   help="Split manifest -- used only for class names / count.")
    p.add_argument("--output-dir", default="outputs",
                   help="Where the per-arch summaries / checkpoints live.")
    p.add_argument("--output", default="outputs/gradcam_compare.png",
                   help="Where to write the side-by-side figure.")
    p.add_argument("--true-label", default=None,
                   help=(
                       "Override the inferred true label. If unset we try "
                       "to read it from the image's parent directory name."
                   ))
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)

    manifest = load_manifest(args.manifest)
    class_names = [
        manifest.idx_to_class[i] for i in range(manifest.num_classes)
    ]

    # Preprocess the image *once* with the shared eval transform so every
    # architecture sees the exact same input pixels. Otherwise the
    # comparison would be confounded by per-arch preprocessing.
    pil_image = Image.open(args.image).convert("RGB")
    image_tensor = get_eval_transforms()(pil_image).unsqueeze(0).to(device)

    true_label = args.true_label or _infer_true_label(
        Path(args.image), class_names
    )

    print(f"Image: {args.image}")
    if true_label is not None:
        print(f"True label: {true_label}")

    per_arch: Dict[str, Tuple[np.ndarray, np.ndarray, List[Tuple[int, float]]]] = {}
    for arch in ARCHITECTURES:
        try:
            ckpt = find_best_checkpoint(arch, output_dir)
        except FileNotFoundError as e:
            print(f"  [skip] {arch}: {e}")
            continue

        early_heatmap, late_heatmap, top_predictions = predict_and_attribute(
            arch=arch,
            checkpoint_path=ckpt,
            image_tensor=image_tensor,
            num_classes=manifest.num_classes,
            device=device,
        )
        per_arch[arch] = (early_heatmap, late_heatmap, top_predictions)

        pred_idx = top_predictions[0][0]
        is_correct = (
            true_label is not None and class_names[pred_idx] == true_label
        )
        marker = "ok " if is_correct else ("err" if true_label else "   ")
        top3_str = "  ".join(
            f"{class_names[idx]} {prob * 100:.1f}%"
            for idx, prob in top_predictions
        )
        print(f"  [{marker}] {ARCH_DISPLAY_NAMES[arch]:<16}  {top3_str}")

    if not per_arch:
        print(
            "No architectures were evaluable -- have any sweeps been run?",
            file=sys.stderr,
        )
        return 1

    out_path = Path(args.output)
    build_figure(
        image_tensor=image_tensor,
        per_arch=per_arch,
        class_names=class_names,
        true_label=true_label,
        output_path=out_path,
    )
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
