#!/usr/bin/env python3
"""
scripts/gradcam_compare.py
==========================

Run Grad-CAM on the *same* image across all three best-trained
architectures and produce a single side-by-side figure that shows:

  [ original | MobileNetV2 | EfficientNet-B0 | ResNet50 ]

Each per-arch panel is titled with the model's top-1 prediction and
softmax confidence. If the image's parent directory name matches one of
the dataset classes (which is the case for any image pulled directly out
of `dataset/<Class>/...`), the title is colored green for a correct
prediction and red for a wrong one.

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
) -> Tuple[np.ndarray, List[Tuple[int, float]]]:
    """Load `arch`, run a forward pass + Grad-CAM, return:
        (heatmap, top_k_predictions)
    where `top_k_predictions` is a list of (class_idx, probability) pairs
    sorted by probability descending. The first entry is top-1.

    We do one `torch.no_grad()` forward to read softmax probabilities
    (Grad-CAM by itself only returns the argmax). Then we ask
    `compute_gradcam` to attribute the *top-1* class so the heatmap and
    the headline prediction agree.
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

    # Pass 2: Grad-CAM attributing the top-1 predicted class.
    pred_idx = top_predictions[0][0]
    heatmap, _ = compute_gradcam(
        model=built.model,
        image_tensor=image_tensor,
        target_layer_name=built.gradcam_target_layer,
        target_class=pred_idx,
    )
    return heatmap, top_predictions


# ---------------------------------------------------------------------------
# Figure assembly
# ---------------------------------------------------------------------------

def build_figure(
    image_tensor: torch.Tensor,
    per_arch: Dict[str, Tuple[np.ndarray, List[Tuple[int, float]]]],
    class_names: List[str],
    true_label: Optional[str],
    output_path: Path,
) -> None:
    """Compose a 1-row figure: original on the left, then one Grad-CAM
    overlay per architecture, each titled with its top-3 predictions.
    """
    import matplotlib
    matplotlib.use("Agg")  # headless: no display server required.
    import matplotlib.pyplot as plt

    # We display the *cropped, denormalized* tensor (224x224 view of the
    # input) so the heatmap aligns to what the model actually saw, not
    # to the larger original.
    display_image = denormalize_for_display(image_tensor[0])

    n_cols = 1 + len(per_arch)
    # Figure is a touch taller than before to make room for the 3-line
    # ranked prediction title above each overlay panel.
    fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 5.2), dpi=150)

    # Column 0: input image.
    axes[0].imshow(display_image)
    input_title = "Input"
    if true_label is not None:
        input_title += f"\nTrue: {true_label}"
    axes[0].set_title(input_title, fontsize=11)
    axes[0].axis("off")

    # Columns 1..N: per-arch overlays.
    for ax, arch in zip(axes[1:], ARCHITECTURES):
        if arch not in per_arch:
            ax.axis("off")
            ax.set_title(f"{ARCH_DISPLAY_NAMES[arch]}\n(no checkpoint)",
                         fontsize=10, color="gray")
            continue

        heatmap, top_predictions = per_arch[arch]
        overlay = overlay_heatmap_on_image(display_image, heatmap)
        ax.imshow(overlay)

        # Build a ranked, monospace-friendly title showing top-1..top-k.
        # Using monospace lets the percentages line up across rows even
        # though class names vary in length.
        title_lines = [ARCH_DISPLAY_NAMES[arch]]
        for rank, (idx, prob) in enumerate(top_predictions, start=1):
            title_lines.append(
                f"{rank}. {class_names[idx]:<14} {prob * 100:5.1f}%"
            )
        title = "\n".join(title_lines)

        # Color by correctness of the *top-1* prediction. We don't try to
        # color individual lines because matplotlib titles are one color
        # per call -- the rank-1 line is what the audience reads first.
        pred_name = class_names[top_predictions[0][0]]
        if true_label is not None:
            color = "green" if pred_name == true_label else "red"
            ax.set_title(title, fontsize=9, color=color, fontfamily="monospace")
        else:
            ax.set_title(title, fontsize=9, fontfamily="monospace")
        ax.axis("off")

    fig.suptitle(
        "Grad-CAM comparison across architectures",
        fontsize=13, y=1.02,
    )
    fig.tight_layout()
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

    per_arch: Dict[str, Tuple[np.ndarray, List[Tuple[int, float]]]] = {}
    for arch in ARCHITECTURES:
        try:
            ckpt = find_best_checkpoint(arch, output_dir)
        except FileNotFoundError as e:
            print(f"  [skip] {arch}: {e}")
            continue

        heatmap, top_predictions = predict_and_attribute(
            arch=arch,
            checkpoint_path=ckpt,
            image_tensor=image_tensor,
            num_classes=manifest.num_classes,
            device=device,
        )
        per_arch[arch] = (heatmap, top_predictions)

        # Plain-text summary line for stdout -- shows the same top-3
        # ranking that ends up in the figure title, useful when running
        # over many images from a shell script.
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
