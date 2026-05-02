#!/usr/bin/env python3
"""
scripts/analyze_confusions.py
=============================

Post-hoc confusion analysis across all three trained architectures.

Reads `outputs/eval/<arch>/metrics.json` (produced by `src/evaluate.py`)
and writes:

    outputs/eval/<arch>/per_class_accuracy.png   -- sorted per-class bar chart
    outputs/eval/<arch>/top_confusions.csv       -- ranked (true -> pred) pairs
    outputs/eval/<arch>/top_confusions.txt       -- same, human-readable
    outputs/eval/confusion_matrices_3up.png      -- side-by-side comparison

Architectures with missing metrics.json are skipped with a warning -- so
you can re-run this any time, even mid-evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


ARCHITECTURES: List[str] = ["mobilenet_v2", "efficientnet_b0", "resnet50"]

ARCH_DISPLAY_NAMES: Dict[str, str] = {
    "mobilenet_v2": "MobileNetV2",
    "efficientnet_b0": "EfficientNet-B0",
    "resnet50": "ResNet50",
}


def load_metrics(eval_dir: Path, arch: str) -> Optional[Dict]:
    path = eval_dir / arch / "metrics.json"
    if not path.exists():
        print(f"[skip] {path} not found -- run `python -m src.evaluate --arch {arch} ...` first.")
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def plot_per_class_accuracy(
    metrics: Dict,
    arch_display: str,
    output_path: Path,
) -> None:
    """Sorted horizontal bar chart of per-class accuracy.

    Sorting ascending puts the worst classes at the bottom -- the eye is
    drawn to them first, which is what we want for an error-analysis plot.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_class: Dict[str, float] = metrics["per_class_acc"]
    items = sorted(per_class.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    accs = [v * 100 for _, v in items]

    fig, ax = plt.subplots(figsize=(8, max(6, 0.25 * len(names))), dpi=150)
    bars = ax.barh(range(len(names)), accs, color="#1f77b4")

    # Color the worst-performing classes red so they stand out.
    threshold = 80.0
    for bar, acc in zip(bars, accs):
        if acc < threshold:
            bar.set_color("#d62728")

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Accuracy (%)")
    ax.set_xlim(0, 100)
    ax.set_title(f"{arch_display} - per-class test accuracy (sorted)")
    ax.axvline(threshold, color="gray", linestyle="--", alpha=0.5,
               label=f"{int(threshold)}% threshold")
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def extract_top_confusions(
    metrics: Dict,
    top_n: int = 15,
) -> List[Tuple[str, str, int, float]]:
    """Return the top-N (true_class, pred_class, count, fraction) confusions.

    Diagonal entries are excluded. `fraction` is count / row_total (i.e.
    "of all true X images, what fraction were predicted Y").
    """
    confusion = np.array(metrics["confusion_matrix"], dtype=np.int64)
    class_names: List[str] = metrics["class_names"]
    row_sums = confusion.sum(axis=1)

    pairs: List[Tuple[str, str, int, float]] = []
    n = confusion.shape[0]
    for i in range(n):
        for j in range(n):
            if i == j or confusion[i, j] == 0:
                continue
            count = int(confusion[i, j])
            frac = count / row_sums[i] if row_sums[i] else 0.0
            pairs.append((class_names[i], class_names[j], count, frac))

    pairs.sort(key=lambda t: t[2], reverse=True)
    return pairs[:top_n]


def write_top_confusions(
    pairs: List[Tuple[str, str, int, float]],
    csv_path: Path,
    txt_path: Path,
    arch_display: str,
) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["true_class", "predicted_class", "count", "fraction_of_true"])
        for true_c, pred_c, count, frac in pairs:
            writer.writerow([true_c, pred_c, count, f"{frac:.4f}"])

    lines = [f"Top confusions for {arch_display}", "=" * (22 + len(arch_display)), ""]
    lines.append(f"{'true':<25} -> {'predicted':<25}  count  frac")
    lines.append("-" * 70)
    for true_c, pred_c, count, frac in pairs:
        lines.append(f"{true_c:<25} -> {pred_c:<25}  {count:>4}   {frac*100:5.1f}%")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_confusion_matrices_3up(
    metrics_by_arch: Dict[str, Dict],
    output_path: Path,
) -> bool:
    """Single figure with all three confusion matrices side-by-side.

    Each matrix is row-normalized (so colors are comparable across
    architectures regardless of per-class test set sizes). Class labels
    are shown on the leftmost subplot only to keep the figure compact.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    archs = [a for a in ARCHITECTURES if a in metrics_by_arch]
    if not archs:
        return False

    fig, axes = plt.subplots(1, len(archs), figsize=(7 * len(archs), 7), dpi=150)
    if len(archs) == 1:
        axes = [axes]

    for ax, arch in zip(axes, archs):
        m = metrics_by_arch[arch]
        confusion = np.array(m["confusion_matrix"], dtype=np.float64)
        class_names = m["class_names"]

        row_sums = confusion.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        normalized = confusion / row_sums

        im = ax.imshow(normalized, cmap="Blues", aspect="equal", vmin=0, vmax=1)
        title = f"{ARCH_DISPLAY_NAMES.get(arch, arch)}\ntop-1: {m['top1']*100:.2f}%"
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=90, fontsize=6)

        if ax is axes[0]:
            ax.set_ylabel("True")
            ax.set_yticks(range(len(class_names)))
            ax.set_yticklabels(class_names, fontsize=6)
        else:
            ax.set_yticks([])

    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02,
                 label="Row-normalized fraction")
    fig.suptitle("Confusion matrices across architectures (row-normalized)",
                 fontsize=13, y=1.02)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return True


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate per-class accuracy, top confusions, and 3-up CM figure.",
    )
    p.add_argument("--eval-dir", default="outputs/eval",
                   help="Directory containing <arch>/metrics.json files.")
    p.add_argument("--top-n", type=int, default=15,
                   help="How many confusion pairs to list per architecture.")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    eval_dir = Path(args.eval_dir)

    metrics_by_arch: Dict[str, Dict] = {}

    for arch in ARCHITECTURES:
        m = load_metrics(eval_dir, arch)
        if m is None:
            continue
        metrics_by_arch[arch] = m
        display = ARCH_DISPLAY_NAMES[arch]
        arch_dir = eval_dir / arch

        plot_per_class_accuracy(m, display, arch_dir / "per_class_accuracy.png")
        print(f"[ok]   wrote {arch_dir / 'per_class_accuracy.png'}")

        pairs = extract_top_confusions(m, top_n=args.top_n)
        write_top_confusions(
            pairs,
            csv_path=arch_dir / "top_confusions.csv",
            txt_path=arch_dir / "top_confusions.txt",
            arch_display=display,
        )
        print(f"[ok]   wrote {arch_dir / 'top_confusions.csv'} and .txt")

    if metrics_by_arch:
        out_3up = eval_dir / "confusion_matrices_3up.png"
        if plot_confusion_matrices_3up(metrics_by_arch, out_3up):
            print(f"[ok]   wrote {out_3up}")
    else:
        print("No metrics.json files found -- run `src.evaluate` first.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
