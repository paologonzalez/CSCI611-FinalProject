"""
src/evaluate.py
===============

Test-set evaluation for a trained model.

Why a separate module from `train.py`?
    `train.py:validate` reports just loss/top-1 -- enough to drive Optuna's
    early stopping. The final report (and the slides) want richer metrics:
    top-5 accuracy, per-class accuracy, and a confusion matrix to spot
    systematic confusions between food classes.

What this module produces
-------------------------
A single dict with:
    {
      "top1": 0.812,
      "top5": 0.957,
      "loss": 0.534,
      "per_class_acc": { "Baked Potato": 0.91, ... },
      "confusion_matrix": <NxN numpy array>,  # rows=true, cols=predicted
      "class_names": [...],                   # row/column labels
    }

Plus, when run as a script, it prints a summary table and (optionally)
saves the confusion matrix as a PNG.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data_prep import get_dataloaders
from src.models import build_model


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    class_names: List[str],
    device: torch.device | None = None,
) -> Dict[str, Any]:
    """Run `model` over `loader` and compute evaluation metrics.

    Returns the dict described in the module docstring. The model is left
    on its current device (caller's choice); we just `.to(device)` if a
    device is explicitly passed.
    """
    if device is None:
        device = next(model.parameters()).device

    model.to(device)
    model.eval()

    num_classes = len(class_names)
    criterion = nn.CrossEntropyLoss()

    # Aggregators. We collect raw counts and finalize at the end so
    # floating-point ordering doesn't introduce drift on big datasets.
    total_seen = 0
    total_loss = 0.0
    top1_correct = 0
    top5_correct = 0

    # NxN confusion matrix. Integer counts, indexed [true_label][pred_label].
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    for images, labels in tqdm(loader, desc="evaluate", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_seen += batch_size

        # top-k correctness
        # `topk(5)` returns the indices of the 5 highest-scoring classes.
        # If the true label is in any of those, it counts for top-5.
        # `topk` works even when num_classes < 5 by clamping.
        k = min(5, num_classes)
        _, topk_idx = logits.topk(k, dim=1)            # shape (B, k)
        match = topk_idx.eq(labels.unsqueeze(1))       # shape (B, k), bool
        top1_correct += match[:, 0].sum().item()
        top5_correct += match.any(dim=1).sum().item()

        # Confusion matrix update -- vectorized add-at-index.
        preds = logits.argmax(dim=1)
        np.add.at(
            confusion,
            (labels.detach().cpu().numpy(), preds.detach().cpu().numpy()),
            1,
        )

    # Per-class accuracy = diagonal / row sum. Guard divide-by-zero for any
    # class that happened to have zero test samples (shouldn't happen with
    # our split logic, but defensive).
    row_sums = confusion.sum(axis=1)
    safe_rows = np.where(row_sums == 0, 1, row_sums)
    per_class_acc_arr = confusion.diagonal() / safe_rows

    return {
        "top1": top1_correct / max(total_seen, 1),
        "top5": top5_correct / max(total_seen, 1),
        "loss": total_loss / max(total_seen, 1),
        "per_class_acc": {
            name: float(per_class_acc_arr[idx])
            for idx, name in enumerate(class_names)
        },
        "confusion_matrix": confusion,
        "class_names": class_names,
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def save_confusion_png(
    confusion: np.ndarray,
    class_names: List[str],
    output_path: str | Path,
    normalize: bool = True,
) -> None:
    """Write the confusion matrix to a PNG using matplotlib.

    With 34 classes the figure is large; we make it 12x12 inches at 150
    DPI which is enough to read class names without zooming. Set
    `normalize=True` (the default) to display row-normalized fractions
    rather than raw counts -- much easier to read across imbalanced
    classes.
    """
    # Lazy import keeps the test/eval CLI usable without matplotlib in
    # environments where someone only wants the metric numbers.
    import matplotlib
    matplotlib.use("Agg")  # headless: no display server required.
    import matplotlib.pyplot as plt

    matrix = confusion.astype(np.float64)
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        matrix = matrix / row_sums

    fig, ax = plt.subplots(figsize=(12, 12), dpi=150)
    im = ax.imshow(matrix, cmap="Blues", aspect="equal")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix" + (" (row-normalized)" if normalize else ""))
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a trained checkpoint on the test split.")
    p.add_argument("--arch", required=True, choices=["resnet50", "efficientnet_b0", "mobilenet_v2"])
    p.add_argument("--checkpoint", required=True, help="Path to a .pt file saved by train.py.")
    p.add_argument("--manifest", default="configs/data_split.yaml")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output-dir", default="outputs/eval", help="Where to write metrics + plots.")
    p.add_argument("--no-plot", action="store_true", help="Skip the confusion-matrix PNG.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data first so we know how many classes the model needs.
    manifest, _, _, test_loader = get_dataloaders(
        manifest_path=args.manifest,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Build the architecture skeleton, then load the trained weights.
    built = build_model(args.arch, num_classes=manifest.num_classes, pretrained=False)
    state = torch.load(args.checkpoint, map_location=device)
    built.model.load_state_dict(state["model_state_dict"])

    # Class names ordered by class_idx so confusion-matrix axes line up.
    class_names = [
        manifest.idx_to_class[i] for i in range(manifest.num_classes)
    ]

    metrics = evaluate_model(
        model=built.model,
        loader=test_loader,
        class_names=class_names,
        device=device,
    )

    # Pretty-print summary.
    print(f"=== Evaluation: {args.arch} ===")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  test loss : {metrics['loss']:.4f}")
    print(f"  top-1 acc : {metrics['top1']:.4f}")
    print(f"  top-5 acc : {metrics['top5']:.4f}")

    # Persist metrics JSON + confusion matrix PNG to output_dir.
    out_dir = Path(args.output_dir) / args.arch
    out_dir.mkdir(parents=True, exist_ok=True)

    # Numpy arrays don't serialize as JSON; convert to nested lists.
    serializable = {
        "top1": metrics["top1"],
        "top5": metrics["top5"],
        "loss": metrics["loss"],
        "per_class_acc": metrics["per_class_acc"],
        "confusion_matrix": metrics["confusion_matrix"].tolist(),
        "class_names": metrics["class_names"],
    }
    (out_dir / "metrics.json").write_text(json.dumps(serializable, indent=2))

    if not args.no_plot:
        save_confusion_png(
            metrics["confusion_matrix"],
            class_names=class_names,
            output_path=out_dir / "confusion_matrix.png",
        )
        print(f"  saved confusion matrix -> {out_dir / 'confusion_matrix.png'}")

    print(f"  wrote metrics.json -> {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
