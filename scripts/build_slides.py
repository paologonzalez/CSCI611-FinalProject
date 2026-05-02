#!/usr/bin/env python3
"""
scripts/build_slides.py
=======================

Generate a starter PowerPoint deck for the CSCI 611 final-project
presentation. Re-run any time the underlying outputs change.

Why a generator (vs. just hand-building the deck):
    - The headline numbers (test top-1 / param counts) can be re-read
      from the per-arch summary JSONs each run, so the slides never
      drift from the actual results.
    - Images (size-vs-accuracy plot, confusion matrices, Grad-CAM) are
      pulled straight from `outputs/`, so any rerun of `report.py` /
      `analyze_confusions.py` flows straight into the deck.

The output is a starter deck -- expect to polish in PowerPoint after.

Usage
-----
From the repo root, with the venv activated:

    python scripts/build_slides.py
    # or with a custom output path:
    python scripts/build_slides.py --output ../slides.pptx
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor


# ---------------------------------------------------------------------------
# Constants -- the deck's "physical" layout
# ---------------------------------------------------------------------------

# 16:9 widescreen slide size (PowerPoint default).
SLIDE_WIDTH = Inches(13.333)
SLIDE_HEIGHT = Inches(7.5)

# Default text margins.
LEFT_MARGIN = Inches(0.6)
RIGHT_MARGIN = Inches(0.6)
TOP_MARGIN_BODY = Inches(1.6)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_text(tf, text: str, size: int = 18, bold: bool = False,
              color: Optional[RGBColor] = None) -> None:
    tf.text = text
    p = tf.paragraphs[0]
    for run in p.runs:
        run.font.size = Pt(size)
        run.font.bold = bold
        if color is not None:
            run.font.color.rgb = color


def _add_title(slide, text: str) -> None:
    """Set the slide's built-in title placeholder."""
    if slide.shapes.title is not None:
        title_tf = slide.shapes.title.text_frame
        title_tf.text = text
        for run in title_tf.paragraphs[0].runs:
            run.font.size = Pt(32)
            run.font.bold = True


def _add_bullets(slide, items: List[str], left: Inches = LEFT_MARGIN,
                 top: Inches = TOP_MARGIN_BODY,
                 width: Optional[Inches] = None,
                 height: Optional[Inches] = None,
                 base_size: int = 18) -> None:
    """Add a single text box with one bullet per item.

    Items can be plain strings or "  - subbullet" with leading whitespace
    to indicate indentation (multiples of 2 spaces => one indent level).
    """
    if width is None:
        width = SLIDE_WIDTH - left - RIGHT_MARGIN
    if height is None:
        height = SLIDE_HEIGHT - top - Inches(0.4)

    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True

    for i, item in enumerate(items):
        # Determine indent level by counting leading spaces (2-space units).
        stripped = item.lstrip(" ")
        indent = (len(item) - len(stripped)) // 2
        # Strip leading dash/bullet characters the user might have included.
        if stripped.startswith("- "):
            stripped = stripped[2:]

        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()

        p.level = indent
        p.text = stripped
        for run in p.runs:
            # Slightly smaller font for deeper indents.
            run.font.size = Pt(max(base_size - 2 * indent, 12))


def _add_image(slide, image_path: Path, left: Inches, top: Inches,
               width: Optional[Inches] = None,
               height: Optional[Inches] = None) -> Optional[object]:
    """Insert an image if it exists; otherwise insert a placeholder note."""
    if not image_path.exists():
        tb = slide.shapes.add_textbox(left, top,
                                      width or Inches(4),
                                      height or Inches(0.6))
        _set_text(tb.text_frame,
                  f"[image missing: {image_path.name}]",
                  size=14, color=RGBColor(0xC0, 0x40, 0x40))
        return None

    kwargs = {}
    if width is not None:
        kwargs["width"] = width
    if height is not None:
        kwargs["height"] = height
    return slide.shapes.add_picture(str(image_path), left, top, **kwargs)


def _add_subtitle_box(slide, text: str, top: Inches = Inches(0.95),
                      size: int = 16, italic: bool = True) -> None:
    """Add an italic subtitle line under the slide title."""
    tb = slide.shapes.add_textbox(
        LEFT_MARGIN, top, SLIDE_WIDTH - LEFT_MARGIN - RIGHT_MARGIN, Inches(0.5)
    )
    tf = tb.text_frame
    tf.text = text
    for run in tf.paragraphs[0].runs:
        run.font.size = Pt(size)
        run.font.italic = italic
        run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)


def _add_code_box(slide, code: str, left: Inches, top: Inches,
                  width: Inches, height: Inches, size: int = 12) -> None:
    """Add a monospace code text box."""
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    lines = code.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = line
        for run in p.runs:
            run.font.name = "Consolas"
            run.font.size = Pt(size)
            run.font.color.rgb = RGBColor(0x20, 0x20, 0x20)


# ---------------------------------------------------------------------------
# Live data: pull headline numbers straight from per-arch summaries
# ---------------------------------------------------------------------------

@dataclass
class ArchHeadline:
    arch: str
    display: str
    params: int
    val_acc: float
    test_top1: Optional[float]
    test_top5: Optional[float]


def load_headlines(output_dir: Path) -> List[ArchHeadline]:
    """Read each arch's summary + test metrics so the slides cite the
    actual current numbers, not whatever was true when the deck was built.
    """
    archs = [
        ("mobilenet_v2", "MobileNetV2"),
        ("efficientnet_b0", "EfficientNet-B0"),
        ("resnet50", "ResNet50"),
    ]

    out: List[ArchHeadline] = []
    for arch, display in archs:
        summary_path = output_dir / f"{arch}_best_summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

        # Param count is in the trial's checkpoint metadata file.
        trial_n = summary["best_trial_number"]
        meta_path = (
            output_dir / "checkpoints" / arch
            / f"trial_{trial_n}" / f"{arch}_trial{trial_n}_best.json"
        )
        params = 0
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            params = int(meta.get("extra_metadata", {}).get("param_count", 0))

        # Test metrics from src/evaluate.py output.
        eval_path = output_dir / "eval" / arch / "metrics.json"
        test_top1 = test_top5 = None
        if eval_path.exists():
            em = json.loads(eval_path.read_text(encoding="utf-8"))
            test_top1 = float(em["top1"])
            test_top5 = float(em["top5"])

        out.append(ArchHeadline(
            arch=arch,
            display=display,
            params=params,
            val_acc=float(summary["best_val_acc"]),
            test_top1=test_top1,
            test_top5=test_top5,
        ))
    return out


def _fmt_params(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M"
    if n >= 1_000:
        return f"{n / 1e3:.1f}k"
    return str(n)


def _fmt_pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x * 100:.2f}%"


# ---------------------------------------------------------------------------
# Slide builders
# ---------------------------------------------------------------------------

def build_title(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank layout

    # Centered title.
    tb = slide.shapes.add_textbox(
        Inches(1), Inches(2.2), SLIDE_WIDTH - Inches(2), Inches(1.5)
    )
    tf = tb.text_frame
    tf.text = "Hyperparameter Optimization for Image Classification"
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    for run in p.runs:
        run.font.size = Pt(40)
        run.font.bold = True

    sub = slide.shapes.add_textbox(
        Inches(1), Inches(3.7), SLIDE_WIDTH - Inches(2), Inches(0.6)
    )
    sub_tf = sub.text_frame
    sub_tf.text = "A reusable methodology for picking the right network"
    sp = sub_tf.paragraphs[0]
    sp.alignment = PP_ALIGN.CENTER
    for run in sp.runs:
        run.font.size = Pt(22)
        run.font.italic = True
        run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    # Course line at bottom.
    foot = slide.shapes.add_textbox(
        Inches(1), Inches(5.6), SLIDE_WIDTH - Inches(2), Inches(0.5)
    )
    foot_tf = foot.text_frame
    foot_tf.text = "CSCI 611  -  California State University, Chico  -  Spring 2026"
    fp = foot_tf.paragraphs[0]
    fp.alignment = PP_ALIGN.CENTER
    for run in fp.runs:
        run.font.size = Pt(16)
        run.font.color.rgb = RGBColor(0x80, 0x80, 0x80)


def build_team(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    tb = slide.shapes.add_textbox(
        Inches(1), Inches(2.8), SLIDE_WIDTH - Inches(2), Inches(1.0)
    )
    tf = tb.text_frame
    tf.text = "Team"
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    for run in p.runs:
        run.font.size = Pt(36)
        run.font.bold = True

    names = slide.shapes.add_textbox(
        Inches(1), Inches(3.9), SLIDE_WIDTH - Inches(2), Inches(0.8)
    )
    n_tf = names.text_frame
    n_tf.text = "Jacob Celniker  -  Anand  -  Paolo"
    np_ = n_tf.paragraphs[0]
    np_.alignment = PP_ALIGN.CENTER
    for run in np_.runs:
        run.font.size = Pt(24)


def build_context(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[5])  # title only
    _add_title(slide, "Context & Relevant Work")
    _add_subtitle_box(slide,
                      "The question isn't which model -- it's how to pick one.")

    bullets = [
        "Modern CV offers dozens of pretrained CNNs; choice is usually ad-hoc.",
        "Real cost matters: parameters, latency, memory, deploy target. No universal winner.",
        "Built on three standard pieces of work, stitched together:",
        "  Transfer learning from ImageNet (Donahue '14, Yosinski '14) -- 24k images is enough.",
        "  Three architecture families: ResNet (He '16), EfficientNet (Tan & Le '19), MobileNetV2 (Sandler '18).",
        "  Optuna for HPO (Akiba '19) -- TPE sampling + median pruning makes the comparison fair.",
        "  Grad-CAM (Selvaraju '17) -- interpretability, not just numbers.",
        "Food classification is the test bed. The methodology is the contribution.",
    ]
    _add_bullets(slide, bullets, top=Inches(1.7))


def build_framework(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    _add_title(slide, "High-Level Framework")
    _add_subtitle_box(slide, "Three architectures, one harness, one report.")

    bullets = [
        "Apples-to-apples by construction. A single YAML manifest pins every image's split (seed=42).",
        "Per-arch entry points share the training scaffold: same loader, same run_training, same eval.",
        "Per-arch knobs reflect domain wisdom, not brute force:",
        "  ResNet50  ->  label_smoothing  (high capacity -> regularize)",
        "  EfficientNet-B0  ->  cosine_lr_schedule  (its original recipe)",
        "  MobileNetV2  ->  freeze_backbone  (small model -> maybe just train the head)",
        "Grad-CAM travels with the model -- each architecture exposes its own target conv layer.",
        "One command produces the report: scripts/report.py -> markdown + cross-arch plots.",
    ]
    _add_bullets(slide, bullets, top=Inches(1.7))


def build_implementation_1(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    _add_title(slide, "Implementation: Search Space + Pruning")

    bullets = [
        "Common search space: lr (log), weight_decay (log), dropout, batch_size, optimizer.",
        "TPE sampler beats random search after ~10 trials.",
        "MedianPruner kills weak trials early -- 28 of 76 trials pruned (~37% compute saved).",
    ]
    _add_bullets(slide, bullets, top=Inches(1.4),
                 height=Inches(1.7), base_size=18)

    code = (
        "# src/tune_optuna.py\n"
        "def sample_common_hparams(trial, space):\n"
        "    return {\n"
        '        "lr":           trial.suggest_float("lr", 1e-5, 1e-2, log=True),\n'
        '        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),\n'
        '        "dropout":      trial.suggest_float("dropout", 0.0, 0.5),\n'
        '        "batch_size":   trial.suggest_categorical("batch_size", [16, 32, 64]),\n'
        '        "optimizer":    trial.suggest_categorical("optimizer", ["adamw", "sgd"]),\n'
        "    }\n"
    )
    _add_code_box(slide, code,
                  left=Inches(0.6), top=Inches(3.4),
                  width=SLIDE_WIDTH - Inches(1.2), height=Inches(3.5),
                  size=14)


def build_implementation_2(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    _add_title(slide, "Implementation: Pruning hook + Grad-CAM")

    bullets = [
        "Each epoch reports val_acc to Optuna; pruner can stop a bad trial mid-run.",
        "Best-by-val-acc checkpoint saved on every improvement -- a crash never costs the best weights.",
        "Grad-CAM comparison runs all three best models on the SAME image; reports top-3 per model.",
    ]
    _add_bullets(slide, bullets, top=Inches(1.4),
                 height=Inches(1.7), base_size=18)

    code = (
        "# src/train.py -- pruning is a 3-line hook\n"
        "if optuna_trial is not None:\n"
        "    optuna_trial.report(val_stats['acc'], step=epoch)\n"
        "    if optuna_trial.should_prune():\n"
        "        raise optuna.TrialPruned(...)\n"
        "\n"
        "# src/gradcam.py -- class-discriminative heatmap\n"
        "score = logits[0, target_class]; score.backward()\n"
        "alpha_k = self._gradients[0].mean(dim=(1, 2))\n"
        "cam = F.relu((alpha_k[:, None, None] * self._activations[0]).sum(dim=0))\n"
    )
    _add_code_box(slide, code,
                  left=Inches(0.6), top=Inches(3.3),
                  width=SLIDE_WIDTH - Inches(1.2), height=Inches(4.0),
                  size=13)


def build_results_headline(prs: Presentation, headlines: List[ArchHeadline],
                           output_dir: Path) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    _add_title(slide, "Results: Size vs. Accuracy")
    _add_subtitle_box(
        slide,
        "EfficientNet-B0 matches ResNet50 within 0.5% -- at less than 1/5 the parameters."
    )

    # Image on the left.
    _add_image(slide,
               output_dir / "report" / "size_vs_accuracy.png",
               left=Inches(0.5), top=Inches(1.7), height=Inches(5.4))

    # Table on the right.
    table_left = Inches(7.0)
    table_top = Inches(2.0)
    table_w = Inches(5.8)
    rows = 1 + len(headlines)
    cols = 5
    tbl = slide.shapes.add_table(
        rows, cols, table_left, table_top, table_w, Inches(2.2)
    ).table

    headers = ["Arch", "Params", "Best val", "Test top-1", "Test top-5"]
    for c, h in enumerate(headers):
        tbl.cell(0, c).text = h
        for run in tbl.cell(0, c).text_frame.paragraphs[0].runs:
            run.font.bold = True
            run.font.size = Pt(13)

    for r, h in enumerate(headlines, start=1):
        values = [
            h.display,
            _fmt_params(h.params) if h.params else "n/a",
            _fmt_pct(h.val_acc),
            _fmt_pct(h.test_top1),
            _fmt_pct(h.test_top5),
        ]
        for c, v in enumerate(values):
            tbl.cell(r, c).text = v
            for run in tbl.cell(r, c).text_frame.paragraphs[0].runs:
                run.font.size = Pt(12)

    # Reference baselines below the table.
    ref = slide.shapes.add_textbox(
        Inches(7.0), Inches(5.0), Inches(5.8), Inches(2.0)
    )
    rt = ref.text_frame
    rt.word_wrap = True
    rt.text = "Reference (torchvision ImageNet, 1000 classes):"
    for run in rt.paragraphs[0].runs:
        run.font.size = Pt(13)
        run.font.bold = True

    p = rt.add_paragraph()
    p.text = ("MobileNetV2 ~72%   EfficientNet-B0 ~77%   ResNet50 ~76%. "
              "All three reach >90% here -- transfer learning carries the load.")
    for run in p.runs:
        run.font.size = Pt(12)


def build_gradcam_slide(prs: Presentation, output_dir: Path) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    _add_title(slide, "Where the Models Look (Grad-CAM)")

    bullets = [
        "Same image, every architecture; top-3 predictions per model.",
        "Easy classes (sushi, pizza) -- all three attend to the food itself.",
        "Confused pairs (Taco/Taquito, apple_pie/cheesecake) -- lighter models spread attention; ResNet focuses tighter.",
        "Top-3 reveals the SHAPE of disagreement, not just the winner.",
    ]
    _add_bullets(slide, bullets, top=Inches(1.4),
                 height=Inches(2.0), base_size=16)

    # Prefer the new compare image if it exists; fall back to the
    # single-arch gradcam.png from the original script.
    candidate = output_dir / "gradcam_compare.png"
    if not candidate.exists():
        candidate = output_dir / "gradcam.png"
    _add_image(slide, candidate,
               left=Inches(1.5), top=Inches(3.4), width=Inches(10.3))


def build_confusions_slide(prs: Presentation, output_dir: Path) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    _add_title(slide, "Per-class & Confusion Analysis")

    bullets = [
        "All three architectures share the same hard cases -- dataset signal, not model failure:",
        "  Taco -> Taquito  (top confusion across the board)",
        "  apple_pie <-> cheesecake",
        "  Hot Dog <-> Sandwich",
        "Top-5 saturated (>99%) -- residual error is fine-grained class confusion.",
    ]
    _add_bullets(slide, bullets, top=Inches(1.4),
                 height=Inches(2.0), base_size=16)

    _add_image(slide,
               output_dir / "eval" / "confusion_matrices_3up.png",
               left=Inches(0.7), top=Inches(3.4), width=Inches(11.9))


def build_what_worked(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    _add_title(slide, "What Worked vs. What Didn't")

    # Two columns.
    col_w = Inches(6.0)

    left = slide.shapes.add_textbox(Inches(0.6), Inches(1.6), col_w, Inches(5.5))
    lt = left.text_frame
    lt.word_wrap = True
    lt.text = "Worked"
    for run in lt.paragraphs[0].runs:
        run.font.size = Pt(22); run.font.bold = True
        run.font.color.rgb = RGBColor(0x1f, 0x77, 0xb4)

    for line in [
        "Shared manifest + shared training loop -- 100% reproducible, fully parallel across teammates.",
        "MedianPruner saved ~37% of trial compute (28/76 pruned).",
        "Letting Optuna pick the optimizer -- ResNet preferred AdamW; the others preferred SGD. We'd have guessed wrong.",
    ]:
        p = lt.add_paragraph()
        p.text = line
        p.level = 0
        for run in p.runs:
            run.font.size = Pt(15)

    right = slide.shapes.add_textbox(
        Inches(6.9), Inches(1.6), col_w, Inches(5.5)
    )
    rt = right.text_frame
    rt.word_wrap = True
    rt.text = "Surprised us"
    for run in rt.paragraphs[0].runs:
        run.font.size = Pt(22); run.font.bold = True
        run.font.color.rgb = RGBColor(0xd6, 0x27, 0x28)

    for line in [
        "freeze_backbone=True LOST. Full fine-tuning beat freezing for MobileNetV2.",
        "cosine_lr_schedule=True LOST for EfficientNet-B0. Constant LR with the right starting value won.",
        "The 'heavy ceiling' assumption nearly broke -- ResNet50 only edged EfficientNet-B0 by 0.5% (within noise).",
        "Honest limit: we tuned for accuracy alone. Real deployment also weighs latency / memory.",
    ]:
        p = rt.add_paragraph()
        p.text = line
        for run in p.runs:
            run.font.size = Pt(15)


def build_conclusion(prs: Presentation) -> None:
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    _add_title(slide, "Conclusion & Future Work")

    # Conclusion (left)
    left = slide.shapes.add_textbox(Inches(0.6), Inches(1.6), Inches(6.0), Inches(5.5))
    lt = left.text_frame
    lt.word_wrap = True
    lt.text = "Conclusion"
    for run in lt.paragraphs[0].runs:
        run.font.size = Pt(22); run.font.bold = True

    for line in [
        "EfficientNet-B0 is the right pick for this dataset -- Pareto-optimal on size vs. accuracy.",
        "The methodology generalizes: shared manifest + per-arch sweeps + pruning-aware HPO + Grad-CAM compare.",
        "3 architectures, 76 trials, 3 machines -- all fully comparable.",
    ]:
        p = lt.add_paragraph()
        p.text = line
        for run in p.runs:
            run.font.size = Pt(15)

    # Future work (right)
    right = slide.shapes.add_textbox(Inches(6.9), Inches(1.6), Inches(6.0), Inches(5.5))
    rt = right.text_frame
    rt.word_wrap = True
    rt.text = "Future work"
    for run in rt.paragraphs[0].runs:
        run.font.size = Pt(22); run.font.bold = True

    for line in [
        "Latency & memory benchmarks on CPU and a phone-class accelerator.",
        "Targeted augmentation for the Taco/Taquito family.",
        "Add a transformer baseline (ViT-B/16, DINOv2) to see if the trade-off shifts.",
        "Demo the harness on a second dataset (Food-101, CIFAR-100) to validate the 'reusable' claim.",
    ]:
        p = rt.add_paragraph()
        p.text = line
        for run in p.runs:
            run.font.size = Pt(15)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate the CSCI 611 final-project starter slides."
    )
    p.add_argument("--output-dir", default="outputs",
                   help="Where the per-arch summaries / plots live.")
    p.add_argument(
        "--output",
        default="../CSCI611_FinalProject_Slides.pptx",
        help="Where to write the .pptx (default: project root, alongside the docx/pdf deliverables).",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    output_dir = Path(args.output_dir)

    prs = Presentation()
    prs.slide_width = SLIDE_WIDTH
    prs.slide_height = SLIDE_HEIGHT

    # Load the headline numbers once so the whole deck cites the same
    # (current) values, even if results.json updates mid-run.
    headlines = load_headlines(output_dir)

    build_title(prs)
    build_team(prs)
    build_context(prs)
    build_framework(prs)
    build_implementation_1(prs)
    build_implementation_2(prs)
    build_results_headline(prs, headlines, output_dir)
    build_gradcam_slide(prs, output_dir)
    build_confusions_slide(prs, output_dir)
    build_what_worked(prs)
    build_conclusion(prs)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    prs.save(out_path)
    print(f"Wrote {out_path.resolve()}")
    print(f"  {len(prs.slides)} slides")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
