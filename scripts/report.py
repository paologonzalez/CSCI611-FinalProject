#!/usr/bin/env python3
"""
scripts/report.py
=================

Cross-architecture analysis & report generator.

Why this script exists
----------------------
After all three teammates finish their per-architecture Optuna sweeps
(`src/train_resnet.py`, `src/train_efficientnet.py`,
`src/train_mobilenet.py`), each leaves behind:

    outputs/optuna_studies/<arch>.db          -- the full study
    outputs/<arch>_best_summary.json          -- best-trial summary
    outputs/eval/<arch>/metrics.json          -- (optional) test metrics
                                                 from `src/evaluate.py`

This script loads all of those, produces analysis plots, and writes a
single markdown report that summarizes the comparison. The end result is
the "Eval & analyze" box at the right edge of the elevator-pitch
diagram: one place that answers "which architecture wins on
accuracy / size / cost, and what hyperparameters got it there."

What gets produced
------------------
Under `outputs/report/`:

    per_arch/<arch>/
        optimization_history.png   -- val_acc vs trial number
        param_importances.png      -- which hparams mattered most (fANOVA)
        slice.png                  -- val_acc vs each hparam individually
        parallel_coordinate.png    -- multi-hparam interactions
    comparison_table.csv           -- one row per architecture
    size_vs_accuracy.png           -- the project's central trade-off plot
    accuracy_ranked.png            -- bar chart of best val acc per arch
    report.md                      -- combined markdown report (read me!)

Robustness
----------
This script is safe to run mid-project: if only one or two
architectures have been trained, it will skip the missing ones and
produce a partial report. Same for missing test metrics.

Usage
-----
From the repo root, with the venv activated:

    python scripts/report.py
    # or, with custom paths:
    python scripts/report.py --output-dir outputs --report-dir outputs/report
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# We import optuna only after argparse so that --help doesn't require it
# to be installed. (Anyone running --help is probably trying to figure out
# what this script does, not run it.)


# ---------------------------------------------------------------------------
# Project-wide configuration
# ---------------------------------------------------------------------------

# The three architectures we benchmark. Order here drives the order in the
# comparison table and trade-off plot. Sorted lightest -> heaviest so the
# trade-off curve reads left-to-right by model size.
ARCHITECTURES: List[str] = [
    "mobilenet_v2",
    "efficientnet_b0",
    "resnet50",
]

# Display names used in plots and the markdown report. Keeping these in one
# place makes it easy to re-style without hunting through the file.
ARCH_DISPLAY_NAMES: Dict[str, str] = {
    "mobilenet_v2": "MobileNetV2",
    "efficientnet_b0": "EfficientNet-B0",
    "resnet50": "ResNet50",
}


# ---------------------------------------------------------------------------
# Data classes for collected results
# ---------------------------------------------------------------------------

@dataclass
class ArchResult:
    """Everything we know about one architecture after its sweep + eval.

    Some fields are Optional because they depend on whether downstream
    steps have been run yet:
      * `study` requires the per-arch train script to have run at least
        one trial.
      * `test_metrics` requires `src/evaluate.py` to have run on the best
        checkpoint.
      * `param_count` is read from the best trial's checkpoint metadata
        and is therefore tied to the best trial existing.
    """

    arch: str
    display_name: str
    study: Optional[Any] = None       # optuna.Study, untyped to avoid early import
    best_summary: Optional[Dict[str, Any]] = None
    test_metrics: Optional[Dict[str, Any]] = None
    param_count: Optional[int] = None
    n_trials: int = 0
    n_pruned: int = 0
    n_complete: int = 0
    n_failed: int = 0
    notes: List[str] = field(default_factory=list)

    @property
    def best_val_acc(self) -> Optional[float]:
        """Best validation accuracy seen across all trials (None if no trials)."""
        if self.best_summary is not None:
            return float(self.best_summary.get("best_val_acc", float("nan")))
        if self.study is not None and self.n_complete > 0:
            return float(self.study.best_value)
        return None

    @property
    def best_hparams(self) -> Dict[str, Any]:
        if self.best_summary is not None:
            return dict(self.best_summary.get("best_hparams", {}))
        if self.study is not None and self.n_complete > 0:
            return dict(self.study.best_trial.params)
        return {}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_study_if_present(arch: str, output_dir: Path) -> Optional[Any]:
    """Open an Optuna study from its SQLite file, or return None if missing.

    The studies are persisted by `tune_optuna.create_or_load_study`; the
    file path is therefore predictable.
    """
    import optuna

    db_path = output_dir / "optuna_studies" / f"{arch}.db"
    if not db_path.exists():
        return None

    return optuna.load_study(
        study_name=arch,
        storage=f"sqlite:///{db_path}",
    )


def _load_json_if_present(path: Path) -> Optional[Dict[str, Any]]:
    """Read a JSON file or return None if it doesn't exist."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _read_param_count_from_best_checkpoint(
    arch: str,
    output_dir: Path,
    best_trial_number: Optional[int],
) -> Optional[int]:
    """Pull `param_count` out of the best trial's checkpoint metadata JSON.

    `train.run_training` saves a `<tag>_best.json` next to each
    `<tag>_best.pt`. The metadata dict has shape:
        {"epoch": ..., "val_acc": ..., "extra_metadata": {"param_count": ...}}
    """
    if best_trial_number is None:
        return None

    candidate = (
        output_dir
        / "checkpoints"
        / arch
        / f"trial_{best_trial_number}"
        / f"{arch}_trial{best_trial_number}_best.json"
    )
    meta = _load_json_if_present(candidate)
    if meta is None:
        return None

    extra = meta.get("extra_metadata", {})
    val = extra.get("param_count")
    return int(val) if val is not None else None


def collect_results(output_dir: Path) -> List[ArchResult]:
    """Build an ArchResult for every architecture we know about.

    Architectures that have not been trained yet still get a result
    object, just with `study=None` and a helpful note. That keeps the
    report consistent across stages of the project.
    """
    results: List[ArchResult] = []

    for arch in ARCHITECTURES:
        result = ArchResult(arch=arch, display_name=ARCH_DISPLAY_NAMES[arch])

        # 1. Optuna study (the bulk of the data).
        result.study = _load_study_if_present(arch, output_dir)
        if result.study is None:
            result.notes.append(
                f"No Optuna study found at outputs/optuna_studies/{arch}.db -- "
                f"has anyone run `python -m src.train_{arch}` yet?"
            )
        else:
            # Tally trial states. We use string comparison rather than the
            # enum so this works across optuna versions.
            for trial in result.study.trials:
                state = trial.state.name
                result.n_trials += 1
                if state == "COMPLETE":
                    result.n_complete += 1
                elif state == "PRUNED":
                    result.n_pruned += 1
                elif state == "FAIL":
                    result.n_failed += 1

        # 2. Best-trial summary JSON (a small denormalized convenience
        #    file that the per-arch training scripts write at the end).
        result.best_summary = _load_json_if_present(
            output_dir / f"{arch}_best_summary.json"
        )

        # 3. Param count from the best trial's checkpoint metadata.
        best_trial_number = (
            result.best_summary.get("best_trial_number")
            if result.best_summary is not None
            else None
        )
        result.param_count = _read_param_count_from_best_checkpoint(
            arch, output_dir, best_trial_number
        )

        # 4. Test-set metrics if `src/evaluate.py` has been run.
        result.test_metrics = _load_json_if_present(
            output_dir / "eval" / arch / "metrics.json"
        )
        if result.test_metrics is None and result.study is not None:
            result.notes.append(
                "No test-set metrics. Run `python -m src.evaluate "
                f"--arch {arch} --checkpoint <best.pt>` after training."
            )

        results.append(result)

    return results


# ---------------------------------------------------------------------------
# Per-architecture plots (Optuna's built-in matplotlib viz)
# ---------------------------------------------------------------------------

def _save_optuna_plots_for_arch(
    result: ArchResult,
    output_dir: Path,
) -> List[str]:
    """Save Optuna's four standard analysis plots for one architecture.

    Each plot answers a different question:

      * `optimization_history` -- "Did the search keep improving, or did
        it plateau?" If the curve goes flat early, you may have under-
        searched OR found a true ceiling for that arch.
      * `param_importances` -- "Which hparams actually mattered?" Optuna
        uses fANOVA to estimate the variance contribution of each
        hyperparameter to the objective. The bar plot lets you say
        things like "lr explains 70% of the variance, batch_size <5%".
      * `slice` -- "How does the objective vary along each hparam,
        marginalized over the others?" Useful for confirming whether
        the search hit the boundary of a range you set (which means the
        range was too narrow).
      * `parallel_coordinate` -- "What patterns in joint hparam
        configurations led to the best trials?" Each trial is a polyline
        across axes; high-performing trials cluster.

    Returns a list of paths actually written. If a plot can't be made
    (e.g., not enough completed trials yet), it's silently skipped and
    a note is appended to the result.
    """
    if result.study is None or result.n_complete == 0:
        result.notes.append(
            "Skipping per-arch plots -- no completed trials available."
        )
        return []

    # Lazy imports keep --help fast for users who don't have these libs.
    import matplotlib
    matplotlib.use("Agg")  # headless: no display server required.
    import matplotlib.pyplot as plt
    from optuna.visualization import matplotlib as ovm

    arch_dir = output_dir / "per_arch" / result.arch
    arch_dir.mkdir(parents=True, exist_ok=True)

    written: List[str] = []

    # The matplotlib visualization module returns matplotlib Axes
    # objects. We grab `ax.figure`, save, close. Each plot is wrapped in
    # a try/except because a few (e.g., param_importances) require >= 2
    # completed trials with varying parameters; we don't want one missing
    # plot to abort the rest.

    plots = [
        ("optimization_history.png", lambda: ovm.plot_optimization_history(result.study)),
        ("param_importances.png",    lambda: ovm.plot_param_importances(result.study)),
        ("slice.png",                lambda: ovm.plot_slice(result.study)),
        ("parallel_coordinate.png",  lambda: ovm.plot_parallel_coordinate(result.study)),
    ]

    for filename, plot_fn in plots:
        try:
            ax = plot_fn()
            # Some optuna viz return a single Axes, some return an array
            # of Axes (e.g. plot_slice with multiple params). Either way,
            # we can grab the figure off the first one.
            fig = ax.figure if hasattr(ax, "figure") else ax.flat[0].figure
            fig.suptitle(f"{result.display_name} -- {filename.removesuffix('.png')}")
            fig.tight_layout()
            out_path = arch_dir / filename
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            written.append(str(out_path))
        except Exception as e:
            # Common reasons: too few trials for fANOVA, or all trials
            # used identical hparams in a category. Log and move on.
            result.notes.append(f"Could not produce {filename}: {e}")

    return written


# ---------------------------------------------------------------------------
# Cross-architecture plots
# ---------------------------------------------------------------------------

def _plot_size_vs_accuracy(results: List[ArchResult], output_path: Path) -> bool:
    """Scatter of (param_count) vs (best val accuracy) per architecture.

    This is the central trade-off the proposal asks about: do you really
    need the heavy model, or does the light one come close? Each point
    is annotated with the architecture name; we draw light grid lines so
    the relative positions are easy to read.

    Returns True if the plot was written, False if there isn't enough
    data yet (e.g., no architectures with both metrics).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = [
        (r, r.param_count, r.best_val_acc)
        for r in results
        if r.param_count is not None and r.best_val_acc is not None
    ]
    if not points:
        return False

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    for r, params, acc in points:
        # Plot point + label. Convert param count to millions for axis
        # readability (e.g. "3.5M" instead of "3504872").
        ax.scatter(params / 1e6, acc * 100, s=120, zorder=3)
        ax.annotate(
            r.display_name,
            (params / 1e6, acc * 100),
            xytext=(8, 6),
            textcoords="offset points",
            fontsize=10,
        )

    ax.set_xlabel("Trainable parameters (millions)")
    ax.set_ylabel("Best validation accuracy (%)")
    ax.set_title("Size vs. accuracy across architectures")
    ax.grid(True, linestyle="--", alpha=0.4, zorder=0)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_accuracy_ranked(results: List[ArchResult], output_path: Path) -> bool:
    """Horizontal bar chart of best validation accuracy per architecture.

    Lets the reader rank the architectures at a glance. We also overlay
    the test-set top-1 (when present) as a hatched bar behind the val
    bar -- a quick way to spot architectures that overfit (val >> test).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eligible = [r for r in results if r.best_val_acc is not None]
    if not eligible:
        return False

    names = [r.display_name for r in eligible]
    vals = [r.best_val_acc * 100 for r in eligible]
    tests = [
        (r.test_metrics["top1"] * 100) if (r.test_metrics is not None) else None
        for r in eligible
    ]

    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    y = list(range(len(eligible)))

    # Background hatched bars: test top-1 (where present).
    for i, t in enumerate(tests):
        if t is not None:
            ax.barh(
                i, t, color="lightgray", edgecolor="gray",
                hatch="///", zorder=1,
                label="Test top-1" if i == 0 else None,
            )

    # Foreground filled bars: best val accuracy.
    ax.barh(y, vals, color="#1f77b4", zorder=2,
            label="Best val acc")

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("Best validation vs. test accuracy")
    ax.set_xlim(0, 100)
    ax.grid(True, axis="x", linestyle="--", alpha=0.4, zorder=0)
    if any(t is not None for t in tests):
        ax.legend(loc="lower right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return True


# ---------------------------------------------------------------------------
# Comparison table (CSV)
# ---------------------------------------------------------------------------

def write_comparison_csv(results: List[ArchResult], output_path: Path) -> None:
    """One row per architecture. Suitable for opening in Excel/Sheets.

    Columns:
      arch, display_name, param_count, best_val_acc, test_top1, test_top5,
      n_trials, n_complete, n_pruned, n_failed,
      best_lr, best_weight_decay, best_optimizer, best_batch_size,
      best_dropout, extra_hparams_json
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "arch", "display_name", "param_count",
        "best_val_acc", "test_top1", "test_top5",
        "n_trials", "n_complete", "n_pruned", "n_failed",
        "best_lr", "best_weight_decay", "best_optimizer",
        "best_batch_size", "best_dropout", "extra_hparams_json",
    ]

    # Hyperparameters that belong to the common search space and so get
    # their own dedicated columns. Anything else (the per-arch extras
    # like `label_smoothing`) is bundled into the JSON column.
    COMMON_KEYS = {"lr", "weight_decay", "optimizer", "batch_size", "dropout"}

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            hp = r.best_hparams
            extras = {k: v for k, v in hp.items() if k not in COMMON_KEYS}
            row = {
                "arch": r.arch,
                "display_name": r.display_name,
                "param_count": r.param_count,
                "best_val_acc": r.best_val_acc,
                "test_top1": r.test_metrics["top1"] if r.test_metrics else None,
                "test_top5": r.test_metrics["top5"] if r.test_metrics else None,
                "n_trials": r.n_trials,
                "n_complete": r.n_complete,
                "n_pruned": r.n_pruned,
                "n_failed": r.n_failed,
                "best_lr": hp.get("lr"),
                "best_weight_decay": hp.get("weight_decay"),
                "best_optimizer": hp.get("optimizer"),
                "best_batch_size": hp.get("batch_size"),
                "best_dropout": hp.get("dropout"),
                "extra_hparams_json": json.dumps(extras) if extras else "",
            }
            writer.writerow(row)


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def _format_acc(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def _format_count(value: Optional[int]) -> str:
    if value is None:
        return "n/a"
    if value >= 1_000_000:
        return f"{value / 1e6:.1f}M"
    if value >= 1_000:
        return f"{value / 1e3:.1f}k"
    return str(value)


def write_markdown_report(
    results: List[ArchResult],
    report_dir: Path,
    plots_written: Dict[str, List[str]],
    plots_cross: Dict[str, bool],
) -> Path:
    """Write `report.md` -- the human-readable summary.

    The report has three sections:
      1. Overview table -- one line per architecture.
      2. Cross-architecture plots -- size-vs-accuracy and ranked accuracy.
      3. Per-architecture sections -- best hparams + the four Optuna plots.

    All image paths in the markdown are *relative to the report file*,
    so the report renders correctly when emailed or zipped up.
    """
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "report.md"

    lines: List[str] = []

    # -------- Header -------------------------------------------------------
    lines.append("# CSCI611 Final Project -- Architecture Comparison Report")
    lines.append("")
    lines.append(
        "Auto-generated by `scripts/report.py`. Re-run that script after "
        "any new training or evaluation to refresh the numbers and plots."
    )
    lines.append("")

    # -------- Overview table ----------------------------------------------
    lines.append("## Overview")
    lines.append("")
    lines.append(
        "| Architecture | Params | Best val acc | Test top-1 | Test top-5 | "
        "Trials (complete / pruned / failed) |"
    )
    lines.append(
        "|---|---|---|---|---|---|"
    )
    for r in results:
        test_top1 = _format_acc(r.test_metrics["top1"]) if r.test_metrics else "n/a"
        test_top5 = _format_acc(r.test_metrics["top5"]) if r.test_metrics else "n/a"
        trial_breakdown = f"{r.n_complete} / {r.n_pruned} / {r.n_failed}"
        lines.append(
            f"| {r.display_name} | {_format_count(r.param_count)} "
            f"| {_format_acc(r.best_val_acc)} "
            f"| {test_top1} | {test_top5} "
            f"| {trial_breakdown} |"
        )
    lines.append("")

    # -------- Cross-architecture plots ------------------------------------
    lines.append("## Cross-architecture plots")
    lines.append("")
    if plots_cross.get("size_vs_accuracy"):
        lines.append("### Size vs. accuracy")
        lines.append("")
        lines.append(
            "The central trade-off the project asks about. Look for the "
            "architecture closest to the upper-left corner -- highest "
            "accuracy at the lowest parameter count."
        )
        lines.append("")
        lines.append("![size vs accuracy](size_vs_accuracy.png)")
        lines.append("")
    else:
        lines.append(
            "_size_vs_accuracy.png skipped: not enough data yet (need both "
            "param count and best val acc for at least one architecture)._"
        )
        lines.append("")

    if plots_cross.get("accuracy_ranked"):
        lines.append("### Accuracy ranking")
        lines.append("")
        lines.append(
            "Best validation accuracy per architecture (filled bar) with "
            "test-set top-1 overlaid as a hatched background bar. A large "
            "gap between val and test indicates overfitting to the val "
            "split during HPO."
        )
        lines.append("")
        lines.append("![accuracy ranked](accuracy_ranked.png)")
        lines.append("")

    # -------- Per-architecture sections -----------------------------------
    lines.append("## Per-architecture detail")
    lines.append("")
    for r in results:
        lines.append(f"### {r.display_name}")
        lines.append("")

        if r.study is None:
            lines.append("_No study found yet._")
            for note in r.notes:
                lines.append(f"- {note}")
            lines.append("")
            continue

        lines.append(f"- **Trials**: {r.n_trials} total "
                     f"({r.n_complete} complete, {r.n_pruned} pruned, "
                     f"{r.n_failed} failed)")
        lines.append(f"- **Best validation accuracy**: {_format_acc(r.best_val_acc)}")
        if r.test_metrics is not None:
            lines.append(f"- **Test top-1**: {_format_acc(r.test_metrics['top1'])}")
            lines.append(f"- **Test top-5**: {_format_acc(r.test_metrics['top5'])}")
        if r.param_count is not None:
            lines.append(f"- **Parameter count**: {_format_count(r.param_count)}")

        if r.best_hparams:
            lines.append("- **Best hyperparameters**:")
            for k, v in r.best_hparams.items():
                # Format floats compactly to keep the report readable.
                if isinstance(v, float):
                    formatted = f"{v:.4g}"
                else:
                    formatted = str(v)
                lines.append(f"    - `{k}`: {formatted}")

        # Plots (if any were saved for this arch)
        for plot_path in plots_written.get(r.arch, []):
            relative = Path(plot_path).relative_to(report_dir)
            label = Path(plot_path).stem.replace("_", " ")
            lines.append("")
            lines.append(f"![{label}]({relative})")

        if r.notes:
            lines.append("")
            lines.append("**Notes:**")
            for note in r.notes:
                lines.append(f"- {note}")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate cross-architecture analysis report.",
    )
    p.add_argument(
        "--output-dir",
        default="outputs",
        help="Where the per-arch training scripts wrote their results.",
    )
    p.add_argument(
        "--report-dir",
        default="outputs/report",
        help="Where this script writes plots, CSV, and report.md.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()

    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)

    # Step 1: gather everything we know about each architecture.
    results = collect_results(output_dir)

    # Hard-fail only if literally nothing has been trained -- otherwise
    # generate a partial report.
    if not any(r.study is not None for r in results):
        print(
            "No Optuna studies found under "
            f"{(output_dir / 'optuna_studies').resolve()}. "
            "Run one of `python -m src.train_<arch>` first.",
            file=sys.stderr,
        )
        return 1

    # Step 2: per-architecture Optuna plots.
    plots_written: Dict[str, List[str]] = {}
    for r in results:
        plots_written[r.arch] = _save_optuna_plots_for_arch(r, report_dir)

    # Step 3: cross-architecture plots.
    plots_cross: Dict[str, bool] = {
        "size_vs_accuracy": _plot_size_vs_accuracy(
            results, report_dir / "size_vs_accuracy.png"
        ),
        "accuracy_ranked": _plot_accuracy_ranked(
            results, report_dir / "accuracy_ranked.png"
        ),
    }

    # Step 4: comparison CSV.
    write_comparison_csv(results, report_dir / "comparison_table.csv")

    # Step 5: markdown report tying everything together.
    report_path = write_markdown_report(
        results, report_dir, plots_written, plots_cross
    )

    # Step 6: friendly stdout summary so the user knows what was written.
    print("=== Report generation complete ===")
    print(f"  Report:           {report_path}")
    print(f"  Comparison table: {report_dir / 'comparison_table.csv'}")
    for arch, paths in plots_written.items():
        print(f"  {arch}: {len(paths)} per-arch plot(s)")
    for name, ok in plots_cross.items():
        print(f"  {name}: {'written' if ok else 'skipped (insufficient data)'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
