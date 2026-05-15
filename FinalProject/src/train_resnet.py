"""
src/train_resnet.py
===================

Per-architecture entry point for **ResNet50** -- the "heavy /
performance ceiling" model in the three-way comparison.

This script is meant to be run on Jacob's, Anand's, or Paolo's machine
*independently of the other two*: while one teammate runs this script,
another can run `train_efficientnet.py` and the third can run
`train_mobilenet.py`. They share no state -- each writes its own Optuna
SQLite database and its own checkpoint directory.

Usage
-----
From the repo root, with the venv activated and the dataset at
`../dataset/`:

    python -m src.train_resnet \\
        --manifest configs/data_split.yaml \\
        --trials 20 \\
        --epochs 8 \\
        --num-workers 4

Outputs (under ./outputs/ by default)
-------------------------------------
    outputs/
        optuna_studies/resnet50.db        -- Optuna study, resumable
        checkpoints/resnet50/trial_<N>/   -- best.pt for each trial
        resnet50_best_summary.json        -- best trial's hp + val acc
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import optuna

from src.tune_optuna import (
    CommonSearchSpace,
    create_or_load_study,
    make_objective,
)


# ---------------------------------------------------------------------------
# Architecture-specific search-space extension
# ---------------------------------------------------------------------------

def resnet_extra_space(trial: optuna.Trial) -> Dict[str, Any]:
    """Add ResNet-specific knobs on top of the common space.

    `label_smoothing`: ResNet50 has high capacity; light label smoothing
    (epsilon ~ 0--0.1) is a well-known regularizer that often nudges
    val accuracy up by a percent or two on moderately-sized datasets.
    """
    return {
        "label_smoothing": trial.suggest_float("label_smoothing", 0.0, 0.1),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run an Optuna hyperparameter search for ResNet50."
    )
    p.add_argument(
        "--manifest",
        default="configs/data_split.yaml",
        help="Split manifest produced by scripts/generate_split_manifest.py.",
    )
    p.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Number of Optuna trials. The MedianPruner will end weak trials early.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=8,
        help="Maximum epochs per trial. Pruning may stop a trial earlier.",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers. Set to 0 if you hit Windows multiprocessing issues.",
    )
    p.add_argument(
        "--output-dir",
        default="outputs",
        help="Root directory for checkpoints and Optuna study.",
    )
    p.add_argument(
        "--study-name",
        default="resnet50",
        help="Optuna study name. The SQLite DB will be <output-dir>/optuna_studies/<study-name>.db.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    output_dir = Path(args.output_dir)

    # Build the persistent study. Re-running this script with the same
    # study name will append more trials to the existing DB rather than
    # overwriting -- handy if you want to extend a search.
    study = create_or_load_study(
        study_name=args.study_name,
        storage_dir=output_dir / "optuna_studies",
    )

    # Build the objective with ResNet-specific extra search space.
    objective = make_objective(
        arch="resnet50",
        manifest_path=args.manifest,
        num_workers=args.num_workers,
        epochs_per_trial=args.epochs,
        checkpoint_dir=output_dir / "checkpoints" / "resnet50",
        common_space=CommonSearchSpace(),
        extra_space_fn=resnet_extra_space,
    )

    # Run the search. `n_trials` is per-call, so previous trials in the
    # study DB are not rerun.
    study.optimize(objective, n_trials=args.trials, gc_after_trial=True)

    # Summarize the best trial.
    best = study.best_trial
    print("\n=== ResNet50 search complete ===")
    print(f"  best trial number: {best.number}")
    print(f"  best val acc:      {best.value:.4f}")
    print(f"  best hyperparams:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    # Persist the best-trial summary as JSON. Matches what the other two
    # entry points write so the final-comparison script can read all three
    # uniformly.
    summary = {
        "arch": "resnet50",
        "best_trial_number": best.number,
        "best_val_acc": best.value,
        "best_hparams": best.params,
        "n_trials": len(study.trials),
    }
    summary_path = output_dir / f"{args.study_name}_best_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"  wrote summary -> {summary_path}")


if __name__ == "__main__":
    main()
