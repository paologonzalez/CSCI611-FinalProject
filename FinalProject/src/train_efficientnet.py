"""
src/train_efficientnet.py
=========================

Per-architecture entry point for **EfficientNet-B0** -- the "intermediate
/ compound-scaled" model in the three-way comparison.

Run this on its own machine in parallel with `train_resnet.py` and
`train_mobilenet.py`. It writes its own Optuna study (sqlite) and its
own checkpoint directory; nothing is shared.

Usage
-----
    python -m src.train_efficientnet \\
        --manifest configs/data_split.yaml \\
        --trials 25 \\
        --epochs 10 \\
        --num-workers 4

Outputs (under ./outputs/ by default)
-------------------------------------
    outputs/
        optuna_studies/efficientnet_b0.db
        checkpoints/efficientnet_b0/trial_<N>/
        efficientnet_b0_best_summary.json
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

def efficientnet_extra_space(trial: optuna.Trial) -> Dict[str, Any]:
    """Add EfficientNet-specific knobs on top of the common space.

    `cosine_lr_schedule`: EfficientNet's original training recipe used a
    cosine learning-rate schedule. We expose it as a categorical knob so
    Optuna can decide whether the cosine schedule helps over a plain
    constant LR for our (much smaller) dataset and budget.
    """
    return {
        "cosine_lr_schedule": trial.suggest_categorical(
            "cosine_lr_schedule", [True, False]
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run an Optuna hyperparameter search for EfficientNet-B0."
    )
    p.add_argument("--manifest", default="configs/data_split.yaml")
    p.add_argument(
        "--trials",
        type=int,
        default=25,
        help="Number of Optuna trials. EfficientNet-B0 is small, so 25 is affordable.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Maximum epochs per trial.",
    )
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--study-name", default="efficientnet_b0")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)

    study = create_or_load_study(
        study_name=args.study_name,
        storage_dir=output_dir / "optuna_studies",
    )

    objective = make_objective(
        arch="efficientnet_b0",
        manifest_path=args.manifest,
        num_workers=args.num_workers,
        epochs_per_trial=args.epochs,
        checkpoint_dir=output_dir / "checkpoints" / "efficientnet_b0",
        common_space=CommonSearchSpace(),
        extra_space_fn=efficientnet_extra_space,
    )

    study.optimize(objective, n_trials=args.trials, gc_after_trial=True)

    best = study.best_trial
    print("\n=== EfficientNet-B0 search complete ===")
    print(f"  best trial number: {best.number}")
    print(f"  best val acc:      {best.value:.4f}")
    print(f"  best hyperparams:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    summary = {
        "arch": "efficientnet_b0",
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
