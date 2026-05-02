"""
src/train_mobilenet.py
======================

Per-architecture entry point for **MobileNetV2** -- the "lightweight /
efficiency baseline" in the three-way comparison.

Run this on its own machine in parallel with `train_resnet.py` and
`train_efficientnet.py`.

Because MobileNetV2 is the cheapest of the three to train, the defaults
here run more trials (30) so the search can be more thorough -- the goal
is to give the lightweight model its best shot at the cost/accuracy
trade-off question.

Usage
-----
    python -m src.train_mobilenet \\
        --manifest configs/data_split.yaml \\
        --trials 30 \\
        --epochs 12 \\
        --num-workers 4

Outputs (under ./outputs/ by default)
-------------------------------------
    outputs/
        optuna_studies/mobilenet_v2.db
        checkpoints/mobilenet_v2/trial_<N>/
        mobilenet_v2_best_summary.json
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

def mobilenet_extra_space(trial: optuna.Trial) -> Dict[str, Any]:
    """Add MobileNetV2-specific knobs.

    `freeze_backbone`: MobileNetV2 is the smallest model; freezing its
    backbone and only training the head can converge faster and avoid
    overfitting on a moderate-size dataset. Optuna decides whether
    freezing helps for this dataset.
    """
    return {
        "freeze_backbone": trial.suggest_categorical(
            "freeze_backbone", [True, False]
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run an Optuna hyperparameter search for MobileNetV2."
    )
    p.add_argument("--manifest", default="configs/data_split.yaml")
    p.add_argument(
        "--trials",
        type=int,
        default=30,
        help="Number of Optuna trials. MobileNetV2 is cheap; 30 is reasonable.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=12,
        help="Maximum epochs per trial.",
    )
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--study-name", default="mobilenet_v2")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)

    study = create_or_load_study(
        study_name=args.study_name,
        storage_dir=output_dir / "optuna_studies",
    )

    objective = make_objective(
        arch="mobilenet_v2",
        manifest_path=args.manifest,
        num_workers=args.num_workers,
        epochs_per_trial=args.epochs,
        checkpoint_dir=output_dir / "checkpoints" / "mobilenet_v2",
        common_space=CommonSearchSpace(),
        extra_space_fn=mobilenet_extra_space,
    )

    study.optimize(objective, n_trials=args.trials, gc_after_trial=True)

    best = study.best_trial
    print("\n=== MobileNetV2 search complete ===")
    print(f"  best trial number: {best.number}")
    print(f"  best val acc:      {best.value:.4f}")
    print(f"  best hyperparams:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    summary = {
        "arch": "mobilenet_v2",
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
