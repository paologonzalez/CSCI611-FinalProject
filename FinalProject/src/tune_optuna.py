"""
src/tune_optuna.py
==================

Shared Optuna utilities used by all three per-architecture entry points
(`src/train_resnet.py`, `src/train_efficientnet.py`, `src/train_mobilenet.py`).

What's centralized here
-----------------------
1. **The general search space.** Hyperparameters that apply to every
   architecture -- learning rate, weight decay, optimizer choice, batch
   size, dropout -- are sampled here. Per-arch entry points can extend
   the space (e.g. with arch-specific "freeze early layers" toggles)
   without duplicating the boilerplate.

2. **The objective scaffold.** `make_objective(...)` returns a callable
   that Optuna can `study.optimize` over. It assembles dataloaders, model,
   optimizer, and calls `train.run_training`, returning the best val acc.
   The arch-specific entry point only has to provide the *builder* and
   any extra hyperparameter samplers.

3. **Study creation.** `create_or_load_study(...)` returns a study with
   a sensible default sampler+pruner. We use `MedianPruner` so unpromising
   trials are killed after a few epochs, and `TPESampler` because it
   converges much faster than random search on continuous spaces.

Storage choice
--------------
Studies are persisted to a SQLite file under `outputs/optuna_studies/`.
This means:
  * If a run crashes, you can `study.optimize(...)` again and it picks up.
  * Multiple processes (or even multiple machines, with shared storage)
    can contribute trials to the same study.

For this project each teammate runs their own arch on their own machine,
so each gets its own SQLite file. No cross-machine sync required.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import optuna
import torch
import torch.nn as nn

from src.data_prep import get_dataloaders
from src.models import BuiltModel, build_model
from src.train import TrainingResult, run_training


# ---------------------------------------------------------------------------
# Common search space
# ---------------------------------------------------------------------------

@dataclass
class CommonSearchSpace:
    """The hyperparameters every architecture searches over.

    These mirror the standard tuning levers in image classification:
      * learning rate (log scale, by far the most important)
      * weight decay (log scale, regularization)
      * optimizer choice (SGD-momentum vs AdamW)
      * batch size (categorical, hardware-bounded)
      * dropout in the new classifier head
    """

    lr_min: float = 1e-5
    lr_max: float = 1e-2
    weight_decay_min: float = 1e-6
    weight_decay_max: float = 1e-3
    dropout_min: float = 0.0
    dropout_max: float = 0.5
    batch_sizes: tuple = (16, 32, 64)
    optimizer_choices: tuple = ("adamw", "sgd")


def sample_common_hparams(
    trial: optuna.Trial,
    space: CommonSearchSpace,
) -> Dict[str, Any]:
    """Draw one hyperparameter configuration from the common space.

    Returns a plain dict so callers can log it (e.g. into the checkpoint
    metadata) without having to round-trip through Optuna again.
    """
    return {
        # Log-uniform is correct for learning rate / weight decay because
        # we care about order-of-magnitude differences (1e-3 vs 1e-4)
        # rather than additive ones.
        "lr": trial.suggest_float("lr", space.lr_min, space.lr_max, log=True),
        "weight_decay": trial.suggest_float(
            "weight_decay", space.weight_decay_min, space.weight_decay_max, log=True
        ),
        "dropout": trial.suggest_float(
            "dropout", space.dropout_min, space.dropout_max
        ),
        "batch_size": trial.suggest_categorical("batch_size", list(space.batch_sizes)),
        "optimizer": trial.suggest_categorical("optimizer", list(space.optimizer_choices)),
    }


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------

def build_optimizer(
    model: nn.Module,
    name: str,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Translate the optimizer name from the search space to a real instance.

    For SGD we set momentum=0.9 (a near-universal default).

    Only parameters with `requires_grad=True` are passed to the optimizer.
    This matters when the per-arch script asks Optuna to "freeze the
    backbone": those frozen parameters should not appear in the optimizer's
    parameter group at all.
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise ValueError(
            "No trainable parameters found. Did you accidentally freeze "
            "everything (including the classifier head)?"
        )

    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            trainable,
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay,
            nesterov=True,
        )
    raise ValueError(f"Unknown optimizer: {name!r}")


# ---------------------------------------------------------------------------
# Optional-extra-hyperparameter handlers
# ---------------------------------------------------------------------------
# The per-arch entry points use `extra_space_fn` to add knobs like
# label_smoothing, cosine_lr_schedule, and freeze_backbone. Each of these
# needs to actually take effect somewhere in the training pipeline. The
# helpers below recognize those well-known keys and apply them. If a key is
# absent from `hparams`, the helper is a no-op -- so per-arch scripts only
# pay for the knobs they actually sample.

def _freeze_backbone_if_requested(
    model: nn.Module,
    hparams: Dict[str, Any],
) -> None:
    """If `freeze_backbone=True`, set requires_grad=False on every parameter
    that isn't part of the classifier head.

    The classifier head in our three architectures is named either
    `fc` (ResNet) or `classifier` (EfficientNet, MobileNetV2). We freeze
    everything else.
    """
    if not hparams.get("freeze_backbone", False):
        return
    for name, param in model.named_parameters():
        if not (name.startswith("fc.") or name.startswith("classifier.")):
            param.requires_grad = False


def _build_criterion(hparams: Dict[str, Any]) -> nn.Module:
    """Construct the loss function, using `label_smoothing` if sampled.

    `label_smoothing=0.0` is mathematically equivalent to plain
    cross-entropy, so this is safe to call unconditionally.
    """
    label_smoothing = float(hparams.get("label_smoothing", 0.0))
    return nn.CrossEntropyLoss(label_smoothing=label_smoothing)


def _build_scheduler(
    optimizer: torch.optim.Optimizer,
    hparams: Dict[str, Any],
    epochs: int,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Construct an LR scheduler if `cosine_lr_schedule=True` was sampled.

    Cosine annealing decays the learning rate smoothly from `lr` to ~0 over
    `epochs` epochs. A common improvement on a constant LR for fine-tuning.
    """
    if hparams.get("cosine_lr_schedule", False):
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs, 1)
        )
    return None


# ---------------------------------------------------------------------------
# Study creation
# ---------------------------------------------------------------------------

def create_or_load_study(
    study_name: str,
    storage_dir: str | Path = "outputs/optuna_studies",
    direction: str = "maximize",
    seed: int = 42,
) -> optuna.Study:
    """Create (or resume) an Optuna study persisted to SQLite.

    Sampler: TPE -- a tree-structured Parzen estimator that handles mixed
    continuous/categorical spaces and beats random search by a wide margin
    after ~10 trials.

    Pruner: MedianPruner -- after `n_startup_trials` warm-up trials,
    prunes any trial whose intermediate value at step `s` is below the
    median of completed trials at the same step.
    """
    storage_dir = Path(storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)
    storage_url = f"sqlite:///{storage_dir / f'{study_name}.db'}"

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,    # gather some baselines before pruning anyone
        n_warmup_steps=2,      # let each trial show 2 epochs before pruning
    )

    return optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,    # resume on rerun; never overwrite history
        direction=direction,
        sampler=sampler,
        pruner=pruner,
    )


# ---------------------------------------------------------------------------
# Objective scaffold
# ---------------------------------------------------------------------------

def make_objective(
    arch: str,
    manifest_path: str | Path,
    *,
    num_workers: int = 4,
    epochs_per_trial: int = 8,
    checkpoint_dir: Optional[str | Path] = None,
    common_space: Optional[CommonSearchSpace] = None,
    extra_space_fn: Optional[Callable[[optuna.Trial], Dict[str, Any]]] = None,
) -> Callable[[optuna.Trial], float]:
    """Build the objective callable for `study.optimize(...)`.

    Parameters
    ----------
    arch:
        One of the names registered in `src.models.ARCHITECTURES`.
    manifest_path:
        Path to the YAML produced by `generate_split_manifest.py`.
    num_workers:
        DataLoader workers to use for *every* trial.
    epochs_per_trial:
        Each Optuna trial trains for at most this many epochs.
        Pruning may end the trial sooner.
    checkpoint_dir:
        Where to save best-by-val-acc checkpoints. Each trial gets its
        own subdirectory keyed by trial number.
    common_space:
        Knobs for the shared search space; defaults are reasonable.
    extra_space_fn:
        Optional hook for the per-architecture script to add more
        hyperparameters (e.g., LR schedule, freeze-early-layers toggle).
        Must take a Trial and return a dict; those entries are merged
        into the metadata dict written to the checkpoint.

    Returns
    -------
    A callable suitable for `study.optimize(...)`. It returns the best
    validation accuracy seen during the trial (which Optuna will maximize).
    """
    if common_space is None:
        common_space = CommonSearchSpace()

    def objective(trial: optuna.Trial) -> float:
        # 1. Sample a hyperparameter configuration.
        common_hp = sample_common_hparams(trial, common_space)
        extra_hp = extra_space_fn(trial) if extra_space_fn is not None else {}
        hparams = {**common_hp, **extra_hp}

        # 2. Build the data pipeline. We rebuild loaders here (not outside
        #    the objective) because batch_size is a sampled hyperparameter
        #    and so it varies across trials.
        manifest, train_loader, val_loader, _ = get_dataloaders(
            manifest_path=manifest_path,
            batch_size=hparams["batch_size"],
            num_workers=num_workers,
        )

        # 3. Build the model with the sampled dropout, on the right device.
        built: BuiltModel = build_model(
            arch=arch,
            num_classes=manifest.num_classes,
            dropout=hparams["dropout"],
            pretrained=True,
        )

        # 3a. Apply optional per-arch extras BEFORE building the optimizer
        #     so frozen params are excluded from optimization.
        _freeze_backbone_if_requested(built.model, hparams)

        # 4. Build the optimizer using sampled lr/optimizer/weight_decay.
        optimizer = build_optimizer(
            built.model,
            name=hparams["optimizer"],
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"],
        )

        # 4a. Loss function (honors label_smoothing if it was sampled).
        criterion = _build_criterion(hparams)

        # 4b. Optional LR scheduler (cosine annealing if sampled).
        scheduler = _build_scheduler(optimizer, hparams, epochs=epochs_per_trial)

        # 5. Run training. If the trial gets pruned, run_training raises
        #    `optuna.TrialPruned`, which Optuna catches and records.
        per_trial_dir: Optional[Path] = None
        if checkpoint_dir is not None:
            per_trial_dir = Path(checkpoint_dir) / f"trial_{trial.number}"

        result: TrainingResult = run_training(
            model=built.model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs_per_trial,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            checkpoint_dir=per_trial_dir,
            checkpoint_tag=f"{arch}_trial{trial.number}",
            optuna_trial=trial,
            extra_metadata={
                "arch": arch,
                "hparams": hparams,
                "param_count": built.param_count,
            },
        )

        return result.best_val_acc

    return objective
