"""
src/train.py
============

Shared training-loop machinery used by all three per-architecture entry
points (`src/train_resnet.py`, `src/train_efficientnet.py`,
`src/train_mobilenet.py`).

This file *intentionally* contains no architecture-specific code. The
per-arch entry points are responsible for:
    1. Building the model (via `src/models.py`).
    2. Defining the Optuna search space.
    3. Calling `run_training(...)` from this file with concrete settings.

That separation is what makes parallel training across teammates' machines
clean: each teammate runs one of the three entry points, but they all
share the same training/eval/checkpointing logic so results are
apples-to-apples.

Optuna integration
------------------
`run_training` accepts an optional `optuna_trial`. When provided:
    * After every epoch we report the current val accuracy via
      `trial.report(val_acc, epoch)`.
    * If the Optuna pruner decides this trial is unpromising, we raise
      `optuna.TrialPruned`, ending the run early and saving wall-clock time.

This pruning-aware loop is critical: with three teammates each running ~20
trials, naive training (no early stopping, no pruning) would burn far more
GPU/CPU hours than the project budget allows.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Optuna is imported lazily inside functions that touch it. This keeps
# `import src.train` working in environments where Optuna isn't installed
# (e.g., a teammate just running evaluation on a saved checkpoint).


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TrainingResult:
    """What `run_training` returns.

    `history` is a list of per-epoch dicts: train_loss, train_acc,
    val_loss, val_acc, epoch_seconds. Useful for plotting learning curves.

    `best_val_acc` is the maximum val accuracy seen across all epochs.

    `best_checkpoint_path` is where the best-by-val-acc weights were saved
    (if `checkpoint_dir` was provided to `run_training`).
    """

    best_val_acc: float
    best_epoch: int
    history: List[Dict[str, float]] = field(default_factory=list)
    best_checkpoint_path: Optional[Path] = None


# ---------------------------------------------------------------------------
# Per-epoch primitives
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    progress_label: str = "train",
) -> Dict[str, float]:
    """Run a single training pass over `loader`.

    Returns a dict with `loss` (mean over all samples) and `acc` (top-1).
    """
    model.train()  # enables dropout, batchnorm running-stat updates, etc.

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    # Wrap the loader in tqdm so the user sees progress per batch. Optuna
    # trials are usually short, so a per-batch bar is more informative than
    # a once-per-epoch print.
    for images, labels in tqdm(loader, desc=progress_label, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Standard supervised step: zero grads -> forward -> loss ->
        # backward -> optimizer step.
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        # Aggregate stats. We weight by batch size so the final mean is
        # correct even on the last (possibly smaller) batch.
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_seen += batch_size

    return {
        "loss": total_loss / max(total_seen, 1),
        "acc": total_correct / max(total_seen, 1),
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    progress_label: str = "val",
) -> Dict[str, float]:
    """Run a single evaluation pass with grads disabled.

    `@torch.no_grad()` is critical: it disables autograd bookkeeping which
    cuts memory use roughly in half and speeds up the pass.
    """
    model.eval()  # disables dropout, freezes batchnorm running stats.

    total_loss = 0.0
    total_correct = 0
    total_seen = 0

    for images, labels in tqdm(loader, desc=progress_label, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_seen += batch_size

    return {
        "loss": total_loss / max(total_seen, 1),
        "acc": total_correct / max(total_seen, 1),
    }


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------

def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    criterion: Optional[nn.Module] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: Optional[torch.device] = None,
    checkpoint_dir: Optional[str | Path] = None,
    checkpoint_tag: str = "model",
    optuna_trial: Optional[Any] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> TrainingResult:
    """Train `model` for `epochs` epochs, tracking the best val accuracy.

    Parameters
    ----------
    model:
        The network to train. Already constructed (e.g. via models.build_*).
    train_loader / val_loader:
        DataLoaders from `data_prep.get_dataloaders`.
    epochs:
        Number of epochs to run. Each Optuna trial typically uses a small
        number (5--15) so the search can cover many configurations.
    optimizer:
        e.g. `torch.optim.AdamW(model.parameters(), lr=...)`.
    criterion:
        Loss function. Defaults to `nn.CrossEntropyLoss()` for multi-class.
    scheduler:
        Optional LR scheduler. Stepped once per epoch after validation.
    device:
        torch device. Defaults to CUDA if available else CPU.
    checkpoint_dir:
        If given, save the best weights (by val acc) to this directory.
    checkpoint_tag:
        Filename stem inside `checkpoint_dir`, e.g. "resnet50_trial7".
    optuna_trial:
        If given, report intermediate val accuracy and honor pruning
        decisions. Pass through whatever you got from your Optuna objective.
    extra_metadata:
        Free-form dict written next to the checkpoint. Use it to record the
        hyperparameter values for the run so it's reproducible from disk.
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Track the best val accuracy across all epochs. We save weights
    # whenever a new best is found, so even if we crash mid-run the best
    # checkpoint is already on disk.
    best_val_acc = -1.0
    best_epoch = -1
    history: List[Dict[str, float]] = []
    best_checkpoint_path: Optional[Path] = None

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        epoch_start = time.time()

        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            progress_label=f"epoch {epoch + 1}/{epochs} train",
        )
        val_stats = validate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            progress_label=f"epoch {epoch + 1}/{epochs} val",
        )

        # Step LR scheduler once per epoch. Some schedulers (like
        # ReduceLROnPlateau) need the val metric; we pass val_loss in that
        # case via duck-typing.
        if scheduler is not None:
            try:
                scheduler.step(val_stats["loss"])  # ReduceLROnPlateau-style
            except TypeError:
                scheduler.step()  # most other schedulers

        epoch_seconds = time.time() - epoch_start
        record = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "train_acc": train_stats["acc"],
            "val_loss": val_stats["loss"],
            "val_acc": val_stats["acc"],
            "epoch_seconds": epoch_seconds,
        }
        history.append(record)

        # Human-readable summary line. Easier to scan a tail -f of the log
        # than the per-batch tqdm bars.
        print(
            f"[epoch {epoch + 1}/{epochs}] "
            f"train_loss={train_stats['loss']:.4f} "
            f"train_acc={train_stats['acc']:.4f} "
            f"val_loss={val_stats['loss']:.4f} "
            f"val_acc={val_stats['acc']:.4f} "
            f"({epoch_seconds:.1f}s)"
        )

        # Save best-by-val-acc checkpoint.
        if val_stats["acc"] > best_val_acc:
            best_val_acc = val_stats["acc"]
            best_epoch = epoch
            if checkpoint_dir is not None:
                best_checkpoint_path = checkpoint_dir / f"{checkpoint_tag}_best.pt"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "epoch": epoch,
                        "val_acc": val_stats["acc"],
                        "extra_metadata": extra_metadata or {},
                    },
                    best_checkpoint_path,
                )
                # Also dump the metadata as JSON next to the checkpoint so
                # humans can read it without loading torch.
                meta_path = checkpoint_dir / f"{checkpoint_tag}_best.json"
                with meta_path.open("w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "epoch": epoch,
                            "val_acc": val_stats["acc"],
                            "extra_metadata": extra_metadata or {},
                            "history_so_far": history,
                        },
                        f,
                        indent=2,
                    )

        # Optuna integration: report and possibly prune.
        if optuna_trial is not None:
            # Lazy import so this module imports cleanly without optuna.
            import optuna

            optuna_trial.report(val_stats["acc"], step=epoch)
            if optuna_trial.should_prune():
                # Pruning ends the trial early; the study uses this trial
                # as a "negative example" for the search.
                raise optuna.TrialPruned(
                    f"Pruned at epoch {epoch} with val_acc={val_stats['acc']:.4f}"
                )

    return TrainingResult(
        best_val_acc=best_val_acc,
        best_epoch=best_epoch,
        history=history,
        best_checkpoint_path=best_checkpoint_path,
    )
