#!/usr/bin/env python3
"""
scripts/benchmark_speed.py
==========================

Inference-speed benchmark for the three best-trial checkpoints.

What it measures
----------------
For each architecture (ResNet50 / EfficientNet-B0 / MobileNetV2):

  latency   -- median / p95 / p99 wall-clock ms for a single image
               (batch_size=1, model.eval(), no grad).
               Single-image latency matters for interactive / per-request use.

  throughput -- images per second at a configurable batch size (default 32).
                Measures how fast the model can chew through a queue of inputs,
                e.g. batch inference on a server.

  model size -- checkpoint .pt file size in MB; a rough proxy for deployment
                footprint when FLOPs / MACs aren't counted.

Why these numbers complement accuracy
--------------------------------------
The project report already has accuracy and parameter count. Speed fills in
the "cost to deploy" dimension: a 1% accuracy drop might be worth it if the
model is 3x faster and fits in half the memory.

Auto-discovery
--------------
The script reads `outputs/<arch>_best_summary.json` (written by the per-arch
train scripts) to find each architecture's best trial number, then resolves
the checkpoint path as:

    <checkpoint_dir>/<arch>/trial_<n>/<arch>_trial<n>_best.pt

If the summary JSON is missing, the script falls back to scanning the
checkpoint directory for the numerically largest trial and using that.

Usage
-----
From the repo root with the venv active:

    python scripts/benchmark_speed.py

    # custom paths / settings
    python scripts/benchmark_speed.py \\
        --checkpoint-dir outputs/checkpoints \\
        --summary-dir    outputs \\
        --manifest       configs/data_split.yaml \\
        --batch-size     64 \\
        --n-repeats      300 \\
        --warmup         50 \\
        --output-dir     outputs/speed_benchmark \\
        --device         cpu

Output
------
  outputs/speed_benchmark/results.json   -- all numbers as JSON
  outputs/speed_benchmark/speed.png      -- throughput bar chart (images/s)
  stdout                                 -- ASCII summary table
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from src.data_prep import load_manifest
from src.models import build_model


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def _find_best_checkpoint(
    arch: str,
    checkpoint_dir: Path,
    summary_dir: Path,
) -> Optional[Path]:
    """Return the path to the best-trial checkpoint for `arch`.

    Tries three strategies in order:
    1. Read the `_best_summary.json` written by the train script.
    2. Scan the checkpoint directory and pick the highest trial number.
    3. Return None (caller will skip this architecture).
    """
    # Strategy 1: trust the summary JSON.
    summary_path = summary_dir / f"{arch}_best_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        trial_n = summary["best_trial_number"]
        ckpt = checkpoint_dir / arch / f"trial_{trial_n}" / f"{arch}_trial{trial_n}_best.pt"
        if ckpt.exists():
            return ckpt

    # Strategy 2: scan for the highest-numbered trial directory.
    arch_dir = checkpoint_dir / arch
    if arch_dir.is_dir():
        trial_dirs = sorted(arch_dir.glob("trial_*"), key=lambda p: int(p.name.split("_")[1]))
        for trial_dir in reversed(trial_dirs):
            candidates = list(trial_dir.glob("*.pt"))
            if candidates:
                return candidates[0]

    return None


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_latency(
    model: nn.Module,
    device: torch.device,
    n_repeats: int,
    warmup: int,
) -> Dict[str, float]:
    """Time single-image (batch=1) forward passes.

    Returns median, p95, and p99 in milliseconds.
    """
    dummy = torch.randn(1, 3, 224, 224, device=device)

    # Warmup: let the runtime JIT-compile kernels and fill caches.
    for _ in range(warmup):
        with torch.no_grad():
            model(dummy)
    _sync(device)

    times_ms: List[float] = []
    with torch.no_grad():
        for _ in range(n_repeats):
            _sync(device)
            t0 = time.perf_counter()
            model(dummy)
            _sync(device)
            times_ms.append((time.perf_counter() - t0) * 1e3)

    arr = np.array(times_ms)
    return {
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
    }


def _time_throughput(
    model: nn.Module,
    device: torch.device,
    batch_size: int,
    n_repeats: int,
    warmup: int,
) -> Dict[str, float]:
    """Time batched forward passes and compute images/second."""
    dummy = torch.randn(batch_size, 3, 224, 224, device=device)

    for _ in range(warmup):
        with torch.no_grad():
            model(dummy)
    _sync(device)

    with torch.no_grad():
        _sync(device)
        t0 = time.perf_counter()
        for _ in range(n_repeats):
            model(dummy)
        _sync(device)
        elapsed = time.perf_counter() - t0

    total_images = batch_size * n_repeats
    return {
        "batch_size": batch_size,
        "images_per_sec": total_images / elapsed,
        "batch_ms": (elapsed / n_repeats) * 1e3,
    }


# ---------------------------------------------------------------------------
# Per-architecture benchmark
# ---------------------------------------------------------------------------

def benchmark_arch(
    arch: str,
    checkpoint_path: Path,
    num_classes: int,
    device: torch.device,
    batch_size: int,
    n_repeats: int,
    warmup: int,
) -> dict:
    built = build_model(arch, num_classes=num_classes, pretrained=False)
    state = torch.load(checkpoint_path, map_location=device)
    built.model.load_state_dict(state["model_state_dict"])
    built.model.to(device)
    built.model.eval()

    latency = _time_latency(built.model, device, n_repeats=n_repeats, warmup=warmup)
    throughput = _time_throughput(built.model, device, batch_size=batch_size, n_repeats=n_repeats, warmup=warmup)

    return {
        "arch": arch,
        "checkpoint": str(checkpoint_path),
        "checkpoint_mb": checkpoint_path.stat().st_size / 1e6,
        "param_count": built.param_count,
        "device": str(device),
        "latency": latency,
        "throughput": throughput,
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _print_table(results: List[dict]) -> None:
    header = f"{'Architecture':<20} {'Params':>8}  {'Ckpt MB':>8}  {'Latency (med ms)':>18}  {'p95 ms':>8}  {'Throughput (img/s)':>20}"
    print()
    print(header)
    print("-" * len(header))
    for r in results:
        lat = r["latency"]
        thr = r["throughput"]
        print(
            f"{r['arch']:<20} "
            f"{r['param_count']/1e6:>7.1f}M  "
            f"{r['checkpoint_mb']:>8.1f}  "
            f"{lat['median_ms']:>18.2f}  "
            f"{lat['p95_ms']:>8.2f}  "
            f"{thr['images_per_sec']:>20.1f}"
        )
    print()


def _save_plot(results: List[dict], output_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["arch"] for r in results]
    throughputs = [r["throughput"]["images_per_sec"] for r in results]
    latencies = [r["latency"]["median_ms"] for r in results]
    colors = ["#4878CF", "#6ACC65", "#D65F5F"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.bar(names, throughputs, color=colors)
    ax1.set_title(f"Throughput (batch={results[0]['throughput']['batch_size']})")
    ax1.set_ylabel("Images / second")
    ax1.set_xlabel("Architecture")
    for i, v in enumerate(throughputs):
        ax1.text(i, v + max(throughputs) * 0.01, f"{v:.0f}", ha="center", fontsize=9)

    ax2.bar(names, latencies, color=colors)
    ax2.set_title("Single-image latency (batch=1)")
    ax2.set_ylabel("Median ms")
    ax2.set_xlabel("Architecture")
    for i, v in enumerate(latencies):
        ax2.text(i, v + max(latencies) * 0.01, f"{v:.1f}", ha="center", fontsize=9)

    fig.suptitle(f"Inference speed comparison — device: {results[0]['device']}", fontsize=11)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  saved speed plot -> {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

ARCHS = ["resnet50", "efficientnet_b0", "mobilenet_v2"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark inference speed of the best checkpoint per architecture.")
    p.add_argument("--checkpoint-dir", default="outputs/checkpoints")
    p.add_argument("--summary-dir",    default="outputs")
    p.add_argument("--manifest",       default="configs/data_split.yaml")
    p.add_argument("--batch-size",     type=int, default=32, help="Batch size used for throughput measurement.")
    p.add_argument("--n-repeats",      type=int, default=200, help="Forward passes per measurement.")
    p.add_argument("--warmup",         type=int, default=20,  help="Warm-up forward passes before timing.")
    p.add_argument("--output-dir",     default="outputs/speed_benchmark")
    p.add_argument("--device",         default=None, help="'cpu' or 'cuda'. Auto-detects if omitted.")
    p.add_argument("--no-plot",        action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(device)}")

    manifest = load_manifest(args.manifest)
    checkpoint_dir = Path(args.checkpoint_dir)
    summary_dir = Path(args.summary_dir)
    output_dir = Path(args.output_dir)

    results = []
    for arch in ARCHS:
        ckpt = _find_best_checkpoint(arch, checkpoint_dir, summary_dir)
        if ckpt is None:
            print(f"  [{arch}] no checkpoint found — skipping")
            continue

        print(f"  [{arch}] benchmarking {ckpt} ...")
        result = benchmark_arch(
            arch=arch,
            checkpoint_path=ckpt,
            num_classes=manifest.num_classes,
            device=device,
            batch_size=args.batch_size,
            n_repeats=args.n_repeats,
            warmup=args.warmup,
        )
        results.append(result)

    if not results:
        print("No checkpoints found. Run the training scripts first.")
        return

    _print_table(results)

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results.json"
    json_path.write_text(json.dumps(results, indent=2))
    print(f"  wrote results -> {json_path}")

    if not args.no_plot:
        _save_plot(results, output_dir / "speed.png")


if __name__ == "__main__":
    main()
