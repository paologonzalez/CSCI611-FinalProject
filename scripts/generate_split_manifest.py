#!/usr/bin/env python3
"""
Generate a reproducible train/val/test split manifest for an image-classification dataset.

Expected dataset layout:
dataset_root/
    Class A/
        img1.jpg
        img2.png
    Class B/
        img3.jpeg
        ...

Outputs a YAML file like:
dataset_root: data/raw
seed: 42
splits:
  train:
    - path: Class A/img1.jpg
      label: Class A
  val:
    - path: Class B/img7.jpg
      label: Class B
  test:
    - path: Class A/img9.jpg
      label: Class A
class_to_idx:
  Class A: 0
  Class B: 1
"""

from __future__ import annotations

import argparse
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a YAML manifest for train/val/test image splits."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Root directory containing one subdirectory per class.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="configs/data_split.yaml",
        help="Output YAML file path.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fraction of each class used for training.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fraction of each class used for validation.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Fraction of each class used for testing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits.",
    )
    parser.add_argument(
        "--copy-existing-names",
        action="store_true",
        help=(
            "Optional flag if you want to preserve files whose names already suggest "
            "'Train' or 'Test'. By default, filenames are ignored and split is done fresh."
        ),
    )
    return parser.parse_args()


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total:.6f} "
            f"(train={train_ratio}, val={val_ratio}, test={test_ratio})."
        )


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def collect_images(dataset_root: Path) -> Dict[str, List[Path]]:
    """
    Returns a dict:
        {
            "class_name": [absolute_path1, absolute_path2, ...],
            ...
        }
    """
    class_to_images: Dict[str, List[Path]] = defaultdict(list)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    class_dirs = [p for p in dataset_root.iterdir() if p.is_dir()]
    if not class_dirs:
        raise ValueError(
            f"No class subdirectories found under dataset root: {dataset_root}"
        )

    for class_dir in sorted(class_dirs):
        class_name = class_dir.name
        images = sorted(
            [p for p in class_dir.rglob("*") if is_image_file(p)]
        )
        if images:
            class_to_images[class_name].extend(images)

    if not class_to_images:
        raise ValueError(f"No image files found under: {dataset_root}")

    return dict(class_to_images)


def split_class_images(
    images: List[Path],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    rng: random.Random,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """
    Split one class into train/val/test while guaranteeing at least one sample
    in non-empty splits whenever possible.
    """
    n = len(images)
    shuffled = images[:]
    rng.shuffle(shuffled)

    if n == 1:
        return shuffled, [], []
    if n == 2:
        return [shuffled[0]], [], [shuffled[1]]

    train_count = int(n * train_ratio)
    val_count = int(n * val_ratio)
    test_count = n - train_count - val_count

    # Ensure no empty test split if test_ratio > 0 and n is large enough
    if test_ratio > 0 and test_count == 0:
        test_count = 1
        if train_count > 1:
            train_count -= 1
        elif val_count > 0:
            val_count -= 1

    # Ensure no empty val split if val_ratio > 0 and enough samples exist
    if val_ratio > 0 and val_count == 0 and n >= 5:
        val_count = 1
        if train_count > 1:
            train_count -= 1
        elif test_count > 1:
            test_count -= 1

    # Final safety
    if train_count <= 0:
        train_count = max(1, n - val_count - test_count)

    train = shuffled[:train_count]
    val = shuffled[train_count:train_count + val_count]
    test = shuffled[train_count + val_count:]

    return train, val, test


def build_manifest(
    dataset_root: Path,
    class_to_images: Dict[str, List[Path]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict:
    rng = random.Random(seed)

    manifest = {
        "dataset_root": str(dataset_root),
        "seed": seed,
        "splits": {
            "train": [],
            "val": [],
            "test": [],
        },
        "class_to_idx": {},
        "summary": {
            "num_classes": 0,
            "total_images": 0,
            "train_images": 0,
            "val_images": 0,
            "test_images": 0,
            "per_class_counts": {},
        },
    }

    class_names = sorted(class_to_images.keys())
    manifest["class_to_idx"] = {name: idx for idx, name in enumerate(class_names)}
    manifest["summary"]["num_classes"] = len(class_names)

    for class_name in class_names:
        images = class_to_images[class_name]
        train_imgs, val_imgs, test_imgs = split_class_images(
            images=images,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            rng=rng,
        )

        manifest["summary"]["per_class_counts"][class_name] = {
            "total": len(images),
            "train": len(train_imgs),
            "val": len(val_imgs),
            "test": len(test_imgs),
        }

        for split_name, split_paths in (
            ("train", train_imgs),
            ("val", val_imgs),
            ("test", test_imgs),
        ):
            for img_path in split_paths:
                manifest["splits"][split_name].append(
                    {
                        "path": str(img_path.relative_to(dataset_root)),
                        "label": class_name,
                        "class_idx": manifest["class_to_idx"][class_name],
                    }
                )

    manifest["summary"]["total_images"] = sum(
        len(v) for v in class_to_images.values()
    )
    manifest["summary"]["train_images"] = len(manifest["splits"]["train"])
    manifest["summary"]["val_images"] = len(manifest["splits"]["val"])
    manifest["summary"]["test_images"] = len(manifest["splits"]["test"])

    return manifest


def save_yaml(data: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)


def main() -> None:
    args = parse_args()
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    dataset_root = Path(args.dataset_root).resolve()
    output_path = Path(args.output).resolve()

    class_to_images = collect_images(dataset_root)
    manifest = build_manifest(
        dataset_root=dataset_root,
        class_to_images=class_to_images,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    manifest["dataset_root"] = os.path.relpath(dataset_root, output_path.parent)

    save_yaml(manifest, output_path)

    print(f"Saved split manifest to: {output_path}")
    print("--- Summary ---")
    print(f"Classes: {manifest['summary']['num_classes']}")
    print(f"Total images: {manifest['summary']['total_images']}")
    print(f"Train: {manifest['summary']['train_images']}")
    print(f"Val:   {manifest['summary']['val_images']}")
    print(f"Test:  {manifest['summary']['test_images']}")


if __name__ == "__main__":
    main()