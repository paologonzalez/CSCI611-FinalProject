"""
src/data_prep.py
================

Data pipeline for the food-image-classification project.

Responsibilities
----------------
1. Read the YAML split manifest produced by `scripts/generate_split_manifest.py`.
2. Resolve every image path to an absolute filesystem path, using the
   `dataset_root` field in the YAML interpreted *relative to the YAML file
   itself*. This makes the manifest portable across teammates' machines.
3. Provide a torch `Dataset` for one split (train / val / test).
4. Provide torchvision transform pipelines:
       - training:    resize -> random crop -> random horizontal flip
                      -> color jitter -> ToTensor -> ImageNet normalize
       - val/test:    resize -> center crop -> ToTensor -> ImageNet normalize
5. Provide a single factory `get_dataloaders(...)` that returns DataLoaders
   for all three splits, configured consistently.

Why ImageNet normalization?
   All three architectures we benchmark (ResNet, EfficientNet, MobileNetV2)
   are loaded with ImageNet-pretrained weights. The pretrained convolutional
   filters were fitted to the ImageNet pixel distribution
   (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), so we must
   normalize our food images with those statistics for transfer learning to
   work properly.

Why resize to 256 then crop to 224?
   224x224 is the canonical input size for ImageNet models. Resizing the
   short side to 256 first and then cropping to 224 is the standard recipe
   from the PyTorch model zoo and gives a small "context buffer" so random
   crops at training time still see slightly different views of the image.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import yaml
from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# Some Kaggle PNG files are saved as palette-mode images with a transparency
# byte; PIL emits a warning for each one ("Palette images with Transparency
# expressed in bytes should be converted to RGBA images"). Our pipeline
# converts every image to RGB immediately after opening, so the warning is
# noise -- silencing it keeps the training console readable.
warnings.filterwarnings(
    "ignore",
    message=".*Palette images with Transparency.*",
    category=UserWarning,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ImageNet channel statistics. All torchvision pretrained models expect these.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Default input resolution for the three architectures we use. Each of
# ResNet / EfficientNet-B0 / MobileNetV2 is trained on 224x224 in torchvision.
DEFAULT_IMAGE_SIZE = 224

# Slightly larger pre-crop size; see module docstring for the rationale.
DEFAULT_RESIZE_SIZE = 256


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------

@dataclass
class Manifest:
    """In-memory representation of `configs/data_split.yaml`.

    Attributes
    ----------
    yaml_path:
        Where the manifest was loaded from. Kept around because we resolve
        `dataset_root` relative to this file's parent directory.
    dataset_root:
        Absolute path to the directory that contains the per-class image
        subdirectories.
    seed:
        The seed used when generating the split. Stored only for logging.
    splits:
        Maps split name ("train" / "val" / "test") to a list of dicts of the
        form {"path": ..., "label": ..., "class_idx": ...}.
    class_to_idx:
        Maps human-readable class name to integer index.
    """

    yaml_path: Path
    dataset_root: Path
    seed: int
    splits: Dict[str, List[dict]]
    class_to_idx: Dict[str, int]

    @property
    def num_classes(self) -> int:
        return len(self.class_to_idx)

    @property
    def idx_to_class(self) -> Dict[int, str]:
        # Reverse lookup, used when displaying predictions in human terms.
        return {idx: name for name, idx in self.class_to_idx.items()}


def load_manifest(yaml_path: str | Path) -> Manifest:
    """Read the split manifest and resolve `dataset_root` portably.

    The manifest stores `dataset_root` as a path relative to the YAML file
    itself (e.g. ``../../dataset``). We resolve it here so the rest of the
    pipeline only ever deals with absolute paths.
    """
    yaml_path = Path(yaml_path).resolve()
    with yaml_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Resolve relative to the YAML file's directory, NOT the current working
    # directory. This is the whole point of the relative-path convention: it
    # lets every teammate share the same committed manifest.
    dataset_root = (yaml_path.parent / cfg["dataset_root"]).resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"dataset_root from {yaml_path} resolves to {dataset_root} "
            f"but that path does not exist. Did you place the dataset at "
            f"the expected location (sibling of the repo)?"
        )

    return Manifest(
        yaml_path=yaml_path,
        dataset_root=dataset_root,
        seed=int(cfg.get("seed", 0)),
        splits=cfg["splits"],
        class_to_idx=cfg["class_to_idx"],
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class FoodDataset(Dataset):
    """A `torch.utils.data.Dataset` over one split of the manifest.

    Each item the DataLoader pulls out of this dataset is a tuple of:
        (image_tensor, label_idx)
    where `image_tensor` is shape (3, H, W), already normalized for ImageNet
    pretrained models, and `label_idx` is an int in ``[0, num_classes)``.
    """

    def __init__(
        self,
        manifest: Manifest,
        split: str,
        transform: transforms.Compose,
    ) -> None:
        if split not in manifest.splits:
            raise ValueError(
                f"split={split!r} not in manifest. Available: "
                f"{list(manifest.splits.keys())}"
            )

        self.manifest = manifest
        self.split = split
        self.transform = transform

        # Cache the list so __getitem__ stays O(1).
        self.samples: List[Tuple[Path, int]] = [
            (manifest.dataset_root / entry["path"], int(entry["class_idx"]))
            for entry in manifest.splits[split]
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label_idx = self.samples[idx]

        # Some Kaggle images are saved with non-standard color modes (RGBA,
        # palette, grayscale). Force RGB so every tensor has 3 channels.
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            tensor = self.transform(img)

        return tensor, label_idx


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_train_transforms(image_size: int = DEFAULT_IMAGE_SIZE) -> transforms.Compose:
    """Augmentation pipeline for the *training* split.

    The augmentations are deliberately conservative -- strong augmentations
    can hurt convergence speed when the dataset is moderate size (~24k) and
    we are doing many short Optuna trials. They include:
      * RandomResizedCrop: simulates different framings/zooms of the food.
      * RandomHorizontalFlip: most foods are symmetric horizontally.
      * ColorJitter: handles photo-lighting variation across the dataset.
    """
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_eval_transforms(
    image_size: int = DEFAULT_IMAGE_SIZE,
    resize_size: int = DEFAULT_RESIZE_SIZE,
) -> transforms.Compose:
    """Deterministic pipeline for val and test splits.

    No randomness at evaluation time. Resize the short edge to `resize_size`
    then center-crop to `image_size`, matching the torchvision ImageNet eval
    recipe.
    """
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_dataloaders(
    manifest_path: str | Path,
    batch_size: int = 32,
    image_size: int = DEFAULT_IMAGE_SIZE,
    num_workers: int = 4,
    pin_memory: bool | None = None,
) -> Tuple[Manifest, DataLoader, DataLoader, DataLoader]:
    """Build train / val / test DataLoaders from a manifest path.

    Returns the loaded Manifest as well so callers can read `num_classes`
    and `class_to_idx` without re-reading the YAML.

    Parameters
    ----------
    manifest_path:
        Path to the YAML manifest produced by generate_split_manifest.py.
    batch_size:
        Mini-batch size shared across all three loaders.
    image_size:
        Side length of the square crop fed to the network.
    num_workers:
        DataLoader worker processes. Set to 0 on Windows if you hit
        multiprocessing issues.
    pin_memory:
        Whether DataLoaders should pin host memory for faster GPU transfer.
        If left as None we auto-detect from CUDA availability.
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    manifest = load_manifest(manifest_path)

    train_tf = get_train_transforms(image_size)
    eval_tf = get_eval_transforms(image_size)

    train_ds = FoodDataset(manifest, split="train", transform=train_tf)
    val_ds = FoodDataset(manifest, split="val", transform=eval_tf)
    test_ds = FoodDataset(manifest, split="test", transform=eval_tf)

    # Common loader kwargs. `persistent_workers=True` keeps worker processes
    # alive across epochs, which avoids paying their startup cost every time
    # we restart iteration; only enabled when num_workers > 0.
    common_kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **common_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **common_kwargs)

    return manifest, train_loader, val_loader, test_loader
