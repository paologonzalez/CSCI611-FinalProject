"""
src/models.py
=============

Model factories for the three architectures we benchmark in this project:

    * ResNet50            -- "Heavy" / performance ceiling.
    * EfficientNet-B0     -- "Intermediate", compound-scaled.
    * MobileNetV2         -- "Light" / efficiency baseline.

Each factory:
  1. Loads the architecture from torchvision with optional ImageNet-pretrained
     weights (the default; the whole point of using these architectures is
     transfer learning -- training from scratch on 24k images would severely
     under-perform).
  2. Replaces the final classification layer so the output dimension matches
     our number of food classes (34 in the current dataset).
  3. Optionally inserts dropout before the final layer so we can tune
     regularization with Optuna.

Each factory also returns metadata that downstream tools need:
  * `param_count`        -- a quick sanity check that "heavy < intermediate
                            < light" actually holds.
  * `gradcam_target_layer`-- the name (in `model.named_modules()`) of the
                            final convolutional block. Grad-CAM hooks this
                            layer to produce attention heatmaps.

Why expose `gradcam_target_layer` here?
   The "right" target layer depends on the architecture. If we hard-coded
   one in `gradcam.py`, swapping architectures would silently produce
   meaningless heatmaps. Centralizing it next to the model definition keeps
   the two in sync.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V2_Weights,
    ResNet50_Weights,
    efficientnet_b0,
    mobilenet_v2,
    resnet50,
)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class BuiltModel:
    """Bundle of (model, metadata) returned by every builder.

    `name`:
        Short string ID used in checkpoints / log filenames / plots.
    `model`:
        The actual `nn.Module`, ready to .to(device) and train.
    `param_count`:
        Total trainable parameters. Reported in the per-arch results table.
    `gradcam_target_layer`:
        Dot-path inside the model of the final feature-extracting conv
        block. Grad-CAM uses this to compute class-discriminative heatmaps.
    """

    name: str
    model: nn.Module
    param_count: int
    gradcam_target_layer: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_trainable_params(model: nn.Module) -> int:
    """Sum the number of parameters that have requires_grad=True.

    This is what people usually mean by "model size" when comparing
    architectures, since fixed (frozen) parameters don't contribute to
    training cost or to overfitting capacity.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _build_classifier_head(
    in_features: int,
    num_classes: int,
    dropout: float,
) -> nn.Module:
    """Construct the final classification block.

    All three architectures share the same head shape:
        Dropout -> Linear(in_features -> num_classes)

    Why a Dropout layer here even though most of these models already use
    dropout internally? The torchvision implementations vary, and exposing
    one explicit knob lets Optuna tune regularization uniformly across the
    three architectures.
    """
    return nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_resnet(num_classes: int, dropout: float = 0.0, pretrained: bool = True) -> BuiltModel:
    """Build ResNet50 -- our "heavy" / accuracy-ceiling model.

    ResNet50 is the deepest of our three architectures (~25M parameters).
    We pick it as the upper bound on what a moderate-depth CNN can do on
    this dataset; if the lighter networks come close to its accuracy, that
    tells us the task does not really require the extra capacity.
    """
    weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = resnet50(weights=weights)

    # ResNet's final layer is `model.fc`: a Linear(2048 -> 1000) trained on
    # ImageNet. Replace it with our food-classification head.
    in_features = model.fc.in_features
    model.fc = _build_classifier_head(in_features, num_classes, dropout)

    return BuiltModel(
        name="resnet50",
        model=model,
        param_count=_count_trainable_params(model),
        # `layer4` is the last residual stage. Its output spatial size is
        # 7x7 for a 224x224 input, which is the standard Grad-CAM target.
        gradcam_target_layer="layer4",
    )


def build_efficientnet(num_classes: int, dropout: float = 0.2, pretrained: bool = True) -> BuiltModel:
    """Build EfficientNet-B0 -- the "intermediate" architecture.

    EfficientNet uses compound scaling (depth, width, resolution scaled
    together by a single factor) and inverted residual + squeeze-excite
    blocks. B0 is the smallest variant (~5M parameters) and is the natural
    middle-ground choice between MobileNetV2 and ResNet50 for this study.
    """
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)

    # EfficientNet's classifier is a Sequential ending in
    #     Dropout -> Linear(1280 -> 1000)
    # Index [1] is the Linear; we read its in_features and rebuild the head.
    in_features = model.classifier[1].in_features
    model.classifier = _build_classifier_head(in_features, num_classes, dropout)

    return BuiltModel(
        name="efficientnet_b0",
        model=model,
        param_count=_count_trainable_params(model),
        # `features[-1]` is the last conv block (the one before the global
        # pool). Its output is what we want Grad-CAM to attribute to.
        gradcam_target_layer="features.8",
    )


def build_mobilenet(num_classes: int, dropout: float = 0.2, pretrained: bool = True) -> BuiltModel:
    """Build MobileNetV2 -- our "light" / efficiency baseline.

    MobileNetV2 is built around depthwise-separable convolutions and
    inverted residual blocks, optimized for inference on mobile and edge
    hardware (~3.5M parameters). It is the fastest of the three at
    inference time and the smallest on disk; if it nearly matches the
    other two, it wins the cost/quality trade-off the project asks about.
    """
    weights = MobileNet_V2_Weights.IMAGENET1K_V2 if pretrained else None
    model = mobilenet_v2(weights=weights)

    # MobileNetV2's classifier is also Sequential:
    #     Dropout -> Linear(1280 -> 1000)
    in_features = model.classifier[1].in_features
    model.classifier = _build_classifier_head(in_features, num_classes, dropout)

    return BuiltModel(
        name="mobilenet_v2",
        model=model,
        param_count=_count_trainable_params(model),
        # `features[-1]` is the final 1x1 conv expansion before the pool.
        gradcam_target_layer="features.18",
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Mapping from short name -> builder function. Used by tune_optuna.py and
# evaluate.py so they can pick a model by string without import gymnastics.
ARCHITECTURES: dict[str, Callable[..., BuiltModel]] = {
    "resnet50": build_resnet,
    "efficientnet_b0": build_efficientnet,
    "mobilenet_v2": build_mobilenet,
}


def build_model(arch: str, num_classes: int, dropout: float = 0.0, pretrained: bool = True) -> BuiltModel:
    """Look up an architecture by name and call its builder.

    Raises a KeyError with a friendly message if the architecture is unknown.
    """
    if arch not in ARCHITECTURES:
        raise KeyError(
            f"Unknown architecture {arch!r}. Available: {sorted(ARCHITECTURES)}"
        )
    return ARCHITECTURES[arch](num_classes=num_classes, dropout=dropout, pretrained=pretrained)


# ---------------------------------------------------------------------------
# Module resolution helper
# ---------------------------------------------------------------------------

def get_submodule_by_name(model: nn.Module, dotted_name: str) -> nn.Module:
    """Resolve a dotted attribute path inside a model.

    Used by Grad-CAM to walk e.g. ``"features.8"`` -> ``model.features[8]``.
    Supports both attribute access (``layer4`` -> ``model.layer4``) and
    integer indexing into ``nn.Sequential`` / ``nn.ModuleList``.
    """
    submodule: nn.Module = model
    for part in dotted_name.split("."):
        if part.isdigit():
            submodule = submodule[int(part)]  # type: ignore[index]
        else:
            submodule = getattr(submodule, part)
    return submodule
