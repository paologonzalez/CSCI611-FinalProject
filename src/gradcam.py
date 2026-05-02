"""
src/gradcam.py
==============

Grad-CAM (Gradient-weighted Class Activation Mapping) for our trained
food-classification models.

Why Grad-CAM matters for this project
-------------------------------------
The proposal/pitch promise a *visualization* component of the comparison.
A confusion matrix tells us *which* classes a model gets wrong; Grad-CAM
tells us *why* by highlighting the image regions that drove a particular
prediction. If MobileNetV2 routinely focuses on background plates while
ResNet50 focuses on the food itself, that's a story worth telling in the
report -- and one that pure accuracy numbers would hide.

Algorithm in one paragraph
--------------------------
For a target class c and a chosen convolutional layer L:
  1. Forward the image through the network and record L's output
     activations A (shape: channels x H x W).
  2. Compute the gradient of the target class score with respect to A.
  3. Average that gradient over spatial dims to get a per-channel weight.
  4. Take the weighted sum of A across channels -> a single H x W map.
  5. ReLU + normalize. That's the heatmap. Upsample to the input
     resolution and overlay on the original image.

Implementation notes
--------------------
We register a forward hook (to capture the activation tensor A) and a
backward hook (to capture the gradient with respect to A). The two hooks
run for any single inference; we then do the weighting math on CPU/GPU
tensors. The hooks are removed at the end so the model is left clean.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from src.data_prep import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    get_eval_transforms,
    load_manifest,
)
from src.models import build_model, get_submodule_by_name


# ---------------------------------------------------------------------------
# Core Grad-CAM class
# ---------------------------------------------------------------------------

class GradCAM:
    """Grad-CAM heatmap generator for one (model, target_layer) pair.

    Use as:
        cam = GradCAM(model, target_layer)
        heatmap = cam(image_tensor, target_class=42)
        cam.remove_hooks()

    Or with the convenience function `compute_gradcam(...)` below, which
    handles hook lifecycle for you.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer

        # Cache for the forward activation and its gradient. Filled in by
        # the hooks during forward/backward.
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        # Register hooks. `register_forward_hook` runs after the layer's
        # forward pass; `register_full_backward_hook` runs during the
        # backward pass with the gradient flowing into the layer's output.
        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        # `output` is the activation A. Detach so we don't accidentally
        # build a second graph through this cached tensor.
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        # `grad_output[0]` is dL/dA -- the gradient of the loss with
        # respect to A. That's what we'll average across spatial dims.
        self._gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        """Detach the hooks. Call this when you're done with the GradCAM."""
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def __call__(
        self,
        image_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int]:
        """Compute the heatmap for `image_tensor`.

        `image_tensor` must be shape (1, 3, H, W), already normalized. If
        you have a (3, H, W) tensor, unsqueeze a batch dim first.

        `target_class`:
            If None, we use the model's argmax prediction (i.e., explain
            the prediction the model actually made). Pass an int to force
            attribution for a specific class.

        Returns
        -------
        heatmap:
            np.ndarray of shape (H, W), values in [0, 1], same spatial
            dims as the input image_tensor.
        target_class:
            The class index that was attributed (echoes the argument or
            the predicted class).
        """
        if image_tensor.dim() != 4 or image_tensor.size(0) != 1:
            raise ValueError(
                "GradCAM expects a single-image batch tensor of shape "
                f"(1, 3, H, W); got {tuple(image_tensor.shape)}."
            )

        # We need gradients flowing through the model, but we don't want
        # to update parameters -- so put the model in eval mode and zero
        # any old grads. `torch.enable_grad()` is harmless inside an
        # outer `no_grad`, so this also works from notebooks.
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            logits = self.model(image_tensor)  # (1, num_classes)
            if target_class is None:
                target_class = int(logits.argmax(dim=1).item())

            # Backprop just the target class score. We don't need a
            # full softmax/CE loss; the gradient of one output element
            # w.r.t. A is what Grad-CAM uses.
            score = logits[0, target_class]
            score.backward()

        if self._activations is None or self._gradients is None:
            raise RuntimeError("Forward/backward did not populate hooks.")

        activations = self._activations[0]  # (C, H', W')
        gradients = self._gradients[0]      # (C, H', W')

        # Weight each channel by its mean gradient (the alpha_k in the
        # Grad-CAM paper).
        channel_weights = gradients.mean(dim=(1, 2))    # (C,)

        # Weighted sum across channels -> coarse class-activation map.
        cam = (channel_weights[:, None, None] * activations).sum(dim=0)  # (H', W')

        # ReLU: only positive evidence supports the class.
        cam = F.relu(cam)

        # Upsample to input image's spatial size (e.g. 224x224).
        cam = cam.unsqueeze(0).unsqueeze(0)  # (1,1,H',W') for interpolate
        cam = F.interpolate(
            cam,
            size=image_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1] for visualization.
        cam_min, cam_max = float(cam.min()), float(cam.max())
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            # Constant heatmap (rare but possible if every gradient is 0).
            cam = np.zeros_like(cam)

        return cam, target_class


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

def compute_gradcam(
    model: nn.Module,
    image_tensor: torch.Tensor,
    target_layer_name: str,
    target_class: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """One-shot Grad-CAM. Handles hook setup/teardown so you don't have to.

    `target_layer_name` is the dotted name from
    `BuiltModel.gradcam_target_layer` (e.g. ``"layer4"`` for ResNet).
    """
    target_layer = get_submodule_by_name(model, target_layer_name)
    cam = GradCAM(model, target_layer)
    try:
        return cam(image_tensor, target_class=target_class)
    finally:
        cam.remove_hooks()


def overlay_heatmap_on_image(
    pil_image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    """Blend a Grad-CAM heatmap onto a PIL image.

    The original image is kept underneath at full opacity; the heatmap is
    rendered as a translucent jet colormap on top. We resize the heatmap
    to the image's exact dimensions in case the input was cropped to a
    different size than the original.
    """
    # Lazy import: only needed when actually saving figures.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.cm as cm

    pil_rgb = pil_image.convert("RGB")
    heatmap_resized = np.array(
        Image.fromarray((heatmap * 255).astype(np.uint8)).resize(pil_rgb.size, Image.BILINEAR)
    ) / 255.0

    # `cm.jet` returns RGBA; drop alpha and scale to 0-255.
    colored = (cm.jet(heatmap_resized)[..., :3] * 255).astype(np.uint8)
    colored_pil = Image.fromarray(colored)

    return Image.blend(pil_rgb, colored_pil, alpha=alpha)


def denormalize_for_display(
    image_tensor: torch.Tensor,
) -> Image.Image:
    """Reverse the ImageNet normalization so we can display the input.

    `image_tensor` is expected to be shape (3, H, W) -- the same kind of
    tensor that came out of `get_eval_transforms`. We undo the channel
    normalization, clip to [0,1], and convert to a PIL Image for blending.
    """
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    img = image_tensor.detach().cpu() * std + mean
    img = img.clamp(0.0, 1.0)
    arr = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render a Grad-CAM heatmap on a sample image.")
    p.add_argument("--arch", required=True, choices=["resnet50", "efficientnet_b0", "mobilenet_v2"])
    p.add_argument("--checkpoint", required=True, help="Path to a .pt file saved by train.py.")
    p.add_argument("--image", required=True, help="Path to an image to explain.")
    p.add_argument("--manifest", default="configs/data_split.yaml")
    p.add_argument("--output", default="outputs/gradcam.png", help="Where to save the overlay.")
    p.add_argument("--target-class", type=int, default=None,
                   help="Class index to attribute. Defaults to the model's prediction.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # We only need the manifest for num_classes and class names; no
    # DataLoader needed for a single-image visualization.
    manifest = load_manifest(args.manifest)
    class_names: List[str] = [manifest.idx_to_class[i] for i in range(manifest.num_classes)]

    built = build_model(args.arch, num_classes=manifest.num_classes, pretrained=False)
    state = torch.load(args.checkpoint, map_location=device)
    built.model.load_state_dict(state["model_state_dict"])
    built.model.to(device)

    # Preprocess the image with the eval pipeline -- same as test time.
    transform = get_eval_transforms()
    pil_image = Image.open(args.image).convert("RGB")
    image_tensor = transform(pil_image).unsqueeze(0).to(device)

    heatmap, predicted_class = compute_gradcam(
        model=built.model,
        image_tensor=image_tensor,
        target_layer_name=built.gradcam_target_layer,
        target_class=args.target_class,
    )

    # Rebuild the displayable image from the (cropped, normalized) tensor
    # so the heatmap aligns with what the model actually saw.
    display_image = denormalize_for_display(image_tensor[0])
    overlay = overlay_heatmap_on_image(display_image, heatmap)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output_path)

    print(f"Predicted class: {class_names[predicted_class]} (idx={predicted_class})")
    print(f"Wrote overlay -> {output_path}")


if __name__ == "__main__":
    main()
