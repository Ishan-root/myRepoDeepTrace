# src/gradcam.py
#
# Self-contained Grad-CAM for DeepTrace (ResNet18, 2 classes).
#
# Imports only stdlib, numpy, torch and PIL, so it can be imported both as
# `from gradcam import ...` (scripts run from src/) and as
# `from src.gradcam import ...` (the Gradio app).
#
# Design notes:
# - Activations are captured with FORWARD hooks. Gradients are obtained with
#   torch.autograd.grad(score, activations). No backward hooks (ResNet's
#   in-place ReLU makes them unreliable) and no .backward() (which would
#   write into param.grad).
# - The verdict (logits/probs) and the heatmaps come from ONE forward pass,
#   so they can never disagree.
# - All hooks are removed in a `finally` block; the model's weights, its
#   training flag and param.grad are never touched.

import base64
import io
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

IMAGE_SIZE = 224
_EPS = 1e-8


@dataclass
class GradCAMResult:
    logits: torch.Tensor          # shape (1, num_classes), detached
    probs: torch.Tensor           # shape (1, num_classes), softmax, detached
    class_idx: int                # class the heatmaps explain
    cams: dict = field(default_factory=dict)  # name -> float32 ndarray (224, 224) in [0, 1]


def default_layers(model):
    """Last block of layer2/3/4 of a torchvision ResNet. layer4 is the primary map."""
    return {
        "layer2": model.layer2[-1],
        "layer3": model.layer3[-1],
        "layer4": model.layer4[-1],
    }


def _normalize_cam(cam):
    """Min-max normalize a 2D float tensor to [0, 1]. A flat map becomes all zeros (never NaN)."""
    cam_min = cam.min()
    cam_max = cam.max()
    span = cam_max - cam_min
    if not torch.isfinite(span) or span.item() <= _EPS:
        return torch.zeros_like(cam)
    return ((cam - cam_min) / span).clamp_(0.0, 1.0)


class GradCAM:
    """
    Usage:
        cam = GradCAM(model)                      # model must already be in eval()
        result = cam(input_tensor)                # input_tensor: (1, 3, 224, 224) on the model's device
        heat = result.cams["layer4"]              # float32 (224, 224) in [0, 1]
    """

    def __init__(self, model, layers=None):
        self.model = model
        self.layers = dict(layers) if layers is not None else default_layers(model)
        if not self.layers:
            raise ValueError("GradCAM needs at least one target layer")

    def __call__(self, input_tensor, class_idx=None):
        if self.model.training:
            raise AssertionError("GradCAM expects the model in eval() mode")
        if input_tensor.dim() != 4 or input_tensor.shape[0] != 1:
            raise ValueError(f"expected input of shape (1, C, H, W), got {tuple(input_tensor.shape)}")

        names = list(self.layers.keys())
        activations = {}
        handles = []

        def make_hook(name):
            def hook(module, inputs, output):
                activations[name] = output
            return hook

        try:
            for name in names:
                handles.append(self.layers[name].register_forward_hook(make_hook(name)))

            with torch.enable_grad():
                # A fresh leaf that requires grad guarantees an autograd graph
                # even if someone froze the model's parameters. Values are unchanged.
                x = input_tensor.detach().clone().requires_grad_(True)
                logits = self.model(x)

                if class_idx is None:
                    class_idx = int(torch.argmax(logits[0]).item())
                class_idx = int(class_idx)
                if not 0 <= class_idx < logits.shape[1]:
                    raise ValueError(f"class_idx {class_idx} out of range for {logits.shape[1]} classes")

                missing = [n for n in names if n not in activations]
                if missing:
                    raise RuntimeError(f"forward hooks did not fire for layers: {missing}")

                score = logits[0, class_idx]          # the logit, not the softmax
                acts = [activations[n] for n in names]
                grads = torch.autograd.grad(score, acts, retain_graph=False, allow_unused=False)

            cams = {}
            with torch.no_grad():
                for name, act, grad in zip(names, acts, grads):
                    act = act.detach()
                    grad = grad.detach()
                    alpha = grad.mean(dim=(2, 3), keepdim=True)                # (1, C, 1, 1)
                    cam = F.relu((alpha * act).sum(dim=1, keepdim=True))       # (1, 1, h, w)
                    cam = F.interpolate(cam, size=(IMAGE_SIZE, IMAGE_SIZE),
                                        mode="bilinear", align_corners=False)
                    cam = _normalize_cam(cam[0, 0].float())
                    cams[name] = cam.cpu().numpy().astype(np.float32, copy=False)

                logits_d = logits.detach()
                probs = F.softmax(logits_d, dim=1)

            return GradCAMResult(logits=logits_d, probs=probs, class_idx=class_idx, cams=cams)
        finally:
            for h in handles:
                h.remove()
            activations.clear()


# ---------------------------------------------------------------------------
# Rendering helpers (numpy + PIL only)
# ---------------------------------------------------------------------------

# Piecewise-linear "heat" colormap: dark blue -> blue -> cyan -> green -> yellow -> red.
_HEAT_STOPS = np.array([0.00, 0.20, 0.40, 0.60, 0.80, 1.00])
_HEAT_COLORS = np.array([
    [ 20,  20, 120],
    [ 30,  90, 230],
    [ 30, 200, 220],
    [ 60, 210,  70],
    [250, 220,  40],
    [230,  40,  30],
], dtype=np.float64)


def _build_lut(stops, colors):
    xs = np.linspace(0.0, 1.0, 256)
    lut = np.stack([np.interp(xs, stops, colors[:, c]) for c in range(3)], axis=1)
    return np.clip(np.round(lut), 0, 255).astype(np.uint8)


HEAT_LUT = _build_lut(_HEAT_STOPS, _HEAT_COLORS)


def colorize(cam, lut=HEAT_LUT):
    """(H, W) float map in [0, 1] -> (H, W, 3) uint8 RGB."""
    cam = np.nan_to_num(np.asarray(cam, dtype=np.float32), nan=0.0, posinf=1.0, neginf=0.0)
    idx = np.clip(np.round(cam * 255.0), 0, 255).astype(np.uint8)
    return lut[idx]


def resized_rgb(pil_image):
    """The 224x224 RGB image the model actually sees (before ToTensor/Normalize)."""
    return pil_image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)


def overlay(pil_resized_224, cam, alpha=0.85, dim=0.55):
    """
    Blend the heatmap onto the 224x224 resized RGB image.

    The photo is turned grey and dimmed (`dim`) so the heat colours stand out even
    on very colourful images. Heat opacity grows with importance: unimportant
    regions show the grey photo, important regions show strong colour (up to `alpha`).
    """
    rgb = np.asarray(pil_resized_224.convert("RGB"), dtype=np.float32)
    cam = np.asarray(cam, dtype=np.float32)
    if rgb.shape[:2] != cam.shape:
        raise ValueError(f"image {rgb.shape[:2]} and cam {cam.shape} sizes differ; "
                         f"pass the 224x224 resized image (see resized_rgb)")
    gray = rgb @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    base = np.repeat((gray * dim)[..., None], 3, axis=2)
    heat = colorize(cam).astype(np.float32)
    c = np.clip(np.nan_to_num(cam, nan=0.0), 0.0, 1.0)
    a = (alpha * np.sqrt(c))[..., None]
    out = base * (1.0 - a) + heat * a
    return Image.fromarray(np.clip(np.round(out), 0, 255).astype(np.uint8))


def to_png_base64(pil_image):
    """PIL image -> base64 PNG string (for <img src="data:image/png;base64,...">)."""
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")
