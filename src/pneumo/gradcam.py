"""Grad-CAM (Selvaraju et al., 2017) for timm CNNs, plus two checks of whether it can be trusted.

* `randomization_sanity_check` - Adebayo et al., 2018: if the maps barely change when the
  trained weights are replaced by random ones, they describe the image, not the model.
* `border_energy` - share of the heatmap outside the central region. Attention on the
  image border (text markers, edges, collimation) is a classic shortcut in CXR models.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch import nn


def gradcam(model: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (heatmaps (B,H,W) in [0,1] at input resolution, logits (B,))."""
    model.eval()
    with torch.enable_grad():
        feats = model.forward_features(x)        # (B, C, h, w) - last conv block
        if feats.ndim != 4:
            raise ValueError("Grad-CAM needs a CNN whose forward_features returns (B,C,H,W)")
        feats.retain_grad()
        logits = model.forward_head(feats).squeeze(1)
        model.zero_grad(set_to_none=True)
        logits.sum().backward()                   # each sample's grads only depend on its own logit
    weights = feats.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * feats).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
    flat = cam.flatten(1)
    cam = (cam - flat.min(1).values[:, None, None]) / (flat.max(1).values - flat.min(1).values + 1e-8)[:, None, None]
    return cam.detach(), logits.detach()


def border_energy(cams: torch.Tensor, margin: float = 0.15) -> np.ndarray:
    """Fraction of each heatmap's mass lying in the outer `margin` band of the image."""
    _, h, w = cams.shape
    mh, mw = int(h * margin), int(w * margin)
    inner = cams[:, mh:h - mh, mw:w - mw].flatten(1).sum(1)
    total = cams.flatten(1).sum(1) + 1e-8
    return (1 - inner / total).cpu().numpy()


def randomize_weights(model: nn.Module, seed: int = 0) -> nn.Module:
    rand = copy.deepcopy(model)
    torch.manual_seed(seed)
    for module in rand.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
    return rand.eval()


def randomization_sanity_check(model: nn.Module, x: torch.Tensor, batch_size: int = 32) -> dict[str, float]:
    """Spearman rho between trained and random-weight Grad-CAM maps.

    A map that is constant (all zeros - common for random weights, where ReLU removes every
    negative contribution) has no defined correlation. Those cases are counted separately
    instead of being treated as rho = 0, which would look like a pass.
    """
    random_model = randomize_weights(model)
    rhos, n_blank_trained, n_blank_random = [], 0, 0
    for i in range(0, len(x), batch_size):
        trained, _ = gradcam(model, x[i:i + batch_size])
        random, _ = gradcam(random_model, x[i:i + batch_size])
        for a, b in zip(trained, random):
            a, b = a.flatten().cpu().numpy(), b.flatten().cpu().numpy()
            blank_a, blank_b = np.ptp(a) == 0, np.ptp(b) == 0
            n_blank_trained += int(blank_a)
            n_blank_random += int(blank_b)
            if not (blank_a or blank_b):
                rhos.append(spearmanr(a, b).statistic)
    rhos = np.asarray(rhos, dtype=float)
    return {
        "spearman_mean": float(rhos.mean()) if len(rhos) else float("nan"),
        "spearman_std": float(rhos.std()) if len(rhos) else float("nan"),
        "n_valid": int(len(rhos)),
        "n_blank_trained": n_blank_trained,
        "n_blank_random": n_blank_random,
        "n": int(len(x)),
    }
