"""Model construction, temperature scaling, inference and checkpoint I/O."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import timm
import torch
from scipy.special import expit
from torch import nn


def build_model(name: str, pretrained: bool, drop_rate: float = 0.0, freeze_backbone: bool = False) -> nn.Module:
    model = timm.create_model(name, pretrained=pretrained, num_classes=1, drop_rate=drop_rate)
    if freeze_backbone:
        head = model.get_classifier()
        for p in model.parameters():
            p.requires_grad = False
        for p in head.parameters():
            p.requires_grad = True
    return model


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Temperature scaling (Guo et al., 2017), fitted on validation logits."""
    z = torch.tensor(logits, dtype=torch.float64)
    y = torch.tensor(labels, dtype=torch.float64)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    # Bounded: on a small, (near-)separable validation set the unconstrained optimum is T -> 0,
    # which would turn every probability into 0 or 1.
    return float(np.clip(log_t.detach().exp().item(), 0.25, 10.0))


@torch.no_grad()
def predict_logits(model: nn.Module, loader, device: str, amp: bool = False) -> np.ndarray:
    model.eval()
    out = []
    for x, _, _ in loader:
        with torch.autocast(device_type="cuda", enabled=amp and device == "cuda"):
            out.append(model(x.to(device, non_blocking=True)).float().squeeze(1).cpu())
    return torch.cat(out).numpy()


def sigmoid(z: np.ndarray) -> np.ndarray:
    return expit(z)


def save_checkpoint(path: Path, model: nn.Module, cfg: dict, extra: dict | None = None) -> None:
    torch.save({"state_dict": model.state_dict(), "config": cfg, **(extra or {})}, path)


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[nn.Module, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    m = ckpt["config"]["model"]
    model = build_model(m["name"], pretrained=False, drop_rate=m.get("drop_rate", 0.0))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt
