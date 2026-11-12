"""Train one configuration, select the best epoch on validation AUROC, calibrate, pick thresholds.

    python -m pneumo.train --config configs/default.yaml [--set train.epochs=20 ...]
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch import nn

from .config import load_config, seed_everything
from .data import load_manifest, make_loader, select_subset
from .metrics import choose_thresholds
from .models import build_model, fit_temperature, predict_logits, save_checkpoint, sigmoid


def cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def freeze_frozen_batchnorm(model: nn.Module) -> None:
    """Keep running statistics of frozen BatchNorm layers fixed (true linear probing)."""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm) and not any(p.requires_grad for p in m.parameters()):
            m.eval()


def train(cfg: dict) -> Path:
    seed_everything(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run_dir = Path(cfg["output_dir"]) / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    d, m, t = cfg["data"], cfg["model"], cfg["train"]
    df = load_manifest(d["manifest"])
    train_df, val_df = select_subset(df, "train"), select_subset(df, "val")
    train_loader = make_loader(train_df, d["img_size"], True, t["batch_size"], d["num_workers"])
    val_loader = make_loader(val_df, d["img_size"], False, t["batch_size"], d["num_workers"])
    print(f"train {len(train_df)} imgs ({train_df.label.mean():.1%} pneumonia) | val {len(val_df)} imgs | {device}")

    model = build_model(m["name"], m["pretrained"], m["drop_rate"], m["freeze_backbone"]).to(device)
    n_pos, n_neg = int(train_df.label.sum()), int((1 - train_df.label).sum())
    pos_weight = n_neg / n_pos if t["pos_weight"] == "auto" else float(t["pos_weight"])
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=t["lr"], weight_decay=t["weight_decay"])
    steps_per_epoch = len(train_loader)
    scheduler = cosine_with_warmup(optimizer, t["warmup_epochs"] * steps_per_epoch, t["epochs"] * steps_per_epoch)
    use_amp = t["amp"] and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history, best_auc, best_epoch, stale = [], -1.0, -1, 0
    for epoch in range(1, t["epochs"] + 1):
        model.train()
        if m["freeze_backbone"]:
            freeze_frozen_batchnorm(model)
        start, running = time.time(), 0.0
        for x, y, _ in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                loss = criterion(model(x).squeeze(1), y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item() * len(x)

        val_logits = predict_logits(model, val_loader, device, use_amp)
        val_auc = roc_auc_score(val_df.label, val_logits)
        history.append({"epoch": epoch, "train_loss": running / len(train_loader.dataset),
                        "val_auroc": val_auc, "lr": scheduler.get_last_lr()[0], "seconds": time.time() - start})
        print(f"epoch {epoch:2d}  loss {history[-1]['train_loss']:.4f}  val AUROC {val_auc:.4f}  "
              f"({history[-1]['seconds']:.0f}s)")
        if val_auc > best_auc:
            best_auc, best_epoch, stale = val_auc, epoch, 0
            save_checkpoint(run_dir / "best.pt", model, cfg)
        else:
            stale += 1
            if stale >= t["patience"]:
                print(f"early stopping: no val improvement for {stale} epochs")
                break

    # Reload the best epoch, calibrate on validation and freeze the operating points.
    model.load_state_dict(torch.load(run_dir / "best.pt", weights_only=False)["state_dict"])
    val_logits = predict_logits(model, val_loader, device, use_amp)
    temperature = fit_temperature(val_logits, val_df.label.to_numpy())
    thresholds = choose_thresholds(val_df.label.to_numpy(), sigmoid(val_logits / temperature),
                                   cfg["eval"]["high_sens_target"])
    save_checkpoint(run_dir / "best.pt", model, cfg, {"temperature": temperature, "thresholds": thresholds,
                                                     "best_epoch": best_epoch, "val_auroc": best_auc})
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"best epoch {best_epoch} (val AUROC {best_auc:.4f}); temperature {temperature:.3f}; thresholds {thresholds}")
    return run_dir


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[], help="overrides like train.epochs=20")
    args = ap.parse_args()
    train(load_config(args.config, args.set))


if __name__ == "__main__":
    main()
