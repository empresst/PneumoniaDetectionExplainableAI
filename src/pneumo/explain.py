"""Grad-CAM galleries and quantitative explainability checks for a trained run.

    python -m pneumo.explain --run runs/densenet121_224

Needs `pneumo.evaluate` to have run first (uses its predictions.csv). Per subset it writes:
  gradcam_gallery.png  - most confident true/false positives/negatives with heatmaps
  explain.json         - randomization sanity check + border-attention statistics
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from .data import CXRDataset, build_transform, denormalize  # noqa: E402
from .gradcam import border_energy, gradcam, randomization_sanity_check  # noqa: E402
from .models import load_checkpoint  # noqa: E402


def _batch(df: pd.DataFrame, img_size: int) -> torch.Tensor:
    ds = CXRDataset(df, build_transform(img_size, train=False))
    return torch.stack([ds[i][0] for i in range(len(ds))])


def outcome(df: pd.DataFrame, threshold: float) -> pd.Series:
    pred = df.prob >= threshold
    return np.select([pred & (df.label == 1), ~pred & (df.label == 0), pred & (df.label == 0)],
                     ["TP", "TN", "FP"], "FN")


def gallery(model, df: pd.DataFrame, img_size: int, device: str, title: str, out: Path, per_group: int = 3) -> None:
    picks = []
    for kind in ["TP", "TN", "FP", "FN"]:
        g = df[df.outcome == kind]
        # most confident examples: errors the model was surest about are the most informative
        g = g.reindex((g.prob - 0.5).abs().sort_values(ascending=False).index).head(per_group)
        picks.append(g)
    picks = pd.concat(picks)
    if picks.empty:
        return
    x = _batch(picks, img_size).to(device)
    cams, _ = gradcam(model, x)
    fig, axes = plt.subplots(2, len(picks), figsize=(2.6 * len(picks), 5.6), squeeze=False)
    for j, (row, img, cam) in enumerate(zip(picks.itertuples(), x, cams)):
        gray = denormalize(img).cpu().numpy()
        axes[0, j].imshow(gray, cmap="gray")
        axes[0, j].set_title(f"{row.outcome}  p={row.prob:.2f}", fontsize=9)
        axes[1, j].imshow(gray, cmap="gray")
        axes[1, j].imshow(cam.cpu().numpy(), cmap="jet", alpha=0.4)
        for a in axes[:, j]:
            a.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def explain(run_dir: Path, max_images: int = 400, batch_size: int = 32) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_checkpoint(run_dir / "best.pt", device)
    img_size, threshold = ckpt["config"]["data"]["img_size"], ckpt["thresholds"]["youden"]
    results = {}
    for pred_csv in sorted((run_dir / "eval").glob("*/predictions.csv")):
        name = pred_csv.parent.name
        df = pd.read_csv(pred_csv)
        df["outcome"] = outcome(df, threshold)
        sample = df.sample(min(max_images, len(df)), random_state=0)

        energies = []
        for i in range(0, len(sample), batch_size):
            cams, _ = gradcam(model, _batch(sample.iloc[i:i + batch_size], img_size).to(device))
            energies.append(border_energy(cams))
        sample["border_energy"] = np.concatenate(energies)

        sanity = randomization_sanity_check(model, _batch(sample.head(32), img_size).to(device))
        res = {
            "randomization_sanity_check": sanity,
            "border_energy_mean": float(sample.border_energy.mean()),
            "border_energy_by_outcome": sample.groupby("outcome").border_energy.mean().round(4).to_dict(),
            "n_images": len(sample),
        }
        (pred_csv.parent / "explain.json").write_text(json.dumps(res, indent=2))
        gallery(model, df, img_size, device, f"Grad-CAM - {name}", pred_csv.parent / "gradcam_gallery.png")
        print(f"{name:26s} sanity rho={sanity['spearman_mean']:.2f}  border energy={res['border_energy_mean']:.3f}")
        results[name] = res
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--max-images", type=int, default=400)
    args = ap.parse_args()
    explain(args.run, args.max_images)


if __name__ == "__main__":
    main()
