"""Evaluate a trained run on the internal test set and the external RSNA set.

    python -m pneumo.evaluate --run runs/densenet121_224

Writes, per subset: metrics.json (with patient-level bootstrap 95% CIs), predictions.csv,
ROC / PR / reliability / confusion-matrix plots, and - for RSNA - a subgroup table by
sex, age band and view position (PA vs AP).
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
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve  # noqa: E402

from .data import load_manifest, make_loader, select_subset  # noqa: E402
from .metrics import expected_calibration_error, full_report, point_metrics  # noqa: E402
from .models import load_checkpoint, predict_logits, sigmoid  # noqa: E402


def plot_curves(y, p, threshold: float, title: str, out: Path) -> None:
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.2))
    fpr, tpr, _ = roc_curve(y, p)
    ax[0].plot(fpr, tpr, lw=2)
    ax[0].plot([0, 1], [0, 1], "--", c="grey", lw=1)
    ax[0].set(title=f"ROC (AUROC {roc_auc_score(y, p):.3f})", xlabel="1 - specificity", ylabel="sensitivity")

    prec, rec, _ = precision_recall_curve(y, p)
    ax[1].plot(rec, prec, lw=2)
    ax[1].axhline(y.mean(), ls="--", c="grey", lw=1)
    ax[1].set(title="Precision-recall", xlabel="recall", ylabel="precision")

    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, 9)
    conf = [p[idx == b].mean() for b in range(10) if (idx == b).any()]
    acc = [y[idx == b].mean() for b in range(10) if (idx == b).any()]
    ax[2].plot([0, 1], [0, 1], "--", c="grey", lw=1)
    ax[2].plot(conf, acc, "o-", lw=2)
    ax[2].set(title=f"Reliability (ECE {expected_calibration_error(y, p):.3f})",
              xlabel="predicted probability", ylabel="observed frequency")

    m = point_metrics(y, p, threshold)
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    ax[3].imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax[3].text(j, i, str(v), ha="center", va="center", fontsize=13,
                   color="white" if v > cm.max() / 2 else "black")
    ax[3].set(title=f"Confusion @ {threshold:.2f}", xticks=[0, 1], yticks=[0, 1],
              xticklabels=["pred normal", "pred pneumonia"], yticklabels=["normal", "pneumonia"])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def subgroup_table(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    df = df.copy()
    df["age_band"] = pd.cut(df["age"], [0, 18, 40, 60, 200], right=False, labels=["<18", "18-39", "40-59", "60+"])
    rows = []
    for col in ["sex", "age_band", "view"]:
        for value, g in df.groupby(col, observed=True):
            if g.label.nunique() < 2 or len(g) < 30:
                continue
            m = point_metrics(g.label.to_numpy(), g.prob.to_numpy(), threshold)
            rows.append({"attribute": col, "group": value, "n": len(g), "prevalence": g.label.mean(),
                         "auroc": roc_auc_score(g.label, g.prob),
                         "sensitivity": m["sensitivity"], "specificity": m["specificity"]})
    return pd.DataFrame(rows)


def evaluate(run_dir: Path, subsets: list[str] | None = None, batch_size: int = 64) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_checkpoint(run_dir / "best.pt", device)
    cfg = ckpt["config"]
    temperature, thresholds = ckpt["temperature"], ckpt["thresholds"]
    df = load_manifest(cfg["data"]["manifest"])
    if subsets is None:
        subsets = ["test"] + [s for s in cfg["eval"]["external_subsets"] if (df["source"] == "rsna").any()]

    summary = {}
    for name in subsets:
        sub = select_subset(df, name)
        if sub.empty:
            print(f"skipping {name}: no images")
            continue
        loader = make_loader(sub, cfg["data"]["img_size"], False, batch_size, cfg["data"]["num_workers"])
        logits = predict_logits(model, loader, device, amp=device == "cuda")
        sub["prob_uncalibrated"] = sigmoid(logits)
        sub["prob"] = sigmoid(logits / temperature)

        out = run_dir / "eval" / name
        out.mkdir(parents=True, exist_ok=True)
        report = full_report(sub.label, sub.prob, sub.patient_id, thresholds, cfg["eval"]["bootstrap"])
        report["ece_uncalibrated"] = expected_calibration_error(sub.label.to_numpy(), sub.prob_uncalibrated.to_numpy())
        (out / "metrics.json").write_text(json.dumps(report, indent=2))
        sub.drop(columns=["src_path"], errors="ignore").to_csv(out / "predictions.csv", index=False)
        plot_curves(sub.label.to_numpy(), sub.prob.to_numpy(), thresholds["youden"],
                    f"{cfg['run_name']} - {name}", out / "curves.png")
        if name.startswith("rsna"):
            subgroup_table(sub, thresholds["youden"]).to_csv(out / "subgroups.csv", index=False)
        ci = report["ci95"]["auroc"]
        print(f"{name:26s} n={report['n']:5d}  AUROC {report['auroc']:.3f} [{ci[0]:.3f}-{ci[1]:.3f}]  "
              f"AUPRC {report['auprc']:.3f}  ECE {report['ece']:.3f}")
        summary[name] = report
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--subsets", nargs="*")
    args = ap.parse_args()
    evaluate(args.run, args.subsets)


if __name__ == "__main__":
    main()
