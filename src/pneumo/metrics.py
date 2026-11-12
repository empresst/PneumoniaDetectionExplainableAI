"""Classification metrics, operating-point selection, calibration error and bootstrap CIs."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


def choose_thresholds(y: np.ndarray, p: np.ndarray, high_sens_target: float) -> dict[str, float]:
    """Operating points chosen on the *validation* set and then frozen for every test set."""
    fpr, tpr, thr = roc_curve(y, p)
    thr = np.clip(thr, 0, 1)  # sklearn puts +inf as the first threshold
    youden = float(thr[np.argmax(tpr - fpr)])
    ok = np.where(tpr >= high_sens_target)[0]
    high_sens = float(thr[ok[0]]) if len(ok) else float(thr[-1])
    return {"default_0.5": 0.5, "youden": youden, f"sens>={high_sens_target:.2f}": high_sens}


def point_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    pred = p >= threshold
    tp = int(np.sum(pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    safe = lambda a, b: a / b if b else float("nan")  # noqa: E731
    sens, spec, ppv = safe(tp, tp + fn), safe(tn, tn + fp), safe(tp, tp + fp)
    return {
        "threshold": threshold,
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "npv": safe(tn, tn + fn),
        "f1": safe(2 * ppv * sens, ppv + sens),
        "accuracy": (tp + tn) / len(y),
        "balanced_accuracy": (sens + spec) / 2,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def ranking_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": expected_calibration_error(y, p),
        "prevalence": float(y.mean()),
        "n": int(len(y)),
    }


def bootstrap_ci(y: np.ndarray, p: np.ndarray, groups: np.ndarray, thresholds: dict[str, float],
                 n_boot: int, seed: int = 0) -> dict[str, list[float]]:
    """95% percentile CIs, resampling *patients* (not images) so correlated images stay together."""
    rng = np.random.default_rng(seed)
    uniq, inverse = np.unique(groups, return_inverse=True)
    members = [np.where(inverse == g)[0] for g in range(len(uniq))]
    samples: dict[str, list[float]] = {}
    for _ in range(n_boot):
        idx = np.concatenate([members[g] for g in rng.integers(0, len(uniq), len(uniq))])
        yb, pb = y[idx], p[idx]
        if yb.min() == yb.max():
            continue
        stats = {k: v for k, v in ranking_metrics(yb, pb).items() if k in {"auroc", "auprc", "ece"}}
        for name, t in thresholds.items():
            m = point_metrics(yb, pb, t)
            stats[f"{name}/sensitivity"] = m["sensitivity"]
            stats[f"{name}/specificity"] = m["specificity"]
        for k, v in stats.items():
            samples.setdefault(k, []).append(v)
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in samples.items()}


def full_report(y, p, groups, thresholds, n_boot: int) -> dict:
    y, p = np.asarray(y, dtype=int), np.asarray(p, dtype=float)
    return {
        **ranking_metrics(y, p),
        "operating_points": {name: point_metrics(y, p, t) for name, t in thresholds.items()},
        "ci95": bootstrap_ci(y, p, np.asarray(groups), thresholds, n_boot),
    }
