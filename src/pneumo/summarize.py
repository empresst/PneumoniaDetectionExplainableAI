"""Collect every run into one Markdown results table (results/RESULTS.md) + JSON.

    python -m pneumo.summarize --runs runs --out results
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _fmt(value: float, ci: list[float] | None = None) -> str:
    return f"{value:.3f} [{ci[0]:.3f}–{ci[1]:.3f}]" if ci else f"{value:.3f}"


def _sanity(check: dict) -> str:
    if "n_valid" not in check:  # explain.json from before blank maps were counted
        return f"{check['spearman_mean']:.2f} (unverified)"
    return f"{check['spearman_mean']:.2f} ({check['n_valid']}/{check['n']} valid)"


def _subgroups(runs: Path, subset: str = "rsna_opacity_vs_normal") -> str:
    import pandas as pd

    parts = []
    for csv in sorted(runs.glob(f"*/eval/{subset}/subgroups.csv")):
        df = pd.read_csv(csv)
        if df.empty:
            continue
        lines = ["| Attribute | Group | n | Prevalence | AUROC | Sensitivity* | Specificity* |", "|---|---|---|---|---|---|---|"]
        lines += [f"| {r.attribute} | {r.group} | {r.n} | {r.prevalence:.2f} | {r.auroc:.3f} | "
                  f"{r.sensitivity:.3f} | {r.specificity:.3f} |" for r in df.itertuples()]
        parts.append(f"### {csv.parts[-4]}\n\n" + "\n".join(lines))
    if not parts:
        return ""
    return (f"\n\n## Subgroups — RSNA, {subset}\n\nPoint estimates only (no CIs); groups with < 30 "
            "images or a single class are omitted. Age bands from DICOM metadata.\n\n" + "\n\n".join(parts) + "\n")


def summarize(runs: Path, out: Path) -> str:
    rows, raw = [], {}
    for metrics_path in sorted(runs.glob("*/eval/*/metrics.json")):
        run, subset = metrics_path.parts[-4], metrics_path.parts[-2]
        m = json.loads(metrics_path.read_text())
        explain_path = metrics_path.parent / "explain.json"
        ex = json.loads(explain_path.read_text()) if explain_path.exists() else {}
        op = m["operating_points"]["youden"]
        ci = m["ci95"]
        rows.append("| " + " | ".join([
            run, subset, str(m["n"]),
            _fmt(m["auroc"], ci.get("auroc")),
            _fmt(m["auprc"], ci.get("auprc")),
            _fmt(op["sensitivity"], ci.get("youden/sensitivity")),
            _fmt(op["specificity"], ci.get("youden/specificity")),
            f"{m['ece_uncalibrated']:.3f} → {m['ece']:.3f}",
            f"{ex['border_energy_mean']:.2f}" if ex else "–",
            _sanity(ex["randomization_sanity_check"]) if ex else "–",
        ]) + " |")
        raw.setdefault(run, {})[subset] = {"metrics": m, "explain": ex}

    header = (
        "| Run | Test set | n | AUROC [95% CI] | AUPRC [95% CI] | Sensitivity* | Specificity* "
        "| ECE before → after calibration | Border attention† | Sanity ρ‡ |\n"
        "|---|---|---|---|---|---|---|---|---|---|"
    )
    notes = (
        "\n\n\\* At the Youden-optimal threshold chosen on the validation set and frozen for all test sets.  \n"
        "† Share of Grad-CAM mass in the outer 15% border band (a uniform map scores ≈ 0.50 at 224 px; "
        "lower = attention on the lung fields).  \n"
        "‡ Spearman correlation between Grad-CAM maps of the trained model and of the same "
        "network with random weights (Adebayo et al., 2018); close to 0 = maps depend on what was learned. "
        "Images where either map is blank have no defined correlation and are excluded; "
        "the count of valid comparisons is shown in brackets.\n"
    )
    table = header + "\n" + "\n".join(rows) + notes + _subgroups(runs)
    out.mkdir(parents=True, exist_ok=True)
    (out / "RESULTS.md").write_text("# Results\n\n" + table)
    (out / "results.json").write_text(json.dumps(raw, indent=2))
    print(table)
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args()
    summarize(args.runs, args.out)


if __name__ == "__main__":
    main()
