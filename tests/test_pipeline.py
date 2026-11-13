"""End-to-end: prepare -> train -> evaluate -> explain -> summarize on synthetic data (CPU, ~1 min)."""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from pneumo.config import load_config
from pneumo.evaluate import evaluate
from pneumo.explain import explain
from pneumo.summarize import summarize
from pneumo.train import train

ROOT = Path(__file__).resolve().parents[1]


def test_full_pipeline(pediatric_root, rsna_root, tmp_path):
    data = tmp_path / "data"
    subprocess.run([sys.executable, "-m", "pneumo.prepare", "--pediatric", str(pediatric_root),
                    "--rsna", str(rsna_root), "--rsna-n", "0", "--out", str(data),
                    "--cache-size", "64", "--workers", "2"], check=True)
    manifest = pd.read_csv(data / "manifest.csv")
    assert set(manifest.source) == {"pediatric", "rsna"}
    assert manifest.loc[manifest.source == "rsna", "view"].notna().all()

    cfg = load_config(ROOT / "configs" / "smoke.yaml",
                      [f"data.manifest={data / 'manifest.csv'}", f"output_dir={tmp_path / 'runs'}"])
    run_dir = train(cfg)
    ckpt_keys = __import__("torch").load(run_dir / "best.pt", weights_only=False).keys()
    assert {"temperature", "thresholds", "best_epoch"} <= set(ckpt_keys)

    summary = evaluate(run_dir)
    assert set(summary) == {"test", "rsna_opacity_vs_normal", "rsna_opacity_vs_all"}
    assert summary["rsna_opacity_vs_normal"]["n"] == 40
    for name in summary:
        assert (run_dir / "eval" / name / "curves.png").exists()
    assert (run_dir / "eval" / "rsna_opacity_vs_all" / "subgroups.csv").exists()

    results = explain(run_dir, max_images=40)
    assert (run_dir / "eval" / "test" / "gradcam_gallery.png").exists()
    sanity = results["test"]["randomization_sanity_check"]
    assert sanity["n_valid"] + max(sanity["n_blank_trained"], sanity["n_blank_random"]) >= 1

    table = summarize(tmp_path / "runs", tmp_path / "results")
    assert "smoke" in table and json.loads((tmp_path / "results" / "results.json").read_text())
