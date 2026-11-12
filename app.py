"""Gradio demo: upload a chest X-ray, get a calibrated probability and a Grad-CAM overlay.

    python app.py --checkpoint runs/densenet121_224/best.pt [--share]

Research prototype only - not a medical device and not for clinical use.
"""
from __future__ import annotations

import argparse

import gradio as gr
import matplotlib.cm as cm
import numpy as np
import torch
from PIL import Image

from pneumo.data import build_transform
from pneumo.gradcam import gradcam
from pneumo.models import load_checkpoint
from pneumo.prepare import letterbox

DISCLAIMER = ("**Research prototype - not a medical device.** Trained on pediatric X-rays from a single "
              "hospital; performance drops on other populations (see the README). Do not use for diagnosis.")


def build_app(checkpoint: str) -> gr.Blocks:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_checkpoint(checkpoint, device)
    img_size, temperature = ckpt["config"]["data"]["img_size"], ckpt["temperature"]
    threshold = ckpt["thresholds"]["youden"]
    transform = build_transform(img_size, train=False)

    def predict(image: Image.Image):
        img = letterbox(image.convert("L"), img_size)
        x = transform(img).unsqueeze(0).to(device)
        cam, logit = gradcam(model, x)
        prob = float(torch.sigmoid(logit / temperature))
        heat = cm.jet(cam[0].cpu().numpy())[..., :3]
        gray = np.asarray(img, dtype=np.float32)[..., None] / 255
        overlay = Image.fromarray(np.uint8(255 * (0.6 * gray + 0.4 * heat)))
        verdict = "PNEUMONIA" if prob >= threshold else "NORMAL"
        return {"pneumonia": prob, "normal": 1 - prob}, overlay, f"Prediction at operating threshold {threshold:.2f}: **{verdict}**"

    with gr.Blocks(title="Pneumonia CXR - explainable classifier") as demo:
        gr.Markdown("# Pneumonia detection from chest X-rays\n" + DISCLAIMER)
        with gr.Row():
            inp = gr.Image(type="pil", label="Chest X-ray (PA/AP)")
            with gr.Column():
                label = gr.Label(label="Calibrated probability")
                verdict = gr.Markdown()
                overlay = gr.Image(label="Grad-CAM: regions driving the prediction")
        gr.Button("Analyse").click(predict, inp, [label, overlay, verdict])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="runs/densenet121_224/best.pt")
    ap.add_argument("--share", action="store_true", help="public link (useful on Colab)")
    args = ap.parse_args()
    build_app(args.checkpoint).launch(share=args.share)
