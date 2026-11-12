# Explainable Pneumonia Detection from Chest X-Rays — with External Validation

A chest X-ray pneumonia classifier built to answer a harder question than *"what accuracy can I get?"*:
**does the model still work on a different population, and can its explanations be trusted?**

- **Leakage-free evaluation:** patient-level, stratified train/val/test split.
- **External validation:** trained on pediatric X-rays from one hospital in China, tested on adult X-rays from the US (RSNA).
- **Clinically meaningful metrics:** AUROC and AUPRC with patient-level bootstrap 95% CIs, sensitivity and specificity at operating points frozen on validation, calibration (ECE) before and after temperature scaling, and subgroup analysis by sex, age and view.
- **Explanations that are checked:** Grad-CAM plus a weight-randomization sanity check and a quantitative shortcut-learning probe.
- **Engineering:** controlled ablations, a reproducible config-driven pipeline, tests, and a Gradio demo.

> ⚠️ Research prototype. Not a medical device and not for clinical use.

---

## 1. Motivation — and what was wrong with v1

[v1 of this project](notebooks/legacy_v1_mobilenet.ipynb) reported ~87% test accuracy with a frozen MobileNetV2.
On review, that number could not be trusted:

| Problem in v1 | Consequence | Fix in v2 |
|---|---|---|
| Validation images sampled from the training folder independently of the training images | Train/val overlap, so tuning and early stopping were done on seen data | Disjoint split |
| Split by image, not patient (the dataset contains several images per child) | Same patient in train and test, which inflates scores | `StratifiedGroupKFold` on patient IDs parsed from file names |
| Grad-CAM computed on a freshly initialized, **untrained** model | Heatmaps explained nothing | Grad-CAM on the trained checkpoint + randomization sanity check |
| `rescale=1/255` instead of the backbone's ImageNet normalization | Degraded pretrained features | ImageNet mean/std normalization |
| 500 of ~5,200 training images, 128 px, 3 epochs, frozen backbone | Underfit baseline | Full data, 224 px, fine-tuning, cosine LR, early stopping |
| Accuracy at a fixed 0.5 threshold | Hides the sensitivity/specificity trade-off clinicians care about | ROC/PR, operating points chosen on validation |
| Single in-distribution test set | No evidence of generalization | External RSNA test set |

The v1 configuration is kept as a baseline (`configs/ablation_mobilenetv2_frozen_128.yaml`) so the effect of each fix can be measured.

## 2. Data

| Role | Dataset | Population | Images used |
|---|---|---|---|
| Train / val / internal test | [Kermany et al. 2018](https://data.mendeley.com/datasets/rscbjbr9sd/2) ([Kaggle mirror](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)), CC BY 4.0 | Children aged 1–5, Guangzhou Women and Children's Medical Center | ~5,860, re-split 70/15/15 by patient |
| External test only | [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge) (from NIH ChestX-ray14) | Adults, US, PA and AP views | Stratified sample of 6,000 (configurable) |

The external set is evaluated in two ways:
- `rsna_opacity_vs_normal`: *Lung Opacity* vs *Normal*, the same task the model was trained on.
- `rsna_opacity_vs_all`: also counts *No Lung Opacity / Not Normal* (other pathologies) as negatives. This is closer to real triage.

No dataset is redistributed in this repository.

## 3. Method

- **Preprocessing:** pad to square (keeps anatomical proportions), resize, grayscale → 3 channels, ImageNet normalization. DICOMs are rescaled by bit depth, not per image, so intensities stay comparable to the JPEG training data.
- **Augmentation:** small random-resized-crop, rotation (±7°), translation, and brightness/contrast jitter. **No horizontal flip**, because it would place the heart on the wrong side.
- **Model:** ImageNet-pretrained CNN from `timm`. The main model is DenseNet-121, the CheXNet backbone. It uses a single logit with `BCEWithLogitsLoss` and `pos_weight = n_neg / n_pos` for class imbalance.
- **Optimization:** AdamW, 1 warmup epoch then cosine decay, mixed precision. The checkpoint with the best validation AUROC is kept (patience 4).
- **Calibration:** temperature scaling fitted on the validation set.
- **Operating points** (fixed on validation, then frozen for every test set): Youden's J, and a high-sensitivity point (sensitivity ≥ 0.95, as needed for screening).
- **Uncertainty:** 95% percentile bootstrap CIs (2,000 resamples), resampling **patients** rather than images.

### Explainability checks
1. **Grad-CAM galleries** of the most confident true/false positives/negatives on each test set.
2. **Randomization sanity check** ([Adebayo et al., 2018](https://arxiv.org/abs/1810.03292)): Spearman correlation between the Grad-CAM maps of the trained model and of the same architecture with random weights. A high correlation means the "explanation" mostly reflects image edges, not learned evidence.
3. **Border-attention score:** share of Grad-CAM mass in the outer 15% of the image. CXR models are known to latch onto text markers, edges and collimation ([Zech et al., 2018](https://doi.org/10.1371/journal.pmed.1002683); [DeGrave et al., 2021](https://doi.org/10.1038/s42256-021-00338-7)). A uniform map scores 0.51, and lower means attention on the lung fields.

## 4. Results

> 🔜 **Pending:** run [`notebooks/kaggle_pipeline.ipynb`](notebooks/kaggle_pipeline.ipynb) on a GPU. `python -m pneumo.summarize` then writes the full table to `results/RESULTS.md`, and the key rows are copied here.

| Model | Internal test AUROC [95% CI] | RSNA AUROC (opacity vs normal) | RSNA AUROC (vs all) | Sens / Spec @ Youden (internal) |
|---|---|---|---|---|
| v1 setup: MobileNetV2 frozen, 128 px | – | – | – | – |
| MobileNetV2 fine-tuned, 224 px | – | – | – | – |
| EfficientNet-B0, 224 px | – | – | – | – |
| ResNet-50, 224 px | – | – | – | – |
| **DenseNet-121, 224 px (main)** | – | – | – | – |

Figures produced per run in `runs/<run>/eval/<test set>/`: ROC, PR, reliability diagram, confusion matrix, Grad-CAM gallery, and for RSNA a subgroup table.

## 5. Limitations

- **Population shift by design.** The training data are pediatric and come from a single centre. Adults look different (heart size, bone density, devices), so a performance drop on RSNA is expected. The point is to **measure** it, not hide it.
- **Label differences.** Kermany labels are image-level diagnoses. RSNA "Lung Opacity" labels are radiologist-drawn opacities, which are related to pneumonia but not identical.
- **Grad-CAM is coarse** (7×7 feature map at 224 px for DenseNet-121) and shows correlation, not causation. The sanity check and border score limit, but do not remove, that caveat.
- **No prospective or clinical evaluation**, and no assessment of radiologist + model performance together.

## 6. Reproduce

```bash
pip install -e .            # Python ≥ 3.10
python -m pneumo.prepare --pediatric data/raw/pediatric --rsna data/raw/rsna --out data
bash scripts/run_all.sh     # train + evaluate + explain every config, then summarize
python app.py --checkpoint runs/densenet121_224/best.pt   # demo (pip install gradio)
```

On Kaggle, use [`notebooks/kaggle_pipeline.ipynb`](notebooks/kaggle_pipeline.ipynb) (datasets attached as notebook inputs, no API key needed). On Colab, use [`notebooks/colab_pipeline.ipynb`](notebooks/colab_pipeline.ipynb), which downloads the data and saves runs to Google Drive.
To test the whole pipeline on synthetic data on CPU (about 30 s): `pip install -r requirements-dev.txt && pytest`.

```
configs/        experiment configs (default.yaml + ablations; `base:` inheritance)
src/pneumo/
  prepare.py    manifest, patient-level split, RSNA metadata, image cache
  data.py       dataset, transforms, named evaluation subsets
  models.py     timm models, temperature scaling, checkpoints
  train.py      training loop, early stopping, calibration, threshold selection
  metrics.py    AUROC/AUPRC, operating points, ECE, patient-level bootstrap
  evaluate.py   test-set metrics, plots, subgroup analysis
  gradcam.py    Grad-CAM, randomization sanity check, border-attention score
  explain.py    Grad-CAM galleries + explainability statistics
  summarize.py  results table across runs
app.py          Gradio demo
tests/          unit tests + end-to-end pipeline test on synthetic data
```

## References

- Kermany D. et al. *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning.* Cell, 2018.
- Rajpurkar P. et al. *CheXNet: Radiologist-Level Pneumonia Detection on Chest X-Rays with Deep Learning.* arXiv:1711.05225, 2017.
- Shih G. et al. *Augmenting the NIH Chest Radiograph Dataset with Expert Annotations of Possible Pneumonia.* Radiology: AI, 2019.
- Selvaraju R. et al. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.* ICCV, 2017.
- Adebayo J. et al. *Sanity Checks for Saliency Maps.* NeurIPS, 2018.
- Guo C. et al. *On Calibration of Modern Neural Networks.* ICML, 2017.
- Zech J. et al. *Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs.* PLOS Medicine, 2018.
- DeGrave A. et al. *AI for radiographic COVID-19 detection selects shortcuts over signal.* Nature Machine Intelligence, 2021.
