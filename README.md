# Explainable Pneumonia Detection from Chest X-Rays — with External Validation

A chest X-ray pneumonia classifier built to answer a harder question than *"what accuracy can I get?"*:
**does the model still work on a different population, and can its explanations be trusted?**

- **Leakage-free evaluation:** patient-level, stratified train/val/test split.
- **External validation:** trained on pediatric X-rays from one hospital in China, tested on adult X-rays from the US (RSNA).
- **Clinically meaningful metrics:** AUROC and AUPRC with patient-level bootstrap 95% CIs, sensitivity and specificity at operating points frozen on validation, calibration (ECE) before and after temperature scaling, and subgroup analysis by sex, age and view.
- **Explanations that are tested, not just shown:** Grad-CAM plus a weight-randomization sanity check and a border-attention (shortcut) probe — which flagged a problem with the best-performing model (see §4).
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
3. **Border-attention score:** share of Grad-CAM mass in the outer 15% of the image. CXR models are known to latch onto text markers, edges and collimation ([Zech et al., 2018](https://doi.org/10.1371/journal.pmed.1002683); [DeGrave et al., 2021](https://doi.org/10.1038/s42256-021-00338-7)). A uniform map scores ≈ 0.50 at 224 px, and lower means attention on the lung fields.

## 4. Results

Single training run per configuration (seed 42) on a Kaggle T4 GPU; 95% CIs from patient-level bootstrap over the test images (they do **not** include training randomness). A second run (with deterministic GPU operations and fixed seeds) gave identical per-epoch training losses and identical AUROC, AUPRC, sensitivity, specificity, ECE and border scores; only the sanity-check values differ, because that check was changed between runs. Full tables — AUPRC, calibration, explainability scores, subgroups: [`results/RESULTS.md`](results/RESULTS.md).

| Model | **External** RSNA AUROC: opacity vs normal | **External** RSNA AUROC: opacity vs all | Internal test AUROC | Internal Sens / Spec @ Youden |
|---|---|---|---|---|
| v1 setup: MobileNetV2 frozen, 128 px | 0.716 [0.699–0.733] | 0.633 [0.618–0.650] | 0.977 [0.968–0.986] | 0.904 / 0.927 |
| MobileNetV2 fine-tuned, 224 px | 0.904 [0.894–0.914] | 0.774 [0.762–0.788] | 0.995 [0.991–0.998] | 0.959 / 0.978 |
| EfficientNet-B0, 224 px | 0.917 [0.907–0.926] | 0.779 [0.766–0.791] | 0.997 [0.994–0.999] | 0.983 / 0.970 |
| ResNet-50, 224 px | 0.929 [0.920–0.937] | 0.784 [0.771–0.796] | 0.995 [0.988–0.999] | 0.950 / 0.996 |
| DenseNet-121, 224 px | 0.940 [0.932–0.948] | 0.796 [0.783–0.809] | 0.995 [0.991–0.998] | 0.956 / 0.961 |

### Findings

1. **The internal test set is saturated; the external one is informative.** Every fine-tuned model scores ≈ 0.995 internally, so the pediatric test cannot separate them. The Kermany dataset is known to be unusually easy and shortcut-prone, so the internal number is not the headline.
2. **Fixing v1's setup matters most.** Fine-tuning at 224 px instead of a frozen backbone at 128 px raises external AUROC (opacity vs normal) from 0.72 to 0.90 with the same MobileNetV2 backbone.
3. **No architecture is shown to be best.** DenseNet-121 has the highest external AUROC, but each model was trained once and the DenseNet and ResNet-50 intervals overlap; differences of ~0.01 can be within seed-to-seed variation.
4. **The model detects "abnormal", not specifically "pneumonia".** External AUROC drops from 0.94 (opacity vs normal) to 0.80 when RSNA images with *other* abnormalities are counted as negatives (DenseNet-121). Sensitivity stays ≈ 0.90, but specificity falls from 0.96 (internal) to 0.86 (vs normal) and 0.55 (vs all).
5. **Calibration does not transfer.** Temperature scaling fitted on pediatric validation data gives ECE 0.016 internally but 0.14 (vs normal) and 0.42 (vs all) on RSNA, essentially unchanged by calibration. Part of this is prevalence shift (73% pneumonia in training vs 22.5% in the RSNA "vs all" set), which temperature scaling cannot correct.
6. **Projection (AP vs PA) is a confounder.** In RSNA, 74% of AP images but only 17% of PA images are positive. Within a single view, DenseNet's AUROC is 0.88–0.90, below the pooled 0.94, so part of the pooled score comes from telling the two views apart. At the frozen threshold, specificity is 0.45 on AP images and sensitivity 0.67 on PA images. Results by sex and age band are flat (AUROC within ~0.02 for DenseNet).
7. **The best-performing model looks at the image borders.** DenseNet-121's Grad-CAM puts 0.67–0.74 of its mass in the outer 15% band, above the ≈ 0.50 of a uniform map — a typical sign of shortcut features (borders, text markers, or the black padding added when images are made square). On the external set ResNet-50 scores 0.30–0.34 (0.61 internally) with a slightly lower external AUROC, so better external AUROC did not coincide with more plausible attention. DenseNet's randomization sanity correlation is low (|ρ| ≤ 0.14, 128/128 valid comparisons); for ResNet-50 many random-weight maps were blank (44–72 of 128 valid), so its check is less conclusive.

Figures per run in `runs/<run>/eval/<test set>/` (Kaggle output): ROC, PR, reliability diagram, confusion matrix, Grad-CAM gallery, subgroup table.

## 5. Limitations

- **Population shift by design.** The training data are pediatric and come from a single centre. Adults look different (heart size, bone density, devices), so a performance drop on RSNA is expected. The point is to **measure** it, not hide it.
- **Label differences.** Kermany labels are image-level diagnoses. RSNA "Lung Opacity" labels are radiologist-drawn opacities, which are related to pneumonia but not identical.
- **Grad-CAM is coarse** (7×7 feature map at 224 px; 4×4 for the 128-px baseline) and shows correlation, not causation. The border score is a crude proxy; it does not measure overlap with the actual opacity.
- **Single training seed.** Confidence intervals cover test-set sampling only, not training randomness, so architecture rankings are not established.
- **Patient grouping is incomplete.** Normal and pneumonia images use different file-naming schemes, so a child imaged in both classes cannot be linked and could fall into different splits.
- **Sanity check is partial.** Only full-model weight randomization is tested (not cascading, layer-by-layer randomization), on 128 images per test set.
- **No prospective or clinical evaluation**, and no assessment of radiologist + model performance together.

## 6. Future work

- **Lung cropping / masking** (e.g. with a pretrained lung segmentation model) and retraining, to test whether the border attention in finding 7 is a shortcut: compare border attention and external AUROC before vs after.
- **Localization against radiologist boxes:** RSNA provides opacity bounding boxes; measure the share of Grad-CAM mass inside them (cf. Saporta et al., 2022).
- **3–5 training seeds per model** and a paired comparison on the same test images (DeLong or paired bootstrap) before claiming one architecture is better.
- **Prevalence (label-shift) correction** of predicted probabilities for the target population, to separate prevalence shift from genuine miscalibration.
- **View-stratified analysis and training** to reduce reliance on AP/PA differences.

## 7. Reproduce

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
- Saporta A. et al. *Benchmarking saliency methods for chest X-ray interpretation.* Nature Machine Intelligence, 2022.
- Zech J. et al. *Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs.* PLOS Medicine, 2018.
- DeGrave A. et al. *AI for radiographic COVID-19 detection selects shortcuts over signal.* Nature Machine Intelligence, 2021.
