# Results

Kaggle runs: GPU T4, ~30 min for all five models (run 2 gave identical training losses and performance metrics to run 1; the sanity check was revised in between, adding validity counts, and subgroups were added). Internal test: 886 images / 444 patients (pediatric). External: 6,000 RSNA studies (3,342 Lung Opacity + Normal).

| Run | Test set | n | AUROC [95% CI] | AUPRC [95% CI] | Sensitivity* | Specificity* | ECE before → after calibration | Border attention† | Sanity ρ‡ |
|---|---|---|---|---|---|---|---|---|---|
| densenet121_224 | rsna_opacity_vs_all | 6000 | 0.796 [0.783–0.809] | 0.486 [0.459–0.515] | 0.899 [0.882–0.916] | 0.554 [0.539–0.568] | 0.423 → 0.424 | 0.72 | 0.14 (128/128 valid) |
| densenet121_224 | rsna_opacity_vs_normal | 3342 | 0.940 [0.932–0.948] | 0.916 [0.902–0.930] | 0.899 [0.883–0.916] | 0.859 [0.843–0.874] | 0.142 → 0.141 | 0.74 | 0.14 (128/128 valid) |
| densenet121_224 | test | 886 | 0.995 [0.991–0.998] | 0.998 [0.997–0.999] | 0.956 [0.939–0.970] | 0.961 [0.936–0.983] | 0.015 → 0.016 | 0.67 | -0.05 (128/128 valid) |
| efficientnet_b0_224 | rsna_opacity_vs_all | 6000 | 0.779 [0.766–0.791] | 0.439 [0.417–0.463] | 0.933 [0.919–0.946] | 0.432 [0.418–0.446] | 0.418 → 0.412 | 0.50 | -0.07 (128/128 valid) |
| efficientnet_b0_224 | rsna_opacity_vs_normal | 3342 | 0.917 [0.907–0.926] | 0.873 [0.855–0.890] | 0.933 [0.919–0.945] | 0.692 [0.673–0.712] | 0.170 → 0.158 | 0.53 | -0.11 (127/128 valid) |
| efficientnet_b0_224 | test | 886 | 0.997 [0.994–0.999] | 0.999 [0.998–1.000] | 0.983 [0.973–0.992] | 0.970 [0.945–0.991] | 0.022 → 0.022 | 0.39 | 0.02 (128/128 valid) |
| mobilenetv2_224 | rsna_opacity_vs_all | 6000 | 0.774 [0.762–0.788] | 0.460 [0.434–0.488] | 0.907 [0.891–0.922] | 0.472 [0.457–0.486] | 0.456 → 0.451 | 0.45 | 0.11 (128/128 valid) |
| mobilenetv2_224 | rsna_opacity_vs_normal | 3342 | 0.904 [0.894–0.914] | 0.866 [0.848–0.883] | 0.907 [0.892–0.922] | 0.725 [0.705–0.743] | 0.207 → 0.200 | 0.45 | 0.07 (128/128 valid) |
| mobilenetv2_224 | test | 886 | 0.995 [0.991–0.998] | 0.998 [0.997–0.999] | 0.959 [0.940–0.974] | 0.978 [0.958–0.996] | 0.018 → 0.013 | 0.47 | -0.02 (128/128 valid) |
| mobilenetv2_frozen_128 | rsna_opacity_vs_all | 6000 | 0.633 [0.618–0.650] | 0.316 [0.296–0.340] | 0.798 [0.777–0.819] | 0.369 [0.355–0.383] | 0.545 → 0.470 | 0.50 | 0.18 (126/128 valid) |
| mobilenetv2_frozen_128 | rsna_opacity_vs_normal | 3342 | 0.716 [0.699–0.733] | 0.616 [0.588–0.645] | 0.798 [0.776–0.820] | 0.487 [0.467–0.509] | 0.376 → 0.270 | 0.51 | 0.18 (128/128 valid) |
| mobilenetv2_frozen_128 | test | 886 | 0.977 [0.968–0.986] | 0.992 [0.987–0.995] | 0.904 [0.878–0.927] | 0.927 [0.885–0.963] | 0.069 → 0.049 | 0.49 | 0.18 (126/128 valid) |
| resnet50_224 | rsna_opacity_vs_all | 6000 | 0.784 [0.771–0.796] | 0.442 [0.417–0.470] | 0.924 [0.909–0.937] | 0.505 [0.491–0.519] | 0.456 → 0.464 | 0.30 | 0.07 (72/128 valid) |
| resnet50_224 | rsna_opacity_vs_normal | 3342 | 0.929 [0.920–0.937] | 0.883 [0.864–0.901] | 0.924 [0.910–0.937] | 0.804 [0.786–0.821] | 0.182 → 0.187 | 0.34 | 0.06 (68/128 valid) |
| resnet50_224 | test | 886 | 0.995 [0.988–0.999] | 0.997 [0.993–1.000] | 0.950 [0.932–0.966] | 0.996 [0.987–1.000] | 0.019 → 0.014 | 0.61 | 0.10 (44/128 valid) |

\* At the Youden-optimal threshold chosen on the validation set and frozen for all test sets.  
† Share of Grad-CAM mass in the outer 15% border band (a uniform map scores ≈ 0.50 at 224 px; lower = attention on the lung fields).  
‡ Spearman correlation between Grad-CAM maps of the trained model and of the same network with random weights (Adebayo et al., 2018); close to 0 = maps depend on what was learned. Images where either map is blank have no defined correlation and are excluded; the count of valid comparisons is shown in brackets.

## Subgroups — RSNA, rsna_opacity_vs_normal

Point estimates only (no CIs); groups with < 30 images or a single class are omitted. Age bands from DICOM metadata. Sensitivity/specificity at the same frozen Youden threshold.

| Model | Attribute | Group | n | Prevalence | AUROC | Sensitivity* | Specificity* |
|---|---|---|---|---|---|---|---|
| densenet121_224 | sex | F | 1390 | 0.39 | 0.941 | 0.894 | 0.863 |
| densenet121_224 | sex | M | 1952 | 0.42 | 0.940 | 0.903 | 0.857 |
| densenet121_224 | age_band | <18 | 192 | 0.48 | 0.932 | 0.914 | 0.788 |
| densenet121_224 | age_band | 18-39 | 1027 | 0.39 | 0.945 | 0.895 | 0.877 |
| densenet121_224 | age_band | 40-59 | 1424 | 0.39 | 0.946 | 0.904 | 0.859 |
| densenet121_224 | age_band | 60+ | 699 | 0.44 | 0.925 | 0.894 | 0.848 |
| densenet121_224 | view | AP | 1389 | 0.74 | 0.884 | 0.974 | 0.448 |
| densenet121_224 | view | PA | 1953 | 0.17 | 0.899 | 0.667 | 0.951 |
| efficientnet_b0_224 | view | AP | 1389 | 0.74 | 0.837 | 0.996 | 0.121 |
| efficientnet_b0_224 | view | PA | 1953 | 0.17 | 0.855 | 0.734 | 0.820 |
| mobilenetv2_224 | view | AP | 1389 | 0.74 | 0.819 | 0.984 | 0.209 |
| mobilenetv2_224 | view | PA | 1953 | 0.17 | 0.826 | 0.664 | 0.840 |
| mobilenetv2_frozen_128 | view | AP | 1389 | 0.74 | 0.587 | 0.852 | 0.223 |
| mobilenetv2_frozen_128 | view | PA | 1953 | 0.17 | 0.629 | 0.630 | 0.547 |
| resnet50_224 | view | AP | 1389 | 0.74 | 0.822 | 0.985 | 0.297 |
| resnet50_224 | view | PA | 1953 | 0.17 | 0.903 | 0.731 | 0.918 |

Sex and age subgroups for the other models are similarly flat (within ~0.04 AUROC) and are in each run's `eval/rsna_opacity_vs_normal/subgroups.csv`.
