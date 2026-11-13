import numpy as np
import pytest
import torch

from pneumo.gradcam import border_energy, gradcam
from pneumo.metrics import bootstrap_ci, choose_thresholds, expected_calibration_error, point_metrics
from pneumo.models import build_model, fit_temperature
from pneumo.prepare import patient_level_split, pediatric_patient_id, scan_pediatric


@pytest.mark.parametrize("name, expected", [
    ("person1000_virus_1681.jpeg", "person1000"),
    ("person3_bacteria_12.jpeg", "person3"),
    ("IM-0115-0001.jpeg", "IM-0115"),
    ("NORMAL2-IM-0427-0001.jpeg", "NORMAL2-IM-0427"),
    ("weird_name.jpeg", "weird_name"),
])
def test_patient_id(name, expected):
    assert pediatric_patient_id(name) == expected


def test_scan_dedupes_nested_copy_and_skips_macosx(pediatric_root):
    df = scan_pediatric(pediatric_root)
    assert len(df) == 2 * 2 * (30 + 2 + 8)  # classes x images/patient x patients (duplicate copy dropped)
    assert df.patient_id.nunique() == 2 * (30 + 2 + 8)
    assert not df.src_path.str.contains("__MACOSX").any()


def test_split_has_no_patient_leakage(pediatric_root):
    df = scan_pediatric(pediatric_root)
    df["split"] = patient_level_split(df, 0.15, 0.15, seed=0)
    owners = df.groupby("patient_id")["split"].nunique()
    assert (owners == 1).all()
    assert set(df.split) == {"train", "val", "test"}
    for _, g in df.groupby("split"):
        assert 0.3 < g.label.mean() < 0.7  # stratified


def test_ece_and_thresholds():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=20000)
    y = (rng.uniform(size=p.size) < p).astype(int)  # perfectly calibrated by construction
    assert expected_calibration_error(y, p) < 0.02
    th = choose_thresholds(y, p, 0.95)
    assert point_metrics(y, p, th["sens>=0.95"])["sensitivity"] >= 0.95


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 400)
    p = np.clip(y * 0.3 + rng.uniform(size=400) * 0.7, 0, 1)
    ci = bootstrap_ci(y, p, np.arange(400) // 2, {"youden": 0.5}, n_boot=200)
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y, p)
    assert ci["auroc"][0] < auc < ci["auroc"][1]


def test_temperature_recovers_overconfidence():
    rng = np.random.default_rng(2)
    true_logits = rng.normal(0, 2, 5000)
    y = (rng.uniform(size=5000) < 1 / (1 + np.exp(-true_logits))).astype(float)
    assert abs(fit_temperature(true_logits * 3, y) - 3) < 0.3


def test_gradcam_shapes_and_range():
    model = build_model("resnet18", pretrained=False)
    cams, logits = gradcam(model, torch.randn(2, 3, 64, 64))
    assert cams.shape == (2, 64, 64) and logits.shape == (2,)
    assert cams.min() >= 0 and cams.max() <= 1 + 1e-6
    uniform = torch.ones(1, 100, 100)
    assert abs(border_energy(uniform)[0] - (1 - 0.7 ** 2)) < 0.01


def test_frozen_backbone_only_trains_head():
    model = build_model("resnet18", pretrained=False, freeze_backbone=True)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {"fc.weight", "fc.bias"}


def test_dicom_keeps_raw_8bit_intensities(rsna_root):
    """Regression: per-image min-max stretching made RSNA intensities incomparable to the JPEGs."""
    import pydicom

    from pneumo.prepare import load_grayscale

    path = next((rsna_root / "stage_2_train_images").glob("*.dcm"))
    raw = pydicom.dcmread(path).pixel_array
    assert np.array_equal(np.asarray(load_grayscale(str(path))), raw)


def test_sanity_check_counts_blank_maps_instead_of_passing():
    """Regression: blank random-weight maps gave NaN correlations that were silently turned into 0."""
    from torch import nn

    from pneumo.gradcam import randomization_sanity_check

    class Dead(nn.Module):  # feature map is all zeros -> blank Grad-CAM for every image
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 2, 8, stride=8)

        def forward_features(self, x):
            return self.conv(x) * 0

        def forward_head(self, f):
            return f.flatten(1).sum(1, keepdim=True)

        def forward(self, x):
            return self.forward_head(self.forward_features(x))

    out = randomization_sanity_check(Dead(), torch.randn(5, 3, 32, 32), batch_size=2)
    assert out["n_valid"] == 0 and out["n_blank_trained"] == 5
    assert np.isnan(out["spearman_mean"])  # not 0.0
