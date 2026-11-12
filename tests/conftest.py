"""Synthetic stand-ins for both datasets, laid out exactly like the real Kaggle downloads."""
from __future__ import annotations

import zlib

import numpy as np
import pandas as pd
import pytest
from PIL import Image


def _xray(rng: np.random.Generator, sick: bool, size: int = 96) -> np.ndarray:
    img = rng.normal(90, 20, (size, size + 16))
    if sick:  # a bright blob in a lung field, so the task is learnable
        cy, cx = rng.integers(30, 60), rng.integers(25, 80)
        yy, xx = np.ogrid[:size, :size + 16]
        img += 90 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / 120)
    return np.clip(img, 0, 255).astype(np.uint8)


@pytest.fixture(scope="session")
def pediatric_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("raw") / "chest_xray"
    rng = np.random.default_rng(0)
    for copy in (root / "chest_xray", root / "chest_xray" / "chest_xray"):  # Kaggle zip ships a nested duplicate
        for split, n_patients, offset in [("train", 30, 0), ("val", 2, 100), ("test", 8, 200)]:
            for cls in ["NORMAL", "PNEUMONIA"]:
                d = copy / split / cls
                d.mkdir(parents=True, exist_ok=True)
                for pid in range(n_patients):
                    for k in range(2):  # two images per patient -> leakage if split by image
                        name = (f"person{offset + pid}_bacteria_{k}.jpeg" if cls == "PNEUMONIA"
                                else f"IM-{offset + pid:04d}-000{k}.jpeg")
                        if not (d / name).exists():
                            Image.fromarray(_xray(np.random.default_rng(zlib.crc32(name.encode())), cls == "PNEUMONIA")).save(d / name)
    (root / "__MACOSX").mkdir()
    Image.fromarray(_xray(rng, False)).save(root / "__MACOSX" / "junk.jpeg")
    return root


@pytest.fixture(scope="session")
def rsna_root(tmp_path_factory):
    from pydicom.dataset import FileDataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

    root = tmp_path_factory.mktemp("rsna")
    img_dir = root / "stage_2_train_images"
    img_dir.mkdir()
    rng = np.random.default_rng(1)
    labels, classes = [], []
    for i in range(60):
        cls = ["Lung Opacity", "Normal", "No Lung Opacity / Not Normal"][i % 3]
        pid = f"rsna-{i:04d}"
        target = int(cls == "Lung Opacity")
        labels += [{"patientId": pid, "x": None, "y": None, "width": None, "height": None, "Target": target}] * (1 + target)
        classes.append({"patientId": pid, "class": cls})

        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = FileDataset(str(img_dir / f"{pid}.dcm"), {}, file_meta=meta, preamble=b"\0" * 128)
        arr = _xray(rng, target == 1, size=128)[:, :128]
        ds.Rows, ds.Columns = arr.shape
        ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 8, 8, 7, 0
        ds.PatientSex, ds.PatientAge, ds.ViewPosition = ["M", "F"][i % 2], f"{20 + i:03d}Y", ["PA", "AP"][i % 2]
        ds.PixelData = arr.tobytes()
        ds.save_as(img_dir / f"{pid}.dcm", enforce_file_format=True)
    pd.DataFrame(labels).to_csv(root / "stage_2_train_labels.csv", index=False)
    pd.DataFrame(classes).to_csv(root / "stage_2_detailed_class_info.csv", index=False)
    return root
