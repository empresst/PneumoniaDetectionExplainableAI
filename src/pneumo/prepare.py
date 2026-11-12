"""Build the data manifest and a cache of pre-resized images.

Pediatric set (Kermany et al., Kaggle "chest-xray-pneumonia"):
    The official train/val/test folders are pooled and re-split *by patient*, because
    the same child can appear under several file names and the official val split has
    only 16 images. Patient IDs are parsed from file names.

RSNA set (RSNA Pneumonia Detection Challenge, adult CXRs from NIH ChestX-ray14):
    Used only as an external test set - never for training, tuning or thresholding.

Usage:
    python -m pneumo.prepare --pediatric data/raw/chest_xray --rsna data/raw/rsna \
        --out data --rsna-n 6000
"""
from __future__ import annotations

import argparse
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

IMG_EXTS = {".jpeg", ".jpg", ".png"}
_PNEUMONIA_ID = re.compile(r"^(person\d+)_", re.IGNORECASE)
_NORMAL_ID = re.compile(r"^((?:NORMAL2-)?IM-\d+)-", re.IGNORECASE)


def pediatric_patient_id(filename: str) -> str:
    """`person1000_virus_1681.jpeg` -> `person1000`, `NORMAL2-IM-0427-0001.jpeg` -> `NORMAL2-IM-0427`."""
    stem = Path(filename).stem
    for pattern in (_PNEUMONIA_ID, _NORMAL_ID):
        m = pattern.match(stem)
        if m:
            return m.group(1)
    return stem  # unknown naming: treat the image as its own patient


def scan_pediatric(root: Path) -> pd.DataFrame:
    rows, seen = [], set()
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in IMG_EXTS or "__MACOSX" in p.parts:
            continue
        cls = p.parent.name.upper()
        if cls not in {"NORMAL", "PNEUMONIA"}:
            continue
        key = (cls, p.name)  # the Kaggle zip contains a nested duplicate copy of the dataset
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            "src_path": str(p),
            "source": "pediatric",
            "patient_id": pediatric_patient_id(p.name),
            "label": int(cls == "PNEUMONIA"),
            "orig_split": p.parent.parent.name.lower(),
        })
    if not rows:
        raise FileNotFoundError(f"No NORMAL/PNEUMONIA images found under {root}")
    return pd.DataFrame(rows)


def patient_level_split(df: pd.DataFrame, test_frac: float, val_frac: float, seed: int) -> pd.Series:
    """Stratified, grouped split: no patient appears in more than one of train/val/test."""
    split = pd.Series("train", index=df.index)
    sgkf = StratifiedGroupKFold(n_splits=round(1 / test_frac), shuffle=True, random_state=seed)
    _, test_idx = next(sgkf.split(df, df["label"], df["patient_id"]))
    split.iloc[test_idx] = "test"

    rest = df.index[split == "train"]
    val_share = val_frac / (1 - test_frac)
    sgkf = StratifiedGroupKFold(n_splits=round(1 / val_share), shuffle=True, random_state=seed + 1)
    _, val_pos = next(sgkf.split(df.loc[rest], df.loc[rest, "label"], df.loc[rest, "patient_id"]))
    split.loc[rest[val_pos]] = "val"
    return split


def _parse_age(value) -> float:
    """DICOM ages come as '51' (RSNA) or '051Y' (DICOM AS format)."""
    m = re.match(r"^\s*(\d+)", str(value or ""))
    return float(m.group(1)) if m else float("nan")


def scan_rsna(root: Path, n: int, seed: int) -> pd.DataFrame:
    import pydicom

    labels = pd.read_csv(next(root.rglob("stage_2_train_labels.csv")))
    classes = pd.read_csv(next(root.rglob("stage_2_detailed_class_info.csv")))
    df = (labels.groupby("patientId")["Target"].max().rename("label").reset_index()
          .merge(classes.drop_duplicates("patientId"), on="patientId")
          .rename(columns={"class": "rsna_class"}))
    if n and n < len(df):  # stratified subsample to keep caching time reasonable
        df = df.groupby("rsna_class").sample(frac=n / len(df), random_state=seed)
    img_dir = next(root.rglob("stage_2_train_images"))

    rows = []
    for r in df.itertuples():
        path = img_dir / f"{r.patientId}.dcm"
        if not path.exists():
            continue
        meta = pydicom.dcmread(path, stop_before_pixels=True)
        rows.append({
            "src_path": str(path),
            "source": "rsna",
            "patient_id": r.patientId,
            "label": int(r.label),
            "rsna_class": r.rsna_class,  # "Lung Opacity" | "Normal" | "No Lung Opacity / Not Normal"
            "sex": getattr(meta, "PatientSex", None),
            "age": _parse_age(getattr(meta, "PatientAge", None)),
            "view": getattr(meta, "ViewPosition", None),
            "split": "external",
        })
    return pd.DataFrame(rows)


def load_grayscale(path: str) -> Image.Image:
    if path.lower().endswith(".dcm"):
        import pydicom

        ds = pydicom.dcmread(path)
        arr = ds.pixel_array.astype(np.float32)
        max_value = 2 ** int(getattr(ds, "BitsStored", 8)) - 1
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            arr = max_value - arr
        # Fixed rescaling by bit depth, never per-image min-max: stretching each image
        # separately changes intensity statistics relative to the JPEG training data.
        arr = arr / max_value * 255
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    return Image.open(path).convert("L")


def letterbox(img: Image.Image, size: int) -> Image.Image:
    """Pad to square (keeps anatomy proportions), then resize."""
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("L", (side, side), 0)
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    return canvas.resize((size, size), Image.BICUBIC)


def _cache_one(args: tuple[str, str, int]) -> None:
    src, dst, size = args
    if not Path(dst).exists():
        letterbox(load_grayscale(src), size).save(dst)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pediatric", type=Path, required=True)
    ap.add_argument("--rsna", type=Path, help="optional external test set")
    ap.add_argument("--rsna-n", type=int, default=6000, help="0 = use all ~26.7k RSNA images")
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--cache-size", type=int, default=320)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    ped = scan_pediatric(args.pediatric)
    ped["split"] = patient_level_split(ped, args.test_frac, args.val_frac, args.seed)
    frames = [ped]
    if args.rsna:
        frames.append(scan_rsna(args.rsna, args.rsna_n, args.seed))
    df = pd.concat(frames, ignore_index=True)

    cache_dir = args.out / f"cache_{args.cache_size}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    df["path"] = [str(cache_dir / f"{s}_{i:06d}.png") for i, s in zip(df.index, df["source"])]
    jobs = [(s, d, args.cache_size) for s, d in zip(df["src_path"], df["path"])]
    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(_cache_one, jobs, chunksize=64))

    df.to_csv(args.out / "manifest.csv", index=False)
    print(df.groupby(["source", "split"])["label"].agg(images="size", positives="sum"))
    print("patients per split:", ped.groupby("split")["patient_id"].nunique().to_dict())


if __name__ == "__main__":
    main()
