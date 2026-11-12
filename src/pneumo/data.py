"""Datasets, transforms and named evaluation subsets built on top of `data/manifest.csv`."""
from __future__ import annotations

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(img_size: int, train: bool) -> T.Compose:
    # No horizontal flip: it moves the heart to the right side, an anatomically
    # implausible image (dextrocardia is rare) that the test set never contains.
    aug = [
        T.RandomResizedCrop(img_size, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        T.RandomAffine(degrees=7, translate=(0.04, 0.04)),
        T.ColorJitter(brightness=0.2, contrast=0.2),
    ] if train else [T.Resize((img_size, img_size))]
    return T.Compose([
        *aug,
        T.Grayscale(num_output_channels=3),  # ImageNet backbones expect 3 channels
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """(3,H,W) normalized tensor -> (H,W) grayscale in [0,1], for plotting."""
    mean = torch.tensor(IMAGENET_MEAN, device=x.device)[:, None, None]
    std = torch.tensor(IMAGENET_STD, device=x.device)[:, None, None]
    return (x * std + mean).mean(0).clamp(0, 1)


class CXRDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform: T.Compose):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        img = Image.open(row["path"]).convert("L")
        return self.transform(img), torch.tensor(row["label"], dtype=torch.float32), i


def load_manifest(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def select_subset(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Named subsets used throughout training and evaluation."""
    ped = df[df["source"] == "pediatric"]
    rsna = df[df["source"] == "rsna"]
    subsets = {
        "train": lambda: ped[ped["split"] == "train"],
        "val": lambda: ped[ped["split"] == "val"],
        "test": lambda: ped[ped["split"] == "test"],
        "rsna_opacity_vs_normal": lambda: rsna[rsna["rsna_class"].isin(["Lung Opacity", "Normal"])],
        "rsna_opacity_vs_all": lambda: rsna,
    }
    if name not in subsets:
        raise KeyError(f"unknown subset {name!r}; choose from {sorted(subsets)}")
    return subsets[name]().reset_index(drop=True)


def make_loader(df: pd.DataFrame, img_size: int, train: bool, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        CXRDataset(df, build_transform(img_size, train)),
        batch_size=batch_size,
        shuffle=train,
        drop_last=train and len(df) > batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
