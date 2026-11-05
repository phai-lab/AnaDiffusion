"""Whole-brain volume loading from the combined parts CSV and split JSON."""

from __future__ import annotations

import json
from typing import Optional, Tuple

import pandas as pd
import torch
from monai import transforms
from monai.data import DataLoader, Dataset
from torch.utils.data import DistributedSampler


def build_image_transforms(spacing: Tuple[float, float, float], size: Tuple[int, int, int]):
    keys = ["image"]
    return transforms.Compose(
        [
            transforms.LoadImaged(keys=keys),
            transforms.EnsureChannelFirstd(keys=keys),
            transforms.Lambdad(keys=keys, func=lambda x: x[0]),
            transforms.EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
            transforms.EnsureTyped(keys=keys),
            transforms.Spacingd(keys=keys, pixdim=spacing, mode="bilinear"),
            transforms.DivisiblePadd(keys=keys, k=32, mode="constant", constant_values=-1.0),
            transforms.CenterSpatialCropd(keys=keys, roi_size=size),
        ]
    )


def load_split_records(csv_path: str, split_json_path: str, image_key: str):
    """Return {"train", "val", "test"} -> list of {"image": path, "imageID": id}."""
    with open(split_json_path, "r", encoding="utf-8") as f:
        splits = json.load(f)
    membership = {str(i): name for name in ("train", "val", "test") for i in splits.get(name, [])}
    records = {"train": [], "val": [], "test": []}
    for _, row in pd.read_csv(csv_path).iterrows():
        split = membership.get(str(row["imageID"]))
        if split is not None:
            records[split].append({"image": row[image_key], "imageID": str(row["imageID"])})
    return records


def make_dataloaders(
    *,
    csv_path: str,
    split_json_path: str,
    image_key: str,
    spacing: Tuple[float, float, float],
    size: Tuple[int, int, int],
    batch_size: int,
    num_workers: int,
    distributed: bool,
    seed: int,
) -> Tuple[DataLoader, DataLoader, Optional[DistributedSampler]]:
    records = load_split_records(csv_path, split_json_path, image_key)
    transform = build_image_transforms(spacing, size)
    train_ds = Dataset(data=records["train"], transform=transform)
    val_ds = Dataset(data=records["val"], transform=transform)
    sampler = DistributedSampler(train_ds, shuffle=True, seed=seed) if distributed else None
    loader_kwargs = {"num_workers": int(num_workers), "pin_memory": True}
    if int(num_workers) > 0:
        loader_kwargs["persistent_workers"] = True
    train_loader = DataLoader(
        train_ds, batch_size=int(batch_size), shuffle=sampler is None, sampler=sampler, **loader_kwargs
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, sampler
