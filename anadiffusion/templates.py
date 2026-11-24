"""MNI152 template masks and binary morphology on [B,1,D,H,W] tensors."""

from __future__ import annotations

import math
import os
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from monai import transforms

from anadiffusion.data import grid_transforms

PART_NAMES = ("lhemi", "rhemi", "sub")
MASK_FILES = {"lhemi": "lhemi_mask.nii.gz", "rhemi": "rhemi_mask.nii.gz", "sub": "sub_mask.nii.gz"}


def load_template_masks(
    template_dir: str,
    spacing: Tuple[float, float, float],
    whole_shape: Tuple[int, int, int],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Load the three part templates on the whole-brain grid.

    The templates are 1 mm volumes in the same MNI152 space as the preprocessed
    scans, so they are resampled, padded, and cropped exactly like the images.
    Returns {"lhemi", "rhemi", "sub"} -> float masks of shape [1,1,D,H,W].
    """
    paths = {name: os.path.join(template_dir, MASK_FILES[name]) for name in PART_NAMES}
    for path in paths.values():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Template mask not found: {path}")
    keys = list(PART_NAMES)
    transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=keys),
            transforms.EnsureChannelFirstd(keys=keys),
            transforms.Lambdad(keys=keys, func=lambda x: x[0]),
            transforms.EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
            transforms.EnsureTyped(keys=keys),
            *grid_transforms(keys, spacing, whole_shape, mode=("nearest",) * len(keys), pad_value=0.0),
            transforms.Lambdad(keys=keys, func=lambda x: (x > 0).float()),
        ]
    )
    loaded = transform(paths)
    return {name: torch.as_tensor(loaded[name]).float().unsqueeze(0).to(device) for name in PART_NAMES}


def dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    return F.max_pool3d(mask, 2 * radius + 1, 1, radius)


def erode(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    return 1.0 - F.max_pool3d(1.0 - mask, 2 * radius + 1, 1, radius)


def binary_close(mask: torch.Tensor, radius: int) -> torch.Tensor:
    return erode(dilate(mask, radius), radius)


def fill_holes(mask: torch.Tensor, max_iters: int = 256) -> torch.Tensor:
    """Fill enclosed background (ventricles, sulci) by flooding from the volume border."""
    free = 1.0 - mask
    outside = torch.zeros_like(mask)
    for axis in (2, 3, 4):
        for edge in (0, -1):
            index = [slice(None)] * 5
            index[axis] = edge
            outside[tuple(index)] = free[tuple(index)]
    for i in range(max_iters):
        grown = F.max_pool3d(outside, 3, 1, 1) * free
        if i % 8 == 7 and torch.equal(grown, outside):
            break
        outside = grown
    return 1.0 - outside


def boundary_band(mask: torch.Tensor, width: int) -> torch.Tensor:
    """Voxels within `width` of the mask boundary (dilation minus erosion)."""
    return ((dilate(mask, width) - erode(mask, width)) > 0.5).float()


def gaussian_blur3d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x
    radius = max(1, int(math.ceil(3 * sigma)))
    t = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-(t**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    for dim in range(3):
        shape = [1, 1, 1, 1, 1]
        shape[2 + dim] = kernel.numel()
        pad = [0] * 6
        pad[2 * (2 - dim)] = radius
        pad[2 * (2 - dim) + 1] = radius
        x = F.conv3d(F.pad(x, pad, mode="replicate"), kernel.view(shape))
    return x


def brain_envelope(x: torch.Tensor, bg_value: float, margin: float = 0.1) -> torch.Tensor:
    """Solid head support of an image: tissue above background, closed and hole-filled."""
    return fill_holes(binary_close((x > bg_value + margin).float(), 2))
