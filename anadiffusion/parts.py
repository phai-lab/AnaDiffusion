"""Part crops, frozen part experts, and image-space scaffold assembly."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.amp import autocast

from anadiffusion.data import grid_origin
from anadiffusion.models import HemisphereConditioner, alphas_cumprod, build_ddim_scheduler
from anadiffusion.templates import PART_NAMES, gaussian_blur3d

HEMISPHERE_SIDE = {"lhemi": 0, "rhemi": 1}
_PAD = 40
# The preprocessed scans share a 1 mm MNI152 grid of NATIVE_SHAPE voxels. The part
# images were cut from it at these (start, shape) boxes before being resampled and
# padded on their own (data_prep/prep_data.py); the right hemisphere is cut from the
# left-right mirrored scan.
NATIVE_SHAPE = (160, 192, 160)
NATIVE_PART_BOXES = {
    "lhemi": ((64, 0, 0), (96, 192, 160)),
    "rhemi": ((64, 0, 0), (96, 192, 160)),
    "sub": ((0, 0, 0), (160, 128, 96)),
}


def flip_lr(x: torch.Tensor) -> torch.Tensor:
    """Mirror along the left-right axis (first spatial axis)."""
    return torch.flip(x, dims=[2])


def mask_background(x: torch.Tensor, mask: torch.Tensor, bg_value: float) -> torch.Tensor:
    m = (mask > 0.5).float()
    return x * m + bg_value * (1.0 - m)


def part_boxes(
    spacing: Tuple[float, float, float],
    whole_shape: Tuple[int, int, int],
    hemi_shape: Tuple[int, int, int],
    sub_shape: Tuple[int, int, int],
) -> Dict[str, Tuple[Tuple[int, int, int], Tuple[int, int, int]]]:
    """Crop boxes (start, size) of the three parts on the whole-brain grid.

    Each box covers the same anatomy as the part images the experts were trained
    on, voxel for voxel. The part images were resampled and padded separately from
    the whole-brain scans, so their grids are offset from the whole-brain grid; the
    offset is computed here and rounded to the nearest voxel.
    """
    origin = grid_origin(NATIVE_SHAPE, spacing, whole_shape)
    # left-right position of whole-brain voxel 0 once the scan is mirrored
    mirrored_origin = NATIVE_SHAPE[0] - 1 - origin[0] - spacing[0] * (whole_shape[0] - 1)
    boxes = {}
    for name in PART_NAMES:
        size = tuple(sub_shape if name == "sub" else hemi_shape)
        native_start, native_shape = NATIVE_PART_BOXES[name]
        part_origin = grid_origin(native_shape, spacing, size)
        start = [(native_start[a] + part_origin[a] - origin[a]) / spacing[a] for a in range(3)]
        if name == "rhemi":
            # the crop is mirrored after cropping, so the box is placed in the mirrored scan
            start[0] = whole_shape[0] - size[0] - (native_start[0] + part_origin[0] - mirrored_origin) / spacing[0]
        boxes[name] = (tuple(int(round(s)) for s in start), size)
    return boxes


def _clamp_start(start, size, shape):
    return [int(min(max(int(v), -_PAD), int(n) + _PAD - int(s))) for v, s, n in zip(start, size, shape)]


def crop_padded(x: torch.Tensor, start, size, bg_value: float) -> torch.Tensor:
    """Crop a box that may extend outside the volume; outside voxels are background."""
    start = _clamp_start(start, size, x.shape[-3:])
    xp = F.pad(x, (_PAD, _PAD) * 3, value=bg_value)
    s = [v + _PAD for v in start]
    return xp[:, :, s[0] : s[0] + size[0], s[1] : s[1] + size[1], s[2] : s[2] + size[2]]


def paste_padded(canvas: torch.Tensor, patch: torch.Tensor, start) -> torch.Tensor:
    """Inverse of crop_padded: write `patch` at `start` into a copy of `canvas`."""
    start = _clamp_start(start, patch.shape[-3:], canvas.shape[-3:])
    cp = F.pad(canvas, (_PAD, _PAD) * 3, value=0.0)
    s = [v + _PAD for v in start]
    cp[:, :, s[0] : s[0] + patch.shape[2], s[1] : s[1] + patch.shape[3], s[2] : s[2] + patch.shape[4]] = patch
    d, h, w = canvas.shape[-3:]
    return cp[:, :, _PAD : _PAD + d, _PAD : _PAD + h, _PAD : _PAD + w]


class PartExperts:
    """Frozen part autoencoders and LDMs that refine part crops by latent SDEdit."""

    def __init__(
        self,
        *,
        hemi_ae,
        hemi_unet,
        hemi_conditioner: HemisphereConditioner,
        sub_ae,
        sub_unet,
        hemi_scale_factor: float,
        sub_scale_factor: float,
        ddim_steps: int,
        device: torch.device,
        amp: bool = True,
    ):
        self.hemi_ae, self.hemi_unet, self.hemi_conditioner = hemi_ae, hemi_unet, hemi_conditioner
        self.sub_ae, self.sub_unet = sub_ae, sub_unet
        self.scale = {"lhemi": float(hemi_scale_factor), "rhemi": float(hemi_scale_factor), "sub": float(sub_scale_factor)}
        self.device = device
        self.amp = bool(amp)
        self.scheduler = build_ddim_scheduler(ddim_steps)
        self.timesteps = [int(t) for t in self.scheduler.timesteps]
        self.abar = alphas_cumprod(device)

    def _models(self, part: str, batch: int = 1):
        if part == "sub":
            return self.sub_ae, self.sub_unet, None
        side = torch.full((batch,), HEMISPHERE_SIDE[part], device=self.device, dtype=torch.long)
        return self.hemi_ae, self.hemi_unet, self.hemi_conditioner(side)

    @torch.no_grad()
    def refine(
        self,
        x_crop: torch.Tensor,
        part: str,
        t_part: int,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """SDEdit a part crop: encode, forward-noise to t_part, denoise with the part LDM, decode.

        Right-hemisphere crops are expected in the left-canonical orientation.
        """
        ae, unet, context = self._models(part, x_crop.shape[0])
        scale = self.scale[part]
        grid = [t for t in self.timesteps if t <= int(t_part)]
        with autocast(device_type="cuda", enabled=self.amp):
            z0 = (ae.encode_stage_2_inputs(x_crop) * scale).float()
        ab = self.abar[grid[0]]
        noise = torch.randn(z0.shape, generator=generator, device=self.device)
        z = ab.sqrt() * z0 + (1.0 - ab).sqrt() * noise
        for t in grid:
            tb = torch.full((z.shape[0],), t, device=self.device, dtype=torch.long)
            with autocast(device_type="cuda", enabled=self.amp):
                eps = unet(z, timesteps=tb, context=context) if context is not None else unet(z, timesteps=tb)
            z = self.scheduler.step(model_output=eps.float(), timestep=t, sample=z)[0].float()
        with autocast(device_type="cuda", enabled=self.amp):
            return ae.decode_stage_2_outputs(z / scale).float()


def assemble(
    x_scaffold: torch.Tensor,
    part_images: Dict[str, torch.Tensor],
    supports: Dict[str, torch.Tensor],
    feather_sigma: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Feather-blend the part assets over the scaffold inside their supports.

    Returns the assembled image and the blend weight w in [0, 1]; voxels with
    w = 0 (CSF, background, omitted boundary regions) keep the scaffold content.
    """
    weights = {}
    for name in PART_NAMES:
        support = supports[name]
        weights[name] = support * gaussian_blur3d(support, feather_sigma).clamp(0.0, 1.0)
    total = sum(weights.values())
    x_parts = sum(weights[n] * part_images[n] for n in PART_NAMES) / total.clamp_min(1e-6)
    w = total.clamp(0.0, 1.0)
    return (1.0 - w) * x_scaffold + w * x_parts, w
