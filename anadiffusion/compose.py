"""Part-to-whole scaffold injection shared by compositional training and inference."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.amp import autocast

from anadiffusion.models import NUM_TRAIN_TIMESTEPS, alphas_cumprod, build_ddim_scheduler
from anadiffusion.parts import (
    PartExperts,
    assemble,
    crop_padded,
    flip_lr,
    mask_background,
    part_boxes,
    paste_padded,
)
from anadiffusion.templates import PART_NAMES, boundary_band, brain_envelope, erode


class ScaffoldInjector:
    """Builds the anatomical scaffold from a partially denoised whole-brain latent and injects it.

    Given the whole-brain latent z_t at timestep t:
      1. the whole-brain denoiser's clean estimate x0_hat is decoded,
      2. the three parts are cropped with the template masks and refined by the
         frozen part experts (latent SDEdit from t_part = min(part_t_ratio * t, part_t_max)),
      3. the refined parts are assembled over the decoded estimate, re-encoded with
         the whole-brain autoencoder, and injected into z_t inside the part supports
         while the noise component of z_t is kept.
    """

    def __init__(
        self,
        *,
        whole_ae,
        whole_unet,
        whole_scale_factor: float,
        experts: PartExperts,
        crop_masks: Dict[str, torch.Tensor],
        paste_masks: Dict[str, torch.Tensor],
        whole_shape: Tuple[int, int, int],
        hemi_shape: Tuple[int, int, int],
        sub_shape: Tuple[int, int, int],
        part_t_ratio: float,
        part_t_max: int = NUM_TRAIN_TIMESTEPS - 1,
        cerebellum_erosion: int = 4,
        feather_sigma: float = 1.5,
        tissue_margin: float = 0.1,
        bg_value: float = -1.0,
        device: torch.device,
        amp: bool = True,
    ):
        self.whole_ae, self.whole_unet = whole_ae, whole_unet
        self.scale = float(whole_scale_factor)
        self.experts = experts
        self.crop_masks, self.paste_masks = crop_masks, paste_masks
        self.boxes = part_boxes(whole_shape, hemi_shape, sub_shape)
        self.part_t_ratio = float(part_t_ratio)
        self.part_t_max = int(min(part_t_max, NUM_TRAIN_TIMESTEPS - 1))
        self.cerebellum_erosion = int(cerebellum_erosion)
        self.feather_sigma = float(feather_sigma)
        self.tissue_margin = float(tissue_margin)
        self.bg = float(bg_value)
        self.device = device
        self.amp = bool(amp)
        self.abar = alphas_cumprod(device)
        with torch.no_grad():
            probe = torch.full((1, 1, *whole_shape), self.bg, device=device)
            self.latent_shape = tuple(self.encode(probe).shape[1:])

    # ----------------------------------------------------------------- whole-brain model
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        with autocast(device_type="cuda", enabled=self.amp):
            return (self.whole_ae.encode_stage_2_inputs(x) * self.scale).float()

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        with autocast(device_type="cuda", enabled=self.amp):
            return self.whole_ae.decode_stage_2_outputs(z / self.scale).float()

    def eps(self, z: torch.Tensor, t: int) -> torch.Tensor:
        tb = torch.full((z.shape[0],), int(t), device=z.device, dtype=torch.long)
        with autocast(device_type="cuda", enabled=self.amp):
            return self.whole_unet(z, timesteps=tb).float()

    @torch.no_grad()
    def clean_estimate(self, z_t: torch.Tensor, t: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (x0_hat latent, decoded image) predicted by the whole-brain denoiser."""
        ab = self.abar[int(t)]
        x0_hat = (z_t - (1.0 - ab).sqrt() * self.eps(z_t, t)) / ab.sqrt()
        return x0_hat, self.decode(x0_hat)

    # --------------------------------------------------------------------- part assets
    def part_timestep(self, t: int, part_t_ratio: Optional[float] = None) -> int:
        ratio = self.part_t_ratio if part_t_ratio is None else float(part_t_ratio)
        return int(min(round(ratio * int(t)), self.part_t_max))

    def crop(self, x: torch.Tensor, part: str) -> torch.Tensor:
        start, size = self.boxes[part]
        crop = crop_padded(mask_background(x, self.crop_masks[part], self.bg), start, size, self.bg)
        return flip_lr(crop) if part == "rhemi" else crop

    def paste(self, part_image: torch.Tensor, part: str, like: torch.Tensor) -> torch.Tensor:
        start, _ = self.boxes[part]
        patch = flip_lr(part_image) if part == "rhemi" else part_image
        return paste_padded(torch.full_like(like, self.bg), patch, start)

    @torch.no_grad()
    def generate_assets(
        self,
        x_hat: torch.Tensor,
        t: int,
        generator: Optional[torch.Generator] = None,
        part_t_ratio: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """Refine each template crop of x_hat with its frozen part expert (part crops)."""
        t_part = self.part_timestep(t, part_t_ratio)
        return {name: self.experts.refine(self.crop(x_hat, name), name, t_part, generator) for name in PART_NAMES}

    @torch.no_grad()
    def supports(self, x_hat: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Part supports: template interior within the head envelope; the cerebellar
        boundary is eroded so the uncertain tissue interface is left to the refiner."""
        envelope = brain_envelope(x_hat, self.bg, self.tissue_margin)
        supports = {}
        for name in PART_NAMES:
            support = self.paste_masks[name] * envelope
            if name == "sub" and self.cerebellum_erosion > 0:
                support = erode(support, self.cerebellum_erosion) * support
            supports[name] = support
        return supports

    @torch.no_grad()
    def composite(self, x_hat: torch.Tensor, assets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        images = {name: self.paste(assets[name], name, x_hat) for name in PART_NAMES}
        return assemble(x_hat, images, self.supports(x_hat), self.feather_sigma)

    # ------------------------------------------------------------------------ injection
    @torch.no_grad()
    def inject(self, z_t: torch.Tensor, t: int, x0_hat: torch.Tensor, x_comp: torch.Tensor, w: torch.Tensor):
        """Replace the clean content of z_t by the re-encoded scaffold inside the part supports.

        z_inj = z_t + sqrt(abar_t) * m * (E(x_comp) - x0_hat) keeps the noise component
        of z_t, so the reverse trajectory continues consistently from z_inj.
        """
        ab = self.abar[int(t)]
        z_comp = self.encode(x_comp)
        m = F.adaptive_avg_pool3d(w, output_size=self.latent_shape[-3:])
        return z_t + ab.sqrt() * m * (z_comp - x0_hat), m

    @torch.no_grad()
    def handoff(
        self,
        z_t: torch.Tensor,
        t: int,
        generator: Optional[torch.Generator] = None,
        part_t_ratio: Optional[float] = None,
        assets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Full scaffold injection at timestep t; `assets` overrides the generated parts (editing)."""
        x0_hat, x_hat = self.clean_estimate(z_t, t)
        if assets is None:
            assets = self.generate_assets(x_hat, t, generator, part_t_ratio)
        x_comp, w = self.composite(x_hat, assets)
        z_inj, m = self.inject(z_t, t, x0_hat, x_comp, w)
        return {"z_inj": z_inj, "m": m, "x0_hat": x0_hat, "x_hat": x_hat, "x_comp": x_comp, "assets": assets}

    def seam_band(self, m: torch.Tensor, width: int) -> torch.Tensor:
        """Latent-space band around the part boundaries of the injection mask."""
        return boundary_band((m > 0.5).float(), int(width))


@torch.no_grad()
def generate(
    injector: ScaffoldInjector,
    *,
    ddim_steps: int,
    handoff_steps: int,
    part_t_ratio: float,
    generator: torch.Generator,
    assets: Optional[Dict[str, torch.Tensor]] = None,
    return_state: bool = False,
):
    """Sample one whole-brain volume: DDIM from pure noise with scaffold injection
    `handoff_steps` steps before the end. Returns the image (and the handoff state)."""
    scheduler = build_ddim_scheduler(ddim_steps)
    timesteps = [int(t) for t in scheduler.timesteps]
    z = torch.randn((1, *injector.latent_shape), generator=generator, device=injector.device)
    state = None
    k = len(timesteps) - int(handoff_steps)
    for i, t in enumerate(timesteps):
        if i == k:
            state = injector.handoff(z, t, generator, part_t_ratio, assets)
            z = state["z_inj"]
        z = scheduler.step(model_output=injector.eps(z, t), timestep=t, sample=z)[0].float()
    x = injector.decode(z)
    return (x, state) if return_state else x
