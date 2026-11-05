"""Model construction, diffusion schedules, and checkpoint I/O."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDIMScheduler, DDPMScheduler

NUM_TRAIN_TIMESTEPS = 1000
BETA_SCHEDULE = dict(schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0195)


def build_ddpm_scheduler() -> DDPMScheduler:
    return DDPMScheduler(num_train_timesteps=NUM_TRAIN_TIMESTEPS, **BETA_SCHEDULE)


def build_ddim_scheduler(num_inference_steps: int) -> DDIMScheduler:
    scheduler = DDIMScheduler(num_train_timesteps=NUM_TRAIN_TIMESTEPS, clip_sample=False, **BETA_SCHEDULE)
    scheduler.set_timesteps(num_inference_steps=int(num_inference_steps))
    return scheduler


def alphas_cumprod(device: torch.device) -> torch.Tensor:
    return build_ddpm_scheduler().alphas_cumprod.to(device=device, dtype=torch.float32)


def build_autoencoder(
    latent_channels: int = 8,
    num_channels: Sequence[int] = (64, 128, 256),
    attention_levels: Sequence[bool] = (False, False, False),
) -> AutoencoderKL:
    return AutoencoderKL(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        num_channels=tuple(num_channels),
        latent_channels=int(latent_channels),
        num_res_blocks=2,
        norm_num_groups=32,
        norm_eps=1e-6,
        attention_levels=tuple(bool(a) for a in attention_levels),
    )


def build_unet(
    latent_channels: int = 8,
    num_channels: Sequence[int] = (256, 256, 512),
    num_head_channels: Sequence[int] = (0, 64, 64),
    cross_attention_dim: Optional[int] = None,
) -> DiffusionModelUNet:
    kwargs: Dict[str, Any] = dict(
        spatial_dims=3,
        in_channels=int(latent_channels),
        out_channels=int(latent_channels),
        num_res_blocks=2,
        num_channels=tuple(num_channels),
        attention_levels=(False, True, True),
        num_head_channels=tuple(num_head_channels),
        norm_num_groups=32,
        norm_eps=1e-6,
        resblock_updown=True,
        upcast_attention=True,
        use_flash_attention=False,
    )
    if cross_attention_dim is not None:
        kwargs.update(with_conditioning=True, cross_attention_dim=int(cross_attention_dim))
    return DiffusionModelUNet(**kwargs)


class HemisphereConditioner(nn.Module):
    """Side embedding for the shared hemisphere LDM (0 = left, 1 = right)."""

    def __init__(self, n_parts: int = 2, d_model: int = 64, p_drop: float = 0.1):
        super().__init__()
        self.n_parts = int(n_parts)
        self.part_emb = nn.Embedding(self.n_parts, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(p_drop)

    def forward(self, part_ids: torch.Tensor) -> torch.Tensor:
        part_ids = part_ids.long().view(-1)
        if torch.any(part_ids < 0) or torch.any(part_ids >= self.n_parts):
            raise ValueError(f"part id out of range [0, {self.n_parts - 1}]")
        ctx = self.drop(self.norm(self.part_emb(part_ids)))
        return ctx.unsqueeze(1)


def _state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    return checkpoint


def load_autoencoder_weights(ae: AutoencoderKL, path: str, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    ae.load_state_dict(_state_dict(checkpoint), strict=False)


def load_unet_weights(unet: nn.Module, path: str, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = _state_dict(checkpoint)
    model_state = unet.state_dict()
    compatible = {
        k: v for k, v in state.items() if k in model_state and tuple(v.shape) == tuple(model_state[k].shape)
    }
    missing, unexpected = unet.load_state_dict(compatible, strict=False)
    print(
        f"[weights] {os.path.basename(path)}: loaded={len(compatible)} "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )


def load_conditioner_weights(conditioner: nn.Module, path: str, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint.get("conditioner_state_dict") if isinstance(checkpoint, dict) else None
    if state is None:
        state = _state_dict(checkpoint)
    conditioner.load_state_dict(state, strict=False)


def read_scale_factor(path: str, keys: Sequence[str] = ("scale_factor",)) -> Optional[float]:
    """Return the latent scale factor stored in a checkpoint, if any."""
    if not path or not os.path.exists(path):
        return None
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    extra = checkpoint.get("extra", {}) if isinstance(checkpoint, dict) else {}
    for key in keys:
        value = extra.get(key) if isinstance(extra, dict) else None
        if value is not None and float(value) > 0:
            return float(value)
    return None


@torch.no_grad()
def estimate_scale_factor(ae: AutoencoderKL, x: torch.Tensor, amp: bool) -> float:
    with torch.autocast("cuda", enabled=amp):
        z = ae.encode_stage_2_inputs(x)
    return float(1.0 / max(torch.std(z).item(), 1e-8))


def save_checkpoint(
    path: str,
    unet: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[torch.amp.GradScaler],
    global_step: int,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "global_step": int(global_step),
            "state_dict": unet.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "extra": extra or {},
        },
        path,
    )


def load_checkpoint(
    path: str,
    unet: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> Tuple[int, Dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    unet.load_state_dict(_state_dict(checkpoint), strict=False)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        return 0, {}
    if optimizer is not None and checkpoint.get("optimizer") is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None and checkpoint.get("scaler") is not None:
        scaler.load_state_dict(checkpoint["scaler"])
    return int(checkpoint.get("global_step", 0)), dict(checkpoint.get("extra") or {})


def freeze(*modules: nn.Module) -> None:
    for module in modules:
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad_(False)
