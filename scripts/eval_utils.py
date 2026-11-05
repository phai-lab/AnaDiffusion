from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from monai.utils import first


def _ensure_outdir(outdir: str | os.PathLike) -> Path:
    path = Path(outdir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _metric_filename(filename: str, metric: str) -> str:
    if "recon_loss" in filename:
        return filename.replace("recon_loss", metric)
    path = Path(filename)
    suffix = path.suffix or ".png"
    return f"{path.stem}_{metric}{suffix}"


def _plot_series(
    values,
    *,
    title: str,
    ylabel: str,
    outdir: str | os.PathLike,
    filename: str,
    label: str,
    color: str,
):
    _ensure_outdir(outdir)
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(values, color=color, linewidth=2.0, label=label)
    ax.set_title(title, fontsize=18)
    ax.set_xlabel("Epochs", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.legend(prop={"size": 12})
    fig.tight_layout()
    fig.savefig(Path(outdir) / filename)
    plt.close(fig)


def _plot_volume_slices(axs, volume: np.ndarray):
    for ax in axs:
        ax.axis("off")
    axs[0].imshow(volume[..., volume.shape[2] // 2], cmap="gray")
    axs[1].imshow(volume[:, volume.shape[1] // 2, :], cmap="gray")
    axs[2].imshow(volume[volume.shape[0] // 2, :, :], cmap="gray")


def _prepare_mask(mask: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mask = mask.to(device=target.device, dtype=target.dtype)
    if mask.dim() == target.dim() - 1:
        mask = mask.unsqueeze(1)
    if mask.shape != target.shape:
        mask = mask.expand_as(target)
    return mask


def psnr(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor | None = None,
    data_range: float = 2.0,
    eps: float = 1e-8,
    cap: float = 99.0,
) -> float:
    """Return the batch-mean PSNR for tensors shaped [B, C, D, H, W]."""
    if mask is None:
        mask = (y > -1.0).float()
    else:
        mask = mask.float()
    mask = _prepare_mask(mask, x)

    batch_size = x.shape[0]
    diff2 = (x - y).pow(2).view(batch_size, -1)
    mask_flat = mask.view(batch_size, -1)

    denom = mask_flat.sum(dim=1).clamp_min(1.0)
    mse = (diff2 * mask_flat).sum(dim=1) / denom
    psnr_vals = 10.0 * torch.log10((data_range**2) / (mse + eps))
    return torch.clamp(psnr_vals, max=cap).mean().item()


def ssim_3d(
    x: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor | None = None,
    data_range: float = 2.0,
    eps: float = 1e-8,
) -> float:
    """Return a global, non-windowed SSIM-like score averaged over the batch."""
    if mask is None:
        mask = y > -1.0
    mask = _prepare_mask(mask, x)

    batch_size = x.shape[0]
    x_flat = x.view(batch_size, -1)
    y_flat = y.view(batch_size, -1)
    mask_flat = mask.view(batch_size, -1)
    denom = mask_flat.sum(dim=1).clamp_min(1.0)

    mu_x = (x_flat * mask_flat).sum(dim=1) / denom
    mu_y = (y_flat * mask_flat).sum(dim=1) / denom

    x_centered = x_flat - mu_x[:, None]
    y_centered = y_flat - mu_y[:, None]
    var_x = (mask_flat * x_centered * x_centered).sum(dim=1) / denom
    var_y = (mask_flat * y_centered * y_centered).sum(dim=1) / denom
    cov_xy = (mask_flat * x_centered * y_centered).sum(dim=1) / denom

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)
    denominator = (mu_x.pow(2) + mu_y.pow(2) + c1) * (var_x + var_y + c2)
    return (numerator / (denominator + eps)).mean().item()


def _gaussian_1d(
    window_size: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    coords = torch.arange(window_size, device=device, dtype=dtype)
    coords = coords - (window_size - 1) / 2.0
    kernel = torch.exp(-(coords**2) / (2 * sigma**2))
    return kernel / (kernel.sum() + 1e-12)


def _gaussian_kernel_3d(
    window_size: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    kernel_1d = _gaussian_1d(window_size, sigma, device, dtype)
    kernel_3d = (
        kernel_1d[:, None, None]
        * kernel_1d[None, :, None]
        * kernel_1d[None, None, :]
    )
    return kernel_3d.unsqueeze(0).unsqueeze(0)


def _as_single_channel_5d(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if tensor.dim() == 3:
        return tensor.unsqueeze(0).unsqueeze(0)
    if tensor.dim() == 4:
        return tensor.unsqueeze(1)
    if tensor.dim() == 5:
        return tensor
    raise ValueError(f"{name} must be shaped [D,H,W], [B,D,H,W], or [B,1,D,H,W].")


@torch.no_grad()
def ssim_3d_heatmap(
    x: torch.Tensor,
    y: torch.Tensor,
    data_range: float = 2.0,
    window_size: int = 11,
    sigma: float = 1.5,
    eps: float = 1e-8,
    mask: torch.Tensor | None = None,
    padding: str = "same",
) -> tuple[torch.Tensor, float]:
    """Compute a local 3D SSIM map and its mean value."""
    x = _as_single_channel_5d(x, "x")
    y = _as_single_channel_5d(y, "y")

    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} and {y.shape}.")
    if x.shape[1] != 1:
        raise ValueError(f"Expected single-channel volumes, got shape {x.shape}.")
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd.")

    kernel = _gaussian_kernel_3d(window_size, sigma, x.device, x.dtype)
    if padding == "same":
        pad = window_size // 2
        pad_tuple = (pad, pad, pad, pad, pad, pad)
        x_input = F.pad(x, pad_tuple, mode="reflect")
        y_input = F.pad(y, pad_tuple, mode="reflect")
    elif padding == "valid":
        x_input = x
        y_input = y
    else:
        raise ValueError("padding must be 'same' or 'valid'.")

    mu_x = F.conv3d(x_input, kernel)
    mu_y = F.conv3d(y_input, kernel)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv3d(x_input * x_input, kernel) - mu_x2
    sigma_y2 = F.conv3d(y_input * y_input, kernel) - mu_y2
    sigma_xy = F.conv3d(x_input * y_input, kernel) - mu_xy
    sigma_x2 = sigma_x2.clamp_min(0.0)
    sigma_y2 = sigma_y2.clamp_min(0.0)

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    numerator = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = numerator / (denominator + eps)

    if mask is None:
        return ssim_map, float(ssim_map.mean().item())

    mask = _as_single_channel_5d(mask, "mask")
    if mask.shape != ssim_map.shape:
        raise ValueError(f"mask shape must match SSIM map shape, got {mask.shape} and {ssim_map.shape}.")
    mask = mask.to(device=ssim_map.device, dtype=ssim_map.dtype)
    ssim_mean = (ssim_map * mask).sum() / (mask.sum() + eps)
    return ssim_map, float(ssim_mean.item())


@torch.no_grad()
def _mean_and_cov(feats: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    if feats.ndim != 2:
        raise ValueError(f"feats must be shaped [N, D], got {feats.shape}.")
    if feats.shape[0] < 2:
        raise ValueError("At least two feature rows are required to estimate covariance.")

    feats = feats.to(dtype=torch.float64)
    mean = feats.mean(dim=0)
    centered = feats - mean
    cov = (centered.T @ centered) / (feats.shape[0] - 1)
    cov = cov + torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device) * eps
    return mean, cov


@torch.no_grad()
def _matrix_sqrt_psd(mat: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    mat = mat.to(dtype=torch.float64)
    mat = 0.5 * (mat + mat.T)
    eigvals, eigvecs = torch.linalg.eigh(mat)
    eigvals = torch.clamp(eigvals, min=eps)
    sqrt_mat = eigvecs @ torch.diag(torch.sqrt(eigvals)) @ eigvecs.T
    return 0.5 * (sqrt_mat + sqrt_mat.T)


@torch.no_grad()
def frechet_distance(
    mu1: torch.Tensor,
    cov1: torch.Tensor,
    mu2: torch.Tensor,
    cov2: torch.Tensor,
    eps: float = 1e-6,
) -> float:
    """Compute the Frechet distance between two Gaussian feature distributions."""
    mu1 = mu1.to(dtype=torch.float64)
    mu2 = mu2.to(dtype=torch.float64)
    cov1 = cov1.to(dtype=torch.float64)
    cov2 = cov2.to(dtype=torch.float64)

    eye = torch.eye(cov1.shape[0], dtype=torch.float64, device=cov1.device)
    cov1 = cov1 + eye * eps
    cov2 = cov2 + eye * eps

    diff = mu1 - mu2
    sqrt_cov1 = _matrix_sqrt_psd(cov1, eps=eps)
    cov_prod = sqrt_cov1 @ cov2 @ sqrt_cov1
    cov_prod_sqrt = _matrix_sqrt_psd(cov_prod, eps=eps)
    trace_term = torch.trace(cov1 + cov2 - 2.0 * cov_prod_sqrt)
    fid = diff.dot(diff) + trace_term
    return float(torch.clamp(fid, min=0.0).item())


@torch.no_grad()
def fid_from_features(
    feats_fake: torch.Tensor,
    feats_real: torch.Tensor,
    eps: float = 1e-6,
) -> float:
    """Compute FID from two feature matrices shaped [N, D]."""
    if feats_real.ndim != 2 or feats_fake.ndim != 2:
        raise ValueError("Feature tensors must be shaped [N, D].")
    if feats_real.shape[1] != feats_fake.shape[1]:
        raise ValueError(f"Feature dimensions must match, got {feats_real.shape} and {feats_fake.shape}.")

    real_mean, real_cov = _mean_and_cov(feats_real, eps=eps)
    fake_mean, fake_cov = _mean_and_cov(feats_fake, eps=eps)
    return frechet_distance(real_mean, real_cov, fake_mean, fake_cov, eps=eps)


def plot_recon_loss(
    epoch_recon_loss_list,
    epoch_ssim_loss_list,
    epoch_psnr_loss_list,
    title: str = "Reconstruction Loss Curve",
    outdir: str | os.PathLike = "ckpts",
    filename: str = "AE_recon_loss.png",
):
    _ensure_outdir(outdir)
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epoch_recon_loss_list, color="blue", linewidth=2.0, label="Reconstruction L1 loss")
    ax.plot(epoch_ssim_loss_list, color="green", linewidth=2.0, label="SSIM")
    ax.set_title(title, fontsize=18)
    ax.set_xlabel("Epochs", fontsize=14)
    ax.set_ylabel("Loss", fontsize=14)
    ax.legend(prop={"size": 12})
    fig.tight_layout()
    fig.savefig(Path(outdir) / filename)
    plt.close(fig)

    _plot_series(
        epoch_ssim_loss_list,
        title=title,
        ylabel="SSIM",
        outdir=outdir,
        filename=_metric_filename(filename, "SSIM"),
        label="SSIM",
        color="green",
    )
    _plot_series(
        epoch_psnr_loss_list,
        title=title,
        ylabel="PSNR",
        outdir=outdir,
        filename=_metric_filename(filename, "PSNR"),
        label="PSNR",
        color="orange",
    )


def plot_adversarial_loss(
    epoch_gen_loss_list,
    epoch_disc_loss_list,
    title: str = "Adversarial Training Curves",
    outdir: str | os.PathLike = "ckpts",
    filename: str = "AE_disc_loss.png",
):
    _ensure_outdir(outdir)
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epoch_gen_loss_list, color="C0", linewidth=2.0, label="Generator")
    ax.plot(epoch_disc_loss_list, color="C1", linewidth=2.0, label="Discriminator")
    ax.set_title(title, fontsize=18)
    ax.set_xlabel("Epochs", fontsize=14)
    ax.set_ylabel("Loss", fontsize=14)
    ax.legend(prop={"size": 12})
    fig.tight_layout()
    fig.savefig(Path(outdir) / filename)
    plt.close(fig)


def plot_reconstructions(
    data_batch,
    reconstruction: torch.Tensor,
    idx: int = 0,
    channel: int = 0,
    title: str = "Reconstruction",
    outdir: str | os.PathLike = "ckpts",
    filename: str = "AE_recons.png",
):
    _ensure_outdir(outdir)
    source = data_batch["image"][idx, channel].detach().float().cpu().numpy()
    recon = reconstruction[idx, channel].detach().float().cpu().numpy()

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(9, 6))
    _plot_volume_slices(axs[0], source)
    _plot_volume_slices(axs[1], recon)
    axs[0, 1].set_title(f"{title}\nOriginal", fontsize=14)
    axs[1, 1].set_title("Reconstruction", fontsize=14)
    fig.tight_layout()
    fig.savefig(Path(outdir) / filename)
    plt.close(fig)


def sample_ldm(
    data_loader,
    autoencoder,
    unet,
    scheduler,
    inferer,
    device: torch.device,
    idx: int = 0,
    channel: int = 0,
    outdir: str | os.PathLike = "ckpts",
    filename: str = "synthetic",
):
    autoencoder.eval()
    unet.eval()
    _ensure_outdir(outdir)

    with torch.no_grad():
        check_data = first(data_loader)
        latent = autoencoder.encode_stage_2_inputs(check_data["image"].to(device))
        noise = torch.randn_like(latent)
        scheduler.set_timesteps(num_inference_steps=1000)
        synthetic_images = inferer.sample(
            input_noise=noise,
            autoencoder_model=autoencoder,
            diffusion_model=unet,
            scheduler=scheduler,
        )

    volume = synthetic_images[idx, channel].detach().float().cpu().numpy()
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(9, 3))
    _plot_volume_slices(axs, volume)
    fig.tight_layout()
    fig.savefig(Path(outdir) / f"{filename}.png")
    plt.close(fig)

    nifti = nib.Nifti1Image(volume.astype(np.float32), np.eye(4))
    nib.save(nifti, Path(outdir) / f"{filename}.nii.gz")


def plot_unet_loss(
    epoch_loss_list,
    title: str = "UNet Loss Curve",
    outdir: str | os.PathLike = "ckpts",
    filename: str = "UNET_loss.png",
):
    _plot_series(
        epoch_loss_list,
        title=title,
        ylabel="Loss",
        outdir=outdir,
        filename=filename,
        label="UNet loss",
        color="C0",
    )
