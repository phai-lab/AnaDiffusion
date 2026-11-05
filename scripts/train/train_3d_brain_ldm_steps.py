import os
import shutil
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from monai import transforms
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader
from monai.utils import first, set_determinism
from torch.amp import autocast, GradScaler
from torch.nn import L1Loss
from tqdm import tqdm
import copy
from datetime import datetime
import json

from generative.inferers import LatentDiffusionInferer
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler


from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,CenterSpatialCrop,
    ScaleIntensityRanged, RandSpatialCropd, RandFlipd, EnsureTyped, CropForegroundd, SpatialPadd,
)
from monai.data import Dataset, CacheDataset, PersistentDataset, DataLoader

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from eval_utils import (  # noqa: E402
    fid_from_features,
    plot_adversarial_loss,
    plot_recon_loss,
    plot_reconstructions,
    plot_unet_loss,
    psnr,
    sample_ldm,
    ssim_3d,
)

from glob import glob
import pandas as pd
from typing import Optional, Dict, Any, Tuple
import nibabel as nib
import numpy as np
import random
import argparse

try:
    # Reuse existing AE trainer from the sibling script.
    from train_3d_brain_ldm_ import train_ae
except Exception:
    train_ae = None

# Avoid shared-memory exhaustion in DataLoader workers (e.g., limited /dev/shm)
mp.set_sharing_strategy("file_system")


def str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "t", "yes", "y")


def _is_placeholder_path(value: str) -> bool:
    return str(value).startswith("<PATH-TO-")


def _validate_public_paths(ap: argparse.ArgumentParser, args: argparse.Namespace):
    placeholders = [
        key
        for key in ("csv", "data_split_json_path", "ae_ckpt", "resume_ckpt", "ldm_ckpt")
        if _is_placeholder_path(getattr(args, key, ""))
    ]
    if placeholders:
        ap.error(f"Replace placeholder path(s) before running: {', '.join(placeholders)}")

def seed_all(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Determinism can slow training but improves reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_dataloaders_from_csv(csv_path, conditions = ['age', 'sex', 'vol', 'group'], train_transforms = None,
                              n_samples = None, data_split_json_path="", batch_size = 1, num_workers=8, seed = 0):
    with open(data_split_json_path, "r") as f:
        splits = json.load(f)

    train_ids = splits["train"]
    val_ids   = splits["val"]
    test_ids  = splits["test"]

    image_ids = {}
    for train_id in train_ids:
        image_ids[train_id] = 'train'
    for val_id in val_ids:
        image_ids[val_id] = 'val'
    for test_id in test_ids:
        image_ids[test_id] = 'test'

    df = pd.read_csv(csv_path)

    train_data, val_data, test_data = [], [], []

    i = 0
    for idx, row in df.iterrows():
        image_id = row['imageID']
        sample = {}
        sample['image'] = row['image']
        for c in conditions:
            sample[c] = row[c]
        if image_ids[image_id] == 'train':
            train_data.append(sample)
            i += 1
            if i == n_samples:
                break
        elif image_ids[image_id] == 'val':
            val_data.append(sample)
        elif image_ids[image_id] == 'test':
            test_data.append(sample)
            
    train_ds = Dataset(data=train_data, transform=train_transforms)
    print(f'Transformed data shape: {train_ds[0]["image"].shape}')
    print(f"Number of training samples: {len(train_ds)}")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_ds = Dataset(data=val_data, transform=train_transforms)
    print(f"Number of validation samples: {len(val_ds)}")
    val_loader = DataLoader(val_ds, batch_size=1, num_workers=num_workers, pin_memory=True)
    test_ds = Dataset(data=test_data, transform=train_transforms)
    print(f"Number of test samples: {len(test_ds)}")
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader

def check_train_data(train_loader, idx = 0, title = 'Train data example', save_path = None):
    check_data = first(train_loader)

    img = check_data["image"][idx, 0]
    fig, axs = plt.subplots(nrows=1, ncols=3)
    for ax in axs:
        ax.axis("off")
    ax = axs[0]
    ax.imshow(img[..., img.shape[2] // 2], cmap="gray")
    ax = axs[1]
    ax.imshow(img[:, img.shape[1] // 2, ...], cmap="gray")
    ax = axs[2]
    ax.imshow(img[img.shape[0] // 2, ...], cmap="gray")
    plt.title(title)
    if not save_path == None:
        plt.savefig(save_path)
    plt.close('all')



def _load_ckpt_into_ae(ae, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = state.get("state_dict", state)
    missing, unexpected = ae.load_state_dict(state, strict=False)

def _save_ckpt(
    path: str,
    unet: torch.nn.Module,
    opt: torch.optim.Optimizer,
    scaler: Optional[GradScaler],
    global_step: int,
    epoch: int,
    extra: Optional[Dict[str, Any]] = None,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "global_step": int(global_step),
        "epoch": int(epoch),
        "state_dict": unet.state_dict(),
        "optimizer": opt.state_dict() if opt is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "extra": extra or {},
    }
    torch.save(payload, path)


def _load_ckpt_into_unet(
    unet: torch.nn.Module,
    ckpt_path: str,
    device,
    opt: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[GradScaler] = None,
) -> Tuple[int, int, Dict[str, Any]]:
    """
    Returns: (global_step, epoch, extra)
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # support raw state_dict checkpoints
    if "state_dict" not in ckpt:
        missing, unexpected = unet.load_state_dict(ckpt, strict=False)
        print(f"[resume] loaded raw state_dict: missing={len(missing)} unexpected={len(unexpected)}")
        return 0, 0, {}

    missing, unexpected = unet.load_state_dict(ckpt["state_dict"], strict=False)
    print(f"[resume] loaded packaged ckpt: missing={len(missing)} unexpected={len(unexpected)}")

    if opt is not None and ckpt.get("optimizer") is not None:
        opt.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])

    gs = int(ckpt.get("global_step", 0))
    ep = int(ckpt.get("epoch", 0))
    extra = ckpt.get("extra", {}) or {}
    return gs, ep, extra


def KL_loss(z_mu, z_sigma):
    kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4])
    return torch.sum(kl_loss) / kl_loss.shape[0]



@torch.no_grad()
def eval_ldm_loss_only(
    *,
    unet: torch.nn.Module,
    autoencoder: AutoencoderKL,
    val_loader,
    device,
    scale_factor: float,
    torch_autocast: bool,
    val_batches: int = 4,
):
    """
    Very cheap eval: just diffusion noise-prediction loss on a few val batches.
    """
    unet.eval()
    autoencoder.eval()

    ddpm = DDPMScheduler(
        num_train_timesteps=1000,
        schedule="scaled_linear_beta",
        beta_start=0.0015,
        beta_end=0.0195,
    )
    inferer_train = LatentDiffusionInferer(ddpm, scale_factor=scale_factor)
    amp_enabled = bool(torch_autocast and device.type == "cuda")

    val_loss = 0.0
    n_seen = 0
    for i, batch in enumerate(val_loader):
        if i >= val_batches:
            break
        images = batch["image"].to(device)
        with torch.no_grad():
            z = autoencoder.encode_stage_2_inputs(images) * scale_factor
        noise = torch.randn_like(z)
        t = torch.randint(0, ddpm.num_train_timesteps, (images.shape[0],), device=device).long()
        with autocast(device_type="cuda", enabled=amp_enabled):
            noise_pred = inferer_train(
                inputs=images,
                autoencoder_model=autoencoder,
                diffusion_model=unet,
                noise=noise,
                timesteps=t,
            )
            loss = F.mse_loss(noise_pred.float(), noise.float())
        val_loss += float(loss.item())
        n_seen += 1

    return {"val_loss_simple": val_loss / max(1, n_seen)}


@torch.no_grad()
def eval_ldm_fast(
    *,
    unet: torch.nn.Module,
    autoencoder: AutoencoderKL,
    val_loader,
    device,
    scale_factor: float,
    torch_autocast: bool,
    outdir: str,
    global_step: int,
    eval_n: int = 8,
    ddim_steps: int = 50,
    val_batches: int = 8,
):
    """
    Fast evaluation:
      - val diffusion loss on a few batches
      - DDIM sampling with small N
      - cheap diversity metrics
    """
    unet.eval()
    autoencoder.eval()
    amp_enabled = bool(torch_autocast and device.type == "cuda")

    # Validation diffusion loss.
    ddpm = DDPMScheduler(
        num_train_timesteps=1000,
        schedule="scaled_linear_beta",
        beta_start=0.0015,
        beta_end=0.0195,
    )
    inferer_train = LatentDiffusionInferer(ddpm, scale_factor=scale_factor)

    val_loss = 0.0
    n_seen = 0
    for i, batch in enumerate(val_loader):
        if i >= val_batches:
            break
        images = batch["image"].to(device)

        with torch.no_grad():
            z = autoencoder.encode_stage_2_inputs(images) * scale_factor

        noise = torch.randn_like(z)
        t = torch.randint(0, ddpm.num_train_timesteps, (images.shape[0],), device=device).long()

        with autocast(device_type="cuda", enabled=amp_enabled):
            noise_pred = inferer_train(
                inputs=images,
                autoencoder_model=autoencoder,
                diffusion_model=unet,
                noise=noise,
                timesteps=t,
            )
            loss = F.mse_loss(noise_pred.float(), noise.float())

        val_loss += loss.item()
        n_seen += 1

    val_loss = val_loss / max(1, n_seen)

    # DDIM sampling in latent space.
    ddim = DDIMScheduler(
        num_train_timesteps=1000,
        schedule="scaled_linear_beta",
        beta_start=0.0015,
        beta_end=0.0195,
        clip_sample=False,
    )

    ref = next(iter(val_loader))["image"].to(device)
    with torch.no_grad():
        z_ref = autoencoder.encode_stage_2_inputs(ref[:1]) * scale_factor
    latent_shape = (eval_n,) + tuple(z_ref.shape[1:])  # (N, C, H, W, D)

    z = torch.randn(latent_shape, device=device)
    ddim.set_timesteps(num_inference_steps=ddim_steps)

    for t in ddim.timesteps:
        t_int = int(t.item()) if hasattr(t, "item") else int(t)
        t_batch = torch.full((eval_n,), t_int, device=device, dtype=torch.long)
        with autocast(device_type="cuda", enabled=amp_enabled):
            eps = unet(z, timesteps=t_batch)
        z, _ = ddim.step(eps, t_int, z)

    with torch.no_grad():
        x_gen = autoencoder.decode_stage_2_outputs(z / scale_factor)  # (N,1,H,W,D)

    # Cheap diversity metrics.
    with torch.no_grad():
        z_back = autoencoder.encode_stage_2_inputs(x_gen) * scale_factor
    latent_std = float(z_back.std().item())

    x_flat = x_gen.float().view(eval_n, -1)
    x_flat = (x_flat - x_flat.mean(dim=1, keepdim=True)) / (x_flat.std(dim=1, keepdim=True) + 1e-6)

    dsum = 0.0
    cnt = 0
    for i in range(eval_n):
        for j in range(i + 1, eval_n):
            dsum += torch.mean((x_flat[i] - x_flat[j]) ** 2).item()
            cnt += 1
    self_l2 = float(dsum / max(1, cnt))

    # Save generated 3D volumes for quick visual inspection + a compact snapshot.
    step_dir = os.path.join(outdir, "eval_samples", f"step{global_step:09d}")
    os.makedirs(step_dir, exist_ok=True)
    affine = np.eye(4, dtype=np.float32)
    ref_meta = None
    try:
        ref_meta = ref
    except Exception:
        ref_meta = None
    if hasattr(ref_meta, "affine"):
        aff = ref_meta.affine
        if isinstance(aff, torch.Tensor):
            aff = aff.detach().cpu().numpy()
        aff = np.asarray(aff)
        if aff.ndim == 3:
            aff = aff[0]
        if aff.shape == (4, 4):
            affine = aff.astype(np.float32)

    x_gen_cpu = x_gen.detach().cpu()
    for i in range(x_gen_cpu.shape[0]):
        arr = x_gen_cpu[i, 0].numpy()
        nib.save(nib.Nifti1Image(arr, affine), os.path.join(step_dir, f"sample_{i:03d}.nii.gz"))

    torch.save(
        {
            "global_step": global_step,
            "val_loss": val_loss,
            "latent_std": latent_std,
            "self_l2": self_l2,
            "x_gen": x_gen_cpu,
        },
        os.path.join(step_dir, "eval_snapshot.pt"),
    )

    metrics = {"val_loss": val_loss, "latent_std": latent_std, "self_l2": self_l2}
    return metrics


def train_ldm_steps(
    *,
    unet,
    train_loader,
    val_loader,
    autoencoder,
    max_steps: int,
    lr: float = 1e-4,
    scale_factor: Optional[float] = None,
    torch_autocast: bool = True,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    outdir: str = "ckpts",
    ckpt_every: int = 10_000,
    last_every: int = 1_000,
    eval_every: int = 5_000,
    simple_eval_every: int = 2_000,
    eval_n: int = 8,
    ddim_steps: int = 50,
    eval_val_batches: int = 8,
    simple_eval_val_batches: int = 4,
    resume_ckpt: str = "",
):
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        schedule="scaled_linear_beta",
        beta_start=0.0015,
        beta_end=0.0195,
    )
    inferer = LatentDiffusionInferer(scheduler, scale_factor=1.0)  # set scale_factor after computed

    autoencoder.eval()
    for p in autoencoder.parameters():
        p.requires_grad = False

    if scale_factor is None:
        with torch.no_grad(), torch.autocast("cuda", enabled=torch_autocast):
            check_data = first(train_loader)
            z = autoencoder.encode_stage_2_inputs(check_data["image"].to(device))
        scale_factor = float(1.0 / torch.std(z).item())
        print(f"[LDM] scale_factor set to {scale_factor}")
    inferer.scale_factor = scale_factor

    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr, weight_decay=1e-2)
    scaler = GradScaler("cuda", enabled=bool(torch_autocast and device.type == "cuda"))

    global_step = 0
    start_epoch = 0
    history = {
        "train_loss": [],
        "simple_eval": [],
        "eval": [],
        "loss_curve": [],
        "epoch_loss_curve": [],
    }

    if resume_ckpt:
        global_step, start_epoch, extra = _load_ckpt_into_unet(
            unet, resume_ckpt, device, opt=optimizer, scaler=scaler
        )
        # restore history if present
        if isinstance(extra, dict):
            history = extra.get("history", history)
            # also restore scale_factor if saved
            if "scale_factor" in extra:
                scale_factor = float(extra["scale_factor"])
                inferer.scale_factor = scale_factor
        if not isinstance(history.get("loss_curve"), list):
            history["loss_curve"] = []
        if not isinstance(history.get("epoch_loss_curve"), list):
            history["epoch_loss_curve"] = []
        print(f"[resume] global_step={global_step}, start_epoch~={start_epoch}, scale_factor={scale_factor}")

    unet.train()
    os.makedirs(outdir, exist_ok=True)

    data_iter = iter(train_loader)
    steps_per_epoch = len(train_loader)
    steps_per_epoch = max(1, int(steps_per_epoch))

    def _update_epoch_loss_curve():
        raw = history.get("loss_curve", [])
        epoch_curve = []
        for i in range(0, len(raw), steps_per_epoch):
            chunk = raw[i : i + steps_per_epoch]
            if chunk:
                epoch_curve.append(float(sum(chunk) / len(chunk)))
        history["epoch_loss_curve"] = epoch_curve

    running = 0.0
    running_n = 0

    pbar = tqdm(range(global_step, max_steps), ncols=110)
    for step in pbar:
        # `step` is the zero-based index of the update we are about to run.
        # `global_step` tracks the number of completed optimizer updates.

        # cycle dataloader without epochs
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        images = batch["image"].to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            z = autoencoder.encode_stage_2_inputs(images) * scale_factor

        noise = torch.randn_like(z)
        timesteps = torch.randint(0, scheduler.num_train_timesteps, (images.shape[0],), device=device).long()

        with autocast(device_type="cuda", enabled=torch_autocast):
            noise_pred = inferer(
                inputs=images,
                autoencoder_model=autoencoder,
                diffusion_model=unet,
                noise=noise,
                timesteps=timesteps,
            )
            loss = F.mse_loss(noise_pred.float(), noise.float())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        global_step = step + 1

        running += float(loss.item())
        running_n += 1
        avg_loss = running / max(1, running_n)
        history["loss_curve"].append(float(loss.item()))

        # pseudo-epoch for logging only
        epoch = int(max(0, global_step - 1) // max(1, steps_per_epoch))

        pbar.set_description(f"step {global_step}/{max_steps} (ep~{epoch})")
        pbar.set_postfix({"loss": f"{avg_loss:.6f}", "sf": f"{scale_factor:.4f}"})

        # log history periodically (lightweight)
        if global_step % 200 == 0:
            history["train_loss"].append({"step": global_step, "loss": avg_loss})

        if (last_every > 0) and (global_step % last_every == 0):
            _update_epoch_loss_curve()
            _save_ckpt(
                os.path.join(outdir, "UNET_last.pt"),
                unet,
                optimizer,
                scaler,
                global_step=global_step,
                epoch=epoch,
                extra={"scale_factor": float(scale_factor), "history": history},
            )
            plot_unet_loss(
                history["epoch_loss_curve"],
                title=f"UNET Epoch-Average Loss_step{global_step}",
                outdir=outdir,
                filename="UNET_loss.png",
            )

        if (ckpt_every > 0) and (global_step % ckpt_every == 0):
            _update_epoch_loss_curve()
            ckpt_path = os.path.join(outdir, f"UNET_step{global_step:09d}.pt")
            _save_ckpt(
                ckpt_path,
                unet,
                optimizer,
                scaler,
                global_step=global_step,
                epoch=epoch,
                extra={"scale_factor": float(scale_factor), "history": history},
            )

        if (eval_every > 0) and (global_step % eval_every == 0):
            metrics = eval_ldm_fast(
                unet=unet,
                autoencoder=autoencoder,
                val_loader=val_loader,
                device=device,
                scale_factor=scale_factor,
                torch_autocast=torch_autocast,
                outdir=outdir,
                global_step=global_step,
                eval_n=eval_n,
                ddim_steps=ddim_steps,
                val_batches=eval_val_batches,
            )
            history["eval"].append({"step": global_step, **metrics})
            unet.train()
        elif (simple_eval_every > 0) and (global_step % simple_eval_every == 0):
            metrics = eval_ldm_loss_only(
                unet=unet,
                autoencoder=autoencoder,
                val_loader=val_loader,
                device=device,
                scale_factor=scale_factor,
                torch_autocast=torch_autocast,
                val_batches=simple_eval_val_batches,
            )
            history["simple_eval"].append({"step": global_step, **metrics})
            unet.train()

    _update_epoch_loss_curve()
    final_path = os.path.join(outdir, "UNET_last.pt")
    _save_ckpt(
        final_path,
        unet,
        optimizer,
        scaler,
        global_step=global_step,
        epoch=int(global_step // max(1, steps_per_epoch)),
        extra={"scale_factor": float(scale_factor), "history": history},
    )
    plot_unet_loss(
        history["epoch_loss_curve"],
        title=f"UNET Epoch-Average Loss_step{global_step}",
        outdir=outdir,
        filename="UNET_loss.png",
    )

def main():
    pre_ap = argparse.ArgumentParser(add_help=False)
    pre_ap.add_argument("--config", default="", help="Path to JSON config file with argument defaults.")
    pre_args, _ = pre_ap.parse_known_args()

    ap = argparse.ArgumentParser(
        description="Train MONAI 3D LDM (AE -> LDM) from a CSV of file paths.",
        parents=[pre_ap],
    )
    ap.add_argument("--csv", default="", help="Path to CSV with columns: image[/path], sex, age, vol, [target_label]")
    ap.add_argument("--data_split_json_path", default="", help="JSON file with train/val/test imageID splits.")
    ap.add_argument("--spacing", default="1,1,1", help="Target spacing mm (e.g., 1,1,1)")
    ap.add_argument("--size", default="160,224,160", help="Volume D,H,W (e.g., 160,224,160)")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--torch_autocast", default=True, help='Use torch autocast to accelerate: True or False.')
    ap.add_argument("--n_samples", default="ALL")
    ap.add_argument("--train_val_split", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)


    ap.add_argument("--stage", choices=["ae", "ldm", "both"], default="both")


    # AE config
    ap.add_argument("--ae_latent_ch", type=int, default=8)
    ap.add_argument("--ae_num_channels", default="64,128,256")
    ap.add_argument("--ae_attention_levels", default="0,0,0", help="Comma-separated binary flags for attention at each AE level (e.g., 0,0,1)")
    ap.add_argument("--ae_ckpt", default="", help="Path to pretrained AE .pt")

    ap.add_argument("--max_steps", type=int, default=120_000, help="Total number of optimizer steps for step-based LDM training.")
    ap.add_argument("--ckpt_every", type=int, default=10_000, help="Save packaged UNet checkpoint every N steps.")
    ap.add_argument("--last_every", type=int, default=1_000, help="Overwrite UNET_last.pt every N steps for easy resume.")
    ap.add_argument("--eval_every", type=int, default=10_000, help="Run full fast eval + sample generation every N steps.")
    ap.add_argument("--simple_eval_every", type=int, default=2_000, help="Run cheap val-loss-only eval every N steps (excluding full-eval steps).")
    ap.add_argument("--eval_n", type=int, default=8, help="Number of generated 3D volumes during full fast eval.")
    ap.add_argument("--ddim_steps", type=int, default=50, help="DDIM denoising steps for fast eval sampling.")
    ap.add_argument("--eval_val_batches", type=int, default=8, help="Number of val batches for full fast eval.")
    ap.add_argument("--simple_eval_val_batches", type=int, default=4, help="Number of val batches for simple eval.")
    ap.add_argument("--resume_ckpt", default="", help="Path to a packaged UNET_step*.pt to resume")

    # LDM config
    ap.add_argument("--ldm_epochs", type=int, default=150)
    ap.add_argument("--ldm_lr", type=float, default=1e-4)
    ap.add_argument("--ldm_use_cond", default="False", help="Use [part_vol_norm,sex,age] conditioning")
    ap.add_argument("--ldm_num_channels", default="256,256,512")
    ap.add_argument("--ldm_num_head_channels", default="0,64,64")
    ap.add_argument("--ldm_ckpt", default="", help="Resume UNet weights (optional)")
    ap.add_argument("--ldm_sample_every", type=int, default=25, help="Synthesize samples every N epochs")


    ap.add_argument("--outdir", default="ckpts")
    ap.add_argument("--out_prefix", default="")
    ap.add_argument("--out_postfix", default = datetime.now().strftime('%Y%m%d_%H%M%S'))


    if pre_args.config:
        with open(pre_args.config, "r", encoding="utf-8") as f:
            cfg_defaults = json.load(f)
        if not isinstance(cfg_defaults, dict):
            raise ValueError(f"Config must be a JSON object: {pre_args.config}")
        known_keys = {a.dest for a in ap._actions}
        unknown_keys = sorted(set(cfg_defaults.keys()) - known_keys)
        if unknown_keys:
            raise ValueError(
                f"Unknown keys in config {pre_args.config}: {unknown_keys}"
            )
        ap.set_defaults(**cfg_defaults)

    args = ap.parse_args()
    if not args.csv:
        ap.error("--csv is required (pass on CLI or in --config).")
    if not args.data_split_json_path:
        ap.error("--data_split_json_path is required (pass on CLI or in --config).")
    if not args.ae_ckpt:
        ap.error("--ae_ckpt is required (pass on CLI or in --config).")
    _validate_public_paths(ap, args)

    os.makedirs(args.outdir, exist_ok=True)
    experiment_dir = os.path.join(args.outdir, f"{args.out_prefix}_{args.out_postfix}")
    os.makedirs(experiment_dir, exist_ok=True)
    args.outdir = experiment_dir

    print(f"\nOutput dir: {args.outdir}\n")

    cfg = vars(args).copy()

    # Make non-JSON types serializable (e.g., tuples)
    for k,v in list(cfg.items()):
        if isinstance(v, tuple): cfg[k] = list(v)
    with open(os.path.join(args.outdir, 'args.json'), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)

    spacing = tuple(float(x) for x in args.spacing.split(","))
    size    = tuple(int(x)   for x in args.size.split(","))

    channel = 0
    keys = ['image']
    train_transforms = transforms.Compose(
        [
        transforms.LoadImaged(keys=keys),
        transforms.EnsureChannelFirstd(keys=keys),
        transforms.Lambdad(keys=keys, func=lambda x: x[channel, :, :, :]),
        transforms.EnsureChannelFirstd(keys=keys, channel_dim="no_channel"),
        transforms.EnsureTyped(keys=keys),
        transforms.Spacingd(keys=keys, pixdim=spacing, mode=("bilinear")),
        transforms.DivisiblePadd(keys=keys, k=32, mode="constant",constant_values=-1.0),
        transforms.CenterSpatialCropd(keys=keys,roi_size=size),
        ]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}\n")
    
    random.seed(args.seed)
    seed_all(args.seed)

    try:
        n_samples = int(args.n_samples)
    except:
        n_samples = args.n_samples
    train_loader, val_loader, test_loader = make_dataloaders_from_csv(args.csv, conditions = ['age', 'sex', 'vol','group'], train_transforms = train_transforms,
                                                         n_samples=n_samples, data_split_json_path = args.data_split_json_path, batch_size = args.batch, 
                                                         num_workers=args.workers, seed=args.seed)

    ae_num_channels = tuple(int(x) for x in args.ae_num_channels.split(","))
    ae_latent_ch = int(args.ae_latent_ch)
    ae_attention_levels = tuple(bool(int(x)) for x in args.ae_attention_levels.split(","))


    autoencoder = AutoencoderKL(
        spatial_dims=3,
        in_channels=1,
        out_channels=1, 
        num_channels=ae_num_channels,
        latent_channels=ae_latent_ch,
        num_res_blocks=2,
        norm_num_groups=32,
        norm_eps=1e-06,
        attention_levels=ae_attention_levels,
    )
    autoencoder.to(device)

    if args.ae_ckpt != "":
        _load_ckpt_into_ae(autoencoder, args.ae_ckpt, device)
    

    ldm_num_channels = tuple(int(x) for x in args.ldm_num_channels.split(","))
    ldm_num_head_channels = tuple(int(x) for x in args.ldm_num_head_channels.split(","))
    unet = DiffusionModelUNet(
        spatial_dims=3,
        in_channels=ae_latent_ch,
        out_channels=ae_latent_ch,
        num_res_blocks=2,
        num_channels=ldm_num_channels,
        attention_levels=(False, True, True),
        num_head_channels=ldm_num_head_channels,
        norm_num_groups=32, norm_eps=1e-6,
        resblock_updown=True,
        upcast_attention=True,
        use_flash_attention=False
    )
    unet.to(device)

    scale_factor = None

    if args.ldm_ckpt != "":
        _load_ckpt_into_unet(unet, args.ldm_ckpt, device)
        ckpt = torch.load(args.ldm_ckpt, map_location=device)
        try:
            ckpt = torch.load(args.ldm_ckpt, map_location=device)
            scale_factor = float(ckpt['extra']['scale_factor'])
            print('Loading scale factor = ', scale_factor)
        except:
            scale_factor = None
    torch_autocast = str2bool(args.torch_autocast)

    if args.stage in ["ldm", "both"]:
        train_ldm_steps(
            unet=unet,
            train_loader=train_loader,
            val_loader=val_loader,
            autoencoder=autoencoder,
            max_steps=args.max_steps,
            lr=args.ldm_lr,
            scale_factor=scale_factor,
            torch_autocast=torch_autocast,
            device=device,
            outdir=args.outdir,
            ckpt_every=args.ckpt_every,
            last_every=args.last_every,
            eval_every=args.eval_every,
            simple_eval_every=args.simple_eval_every,
            eval_n=args.eval_n,
            ddim_steps=args.ddim_steps,
            eval_val_batches=args.eval_val_batches,
            simple_eval_val_batches=args.simple_eval_val_batches,
            resume_ckpt=args.resume_ckpt,
        )


if __name__ == "__main__":
   main()
