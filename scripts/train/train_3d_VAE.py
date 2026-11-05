import os
import shutil
import tempfile
import sys
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
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from torch.nn import L1Loss
from tqdm import tqdm
import copy
from datetime import datetime
import json

from generative.inferers import LatentDiffusionInferer
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from generative.networks.schedulers import DDPMScheduler

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
    ScaleIntensityRanged, RandSpatialCropd, RandFlipd, EnsureTyped, CropForegroundd, SpatialPadd
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

# Avoid shared-memory exhaustion in DataLoader workers (e.g., limited /dev/shm)
mp.set_sharing_strategy("file_system")

def seed_all(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Determinism can slow training but improves reproducibility.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _is_placeholder_path(value: str) -> bool:
    return str(value).startswith("<PATH-TO-")


def _validate_public_paths(ap: argparse.ArgumentParser, args: argparse.Namespace):
    placeholders = [
        key
        for key in ("csv", "data_split_json_path", "ae_ckpt")
        if _is_placeholder_path(getattr(args, key, ""))
    ]
    if placeholders:
        ap.error(f"Replace placeholder path(s) before running: {', '.join(placeholders)}")


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


def _save_ckpt(path: str, unet: torch.nn.Module, opt, scaler, epoch: int, global_step: int,
               ema: Optional[Any] = None, extra: Optional[Dict[str, Any]] = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "state_dict": unet.state_dict(),
        "optimizer": opt.state_dict() if opt is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "extra": extra or {},
    }
    if ema is not None:
        payload["ema"] = {k: v.cpu() for k, v in ema.shadow.items()}
    torch.save(payload, path)


def KL_loss(z_mu, z_sigma):
    kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4])
    return torch.sum(kl_loss) / kl_loss.shape[0]


def train_ae(autoencoder, train_loader, val_loader = None, val_interval = 1, ae_epochs = 100, 
             adv_weight = 0.01, perceptual_weight = 0.001, kl_weight = 1e-6, 
             lr=1e-4, device = torch.device("cuda" if torch.cuda.is_available() else "cpu"), outdir = 'ckpts'):

    discriminator = PatchDiscriminator(spatial_dims=3, num_layers_d=3, num_channels=32, in_channels=1, out_channels=1)
    discriminator.to(device)


    l1_loss = L1Loss()
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    loss_perceptual = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2)
    loss_perceptual.to(device)



    optimizer_g = torch.optim.AdamW(autoencoder.parameters(), lr=lr, betas=(0.5, 0.9))
    optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=lr*0.5, betas=(0.5, 0.9))



    n_epochs = ae_epochs
    autoencoder_warm_up_n_epochs = int(0.2 * n_epochs)
    epoch_recon_loss_list = []
    epoch_gen_loss_list = []
    epoch_disc_loss_list = []
    epoch_ssim_train_list, epoch_psnr_train_list = [], []
    val_recon_epoch_loss_list = []
    val_recon_ema_list = []
    epoch_ssim_val_list, epoch_psnr_val_list = [], []
    intermediary_images = []
    n_example_images = 4

    ae_best = None
    ae_best_loss = float('inf')

    ema_alpha = 0.2
    val_loss_ema = None
    ae_best_ema = float("inf")


    for epoch in range(n_epochs):
        autoencoder.train()
        discriminator.train()
        epoch_loss = 0
        gen_epoch_loss = 0
        disc_epoch_loss = 0
        ssim_train, psnr_train = 0, 0
        ssim_val, psnr_val = 0, 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110, file=sys.__stdout__)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in progress_bar:
            images = batch["image"].to(device)

            optimizer_g.zero_grad(set_to_none=True)
            reconstruction, z_mu, z_sigma = autoencoder(images)
            kl_loss = KL_loss(z_mu, z_sigma)

            recons_loss = l1_loss(reconstruction.float(), images.float())
            p_loss = loss_perceptual(reconstruction.float(), images.float())
            loss_g = recons_loss + kl_weight * kl_loss + perceptual_weight * p_loss
            psnr_train += psnr(reconstruction.float(), images.float())
            ssim_train += ssim_3d(reconstruction.float(), images.float())

            if epoch > autoencoder_warm_up_n_epochs:
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                loss_g += adv_weight * generator_loss

            loss_g.backward()
            optimizer_g.step()

            if epoch > autoencoder_warm_up_n_epochs:
                optimizer_d.zero_grad(set_to_none=True)
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

                loss_d = adv_weight * discriminator_loss

                loss_d.backward()
                optimizer_d.step()

            epoch_loss += recons_loss.item()
            if epoch > autoencoder_warm_up_n_epochs:
                gen_epoch_loss += generator_loss.item()
                disc_epoch_loss += discriminator_loss.item()

            progress_bar.set_postfix(
                {
                    "recons_loss": epoch_loss / (step + 1),
                    "gen_loss": gen_epoch_loss / (step + 1),
                    "disc_loss": disc_epoch_loss / (step + 1),
                    "SSIM_train": ssim_train / (step + 1),
                    "PSNR_train": psnr_train / (step + 1),

                }
            )
        epoch_recon_loss_list.append(epoch_loss / (step + 1))
        epoch_gen_loss_list.append(gen_epoch_loss / (step + 1))
        epoch_disc_loss_list.append(disc_epoch_loss / (step + 1))
        epoch_ssim_train_list.append(np.mean(ssim_train / (step + 1)))
        epoch_psnr_train_list.append(np.mean(psnr_train / (step + 1)))

        plot_recon_loss(epoch_recon_loss_list, epoch_ssim_train_list, epoch_psnr_train_list, title = f'Train Reconstruction Loss Curve_ep{epoch+1}', outdir = outdir, filename = 'AE_train_recon_loss.png')
        plot_adversarial_loss(epoch_gen_loss_list, epoch_disc_loss_list, title = f'Adversarial Training Curves_ep{epoch+1}', outdir = outdir, filename = 'AE_disc_loss.png')
        plot_reconstructions(batch, reconstruction, idx = 0, channel=0, title = f'Train Sample Reconstructions_ep{epoch+1}', outdir = outdir, filename = 'AE_train_recons.png')

        if val_loader is not None and (epoch + 1) % val_interval == 0:
            autoencoder.eval()
            val_epoch_loss = 0.0
            psnr_val = 0.0
            ssim_val = 0.0
            val_progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), ncols=70, file=sys.__stdout__)
            val_progress_bar.set_description(f"Val Epoch {epoch}")
            with torch.no_grad():
                for val_step, val_batch in val_progress_bar:
                    val_images = val_batch["image"].to(device)
                    val_reconstruction, _, _ = autoencoder(val_images)
                    val_recons_loss = l1_loss(val_reconstruction.float(), val_images.float())
                    val_epoch_loss += val_recons_loss.item()
                    psnr_val += psnr(val_reconstruction.float(), val_images.float())
                    ssim_val += ssim_3d(val_reconstruction.float(), val_images.float())
                    val_progress_bar.set_postfix(
                        {
                            "val_recons_loss": val_epoch_loss / (val_step + 1),
                            "SSIM_val": ssim_val / (val_step + 1),
                            "PSNR_val": psnr_val / (val_step + 1),
                        }
                    )

            val_recon_epoch_loss_list.append(val_epoch_loss / (val_step + 1))
            val_loss = val_epoch_loss / (val_step + 1)

            if val_loss_ema is None:
                val_loss_ema = val_loss
            else:
                val_loss_ema = ema_alpha * val_loss + (1.0 - ema_alpha) * val_loss_ema
            val_recon_ema_list.append(float(val_loss_ema))

            epoch_ssim_val_list.append(ssim_val / (val_step + 1))
            epoch_psnr_val_list.append(np.mean(psnr_val / (val_step + 1)))
            plot_recon_loss(val_recon_epoch_loss_list, epoch_ssim_val_list, epoch_psnr_val_list, title = f'Val Reconstruction Loss Curve_ep{epoch+1}', outdir = outdir, filename = 'AE_val_recon_loss.png')
            print(f"Val epoch {epoch}: raw={val_loss:.6f}, ema={val_loss_ema:.6f}, best_ema={ae_best_ema:.6f}")


            if val_loss_ema < ae_best_ema:
                print(f"Updating best AE checkpoint (EMA improved): raw={val_loss:.6f}, ema={val_loss_ema:.6f}")
                ae_best_ema = val_loss_ema
                ae_best_loss = val_loss
                ae_best = copy.deepcopy(autoencoder)
                _save_ckpt(os.path.join(outdir, "AE_best.pt"), autoencoder, opt=optimizer_g, scaler=None, epoch=epoch+1, global_step=None, 
                            extra={"opt_d": optimizer_d, "train_rec_loss": epoch_recon_loss_list, "train_disc_loss": epoch_disc_loss_list, 
                                   "train_gen_loss": epoch_gen_loss_list, "train_ssim": epoch_ssim_train_list, "train_psnr": epoch_psnr_train_list,
                                   "val_rec_loss": val_recon_epoch_loss_list, "val_rec_ema": val_recon_ema_list,
                                   "val_ssim": epoch_ssim_val_list, "val_psnr": epoch_psnr_val_list})
                plot_reconstructions(val_batch, val_reconstruction, idx = 0, channel=0, title = f'Val Sample Reconstructions_ep{epoch+1}', outdir = outdir, filename = 'AE_val_recons.png')
            if epoch + 1 == n_epochs:
                print('Final epoch, saving last AE checkpoint and val reconstructions...')
                plot_reconstructions(val_batch, val_reconstruction, idx = 0, channel=0, title = f'Val Sample Reconstructions_ep{epoch+1}', outdir = outdir, filename = 'AE_last_val_recons.png')
        _save_ckpt(os.path.join(outdir, "AE_last.pt"), autoencoder, opt=optimizer_g, scaler=None, epoch=epoch+1, global_step=None, 
                        extra={"opt_d": optimizer_d, "train_rec_loss": epoch_recon_loss_list, "train_disc_loss": epoch_disc_loss_list, 
                               "train_gen_loss": epoch_gen_loss_list, "train_ssim": epoch_ssim_train_list, "train_psnr": epoch_psnr_train_list,
                               "val_rec_loss": val_recon_epoch_loss_list, "val_rec_ema": val_recon_ema_list,
                               "val_ssim": epoch_ssim_val_list, "val_psnr": epoch_psnr_val_list})
    del discriminator
    del loss_perceptual
    torch.cuda.empty_cache()
    print('Returning the last and best AEs.')
    return autoencoder, ae_best


def main():
    pre_ap = argparse.ArgumentParser(add_help=False)
    pre_ap.add_argument("--config", default="", help="Path to JSON config file with argument defaults.")
    pre_args, _ = pre_ap.parse_known_args()

    ap = argparse.ArgumentParser(
        description="Train MONAI 3D LDM (AE -> LDM) from a CSV of file paths.",
        parents=[pre_ap],
    )
    ap.add_argument("--csv", default="", help="Path to CSV with columns: image[/path], sex, age, vol, [target_label]")
    ap.add_argument("--spacing", default="1,1,1", help="Target spacing mm (e.g., 1,1,1)")
    ap.add_argument("--size", default="160,224,160", help="Volume D,H,W (e.g., 160,224,160)")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--torch_autocast", default=True, help='Use torch autocast to accelerate: True or False.')
    ap.add_argument("--n_samples", default="ALL")
    ap.add_argument("--data_split_json_path", type=str, default="", help='Path to JSON file containing train/val/test splits of image IDs.')
    ap.add_argument("--seed", type=int, default=0)


    # AE config
    ap.add_argument("--ae_epochs", type=int, default=50)
    ap.add_argument("--ae_lr", type=float, default=1e-4)
    ap.add_argument("--ae_latent_ch", type=int, default=8)
    ap.add_argument("--ae_kl", type=float, default=1e-6)
    ap.add_argument("--ae_adv_weight", type=float, default=0.01)
    ap.add_argument("--ae_perceptual_weight", type=float, default=0.001)
    ap.add_argument("--ae_kl_weight", type=float, default=1e-6)
    ap.add_argument("--ae_num_channels", default="64,128,256")
    ap.add_argument("--ae_attention_levels", default="0,0,0", help="Comma-separated binary flags for attention at each AE level (e.g., 0,0,1)")
    ap.add_argument("--ae_ckpt", default="", help="Path to pretrained AE .pt (optional)")


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
    _validate_public_paths(ap, args)

    os.makedirs(args.outdir, exist_ok=True)
    experiment_dir = os.path.join(args.outdir, f"{args.out_prefix}_{args.out_postfix}")
    os.makedirs(experiment_dir, exist_ok=True)
    args.outdir = experiment_dir
    
    print(f"\nOutput dir: {args.outdir}\n")

    import sys
    log_path = os.path.join(args.outdir, 'run.log')
    log = open(log_path, "w", buffering=1)
    sys.stdout = log
    sys.stderr = log  # tracebacks too

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

    last_ae, best_ae = train_ae(autoencoder, train_loader, val_loader, val_interval = 1, ae_epochs = args.ae_epochs, 
                                    adv_weight = args.ae_adv_weight, perceptual_weight = args.ae_perceptual_weight, kl_weight = args.ae_kl_weight, 
                                    lr=args.ae_lr, device = device, outdir = args.outdir)
    

if __name__ == "__main__":
   main()
