"""Compositional whole-brain generation and part editing.

Generation: DDIM sampling of the whole-brain LDM from pure noise; `handoff_steps`
steps before the end the partially denoised estimate is decoded, its template
crops are refined by the frozen part experts, assembled into a scaffold,
re-encoded, injected into the whole-brain latent, and the remaining steps are
denoised by the compositional whole-brain model.

Editing: `--edit --recipient i --donor j --part sub|lhemi|rhemi` replaces one
part asset of sample i by the corresponding asset of sample j before injection.

    python scripts/infer/infer_compositional.py --config configs/infer/infer_compositional_spacing1p5.json
    torchrun --standalone --nproc_per_node=4 scripts/infer/infer_compositional.py --config ...
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from anadiffusion import models  # noqa: E402
from anadiffusion.compose import ScaffoldInjector, generate  # noqa: E402
from anadiffusion.config import build_parser, parse_floats, parse_ints, resolve  # noqa: E402
from anadiffusion.parts import PartExperts  # noqa: E402
from anadiffusion.templates import PART_NAMES, load_template_masks  # noqa: E402

DEFAULTS = {
    "spacing": "1.5,1.5,1.5",
    "size": "128,128,128",
    "hemi_size": "64,128,128",
    "sub_size": "128,96,64",
    "torch_autocast": True,
    "crop_template_dir": "data/template/dilated_r2",
    "paste_template_dir": "data/template/dilated_r0",
    "whole_ae_ckpt": "<PATH-TO-WHOLE-BRAIN-AE-CHECKPOINT>",
    "whole_unet_ckpt": "<PATH-TO-COMPOSITIONAL-WHOLE-BRAIN-UNET-CHECKPOINT>",
    "hemi_ae_ckpt": "<PATH-TO-HEMI-AE-CHECKPOINT>",
    "hemi_unet_ckpt": "<PATH-TO-HEMI-UNET-CHECKPOINT>",
    "sub_ae_ckpt": "<PATH-TO-SUB-AE-CHECKPOINT>",
    "sub_unet_ckpt": "<PATH-TO-SUB-UNET-CHECKPOINT>",
    "ae_latent_ch": 8,
    "ae_num_channels": "64,128,256",
    "ldm_num_channels": "256,256,512",
    "ldm_num_head_channels": "0,64,64",
    "hemi_cond_dim": 64,
    # sampling
    "ddim_steps": 50,
    "handoff_steps": 10,
    "part_t_ratio": 1.0,
    "part_t_max": 300,
    "part_ddim_steps": 100,
    "cerebellum_erosion": 4,
    "feather_sigma": 1.5,
    "tissue_margin": 0.1,
    "bg_value": -1.0,
    "num_samples": 291,
    "seed": 1017,
    "save_assets": False,
    "outdir": "samples/compositional",
    # editing
    "edit": False,
    "recipient": 0,
    "donor": 1,
    "part": "sub",
}


def build_injector(args, device):
    amp = bool(args.torch_autocast) and device.type == "cuda"
    spacing, size = parse_floats(args.spacing), parse_ints(args.size)
    hemi_size, sub_size = parse_ints(args.hemi_size), parse_ints(args.sub_size)
    ae_kwargs = dict(latent_channels=args.ae_latent_ch, num_channels=parse_ints(args.ae_num_channels))
    unet_kwargs = dict(
        latent_channels=args.ae_latent_ch,
        num_channels=parse_ints(args.ldm_num_channels),
        num_head_channels=parse_ints(args.ldm_num_head_channels),
    )
    whole_ae = models.build_autoencoder(**ae_kwargs).to(device)
    whole_unet = models.build_unet(**unet_kwargs).to(device)
    hemi_ae = models.build_autoencoder(**ae_kwargs).to(device)
    hemi_unet = models.build_unet(cross_attention_dim=args.hemi_cond_dim, **unet_kwargs).to(device)
    hemi_conditioner = models.HemisphereConditioner(n_parts=2, d_model=args.hemi_cond_dim).to(device)
    sub_ae = models.build_autoencoder(**ae_kwargs).to(device)
    sub_unet = models.build_unet(**unet_kwargs).to(device)
    models.load_autoencoder_weights(whole_ae, args.whole_ae_ckpt, device)
    models.load_unet_weights(whole_unet, args.whole_unet_ckpt, device)
    models.load_autoencoder_weights(hemi_ae, args.hemi_ae_ckpt, device)
    models.load_unet_weights(hemi_unet, args.hemi_unet_ckpt, device)
    models.load_conditioner_weights(hemi_conditioner, args.hemi_unet_ckpt, device)
    models.load_autoencoder_weights(sub_ae, args.sub_ae_ckpt, device)
    models.load_unet_weights(sub_unet, args.sub_unet_ckpt, device)
    models.freeze(whole_ae, whole_unet, hemi_ae, hemi_unet, hemi_conditioner, sub_ae, sub_unet)

    scale = {
        "whole": models.read_scale_factor(args.whole_unet_ckpt, ("whole_scale_factor", "scale_factor")),
        "hemi": models.read_scale_factor(args.hemi_unet_ckpt, ("hemi_scale_factor", "scale_factor")),
        "sub": models.read_scale_factor(args.sub_unet_ckpt, ("sub_scale_factor", "scale_factor")),
    }
    missing = [k for k, v in scale.items() if v is None]
    if missing:
        raise ValueError(f"Latent scale factor missing from checkpoint(s): {', '.join(missing)}")

    experts = PartExperts(
        hemi_ae=hemi_ae,
        hemi_unet=hemi_unet,
        hemi_conditioner=hemi_conditioner,
        sub_ae=sub_ae,
        sub_unet=sub_unet,
        hemi_scale_factor=scale["hemi"],
        sub_scale_factor=scale["sub"],
        ddim_steps=int(args.part_ddim_steps),
        device=device,
        amp=amp,
    )
    return ScaffoldInjector(
        whole_ae=whole_ae,
        whole_unet=whole_unet,
        whole_scale_factor=scale["whole"],
        experts=experts,
        crop_masks=load_template_masks(args.crop_template_dir, spacing, size, device),
        paste_masks=load_template_masks(args.paste_template_dir, spacing, size, device),
        spacing=spacing,
        whole_shape=size,
        hemi_shape=hemi_size,
        sub_shape=sub_size,
        part_t_ratio=float(args.part_t_ratio),
        part_t_max=int(args.part_t_max),
        cerebellum_erosion=int(args.cerebellum_erosion),
        feather_sigma=float(args.feather_sigma),
        tissue_margin=float(args.tissue_margin),
        bg_value=float(args.bg_value),
        device=device,
        amp=amp,
    )


def save_nifti(x: torch.Tensor, affine: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(x[0, 0].detach().cpu().numpy().astype(np.float32), affine), str(path))


def sample(injector, args, index: int, device, assets=None):
    generator = torch.Generator(device=device).manual_seed(int(args.seed) + int(index))
    return generate(
        injector,
        ddim_steps=int(args.ddim_steps),
        handoff_steps=int(args.handoff_steps),
        part_t_ratio=float(args.part_t_ratio),
        generator=generator,
        assets=assets,
        return_state=True,
    )


def main():
    parser = build_parser("Compositional whole-brain MRI generation and part editing.", DEFAULTS)
    args = resolve(parser, DEFAULTS)
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    outdir = Path(args.outdir)
    if rank == 0:
        outdir.mkdir(parents=True, exist_ok=True)
        with open(outdir / "infer_args.json", "w", encoding="utf-8") as f:
            json.dump(vars(args), f, indent=2, sort_keys=True)
    affine = np.diag([*parse_floats(args.spacing), 1.0])
    injector = build_injector(args, device)

    if bool(args.edit):
        if args.part not in PART_NAMES:
            raise ValueError(f"--part must be one of {PART_NAMES}")
        _, donor_state = sample(injector, args, int(args.donor), device)
        _, recipient_state = sample(injector, args, int(args.recipient), device)
        assets = dict(recipient_state["assets"])
        assets[args.part] = donor_state["assets"][args.part]
        edited, _ = sample(injector, args, int(args.recipient), device, assets=assets)
        name = f"recipient_{int(args.recipient):04d}_{args.part}_from_{int(args.donor):04d}"
        save_nifti(edited, affine, outdir / f"{name}.nii.gz")
        save_nifti(recipient_state["x_hat"], affine, outdir / f"{name}_recipient_estimate.nii.gz")
        print(f"saved {outdir / name}.nii.gz")
        return

    indices = [i for i in range(int(args.num_samples)) if i % world == rank]
    start = time.time()
    for n, index in enumerate(indices):
        path = outdir / "images" / f"sample_{index:04d}.nii.gz"
        if path.exists():
            continue
        x, state = sample(injector, args, index, device)
        save_nifti(x, affine, path)
        if bool(args.save_assets):
            for part in PART_NAMES:
                save_nifti(state["assets"][part], affine, outdir / "assets" / f"sample_{index:04d}_{part}.nii.gz")
        if rank == 0 and (n % 10 == 0 or n == len(indices) - 1):
            print(f"{n + 1}/{len(indices)} samples, {(time.time() - start) / (n + 1):.1f}s per sample", flush=True)


if __name__ == "__main__":
    main()
