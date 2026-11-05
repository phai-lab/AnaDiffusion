"""MedicalNet FID of generated cohorts against the real test split.

Features are the global-average-pooled layer-4 activations of the pretrained
MedicalNet ResNet-10 (Torch Hub, `warvito/MedicalNet-models`). FID is reported
on five supports derived from the MNI152 template masks: the whole brain, the
left and right hemispheres, the cerebellar-brainstem complex (each cropped to
the part box and z-scored inside its mask), and the seam band around the part
boundaries. Values are reported as FID x 10^4, the unit used in the paper.

    python scripts/eval/medicalnet_fid.py --config configs/eval/medicalnet_fid_spacing1p5.json \
        --cohort name=<DIR-WITH-GENERATED-NIFTI> [name2=<DIR2> ...]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from anadiffusion.config import load_config, parse_floats, parse_ints  # noqa: E402
from anadiffusion.data import build_image_transforms, load_split_records  # noqa: E402
from anadiffusion.templates import PART_NAMES, boundary_band, load_template_masks  # noqa: E402

DEFAULTS = {
    "csv": "<PATH-TO-WHOLE-BRAIN-AND-PARTS-CSV>",
    "data_split_json_path": "<PATH-TO-PATIENT-SPLIT-JSON>",
    "whole_key": "whole_brain",
    "spacing": "1.5,1.5,1.5",
    "size": "128,128,128",
    "hemi_size": "64,128,128",
    "sub_size": "128,96,64",
    "template_dir": "data/template/dilated_r2",
    "seam_radius": 2,
    "num_samples": 291,
    "real_feature_cache": "",
    "out": "eval/medicalnet_fid.json",
}
HUB_REPOSITORY = "warvito/MedicalNet-models"
HUB_MODEL = "medicalnet_resnet10_23datasets"
FID_SCALE = 1e4


class Region:
    """A fixed evaluation support: optional mask and optional crop window."""

    def __init__(self, mask: Optional[torch.Tensor], window: Optional[Tuple[slice, slice, slice]]):
        self.mask, self.window = mask, window


def crop_window(mask: torch.Tensor, size: Tuple[int, int, int]) -> Tuple[slice, slice, slice]:
    """Fixed-size window centred on the mask bounding box, clamped to the volume."""
    active = torch.nonzero(mask[0, 0] > 0.5, as_tuple=False)
    centre = (active.amin(0) + active.amax(0) + 1) // 2
    window = []
    for axis in range(3):
        extent = int(mask.shape[2 + axis])
        start = int(centre[axis]) - size[axis] // 2
        start = max(0, min(start, extent - size[axis]))
        window.append(slice(start, start + size[axis]))
    return tuple(window)


def build_regions(masks: Dict[str, torch.Tensor], hemi_size, sub_size, seam_radius: int) -> Dict[str, Region]:
    seam = torch.stack([boundary_band(masks[n], seam_radius) for n in PART_NAMES]).amax(0)
    return {
        "whole_brain": Region(None, None),
        "left_hemi": Region(masks["lhemi"], crop_window(masks["lhemi"], hemi_size)),
        "right_hemi": Region(masks["rhemi"], crop_window(masks["rhemi"], hemi_size)),
        "cerebellum_brainstem": Region(masks["sub"], crop_window(masks["sub"], sub_size)),
        "seam": Region(seam, None),
    }


def preprocess(volume: torch.Tensor, region: Region) -> torch.Tensor:
    """Crop to the region window, z-score inside the support, zero elsewhere."""
    support = torch.ones_like(volume, dtype=torch.bool) if region.mask is None else region.mask > 0.5
    if region.window is not None:
        volume = volume[(..., *region.window)]
        support = support[(..., *region.window)]
    active = volume[support]
    normalized = (volume - active.mean()) / active.std(unbiased=False)
    return torch.where(support, normalized, torch.zeros_like(normalized))


def load_feature_model(device: torch.device):
    model = torch.hub.load(HUB_REPOSITORY, model=HUB_MODEL, trust_repo=True).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


@torch.no_grad()
def extract_features(model, volumes, regions: Dict[str, Region], device) -> Dict[str, torch.Tensor]:
    features = {name: [] for name in regions}
    for volume in volumes:
        volume = volume.to(device=device, dtype=torch.float32)
        for name, region in regions.items():
            output = model(preprocess(volume, region))
            features[name].append(F.adaptive_avg_pool3d(output, 1).flatten(1).double().cpu())
    return {name: torch.cat(values, 0) for name, values in features.items()}


def frechet_distance(generated: torch.Tensor, real: torch.Tensor) -> float:
    """FID between two equally sized feature sets, computed from centred factors."""
    n = min(generated.shape[0], real.shape[0])
    generated, real = generated[:n].double(), real[:n].double()
    origin = generated[0].clone()
    generated, real = generated - origin, real - origin
    mean_g, mean_r = generated.mean(0), real.mean(0)
    factor_g = (generated - mean_g) / math.sqrt(n - 1)
    factor_r = (real - mean_r) / math.sqrt(n - 1)
    nuclear = torch.linalg.svdvals(factor_g @ factor_r.T).sum()
    value = (mean_g - mean_r).dot(mean_g - mean_r) + factor_g.square().sum() + factor_r.square().sum() - 2.0 * nuclear
    return max(0.0, float(value))


def real_volumes(args, n_max: int):
    records = load_split_records(args.csv, args.data_split_json_path, args.whole_key)["test"][:n_max]
    transform = build_image_transforms(parse_floats(args.spacing), parse_ints(args.size))
    for record in records:
        yield transform({"image": record["image"]})["image"].unsqueeze(0).float()


def generated_volumes(directory: str, n_max: int):
    for path in sorted(glob.glob(os.path.join(directory, "*.nii.gz")))[:n_max]:
        yield torch.from_numpy(np.asarray(nib.load(path).get_fdata(), dtype=np.float32))[None, None]


def main():
    parser = argparse.ArgumentParser(description="MedicalNet FID on template-defined supports.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--cohort", nargs="+", required=True, help="name=<directory of generated .nii.gz>")
    cli = parser.parse_args()
    config = dict(DEFAULTS)
    config.update(load_config(cli.config))
    args = argparse.Namespace(**config)
    placeholders = [k for k, v in config.items() if isinstance(v, str) and v.startswith("<PATH-TO-")]
    if placeholders:
        parser.error(f"Replace placeholder path(s) in the config before running: {', '.join(placeholders)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_feature_model(device)
    masks = load_template_masks(args.template_dir, parse_floats(args.spacing), parse_ints(args.size), device)
    regions = build_regions(masks, parse_ints(args.hemi_size), parse_ints(args.sub_size), int(args.seam_radius))

    cache = Path(args.real_feature_cache) if args.real_feature_cache else None
    if cache is not None and cache.exists():
        real = torch.load(cache, weights_only=False)
    else:
        real = extract_features(model, real_volumes(args, int(args.num_samples)), regions, device)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(real, cache)
    print(f"real test volumes: {real['whole_brain'].shape[0]}")

    out_path = Path(args.out)
    results = json.load(open(out_path)) if out_path.exists() else {}
    for spec in cli.cohort:
        name, directory = spec.split("=", 1) if "=" in spec else (Path(spec).name, spec)
        generated = extract_features(model, generated_volumes(directory, int(args.num_samples)), regions, device)
        results[name] = {region: FID_SCALE * frechet_distance(generated[region], real[region]) for region in regions}
        results[name]["n_generated"] = int(generated["whole_brain"].shape[0])
        print(name, json.dumps({k: round(v, 2) if isinstance(v, float) else v for k, v in results[name].items()}))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
