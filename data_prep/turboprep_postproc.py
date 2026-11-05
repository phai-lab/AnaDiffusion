#!/usr/bin/env python3
"""Normalize and crop turboprep-style NIfTI outputs for AnaDiff training."""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from typing import Iterable

import nibabel as nib
import numpy as np
from monai import transforms
from tqdm import tqdm


LOGGER = logging.getLogger("turboprep_postproc")


def normalize_mri(vol: np.ndarray, mask: np.ndarray, pct: tuple[float, float] = (1, 99)) -> np.ndarray:
    """Robustly scale intensities inside `mask` to [-1, 1]."""
    x = vol.astype(np.float32)
    mask_bool = mask.astype(bool)
    vals = x[mask_bool] if mask_bool.any() else x.ravel()
    lo, hi = np.percentile(vals, [pct[0], pct[1]])
    if hi <= lo:
        lo, hi = float(x.min()), float(x.max())
    clipped = np.clip(x, lo, hi)
    scaled = (clipped - lo) / (hi - lo + 1e-8)
    return (scaled * 2.0 - 1.0).astype(np.float32)


def _is_placeholder(path: str) -> bool:
    return str(path).startswith("<PATH-TO-")


def _parse_sample_limit(value: str) -> int | None:
    if str(value).upper() == "ALL":
        return None
    limit = int(value)
    if limit < 1:
        raise argparse.ArgumentTypeError("--n_samples must be ALL or a positive integer")
    return limit


def _iter_subject_dirs(root: Path, n_samples: int | None) -> Iterable[Path]:
    entries = sorted(p for p in root.iterdir() if p.is_dir())
    if n_samples is not None:
        entries = entries[:n_samples]
    return entries


def build_postprocess_transform(spatial_size: tuple[int, int, int]):
    keys = ["normalized", "mask", "segm"]
    return transforms.Compose(
        [
            transforms.LoadImaged(keys=keys),
            transforms.EnsureChannelFirstd(keys=keys),
            transforms.EnsureTyped(keys=keys, track_meta=False),
            transforms.CropForegroundd(keys=keys, source_key="mask"),
            transforms.DivisiblePadd(keys=keys, k=32, mode="constant", constant_values=-1.0),
            transforms.SpatialPadd(keys=keys, spatial_size=spatial_size, mode="constant", constant_values=-1.0),
        ]
    )


def _squeeze_channel(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 4 and arr.shape[0] == 1:
        return arr[0]
    return arr


def process_subject(subject_dir: Path, outdir: Path, data_transforms) -> bool:
    t1_path = subject_dir / "normalized.nii.gz"
    mask_path = subject_dir / "mask.nii.gz"
    segm_path = subject_dir / "segm.nii.gz"

    missing = [p.name for p in (t1_path, mask_path, segm_path) if not p.exists()]
    if missing:
        LOGGER.warning("Skipping %s; missing %s", subject_dir.name, ", ".join(missing))
        return False

    t1_img = nib.load(str(t1_path))
    t1 = t1_img.get_fdata().astype(np.float32)
    mask = nib.load(str(mask_path)).get_fdata().astype(np.uint8)
    norm_t1 = normalize_mri(t1, mask)

    subject_out = outdir / subject_dir.name
    subject_out.mkdir(parents=True, exist_ok=True)
    out_norm_path = subject_out / "normalized.nii.gz"
    out_mask_path = subject_out / "mask.nii.gz"
    out_segm_path = subject_out / "segm.nii.gz"

    nib.save(nib.Nifti1Image(norm_t1, t1_img.affine, t1_img.header), str(out_norm_path))
    shutil.copy2(mask_path, out_mask_path)
    shutil.copy2(segm_path, out_segm_path)

    out = data_transforms(
        {
            "normalized": str(out_norm_path),
            "mask": str(out_mask_path),
            "segm": str(out_segm_path),
        }
    )
    norm_out = _squeeze_channel(np.asarray(out["normalized"]))
    mask_out = _squeeze_channel(np.asarray(out["mask"]))
    segm_out = _squeeze_channel(np.asarray(out["segm"]))

    mask_out = (mask_out > 0).astype(np.uint8)
    segm_out = np.where(segm_out > 0, segm_out, 0).astype(np.int16)
    norm_out = np.where(mask_out > 0, norm_out, -1.0).astype(np.float32)
    nib.save(nib.Nifti1Image(norm_out, t1_img.affine, t1_img.header), str(out_norm_path))
    nib.save(nib.Nifti1Image(mask_out, t1_img.affine, t1_img.header), str(out_mask_path))
    nib.save(nib.Nifti1Image(segm_out, t1_img.affine, t1_img.header), str(out_segm_path))
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize and crop turboprep-style outputs containing normalized.nii.gz, mask.nii.gz, and segm.nii.gz."
    )
    parser.add_argument("--root", default="<PATH-TO-TURBOPREP-OUTPUT>", help="Directory containing one subdirectory per subject.")
    parser.add_argument("--outdir", default="<PATH-TO-PROCESSED-TURBOPREP-OUTPUT>", help="Directory to write processed subjects.")
    parser.add_argument("--n_samples", default="ALL", help="Number of subjects to process, or ALL.")
    parser.add_argument("--spatial_size", default="160,192,160", help="Output spatial size as D,H,W.")
    parser.add_argument("--fail_fast", action="store_true", help="Stop on the first processing error.")
    args = parser.parse_args()

    if _is_placeholder(args.root) or _is_placeholder(args.outdir):
        parser.error("Replace <PATH-TO-...> placeholders with real paths before running.")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    root = Path(args.root).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    if not root.is_dir():
        parser.error(f"--root does not exist or is not a directory: {root}")
    outdir.mkdir(parents=True, exist_ok=True)

    spatial_size = tuple(int(x.strip()) for x in args.spatial_size.split(","))
    if len(spatial_size) != 3:
        parser.error("--spatial_size must have three comma-separated integers")
    n_samples = _parse_sample_limit(args.n_samples)
    data_transforms = build_postprocess_transform(spatial_size)

    processed = 0
    failed = 0
    for subject_dir in tqdm(list(_iter_subject_dirs(root, n_samples)), desc="subjects"):
        try:
            if process_subject(subject_dir, outdir, data_transforms):
                processed += 1
        except Exception as exc:
            failed += 1
            LOGGER.exception("Failed to process %s: %s", subject_dir.name, exc)
            if args.fail_fast:
                raise

    LOGGER.info("Finished. processed=%d failed=%d outdir=%s", processed, failed, outdir)


if __name__ == "__main__":
    main()
