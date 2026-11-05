#!/usr/bin/env python3
"""Create AnaDiff part volumes and CSV manifests from processed MRI data."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm


LOGGER = logging.getLogger("prep_data")

PARTS: dict[str, dict[str, Any]] = {
    "left_hemi": {"prefix": "lhemi", "labels": [2, 3, 4, 5, 10, 11, 12, 13, 17, 18, 26, 28]},
    "right_hemi": {"prefix": "rhemi", "labels": [41, 42, 43, 44, 49, 50, 51, 52, 53, 54, 58, 60]},
    "right_hemi_mirror": {"prefix": "rhemi_mirror", "labels": [41, 42, 43, 44, 49, 50, 51, 52, 53, 54, 58, 60]},
    "sub": {"prefix": "sub", "labels": [7, 8, 46, 47, 14, 15, 16]},
    "whole_brain": {"prefix": "whole_brain", "labels": "whole"},
    "left": {"prefix": "left", "labels": "left"},
    "right": {"prefix": "right", "labels": "right"},
    "right_mirror": {"prefix": "right_mirror", "labels": "right_mirror"},
}

SEX = {"F": 0, "M": 1, "X": 2}
GROUP = {"CN": 0, "MCI": 1, "AD": 2, "SMC": 3, "EMCI": 4, "LMCI": 5}


def normalize_mri(vol: np.ndarray, mask: np.ndarray, pct: tuple[float, float] = (1, 99)) -> np.ndarray:
    x = vol.astype(np.float32)
    mask_bool = mask.astype(bool)
    vals = x[mask_bool] if mask_bool.any() else x.ravel()
    lo, hi = np.percentile(vals, [pct[0], pct[1]])
    if hi <= lo:
        lo, hi = float(x.min()), float(x.max())
    clipped = np.clip(x, lo, hi)
    scaled = (clipped - lo) / (hi - lo + 1e-8)
    return (scaled * 2.0 - 1.0).astype(np.float32)


def save_nii_like(arr: np.ndarray, ref_img: nib.Nifti1Image, out_path: Path, dtype=None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if dtype is not None:
        arr = arr.astype(dtype)
    nib.save(nib.Nifti1Image(arr, ref_img.affine, ref_img.header), str(out_path))


def _load_head_mask(mask_path: Path | None, seg: np.ndarray) -> np.ndarray:
    if mask_path is not None and mask_path.exists():
        return nib.load(str(mask_path)).get_fdata().astype(np.uint8)
    return (seg != 0).astype(np.uint8)


def _part_mask(seg: np.ndarray, head_mask: np.ndarray, labels: Any) -> np.ndarray:
    if labels == "whole":
        return head_mask.copy().astype(np.uint8)
    if labels == "left":
        mid = seg.shape[0] // 2
        mask = head_mask.copy()
        mask[:mid, :, :] = 0
        return mask.astype(np.uint8)
    if labels == "right" or labels == "right_mirror":
        mid = seg.shape[0] // 2
        mask = head_mask.copy()
        mask[mid:, :, :] = 0
        return mask.astype(np.uint8)
    return np.isin(seg, labels).astype(np.uint8)


def _crop_part(arr: np.ndarray, mask: np.ndarray, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    if prefix == "rhemi":
        return arr[:96, :, :], mask[:96, :, :]
    if prefix in {"lhemi", "rhemi_mirror"}:
        return arr[-96:, :, :], mask[-96:, :, :]
    if prefix == "sub":
        return arr[:, :128, :96], mask[:, :128, :96]
    return arr, mask


def save_part_nii(
    *,
    t1_path: Path,
    seg_path: Path,
    mask_path: Path | None,
    labels: Any,
    prefix: str,
    postfix: str,
    save_dir: Path,
    whole_brain_norm: bool = False,
) -> tuple[str, str, float]:
    seg_img = nib.load(str(seg_path))
    seg = seg_img.get_fdata().astype(np.int32)
    head_mask = _load_head_mask(mask_path, seg)

    t1_img = nib.load(str(t1_path))
    t1 = t1_img.get_fdata().astype(np.float32)
    if whole_brain_norm:
        t1 = normalize_mri(t1, head_mask)

    mask = _part_mask(seg, head_mask, labels)
    if prefix in {"rhemi_mirror", "right_mirror"}:
        mask = mask[::-1, :, :]
        part_arr = np.where(mask > 0, t1[::-1, :, :], -1.0)
    else:
        part_arr = np.where(mask > 0, t1, -1.0)

    part_arr, mask = _crop_part(part_arr, mask, prefix)
    head_voxels = max(float(head_mask.sum()), 1.0)
    part_vol = float(mask.sum()) / head_voxels

    subject_id = t1_path.parent.name
    filename = f"{prefix}_{subject_id}{postfix}"
    mask_dir = save_dir.parent / f"{save_dir.name}_mask"
    part_path = save_dir / filename
    part_mask_path = mask_dir / f"mask_{filename}"
    save_nii_like(part_arr, t1_img, part_path, dtype=np.float32)
    save_nii_like(mask, t1_img, part_mask_path, dtype=np.uint8)
    return str(part_path), str(part_mask_path), part_vol


def _is_placeholder(path: str) -> bool:
    return str(path).startswith("<PATH-TO-")


def _metadata_for_image(image_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
    if image_id not in metadata:
        raise KeyError(f"Missing metadata for imageID={image_id}")
    row = metadata[image_id]
    sex = row.get("Sex")
    group = row.get("Group")
    if isinstance(sex, str):
        sex = SEX[sex]
    if isinstance(group, str):
        group = GROUP[group]
    return {
        "age": float(row["Age"]) / 100.0,
        "sex": int(sex),
        "group": int(group),
    }


def _write_combined_csv(outdir: Path, prefixes: list[str], postfix: str) -> Path:
    combined = None
    for prefix in prefixes:
        part_csv = outdir / f"{prefix}_{postfix}.csv"
        part_df = pd.read_csv(part_csv).sort_values("imageID", ignore_index=True)
        if combined is None:
            combined = part_df[["imageID", "age", "sex", "group"]].copy()
        cols = part_df[["imageID", "image", "mask", "part", "vol"]].rename(
            columns={
                "image": prefix,
                "mask": f"{prefix}_mask",
                "part": f"{prefix}_part",
                "vol": f"{prefix}_vol",
            }
        )
        combined = combined.merge(cols, on="imageID", how="inner")
    if combined is None:
        raise ValueError("No part CSVs were available to combine.")
    out_path = outdir / f"whole_brain+3parts+masks_{postfix}.csv"
    combined.to_csv(out_path, index=False)
    return out_path


def _write_hemi_pair_csv(outdir: Path, combined_csv: Path, postfix: str) -> Path | None:
    df = pd.read_csv(combined_csv)
    required = {"lhemi", "lhemi_mask", "lhemi_vol", "rhemi_mirror", "rhemi_mirror_mask", "rhemi_mirror_vol"}
    if not required.issubset(df.columns):
        LOGGER.warning("Skipping paired hemi CSV; combined CSV does not contain both lhemi and rhemi_mirror columns.")
        return None
    rows = []
    for _, img in df.iterrows():
        for part, pair in [("lhemi", "rhemi_mirror"), ("rhemi_mirror", "lhemi")]:
            rows.append(
                {
                    "imageID": img["imageID"],
                    "age": img["age"],
                    "sex": img["sex"],
                    "group": img["group"],
                    "image": img[part],
                    "mask": img[f"{part}_mask"],
                    "part": 0 if part == "lhemi" else 1,
                    "vol": img[f"{part}_vol"],
                    "pair": img[pair],
                    "pair_mask": img[f"{pair}_mask"],
                    "pair_part": 0 if pair == "lhemi" else 1,
                    "pair_vol": img[f"{pair}_vol"],
                }
            )
    out_path = outdir / f"hemi_{postfix}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare whole-brain, hemisphere, and substructure NIfTI files plus CSV metadata for AnaDiff."
    )
    parser.add_argument("--root", default="<PATH-TO-PROCESSED-TURBOPREP-OUTPUT>", help="Directory with processed subject folders.")
    parser.add_argument("--metadata_json", default="<PATH-TO-METADATA-JSON>", help="JSON keyed by imageID with Age, Sex, and Group fields.")
    parser.add_argument(
        "--part",
        default="whole_brain,left_hemi,right_hemi_mirror,sub",
        help=f"Comma-separated parts. Available: {','.join(PARTS)}",
    )
    parser.add_argument("--outdir", default="<PATH-TO-PROCESSED-PARTS-DIR>", help="Output directory for part NIfTI files and CSVs.")
    parser.add_argument("--postfix", default="processed", help="Manifest/output suffix, for example release_v1.")
    parser.add_argument("--image_postfix", default="_normalized.nii.gz", help="Suffix for written NIfTI filenames.")
    parser.add_argument("--whole_brain_norm", action="store_true", help="Apply robust whole-brain normalization before masking parts.")
    parser.add_argument("--combine", action=argparse.BooleanOptionalAction, default=True, help="Write combined whole/part CSV.")
    parser.add_argument("--write_hemi_csv", action=argparse.BooleanOptionalAction, default=True, help="Write paired left/right-hemi CSV when possible.")
    parser.add_argument("--fail_fast", action="store_true", help="Stop on first failed subject.")
    args = parser.parse_args()

    for path_arg in (args.root, args.metadata_json, args.outdir):
        if _is_placeholder(path_arg):
            parser.error("Replace <PATH-TO-...> placeholders with real paths before running.")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    root = Path(args.root).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    metadata_path = Path(args.metadata_json).expanduser().resolve()
    if not root.is_dir():
        parser.error(f"--root does not exist or is not a directory: {root}")
    if not metadata_path.is_file():
        parser.error(f"--metadata_json does not exist: {metadata_path}")
    outdir.mkdir(parents=True, exist_ok=True)

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    requested_parts = [part.strip() for part in args.part.split(",") if part.strip()]
    unknown_parts = sorted(set(requested_parts) - set(PARTS))
    if unknown_parts:
        parser.error(f"Unknown part(s): {unknown_parts}. Available: {sorted(PARTS)}")

    written_prefixes = []
    for part in requested_parts:
        spec = PARTS[part]
        prefix = spec["prefix"]
        labels = spec["labels"]
        save_dir = outdir / f"turboprepout_{prefix}_{args.postfix}"
        save_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        failed = 0

        for subject_dir in tqdm(sorted(p for p in root.iterdir() if p.is_dir()), desc=part):
            try:
                t1_path = subject_dir / "normalized.nii.gz"
                seg_path = subject_dir / "segm.nii.gz"
                mask_path = subject_dir / "mask.nii.gz"
                if not t1_path.exists() or not seg_path.exists():
                    raise FileNotFoundError(f"Missing normalized.nii.gz or segm.nii.gz in {subject_dir}")
                part_path, part_mask_path, part_vol = save_part_nii(
                    t1_path=t1_path,
                    seg_path=seg_path,
                    mask_path=mask_path if mask_path.exists() else None,
                    labels=labels,
                    prefix=prefix,
                    postfix=args.image_postfix,
                    save_dir=save_dir,
                    whole_brain_norm=args.whole_brain_norm,
                )
                image_id = subject_dir.name
                meta = _metadata_for_image(image_id, metadata)
                rows.append(
                    {
                        "imageID": image_id,
                        "image": part_path,
                        "mask": part_mask_path,
                        "part": part,
                        "vol": part_vol,
                        **meta,
                    }
                )
            except Exception as exc:
                failed += 1
                LOGGER.exception("Failed part=%s subject=%s: %s", part, subject_dir.name, exc)
                if args.fail_fast:
                    raise

        part_df = pd.DataFrame(rows)
        df_path = outdir / f"{prefix}_{args.postfix}.csv"
        part_df.to_csv(df_path, index=False)
        written_prefixes.append(prefix)
        LOGGER.info("Wrote %s rows=%d failed=%d", df_path, len(part_df), failed)

    combined_csv = None
    if args.combine:
        combined_csv = _write_combined_csv(outdir, written_prefixes, args.postfix)
        LOGGER.info("Wrote %s", combined_csv)

    if args.write_hemi_csv and combined_csv is not None:
        hemi_csv = _write_hemi_pair_csv(outdir, combined_csv, args.postfix)
        if hemi_csv is not None:
            LOGGER.info("Wrote %s", hemi_csv)


if __name__ == "__main__":
    main()
