# Data Preparation

`turboprep_postproc.py` normalizes, foreground-crops, pads, and writes each subject folder with:

- `normalized.nii.gz`
- `mask.nii.gz`
- `segm.nii.gz`

`prep_data.py` creates whole-brain, hemisphere, mirrored hemisphere, and substructure NIfTI files plus CSV manifests for training.

The scripts validate `<PATH-TO-...>` placeholders and stop with a clear error until real paths are supplied.
