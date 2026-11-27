"""Ingest the swc_female_rat <-> whs_sd_rat deformation fields.

The Waxholm Space (WHS) Sprague Dawley rat template (BrainGlobe
``whs_sd_rat_39um``) was registered to the SWC female Lister Hooded rat STPT
template (BrainGlobe ``swc_female_rat_25um``) with FireANTs. The STPT template
was the fixed image and WHS the moving image. FireANTs saved the result as an
ANTs-style composed displacement field (initial affine plus deformation).

Both images were given unit spacing, zero origin and identity direction during
registration. As a result, the ANTs field is a voxel displacement:

    whs_index = stpt_index + displacement(stpt_index)

Here ``stpt_index`` is a 25 um voxel on the STPT grid and ``whs_index`` is a
39 um voxel on the WHS grid. In NIfTI (x, y, z) order both images are in
BrainGlobe ASR orientation, so no reordering or flipping is needed.

CCF translator expects inputs and outputs to be at the same resolution, so:

1. ``resize_input`` rescales the target coordinates from the 39 um WHS grid to
   a 25 um WHS grid. This gives ``swc_female_rat_pull_whs_sd_rat`` (25 um).
2. ``resize_transformation`` resamples that field to 39 um, and
   ``invert_deformation`` inverts it onto the native WHS grid. This gives
   ``whs_sd_rat_pull_swc_female_rat`` (39 um).

Usage:
    python swc_female_rat.py /path/to/composed_warp_ants.nii.gz
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

from brainglobe_ccf_translator.deformation.apply_deformation import (
    resize_input,
    resize_transformation,
)
from brainglobe_ccf_translator.deformation.forward_transform import (
    invert_deformation,
)

VERSION = "1.0"
SWC_SPACE = "swc_female_rat"
WHS_SPACE = "whs_sd_rat"
SWC_MICRON = 25
WHS_MICRON = 39
# BrainGlobe whs_sd_rat_39um shape in ASR order.
WHS_SHAPE = np.array([1024, 512, 512])


def load_ants_warp(path):
    """Return the ANTs displacement as (3, A, S, R) in WHS 39 um voxels."""
    img = nib.load(str(path))
    arr = np.asanyarray(img.dataobj)
    if arr.ndim == 5:
        arr = arr[:, :, :, 0, :]
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"Unexpected ANTs warp shape {arr.shape}")
    spacing = np.abs(np.diag(img.affine)[:3])
    if not np.allclose(spacing, 1) or not np.allclose(img.affine[:3, 3], 0):
        raise ValueError(
            "Expected a unit-spacing, zero-origin warp "
            "(displacements in voxels); got affine\n"
            f"{img.affine}"
        )
    return np.transpose(arr, (3, 0, 1, 2)).astype(np.float32)


def save_volume(volume, voxel_size_micron, file_name):
    file_name.parent.mkdir(parents=True, exist_ok=True)
    affine = np.eye(4)
    affine[:3, :3] *= voxel_size_micron
    volume = np.transpose(volume, (1, 2, 3, 0)).astype(np.float32)
    image = nib.Nifti1Image(volume, affine=affine)
    image.header.set_xyzt_units(3)
    nib.save(image, str(file_name))
    print(f"saved {file_name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ants_warp", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("~/.brainglobe/deformation_fields").expanduser(),
    )
    args = parser.parse_args()

    # STPT grid (25 um) pulling from WHS. Rescale the target coordinates
    # from the WHS 39 um grid to the WHS grid at 25 um.
    warp = load_ants_warp(args.ants_warp)
    whs_shape_at_swc = np.round(WHS_SHAPE * WHS_MICRON / SWC_MICRON).astype(
        int
    )
    swc_field = resize_input(
        warp, (1, *WHS_SHAPE), (1, *whs_shape_at_swc)
    ).astype(np.float32)
    del warp
    save_volume(
        swc_field,
        SWC_MICRON,
        args.output_dir
        / SWC_SPACE
        / f"{SWC_SPACE}_pull_{WHS_SPACE}_v{VERSION}.nii.gz",
    )

    # WHS grid (39 um) pulling from STPT. Bring the field to 39 um, then
    # invert it onto the native WHS grid.
    swc_shape_at_whs = np.round(
        np.array(swc_field.shape[1:]) * SWC_MICRON / WHS_MICRON
    ).astype(int)
    swc_field = resize_transformation(swc_field, swc_shape_at_whs)
    whs_field = invert_deformation(swc_field, WHS_SHAPE)
    del swc_field
    save_volume(
        whs_field,
        WHS_MICRON,
        args.output_dir
        / WHS_SPACE
        / f"{WHS_SPACE}_pull_{SWC_SPACE}_v{VERSION}.nii.gz",
    )


if __name__ == "__main__":
    main()
