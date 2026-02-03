"""
the provided deformation field takes a 32um input vol and maps it to 25um.
We need to rescale the input to 25um as ccf translator expects inputs and
outputs to be the same resolution.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import tifffile
import brainglobe_atlasapi
from brainglobe_ccf_translator.deformation.forward_transform import (
    invert_deformation,
)


VERSION = "1.0"
voxel_size_micron = 25
perens = brainglobe_atlasapi.BrainGlobeAtlas('perens_stereotaxic_mri_mouse_25um')


def open_deformation_field(deformation_paths):
    deformation_arr = np.stack([tifffile.imread(p) for p in deformation_paths])
    return deformation_arr


def save_volume(volume, file_name):
    affine = np.eye(4)
    affine[:3, :3] *= voxel_size_micron
    volume = np.transpose(volume, (1, 2, 3, 0))
    image = nib.Nifti1Image(volume, affine=affine)
    image.header.set_xyzt_units(3)
    nib.save(image, file_name)


root_dir = Path("~/brainglobe_workingdir/dorr_mri/").expanduser()
deformation_paths = [
    f"{root_dir}/deformation_field_0.tiff",
    f"{root_dir}/deformation_field_1.tiff",
    f"{root_dir}/deformation_field_2.tiff",
]
deformation_arr = open_deformation_field(deformation_paths)

file_name = Path(
    "~/.brainglobe/deformation_fields/dorr_mouse_mri/"
).expanduser()
file_name.mkdir(parents=True, exist_ok=True)
save_volume(
    deformation_arr,
    f"{file_name}/dorr_mouse_mri_pull_perens_stereotaxic_mri_mouse.nii.gz",
)

invert_arr = invert_deformation(deformation_arr, output_shape=np.array(perens.shape))

file_name = Path(
    "~/.brainglobe/deformation_fields/perens_stereotaxic_mri_mouse/"
).expanduser()

save_volume(
    invert_arr,
    f"{file_name}/perens_stereotaxic_mri_mouse_pull_dorr_mouse_mri.nii.gz",
)
