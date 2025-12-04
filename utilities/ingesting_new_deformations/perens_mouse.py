"""I wanted to include the Gubra multimodal atlas but ran into a few issues.
First the origin of the volumes is not zero so this had to be corrected for in the deformations.
Second, they transform between two volumes of different resolutions, something which I suspect may cause
problems for CCF translator in the future, here is my attempt to correct for these issues.
In the end it is close to the Gubra result, maybe off by less than 1 voxel.


This script assumes you have downloaded the Gubra atlas files available from:
https://www.neuropedia.dk/wp-content/uploads/Multimodal_mouse_brain_atlas_files_v2.7z
"""

import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas

VERSION = "1.1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from brainglobe_ccf_translator.deformation.apply_deformation import (
    resize_input,
)
from brainglobe_ccf_translator.deformation.forward_transform import (
    interpolate_volume,
    invert_deformation,
)


def _reset_offsets(img: nib.Nifti1Image) -> None:
    for axis, key in enumerate("xyz"):
        img.header[f"qoffset_{key}"] = 0
        img.affine[axis, -1] = 0


def _ensure_parent(path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def zero_origin_image(
    input_path: Path, output_path: Path
) -> tuple[int, int, int]:
    img = nib.load(str(input_path))
    data = np.asanyarray(img.dataobj)
    _reset_offsets(img)
    _ensure_parent(output_path)
    nib.save(nib.Nifti1Image(data, img.affine, img.header), str(output_path))
    return data.shape


def _lsfm_mask(arr: np.ndarray) -> np.ndarray:
    """
    This is specific to the gubra deformations.
    For some reason they have multiple voxels which use 1000 as the fill value.
    """
    return (
        (arr[:, :, :, :, 1] == 1000)
        | (arr[:, :, :, :, 2] == 1000)
        | (arr[:, :, :, :, 0] == 0)
    )


def _resize_vector_field(
    arr: np.ndarray, source_shape: tuple[int, int, int], fill_missing: bool
) -> np.ndarray:
    vec = np.transpose(arr, [3, 0, 1, 2])
    vec = resize_input(vec, (1, *source_shape), vec.shape)
    if fill_missing:
        interpolation_mask = np.ones(vec[0].shape, dtype=bool)
        for axis in range(vec.shape[0]):
            vec[axis] = interpolate_volume(vec[axis], interpolation_mask)
    vec = resize_input(vec, vec.shape, (1, *source_shape))
    return np.transpose(vec, [1, 2, 3, 0])


def _reorient_vector_field(arr, source_shape, new_order, flips):
    vec = np.transpose(arr, [3, 0, 1, 2])
    vec = resize_input(vec, (1, *source_shape), vec.shape)
    new_order = np.array(new_order)
    vec = np.transpose(vec, [0, *(new_order + 1)])
    vec = vec[new_order]
    if flips[0]:
        vec = vec[:, ::-1]
        vec[0] *= -1
    if flips[1]:
        vec = vec[:, :, ::-1]
        vec[1] *= -1
    if flips[2]:
        vec = vec[:, :, :, ::-1]
        vec[2] *= -1
    source_shape = np.array(source_shape)
    source_shape = source_shape[new_order]
    vec = resize_input(vec, vec.shape, (1, *source_shape))
    vec = np.transpose(vec, [1, 2, 3, 0])
    return vec


def process_deformation_field(
    input_path: Path,
    output_path: Path,
    source_shape: tuple[int, int, int],
    *,
    mask_fn=None,
    mask_label: str | None = None,
    resize_to_source: bool = False,
    fill_missing: bool = False,
    crop_input=[[0, 0], [0, 0], [0, 0]],
    crop_output=[[0, 0], [0, 0], [0, 0]],
    new_order=[0, 1, 2],
    flips=[False, False, False],
) -> None:
    img = nib.load(str(input_path))
    arr = img.get_fdata()
    if mask_fn:
        mask = mask_fn(arr)
        arr[mask, :] = np.nan

    diff = (np.array(img.shape[:3]) - np.array(source_shape)) / 2.0
    arr[:, :, :, :, 0] -= img.affine[0, 0] * diff[0]
    arr[:, :, :, :, 1] -= img.affine[1, 1] * diff[1]
    arr[:, :, :, :, 2] -= img.affine[2, 2] * diff[2]

    arr = arr[
        crop_output[0][0] : (arr.shape[0] - crop_output[0][1]),
        crop_output[1][0] : (arr.shape[1] - crop_output[1][1]),
        crop_output[2][0] : (arr.shape[2] - crop_output[2][1]),
        :,
        :,
    ]
    arr[:, :, :, :, 0] -= crop_input[0][0]
    arr[:, :, :, :, 1] -= crop_input[1][0]
    arr[:, :, :, :, 2] -= crop_input[2][0]

    arr = np.squeeze(arr, axis=3)
    if resize_to_source:
        arr = _resize_vector_field(arr, source_shape, fill_missing)
    arr = _reorient_vector_field(arr, source_shape, new_order, flips)
    _reset_offsets(img)
    _ensure_parent(output_path)
    nib.save(nib.Nifti1Image(arr, img.affine, img.header), str(output_path))


def invert_and_save(
    input_path: Path,
    reference_path: Path,
    output_path: Path,
    output_shape,
    *,
    crop_input=[[0, 0], [0, 0], [0, 0]],
    crop_output=[[0, 0], [0, 0], [0, 0]],
) -> None:
    img = nib.load(str(input_path))
    arr = np.transpose(img.get_fdata(), [3, 0, 1, 2])
    invert_arr = invert_deformation(arr, output_shape)
    invert_arr = np.transpose(invert_arr, [1, 2, 3, 0])
    invert_arr = invert_arr[
        crop_output[0][0] : (invert_arr.shape[0] - crop_output[0][1]),
        crop_output[1][0] : (invert_arr.shape[1] - crop_output[1][1]),
        crop_output[2][0] : (invert_arr.shape[2] - crop_output[2][1]),
        :,
    ]
    invert_arr[:, :, :, 0] -= crop_input[0][0]
    invert_arr[:, :, :, 1] -= crop_input[1][0]
    invert_arr[:, :, :, 2] -= crop_input[2][0]
    _ensure_parent(output_path)
    nib.save(
        nib.Nifti1Image(invert_arr, img.affine, img.header), str(output_path)
    )


root_path = Path(
    "~/brainglobe_workingdir/perens_stereotaxic_mri_mouse/Multimodal_mouse_brain_atlas_files"
)
out_path = Path("~/.brainglobe/").expanduser()
mri_dir = root_path / "MRI_space_oriented"
ccfv3_oriented_dir = root_path / "AIBS_CCFv3_space_oriented"
ccfv3_original_dir = root_path / "AIBS_CCFv3_space_original"
deformation_dir = root_path / "Deformation_fields"
allen_mouse_dir = out_path / "deformation_fields" / "allen_mouse"
def_out_dir = out_path / "deformation_fields"


zero_origin_image(
    mri_dir / "mri_temp.nii.gz", mri_dir / "mri_new_header.nii.gz"
)
ccfv3_shape = zero_origin_image(
    ccfv3_oriented_dir / "ccfv3_temp.nii.gz",
    ccfv3_oriented_dir / "ccfv3_new_header.nii.gz",
)


ccfv3_template = ccfv3_oriented_dir / "ccfv3_temp.nii.gz"

ccfv3_original_shape = zero_origin_image(
    ccfv3_original_dir / "ccfv3_orig_temp.nii.gz",
    ccfv3_original_dir / "ccfv3_orig_new_header.nii.gz",
)
atlas = BrainGlobeAtlas("allen_mouse_25um")

process_deformation_field(
    deformation_dir / "ccfv3_orig_2_mri_deffield.nii.gz",
    def_out_dir
    / "perens_stereotaxic_mri_mouse"
    / f"perens_stereotaxic_mri_mouse_pull_allen_mouse_v{VERSION}.nii.gz",
    ccfv3_original_shape,
    resize_to_source=True,
    crop_input=[[0, 0], [70, 70], [0, 0]],
    new_order=[1, 2, 0],
    flips=[False, True, False],
)


invert_and_save(
    def_out_dir
    / "perens_stereotaxic_mri_mouse"
    / f"perens_stereotaxic_mri_mouse_pull_allen_mouse_v{VERSION}.nii.gz",
    ccfv3_template,
    def_out_dir
    / "allen_mouse"
    / f"allen_mouse_pull_perens_stereotaxic_mri_mouse_v{VERSION}.nii.gz",
    atlas.shape,
    crop_input=[[0, 0], [0, 0], [0, 0]],
    crop_output=[[0, 0], [0, 0], [0, 0]],
)


process_deformation_field(
    deformation_dir / "ccfv3_orig_2_lsfm_deffield.nii.gz",
    def_out_dir
    / "perens_multimodal_lsfm"
    / f"perens_multimodal_lsfm_pull_allen_mouse_v{VERSION}.nii.gz",
    ccfv3_original_shape,
    resize_to_source=True,
    crop_input=[[0, 0], [70, 70], [0, 0]],
    mask_fn=_lsfm_mask,
    new_order=[1, 2, 0],
    flips=[False, True, False],
    fill_missing=True,
)
invert_and_save(
    def_out_dir
    / "perens_multimodal_lsfm"
    / f"perens_multimodal_lsfm_pull_allen_mouse_v{VERSION}.nii.gz",
    ccfv3_template,
    def_out_dir
    / "allen_mouse"
    / f"allen_mouse_pull_perens_multimodal_lsfm_v{VERSION}.nii.gz",
    atlas.shape,
    crop_input=[[0, 0], [0, 0], [0, 0]],
    crop_output=[[0, 0], [0, 0], [0, 0]],
)
