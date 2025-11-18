"""I wanted to include the Gubra multimodal atlas but ran into a few issues.
First the origin of the volumes is not zero so this had to be corrected for in the deformations.
Second, they transform between two volumes of different resolutions, something which I suspect may cause
problems for CCF translator in the future, here is my attempt to correct for these issues.
In the end it is close to the Gubra result, maybe off by less than 1 voxel.
"""

import nibabel as nib
import numpy as np
from pathlib import Path
import sys
import pandas as pd

VERSION = "1.1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from brainglobe_ccf_translator.deformation.forward_transform import (
    invert_deformation,
    interpolate_volume,
)
from brainglobe_ccf_translator.deformation.apply_deformation import (
    resize_input,
)


def _reset_offsets(img: nib.Nifti1Image) -> None:
    for axis, key in enumerate("xyz"):
        img.header[f"qoffset_{key}"] = 0
        img.affine[axis, -1] = 0


def _ensure_parent(path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def zero_origin_image(input_path: Path, output_path: Path) -> tuple[int, int, int]:
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


def process_deformation_field(
    input_path: Path,
    output_path: Path,
    source_shape: tuple[int, int, int],
    *,
    mask_fn=None,
    mask_label: str | None = None,
    resize_to_source: bool = False,
    fill_missing: bool = False,
) -> None:
    img = nib.load(str(input_path))
    arr = img.get_fdata()
    diff = (np.array(img.shape[:3]) - np.array(source_shape)) / 2.0
    arr[:, :, :, :, 0] -= img.affine[0, 0] * diff[0]
    arr[:, :, :, :, 1] -= img.affine[1, 1] * diff[1]
    arr[:, :, :, :, 2] -= img.affine[2, 2] * diff[2]
    if mask_fn:
        mask = mask_fn(arr)
        arr[mask, :] = np.nan
        if mask_label:
            print(mask_label)
            print(np.mean(mask))
    arr = np.squeeze(arr, axis=3)
    if resize_to_source:
        arr = _resize_vector_field(arr, source_shape, fill_missing)
    _reset_offsets(img)
    _ensure_parent(output_path)
    nib.save(nib.Nifti1Image(arr, img.affine, img.header), str(output_path))


def invert_and_save(
    input_path: Path,
    reference_path: Path,
    output_path: Path,
    *,
    crop: dict[int, slice] | None = None,
) -> None:
    output_shape = nib.load(str(reference_path)).shape
    img = nib.load(str(input_path))
    arr = np.transpose(img.get_fdata(), [3, 0, 1, 2])
    invert_arr = invert_deformation(arr, output_shape)
    invert_arr = np.transpose(invert_arr, [1, 2, 3, 0])
    if crop:
        slices = [slice(None)] * invert_arr.ndim
        for axis, axis_slice in crop.items():
            slices[axis] = axis_slice
        invert_arr = invert_arr[tuple(slices)]
    _ensure_parent(output_path)
    nib.save(nib.Nifti1Image(invert_arr, img.affine, img.header), str(output_path))


root_path = Path("/home/harryc/github/gubra/Multimodal_mouse_brain_atlas_files")
out_path = Path("~/.brainglobe/").expanduser()
mri_dir = root_path / "MRI_space_oriented"
ccfv3_oriented_dir = root_path / "AIBS_CCFv3_space_oriented"
ccfv3_original_dir = root_path / "AIBS_CCFv3_space_original"
deformation_dir = root_path / "Deformation_fields"
perens_stpt_dir = out_path / "deformation_fields" / "perens_stpt_mouse"
perens_mri_dir = out_path / "deformation_fields" / "perens_mri_mouse"
perens_lsfm_dir = out_path / "deformation_fields" / "perens_lsfm_mouse"

def create_metadata_dict(
    source_space,
    target_space,
    source_age_pnd,
    target_age_pnd,
    transformation_resolution_micron,
    affine=np.eye(4),
    dim_order=np.arange(3),
    key_age=True,
    source_key_age=False,
    target_key_age=False,
    padding_micron="[[0,0], [0,0], [0,0]]",
    dim_flip="[False, False, False]",
    vector=1,
    out_dir = ""
):
    filename = f"{source_space}_pull_{target_space}_v{VERSION}.nii.gz"
    img = nib.load(f"{out_dir}/{filename}")
    X_physical_size_micron,Y_physical_size_micron,Z_physical_size_micron,_ = np.array(img.shape) * transformation_resolution_micron

    filename = f"{target_space}_pull_{source_space}_v{VERSION}.nii.gz"
    img = nib.load(f"{out_dir}/{filename}")
    target_X_physical_size_micron,target_Y_physical_size_micron,target_Z_physical_size_micron,_ = np.array(img.shape) * transformation_resolution_micron
    return {
        "filename": filename,
        "source_space": source_space,
        "target_space": target_space,
        "affine": affine,
        "dim_order": dim_order,
        "key_age": key_age,
        "source_age_pnd": source_age_pnd,
        "target_age_pnd": target_age_pnd,
        "source_key_age": source_key_age,
        "target_key_age": target_key_age,
        "padding_micron": padding_micron,
        "transformation_resolution_micron": transformation_resolution_micron,
        "X_physical_size_micron":X_physical_size_micron,
        "Y_physical_size_micron":Y_physical_size_micron,
        "Z_physical_size_micron":Z_physical_size_micron,
        "target_X_physical_size_micron":target_X_physical_size_micron,
        "target_Y_physical_size_micron":target_Y_physical_size_micron,
        "target_Z_physical_size_micron":target_Z_physical_size_micron,
        "dim_flip": dim_flip,
        "vector": vector,
    }


zero_origin_image(mri_dir / "mri_temp.nii.gz", mri_dir / "mri_new_header.nii.gz")
ccfv3_shape = zero_origin_image(
    ccfv3_oriented_dir / "ccfv3_temp.nii.gz",
    ccfv3_oriented_dir / "ccfv3_new_header.nii.gz",
)

# process_deformation_field(
#     deformation_dir / "ccfv3_2_mri_deffield.nii.gz",
#     perens_stpt_dir / "perens_mri_mouse_pull_perens_stpt_mouse.nii.gz",
#     ccfv3_shape,
# )

# process_deformation_field(
#     deformation_dir / "ccfv3_2_lsfm_deffield.nii.gz",
#     perens_stpt_dir / "perens_lsfm_mouse_pull_perens_stpt_mouse.nii.gz",
#     ccfv3_shape,
#     mask_fn=_lsfm_mask,
#     mask_label="lsfm nan percent",
#     resize_to_source=True,
#     fill_missing=True,
# )


ccfv3_template = ccfv3_oriented_dir / "ccfv3_temp.nii.gz"

# invert_and_save(
#     perens_stpt_dir / "perens_mri_mouse_pull_perens_stpt_mouse.nii.gz",
#     ccfv3_template,
#     perens_mri_dir / "perens_stpt_mouse_pull_perens_mri_mouse.nii.gz",
# )

# invert_and_save(
#     perens_stpt_dir / "perens_lsfm_mouse_pull_perens_stpt_mouse.nii.gz",
#     ccfv3_template,
#     perens_lsfm_dir / "perens_stpt_mouse_pull_perens_lsfm_mouse.nii.gz",
# )

ccfv3_original_shape = zero_origin_image(
    ccfv3_original_dir / "ccfv3_orig_temp.nii.gz",
    ccfv3_original_dir / "ccfv3_orig_new_header.nii.gz",
)

process_deformation_field(
    deformation_dir / "ccfv3_orig_2_mri_deffield.nii.gz",
    perens_mri_dir / f"perens_stereotaxic_mri_mouse_pull_allen_mouse_v{VERSION}.nii.gz",
    ccfv3_original_shape,
)

invert_and_save(
    perens_mri_dir / f"perens_stereotaxic_mri_mouse_pull_allen_mouse_v{VERSION}.nii.gz",
    ccfv3_template,
    perens_mri_dir / f"allen_mouse_pull_perens_stereotaxic_mri_mouse_v{VERSION}.nii.gz",
    crop={1: slice(70, -70)},
)


process_deformation_field(
    deformation_dir / "ccfv3_orig_2_lsfm_deffield.nii.gz",
    perens_mri_dir / f"perens_multimodal_lsfm_mouse_pull_allen_mouse_v{VERSION}.nii.gz",
    ccfv3_original_shape,
)

invert_and_save(
    perens_mri_dir / f"perens_multimodal_lsfm_mouse_pull_allen_mouse_v{VERSION}.nii.gz",
    ccfv3_template,
    perens_mri_dir / f"allen_mouse_pull_perens_multimodal_lsfm_mouse_v{VERSION}.nii.gz",
    crop={1: slice(70, -70)},
)


pd.DataFrame([
    create_metadata_dict(
        source_space = "perens_multimodal_lsfm_mouse",
        target_space = "allen_mouse",
        transformation_resolution_micron=25,
        source_age_pnd=56,
        target_age_pnd=56,
        out_dir=perens_mri_dir
        ),

])