"""I wanted to include the Gubra multimodal atlas but ran into a few issues. 
First the origin of the volumes is not zero so this had to be corrected for in the deformations.
Second, they transform between two volumes of different resolutions, something which I suspect may cause 
problems for CCF translator in the future, here is my attempt to correct for these issues. 
In the end it is close to the Gubra result, maybe off by less than 1 voxel. 
"""

import matplotlib.pyplot as plt
import nibabel as nib
import os
import numpy as np
from scipy.ndimage import zoom
from ccf_translator.deformation.interpolation.NearestNDInterpolator import (
    NearestNDInterpolator,
)
from ccf_translator.deformation.forward_transform import create_deformation_coords
from scipy.ndimage import binary_dilation, binary_erosion


def invert_deformation(deformation_arr_transpose, output_shape):
    deformed_coords = create_deformation_coords(deformation_arr_transpose)
    new_coords = np.round(deformed_coords).astype(int)
    new_coords[0][new_coords[0] >= output_shape[0]] = output_shape[0] - 1
    new_coords[1][new_coords[1] >= output_shape[1]] = output_shape[1] - 1
    new_coords[2][new_coords[2] >= output_shape[2]] = output_shape[2] - 1
    new_coords[0][new_coords[0] < 0] = 0
    new_coords[1][new_coords[1] < 0] = 0
    new_coords[2][new_coords[2] < 0] = 0
    reversed_deform = np.zeros((3, *output_shape))
    reversed_deform[:] = np.nan
    reversed_deform[:, new_coords[0], new_coords[1], new_coords[2]] = (
        -deformation_arr_transpose
    )
    # Assuming `img` is your image array
    mask = np.isnan(reversed_deform[0])
    # Create a mask for NaNs that are at the edge for efficiency
    edge_mask = mask & ~binary_dilation(~mask)
    eroded_img = binary_erosion(edge_mask, structure=np.ones((3, 3, 3)))
    for i in range(3):
        reversed_deform[i] = interpolate_volume(reversed_deform[i], mask=~eroded_img)
    return reversed_deform


def interpolate_volume(volume, mask):
    # Get the shape of the volume
    shape = volume.shape
    # Create a grid of points in the volume
    grid = np.mgrid[0 : shape[0], 0 : shape[1], 0 : shape[2]]
    points = grid.reshape((3, -1)).T
    # Flatten the volume
    mask = mask.flatten()
    values = volume.flatten()
    nan_pos = np.isnan(values)
    interp_mask = ~nan_pos & mask
    # Create the interpolator
    interpolator = NearestNDInterpolator(points[interp_mask], values[interp_mask])
    # Interpolate the volume
    out_mask = nan_pos & mask
    values[out_mask] = interpolator(points[out_mask], k=30, weights="distance")
    # Reshape the interpolated volume to the original shape
    interpolated_volume = values.reshape(shape)
    return interpolated_volume


def calculate_offset(original_input_shape, output_shape):
    x = np.arange(0, original_input_shape[1], original_input_shape[1] / output_shape[1])
    y = np.arange(0, original_input_shape[2], original_input_shape[2] / output_shape[2])
    z = np.arange(0, original_input_shape[3], original_input_shape[3] / output_shape[3])
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    original_coordinates = np.stack([X, Y, Z])
    target_coordinates = np.indices(output_shape[1:])
    coordinate_difference = original_coordinates - target_coordinates
    return coordinate_difference


def resize_input(arr, original_input_shape, new_input_shape):
    output_shape = arr.shape
    initial_offset = calculate_offset(original_input_shape, output_shape)
    arr -= initial_offset
    scale = np.array(new_input_shape) / np.array(original_input_shape)
    arr[0] *= scale[1]
    arr[1] *= scale[2]
    arr[2] *= scale[3]
    new_offset = calculate_offset(new_input_shape, output_shape)
    arr += new_offset
    return arr


root_path = r"/home/harryc/github/gubra/Multimodal_mouse_brain_atlas_files"
out_path = r"/home/harryc/github/ccf_translator/ccf_translator/"

##set the origin as zero in the transform file if you want to use that with these files
##here we just remove the offsets from the volume that is to be registered
img = nib.load(f"{root_path}/MRI_space_oriented/mri_temp.nii.gz")
img.header["qoffset_x"] = 0
img.header["qoffset_y"] = 0
img.header["qoffset_z"] = 0
source_arr = np.asanyarray(img.dataobj)
img.affine[0, 3] = 0
img.affine[1, 3] = 0
img.affine[2, 3] = 0
out_im = nib.Nifti1Image(source_arr, img.affine, img.header)
nib.save(out_im, f"{root_path}/MRI_space_oriented/mri_new_header.nii.gz")


##set the origin as zero in the transform file if you want to use that with these files
##here we just remove the offsets from the volume that is to be registered
img = nib.load(f"{root_path}/AIBS_CCFv3_space_oriented/ccfv3_temp.nii.gz")
img.header["qoffset_x"] = 0
img.header["qoffset_y"] = 0
img.header["qoffset_z"] = 0
source_arr = np.asanyarray(img.dataobj)
img.affine[0, 3] = 0
img.affine[1, 3] = 0
img.affine[2, 3] = 0
out_im = nib.Nifti1Image(source_arr, img.affine, img.header)
nib.save(out_im, f"{root_path}/AIBS_CCFv3_space_oriented/ccfv3_new_header.nii.gz")


##here we correct the deformation matrix so it has an origin of zero
img = nib.load(f"{root_path}/Deformation_fields/ccfv3_2_mri_deffield.nii.gz")
##here we correct the deformation matrix so it has an origin of zero
target_shape = np.array(img.shape[:3])
diff = (target_shape / 2) - (np.array(source_arr.shape) / 2)
arr = img.get_fdata()
# mask = arr[:,:,:,:,1]==1000
# arr[mask,:] = np.nan
###These numbers can be calculated as the midpoint of the source volume minus midpoint of target (maybe other way round)
arr[:, :, :, :, 0] -= img.affine[0, 0] * diff[0]
arr[:, :, :, :, 1] -= img.affine[1, 1] * diff[1]
arr[:, :, :, :, 2] -= img.affine[2, 2] * diff[2]
img.header["qoffset_x"] = 0
img.header["qoffset_y"] = 0
img.header["qoffset_z"] = 0
img.affine[0, -1] = 0
img.affine[1, -1] = 0
img.affine[2, -1] = 0
arr = np.squeeze(arr, 3)
# arr = np.transpose(arr, [3,0,1,2])
# arr = resize_input(arr, (1,*source_arr.shape), arr.shape)
# t_mask = np.ones(arr[0].shape).astype(bool)
# print('mri nan percent')
# print(np.sum(mask) / np.sum(t_mask))
# for i in range(3):
#     arr[i] = interpolate_volume(arr[i],t_mask)
# arr = resize_input(arr, arr.shape, (1,*source_arr.shape))
# arr = np.transpose(arr, [1,2,3,0])
out_im = nib.Nifti1Image(arr, img.affine, img.header)
save_path = f"{out_path}/metadata/deformation_fields/perens_stpt_mouse/"
if not os.path.exists(save_path):
    os.makedirs(save_path)
nib.save(out_im, f"{save_path}/perens_mri_mouse_pull_perens_stpt_mouse.nii.gz")

#######################################
img = nib.load(f"{root_path}/Deformation_fields/ccfv3_2_lsfm_deffield.nii.gz")
##here we correct the deformation matrix so it has an origin of zero
target_shape = np.array(img.shape[:3])
diff = (target_shape / 2) - (np.array(source_arr.shape) / 2)
arr = img.get_fdata()
mask = (
    (arr[:, :, :, :, 1] == 1000)
    | (arr[:, :, :, :, 2] == 1000)
    | (arr[:, :, :, :, 0] == 0)
)
arr[mask, :] = np.nan

###These numbers can be calculated as the midpoint of the source volume minus midpoint of target (maybe other way round)
arr[:, :, :, :, 0] -= img.affine[0, 0] * diff[0]
arr[:, :, :, :, 1] -= img.affine[1, 1] * diff[1]
arr[:, :, :, :, 2] -= img.affine[2, 2] * diff[2]
img.header["qoffset_x"] = 0
img.header["qoffset_y"] = 0
img.header["qoffset_z"] = 0
img.affine[0, -1] = 0
img.affine[1, -1] = 0
img.affine[2, -1] = 0
arr = np.squeeze(arr, 3)
arr = np.transpose(arr, [3, 0, 1, 2])
arr = resize_input(arr, (1, *source_arr.shape), arr.shape)
t_mask = np.ones(arr[0].shape).astype(bool)
print("lsfm nan percent")
print(np.sum(mask) / np.sum(t_mask))
for i in range(3):
    arr[i] = interpolate_volume(arr[i], t_mask)
arr = resize_input(arr, arr.shape, (1, *source_arr.shape))
arr = np.transpose(arr, [1, 2, 3, 0])
out_im = nib.Nifti1Image(arr, img.affine, img.header)
save_path = f"{out_path}/metadata/deformation_fields/perens_stpt_mouse/"
if not os.path.exists(save_path):
    os.makedirs(save_path)
nib.save(out_im, f"{save_path}/perens_lsfm_mouse_pull_perens_stpt_mouse.nii.gz")

######################################
##set the origin as zero in the transform file if you want to use that with these files
##here we just remove the offsets from the volume that is to be registered
img = nib.load(f"{root_path}/AIBS_CCFv3_space_original/ccfv3_orig_temp.nii.gz")

img.header["qoffset_x"] = 0
img.header["qoffset_y"] = 0
img.header["qoffset_z"] = 0
source_arr = np.asanyarray(img.dataobj)
img.affine[0, 3] = 0
img.affine[1, 3] = 0
img.affine[2, 3] = 0
out_im = nib.Nifti1Image(source_arr, img.affine, img.header)
nib.save(out_im, f"{root_path}/AIBS_CCFv3_space_original/ccfv3_orig_new_header.nii.gz")
#######################################
img = nib.load(f"{root_path}/Deformation_fields/ccfv3_orig_2_mri_deffield.nii.gz")
##here we correct the deformation matrix so it has an origin of zero
target_shape = np.array(img.shape[:3])
diff = (target_shape / 2) - (np.array(source_arr.shape) / 2)
arr = img.get_fdata()
###These numbers can be calculated as the midpoint of the source volume minus midpoint of target (maybe other way round)
arr[:, :, :, :, 0] -= img.affine[0, 0] * diff[0]
arr[:, :, :, :, 1] -= img.affine[1, 1] * diff[1]
arr[:, :, :, :, 2] -= img.affine[2, 2] * diff[2]
img.header["qoffset_x"] = 0
img.header["qoffset_y"] = 0
img.header["qoffset_z"] = 0
img.affine[0, -1] = 0
img.affine[1, -1] = 0
img.affine[2, -1] = 0
arr = np.squeeze(arr, 3)
arr = np.transpose(arr, [3, 0, 1, 2])
# arr = resize_input(arr, (1,*source_arr.shape), arr.shape)
# t_mask = np.ones(arr[0].shape).astype(bool)
# print('mri nan percent')
# print(np.sum(mask) / np.sum(t_mask))
# arr = resize_input(arr, arr.shape, (1,*source_arr.shape))
arr = np.transpose(arr, [1, 2, 3, 0])
out_im = nib.Nifti1Image(arr, img.affine, img.header)
save_path = f"{out_path}/metadata/deformation_fields/perens_mri_mouse/"
if not os.path.exists(save_path):
    os.makedirs(save_path)
nib.save(out_im, f"{save_path}/perens_mri_mouse_pull_allen_mouse.nii.gz")
#######################################


output_shape = nib.load(
    f"{root_path}/AIBS_CCFv3_space_oriented/ccfv3_temp.nii.gz"
).shape
save_path = f"{out_path}/metadata/deformation_fields/perens_stpt_mouse/"
img = nib.load(f"{save_path}/perens_mri_mouse_pull_perens_stpt_mouse.nii.gz")
arr = img.get_fdata()
arr = np.transpose(arr, [3, 0, 1, 2])
invert_arr = invert_deformation(arr, output_shape)
invert_arr = np.transpose(invert_arr, [1, 2, 3, 0])
out_im = nib.Nifti1Image(invert_arr, img.affine, img.header)
save_path = f"{out_path}/metadata/deformation_fields/perens_mri_mouse/"
nib.save(out_im, f"{save_path}/perens_stpt_mouse_pull_perens_mri_mouse.nii.gz")


output_shape = nib.load(
    f"{root_path}/AIBS_CCFv3_space_oriented/ccfv3_temp.nii.gz"
).shape
save_path = f"{out_path}/metadata/deformation_fields/perens_stpt_mouse/"
img = nib.load(f"{save_path}/perens_lsfm_mouse_pull_perens_stpt_mouse.nii.gz")
arr = img.get_fdata()
arr = np.transpose(arr, [3, 0, 1, 2])
invert_arr = invert_deformation(arr, output_shape)
invert_arr = np.transpose(invert_arr, [1, 2, 3, 0])
out_im = nib.Nifti1Image(invert_arr, img.affine, img.header)
save_path = f"{out_path}/metadata/deformation_fields/perens_lsfm_mouse/"
nib.save(out_im, f"{save_path}/perens_stpt_mouse_pull_perens_lsfm_mouse.nii.gz")

output_shape = nib.load(
    f"{root_path}/AIBS_CCFv3_space_oriented/ccfv3_temp.nii.gz"
).shape
save_path = f"{out_path}/metadata/deformation_fields/allen_mouse/"
img = nib.load(f"{save_path}/perens_mri_mouse_pull_allen_mouse.nii.gz")
arr = img.get_fdata()
arr = np.transpose(arr, [3, 0, 1, 2])
invert_arr = invert_deformation(arr, output_shape)
invert_arr = np.transpose(invert_arr, [1, 2, 3, 0])
invert_arr = invert_arr[:, 70:-70, :, :]
out_im = nib.Nifti1Image(invert_arr, img.affine, img.header)
save_path = f"{out_path}/metadata/deformation_fields/perens_mri_mouse/"
nib.save(out_im, f"{save_path}/allen_mouse_pull_perens_mri_mouse.nii.gz")
