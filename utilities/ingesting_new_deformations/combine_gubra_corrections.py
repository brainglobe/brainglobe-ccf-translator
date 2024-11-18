###Jobs for tomorrow
###Interpolate missing values so that the arrays are complete
###Invert transforms on big computer
###Calculate transform from oriented to orig
###Finalise metadata


from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
import scipy
import json
import os
from scipy.ndimage import map_coordinates
import requests
from brainglobe_ccf_translator.deformation.forward_transform import (
    create_deformation_coords,
)
from brainglobe_ccf_translator.deformation.interpolation.NearestNDInterpolator import (
    NearestNDInterpolator,
)
from scipy.ndimage import binary_dilation, binary_erosion

root_path = r"/home/harryc/github/gubra/Multimodal_mouse_brain_atlas_files"
out_path = r"/home/harryc/github/brainglobe_ccf_translator/brainglobe_ccf_translator/"


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


def create_deformation_coords(deformation_arr):
    coords = np.mgrid[
        0 : deformation_arr.shape[1],
        0 : deformation_arr.shape[2],
        0 : deformation_arr.shape[3],
    ]
    deformed_coords = coords + deformation_arr
    return deformed_coords


def open_transformation(transform_path):
    deformation_img = nib.load(transform_path)
    deformation = np.asarray(deformation_img.dataobj)
    deformation = np.transpose(deformation, (3, 0, 1, 2))
    return deformation


def combine_deformations(deformation_a, deformation_b):
    deformation_a = apply_transform(
        deformation_a, deformation_b, order=2, apply_to_coords=True
    )
    return deformation_a + deformation_b


def apply_transform(data, deformation, order, apply_to_coords=False):
    deformation_coords = create_deformation_coords(deformation)
    if apply_to_coords:
        out_data = np.empty(deformation.shape)
        for i in range(data.shape[0]):
            out_data[i] = scipy.ndimage.map_coordinates(
                data[i], deformation_coords, order=order
            )
    else:
        out_data = np.empty(deformation[0].shape)
        out_data = scipy.ndimage.map_coordinates(data, deformation_coords, order=order)
    return out_data


def resize_transform(arr, scale):
    """performs a regular grid interpolation"""
    # Original indices
    _, z_indices, y_indices, x_indices = np.indices(arr.shape)
    # New indices with double the resolution
    x_new_indices = np.linspace(
        x_indices.min(), x_indices.max(), int(arr.shape[3] * scale[3])
    )
    y_new_indices = np.linspace(
        y_indices.min(), y_indices.max(), int(arr.shape[2] * scale[2])
    )
    z_new_indices = np.linspace(
        z_indices.min(), z_indices.max(), int(arr.shape[1] * scale[1])
    )

    # Create a grid of new indices
    new_indices = np.meshgrid(
        z_new_indices, y_new_indices, x_new_indices, indexing="ij"
    )
    new_shape = np.array(arr.shape)
    new_shape[1] = new_shape[1] * scale[1]
    new_shape[2] = new_shape[2] * scale[2]
    new_shape[3] = new_shape[3] * scale[3]
    new_array = np.zeros(new_shape)
    for i in range(3):
        # New array with double the resolution
        new_array[i] = map_coordinates(arr[i], new_indices, order=1)
    new_array[0] = new_array[0] * scale[1]
    new_array[1] = new_array[1] * scale[2]
    new_array[2] = new_array[2] * scale[3]
    return new_array


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
    out_arr = arr.copy()
    """This function changes the input size of a transformation without changing the output"""
    output_shape = out_arr.shape
    initial_offset = calculate_offset(original_input_shape, output_shape)
    out_arr -= initial_offset
    scale = np.array(new_input_shape) / np.array(original_input_shape)
    out_arr[0] *= scale[1]
    out_arr[1] *= scale[2]
    out_arr[2] *= scale[3]
    new_offset = calculate_offset(new_input_shape, output_shape)
    out_arr += new_offset
    return out_arr


perens_stpt = nib.load(f"{root_path}/AIBS_CCFv3_space_original/ccfv3_orig_temp.nii.gz")
perens_stpt_arr = perens_stpt.get_fdata()

save_path = r"/home/harryc/github/brainglobe_ccf_translator/brainglobe_ccf_translator_local/metadata/deformation_fields/perens_mri_mouse/"
first = nib.load(f"{save_path}/perens_mri_mouse_pull_allen_mouse.nii.gz")
##here we correct the deformation matrix so it has an origin of zero
first_arr = first.get_fdata()
first_arr = np.transpose(first_arr, (3, 0, 1, 2))

save_path = r"/home/harryc/github/brainglobe_ccf_translator/brainglobe_ccf_translator_local/metadata/deformation_fields/perens_stpt_mouse/"
second = nib.load(f"{save_path}/perens_stpt_mouse_pull_perens_mri_mouse.nii.gz")
second_arr = second.get_fdata()
second_arr = np.transpose(second_arr, (3, 0, 1, 2))

###The first arr must have the same input size as the second when adding together
first_arr_output_size = np.array(first_arr.shape[1:])
second_arr_output_size = np.array(second_arr.shape[1:])
first_arr = resize_input(
    first_arr, (1, *perens_stpt_arr.shape), (1, *first_arr.shape[1:])
)
ccf2orig = combine_deformations(first_arr, second_arr)
ccf2orig = resize_input(
    ccf2orig, (1, *first_arr.shape[1:]), (1, *perens_stpt_arr.shape)
)


# test_transform = apply_transform(perens_stpt_arr, ccf2orig, order=1)
# transform_out = np.transpose(ccf2orig, [0,2,3,1])[:,70:-70][:,::-1]
ccf2orig = np.transpose(ccf2orig, [1, 2, 3, 0])
transform_out = nib.Nifti1Image(ccf2orig, np.eye(4))

save_path = r"/home/harryc/github/brainglobe_ccf_translator/brainglobe_ccf_translator_local/metadata/deformation_fields/perens_stpt_mouse"
nib.save(transform_out, f"{save_path}/perens_stpt_mouse_pull_allen_mouse.nii.gz")
img = nib.load(f"{save_path}/perens_stpt_mouse_pull_allen_mouse.nii.gz")
arr = img.get_fdata()
arr = np.transpose(arr, [3, 0, 1, 2])
arr[1] -= 70

arr = np.transpose(arr, [1, 2, 3, 0])

transform_out = nib.Nifti1Image(arr, np.eye(4))

nib.save(transform_out, f"{save_path}/perens_stpt_mouse_pull_allen_mouse.nii.gz")

# test_out = nib.Nifti1Image(test_transform, perens_stpt.affine, perens_stpt.header)
# nib.save(test_out,f"stpt_warped_2_stpt.nii.gz" )
img = nib.load(f"{save_path}/perens_stpt_mouse_pull_allen_mouse.nii.gz")
arr = img.get_fdata()
arr = np.transpose(arr, [3, 0, 1, 2])
out_arr = invert_deformation(arr, output_shape=(456, 668, 320))

out_arr = out_arr[:, :, 70:-70, :]
out_arr[1] += 140

out_arr = out_arr.transpose([0, 2, 3, 1])
out_arr = out_arr[[1, 2, 0]]
out_arr = out_arr[:, :, ::-1, :]
out_arr[1] *= -1


out_arr = np.transpose(out_arr, [1, 2, 3, 0])
nib.save(
    nib.Nifti1Image(out_arr, np.eye(4)),
    f"/home/harryc/github/brainglobe_ccf_translator/brainglobe_ccf_translator_local/metadata/deformation_fields/allen_mouse/allen_mouse_pull_perens_stpt_mouse.nii.gz",
)
