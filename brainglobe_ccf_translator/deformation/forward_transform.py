import numpy as np
from .interpolation.NearestNDInterpolator import NearestNDInterpolator
from .apply_deformation import create_deformation_coords
import nibabel as nib
from scipy.ndimage import binary_dilation, binary_erosion


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
    values[out_mask] = interpolator(points[out_mask], k=40)
    # Reshape the interpolated volume to the original shape
    interpolated_volume = values.reshape(shape)
    return interpolated_volume


def invert_transformation_volume(forward_arr):
    coords = np.mgrid[
        0 : forward_arr.shape[1], 0 : forward_arr.shape[2], 0 : forward_arr.shape[3]
    ]
    output = np.zeros(forward_arr.shape)
    output[:] = np.nan
    temp_forward = create_deformation_coords(forward_arr)
    temp_forward_mask = np.isnan(temp_forward[0])
    temp_forward = temp_forward[:, ~temp_forward_mask]
    coords = coords[:, ~temp_forward_mask]
    temp_forward = np.round(temp_forward).astype(int)
    output[:, temp_forward[0], temp_forward[1], temp_forward[2]] = forward_arr[
        :, coords[0], coords[1], coords[2]
    ]
    return output

def invert_deformation(deformation_arr_transpose, output_shape = None):
    if output_shape is None:
        output_shape = deformation_arr_transpose.shape[1:]
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
