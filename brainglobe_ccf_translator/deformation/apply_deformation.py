import numpy as np
import nibabel as nib
import scipy
import json
import os
from scipy.ndimage import map_coordinates
import requests

def invert_dim_order(order):
    inverse = [0] * len(order)
    for i, o in enumerate(order):
        inverse[o] = i
    return inverse

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


def combine_deformations(deformation_a, deformation_b):
    deformation_a = apply_transform(
        deformation_a, deformation_b, order=1, apply_to_coords=True
    )
    return deformation_a + deformation_b


def resize_transform(arr, scale):
    """performs a regular grid interpolation"""
    _, z_indices, y_indices, x_indices = np.indices(arr.shape)
    x_new_indices = np.linspace(
        x_indices.min(), x_indices.max(), int(arr.shape[3] * scale[3])
    )
    y_new_indices = np.linspace(
        y_indices.min(), y_indices.max(), int(arr.shape[2] * scale[2])
    )
    z_new_indices = np.linspace(
        z_indices.min(), z_indices.max(), int(arr.shape[1] * scale[1])
    )

    new_indices = np.meshgrid(
        z_new_indices, y_new_indices, x_new_indices, indexing="ij"
    )
    new_shape = np.array(arr.shape)
    new_shape[1] = new_shape[1] * scale[1]
    new_shape[2] = new_shape[2] * scale[2]
    new_shape[3] = new_shape[3] * scale[3]
    new_array = np.zeros(new_shape)
    for i in range(3):
        new_array[i] = map_coordinates(arr[i], new_indices, order=1)
    new_array[0] = new_array[0] * scale[1]
    new_array[1] = new_array[1] * scale[2]
    new_array[2] = new_array[2] * scale[3]
    return new_array


def resize_transformation(deform_arr, array_size):
    if (np.array(deform_arr.shape)[1:] != np.array(array_size)).all():
        scale = (1, *(np.array(array_size) / np.array(deform_arr.shape)[1:]))
        deform_arr = resize_transform(deform_arr, scale)
    return deform_arr


def pad_neg(array, padding, mode):
    padding = np.round(padding).astype(int)
    for i in range(len(padding)):
        if padding[i][0] < 0:
            array = np.delete(array, np.s_[: abs(padding[i][0])], axis=i)
            padding[i][0] = 0
        if padding[i][1] < 0:
            array = np.delete(array, np.s_[-abs(padding[i][1]) :], axis=i)
            padding[i][1] = 0
    array = np.pad(array, padding, mode=mode)
    return array


def download_deformation_field(url, path):
    print("Downloading file from " + url + " to " + path)
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    r = requests.get(url, allow_redirects=True)
    with open(path, "wb") as file:
        file.write(r.content)


def calculate_offset(original_input_shape, output_shape):
    x = np.linspace(0, original_input_shape[1], output_shape[1], endpoint=False)
    y = np.linspace(0, original_input_shape[2], output_shape[2], endpoint=False)
    z = np.linspace(0, original_input_shape[3], output_shape[3], endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    original_coordinates = np.stack([X, Y, Z])
    target_coordinates = np.indices(output_shape[1:])
    coordinate_difference = original_coordinates - target_coordinates
    return coordinate_difference


def resize_input(arr, original_input_shape, new_input_shape):
    out_arr = arr.copy()
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


def extract_metadata(metadata, source_metadata, target_metadata, start, stop):
    translation_metadata = metadata[
        (source_metadata == stop) & (target_metadata == start)
    ]
    if len(translation_metadata) > 1:
        raise Exception("Error more than one matching transformation found.")
    return translation_metadata.to_dict(orient="list")


def handle_padding(deform_arr, temp_padding, original_voxel_size):
    padding = np.array(json.loads(temp_padding))
    x_pad = padding[0] / original_voxel_size
    y_pad = padding[1] / original_voxel_size
    z_pad = padding[2] / original_voxel_size
    temp_padding = np.array([x_pad, y_pad, z_pad])
    deform_padding = np.concatenate(([[0, 0]], temp_padding), axis=0)
    if deform_arr is not None:
        original_shape = deform_arr.shape
        deform_arr = pad_neg(deform_arr, deform_padding, mode="constant")
        for i in range(len(temp_padding)):
            deform_arr[i] += temp_padding[i][0]
        new_shape = deform_arr.shape
        deform_arr = resize_input(deform_arr, original_shape, new_shape)
    return deform_arr, temp_padding


def handle_dim_order(deform_arr, dim_order, target_shape, pad_sum, temp_padding, dim_order_sum):
    dim_order = list(map(int, dim_order[1:-1].split(", ")))
    pad_sum = pad_sum[dim_order]
    if temp_padding is not None:
        temp_padding = temp_padding[dim_order]
    dim_order_sum = dim_order_sum[dim_order]
    if deform_arr is not None:
        target_shape = target_shape[dim_order]
        deform_dim = np.array(dim_order.copy())
        deform_dim = deform_dim + 1
        deform_dim = [0, *deform_dim]
        deform_arr = np.transpose(deform_arr, deform_dim)
        deform_arr = deform_arr[dim_order]
    return deform_arr, target_shape, pad_sum, temp_padding, dim_order_sum


def handle_dim_flip(deform_arr, dim_flip, pad_sum, flip_sum):
    dim_flip = list(map(eval, dim_flip[1:-1].split(", ")))
    for i in range(3):
        if dim_flip[i]:
            pad_sum[i] = pad_sum[i][::-1]
            flip_sum[i] = not flip_sum[i]
            if deform_arr is not None:
                deform_arr[i] *= -1
                deform_arr = np.flip(deform_arr, axis=i + 1)
    return deform_arr, pad_sum, flip_sum


def load_and_combine_deformation(
    deform_arr,
    deform_path,
    vector,
    translation_metadata,
    old_voxel_size,
    final_voxel_size,
    target_shape,
):
    if deform_arr is None:
        deform_arr = open_transformation(deform_path) * vector
        target_shape = np.array(
            (
                translation_metadata["target_X_physical_size_micron"],
                translation_metadata["target_Y_physical_size_micron"],
                translation_metadata["target_Z_physical_size_micron"],
            )
        )
        old_voxel_size = float(
            translation_metadata["transformation_resolution_micron"][0]
        )
        final_voxel_size = old_voxel_size
        target_shape = target_shape / final_voxel_size
        target_shape = target_shape.flatten()
        dim_order = list(
            map(int, translation_metadata["dim_order"][0][1:-1].split(", "))
        )
        target_shape = target_shape[dim_order]
        if (deform_arr.shape[1:] != target_shape).any():
            deform_arr = resize_input(
                deform_arr,
                original_input_shape=(1, *target_shape),
                new_input_shape=(1, *deform_arr.shape[1:]),
            )
    else:
        old_shape = np.array(
            (
                translation_metadata["target_X_physical_size_micron"],
                translation_metadata["target_Y_physical_size_micron"],
                translation_metadata["target_Z_physical_size_micron"],
            )
        )
        new_voxel_size = float(
            translation_metadata["transformation_resolution_micron"][0]
        )
        dim_order = list(
            map(int, translation_metadata["dim_order"][0][1:-1].split(", "))
        )
        deform_b = open_transformation(deform_path) * vector
        temp_shape = old_shape / new_voxel_size
        temp_shape = temp_shape.flatten()
        temp_shape = temp_shape[dim_order]
        deform_b = resize_input(
            deform_b,
            original_input_shape=(1, *(temp_shape)),
            new_input_shape=(1, *deform_b.shape[1:]),
        )
        if new_voxel_size != old_voxel_size:
            deform_b = resize_transformation(
                deform_b,
                (np.array(deform_b.shape[1:]) * (new_voxel_size / old_voxel_size)),
            )
        old_shape = old_shape / final_voxel_size
        old_shape = old_shape.flatten()
        old_shape = old_shape[dim_order]
        deform_b = resize_input(
            deform_b,
            original_input_shape=(1, *deform_b.shape[1:]),
            new_input_shape=(1, *deform_arr.shape[1:]),
        )
        deform_arr = resize_input(
            deform_arr,
            original_input_shape=(1, *deform_arr.shape[1:]),
            new_input_shape=(1, *old_shape),
        )
        deform_arr = combine_deformations(deform_arr, deform_b)
        deform_arr = resize_input(
            deform_arr,
            original_input_shape=(1, *old_shape),
            new_input_shape=(1, *deform_arr.shape[1:]),
        )
    return deform_arr, final_voxel_size, target_shape, old_voxel_size


def combine_route(route, original_voxel_size, base_path, metadata):
    deform_arr = None
    target_shape = None
    final_voxel_size = None
    old_voxel_size = None
    pad_sum = np.zeros((3, 2))
    flip_sum = [False, False, False]
    dim_order_sum = np.array([0, 1, 2])
    source_metadata = (
        metadata["source_space"] + "_P" + metadata["source_age_pnd"].astype(str)
    )
    target_metadata = (
        metadata["target_space"] + "_P" + metadata["target_age_pnd"].astype(str)
    )
    temp_padding = None

    for i in range(1, len(route)):
        start = route[i - 1]
        stop = route[i]
        translation_metadata = extract_metadata(
            metadata, source_metadata, target_metadata, start, stop
        )

        if translation_metadata["padding_micron"][0] != "[[0, 0], [0, 0], [0, 0]]":
            deform_arr, temp_padding = handle_padding(
                deform_arr,
                translation_metadata["padding_micron"][0],
                original_voxel_size,
            )
            pad_sum += temp_padding

        if translation_metadata["dim_order"][0] != "[0, 1, 2]":
            deform_arr, target_shape, pad_sum, temp_padding, dim_order_sum = (
                handle_dim_order(
                    deform_arr,
                    translation_metadata["dim_order"][0],
                    target_shape,
                    pad_sum,
                    temp_padding,
                    dim_order_sum,
                )
            )

        if translation_metadata["dim_flip"][0] != "[False, False, False]":
            deform_arr, pad_sum, flip_sum = handle_dim_flip(
                deform_arr, translation_metadata["dim_flip"][0], pad_sum, flip_sum
            )

        if translation_metadata["file_name"][0] != "False":
            vector = int(translation_metadata["vector"][0])
            deform_path = os.path.join(
                base_path,
                "metadata",
                "deformation_fields",
                translation_metadata["source_space"][0],
                translation_metadata["file_name"][0],
            )
            if not os.path.exists(deform_path):
                target_url = f"https://data-proxy.ebrains.eu/api/v1/buckets/common-coordinate-framework-translator/deformation_fields/{translation_metadata['source_space'][0]}/{translation_metadata['file_name'][0]}"
                download_deformation_field(target_url, deform_path)
            deform_arr, final_voxel_size, target_shape, old_voxel_size = (
                load_and_combine_deformation(
                    deform_arr,
                    deform_path,
                    vector,
                    translation_metadata,
                    old_voxel_size,
                    final_voxel_size,
                    target_shape,
                )
            )

    if deform_arr is not None:
        deform_arr = resize_input(
            deform_arr,
            original_input_shape=(1, *deform_arr.shape[1:]),
            new_input_shape=(1, *target_shape),
        )
    return deform_arr, pad_sum, flip_sum, dim_order_sum, final_voxel_size
