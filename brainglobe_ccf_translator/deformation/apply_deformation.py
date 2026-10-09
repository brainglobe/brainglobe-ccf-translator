import json
import os

import nibabel as nib
import numpy as np
import requests
from scipy.ndimage import map_coordinates

_COORDINATE_CHUNK_BYTES = 16 * 1024**2


def _coordinate_blocks(shape, dtype=np.float64):
    """Reuse a small coordinate buffer, with at least one output plane."""
    plane_bytes = 3 * int(np.prod(shape[1:])) * np.dtype(dtype).itemsize
    depth = max(1, _COORDINATE_CHUNK_BYTES // max(1, plane_bytes))
    depth = min(depth, max(1, shape[0]))
    coordinates = np.empty((3, min(depth, shape[0]), *shape[1:]), dtype=dtype)
    for start in range(0, shape[0], depth):
        stop = min(start + depth, shape[0])
        yield slice(start, stop), coordinates[:, : stop - start]


def _deformation_coordinate_blocks(deformation):
    shape = deformation.shape[1:]
    axes = [np.arange(size) for size in shape]
    dtype = np.result_type(np.int_, deformation.dtype)
    for slab, coordinates in _coordinate_blocks(shape, dtype):
        for axis, ticks in enumerate(axes):
            axis_shape = [1, 1, 1]
            axis_shape[axis] = -1
            if axis == 0:
                ticks = ticks[slab]
            # Add global indices directly to preserve floating-point rounding.
            np.add(
                ticks.reshape(axis_shape),
                deformation[axis, slab],
                out=coordinates[axis],
            )
        yield slab, coordinates


def _map_coordinates_into(data, coordinates, order, output):
    if data.dtype == output.dtype:
        map_coordinates(data, coordinates, order=order, output=output)
    else:
        # Preserve SciPy's rounding to the input dtype before assignment.
        output[...] = map_coordinates(data, coordinates, order=order)


def invert_dim_order(order):
    inverse = [0] * len(order)
    for i, o in enumerate(order):
        inverse[o] = i
    return inverse


def create_deformation_coords(deformation_arr):
    shape = deformation_arr.shape[1:]
    # Preserve promotion from the original integer grid, including float32
    # fields producing float64 coordinates, without allocating a dense grid.
    deformed_coords = np.empty(
        (3, *shape), dtype=np.result_type(np.int_, deformation_arr.dtype)
    )
    for axis, size in enumerate(shape):
        axis_shape = [1, 1, 1]
        axis_shape[axis] = size
        np.add(
            np.arange(size).reshape(axis_shape),
            deformation_arr[axis],
            out=deformed_coords[axis],
        )
    return deformed_coords


def open_transformation(transform_path):
    deformation_img = nib.load(transform_path)
    deformation = np.asarray(deformation_img.dataobj)
    deformation = np.transpose(deformation, (3, 0, 1, 2))
    return deformation


def apply_transform(data, deformation, order, apply_to_coords=False):
    if order > 1:
        # Keep spline prefiltering to once per component, not once per slab.
        blocks = [(slice(None), create_deformation_coords(deformation))]
    else:
        blocks = _deformation_coordinate_blocks(deformation)
    if apply_to_coords:
        out_data = np.empty(deformation.shape)
    else:
        out_data = np.empty(deformation.shape[1:], dtype=data.dtype)
    for slab, coordinates in blocks:
        if apply_to_coords:
            for i in range(data.shape[0]):
                _map_coordinates_into(
                    data[i], coordinates, order, out_data[i, slab]
                )
        else:
            _map_coordinates_into(data, coordinates, order, out_data[slab])
    return out_data


def combine_deformations(deformation_a, deformation_b):
    deformation_a = apply_transform(
        deformation_a, deformation_b, order=1, apply_to_coords=True
    )
    deformation_a += deformation_b
    return deformation_a


def resize_transform(arr, scale):
    """performs a regular grid interpolation"""
    # Axis bounds come from the shape; a dense 4D index grid can use
    # more memory than the deformation field itself.
    x_new_indices = np.linspace(
        0, arr.shape[3] - 1, int(arr.shape[3] * scale[3])
    )
    y_new_indices = np.linspace(
        0, arr.shape[2] - 1, int(arr.shape[2] * scale[2])
    )
    z_new_indices = np.linspace(
        0, arr.shape[1] - 1, int(arr.shape[1] * scale[1])
    )

    new_shape = np.array(arr.shape)
    new_shape[1] = int(new_shape[1] * scale[1])
    new_shape[2] = int(new_shape[2] * scale[2])
    new_shape[3] = int(new_shape[3] * scale[3])
    new_array = np.empty(new_shape)
    for slab, new_indices in _coordinate_blocks(new_shape[1:]):
        # Slice the full linspace so slab boundaries keep identical rounding.
        new_indices[0] = z_new_indices[slab, None, None]
        new_indices[1] = y_new_indices[None, :, None]
        new_indices[2] = x_new_indices[None, None, :]
        for i in range(3):
            _map_coordinates_into(arr[i], new_indices, 1, new_array[i, slab])
    new_array[0] *= scale[1]
    new_array[1] *= scale[2]
    new_array[2] *= scale[3]
    return new_array


def resize_transformation(deform_arr, array_size):
    if (np.array(deform_arr.shape)[1:] != np.array(array_size)).any():
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
    original_input_shape = np.asarray(original_input_shape, dtype=float)
    output_shape = np.asarray(output_shape, dtype=int)

    def _axis_lin(length, samples):
        if samples <= 1:
            return np.zeros(samples, dtype=float)
        span = max(length - 1.0, 0.0)
        return np.linspace(0.0, span, samples)

    coordinate_difference = np.empty((3, *output_shape[1:]))
    for axis, (length, samples) in enumerate(
        zip(original_input_shape[1:], output_shape[1:])
    ):
        axis_shape = [1, 1, 1]
        axis_shape[axis] = samples
        offset = _axis_lin(length, samples) - np.arange(samples)
        coordinate_difference[axis] = offset.reshape(axis_shape)
    return coordinate_difference


def resize_input(arr, original_input_shape, new_input_shape):
    out_arr = arr.copy()
    output_shape = out_arr.shape
    out_arr -= calculate_offset(original_input_shape, output_shape)
    scale = np.array(new_input_shape) / np.array(original_input_shape)
    out_arr[0] *= scale[1]
    out_arr[1] *= scale[2]
    out_arr[2] *= scale[3]
    out_arr += calculate_offset(new_input_shape, output_shape)
    return out_arr


def extract_metadata(metadata, source_metadata, target_metadata, start, stop):
    translation_metadata = metadata[
        (source_metadata == stop) & (target_metadata == start)
    ]
    if len(translation_metadata) > 1:
        raise Exception("Error more than one matching transformation found.")
    return translation_metadata.to_dict(orient="list")


def handle_padding(
    deform_arr, temp_padding, original_voxel_size, target_shape
):
    padding = np.array(json.loads(temp_padding))
    x_pad = padding[0] / original_voxel_size
    y_pad = padding[1] / original_voxel_size
    z_pad = padding[2] / original_voxel_size
    temp_padding = np.array([x_pad, y_pad, z_pad])
    deform_padding = np.concatenate(([[0, 0]], temp_padding), axis=0)
    if deform_arr is not None:
        deform_arr = pad_neg(deform_arr, deform_padding, mode="constant")
        for i in range(len(temp_padding)):
            deform_arr[i] += temp_padding[i][0]
        # Update target_shape to match new deformation field shape
        target_shape = np.array(deform_arr.shape[1:])
    return deform_arr, temp_padding, target_shape


def handle_dim_order(
    deform_arr, dim_order, target_shape, pad_sum, temp_padding, dim_order_sum
):
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


def _open_scaled_transformation(path, vector):
    deformation = open_transformation(path)
    # Newly loaded fields are private to this route. Nibabel's default mmap
    # mode is copy-on-write, so scaling never changes the file on disk.
    if (
        deformation.flags.writeable
        and np.result_type(deformation, vector) == deformation.dtype
    ):
        if vector != 1:
            deformation *= vector
        return deformation
    return deformation * vector


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
        deform_arr = _open_scaled_transformation(deform_path, vector)
        old_voxel_size = float(
            translation_metadata["transformation_resolution_micron"][0]
        )
        final_voxel_size = old_voxel_size
        target_shape = np.array(deform_arr.shape[1:])
    else:
        new_voxel_size = float(
            translation_metadata["transformation_resolution_micron"][0]
        )
        deform_b = _open_scaled_transformation(deform_path, vector)

        if new_voxel_size != old_voxel_size:
            deform_b = resize_transformation(
                deform_b,
                (
                    np.array(deform_b.shape[1:])
                    * (new_voxel_size / old_voxel_size)
                ),
            )

        deform_arr = combine_deformations(deform_arr, deform_b)
        target_shape = np.array(deform_arr.shape[1:])

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
        metadata["source_space"]
        + "_P"
        + metadata["source_age_pnd"].astype(str)
    )
    target_metadata = (
        metadata["target_space"]
        + "_P"
        + metadata["target_age_pnd"].astype(str)
    )
    temp_padding = None

    for i in range(1, len(route)):
        start = route[i - 1]
        stop = route[i]
        translation_metadata = extract_metadata(
            metadata, source_metadata, target_metadata, start, stop
        )

        if (
            translation_metadata["padding_micron"][0]
            != "[[0, 0], [0, 0], [0, 0]]"
        ):
            deform_arr, temp_padding, target_shape = handle_padding(
                deform_arr,
                translation_metadata["padding_micron"][0],
                original_voxel_size,
                target_shape,
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
                deform_arr,
                translation_metadata["dim_flip"][0],
                pad_sum,
                flip_sum,
            )

        if translation_metadata["file_name"][0] != "False":
            vector = int(translation_metadata["vector"][0])
            deform_path = os.path.join(
                base_path,
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
        current_shape = np.array(deform_arr.shape[1:])
        if not np.array_equal(current_shape, target_shape):
            deform_arr = resize_input(
                deform_arr,
                original_input_shape=(1, *deform_arr.shape[1:]),
                new_input_shape=(1, *target_shape),
            )
    return deform_arr, pad_sum, flip_sum, dim_order_sum, final_voxel_size
