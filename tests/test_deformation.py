import numpy as np
import pytest

from brainglobe_ccf_translator.deformation.apply_deformation import (
    apply_transform,
    calculate_offset,
    create_deformation_coords,
    resize_input,
    resize_transform,
)


@pytest.mark.parametrize(
    "shape, scale",
    [
        ((4, 5, 6), (1, 1, 1)),
        ((4, 5, 6), (0.5, 0.4, 0.5)),
        ((4, 5, 6), (2, 1.5, 0.5)),
        ((1, 5, 6), (2, 0.2, 0.5)),
    ],
)
def test_resize_transform_preserves_linear_field(shape, scale):
    # Linear fields have an exact interpolation result, including endpoints.
    z, y, x = np.indices(shape, dtype=float)
    field = np.stack((z + 2 * y, y - x, 3 * x + z))

    result = resize_transform(field, (1, *scale))

    output_shape = tuple(
        int(size * factor) for size, factor in zip(shape, scale)
    )
    axes = [
        np.linspace(0, size - 1, count)
        for size, count in zip(shape, output_shape)
    ]
    z_out, y_out, x_out = np.meshgrid(*axes, indexing="ij")
    expected = np.stack(
        (
            (z_out + 2 * y_out) * scale[0],
            (y_out - x_out) * scale[1],
            (3 * x_out + z_out) * scale[2],
        )
    )

    np.testing.assert_allclose(result, expected, atol=1e-12)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int16])
def test_create_deformation_coords_preserves_values_and_dtype(dtype):
    field = np.arange(72, dtype=dtype).reshape(3, 2, 3, 4)[:, ::-1]
    original = field.copy()
    expected = np.indices(field.shape[1:]) + field

    result = create_deformation_coords(field)

    np.testing.assert_array_equal(result, expected)
    assert result.dtype == expected.dtype
    np.testing.assert_array_equal(field, original)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int16])
def test_apply_transform_preserves_component_rounding(dtype):
    field = np.array([[0.1, 0.7], [-0.3, 0.6], [0.11, 0.33]], dtype=dtype)
    if np.issubdtype(dtype, np.integer):
        field = np.array([[1, 6], [4, -3], [2, 9]], dtype=dtype)
    field = field.reshape(3, 2, 1, 1)
    original = field.copy()
    deformation = np.zeros((3, 1, 1, 1))
    deformation[0] = 0.25
    # Interpolate a quarter of the way between two samples. SciPy rounds
    # to the input dtype before the component is stored as float64.
    expected = 0.75 * field[:, 0].astype(float) + 0.25 * field[:, 1].astype(
        float
    )
    if np.issubdtype(dtype, np.integer):
        expected = np.round(expected)
    expected = expected.astype(dtype).astype(float).reshape(3, 1, 1, 1)

    result = apply_transform(field, deformation, order=1, apply_to_coords=True)

    np.testing.assert_array_equal(result, expected)
    assert result.dtype == np.float64
    np.testing.assert_array_equal(field, original)


def test_calculate_offset_handles_singleton_axis():
    result = calculate_offset((1, 5, 7, 3), (3, 3, 4, 1))

    expected = np.zeros((3, 3, 4, 1))
    expected[0] = np.array([0, 1, 2])[:, None, None]
    expected[1] = np.array([0, 1, 2, 3])[None, :, None]
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int16])
def test_resize_transform_preserves_component_rounding(dtype):
    samples = np.array([[0.1, 0.7], [-0.3, 0.6], [0.11, 0.33]], dtype=dtype)
    if np.issubdtype(dtype, np.integer):
        samples = np.array([[1, 6], [4, -3], [2, 9]], dtype=dtype)
        midpoint = np.array([4, 1, 6], dtype=dtype)
    else:
        midpoint = (samples.astype(float).sum(axis=1) / 2).astype(dtype)
    expected = np.column_stack((samples[:, 0], midpoint, samples[:, 1]))
    expected = expected.astype(float).reshape(3, 3, 1, 1)
    expected[0] *= 1.5
    field = samples.reshape(3, 2, 1, 1)
    original = field.copy()

    result = resize_transform(field, (1, 1.5, 1, 1))

    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(field, original)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_resize_input_identity_preserves_field(dtype):
    field = np.arange(72, dtype=dtype).reshape(3, 2, 3, 4) / 8
    original = field.copy()

    result = resize_input(field, (1, 2, 3, 4), (1, 2, 3, 4))

    np.testing.assert_array_equal(result, original)
    np.testing.assert_array_equal(field, original)
    assert result.dtype == field.dtype
    assert not np.shares_memory(result, field)


def test_resize_input_changes_coordinate_frame():
    field = np.ones((3, 3, 3, 3))
    expected = np.full(field.shape, 2.0)
    expected[0] += np.array([0, 1.5, 3])[:, None, None]
    expected[1] += np.array([0, 1.5, 3])[None, :, None]
    expected[2] += np.array([0, 1.5, 3])[None, None, :]

    result = resize_input(field, (1, 3, 5, 7), (1, 6, 10, 14))

    np.testing.assert_array_equal(result, expected)
    np.testing.assert_array_equal(field, np.ones(field.shape))
