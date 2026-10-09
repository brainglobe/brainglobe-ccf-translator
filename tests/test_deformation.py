import numpy as np
import pytest

from brainglobe_ccf_translator.deformation.apply_deformation import (
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
