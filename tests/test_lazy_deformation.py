import numpy as np
import pytest
from scipy.ndimage import map_coordinates

from brainglobe_ccf_translator.deformation import (
    apply_deformation,
    lazy_deformation,
)
from brainglobe_ccf_translator.deformation.lazy_deformation import (
    LazyDeformation,
)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int16])
def test_lazy_interpolation_matches_scipy_at_boundaries(dtype):
    rng = np.random.default_rng(42)
    field = (rng.normal(size=(3, 9, 1, 13)) * 100).astype(dtype)
    coordinates = rng.uniform(-1, 14, size=(3, 100))
    coordinates[1] = 0
    coordinates[:, :3] = np.array([[0, 8, 8], [0, 0, 0], [0, 12, 0]])
    expected = np.stack(
        [
            map_coordinates(component, coordinates, order=1)
            for component in field
        ]
    )
    # Transpose through the virtual-grid path rather than the direct sampler.
    field_lazy = LazyDeformation.from_array(field).transpose([0, 1, 2])

    actual = field_lazy.sample(coordinates)

    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == expected.dtype


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_lazy_geometry_and_composition_match_dense(dtype, monkeypatch):
    rng = np.random.default_rng(42)
    first = rng.normal(size=(3, 9, 11, 13)).astype(dtype)
    original = first.copy()
    padding = np.array([[0, 0], [-1, 2], [1, -2], [0, 1]])
    dense = apply_deformation.pad_neg(first, padding, "constant")
    dense += np.array([1, 2, 3])[:, None, None, None]
    dense = np.transpose(dense, [0, 3, 1, 2])[[2, 0, 1]]
    dense[1] *= -1
    dense = np.flip(dense, axis=2)
    lazy = LazyDeformation.from_array(first).pad(padding).add_offset([1, 2, 3])
    lazy = lazy.transpose([2, 0, 1]).flip(1)
    second = rng.uniform(-1, 1, size=dense.shape).astype(dtype)
    scale = (1, 0.6, 0.4, 0.3)
    expected = apply_deformation.resize_transform(
        apply_deformation.combine_deformations(dense, second), scale
    )
    # Exercise materialization across many small batches.
    monkeypatch.setattr(lazy_deformation, "_EVALUATION_CHUNK_VOXELS", 7)

    actual = (
        lazy.combine(LazyDeformation.from_array(second))
        .resize(scale)
        .materialize()
    )

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(first, original)


@pytest.mark.parametrize("nonfinite", [np.nan, np.inf])
def test_lazy_boundary_preserves_nonfinite_neighbors(nonfinite):
    field = np.arange(24, dtype=float).reshape(3, 2, 2, 2)
    field[:, 0] = nonfinite
    coordinates = np.array([[1.0, 1.0], [0.0, 1.0], [0.0, 1.0]])
    expected = np.stack(
        [
            map_coordinates(component, coordinates, order=1)
            for component in field
        ]
    )

    actual = (
        LazyDeformation.from_array(field)
        .transpose([0, 1, 2])
        .sample(coordinates)
    )

    np.testing.assert_array_equal(actual, expected)
