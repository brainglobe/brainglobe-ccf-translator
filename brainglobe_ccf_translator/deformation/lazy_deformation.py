"""Evaluate composed fields only at grid points needed by the output."""

import numpy as np
from scipy.ndimage import map_coordinates

_EVALUATION_CHUNK_VOXELS = 16_384


class LazyDeformation:
    """A discrete deformation grid whose values are evaluated on demand.

    Interpolation evaluates the eight neighboring grid points first. This
    preserves interpolation of the composed field, rather than replacing it
    with composition of prematurely downsampled fields.
    """

    def __init__(self, shape, dtype, evaluate, sample=None):
        self.shape = (3, *shape)
        self.dtype = np.dtype(dtype)
        self._evaluate = evaluate
        self._sample = sample

    @classmethod
    def from_array(cls, field):
        def evaluate(indices):
            return field[:, indices[0], indices[1], indices[2]]

        def sample(coordinates):
            return np.stack(
                [
                    map_coordinates(component, coordinates, order=1)
                    for component in field
                ]
            )

        return cls(field.shape[1:], field.dtype, evaluate, sample)

    def sample(self, coordinates):
        if self._sample is not None:
            return self._sample(coordinates)
        shape = np.array(self.shape[1:])
        valid = np.all(
            np.isfinite(coordinates)
            & (coordinates >= 0)
            & (coordinates <= (shape - 1)[:, None]),
            axis=0,
        )
        output = np.zeros((3, coordinates.shape[1]), dtype=self.dtype)
        coordinates = coordinates[:, valid]
        if not coordinates.shape[1]:
            return output
        lower = np.floor(coordinates).astype(np.intp)
        # ndimage reflects an interpolation neighbor beyond the last voxel,
        # even when its weight is zero. This matters for NaN/Inf neighbors.
        upper = np.where(
            lower + 1 >= shape[:, None],
            np.maximum(shape - 2, 0)[:, None],
            lower + 1,
        )
        fraction = coordinates - lower
        # Preserve SciPy's final-weight rounding and corner accumulation order.
        weights = [(1 - f, 1 - (1 - f)) for f in fraction]
        corners = np.empty((3, coordinates.shape[1], 8), dtype=np.intp)
        for corner in range(8):
            for axis in range(3):
                bit = (corner >> (2 - axis)) & 1
                corners[axis, :, corner] = upper[axis] if bit else lower[axis]
        flat_indices = np.ravel_multi_index(corners.reshape(3, -1), shape)
        unique, inverse = np.unique(flat_indices, return_inverse=True)
        values = self._evaluate(np.array(np.unravel_index(unique, shape)))
        values = values[:, inverse].reshape(3, coordinates.shape[1], 8)
        interpolated = np.zeros((3, coordinates.shape[1]))
        # SciPy's compiled interpolation propagates nonfinite values silently.
        with np.errstate(invalid="ignore", over="ignore"):
            for corner in range(8):
                coefficient = values[:, :, corner].astype(float)
                for axis in range(3):
                    bit = (corner >> (2 - axis)) & 1
                    coefficient *= weights[axis][bit]
                interpolated += coefficient
        if np.issubdtype(self.dtype, np.integer):
            interpolated = np.where(
                interpolated > 0,
                np.floor(interpolated + 0.5),
                np.ceil(interpolated - 0.5),
            )
            bounds = np.iinfo(self.dtype)
            interpolated = np.clip(interpolated, bounds.min, bounds.max)
        output[:, valid] = interpolated
        return output

    def resize(self, scale):
        shape = tuple(
            int(size * factor)
            for size, factor in zip(self.shape[1:], scale[1:])
        )
        axes = [
            np.linspace(0, size - 1, count)
            for size, count in zip(self.shape[1:], shape)
        ]

        def evaluate(indices):
            coordinates = np.stack(
                [axis[index] for axis, index in zip(axes, indices)]
            )
            values = self.sample(coordinates).astype(float)
            for axis in range(3):
                values[axis] *= scale[axis + 1]
            return values

        return LazyDeformation(shape, np.float64, evaluate)

    def combine(self, following):
        def evaluate(indices):
            displacement = following._evaluate(indices)
            values = self.sample(indices + displacement).astype(float)
            values += displacement
            return values

        return LazyDeformation(following.shape[1:], np.float64, evaluate)

    def pad(self, padding):
        padding = np.round(padding).astype(int)[1:]
        shape = np.array(self.shape[1:])
        crop_start = np.minimum(np.maximum(-padding[:, 0], 0), shape)
        cropped_shape = shape - crop_start
        cropped_shape -= np.minimum(
            np.maximum(-padding[:, 1], 0), cropped_shape
        )
        pad_start = np.maximum(padding[:, 0], 0)
        output_shape = cropped_shape + pad_start + np.maximum(padding[:, 1], 0)

        def evaluate(indices):
            valid = np.all(
                (indices >= pad_start[:, None])
                & (indices < (pad_start + cropped_shape)[:, None]),
                axis=0,
            )
            values = np.zeros((3, indices.shape[1]), dtype=self.dtype)
            original = (
                indices[:, valid] - pad_start[:, None] + crop_start[:, None]
            )
            values[:, valid] = self._evaluate(original)
            return values

        return LazyDeformation(output_shape, self.dtype, evaluate)

    def add_offset(self, offset):
        def evaluate(indices):
            values = self._evaluate(indices).copy()
            for axis in range(3):
                values[axis] += offset[axis]
            return values

        return LazyDeformation(self.shape[1:], self.dtype, evaluate)

    def transpose(self, order):
        inverse = np.argsort(order)

        def evaluate(indices):
            return self._evaluate(indices[inverse])[order]

        return LazyDeformation(
            np.array(self.shape[1:])[order], self.dtype, evaluate
        )

    def flip(self, axis):
        def evaluate(indices):
            indices = indices.copy()
            indices[axis] = self.shape[axis + 1] - 1 - indices[axis]
            values = self._evaluate(indices).copy()
            values[axis] *= -1
            return values

        return LazyDeformation(self.shape[1:], self.dtype, evaluate)

    def materialize(self):
        output = np.empty(self.shape, dtype=self.dtype)
        flat_output = output.reshape(3, -1)
        shape = self.shape[1:]
        for start in range(0, flat_output.shape[1], _EVALUATION_CHUNK_VOXELS):
            stop = min(start + _EVALUATION_CHUNK_VOXELS, flat_output.shape[1])
            indices = np.array(np.unravel_index(np.arange(start, stop), shape))
            flat_output[:, start:stop] = self._evaluate(indices)
        return output
