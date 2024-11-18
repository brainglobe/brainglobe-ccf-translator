from scipy.interpolate.interpnd import NDInterpolatorBase, _ndim_coords_from_arrays
from scipy.spatial import cKDTree
import numpy as np


class NearestNDInterpolator(NDInterpolatorBase):
    """NearestNDInterpolator(x, y).

    Nearest-neighbor interpolator in N > 1 dimensions.

    .. versionadded:: 0.9

    Methods
    -------
    __call__

    Parameters
    ----------
    x : (npoints, ndims) 2-D ndarray of floats
        Data point coordinates.
    y : (npoints, ) 1-D ndarray of float or complex
        Data values.
    rescale : boolean, optional
        Rescale points to unit cube before performing interpolation.
        This is useful if some of the input dimensions have
        incommensurable units and differ by many orders of magnitude.

        .. versionadded:: 0.14.0
    tree_options : dict, optional
        Options passed to the underlying ``cKDTree``.

        .. versionadded:: 0.17.0

    See Also
    --------
    griddata :
        Interpolate unstructured D-D data.
    LinearNDInterpolator :
        Piecewise linear interpolator in N dimensions.
    CloughTocher2DInterpolator :
        Piecewise cubic, C1 smooth, curvature-minimizing interpolator in 2D.
    interpn : Interpolation on a regular grid or rectilinear grid.
    RegularGridInterpolator : Interpolator on a regular or rectilinear grid
                              in arbitrary dimensions (`interpn` wraps this
                              class).

    Notes
    -----
    Uses ``scipy.spatial.cKDTree``

    .. note:: For data on a regular grid use `interpn` instead.

    Examples
    --------
    We can interpolate values on a 2D plane:

    >>> from scipy.interpolate import NearestNDInterpolator
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> rng = np.random.default_rng()
    >>> x = rng.random(10) - 0.5
    >>> y = rng.random(10) - 0.5
    >>> z = np.hypot(x, y)
    >>> X = np.linspace(min(x), max(x))
    >>> Y = np.linspace(min(y), max(y))
    >>> X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
    >>> interp = NearestNDInterpolator(list(zip(x, y)), z)
    >>> Z = interp(X, Y)
    >>> plt.pcolormesh(X, Y, Z, shading='auto')
    >>> plt.plot(x, y, "ok", label="input point")
    >>> plt.legend()
    >>> plt.colorbar()
    >>> plt.axis("equal")
    >>> plt.show()

    If we want to have a smoother interpolation which is still easy to
    compute, we can specify a k value in the query along with a weights
    argument:

    >>> Z = interp(X, Y, k=4, weights='distance')
    >>> plt.pcolormesh(X, Y, Z, shading='auto')
    >>> plt.plot(x, y, "ok", label="input point")
    >>> plt.legend()
    >>> plt.colorbar()
    >>> plt.axis("equal")
    >>> plt.show()
    """

    def __init__(self, x, y, rescale=False, tree_options=None):
        NDInterpolatorBase.__init__(
            self, x, y, rescale=rescale, need_contiguous=False, need_values=False
        )
        if tree_options is None:
            tree_options = dict()
        self.tree = cKDTree(self.points, **tree_options)
        self.values = np.asarray(y)

    def __call__(self, *args, weights="uniform", **query_options):
        """
        Evaluate interpolator at given points.

        Parameters
        ----------
        x1, x2, ... xn : array-like of float
            Points where to interpolate data at.
            x1, x2, ... xn can be array-like of float with broadcastable shape.
            or x1 can be array-like of float with shape ``(..., ndim)``
        weights : {'uniform', 'distance'}, optional method of averaging the k nearest
            neighbors. uniform means all k returned points are weighted equally,
            whereas distance weights points by the inverse of their distance. If
            k is set to 1 (as is the default), this parameter will have
            no effect.
        **query_options
            This allows ``k``, ``eps``, ``p``, ``distance_upper_bound``, and ``workers``
            being passed to the cKDTree's query function to be explicitly set.
            See `scipy.spatial.cKDTree.query` for an overview of the different options.

            .. versionadded:: 1.12.0

        """
        # For the sake of enabling subclassing, NDInterpolatorBase._set_xi performs
        # some operations which are not required by NearestNDInterpolator.__call__,
        # hence here we operate on xi directly, without calling a parent class function.
        xi = _ndim_coords_from_arrays(args, ndim=self.points.shape[1])
        xi = self._check_call_shape(xi)
        xi = self._scale_x(xi)

        # We need to handle two important cases:
        # (1) the case where xi has trailing dimensions (..., ndim), and
        # (2) the case where y has trailing dimensions
        # We will first flatten xi to deal with case (1),
        # do the computation in flattened array while retaining y's dimensionality,
        # and then reshape the interpolated values back to match xi's shape.

        # Flatten xi for the query
        xi_flat = xi.reshape(-1, xi.shape[-1])
        original_shape = xi.shape
        flattened_shape = xi_flat.shape

        # if distance_upper_bound is set to not be infinite,
        # then we need to consider the case where cKDtree
        # does not find any points within distance_upper_bound to return.
        # It marks those points as having infinte distance, which is what will be used
        # below to mask the array and return only the points that were deemed
        # to have a close enough neighbor to return something useful.
        dist, i = self.tree.query(xi_flat, **query_options)

        valid_mask = np.isfinite(dist)

        # create a holder interp_values array and fill with nans.
        if self.values.ndim > 1:
            interp_shape = flattened_shape[:-1] + self.values.shape[1:]
        else:
            interp_shape = flattened_shape[:-1]

        if np.issubdtype(self.values.dtype, np.complexfloating):
            interp_values = np.full(interp_shape, np.nan, dtype=self.values.dtype)
        else:
            interp_values = np.full(interp_shape, np.nan)

        if dist.ndim == 1:
            interp_values[valid_mask] = self.values[i[valid_mask], ...]
        else:
            valid_mask = valid_mask.all(axis=1)
            if weights == "uniform" and dist.ndim > 1:
                interp_values = np.average(self.values[i], axis=1)
            elif weights == "distance" and dist.ndim > 1:
                power = 2
                interp_values = np.average(
                    self.values[i], axis=1, weights=(1.0 / (dist**power))
                )
            else:
                raise ValueError(
                    "Unknown value %r passed for weights,"
                    "must be either 'uniform' or 'distance'" % (weights)
                )
            interp_values[~valid_mask] = np.nan

        if self.values.ndim > 1:
            new_shape = original_shape[:-1] + self.values.shape[1:]
        else:
            new_shape = original_shape[:-1]
        interp_values = interp_values.reshape(new_shape)

        return interp_values
