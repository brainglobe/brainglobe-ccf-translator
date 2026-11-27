"""Ingest the swc_female_rat <-> whs_sd_rat deformation fields.

The Waxholm Space (WHS) Sprague Dawley rat template (BrainGlobe
``whs_sd_rat_39um``) was registered to the SWC female Lister Hooded rat STPT
template (BrainGlobe ``swc_female_rat_25um``) with FireANTs. The STPT template
was the fixed image and WHS the moving image. FireANTs saved the result as an
ANTs-style composed displacement field (initial affine plus deformation).

Both images were given unit spacing, zero origin and identity direction during
registration. As a result, the ANTs field is a voxel displacement:

    whs_index = stpt_index + displacement(stpt_index)

Here ``stpt_index`` is a 25 um voxel on the STPT grid and ``whs_index`` is a
39 um voxel on the WHS grid. In NIfTI (x, y, z) order both images are in
BrainGlobe ASR orientation, so the indices are the BrainGlobe atlas array
indices of each atlas.

CCF translator expects each field to be defined on the grid of the space it
lives in (``<space>_pull_<other>``). Its displacements are in voxels of
``transformation_resolution_micron``, and they point into the other space at
that same resolution. This script:

1. Rescales the forward field into that convention to make
   ``swc_female_rat_pull_whs_sd_rat`` (on the STPT grid).
2. Numerically inverts it to make ``whs_sd_rat_pull_swc_female_rat`` (on the
   WHS grid). The inversion uses a fixed-point iteration, preconditioned with
   the best-fit global affine, then local Newton refinement.

Usage:
    python swc_female_rat.py /path/to/composed_warp_ants.nii.gz
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates

VERSION = "1.0"
SWC_SPACE = "swc_female_rat"
WHS_SPACE = "whs_sd_rat"
# Native voxel sizes of the images used during registration.
SWC_NATIVE_MICRON = 25.0
WHS_NATIVE_MICRON = 39.0
# BrainGlobe whs_sd_rat_39um shape in ASR order.
WHS_NATIVE_SHAPE = np.array([1024, 512, 512])
# Inversion settings (units: STPT voxels for steps, WHS voxels for error).
NEWTON_TOLERANCE = 0.05
MAX_STEP = 5.0


def load_ants_warp(path):
    """Return the ANTs displacement as (3, A, S, R) float32 in WHS voxels."""
    img = nib.load(str(path))
    arr = np.asanyarray(img.dataobj)
    if arr.ndim == 5:
        arr = arr[:, :, :, 0, :]
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"Unexpected ANTs warp shape {arr.shape}")
    spacing = np.abs(np.diag(img.affine)[:3])
    if not np.allclose(spacing, 1) or not np.allclose(img.affine[:3, 3], 0):
        raise ValueError(
            "Expected a unit-spacing, zero-origin warp "
            "(displacements in voxels); got affine\n"
            f"{img.affine}"
        )
    return np.ascontiguousarray(np.moveaxis(arr, -1, 0), dtype=np.float32)


def _grid(shape, step=1.0):
    return np.meshgrid(
        *[np.arange(n, dtype=np.float32) * step for n in shape],
        indexing="ij",
    )


def _axis_coords(shape, axis, step=1.0):
    """Broadcastable coordinates along one axis (avoids dense 3D grids)."""
    view = [1, 1, 1]
    view[axis] = -1
    return (np.arange(shape[axis], dtype=np.float32) * step).reshape(view)


def make_swc_pull_whs(warp, resolution):
    """CCF translator field on the STPT grid pulling from WHS."""
    native_shape = np.array(warp.shape[1:])
    shape = np.round(native_shape * SWC_NATIVE_MICRON / resolution).astype(int)
    step = resolution / SWC_NATIVE_MICRON  # STPT native voxels per output
    out = np.empty((3, *shape), dtype=np.float32)
    coords = None
    if step != 1:
        coords = _grid(shape, step)
    for axis in range(3):
        native = _axis_coords(shape, axis, step)
        if coords is None:
            out[axis] = warp[axis]
        else:
            out[axis] = map_coordinates(
                warp[axis], coords, order=1, mode="nearest"
            )
        # out = WHS position (in output voxels) - own grid position
        out[axis] += native
        out[axis] *= WHS_NATIVE_MICRON / resolution
        out[axis] -= native / step
    return out


def fit_affine(warp, stride=8):
    """Least-squares affine with whs_index ~= M @ stpt_index + t."""
    sub = warp[:, ::stride, ::stride, ::stride]
    grid = _grid(sub.shape[1:], stride)
    src = np.stack([g.ravel() for g in grid], axis=1)
    dst = src + np.stack([sub[i].ravel() for i in range(3)], axis=1)
    design = np.concatenate([src, np.ones((len(src), 1))], axis=1)
    coef, *_ = np.linalg.lstsq(design, dst, rcond=None)
    return coef[:3].T, coef[3]


def make_whs_pull_swc(
    warp, resolution, iterations=20, newton_iterations=30, slab=16, workers=8
):
    """CCF translator field on the WHS grid pulling from STPT.

    It solves forward(q) = w for STPT coordinates q, where
    forward(q) = M q + t + r(q), and r is the non-affine remainder of the
    ANTs warp. r is extrapolated as a constant outside the STPT grid, so the
    iteration stays well-posed for WHS voxels beyond the STPT field of view
    (such as the spinal cord). Those voxels pull from outside the STPT image
    and receive the background value.
    """
    matrix, offset = fit_affine(warp)
    matrix_inv = np.linalg.inv(matrix)
    stpt_shape = np.array(warp.shape[1:])

    # Non-affine remainder r(q) = q + warp(q) - (M q + t), done in place.
    for axis in range(3):
        warp[axis] += _axis_coords(stpt_shape, axis) - offset[axis]
        for j in range(3):
            warp[axis] -= matrix[axis, j] * _axis_coords(stpt_shape, j)
    remainder = warp

    shape = np.round(WHS_NATIVE_SHAPE * WHS_NATIVE_MICRON / resolution).astype(
        int
    )
    step = resolution / WHS_NATIVE_MICRON  # WHS native voxels per output vox
    out = np.empty((3, *shape), dtype=np.float32)
    n_blocks = int(np.ceil(shape[0] / slab))
    residual_max = np.zeros(n_blocks)
    residual_over = np.zeros(n_blocks, dtype=np.int64)
    residual_count = np.zeros(n_blocks, dtype=np.int64)

    def solve(block):
        start = block * slab
        stop = min(start + slab, shape[0])
        w = np.stack(
            np.meshgrid(
                np.arange(start, stop) * step,
                np.arange(shape[1]) * step,
                np.arange(shape[2]) * step,
                indexing="ij",
            )
        ).reshape(3, -1)

        def forward(q):
            r = np.stack(
                [
                    map_coordinates(remainder[i], q, order=1, mode="nearest")
                    for i in range(3)
                ]
            )
            return matrix @ q + offset[:, None] + r

        # Cheap fixed-point iterations, preconditioned with the global affine.
        q = matrix_inv @ (w - offset[:, None])
        for _ in range(iterations):
            q -= matrix_inv @ (forward(q) - w)

        # Where the local Jacobian differs strongly from the global affine
        # (strong local compression or expansion), the fixed point converges
        # slowly. Refine those points with Newton steps, using a
        # finite-difference Jacobian and limiting each step to MAX_STEP voxels.
        err = np.linalg.norm(forward(q) - w, axis=0)
        todo = np.flatnonzero(err > NEWTON_TOLERANCE)
        for _ in range(newton_iterations):
            if not todo.size:
                break
            qt = q[:, todo]
            fq = forward(qt)
            jac = np.empty((todo.size, 3, 3))
            for j in range(3):
                dq = qt.copy()
                dq[j] += 1.0
                jac[:, :, j] = (forward(dq) - fq).T
            jac += 1e-3 * np.eye(3)
            delta = np.linalg.solve(jac, (fq - w[:, todo]).T[..., None])
            delta = delta[..., 0].T
            norm = np.linalg.norm(delta, axis=0)
            delta *= np.minimum(1.0, MAX_STEP / np.maximum(norm, 1e-9))
            q[:, todo] = qt - delta
            err[todo] = np.linalg.norm(
                forward(q[:, todo]) - w[:, todo], axis=0
            )
            todo = todo[err[todo] > NEWTON_TOLERANCE]

        inside = np.all((q >= 0) & (q <= stpt_shape[:, None] - 1), axis=0)
        residual_max[block] = err[inside].max() if inside.any() else 0.0
        residual_over[block] = np.count_nonzero(err[inside] > 1)
        residual_count[block] = np.count_nonzero(inside)
        q_micron = q * SWC_NATIVE_MICRON
        disp = q_micron / resolution - w / step
        out[:, start:stop] = disp.reshape(3, stop - start, *shape[1:])

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(solve, range(len(residual_max))))
    print(
        "inversion residual inside STPT field of view: max "
        f"{residual_max.max():.4f} WHS voxels, "
        f"{residual_over.sum() / max(1, residual_count.sum()):.2e} "
        "of voxels above 1 WHS voxel"
    )
    return out


def save_field(field, resolution, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    affine = np.diag([resolution, resolution, resolution, 1.0])
    image = nib.Nifti1Image(np.moveaxis(field, 0, -1), affine=affine)
    image.header.set_xyzt_units(3)
    nib.save(image, str(path))
    print(f"saved {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ants_warp", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("~/.brainglobe/deformation_fields").expanduser(),
    )
    parser.add_argument(
        "--swc-resolution", type=float, default=SWC_NATIVE_MICRON
    )
    parser.add_argument(
        "--whs-resolution", type=float, default=WHS_NATIVE_MICRON
    )
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--newton-iterations", type=int, default=30)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    warp = load_ants_warp(args.ants_warp)
    swc_field = make_swc_pull_whs(warp, args.swc_resolution)
    save_field(
        swc_field,
        args.swc_resolution,
        args.output_dir
        / SWC_SPACE
        / f"{SWC_SPACE}_pull_{WHS_SPACE}_v{VERSION}.nii.gz",
    )
    del swc_field
    whs_field = make_whs_pull_swc(
        warp,
        args.whs_resolution,
        iterations=args.iterations,
        newton_iterations=args.newton_iterations,
        workers=args.workers,
    )
    save_field(
        whs_field,
        args.whs_resolution,
        args.output_dir
        / WHS_SPACE
        / f"{WHS_SPACE}_pull_{SWC_SPACE}_v{VERSION}.nii.gz",
    )


if __name__ == "__main__":
    main()
