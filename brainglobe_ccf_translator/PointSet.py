from .deformation import apply_deformation, route_calculation
import pandas as pd
import os
import numpy as np

base_path = os.path.dirname(__file__)


class PointSet:
    def __init__(self, values, space, voxel_size_micron, age_PND):
        self.values = np.array(values).astype(float)
        self.space = space
        self.voxel_size_micron = voxel_size_micron
        self.age_PND = age_PND
        metadata_path = os.path.join(base_path, "metadata", "translation_metadata.csv")
        metadata = pd.read_csv(metadata_path)
        self.metadata = metadata

    def transform(self, target_age=None, target_space=None):
        values = self.values
        if len(values.shape) == 1:
            values = values.reshape(1, -1)
        source = f"{self.space}_P{self.age_PND}"
        target = f"{target_space}_P{target_age}"
        row_template = "{}_physical_size_micron"
        source = f"{self.space}_P{self.age_PND}"
        target = f"{target_space}_P{target_age}"
        G = route_calculation.create_G(self.metadata)
        route = route_calculation.calculate_route(target, source, G)
        deform_arr, pad_sum, flip_sum, dim_order_sum, final_voxel_size = apply_deformation.combine_route(
            route, self.voxel_size_micron, base_path, self.metadata
        )
        previous = '_'.join(route[1].split('_')[:-1])
        space_size_micron = (
            self.metadata[(self.metadata["source_space"] == target_space) & (self.metadata["target_space"] == previous)]
            .iloc[0][
                [
                    row_template.format("X"),
                    row_template.format("Y"),
                    row_template.format("Z"),
                ]
            ]
            .values
        )

        space_size_voxels = space_size_micron / self.voxel_size_micron
        if deform_arr is not None:
            if final_voxel_size != self.voxel_size_micron:
                values = values * (self.voxel_size_micron / final_voxel_size)
            # Create a mask of rows in 'values' that contain NaN
            nan_mask = np.isnan(values).any(axis=1)

            # Only perform the operation on rows that do not contain NaN
            values[~nan_mask] = (
                values[~nan_mask]
                + deform_arr[
                    :,
                    values[~nan_mask, 0].astype(int),
                    values[~nan_mask, 1].astype(int),
                    values[~nan_mask, 2].astype(int),
                ].T
            )

            # For rows that contain NaN, set the entire row to NaN
            values[nan_mask] = np.nan
            if final_voxel_size != self.voxel_size_micron:
                values = values * (final_voxel_size / self.voxel_size_micron)
        for i in range(3):
            values[:, i] = values[:, i] + pad_sum[i][0]

        dim_order_sum = apply_deformation.invert_dim_order(dim_order_sum)
        values = values[:, dim_order_sum]
        flip_sum_reorder = np.array(flip_sum)[dim_order_sum]
        for i in range(len(flip_sum_reorder)):
            if flip_sum_reorder[i]:
                values[:, i] = space_size_voxels[i] - values[:, i]


        self.values = values
        self.age_PND = target_age
        self.space = target_space
