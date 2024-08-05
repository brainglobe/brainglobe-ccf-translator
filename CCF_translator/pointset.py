from .deformation import apply_deformation, route_calculation
import pandas as pd
import os
base_path = os.path.dirname(__file__)


class pointset:
    def __init__(self, values, space, voxel_size_um, age_PND):
        self.values = values
        self.space = space
        self.voxel_size_um = voxel_size_um
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
        row_template = '{}_physical_size_micron'
        space_size_micron = self.metadata[self.metadata['source_space'] == self.space].iloc[0][[row_template.format('X'),
                                                                row_template.format('Y'),
                                                                row_template.format('Z')]].values
        space_size_voxels = space_size_micron/self.voxel_size_um
        source = f"{self.space}_P{self.age_PND}"
        target = f"{target_space}_P{target_age}"
        route = route_calculation.calculate_route(target, source, self.metadata)
        deform_arr, pad_sum, flip_sum, dim_order_sum = apply_deformation.combine_route(
            route, space_size_voxels, base_path, self.metadata
        )
        values = values[:,dim_order_sum]
        space_size_reorder = space_size_voxels[dim_order_sum]
        for i in range(len(flip_sum)):
            if flip_sum[i]:
                values[:,i] = space_size_reorder[i] - values[:,i]
        for i in range(3):
            values[:,i] = values[:,i] +  pad_sum[i][0]
        if deform_arr is not None:
            values = values + deform_arr[:,values[:,0].astype(int),values[:,1].astype(int),values[:,2].astype(int) ].T
        self.values = values
        self.age_PND = target_age
        self.space = target_space

