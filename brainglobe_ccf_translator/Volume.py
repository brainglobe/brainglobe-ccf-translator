import numpy as np
from .deformation import apply_deformation, route_calculation
import pandas as pd
import json
import os
import nibabel as nib

base_path = os.path.dirname(__file__)
"""
At present the order of transformations is:
transpose 
flip 
transform
(if transform doesnt exist we pad, if it does we don't since it is handled by the transform)

So the transform should be in the shape of the output
"""


class Volume:
    def __init__(
        self, values, space, voxel_size_micron, age_PND, segmentation_file=False
    ):
        self.values = values
        self.space = space
        self.voxel_size_micron = voxel_size_micron
        self.age_PND = age_PND
        self.segmentation_file = segmentation_file
        metadata_path = os.path.join(base_path, "metadata", "translation_metadata.csv")
        metadata = pd.read_csv(metadata_path)
        self.metadata = metadata

    def transform(self, target_age, target_space):
        array = self.values
        source = f"{self.space}_P{self.age_PND}"
        target = f"{target_space}_P{target_age}"
        if source == target:
            print("volume is already in that space")
            return
        G = route_calculation.create_G(self.metadata)
        route = route_calculation.calculate_route(source, target, G)
        deform_arr, pad_sum, flip_sum, dim_order_sum, final_voxel_size = (
            apply_deformation.combine_route(
                route, self.voxel_size_micron, base_path, self.metadata
            )
        )
        array = np.transpose(array, dim_order_sum)
        for i in range(len(flip_sum)):
            if flip_sum[i]:
                array = np.flip(array, axis=i)
        if deform_arr is not None:

            # original_input_shape = np.array([456.0, 668.0, 320.0])
            if final_voxel_size != self.voxel_size_micron:
                original_input_shape = np.shape(array)
                original_input_shape = np.array(original_input_shape)[dim_order_sum]
                new_input_shape = np.array(array.shape) * (
                    final_voxel_size / self.voxel_size_micron
                )
                deform_arr = apply_deformation.resize_transform(
                    deform_arr,
                    (1, *([final_voxel_size / self.voxel_size_micron] * 3)),
                )
            order = 0 if self.segmentation_file else 1
            array = apply_deformation.apply_transform(array, deform_arr, order=order)
        else:
            array = apply_deformation.pad_neg(array, pad_sum, mode="constant")
        self.values = array
        self.age_PND = target_age
        self.space = target_space

    def save(self, save_path):
        vol_metadata = {
            "space": self.space,
            "age_PND": self.age_PND,
            "segmentation_file": self.segmentation_file,
        }
        affine = np.eye(4)
        affine[:3, :3] *= self.voxel_size_micron
        image = nib.Nifti1Image(self.values, affine=affine)
        image.header["descrip"] = vol_metadata
        image.header.set_xyzt_units(3)
        nib.save(image, save_path)
