import os
import sys

sys.path.append(os.path.abspath("/home/harryc/github/ccf_translator/"))
import nibabel as nib
import math
import pandas as pd
import numpy as np
from ccf_translator.deformation.forward_transform import invert_deformation
from ccf_translator.deformation.apply_deformation import resize_input, apply_transform
import os
import matplotlib.pyplot as plt

root_path = r"/home/harryc/github/gubra/princeton/"

voxel_size_micron = 20


def open_deformation_field(deformation):
    """this function opens the elastix deformation
    and returns it in the format expected by ccf_translator"""
    deformation_arr = np.asanyarray(deformation.dataobj)
    def_header_dict = dict(deformation.header)
    # x_sign = math.copysign(1,def_header_dict['qoffset_x'])
    # y_sign = math.copysign(1,def_header_dict['qoffset_y'])
    # z_sign = math.copysign(1,def_header_dict['qoffset_z'])
    dim_scale = 1 / def_header_dict["pixdim"][1:4]
    # fopr some reason in this volume we have to ignore q offset
    # dim_scale = np.array([x_sign,y_sign,z_sign]) * dim_scale
    deformation_arr_scaled = np.squeeze(deformation_arr, 3)
    deformation_arr_scaled = np.transpose(deformation_arr_scaled, (3, 0, 1, 2))
    dim_scale_reshaped = dim_scale.reshape(-1, 1, 1, 1)
    deformation_arr_scaled_multiplied = deformation_arr_scaled * dim_scale_reshaped
    return deformation_arr_scaled_multiplied


def save_volume(volume, file_name):
    affine = np.eye(4)
    affine[:3, :3] *= voxel_size_micron
    volume = np.transpose(volume, (1, 2, 3, 0))
    image = nib.Nifti1Image(volume, affine=affine)
    image.header.set_xyzt_units(3)
    nib.save(image, file_name)


source_spaces = ["princeton_mouse"]
target_spaces = ["allen_mouse"]

original_elastix_volume_paths = [
    f"{root_path}/output_directory/deformationField.nii.gz"
]
 
for i in range(len(original_elastix_volume_paths)):
    original_elastix_volume_path = original_elastix_volume_paths[i]
    source = source_spaces[i]
    target = target_spaces[i]
    elastix_img = nib.load(original_elastix_volume_path)
    elastix_arr = open_deformation_field(elastix_img).astype(np.float32)
    elastix_arr = resize_input(
    elastix_arr, (1, *current_input_size), elastix_arr.shape
    )

    elastix_arr = elastix_arr[:,:,:,::-1]
    elastix_arr[2] *= -1
    elastix_arr = resize_input(
        elastix_arr, elastix_arr.shape, (1, *new_input_size)
    )
    elastix_arr = np.transpose(elastix_arr, [0,2,1,3])
    elastix_arr = elastix_arr[[1,0,2]]
    
    save_path = f"/home/harryc/github/ccf_translator/ccf_translator/metadata/deformation_fields/{source}/"
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    save_volume(elastix_arr, f"{save_path}/{source}_pull_{target}.nii.gz")
    inverted_arr = invert_deformation(elastix_arr, new_input_size[[1,0,2]])
    save_path = f"/home/harryc/github/ccf_translator/ccf_translator/metadata/deformation_fields/{target}/"
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    save_volume(inverted_arr, f"{save_path}/{target}_pull_{source}.nii.gz")

