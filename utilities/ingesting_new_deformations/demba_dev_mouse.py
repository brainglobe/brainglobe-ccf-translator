from brainglobe_ccf_translator.deformation.forward_transform import invert_deformation
import numpy as np
from glob import glob
import nibabel as nib
import os
import math

# Here we have a recreation of the intermediate volumes
# We start from the files which came out of elastix
# These can be found in the EBRAINS datasets ie;
# for instance the dataset titled Allen Mouse Brain CCFv3 segmentations transformed to P7 population-averaged serial two-photon tomography data
# Their path is script_with_metadata/deformationField.nii.gz
VERSION = "1.1"



deformation_urls = {
    4: "https://data-proxy.ebrains.eu/api/v1/buckets/d-278f396c-53b6-4d90-9b3f-568d3f23e407/script_with_metadata/deformationField.nii.gz?inline=true",
    7: "https://data-proxy.ebrains.eu/api/v1/buckets/d-d45dc547-64eb-4314-8f73-c6e3d5ab8de0/script_with_metadata/deformationField.nii.gz?inline=true",
    14: "https://data-proxy.ebrains.eu/api/v1/buckets/d-fbd8a406-a114-4c26-a77d-59dc93476682/script_with_metadata/deformationField.nii.gz?inline=true",
    21: "https://data-proxy.ebrains.eu/api/v1/buckets/d-4a0b3a87-4fe4-4a9f-a07e-e35e54681ff8/script_with_metadata/deformationField.nii.gz?inline=true",
    28: "https://data-proxy.ebrains.eu/api/v1/buckets/d-c8395a2f-a6ae-40d0-ad0d-a87e8b9b610b/script_with_metadata/deformationField.nii.gz?inline=true"
    }

key_ages = [56, 28, 21, 14, 7, 4]
space_name = "Demba"
voxel_size_micron = 20
save_path = f"brainglobe_ccf_translator/metadata/deformation_fields/{space_name}"
if not os.path.exists(save_path):
    os.mkdir(save_path)


def open_deformation_field(deformation):
    """this function opens the elastix deformation
    and returns it in the format expected by brainglobe_ccf_translator"""
    deformation_arr = np.asanyarray(deformation.dataobj)
    def_header_dict = dict(deformation.header)
    x_sign = math.copysign(1, def_header_dict["qoffset_x"])
    y_sign = math.copysign(1, def_header_dict["qoffset_y"])
    z_sign = math.copysign(1, def_header_dict["qoffset_z"])
    dim_scale = 1 / def_header_dict["pixdim"][1:4]
    dim_scale = np.array([x_sign, y_sign, z_sign]) * dim_scale
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


for i in range(len(key_ages) - 1):
    age = key_ages[i + 1]
    # using ccft terminology we would say that the elastix deform is in
    # the 28 space pulling values in from 56 (for the p28 volume that is)
    original_elastix_volume_path = deformation_urls[age]
    elastix_img = nib.load(original_elastix_volume_path)
    elastix_arr = open_deformation_field(elastix_img).astype(np.float32)
    elastix_arr = elastix_arr[[2,1,0]]
    elastix_arr = np.transpose(elastix_arr, (0, 3, 2, 1))
    magnitude = key_ages[i] - age
    # here we make it a single day transform so in our example 28 pulling values from 29
    elastix_arr /= magnitude
    save_volume(elastix_arr, f"{save_path}/{age}_pull_{age+1}_v{VERSION}.nii.gz")
    for day in range(1, magnitude + 1):
        temp_arr = elastix_arr.copy()
        temp_arr *= day
        temp_arr = invert_deformation(temp_arr)
        temp_arr /= day
        temp_age = age + day
        save_volume(temp_arr, f"{save_path}/{temp_age}_pull_{temp_age-1}_v{VERSION}.nii.gz")
